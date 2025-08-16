import json
import math
import random
import re
from typing import List, Tuple, Optional, Iterable

from gomoku import Agent
from gomoku.llm import OpenAIGomokuClient
from gomoku.core.models import GameState, Player


Coord = Tuple[int, int]


class Mark5Agent(Agent):

    # ------------------------- lifecycle -------------------------

    def _setup(self):
        self.llm = OpenAIGomokuClient(model="google/gemma-2-9b-it")
        # knobs
        self.max_llm_candidates = 10         # send at most N candidates to the LLM
        self.max_neighbors_radius = 2        # consider empty cells within R of any stone
        self.random_fallback = False         # set True if you want more exploration

    # ------------------------- core API --------------------------

    async def get_move(self, game_state: GameState) -> Coord:
        board = self._board_from_history(game_state)
        me, opp = self.player.value, (Player.WHITE if self.player == Player.BLACK else Player.BLACK).value
        size = game_state.board_size

        legal = list(game_state.get_legal_moves())
        if not legal:
            # Should not happen in normal flow, but be safe.
            return (size // 2, size // 2)

        # 0) FIRST MOVE bias: take (3,3) / center-ish on empty boards
        if len(game_state.move_history) == 0:
            return (size // 2, size // 2)

        # 1) INSTANT WIN: if we can win right now, do it.
        win_now = self._first_winning_move(board, legal, me)
        if win_now is not None:
            return win_now

        # 2) INSTANT BLOCK: if opponent can win next move, block it.
        block_now = self._block_opponent_win(board, legal, me, opp)
        if block_now is not None:
            return block_now

        # 3) MICRO LOOK-AHEAD (1-ply safety): avoid moves that hand
        #    the opponent an immediate win unless our move also wins.
        safe_moves = self._filter_suicidal(board, legal, me, opp)
        if not safe_moves:
            safe_moves = legal  # if everything looks risky, just continue

        # 4) Build candidate set near existing stones and score them
        candidates = self._nearby_moves(board, safe_moves, radius=self.max_neighbors_radius)
        if not candidates:
            candidates = safe_moves

        scored = sorted(candidates, key=lambda rc: -self._heuristic_score(board, rc, me, opp))
        topK = scored[: self.max_llm_candidates]

        # 5) Ask the LLM to choose ONE move among the topK
        llm_choice = await self._ask_llm_move(game_state, board, me, opp, topK)
        if llm_choice and game_state.is_valid_move(*llm_choice):
            return llm_choice

        # 6) Fallbacks: best heuristic -> random
        best = topK[0] if topK else safe_moves[0]
        if self.random_fallback:
            best = random.choice(topK if topK else safe_moves)
        return best

    # ------------------------- board utils -----------------------

    def _board_from_history(self, game_state: GameState) -> List[List[str]]:
        """Rebuild a char board ('.','X','O') purely from move history."""
        size = game_state.board_size
        B = [['.' for _ in range(size)] for _ in range(size)]
        for mv in game_state.move_history:
            B[mv.row][mv.col] = mv.player.value  # 'X' or 'O'
        return B

    @staticmethod
    def _in_bounds(size: int, r: int, c: int) -> bool:
        return 0 <= r < size and 0 <= c < size

    # ------------------------- tactics ---------------------------

    def _first_winning_move(self, board: List[List[str]], legal: Iterable[Coord], who: str) -> Optional[Coord]:
        for r, c in legal:
            if self._is_winning_placement(board, r, c, who):
                return (r, c)
        return None

    def _block_opponent_win(self, board: List[List[str]], legal: Iterable[Coord], me: str, opp: str) -> Optional[Coord]:
        # If opp has any immediate win squares, block one (preferably the one
        # that also improves our threats).
        must_blocks = []
        for r, c in legal:
            if self._is_winning_placement(board, r, c, opp):
                must_blocks.append((r, c))
        if not must_blocks:
            return None
        # pick the block that gives us the best follow-up
        best = max(must_blocks, key=lambda rc: self._threats_after(board, rc, me))
        return best

    def _filter_suicidal(self, board: List[List[str]], legal: Iterable[Coord], me: str, opp: str) -> List[Coord]:
        """Remove moves that let the opponent win instantly on their reply (unless our move also wins)."""
        legal = list(legal)
        if len(legal) <= 1:
            return legal

        safe: List[Coord] = []
        for r, c in legal:
            # If this move wins immediately, it's always safe.
            if self._is_winning_placement(board, r, c, me):
                safe.append((r, c))
                continue

            # Simulate our move; if opponent then has a winning reply, avoid it.
            self._place(board, r, c, me)
            opp_wins = self._first_winning_move(board, self._empties(board), opp)
            self._place(board, r, c, '.')  # undo

            if opp_wins is None:
                safe.append((r, c))

        return safe

    # ------------------------- evaluation ------------------------

    def _nearby_moves(self, board: List[List[str]], legal: Iterable[Coord], radius: int = 2) -> List[Coord]:
        size = len(board)
        occupied = [(r, c) for r in range(size) for c in range(size) if board[r][c] != '.']
        if not occupied:
            return list(legal)
        def is_close(rc: Coord) -> bool:
            r, c = rc
            for rr, cc in occupied:
                if abs(rr - r) <= radius and abs(cc - c) <= radius:
                    return True
            return False
        return [rc for rc in legal if is_close(rc)]

    def _heuristic_score(self, board: List[List[str]], rc: Coord, me: str, opp: str) -> float:
        """Lightweight score combining extension, forks, center bias."""
        r, c = rc
        if self._is_winning_placement(board, r, c, me):
            return 1e9  # win now

        size = len(board)
        center = (size - 1) / 2.0
        center_bias = -0.1 * (abs(r - center) + abs(c - center))

        self._place(board, r, c, me)
        my_best_line = self._best_line_through(board, r, c, me)
        # crude “fork potential”: how many lines reach 4 with at least one open end
        forkish = self._count_open_fours(board, me)

        # block value: does this move reduce opponent’s best line?
        before_opp = self._global_best_line(board, opp)
        # undo, place opp stone to measure baseline, then compute after our placement
        self._place(board, r, c, '.')
        baseline_opp = before_opp
        self._place(board, r, c, me)
        after_opp = self._global_best_line(board, opp)
        block_gain = max(0, baseline_opp - after_opp)
        self._place(board, r, c, '.')  # restore

        return 30 * (my_best_line == 4) + 10 * block_gain + 4 * my_best_line + 2.5 * forkish + center_bias

    # ------------------------- LLM -------------------------------

    async def _ask_llm_move(
        self,
        game_state: GameState,
        board: List[List[str]],
        me: str,
        opp: str,
        candidates: List[Coord],
    ) -> Optional[Coord]:

        board_str = game_state.format_board("standard")  # human-readable
        cand_json = [{"row": r, "col": c} for r, c in candidates]

        system_prompt = f"""
You are a concise, ruthless Gomoku (five-in-a-row) expert on an 8×8 board.
You play as '{me}' and the opponent is '{opp}'. Empty cells are '.'.

PRIORITY LADDER (higher beats lower):
1) If any move wins immediately (makes five in a row in any direction), play it.
2) Otherwise, if the opponent can win next turn anywhere, BLOCK that exact square.
3) Prefer moves that create TWO simultaneous threats (forks), especially open-fours.
4) If no fork, extend your longest line (to 3→4) with at least one open end.
5) Break opponent open-fours and open-threes tactically (don’t over-defend).
6) Keep moves near the current battle; mild center preference only if all else equal.

OUTPUT RULES (STRICT):
- Pick exactly ONE from the provided candidate list.
- Return ONLY valid JSON: {{"row": <int>, "col": <int>}}
- No prose. No comments. No code blocks.
""".strip()

        user_prompt = (
            f"Current board:\n{board_str}\n\n"
            f"Candidates (must choose one):\n{json.dumps(cand_json)}\n\n"
            f"Respond with JSON only."
        )

        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        content = await self.llm.complete(msgs)

        # Robustly extract {"row": r, "col": c}
        m = re.search(r"{\s*\"row\"\s*:\s*(\d+)\s*,\s*\"col\"\s*:\s*(\d+)\s*}", content)
        if not m:
            return None
        r, c = int(m.group(1)), int(m.group(2))
        # ensure it is one of the candidates we offered
        if (r, c) in candidates:
            return (r, c)
        return None

    # ------------------------- line logic ------------------------

    def _is_winning_placement(self, board: List[List[str]], r: int, c: int, who: str) -> bool:
        if board[r][c] != '.':
            return False
        self._place(board, r, c, who)
        win = self._has_five(board, r, c, who)
        self._place(board, r, c, '.')
        return win

    def _has_five(self, board: List[List[str]], r: int, c: int, who: str) -> bool:
        # check 4 directions through (r,c)
        return any(self._count_dir(board, r, c, dr, dc, who) >= 5
                   for dr, dc in [(1,0),(0,1),(1,1),(1,-1)])

    def _count_dir(self, board, r, c, dr, dc, who) -> int:
        size = len(board)
        cnt = 1
        # forward
        rr, cc = r + dr, c + dc
        while self._in_bounds(size, rr, cc) and board[rr][cc] == who:
            cnt += 1; rr += dr; cc += dc
        # backward
        rr, cc = r - dr, c - dc
        while self._in_bounds(size, rr, cc) and board[rr][cc] == who:
            cnt += 1; rr -= dr; cc -= dc
        return cnt

    def _best_line_through(self, board, r, c, who) -> int:
        return max(self._count_dir(board, r, c, dr, dc, who) for dr, dc in [(1,0),(0,1),(1,1),(1,-1)])

    def _global_best_line(self, board, who) -> int:
        size = len(board)
        best = 0
        for r in range(size):
            for c in range(size):
                if board[r][c] == who:
                    for dr, dc in [(1,0),(0,1),(1,1),(1,-1)]:
                        best = max(best, self._streak_len(board, r, c, dr, dc, who))
        return best

    def _streak_len(self, board, r, c, dr, dc, who) -> int:
        # length of contiguous run starting at (r,c) forward direction only
        size = len(board)
        if board[r][c] != who:
            return 0
        length = 0
        rr, cc = r, c
        while self._in_bounds(size, rr, cc) and board[rr][cc] == who:
            length += 1
            rr += dr; cc += dc
        return length

    def _count_open_fours(self, board, who) -> int:
        """Very rough count of open-4 patterns for 'who' after a tentative move."""
        size = len(board)
        total = 0
        for r in range(size):
            for c in range(size):
                for dr, dc in [(1,0),(0,1),(1,1),(1,-1)]:
                    total += self._is_open_four(board, r, c, dr, dc, who)
        return total

    def _is_open_four(self, board, r, c, dr, dc, who) -> int:
        """Return 1 if a sequence of exactly four with at least one open end occurs starting at (r,c)."""
        size = len(board)
        cells = []
        rr, cc = r, c
        for _ in range(4):
            if not self._in_bounds(size, rr, cc):
                return 0
            cells.append(board[rr][cc])
            rr += dr; cc += dc
        if cells.count(who) != 4:
            return 0
        # check ends
        left_r, left_c = r - dr, c - dc
        right_r, right_c = rr, cc
        left_open = self._in_bounds(size, left_r, left_c) and board[left_r][left_c] == '.'
        right_open = self._in_bounds(size, right_r, right_c) and board[right_r][right_c] == '.'
        return 1 if (left_open or right_open) else 0

    def _threats_after(self, board, rc: Coord, me: str) -> float:
        """Place ME on rc and estimate threat growth (used to pick better blocks)."""
        r, c = rc
        self._place(board, r, c, me)
        v = self._count_open_fours(board, me) + self._global_best_line(board, me) * 0.5
        self._place(board, r, c, '.')
        return v

    def _place(self, board, r, c, ch):
        board[r][c] = ch

    def _empties(self, board) -> List[Coord]:
        size = len(board)
        return [(r, c) for r in range(size) for c in range(size) if board[r][c] == '.']