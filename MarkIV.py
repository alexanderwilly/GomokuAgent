from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import time
import math
import hashlib
import json
import re

from gomoku import Agent
from gomoku.llm import OpenAIGomokuClient
from gomoku.core.models import GameState, Player

Coord = Tuple[int, int]
DIRS: List[Tuple[int, int]] = [(1, 0), (0, 1), (1, 1), (1, -1)] 
INF = 10_000_000


def inside(n: int, r: int, c: int) -> bool:
    return 0 <= r < n and 0 <= c < n


def clone_board(gs: GameState) -> List[List[str]]:
    return [row[:] for row in gs.board]


def board_key(board: List[List[str]], to_move: str) -> str:
    m = hashlib.blake2b(digest_size=12)
    for row in board:
        m.update("".join(row).encode())
    m.update(to_move.encode())
    return m.hexdigest()


def place(board: List[List[str]], rc: Coord, ch: str):
    r, c = rc
    board[r][c] = ch


def empty_cells(board: List[List[str]]) -> List[Coord]:
    n = len(board)
    out: List[Coord] = []
    for r in range(n):
        for c in range(n):
            if board[r][c] == ".":
                out.append((r, c))
    return out


def is_win(board: List[List[str]], ch: str) -> bool:
    n = len(board)
    for r in range(n):
        for c in range(n):
            if board[r][c] != ch:
                continue
            for dr, dc in DIRS:
                cnt = 1
                rr, cc = r + dr, c + dc
                while inside(n, rr, cc) and board[rr][cc] == ch:
                    cnt += 1
                    rr += dr
                    cc += dc
                if cnt >= 5:
                    return True
    return False


def line_len(board: List[List[str]], r: int, c: int, dr: int, dc: int, ch: str) -> Tuple[int, int]:
    """Return (stones_in_line_through_rc, open_ends)."""
    n = len(board)
    total = 1
    open_ends = 0
    # forward
    rr, cc = r + dr, c + dc
    while inside(n, rr, cc) and board[rr][cc] == ch:
        total += 1
        rr += dr
        cc += dc
    if inside(n, rr, cc) and board[rr][cc] == ".":
        open_ends += 1
    # backward
    rr, cc = r - dr, c - dc
    while inside(n, rr, cc) and board[rr][cc] == ch:
        total += 1
        rr -= dr
        cc -= dc
    if inside(n, rr, cc) and board[rr][cc] == ".":
        open_ends += 1
    return total, open_ends


def max_after_place(board: List[List[str]], rc: Coord, ch: str) -> Tuple[int, int, int]:
    """Return (best_len, open4s, threes) for ch through rc (assuming ch is at rc)."""
    r, c = rc
    best = 1
    open4s = 0
    threes = 0
    for dr, dc in DIRS:
        ln, ends = line_len(board, r, c, dr, dc, ch)
        best = max(best, ln)
        if ln == 4 and ends >= 1:
            open4s += 1
        if ln >= 3 and ends >= 1:
            threes += 1
    return best, open4s, threes


def manhattan_center_bias(n: int, rc: Coord) -> float:
    ctr = (n - 1) / 2.0
    r, c = rc
    return -(abs(r - ctr) + abs(c - ctr))


def adjacency(board: List[List[str]], rc: Coord, ch: str) -> int:
    n = len(board)
    r, c = rc
    adj = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if inside(n, rr, cc) and board[rr][cc] == ch:
                adj += 1
    return adj


def fmt_board(board: List[List[str]]) -> str:
    n = len(board)
    header = "   " + " ".join(str(c) for c in range(n))
    rows = [header]
    for r in range(n):
        rows.append(f"{r}  " + " ".join(board[r]))
    return "\n".join(rows)


class Mark4Agent(Agent):
    """
    Mark4Agent = strong tactical core (iterative deepening alpha-beta with threat ordering)
    + compact prompt to a small LLM (Gemma 2 9B IT) for candidate selection.
    """

    # ---------- config & LLM ----------
    def _setup(self):
        # Search config
        self.MAX_BEAM = 14
        self.NEIGHBOR_RADIUS = 1
        self.QUIESCENCE_MARGIN = 1
        self.DEPTHS = [2, 3, 4]
        self.MOVE_TIME_BUDGET = 18.5  # keep < 20s

        # LLM: small, deterministic, low token budget
        self.LLM_ENABLED = True
        self.LLM_MODEL = "google/gemma-2-9b-it"
        self.LLM_TOPK = 8
        self.LLM_JSON_RE = re.compile(r"\{[^{}]+\}")
        try:
            self.llm = OpenAIGomokuClient(model=self.LLM_MODEL)
        except Exception:
            # If model not available, fall back to pure search
            self.LLM_ENABLED = False
            self.llm = None

        # Tables
        self.tt: Dict[str, Tuple[int, int]] = {}
        self.killers: Dict[int, Coord] = {}

    # ---------- candidates & ordering ----------
    def _neighbor_candidates(self, board: List[List[str]]) -> List[Coord]:
        n = len(board)
        empties = set(empty_cells(board))
        occ = [(r, c) for r in range(n) for c in range(n) if board[r][c] != "."]
        if not occ:
            base = {(n // 2, n // 2), (n // 2 - 1, n // 2), (n // 2, n // 2 - 1), (n // 2 - 1, n // 2 - 1)}
            return [rc for rc in base if board[rc[0]][rc[1]] == "."]

        S: set[Coord] = set()
        R = self.NEIGHBOR_RADIUS
        for r, c in occ:
            for dr in range(-R, R + 1):
                for dc in range(-R, R + 1):
                    rr, cc = r + dr, c + dc
                    if inside(n, rr, cc) and board[rr][cc] == ".":
                        S.add((rr, cc))
        return list(S) if S else list(empties)

    def _score_move(self, board: List[List[str]], rc: Coord, me: str, opp: str) -> float:
        n = len(board)
        place(board, rc, me)
        best, open4s, threes = max_after_place(board, rc, me)
        place(board, rc, ".")
        center = manhattan_center_bias(n, rc)
        adj = adjacency(board, rc, me)

        dmain = abs(rc[1] - rc[0])
        danti = abs((rc[0] + rc[1]) - (n - 1))
        diag_bias = -0.35 * min(dmain, danti)

        s = 0.0
        if best >= 5:
            s += 50_000
        elif best == 4:
            s += 6_500
        elif best == 3:
            s += 900

        s += open4s * 1200
        s += threes * 280
        s += center * 40
        s += adj * 18
        s += diag_bias * 30
        return s

    def _order_moves(self, board: List[List[str]], moves: List[Coord], me: str, opp: str) -> List[Coord]:
        ranked = []
        killer_bonus = 5000
        killer_set = set(self.killers.values())
        for rc in moves:
            s = self._score_move(board, rc, me, opp)
            if rc in killer_set:
                s += killer_bonus
            ranked.append((s, rc))
        ranked.sort(reverse=True)
        return [rc for _, rc in ranked]

    # ---------- quick win/block checks ----------
    def _immediate_win(self, board: List[List[str]], me: str) -> Optional[Coord]:
        for r, c in empty_cells(board):
            place(board, (r, c), me)
            w = is_win(board, me)
            place(board, (r, c), ".")
            if w:
                return (r, c)
        return None

    def _opponent_wins_next(self, board: List[List[str]], opp: str, cap: int = 48) -> List[Coord]:
        res: List[Coord] = []
        k = 0
        for r, c in empty_cells(board):
            place(board, (r, c), opp)
            if is_win(board, opp):
                res.append((r, c))
            place(board, (r, c), ".")
            k += 1
            if k >= cap:
                break
        return res

    # ---------- evaluation ----------
    def _static_eval(self, board: List[List[str]], me: str, opp: str) -> int:
        n = len(board)
        score = 0
        for r in range(n):
            for c in range(n):
                ch = board[r][c]
                if ch == ".":
                    continue
                for dr, dc in DIRS:
                    ln, ends = line_len(board, r, c, dr, dc, ch)
                    if ln >= 5:
                        return INF if ch == me else -INF
                    val = 0
                    if ln == 4:
                        val = 5000 if ends >= 1 else 1200
                    elif ln == 3:
                        val = 900 if ends >= 1 else 180
                    elif ln == 2:
                        val = 120 if ends >= 1 else 20
                    if ch == me:
                        score += val
                    else:
                        score -= int(val * 0.95)
        return score

    def _quiescent(self, board: List[List[str]], last: Optional[Coord], me: str, opp: str) -> bool:
        if last is None:
            return True
        r, c = last
        best_me, o4_me, t_me = max_after_place(board, (r, c), me)
        best_opp, o4_opp, t_opp = max_after_place(board, (r, c), opp)
        return not ((best_me >= 4 or o4_me or t_me >= 2) or (best_opp >= 4 or o4_opp or t_opp >= 2))

    # ---------- alpha-beta ----------
    def _search(self, board: List[List[str]], depth: int, alpha: int, beta: int,
                me: str, opp: str, to_move: str, last: Optional[Coord],
                hard_deadline: float) -> Tuple[int, Optional[Coord]]:
        if time.time() >= hard_deadline:
            return 0, None

        if is_win(board, me):
            return INF - 1, last
        if is_win(board, opp):
            return -INF + 1, last

        if depth == 0 or self._quiescent(board, last, me, opp):
            return self._static_eval(board, me, opp), None

        key = board_key(board, to_move)
        if key in self.tt and self.tt[key][0] >= depth:
            return self.tt[key][1], None

        raw_moves = self._neighbor_candidates(board)
        legal = [(r, c) for (r, c) in raw_moves if board[r][c] == "."]
        if not legal:
            return self._static_eval(board, me, opp), None

        # immediate win for side to_move
        for rc in legal:
            place(board, rc, to_move)
            if is_win(board, to_move):
                place(board, rc, ".")
                score = INF - (5 - depth) if to_move == me else -INF + (5 - depth)
                self.tt[key] = (depth, score)
                self.killers[depth] = rc
                return score, rc
            place(board, rc, ".")

        order = self._order_moves(board, legal[: self.MAX_BEAM], me, opp)

        best_rc: Optional[Coord] = None
        if to_move == me:
            val = -INF
            for rc in order:
                if time.time() >= hard_deadline:
                    break
                place(board, rc, me)
                sc, _ = self._search(board, depth - 1, alpha, beta, me, opp, opp, rc, hard_deadline)
                place(board, rc, ".")
                if sc > val:
                    val, best_rc = sc, rc
                if val > alpha:
                    alpha = val
                if alpha >= beta:
                    self.killers[depth] = rc
                    break
            self.tt[key] = (depth, val)
            return val, best_rc
        else:
            val = INF
            for rc in order:
                if time.time() >= hard_deadline:
                    break
                place(board, rc, opp)
                sc, _ = self._search(board, depth - 1, alpha, beta, me, opp, me, rc, hard_deadline)
                place(board, rc, ".")
                if sc < val:
                    val, best_rc = sc, rc
                if val < beta:
                    beta = val
                if alpha >= beta:
                    self.killers[depth] = rc
                    break
            self.tt[key] = (depth, val)
            return val, best_rc

    # ---------- compact LLM prompt ----------
    def _system_prompt(self, me: str, opp: str) -> str:
        return (
            "You are a concise Gomoku tactician on an 8x8 board, five-in-a-row wins.\n"
            f"You play as '{me}'. Opponent is '{opp}'.\n"
            "Choose exactly ONE move from the provided CANDIDATE LIST ONLY.\n"
            "Priority order:\n"
            "1) Win immediately.\n"
            "2) If opponent can win next move, BLOCK it.\n"
            "3) Prefer double-threats (two winning lines, open-fours, or strong threes).\n"
            "4) Prefer central/diagonal proximity when choices are equal.\n"
            "Output JSON only: {\"row\": r, \"col\": c} (0-index).\n"
            "No commentary."
        )

    def _build_user_prompt(self, board: List[List[str]], me: str, opp: str,
                           cands: List[Coord]) -> str:
        # annotate each candidate with quick tags the model can use
        lines = [f"Board:\n{fmt_board(board)}", "\nCANDIDATES:"]
        for i, (r, c) in enumerate(cands):
            # tag: immediate win?
            place(board, (r, c), me)
            my_win = is_win(board, me)
            place(board, (r, c), ".")
            # tag: blocks opponent?
            place(board, (r, c), opp)
            opp_wins_if = is_win(board, opp)
            place(board, (r, c), ".")
            # heuristic
            best, o4, threes = max_after_place(board, (r, c), me)
            tags = []
            if my_win:
                tags.append("WIN")
            if opp_wins_if:
                tags.append("BLOCK")
            if o4:
                tags.append(f"open4={o4}")
            if threes >= 2:
                tags.append("double3")
            if best >= 4 and not my_win:
                tags.append("build4")
            tag_s = ", ".join(tags) if tags else "good"
            lines.append(f"- {i}: ({r}, {c})  [{tag_s}]")
        lines.append("\nReturn one JSON object only.")
        return "\n".join(lines)

    async def _llm_pick(self, board: List[List[str]], me: str, opp: str,
                        cands: List[Coord]) -> Optional[Coord]:
        if not self.LLM_ENABLED or not cands:
            return None
        sys = self._system_prompt(me, opp)
        usr = self._build_user_prompt(board, me, opp, cands)
        messages = [
            {"role": "system", "content": sys},
            {"role": "user", "content": usr},
        ]
        try:
            content = await self.llm.complete(messages)  # API wrapper provided by framework
            m = self.LLM_JSON_RE.search(content)
            if not m:
                return None
            obj = json.loads(m.group(0))
            rc = (int(obj["row"]), int(obj["col"]))
            return rc if rc in cands else None
        except Exception:
            return None

    # ---------- main ----------
    async def get_move(self, game_state: GameState) -> Coord:
        n = game_state.board_size
        me = self.player.value
        opp = (Player.WHITE if self.player == Player.BLACK else Player.BLACK).value

        board = clone_board(game_state)
        legal = game_state.get_legal_moves()
        if not legal:
            return (0, 0)

        # Check if win 
        w = self._immediate_win(board, me)
        if w and game_state.is_valid_move(*w):
            return w

        # Must block opponent's immediate win(s)
        opp_threats = self._opponent_wins_next(board, opp)
        if opp_threats:
            best_block, cover = None, -1
            for b in set(opp_threats):
                if game_state.is_valid_move(*b):
                    c = opp_threats.count(b)
                    if c > cover:
                        best_block, cover = b, c
            if best_block:
                return best_block

        # Build candidate set (ordered)
        cands = self._neighbor_candidates(board)
        cands = [rc for rc in cands if game_state.is_valid_move(*rc)]
        if not cands:
            return legal[0]
        cands = self._order_moves(board, cands[: self.MAX_BEAM], me, opp)
        shortlist = cands[: min(self.LLM_TOPK, len(cands))]

        # LLM tie-breaker (tiny prompt, satisfies assignment)
        llm_choice = await self._llm_pick(board, me, opp, shortlist)
        if llm_choice and game_state.is_valid_move(*llm_choice):
            # If LLM picked an immediate win/block among shortlist, use it right away
            place(board, llm_choice, me)
            if is_win(board, me):
                place(board, llm_choice, ".")
                return llm_choice
            place(board, llm_choice, ".")
            # Otherwise keep as PV seed and fall through to search
            cands.sort(key=lambda rc: 0 if rc == llm_choice else 1)

        # Iterative deepening alpha-beta (time bounded)
        start = time.time()
        hard_deadline = start + self.MOVE_TIME_BUDGET

        self.tt.clear()
        self.killers.clear()

        best_move: Optional[Coord] = None
        best_val = -INF

        for depth in self.DEPTHS:
            if time.time() >= hard_deadline:
                break
            val_local = -INF
            move_local: Optional[Coord] = None
            alpha, beta = -INF, INF

            for rc in cands:
                if time.time() >= hard_deadline:
                    break
                place(board, rc, me)
                sc, _ = self._search(board, depth - 1, alpha, beta, me, opp, opp, rc, hard_deadline)
                place(board, rc, ".")
                if sc > val_local:
                    val_local, move_local = sc, rc
                if sc > alpha:
                    alpha = sc

            if move_local is not None:
                best_move, best_val = move_local, val_local
                # put PV first
                cands.sort(key=lambda x: 0 if x == best_move else 1)

        if best_move and game_state.is_valid_move(*best_move):
            return best_move

        # Heuristic fallback
        scored = [(self._score_move(board, rc, me, opp), rc) for rc in shortlist]
        scored.sort(reverse=True)
        return scored[0][1] if scored else legal[0]
