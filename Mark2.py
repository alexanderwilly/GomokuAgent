import re
import json
from typing import Tuple, List

from gomoku import Agent
from gomoku.llm import OpenAIGomokuClient
from gomoku.core.models import Player

Coord = Tuple[int, int]


class Mark2Agent(Agent):


    # ---------------- LLM client ----------------
    def _setup(self):
        self.llm = OpenAIGomokuClient(
            model="google/gemma-2-9b-it",
            temperature=0.03,
            top_p=0.15,
            max_tokens=64,        # tighter generation cap = faster
        )
        self.MAX_PASSES = 2      # keep recursion, but only 1 audit round

    # ---------------- helpers ----------------
    def _empties(self, gs) -> List[Coord]:
        n = gs.board_size
        return [(r, c) for r in range(n) for c in range(n) if gs.board[r][c] == "."]

    def _main_diag_targets(self, gs, diffs=(-1, 0, 1), limit=8) -> List[Coord]:
        """Main diagonal ↘ around center: c - r ∈ diffs; return up to 'limit' empties."""
        n = gs.board_size
        E = set(self._empties(gs))
        out: List[Coord] = []
        for d in diffs:
            for r in range(n):
                c = r + d
                if 0 <= c < n and (r, c) in E:
                    out.append((r, c))
        ctr = (n - 1) / 2
        out.sort(key=lambda rc: (abs(rc[0] - ctr) + abs(rc[1] - ctr), rc[0], rc[1]))
        return out[:limit]

    def _anti_diag_targets(self, gs, sums=(6, 7, 8), limit=8) -> List[Coord]:
        """Anti-diagonal ↙ near center on 8×8: r + c ∈ sums; return up to 'limit' empties."""
        n = gs.board_size
        E = set(self._empties(gs))
        out: List[Coord] = []
        for s in sums:
            for r in range(n):
                c = s - r
                if 0 <= c < n and (r, c) in E:
                    out.append((r, c))
        ctr = (n - 1) / 2
        out.sort(key=lambda rc: (abs(rc[0] - ctr) + abs(rc[1] - ctr), rc[0], rc[1]))
        return out[:limit]

    def _column_targets(self, gs, col=3, limit=8) -> List[Coord]:
        n = gs.board_size
        E = set(self._empties(gs))
        out = [(r, col) for r in range(n) if (r, col) in E]
        ctr = (n - 1) / 2
        out.sort(key=lambda rc: (abs(rc[0] - ctr), rc[0]))
        return out[:limit]

    # ---------------- proposer prompt ----------------
    def _system_proposer(
        self, n: int, me: str, opp: str, last_move_txt: str,
        main_diag: List[Coord], anti_diag: List[Coord], col3: List[Coord],
        i_am_white: bool,
    ) -> str:
        center_cols = [n // 2 - 1, n // 2] if n % 2 == 0 else [n // 2]
        center_rows = center_cols
        return f"""
You are a Gomoku (8×8, five-in-a-row) MOVE SELECTOR.
You are {me}; opponent {opp}. '.' empty. 0-indexed (row,col).
REPLY: ONE JSON OBJECT ONLY. No prose, no backticks.

ORDER:
A) WIN/BLOCK NOW (vertical, horizontal, ↘, ↙):
   - If a move wins NOW for {me} → choose it.
   - Else if {opp} can win NEXT turn → BLOCK it.
   - Include gap-fours: _XXXX, XXXX_, XX_XX, X_XXX, XXX_X (underscore is empty).
   - Start with lines through opponent's last move {last_move_txt}.

B) IF NONE: DOUBLECHENG PLAN
   - Dual diagonals near center:
     • ↘ main-diagonal targets c−r∈{{-1,0,1}}: {main_diag}
     • ↙ anti-diagonal targets r+c∈{{6,7,8}}: {anti_diag}
     • Prefer two open ends and creating a fork (two distinct win threats).
   - If White={i_am_white}: consider fast vertical on col 3 when it builds 4/5 or a fork: {col3}
   - Tie-breakers: diagonal completion (↘/↙) > center (rows {center_rows}, cols {center_cols})
     > adjacency to {me} stones > lowest (row,col).

FINAL SELF-CHECK:
- Re-scan for {me} WIN-NOW; if exists, output it.
- Else re-scan for {opp} BLOCK-NOW (incl. gap-fours). If your choice fails to block, override.

OUTPUT (exactly one line):
{{"row": <int>, "col": <int>}}
""".strip()

    def _user_proposer(self, gs, legal: List[Coord]) -> str:
        # Keep the board (8 lines) — small enough, but avoids long descriptions.
        return (
            f"Board {gs.board_size}x{gs.board_size}:\n"
            f"{gs.format_board('standard')}\n\n"
            f"Current: {gs.current_player.value}\n"
            f"Legal: {legal}\n"
            "Return exactly one JSON object."
        )

    # ---------------- auditor prompt ----------------
    def _system_auditor(self, me: str, opp: str) -> str:
        return f"""
You are a strict Gomoku REFEREE. JSON ONLY. No prose, no backticks.

Rules:
1) If any move wins NOW for {me} → return that move.
2) Else if {opp} can win NEXT turn and the proposed move does NOT block all such wins,
   → return a corrected BLOCKING move (consider gap-fours: _XXXX, XXXX_, XX_XX, X_XXX, XXX_X).
3) Else 1-ply sanity check: if the proposal lets {opp} win immediately next, change to the safest block.

OUTPUT (single-line JSON): {{"row": <int>, "col": <int>}}
""".strip()

    def _user_auditor(self, gs, legal: List[Coord], proposed: Coord) -> str:
        last_txt = "N/A"
        if gs.move_history:
            last = gs.move_history[-1]
            last_txt = f"({last.row}, {last.col}) by {last.player.value}"
        return (
            f"Board:\n{gs.format_board('standard')}\n\n"
            f"Current: {gs.current_player.value}\n"
            f"Last: {last_txt}\n"
            f"Legal: {legal}\n"
            f"Proposed: {proposed}\n"
            "Return final JSON."
        )

    # ---------------- parsing ----------------
    @staticmethod
    def _extract_json(text: str):
        s = text.strip()
        s = s.replace("```json", "").replace("```", "").strip()
        m = re.search(r"\{[^{}]+\}", s, flags=re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None

    # ---------------- main move ----------------
    async def get_move(self, game_state) -> Coord:
        me = self.player.value
        opp = (Player.WHITE if self.player == Player.BLACK else Player.BLACK).value

        legal = game_state.get_legal_moves()
        if not legal:
            return (0, 0)

        # Precompute compact target lists (trimmed to keep prompt small/fast)
        main_diag = self._main_diag_targets(game_state, diffs=[-1, 0, 1], limit=8)
        anti_diag = self._anti_diag_targets(game_state, sums=[6, 7, 8], limit=8)
        col3 = self._column_targets(game_state, col=3, limit=8)

        last_move_txt = "N/A"
        if game_state.move_history:
            lm = game_state.move_history[-1]
            last_move_txt = f"({lm.row}, {lm.col}) by {lm.player.value}"

        # Pass 1: propose
        sys_prop = self._system_proposer(
            game_state.board_size, me, opp, last_move_txt,
            main_diag, anti_diag, col3, i_am_white=(self.player == Player.WHITE)
        )
        user_prop = self._user_proposer(game_state, legal)
        content = await self.llm.complete(
            [{"role": "system", "content": sys_prop},
             {"role": "user", "content": user_prop}]
        )
        obj = self._extract_json(content)
        move: Coord = None  # type: ignore

        if obj and "row" in obj and "col" in obj:
            try:
                candidate = (int(obj["row"]), int(obj["col"]))
                if game_state.is_valid_move(*candidate):
                    move = candidate
            except Exception:
                move = None

        # Fallback to center-ish if parsing/validity fails
        if move is None:
            n = game_state.board_size
            centers = [(n//2, n//2), (n//2, n//2-1), (n//2-1, n//2), (n//2-1, n//2-1)]
            for rc in centers:
                if 0 <= rc[0] < n and 0 <= rc[1] < n and game_state.is_valid_move(*rc):
                    move = rc
                    break
            if move is None:
                move = legal[0]

        # Pass 2: audit
        if self.MAX_PASSES > 1:
            sys_aud = self._system_auditor(me, opp)
            user_aud = self._user_auditor(game_state, legal, move)
            check = await self.llm.complete(
                [{"role": "system", "content": sys_aud},
                 {"role": "user", "content": user_aud}]
            )
            fix = self._extract_json(check)
            if fix and "row" in fix and "col" in fix:
                try:
                    revised = (int(fix["row"]), int(fix["col"]))
                    if game_state.is_valid_move(*revised):
                        move = revised
                except Exception:
                    pass

        return move
