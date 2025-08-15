import random
from typing import Tuple, List
from gomoku.agents.openai_llm_agent import LLMGomokuAgent
from gomoku.core.models import GameState, Player

class PromptEngineerV3(LLMGomokuAgent):
    """
    A time-optimized hybrid agent that uses a fast heuristic scan to select a
    small number of candidate moves, then runs a deep Minimax search only on those
    candidates, ensuring it performs well within the 20-second time limit.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.depth = 3 # Can now afford a deeper search
        self.candidate_move_count = 7 # Only consider the top 7 moves

    def _get_system_prompt(self, game_state: GameState) -> str:
        # This prompt is fast and provides general guidance.
        return """You are a Gomoku grandmaster. Analyze the board. Recommend a general strategy: 'ATTACK' or 'DEFEND'. Respond with a single word. Example: ATTACK"""

    async def get_move(self, game_state: GameState) -> Tuple[int, int]:
        board = game_state.board
        legal_moves = self._get_legal_moves(board)

        if not legal_moves: return None
        # CORRECTED: Use len() to get the size of a standard Python list
        if len(legal_moves) == len(board) * len(board[0]):
            return (len(board) // 2, len(board) // 2)

        self.player = game_state.current_player.value
        self.opponent = Player.WHITE.value if self.player == Player.BLACK.value else Player.BLACK.value
        
        # 1. Get a quick strategic hint from the LLM
        llm_strategy = await self._get_llm_strategy(game_state)

        # 2. Generate a short list of the best candidate moves via a fast heuristic scan
        candidate_moves = self._get_candidate_moves(board, legal_moves, llm_strategy)

        best_score = -float('inf')
        best_move = candidate_moves[0] # Fallback to the best heuristic move

        # 3. Run the deep Minimax search ONLY on the short list of candidates
        for move in candidate_moves:
            row, col = move
            board[row][col] = self.player
            score = self.minimax(board, self.depth - 1, -float('inf'), float('inf'), False)
            board[row][col] = Player.EMPTY.value
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move

    async def _get_llm_strategy(self, game_state: GameState) -> str:
        try:
            return await super()._get_llm_response(game_state)
        except:
            return "ATTACK" # Fallback strategy

    def _get_candidate_moves(self, board, legal_moves, llm_strategy) -> List[Tuple[int, int]]:
        """
        Performs a fast, non-recursive scan to find the most promising moves.
        """
        move_scores = {}
        for move in legal_moves:
            score = 0
            # Score based on immediate threats for me
            score += self._get_heuristic_score(board, move, self.player) * 1.1 # Prioritize offense
            # Score based on immediate threats for my opponent (blocking)
            score += self._get_heuristic_score(board, move, self.opponent)
            
            # Add a small bonus if the move aligns with the LLM's general strategy
            if llm_strategy == "ATTACK":
                score += self._get_heuristic_score(board, move, self.player) * 0.1

            move_scores[move] = score

        # Sort moves by their heuristic score in descending order
        sorted_moves = sorted(move_scores, key=move_scores.get, reverse=True)
        return sorted_moves[:self.candidate_move_count]

    def _get_heuristic_score(self, board, move, player):
        """Calculates a simple score for a move without recursion."""
        row, col = move
        board[row][col] = player
        score = self._evaluate_all_lines(board, player)
        board[row][col] = Player.EMPTY.value
        return score

    # --- Minimax and Evaluation Functions (remain the same) ---

    def minimax(self, board, depth, alpha, beta, is_maximizing):
        if depth == 0 or self._is_game_over(board):
            return self._evaluate_board(board)
        # For performance, we run minimax on a limited set of best moves for sub-trees too
        legal_moves = self._get_candidate_moves(board, self._get_legal_moves(board), "ATTACK")
        if is_maximizing:
            max_eval = -float('inf')
            for move in legal_moves:
                r, c = move; board[r][c] = self.player
                evaluation = self.minimax(board, depth - 1, alpha, beta, False)
                board[r][c] = Player.EMPTY.value
                max_eval = max(max_eval, evaluation)
                alpha = max(alpha, evaluation)
                if beta <= alpha: break
            return max_eval
        else:
            min_eval = float('inf')
            for move in legal_moves:
                r, c = move; board[r][c] = self.opponent
                evaluation = self.minimax(board, depth - 1, alpha, beta, True)
                board[r][c] = Player.EMPTY.value
                min_eval = min(min_eval, evaluation)
                beta = min(beta, evaluation)
                if beta <= alpha: break
            return min_eval

    def _evaluate_board(self, board):
        score = self._evaluate_all_lines(board, self.player) - self._evaluate_all_lines(board, self.opponent) * 1.1
        return score

    def _evaluate_all_lines(self, board, player):
        score = 0; board_size = len(board)
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for r in range(board_size):
            for c in range(board_size):
                for dr, dc in directions:
                    if 0 <= r + 4 * dr < board_size and 0 <= c + 4 * dc < board_size:
                        line = [board[r + i * dr][c + i * dc] for i in range(5)]
                        score += self._score_line(line, player)
        return score

    def _score_line(self, line, player):
        my_stones = line.count(player); empty_stones = line.count(Player.EMPTY.value)
        if my_stones == 5: return 100000
        if my_stones == 4 and empty_stones == 1: return 5000
        if my_stones == 3 and empty_stones == 2: return 500
        if my_stones == 2 and empty_stones == 3: return 50
        return 0

    def _get_legal_moves(self, board):
        return [(r, c) for r in range(len(board)) for c in range(len(board[0])) if board[r][c] == Player.EMPTY.value]

    def _is_game_over(self, board):
        return abs(self._evaluate_board(board)) >= 100000 or not self._get_legal_moves(board)