import random
from typing import Tuple, List
from gomoku.agents.base import Agent
from gomoku.core.models import GameState, Player

class PromptEngineerV2(Agent):
    """
    An advanced rule-based agent using the Minimax algorithm to look ahead,
    with robust pattern recognition and contextual defense. This agent conforms
    to the gomoku-ai framework structure.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.depth = 2 # How many moves to look ahead

    async def get_move(self, game_state: GameState) -> Tuple[int, int]:
        """
        Entry point for the agent. It uses Minimax to find the optimal move.
        """
        board = game_state.board
        legal_moves = self._get_legal_moves(board)

        if len(legal_moves) == len(board) * len(board[0]):
            center = len(board) // 2
            return (center, center)

        self.player = game_state.current_player.value
        self.opponent = Player.WHITE.value if self.player == Player.BLACK.value else Player.BLACK.value
        
        best_score = -float('inf')
        best_move = None

        for move in legal_moves:
            row, col = move
            board[row][col] = self.player # CORRECTED INDEXING
            score = self.minimax(board, self.depth - 1, -float('inf'), float('inf'), False)
            board[row][col] = Player.EMPTY.value # CORRECTED INDEXING
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move if best_move is not None else random.choice(legal_moves)

    def minimax(self, board, depth: int, alpha: float, beta: float, is_maximizing: bool) -> int:
        """
        Core of the lookahead logic using Minimax with Alpha-Beta Pruning.
        """
        if depth == 0 or self._is_game_over(board):
            return self._evaluate_board(board)

        legal_moves = self._get_legal_moves(board)

        if is_maximizing:
            max_eval = -float('inf')
            for move in legal_moves:
                row, col = move
                board[row][col] = self.player # CORRECTED INDEXING
                evaluation = self.minimax(board, depth - 1, alpha, beta, False)
                board[row][col] = Player.EMPTY.value # CORRECTED INDEXING
                max_eval = max(max_eval, evaluation)
                alpha = max(alpha, evaluation)
                if beta <= alpha: break
            return max_eval
        else: # Minimizing player
            min_eval = float('inf')
            for move in legal_moves:
                row, col = move
                board[row][col] = self.opponent # CORRECTED INDEXING
                evaluation = self.minimax(board, depth - 1, alpha, beta, True)
                board[row][col] = Player.EMPTY.value # CORRECTED INDEXING
                min_eval = min(min_eval, evaluation)
                beta = min(beta, evaluation)
                if beta <= alpha: break
            return min_eval

    def _evaluate_board(self, board) -> int:
        score = 0
        score += self._evaluate_all_lines(board, self.player)
        score -= self._evaluate_all_lines(board, self.opponent) * 1.1 # Prioritize defense
        return score

    def _evaluate_all_lines(self, board, player: str) -> int:
        score = 0
        board_size = len(board)
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for r in range(board_size):
            for c in range(board_size):
                for dr, dc in directions:
                    line_of_5 = []
                    # Check if the line of 5 stays within the board bounds
                    if 0 <= r + 4 * dr < board_size and 0 <= c + 4 * dc < board_size:
                        for i in range(5):
                            line_of_5.append(board[r + i * dr][c + i * dc]) # CORRECTED INDEXING
                        score += self._score_line(line_of_5, player)
        return score

    def _score_line(self, line: List[str], player: str) -> int:
        my_stones = line.count(player)
        empty_stones = line.count(Player.EMPTY.value)

        if my_stones == 5: return 100000
        if my_stones == 4 and empty_stones == 1: return 5000
        if my_stones == 3 and empty_stones == 2: return 500
        if my_stones == 2 and empty_stones == 3: return 50
        return 0

    def _get_legal_moves(self, board) -> List[Tuple[int, int]]:
        return [(r, c) for r in range(len(board)) for c in range(len(board[0])) if board[r][c] == Player.EMPTY.value] # CORRECTED INDEXING

    def _is_game_over(self, board) -> bool:
        score = self._evaluate_board(board)
        return abs(score) >= 100000 or not self._get_legal_moves(board)