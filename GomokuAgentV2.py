# Import necessary modules for regular expressions, JSON parsing, and the Gomoku framework
import re
import json
from gomoku import Agent
from gomoku.llm import OpenAIGomokuClient

import random
from typing import Tuple, List
# from gomoku.agents.openai_llm_agent import LLMGomokuAgent
from gomoku.core.models import GameState, Player


class GomokuAgentV2(Agent):
    """
    A Gomoku AI agent that uses a language model to make strategic moves.
    Inherits from the base Agent class provided by the Gomoku framework.
    """

    def _setup(self):
        self.depth = 3
        self.candidate_move_count = 7 

        self.llm = OpenAIGomokuClient(
            model="qwen/qwen-2.5-7b-instruct"
        )

    def _create_system_prompt(self, game_state, player, rival) -> str:
        """Create the system prompt that teaches the LLM how to play Gomoku."""
        return f"""
You are an outsanding Gomoku player. These are the rules of Gomoku:
1. The board is a {game_state.board_size}x{game_state.board_size} grid.
2. Two players take turns placing their symbols.
3. The first player to align **five consecutive pieces** vertically, horizontally, or diagonally wins.
4. You cannot place a piece on an occupied cell.

Game State Input:
- Board size: {game_state.board_size} x {game_state.board_size}.
- Board Representation: an array read left to right, top to bottom. 
- {player} indicates my pieces, {rival} indicates opponent's pieces, and '.' indicates empty spaces.

Your task:
1. First Move Rule: If the board is empty (no {player} or {rival} pieces), place your move at the center: 
    - row = {game_state.board_size//2}
    - col = {game_state.board_size//2}.
2. Otherwise:
    - Analyze the current board and the last move made.
    - Predict the opponent's possible strategies.
    - Choose exactly **one** move for {player} that either maximizes winning chances or blocks immediate threats.
3. The move must be on a empty space.
4. Output only a valid JSON without explanation, 0-indexed, exactly in this format:
```json
{{"row": <row_number>, "col": <col_number>}}
```
""".strip()

    async def get_move(self, game_state: GameState) -> Tuple[int, int]:
        # Get the current player's symbol (e.g., 'X' or 'O')
        player = self.player.value

        # Determine the opponent's symbol by checking which player we are
        rival = (Player.WHITE if self.player == Player.BLACK else Player.BLACK).value

        # Convert the game board to a human-readable string format
        board_str = game_state.format_board("standard")

        board_prompt = f"Current board state:\n{board_str}\n"
        board_prompt += f"Current player: {game_state.current_player.value}\n"
        if game_state.move_history:
            last_move = game_state.move_history[-1]
            board_prompt += f"Last move: {last_move.player.value} at ({last_move.row}, {last_move.col})\n"
            

        # Prepare the conversation messages for the language model
        messages = [
            {
                "role": "system",
                "content": self._create_system_prompt(game_state, player, rival),
            },
            {
                "role": "user",
                "content": f"{board_prompt}\n\nProvide your next move as JSON without explanation.",
            },
        ]

        

        # Send the messages to the language model and get the response
        content = await self.llm.complete(messages)

        print("💡 Response:\n\n")
        print(content)
        print()


        # Parse the LLM response to extract move coordinates
        try:
            # Use regex to find JSON-like content in the response
            if m := re.search(r"{[^}]+}", content, re.DOTALL):
                # Parse the JSON to extract row and column
                move = json.loads(m.group(0))
                row, col = (move["row"], move["col"])

                # Validate that the proposed move is legal
                if game_state.is_valid_move(row, col):
                    return (row, col)
        except json.JSONDecodeError as e:
            # If JSON parsing fails, continue to fallback strategy
            pass

        # Fallback: if LLM response is invalid, choose the first available legal move
        return await self._get_fallback(game_state)

    async def _get_fallback(self, game_state: GameState) -> Tuple[int, int]:
        print("Oops, LLM response was invalid.")
        board = game_state.board
        legal_moves = self._get_legal_moves(board)
        if not legal_moves: return None

        if len(legal_moves) == len(board) * len(board[0]):
            return (len(board) // 2, len(board) // 2)
        
        self.fallback_player = game_state.current_player.value
        self.opponent = Player.WHITE.value if self.fallback_player == Player.BLACK.value else Player.BLACK.value

        # Generate a short list of the best candidate moves via a fast heuristic scan
        candidate_moves = self._get_candidate_moves(board, legal_moves)

        best_score = -float('inf')
        best_move = candidate_moves[0] # Fallback to the best heuristic move


        # Run the deep Minimax search only on the short list of candidates
        for move in candidate_moves:
            row, col = move
            board[row][col] = self.fallback_player
            score = self.minimax(board, self.depth - 1, -float('inf'), float('inf'), False)
            board[row][col] = Player.EMPTY.value
            
            if score > best_score:
                best_score = score
                best_move = move

        return best_move
    
    def _get_candidate_moves(self, board, legal_moves) -> List[Tuple[int, int]]:
        """
        Performs a fast, non-recursive scan to find the most promising moves.
        """
        move_scores = {}
        for move in legal_moves:
            score = 0
            # Score based on immediate threats for me
            score += self._get_heuristic_score(board, move, self.fallback_player) * 1.1 # Prioritize offense
            # Score based on immediate threats for my opponent (blocking)
            score += self._get_heuristic_score(board, move, self.opponent)
            
            score += self._get_heuristic_score(board, move, self.fallback_player) * 0.1

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
        legal_moves = self._get_candidate_moves(board, self._get_legal_moves(board))
        if is_maximizing:
            max_eval = -float('inf')
            for move in legal_moves:
                r, c = move; board[r][c] = self.fallback_player
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
        score = self._evaluate_all_lines(board, self.fallback_player) - self._evaluate_all_lines(board, self.opponent) * 1.1
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


# You are an expert Gomoku (Five-in-a-Row) player. You are {player}, your opponent is {rival}.
# Your task is to choose exactly one best move for {player} based on the current board, following this strategies:
# 1. Control the center of the board early.
# 2. If a move creates a five-in-a-row for {player}, choose it.
# 3. If {rival} can win next turn (e.g., open four or equivalent threat), block it.
# 4. If possible, choose a move that creates two or more simultaneous winning threats (e.g., two open fours).
# 5. If no double threat exists, choose the move that creates the most powerful single threat, forcing {rival} to defend and setting up a future win.

# Output Rules:
# - The move must be on an empty square (marked as '.')
# - The row and col must be valid coordinates on the board (0-indexed)
# - Output only valid JSON in the exact format below.
# - No explanation, reasoning, or extra text. JSON only.

# Format:
# ```json
# {{"row": <row_number>, "col": <col_number>}}
# ```