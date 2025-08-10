# Import necessary modules for regular expressions, JSON parsing, and the Gomoku framework
import re
import json
from gomoku import Agent
from gomoku.llm import OpenAIGomokuClient
from gomoku.core.models import Player


class GomokuAgent(Agent):
    """
    A Gomoku AI agent that uses a language model to make strategic moves.
    Inherits from the base Agent class provided by the Gomoku framework.
    """

    def _setup(self):
        """
        Initialize the agent by setting up the language model client.
        This method is called once when the agent is created.
        """
        # Create an OpenAI-compatible client using the Gemma2 model for move generation
        self.llm = OpenAIGomokuClient(
            model="qwen/qwen-2.5-7b-instruct"
        )

    def _create_system_prompt(self, game_state, player, rival) -> str:
        """Create the system prompt that teaches the LLM how to play Gomoku."""
        return f"""
You are an expert Gomoku (Five-in-a-Row) player. You are {player}, your opponent is {rival}.
Your task is to choose exactly one best move for {player} based on the current board, following this strategies:
1. Control the center of the board early.
2. If a move creates a five-in-a-row for {player}, choose it.
3. If {rival} can win next turn (e.g., open four or equivalent threat), block it.
4. If possible, choose a move that creates two or more simultaneous winning threats (e.g., two open fours).
5. If no double threat exists, choose the move that creates the most powerful single threat, forcing {rival} to defend and setting up a future win.

Output Rules:
- The move must be on an empty square (marked as '.')
- The row and col must be valid coordinates on the board (0-indexed)
- Output only valid JSON in the exact format below.
- No explanation, reasoning, or extra text. JSON only.

Format:
```json
{{"row": <row_number>, "col": <col_number>}}
```

Examples:
```json
{{"row": 5, "col": 2}}
```
```json
{{"row": 1, "col": 2}}
```
```json
{{"row": 0, "col": 3}}
```
""".strip()

    async def get_move(self, game_state):
        """
        Generate the next move for the current game state using an LLM.

        Args:
            game_state: Current state of the Gomoku game board

        Returns:
            tuple: (row, col) coordinates of the chosen move
        """
        # Get the current player's symbol (e.g., 'X' or 'O')
        player = self.player.value

        # Determine the opponent's symbol by checking which player we are
        rival = (Player.WHITE if self.player == Player.BLACK else Player.BLACK).value

        # Convert the game board to a human-readable string format
        board_str = game_state.format_board("standard")

        board_prompt = f"Current board state:\n{board_str}\n"
        board_prompt += f"Current player: {game_state.current_player.value}\n"
        board_prompt += f"Move count: {len(game_state.move_history)}\n"
        # board_size = game_state.board_size
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
        return game_state.get_legal_moves()[-1]