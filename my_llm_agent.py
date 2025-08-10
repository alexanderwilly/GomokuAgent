
import asyncio
import json
import re
from typing import Tuple, List, Dict, Optional

from gomoku.agents.base import Agent
from gomoku.core.models import GameState, Player


try:
    import torch 
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
except ImportError:
    # If the libraries are not available, the agent will fail on first use.
    torch = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore


class MyLLMGomokuAgent(Agent):
    def __init__(
        self,
        name: str,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(name)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Determine device lazily: default to GPU if available
        if device is None:
            if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
        # Internal holders for model and tokenizer
        self._model: Optional[AutoModelForCausalLM] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        # System prompt instructing the model on Gomoku rules and strategy
        self.system_prompt: str = self._build_system_prompt()

    # ------------------------------------------------------------------
    # Model initialisation
    def _load_model(self) -> None:

        if self._model is not None and self._tokenizer is not None:
            return
        if AutoModelForCausalLM is None or AutoTokenizer is None:
            raise ImportError(
                "transformers or torch is not installed. Please install them "
                "before using MyLLMGomokuAgent."
            )
        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # Load model; ``device_map='auto'`` lets transformers decide how to
        # dispatch the model across available devices.  Using ``torch_dtype``
        # ``float16`` helps reduce memory requirements.  If the evaluation
        # environment does not support float16, setting dtype to ``None`` is
        # acceptable.
        if torch is not None:
            dtype = torch.float16 if hasattr(torch, "cuda") and torch.cuda.is_available() else torch.float32
        else:
            dtype = None  # type: ignore
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=dtype, device_map="auto"
        )

    # ------------------------------------------------------------------
    # Prompt construction
    def _build_system_prompt(self) -> str:

        return (
            "You are a seasoned Gomoku (Five in a Row) player. The board is an 8×8 grid "
            "indexed from 0 to 7 along both rows and columns. X and O take turns "
            "placing their symbols on empty squares, and the first to achieve five of "
            "their own symbols consecutively (horizontally, vertically or diagonally) "
            "wins.  Always play as the current player indicated in the prompt.\n\n"
            "Key principles for strong play:\n"
            "- **Centre control:** favour moves towards the middle of the board early in the game.\n"
            "- **Threats and blocks:** create multiple lines of attack while immediately blocking any opponent four‑in‑a‑row threats.\n"
            "- **Forks:** seek positions that simultaneously threaten two or more winning lines.\n"
            "- **Spacing:** avoid spreading moves too thin; build on existing lines.\n\n"
            "You will be given the current board state with '.' representing empty squares, "
            "'X' for the first player and 'O' for the second.  Respond **only** with valid JSON enclosed in triple backticks.  The JSON object must have two keys:\n"
            "- `reasoning`: a brief explanation (one sentence) of why you chose your move.\n"
            "- `move`: an object with integer fields `row` and `col` representing your chosen coordinates.\n\n"
            "Ensure that the move is legal (the square is empty and within 0–7 on both axes).  Do not propose moves outside the board."
        )

    # ------------------------------------------------------------------
    # Message construction
    def _build_messages(self, game_state: GameState) -> List[Dict[str, str]]:

        # Format the board as a string with rows separated by newlines
        board_str = game_state.format_board(formatter="standard")
        user_content = (
            f"Current board state:\n{board_str}\n"
            f"Current player: {game_state.current_player.value}\n"
            f"Move count: {len(game_state.move_history)}\n"
        )
        if game_state.move_history:
            last_move = game_state.move_history[-1]
            user_content += f"Last move: {last_move.player.value} at ({last_move.row}, {last_move.col})\n"
        user_content += "\nPlease provide your next move in the prescribed JSON format."
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]
        return messages

    # ------------------------------------------------------------------
    # LLM generation wrapper
    def _generate_response(self, messages: List[Dict[str, str]]) -> str:

        # Ensure the model and tokenizer are ready
        self._load_model()
        # Type assertions for static type checkers
        assert self._model is not None and self._tokenizer is not None
        tokenizer: AutoTokenizer = self._tokenizer  # type: ignore
        model: AutoModelForCausalLM = self._model  # type: ignore

        # Build the input according to the chat template if supported
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fallback to simple concatenation: system first, then user
            text = "\n\n".join([
                f"[{msg['role'].upper()}] {msg['content']}" for msg in messages
            ])

        # Tokenize and move to device
        model_inputs = tokenizer(text, return_tensors="pt").to(self.device)

        # Generate response
        if torch is not None:
            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                )
        else:
            raise RuntimeError("torch is required for model generation")
        # Remove the input portion from the generated output
        generated_only_ids = generated_ids[:, model_inputs.input_ids.shape[-1] :]
        response = tokenizer.decode(
            generated_only_ids[0], skip_special_tokens=True
        ).strip()
        return response

    # ------------------------------------------------------------------
    # Response parsing
    def _parse_move_response(
        self, response: str, game_state: GameState
    ) -> Tuple[int, int]:
        try:
            # Extract JSON enclosed within ```json ... ``` markers
            match = re.search(r"```json\s*({[\s\S]*?})\s*```", response)
            json_str: Optional[str] = None
            if match:
                json_str = match.group(1)
            else:
                # If explicit code block markers are missing, try to find a brace
                brace_match = re.search(r"\{[\s\S]*\}", response)
                if brace_match:
                    json_str = brace_match.group(0)
            if json_str:
                data = json.loads(json_str)
                # Ensure the move exists
                if isinstance(data, dict) and "move" in data:
                    move = data["move"]
                    row = int(move["row"])
                    col = int(move["col"])
                    # Validate coordinates
                    if game_state.is_valid_move(row, col):
                        return (row, col)
        except Exception as e:
            # Swallow JSON errors and fall back
            print(f"Error parsing LLM response: {e}")
        # If parsing fails or move invalid, fallback
        return self._get_fallback_move(game_state)

    # ------------------------------------------------------------------
    # Fallback strategy
    def _get_fallback_move(self, game_state: GameState) -> Tuple[int, int]:
        center = game_state.board_size // 2
        if game_state.is_valid_move(center, center):
            return (center, center)
        # Use the game's helper to list legal moves
        legal_moves = game_state.get_legal_moves()
        if legal_moves:
            return legal_moves[0]
        # If there are no moves left the board is full; raise an error
        raise RuntimeError("No valid moves available")

    # ------------------------------------------------------------------
    # Main entry point
    async def get_move(self, game_state: GameState) -> Tuple[int, int]:

        try:
            messages = self._build_messages(game_state)
            # Run the synchronous generation in a background thread to avoid
            # blocking the event loop
            response = await asyncio.to_thread(self._generate_response, messages)
            return self._parse_move_response(response, game_state)
        except Exception as err:
            print(f"LLM error for agent {self.agent_id}: {err}")
            return self._get_fallback_move(game_state)