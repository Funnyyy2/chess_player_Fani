import chess
import random
import re
import torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

from chess_tournament import Player 
class TransformerPlayer(Player):
    """
    Tiny LM baseline chess player.
    """

    UCI_REGEX = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", re.IGNORECASE)

    def __init__(
        self,
        name: str = "TinyLMPlayer",
        model_id: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
        temperature: float = 0.7,
        max_new_tokens: int = 8,
    ):
        super().__init__(name)
        self.model_id = model_id
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None

    def _load_model(self):
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
            self.model.to(self.device)
            self.model.eval()

    def _build_prompt(self, fen: str) -> str:
        return f"FEN: {fen}\nMove:"

    def _extract_move(self, text: str) -> Optional[str]:
        match = self.UCI_REGEX.search(text)
        return match.group(1).lower() if match else None

    def _random_legal_from_board(self, board: chess.Board) -> Optional[str]:
        moves = list(board.legal_moves)
        return random.choice(moves).uci() if moves else None

    def get_move(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)
        if not any(board.legal_moves):
            return None

        try:
            self._load_model()
        except Exception:
            return self._random_legal_from_board(board)

        prompt = self._build_prompt(fen)
        
        for _ in range(3):
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=True,
                        temperature=self.temperature,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

                decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                if decoded.startswith(prompt):
                    decoded = decoded[len(prompt):]

                move = self._extract_move(decoded)
                if not move:
                    continue
                try:
                    uci_move = chess.Move.from_uci(move)
                except ValueError:
                    continue

                if uci_move in board.legal_moves:
                    return move

            except Exception:
                continue

        return self._random_legal_from_board(board)
