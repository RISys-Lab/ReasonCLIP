#!/usr/bin/env python3
"""LocCa/SigLIP2 RefCOCO decoder and coordinate protocol."""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Sequence

import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import nn


BOX_BINS = 500
TASK_PREFIX = "aref:"
_BRACKET_BOX = re.compile(
    r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]"
)
_INTEGER = re.compile(r"-?\d+")


def lecun_normal_(tensor: torch.Tensor, fan_in: int) -> None:
    """Flax/JAX LeCun truncated-normal initialization."""

    target_std = 1.0 / math.sqrt(fan_in)
    untruncated_std = target_std / 0.87962566103423978
    nn.init.trunc_normal_(
        tensor,
        mean=0.0,
        std=untruncated_std,
        a=-2.0 * untruncated_std,
        b=2.0 * untruncated_std,
    )


class C4Tokenizer:
    """C4 English 32k SentencePiece wrapper matching Big Vision tokenization."""

    def __init__(self, model_path: Path, max_length: int = 64) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        self.model_path = model_path.resolve()
        digest = hashlib.sha256()
        with self.model_path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        self.model_sha256 = digest.hexdigest()
        self.processor = spm.SentencePieceProcessor(model_file=str(self.model_path))
        self.max_length = int(max_length)
        self.vocab_size = int(self.processor.vocab_size())
        self.pad_id = int(self.processor.piece_to_id("<pad>"))
        self.eos_id = int(self.processor.eos_id())
        self.unk_id = int(self.processor.unk_id())
        if self.vocab_size != 32_000:
            raise RuntimeError(f"Expected C4 vocabulary size 32000, got {self.vocab_size}")
        if self.pad_id < 0 or self.eos_id < 0:
            raise RuntimeError(
                f"C4 tokenizer lacks pad/eos IDs: pad={self.pad_id}, eos={self.eos_id}"
            )

    @staticmethod
    def normalize(text: str) -> str:
        return text.lower()

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        ids = list(self.processor.encode(self.normalize(text), out_type=int))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def encode_sticky(self, text: str) -> list[int]:
        ids = self.encode(text, add_eos=True)
        if len(ids) >= self.max_length:
            ids = ids[: self.max_length - 1] + [self.eos_id]
        return ids + [self.pad_id] * (self.max_length - len(ids))

    def decode(self, token_ids: Sequence[int]) -> str:
        clean = []
        for token_id in token_ids:
            value = int(token_id)
            if value == self.eos_id:
                break
            if value != self.pad_id:
                clean.append(value)
        return self.processor.decode(clean)


def quantize_box_xywh(
    box_xywh: Sequence[float],
    width: int,
    height: int,
    bins: int = BOX_BINS,
) -> tuple[int, int, int, int]:
    """Return LocCa coordinates in [left, bottom, right, top] order."""

    if len(box_xywh) != 4:
        raise ValueError(f"Expected xywh box, got {box_xywh}")
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size {width}x{height}")
    x, y, box_w, box_h = (float(value) for value in box_xywh)
    normalized = (
        x / width,
        (y + box_h) / height,
        (x + box_w) / width,
        y / height,
    )
    return tuple(max(0, min(bins, round(value * bins))) for value in normalized)


def box_string(box_lbrt: Sequence[int]) -> str:
    if len(box_lbrt) != 4:
        raise ValueError(f"Expected four LocCa coordinates, got {box_lbrt}")
    left, bottom, right, top = (int(value) for value in box_lbrt)
    return f"[{left}, {bottom}, {right}, {top}]"


def target_string(expression: str, box_lbrt: Sequence[int]) -> str:
    return f"{TASK_PREFIX} {expression.strip()} : {box_string(box_lbrt)}"


def prompt_string(expression: str) -> str:
    return f"{TASK_PREFIX} {expression.strip()} : "


def parse_box_string(text: str, bins: int = BOX_BINS) -> tuple[int, int, int, int] | None:
    match = _BRACKET_BOX.search(text)
    if match is not None:
        values = tuple(int(item) for item in match.groups())
    else:
        integers = [int(item) for item in _INTEGER.findall(text)]
        if len(integers) < 4:
            return None
        values = tuple(integers[-4:])
    left, bottom, right, top = values
    if not all(0 <= value <= bins for value in values):
        return None
    if right <= left or bottom <= top:
        return None
    return left, bottom, right, top


def dequantize_box_lbrt(
    box_lbrt: Sequence[int],
    width: int,
    height: int,
    bins: int = BOX_BINS,
) -> tuple[float, float, float, float]:
    """Convert [left, bottom, right, top] to pixel xyxy."""

    left, bottom, right, top = (float(value) for value in box_lbrt)
    return (
        left / bins * width,
        top / bins * height,
        right / bins * width,
        bottom / bins * height,
    )


def xywh_to_xyxy(box_xywh: Sequence[float]) -> tuple[float, float, float, float]:
    x, y, width, height = (float(value) for value in box_xywh)
    return x, y, x + width, y + height


def box_iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(value) for value in a)
    bx1, by1, bx2, by2 = (float(value) for value in b)
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def shift_right(labels: torch.Tensor, pad_id: int) -> torch.Tensor:
    shifted = torch.full_like(labels, int(pad_id))
    shifted[:, 1:] = labels[:, :-1]
    return shifted


class LocCaDecoderLayer(nn.Module):
    """Pre-LN self-attention, cross-attention, and GELU MLP block."""

    def __init__(
        self,
        hidden_size: int,
        vision_size: int,
        num_heads: int,
        mlp_size: int,
        dropout: float,
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6, bias=use_bias)
        self.self_attention = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            bias=False,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6, bias=use_bias)
        self.cross_attention = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            bias=False,
            kdim=vision_size,
            vdim=vision_size,
            batch_first=True,
        )
        self.norm3 = nn.LayerNorm(hidden_size, eps=1e-6, bias=use_bias)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_size, bias=use_bias),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(mlp_size, hidden_size, bias=use_bias),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        vision_tokens: torch.Tensor,
        causal_mask: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        normalized = self.norm1(hidden_states)
        attended = self.self_attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        hidden_states = hidden_states + self.dropout(attended)
        normalized = self.norm2(hidden_states)
        attended = self.cross_attention(
            normalized,
            vision_tokens,
            vision_tokens,
            need_weights=False,
        )[0]
        hidden_states = hidden_states + self.dropout(attended)
        return hidden_states + self.mlp(self.norm3(hidden_states))


class LocCaDecoder(nn.Module):
    """Randomly initialized six-layer Base decoder from the LocCa probe."""

    def __init__(
        self,
        vision_size: int,
        vocab_size: int = 32_000,
        max_length: int = 64,
        hidden_size: int = 768,
        num_heads: int = 12,
        mlp_size: int = 3072,
        num_layers: int = 6,
        dropout: float = 0.1,
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.max_length = int(max_length)
        self.hidden_size = int(hidden_size)
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Parameter(torch.empty(1, max_length, hidden_size))
        self.layers = nn.ModuleList(
            LocCaDecoderLayer(
                hidden_size,
                vision_size,
                num_heads,
                mlp_size,
                dropout,
                use_bias=use_bias,
            )
            for _ in range(num_layers)
        )
        self.final_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.output = nn.Linear(hidden_size, vocab_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, std=1.0)
        nn.init.normal_(self.position_embedding, std=1.0 / math.sqrt(self.hidden_size))
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.output:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.normal_(module.bias, std=1e-6)
        for layer in self.layers:
            for attention in (layer.self_attention, layer.cross_attention):
                if attention.in_proj_weight is not None:
                    for projection in attention.in_proj_weight.chunk(3, dim=0):
                        lecun_normal_(projection, attention.embed_dim)
                else:
                    lecun_normal_(attention.q_proj_weight, attention.embed_dim)
                    lecun_normal_(attention.k_proj_weight, attention.kdim)
                    lecun_normal_(attention.v_proj_weight, attention.vdim)
                lecun_normal_(attention.out_proj.weight, attention.embed_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        vision_tokens: torch.Tensor,
        input_ids: torch.Tensor,
        pad_id: int,
    ) -> torch.Tensor:
        length = input_ids.shape[1]
        if length > self.max_length:
            raise ValueError(f"Decoder length {length} exceeds maximum {self.max_length}")
        hidden_states = self.token_embedding(input_ids) + self.position_embedding[:, :length]
        causal_mask = torch.ones(length, length, dtype=torch.bool, device=input_ids.device).triu(1)
        # Big Vision uses a causal mask only. Padding follows EOS during
        # training, so active target positions cannot attend to it.
        key_padding_mask = None
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                vision_tokens,
                causal_mask,
                key_padding_mask,
            )
        return self.output(self.final_norm(hidden_states))

    @torch.no_grad()
    def generate(
        self,
        vision_tokens: torch.Tensor,
        prompt_ids: Sequence[Sequence[int]],
        pad_id: int,
        eos_id: int,
        max_new_tokens: int = 16,
    ) -> list[list[int]]:
        if vision_tokens.shape[0] != len(prompt_ids):
            raise ValueError("Prompt batch does not match vision-token batch")
        generated = [list(items) for items in prompt_ids]
        finished = [False] * len(generated)
        for _ in range(max_new_tokens):
            max_prompt = max(len(items) for items in generated)
            if max_prompt + 1 > self.max_length:
                break
            input_ids = torch.full(
                (len(generated), max_prompt + 1),
                int(pad_id),
                dtype=torch.long,
                device=vision_tokens.device,
            )
            positions = []
            for row, items in enumerate(generated):
                input_ids[row, 1 : len(items) + 1] = torch.tensor(items, device=input_ids.device)
                positions.append(len(items))
            logits = self(vision_tokens, input_ids, pad_id=pad_id)
            next_tokens = [
                int(logits[row, position].argmax().item())
                for row, position in enumerate(positions)
            ]
            for row, token_id in enumerate(next_tokens):
                if not finished[row]:
                    generated[row].append(token_id)
                    finished[row] = token_id == int(eos_id)
            if all(finished):
                break
        return generated


def decoder_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    label_smoothing: float = 0.1,
    reduction: str = "mean",
) -> torch.Tensor:
    """Big Vision weighted_softmax_xent with per-example normalization."""

    if reduction not in {"mean", "sum", "none"}:
        raise ValueError(f"Unsupported decoder loss reduction: {reduction}")
    log_probabilities = F.log_softmax(logits.float(), dim=-1)
    target_log_probabilities = log_probabilities.gather(
        dim=-1,
        index=labels.unsqueeze(-1),
    ).squeeze(-1)
    smoothing = float(label_smoothing)
    confidence = 1.0 - smoothing
    low_confidence = smoothing / max(logits.shape[-1] - 1, 1)
    token_loss = -(
        confidence * target_log_probabilities
        + low_confidence * (log_probabilities.sum(dim=-1) - target_log_probabilities)
    )
    weights = loss_mask.to(token_loss.dtype)
    per_example = (token_loss * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(2e-38)
    if reduction == "mean":
        return per_example.mean()
    if reduction == "sum":
        return per_example.sum()
    return per_example
