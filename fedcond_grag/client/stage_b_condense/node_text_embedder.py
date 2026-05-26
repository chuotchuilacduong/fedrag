"""Frozen local node text embedding utilities for Stage B condensation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn


@dataclass
class NodeTextBank:
    """Local cache of node-level and chunk-level text embeddings."""

    node_embeddings: Tensor
    chunk_embeddings: list[Tensor]
    encoder_name: str
    dim: int

    @property
    def num_nodes(self) -> int:
        return int(self.node_embeddings.size(0))


class HashTextEncoder(nn.Module):
    """Deterministic dependency-free encoder used when SBERT is unavailable."""

    def __init__(self, dim: int = 384):
        super().__init__()
        self.dim = int(dim)

    def encode(self, texts: Sequence[str], convert_to_tensor: bool = True, **_: object) -> Tensor:
        rows = [self._encode_one(text) for text in texts]
        out = torch.stack(rows, dim=0) if rows else torch.empty((0, self.dim))
        return out if convert_to_tensor else out.numpy()

    def _encode_one(self, text: str) -> Tensor:
        vec = torch.zeros(self.dim, dtype=torch.float32)
        tokens = re.findall(r"\w+", text.lower())
        if not tokens:
            return vec
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, byteorder="little", signed=False)
            idx = value % self.dim
            sign = 1.0 if ((value >> 8) & 1) == 0 else -1.0
            vec[idx] += sign
        return torch.nn.functional.normalize(vec, p=2, dim=0)


def load_frozen_encoder(encoder_name: str = "all-MiniLM-L6-v2", dim: int = 384) -> nn.Module:
    """Load a frozen SentenceTransformer if available, else a hash encoder."""

    try:
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(encoder_name)
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
        encoder.eval()
        return encoder
    except Exception:
        encoder = HashTextEncoder(dim=dim)
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
        return encoder


def chunk_text(text: str, *, max_tokens: int = 64) -> list[str]:
    """Simple token-window chunker; keeps Stage B independent of tokenizer deps."""

    tokens = text.split()
    if not tokens:
        return [""]
    return [" ".join(tokens[i : i + max_tokens]) for i in range(0, len(tokens), max_tokens)]


@torch.no_grad()
def encode_texts(encoder: nn.Module, texts: Sequence[str], device: torch.device | None = None) -> Tensor:
    """Encode text with either SentenceTransformer or the hash fallback."""

    if hasattr(encoder, "encode"):
        embeddings = encoder.encode(list(texts), convert_to_tensor=True, show_progress_bar=False)
    else:
        raise TypeError("encoder must expose an encode(texts, convert_to_tensor=True) method")
    if not torch.is_tensor(embeddings):
        embeddings = torch.as_tensor(embeddings)
    embeddings = embeddings.detach().float()
    if device is not None:
        embeddings = embeddings.to(device)
    return embeddings


@torch.no_grad()
def build_text_bank(
    node_texts: Sequence[str],
    *,
    encoder: nn.Module | None = None,
    encoder_name: str = "all-MiniLM-L6-v2",
    dim: int = 384,
    max_chunk_tokens: int = 64,
    device: torch.device | None = None,
    batch_size: int = 4096,
) -> NodeTextBank:
    """Build a frozen local text bank for all nodes — fully batched for speed."""

    encoder = encoder or load_frozen_encoder(encoder_name=encoder_name, dim=dim)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    encoder.eval()

    n = len(node_texts)
    if n == 0:
        return NodeTextBank(
            node_embeddings=torch.empty((0, dim), dtype=torch.float32),
            chunk_embeddings=[],
            encoder_name=encoder_name,
            dim=dim,
        )

    # 1. Chunk every node text, track which chunks belong to which node
    all_chunks: list[str] = []
    node_chunk_counts: list[int] = []
    for text in node_texts:
        chunks = chunk_text(str(text), max_tokens=max_chunk_tokens)
        all_chunks.extend(chunks)
        node_chunk_counts.append(len(chunks))

    # 2. Encode ALL chunks in large batches (single pass, GPU-efficient)
    # Move each batch to CPU immediately to avoid accumulating on GPU
    all_embeddings_list: list[Tensor] = []
    for start in range(0, len(all_chunks), batch_size):
        batch = all_chunks[start : start + batch_size]
        emb = encode_texts(encoder, batch, device=device).cpu()
        all_embeddings_list.append(emb)
    all_embeddings: Tensor = torch.cat(all_embeddings_list, dim=0)  # [total_chunks, d] on CPU

    # 3. Reassemble per-node chunk tensors and compute mean pooling
    chunk_embeddings: list[Tensor] = []
    node_rows: list[Tensor] = []
    offset = 0
    for count in node_chunk_counts:
        node_embs = all_embeddings[offset : offset + count]
        offset += count
        chunk_embeddings.append(node_embs)
        node_rows.append(node_embs.mean(dim=0))

    node_embeddings = torch.stack(node_rows, dim=0)
    return NodeTextBank(
        node_embeddings=node_embeddings,
        chunk_embeddings=chunk_embeddings,
        encoder_name=encoder_name,
        dim=int(node_embeddings.size(1)),
    )


def save_text_bank(bank: NodeTextBank, path: str | Path) -> None:
    """Save local-only text embeddings."""

    payload = {
        "node_embeddings": bank.node_embeddings.detach().cpu(),
        "chunk_embeddings": [chunks.detach().cpu() for chunks in bank.chunk_embeddings],
        "encoder_name": bank.encoder_name,
        "dim": bank.dim,
    }
    torch.save(payload, Path(path))


def load_text_bank(path: str | Path, *, device: torch.device | None = None) -> NodeTextBank:
    """Load a local node text bank cache."""

    payload = torch.load(Path(path), map_location=device or "cpu")
    return NodeTextBank(
        node_embeddings=payload["node_embeddings"].to(device) if device is not None else payload["node_embeddings"],
        chunk_embeddings=[
            chunks.to(device) if device is not None else chunks for chunks in payload["chunk_embeddings"]
        ],
        encoder_name=str(payload["encoder_name"]),
        dim=int(payload["dim"]),
    )
