"""
Sentinel-LLM local demo model.
Author: Ashutosh Sahoo, M.Tech AI, NIT Silchar

A small, CPU-friendly text classifier that proves the LoRA pipeline
end-to-end without needing a GPU or a pretrained LLM:

  hashed token embedding (hashlib, vocab 4096, dim 128)
    -> mean pooling over tokens
    -> 2-layer MLP head (fc1: 128->256, fc2: 256->num_labels)
       where fc1/fc2 are the LoRA-adapted layers

The production LLM version (Qwen2.5-0.5B + PEFT QLoRA) is in
train_lora_peft.py and uses the exact same dataset.jsonl.
"""
import hashlib
import re
import torch
import torch.nn as nn

VOCAB_SIZE = 4096
EMBED_DIM = 128
HIDDEN_DIM = 256

_TOKEN_RE = re.compile(r"[a-z0-9_\.\-/:#]+")

def tokenize(text: str):
    return _TOKEN_RE.findall(text.lower())

def hash_token(token: str, vocab_size: int = VOCAB_SIZE) -> int:
    # Stable across runs / processes (unlike Python's salted hash())
    return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % vocab_size

def encode(text: str, vocab_size: int = VOCAB_SIZE):
    toks = tokenize(text)
    if not toks:
        toks = ["<empty>"]
    return [hash_token(t, vocab_size) for t in toks]

def encode_batch(texts, device="cpu"):
    """Returns (indices tensor [B, L], mask [B, L]) padded to max len in batch."""
    encoded = [encode(t) for t in texts]
    maxlen = max(len(e) for e in encoded)
    idx = torch.zeros(len(texts), maxlen, dtype=torch.long)
    mask = torch.zeros(len(texts), maxlen, dtype=torch.float)
    for i, e in enumerate(encoded):
        idx[i, :len(e)] = torch.tensor(e, dtype=torch.long)
        mask[i, :len(e)] = 1.0
    return idx.to(device), mask.to(device)


class SentinelClassifier(nn.Module):
    def __init__(self, num_labels: int, vocab_size: int = VOCAB_SIZE, embed_dim: int = EMBED_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.15)
        self.fc2 = nn.Linear(hidden_dim, num_labels)

    def forward(self, indices, mask):
        emb = self.embedding(indices)  # [B, L, D]
        mask_e = mask.unsqueeze(-1)   # [B, L, 1]
        pooled = (emb * mask_e).sum(dim=1) / mask_e.sum(dim=1).clamp(min=1.0)  # mean pooling
        h = self.dropout(self.relu(self.fc1(pooled)))
        return self.fc2(h)
