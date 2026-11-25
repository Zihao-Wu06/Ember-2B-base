"""Tokenizer definition and the Engram canonical-id map (shared by scripts and tests)."""
from __future__ import annotations

import re
import unicodedata

import numpy as np
from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

SPECIALS = ["<|endoftext|>", "<|pad|>", "<|im_start|>", "<|im_end|>",
            "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>"] + [f"<|reserved_{i}|>" for i in range(9)]
EOS_ID = 0
PAD_ID = 1
VOCAB_SIZE = 65536  # incl. the 16 specials; token ids fit uint16

# Qwen2-style split: digits one by one (helps arithmetic), letters in runs, whitespace runs kept
# (code indentation), punctuation runs with trailing newlines.
QWEN2_PATTERN = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}"
                 r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")


def new_tokenizer() -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(QWEN2_PATTERN), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    return tok


def new_trainer(vocab_size: int = VOCAB_SIZE, show_progress: bool = False) -> trainers.BpeTrainer:
    return trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=2, special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), max_token_length=24,
        show_progress=show_progress)


def strip_specials(text: str) -> str:
    """Raw documents must never contain a literal special token (it would be encoded as one)."""
    if "<|" in text:
        for s in SPECIALS:
            text = text.replace(s, "")
    return text


def check_tokenizer(tok: Tokenizer, vocab_size: int = VOCAB_SIZE) -> None:
    assert tok.get_vocab_size() == vocab_size, tok.get_vocab_size()
    for i, s in enumerate(SPECIALS):
        assert tok.token_to_id(s) == i, (s, tok.token_to_id(s))


# ---------------------------------------------------------------- Engram canonical map
_WS = re.compile(r"[ \t\r\n]+")


def canon_norm(text: str) -> str:
    """Engram demo normaliser: NFKC -> NFD -> strip accents -> lower -> collapse whitespace -> strip
    (a string that is exactly one space stays one space)."""
    t = unicodedata.normalize("NFKC", text)
    t = unicodedata.normalize("NFD", t)
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn")
    t = _WS.sub(" ", t.lower())
    return t if t == " " else t.strip()


def build_canon_map(tok: Tokenizer) -> tuple[np.ndarray, dict]:
    V = tok.get_vocab_size()
    key_to_id: dict[str, int] = {}
    canon = np.empty(V, dtype=np.int32)
    groups: dict[int, list[str]] = {}
    for i in range(V):
        text = tok.decode([i], skip_special_tokens=False)
        if "�" in text:
            key = "\x00raw:" + tok.id_to_token(i)   # partial UTF-8 byte token: never merged
        else:
            key = canon_norm(text) or text
        cid = key_to_id.setdefault(key, len(key_to_id))
        canon[i] = cid
        groups.setdefault(cid, []).append(text)
    n = len(key_to_id)
    top = sorted(groups.values(), key=len, reverse=True)[:10]
    meta = {"vocab_size": V, "num_canonical": n, "reduction": 1 - n / V,
            "largest_groups": [{"size": len(g), "examples": g[:8]} for g in top]}
    return canon, meta
