import numpy as np

from tok_lib import SPECIALS, build_canon_map, canon_norm, check_tokenizer, new_tokenizer, new_trainer, strip_specials

DOCS = ["Hello world. hello World! The answer is 42.\n",
        "def add(a, b):\n    return a + b\n\nprint(add(1, 2))\n",
        "Let x = 3.14159; then 2x = 6.28318. Café naïve résumé HELLO hello Hello",
        "for i in range(10):\n\tif i % 2 == 0:\n\t\tprint(i)\n"] * 50


def tiny_tokenizer(vocab=330):
    tok = new_tokenizer()
    tok.train_from_iterator(DOCS, new_trainer(vocab), length=len(DOCS))
    return tok


def test_specials_vocab_and_roundtrip():
    tok = tiny_tokenizer()
    check_tokenizer(tok, 330)
    for d in DOCS[:4]:
        assert tok.decode(tok.encode(d).ids) == d
    # digits are split one by one
    ids = tok.encode("12345").ids
    assert len(ids) == 5


def test_canon_norm_rules():
    assert canon_norm(" Hello") == canon_norm("hello") == canon_norm("HELLO") == "hello"
    assert canon_norm("Café") == "cafe"
    assert canon_norm("   ") == " " and canon_norm("\n\n") == " "


def test_canon_map_merges_case_and_space_variants():
    tok = tiny_tokenizer()
    canon, meta = build_canon_map(tok)
    assert canon.shape == (330,) and canon.dtype == np.int32
    assert meta["num_canonical"] < 330
    v = tok.get_vocab()
    variants = [t for t in ("hello", "Ġhello", "Hello", "ĠHello") if t in v]
    assert len(variants) >= 2 and len({int(canon[v[t]]) for t in variants}) == 1
    # specials keep their own ids (each special string is unique after normalisation)
    assert len({int(canon[i]) for i in range(len(SPECIALS))}) == len(SPECIALS)


def test_strip_specials():
    assert strip_specials("a<|endoftext|>b<|im_start|>c") == "abc"
    assert strip_specials("no specials <| here") == "no specials <| here"


def test_split_at_words_encodes_like_the_whole_text():
    """tokenize_to_bins --max-encode-chars: pieces cut inside "<letter> <letter>" encode to exactly the whole text's ids
    with the real tokenizer (its pre-tokenizer starts a new piece at " word"; PG19 books of up to 28M characters)."""
    import os

    import pytest
    tok_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "tokenizer", "tokenizer.json")
    if not os.path.exists(tok_path):
        pytest.skip("needs data/tokenizer/tokenizer.json")
    from tokenizers import Tokenizer
    from tokenize_to_bins import split_at_words
    tok = Tokenizer.from_file(tok_path)
    para = ("Chapter XII.\n\nIt was the best of times, it's said; 1919-11-20: 3.14159 * 42 = 131.95!  \"Café\" naïve "
            "Ångström -- don't, we'll, they're.\r\n    Indented   line\twith\ttabs...\n" * 7 + "αβγ δ ж Москва 東京 "
            "x́y é é ")
    text = (para * 300)[:150_000]
    whole = tok.encode(text).ids
    for n in (40_000, 5_000, 777):
        parts = split_at_words(text, n)
        assert "".join(parts) == text and len(parts) > 1 and max(map(len, parts)) <= n
        ids = []
        for p in parts:
            ids += tok.encode(p).ids
        assert ids == whole, n
