import os

import numpy as np
import pytest
import torch

from moe_llm.data import MixtureLoader


def make_bins(root, sizes):
    rng = np.random.default_rng(0)
    for name, parts in sizes.items():
        os.makedirs(os.path.join(root, name), exist_ok=True)
        for i, n in enumerate(parts):
            rng.integers(0, 60000, n, dtype=np.uint16).tofile(os.path.join(root, name, f"{i:03d}.bin"))


def test_loader_shapes_shift_and_exact_resume(tmp_path):
    make_bins(tmp_path, {"a": [5000, 3000], "b": [9000]})
    w = {"a": 0.3, "b": 0.7}
    L1 = MixtureLoader(str(tmp_path), w, seq_len=64, batch_size=4, seed=1, val_tokens=1000)
    out = [L1.next_batch() for _ in range(6)]
    x, y = out[0]
    assert x.shape == (4, 64) and torch.equal(x[:, 1:], y[:, :-1])

    L2 = MixtureLoader(str(tmp_path), w, seq_len=64, batch_size=4, seed=1, val_tokens=1000)
    for _ in range(2):
        L2.next_batch()
    state = L2.state_dict()
    L3 = MixtureLoader(str(tmp_path), w, seq_len=64, batch_size=4, seed=1, val_tokens=1000)
    L3.load_state_dict(state)
    for i in range(2, 6):
        xb, _ = L3.next_batch()
        assert torch.equal(xb, out[i][0])


def test_train_and_val_are_disjoint_and_epochs_wrap(tmp_path):
    make_bins(tmp_path, {"a": [2000]})
    L = MixtureLoader(str(tmp_path), {"a": 1.0}, seq_len=32, batch_size=8, seed=0, val_tokens=500)
    data = np.fromfile(os.path.join(tmp_path, "a", "000.bin"), dtype=np.uint16).astype(np.int64)
    train_end = L.streams["a"].total - L.val_tokens["a"]
    n_win = L.n_train_windows["a"]
    seen = []
    for _ in range(3 * n_win // 8 + 2):  # > 2 epochs
        x, y = L.next_batch()
        for row in torch.cat([x, y[:, -1:]], 1).numpy():
            start = int(np.flatnonzero(np.all(np.lib.stride_tricks.sliding_window_view(data, 33) == row, axis=1))[0])
            assert start + 33 <= train_end
            seen.append(start)
    assert L.state["epoch"]["a"] >= 2
    vx, vy = next(L.val_batches("a", 1))
    assert vx.shape[1] == 32 and torch.equal(vx[0], torch.from_numpy(data[train_end:train_end + 32]))


def test_multiple_roots_are_merged(tmp_path):
    make_bins(tmp_path / "p1", {"a": [3000], "b": [3000]})
    make_bins(tmp_path / "p2", {"a": [4000]})
    one = MixtureLoader(str(tmp_path / "p1"), {"a": 1.0, "b": 1.0}, seq_len=32, batch_size=2)
    both = MixtureLoader(f"{tmp_path / 'p1'},{tmp_path / 'p2'}", {"a": 1.0, "b": 1.0}, seq_len=32, batch_size=2)
    assert both.streams["a"].total == one.streams["a"].total + 4000
    assert both.streams["b"].total == one.streams["b"].total
    x, _ = both.next_batch()
    assert x.shape == (2, 32)


def _drawn(L, name):
    """Training windows taken from `name` so far (survives epoch wrap-around)."""
    return L.state["epoch"][name] * L.n_train_windows[name] + L.state["cursor"][name]


def test_anneal_switches_mixture_at_step_and_resumes_exactly(tmp_path):
    make_bins(tmp_path, {"a": [20000], "b": [20000]})
    kw = dict(seq_len=16, batch_size=8, seed=3, val_tokens=1000)
    base, ann = {"a": 0.9, "b": 0.1}, {"a": 0.1, "b": 0.9}
    plain = MixtureLoader(str(tmp_path), base, **kw)
    L = MixtureLoader(str(tmp_path), base, anneal_weights=ann, anneal_from=50, **kw)
    out = []
    for step in range(100):
        out.append(L.next_batch()[0])
        if step < 50:   # before the switch: identical to a loader without annealing
            assert torch.equal(out[-1], plain.next_batch()[0])
        if step == 49:
            a_before = _drawn(L, "a")
    a_after = _drawn(L, "a") - a_before
    assert a_before > 320 and a_after < 80, (a_before, a_after)   # expected 360 and 40 of 400 rows
    S = MixtureLoader(str(tmp_path), base, anneal_weights=ann, anneal_from=50, **kw)
    for _ in range(45):
        S.next_batch()
    R = MixtureLoader(str(tmp_path), base, anneal_weights=ann, anneal_from=50, **kw)
    R.load_state_dict(S.state_dict())
    for i in range(45, 60):   # resumed before the switch, reproduces the batches across it
        assert torch.equal(R.next_batch()[0], out[i])


def test_anneal_mix_may_add_sources(tmp_path):
    make_bins(tmp_path, {"a": [5000], "b": [5000]})
    L = MixtureLoader(str(tmp_path), {"a": 1.0}, seq_len=16, batch_size=4, anneal_weights={"b": 1.0}, anneal_from=3)
    assert L.names == ["a", "b"]
    for step in range(6):
        L.next_batch()
        assert (_drawn(L, "a"), _drawn(L, "b")) == (4 * min(step + 1, 3), 4 * max(0, step - 2))


def test_resume_old_state_with_new_anneal_only_sources(tmp_path):
    """Main run: the anneal mix gains a source after training started. A checkpoint written without it
    resumes; batches before the switch are unchanged, and from then on the run equals a loader that had
    the new anneal mix from step 0 (so nothing trained so far is invalidated)."""
    make_bins(tmp_path, {"a": [20000], "b": [20000], "c": [20000], "d": [20000]})
    kw = dict(seq_len=16, batch_size=8, seed=3, val_tokens=1000)
    base = {"a": 0.6, "b": 0.3, "c": 0.1}
    old_ann = {"a": 0.2, "b": 0.6, "c": 0.2}
    new_ann = {"a": 0.2, "b": 0.3, "d": 0.5}   # c dropped, d added
    old = MixtureLoader(str(tmp_path), base, anneal_weights=old_ann, anneal_from=50, **kw)
    ref = MixtureLoader(str(tmp_path), base, anneal_weights=new_ann, anneal_from=50, **kw)
    assert old.names == ["a", "b", "c"] and ref.names == ["a", "b", "c", "d"]
    for _ in range(30):   # stable phase: an extra zero-weight source changes no pick and no window
        assert torch.equal(old.next_batch()[0], ref.next_batch()[0])
    state = old.state_dict()
    assert "d" not in state["cursor"]
    new = MixtureLoader(str(tmp_path), base, anneal_weights=new_ann, anneal_from=50, **kw)
    new.load_state_dict(state)
    c_at_switch = None
    for step in range(30, 100):
        assert torch.equal(new.next_batch()[0], ref.next_batch()[0]), step
        if step == 49:
            c_at_switch = _drawn(new, "c")
    assert _drawn(new, "d") > 0 and _drawn(new, "c") == c_at_switch   # d used, c no longer drawn
    vx, _ = next(new.val_batches("d", 1))   # the new source has its own held-out tail
    assert vx.shape == (8, 16)


def test_planned_epochs_and_anneal_args(tmp_path):
    make_bins(tmp_path, {"a": [5000], "b": [5000]})
    kw = dict(seq_len=16, batch_size=4, val_tokens=1000)
    L = MixtureLoader(str(tmp_path), {"a": 0.75, "b": 0.25}, anneal_weights={"a": 0.25, "b": 0.75},
                      anneal_from=10, **kw)
    ep = L.planned_epochs(30)   # 10 steps at 3:1, then 20 steps at 1:3, 4 rows per step
    assert abs(ep["a"] * L.n_train_windows["a"] - 4 * (10 * 0.75 + 20 * 0.25)) < 1e-9
    assert abs(ep["b"] * L.n_train_windows["b"] - 4 * (10 * 0.25 + 20 * 0.75)) < 1e-9
    plain = MixtureLoader(str(tmp_path), {"a": 0.75, "b": 0.25}, **kw)
    assert abs(plain.planned_epochs(30)["a"] * plain.n_train_windows["a"] - 4 * 30 * 0.75) < 1e-9
    with pytest.raises(ValueError):
        MixtureLoader(str(tmp_path), {"a": 1.0}, anneal_weights={"b": 1.0}, **kw)


def test_mid_mix_switches_between_stable_and_anneal_and_resumes_exactly(tmp_path):
    make_bins(tmp_path, {"a": [20000], "b": [20000], "c": [20000]})
    kw = dict(seq_len=16, batch_size=8, seed=3, val_tokens=1000)
    base, mid, ann = {"a": 0.8, "b": 0.2}, {"a": 0.2, "c": 0.8}, {"b": 0.5, "c": 0.5}
    plain = MixtureLoader(str(tmp_path), base, anneal_weights=ann, anneal_from=60, **kw)
    L = MixtureLoader(str(tmp_path), base, anneal_weights=ann, anneal_from=60, mid_weights=mid, mid_from=30, **kw)
    assert L.names == plain.names == ["a", "b", "c"]
    out, drawn = [], {}
    for step in range(90):
        out.append(L.next_batch()[0])
        if step < 30:   # before the mid switch: identical to the loader without a mid phase
            assert torch.equal(out[-1], plain.next_batch()[0])
        if step in (29, 59):
            drawn[step] = {n: _drawn(L, n) for n in L.names}
    assert drawn[59]["b"] == drawn[29]["b"] and drawn[59]["c"] - drawn[29]["c"] > 150   # mid: no b, mostly c
    assert _drawn(L, "a") == drawn[59]["a"] and _drawn(L, "b") > drawn[59]["b"]           # anneal: no a
    S = MixtureLoader(str(tmp_path), base, anneal_weights=ann, anneal_from=60, mid_weights=mid, mid_from=30, **kw)
    for _ in range(45):
        S.next_batch()
    R = MixtureLoader(str(tmp_path), base, anneal_weights=ann, anneal_from=60, mid_weights=mid, mid_from=30, **kw)
    R.load_state_dict(S.state_dict())
    for i in range(45, 75):   # resumed inside the mid phase, reproduces the batches across the anneal switch
        assert torch.equal(R.next_batch()[0], out[i])


def test_switch_to_mid_mix_keeps_stable_sources_and_resumes_live_state(tmp_path):
    """The main-run switch: the live loader (stable mix + anneal-only sources) wrote a checkpoint; the new one
    has a mid mix that drops a stable source and adds new ones. The stable sources keep their positions and
    window orders, and the live state continues exactly like a loader that had the mid phase scheduled from the
    start (nothing trained so far is invalidated)."""
    make_bins(tmp_path, {s: [20000] for s in "abcde"})
    kw = dict(seq_len=16, batch_size=8, seed=3, val_tokens=1000)
    base = {"a": 0.5, "b": 0.3, "c": 0.2}
    ann = {"a": 0.3, "d": 0.7}                        # d: anneal-only, like edu_python in the live run
    mid = {"a": 0.3, "b": 0.2, "d": 0.2, "e": 0.3}    # c dropped (like code_web), d used earlier, e new
    new_ann = {"a": 0.3, "d": 0.4, "e": 0.3}

    def new_loader():
        return MixtureLoader(str(tmp_path), base, anneal_weights=new_ann, anneal_from=80, mid_weights=mid,
                             mid_from=30, **kw)

    live = MixtureLoader(str(tmp_path), base, anneal_weights=ann, anneal_from=80, **kw)
    ref = new_loader()
    assert live.names == ["a", "b", "c", "d"] and ref.names == ["a", "b", "c", "d", "e"]
    for n in live.names:
        assert np.array_equal(live._perm(n, 0), ref._perm(n, 0)), n
    for _ in range(30):
        assert torch.equal(live.next_batch()[0], ref.next_batch()[0])
    new = new_loader()
    new.load_state_dict(live.state_dict())
    c_at_switch = _drawn(new, "c")
    for step in range(30, 100):
        assert torch.equal(new.next_batch()[0], ref.next_batch()[0]), step
    assert _drawn(new, "c") == c_at_switch and _drawn(new, "e") > 0
    vx, _ = next(new.val_batches("e", 1))
    assert vx.shape == (8, 16)


def test_planned_epochs_with_mid_mix_and_args(tmp_path):
    make_bins(tmp_path, {"a": [5000], "b": [5000]})
    kw = dict(seq_len=16, batch_size=4, val_tokens=1000)
    L = MixtureLoader(str(tmp_path), {"a": 0.75, "b": 0.25}, mid_weights={"a": 0.5, "b": 0.5}, mid_from=10,
                      anneal_weights={"a": 0.25, "b": 0.75}, anneal_from=20, **kw)
    ep = L.planned_epochs(30)   # 10 steps at 3:1, 10 at 1:1, 10 at 1:3, 4 rows per step
    assert abs(ep["a"] * L.n_train_windows["a"] - 4 * (10 * 0.75 + 10 * 0.5 + 10 * 0.25)) < 1e-9
    ep = L.planned_epochs(15)   # ends inside the mid phase
    assert abs(ep["b"] * L.n_train_windows["b"] - 4 * (10 * 0.25 + 5 * 0.5)) < 1e-9
    M = MixtureLoader(str(tmp_path), {"a": 0.75, "b": 0.25}, mid_weights={"a": 0.5, "b": 0.5}, mid_from=10, **kw)
    assert abs(M.planned_epochs(30)["a"] * M.n_train_windows["a"] - 4 * (10 * 0.75 + 20 * 0.5)) < 1e-9
    assert [s for s, _ in L.phases()] == [0, 10, 20] and [s for s, _ in M.phases()] == [0, 10]
    with pytest.raises(ValueError):
        MixtureLoader(str(tmp_path), {"a": 1.0}, mid_weights={"b": 1.0}, **kw)
    with pytest.raises(ValueError):   # the mid mix must start before the anneal
        MixtureLoader(str(tmp_path), {"a": 1.0}, mid_weights={"b": 1.0}, mid_from=30, anneal_weights={"a": 1.0},
                      anneal_from=20, **kw)


def test_main_run_stage2_mixes():
    """MIXES of the main run's stage 2: the live loader's sources keep their positions, the new sources exist
    in data_prep/sources.py, and the mixes have the planned composition (English / math / code)."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    import train as T
    from sources import BY_NAME

    def names(base, mid, ann):
        out = [n for n, w in base.items() if w > 0]
        for later in (mid, ann):
            out += [n for n, w in (later or {}).items() if w > 0 and n not in out]
        return out

    M = T.MIXES
    live = names(M["reasoning"], None, M["reasoning_anneal_code"])
    new = names(M["reasoning"], M["reasoning_s2"], M["reasoning_anneal_code_s2"])
    assert new[:len(live)] == live and len(new) == len(live) + 4
    kind = {"en": "en", "math": "math", "code": "code"}
    for mix, want in (("reasoning_s2", (0.43, 0.28, 0.29)), ("reasoning_anneal_code_s2", (0.42, 0.28, 0.30))):
        w = M[mix]
        assert abs(sum(w.values()) - 1) < 1e-9, mix
        share = {c: sum(v for n, v in w.items() if kind[BY_NAME[n].kind] == c) for c in ("en", "math", "code")}
        assert all(abs(share[c] - x) < 1e-9 for c, x in zip(("en", "math", "code"), want)), (mix, share)
        assert w.get("code_web", 0) == 0
