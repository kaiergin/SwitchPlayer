"""Test H, stage C, the guessers (CURRICULUM "Next experiment (pre-registered)
-- Test H", METHOD: GUESSERS). Pins, on the CPU: every decision's 17 stacks as
frame indices against stage A's own stack rebuild, rows past a life's end
repeating its last stack (LIFE ENDS), and row k equal to decision t+k's row 0
inside a life; gather()'s layout for S, F_k and SIGMA; the network's pairs (S
reads (obs_t, obs_t)), SIGMA's uniform start and its cut renormalisation; a toy
task only the future can solve (F_1 learns it, S cannot); resume continuing a
stamped model; best_within / best_state over every new best; the common
validation set's FIT-overlap exclusion; and the plan's rules on a stand-in
trainer: the grid and its edge rate, the cap from the chosen rates' best
steps, AT THE CAP's one extension and flag, S's sanity rerun, TRAINING FAILED's
rerun at a third of the rate and twice the cap, VOID, and a killed model's
directory cleared and retrained."""
import json
import math
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROBES = ROOT / "benchmarks" / "probes"
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(PROBES))
import probe_move_guesser as mg  # noqa: E402
import train_move_guesser as tg  # noqa: E402


def _game(kinds, ends):
    n = len(kinds)
    frames = np.stack([np.full((3, 84, 84), i % 256, np.uint8) for i in range(n)])
    dec = np.array([-1] * n, np.int32)
    d = 0
    for i, k in enumerate(kinds):
        if k == mg.KIND_MOVE:
            dec[i] = d
            d += 1
    return {"kind": np.array(kinds, np.int8), "end": np.array(ends, np.int8),
            "move": np.array([i % 9 for i in range(n)], np.int8),
            "decision": dec, "frames": frames}


def test_windows_match_the_stack_rebuild_and_life_ends():
    # life 0: start, 3 opening, 25 moves (life lost at the last), life 1: start, 30 moves
    kinds = [0] + [1] * 3 + [2] * 25 + [0] + [2] * 30
    ends = [0] * len(kinds)
    ends[28] = 1
    ends[-1] = 1
    arr = _game(kinds, ends)
    d, ev, mv, win = tg.decision_windows(arr, frame_offset=0)
    rebuilt = mg.stacks_from_frames(arr)
    assert len(d) == 55
    for n in range(len(d)):
        st = np.concatenate([arr["frames"][j] for j in win[n, 0]])
        assert np.array_equal(st, rebuilt[int(d[n])])
    by_dec = {int(x): n for n, x in enumerate(d)}
    # inside a life, row k is decision t+k's row 0
    for n in range(len(d)):
        t = int(d[n])
        i = int(ev[n])
        for k in range(1, tg.K_MAX + 1):
            j = i + k
            if j < len(kinds) and kinds[j] == mg.KIND_MOVE and all(
                    ends[x] == 0 for x in range(i, j)):
                assert list(win[n, k]) == list(win[by_dec[t + k], 0])
    # life 0's last decision (event 28) lost the life at its own step: every
    # later row is the stack after that step, repeated
    n = by_dec[24]
    after = list(win[n, 1])
    assert after[-1] == 28 and all(list(win[n, k]) == after for k in range(1, 17))
    # a decision 5 steps before the loss: rows 1..5 move on, then repeat
    n = by_dec[19]
    assert [int(win[n, k][-1]) for k in range(1, 8)] == [23, 24, 25, 26, 27, 28, 28]
    assert all(list(win[n, k]) == list(win[n, 6]) for k in range(6, 17))


def test_gather_layout():
    frames = torch.arange(40, dtype=torch.uint8).view(40, 1, 1, 1).expand(40, 3, 84, 84).contiguous()
    win = torch.arange(2 * 17 * 4).view(2, 17, 4) % 40
    for arm, offs in tg.ARMS.items():
        x = tg.gather(frames, win, offs)
        assert x.shape == (2, 1 + len(offs), 12, 84, 84)
        cols = [0] + list(offs)
        for b in range(2):
            for r, c in enumerate(cols):
                want = [int(v) for v in win[b, c]]
                got = [int(x[b, r, 3 * f, 0, 0]) for f in range(4)]
                assert got == want


def test_network_pairs_and_cut():
    torch.manual_seed(0)
    obs = torch.randint(0, 255, (3, 17, 12, 84, 84), dtype=torch.uint8)
    s = tg.Guesser(tg.ARMS["S"])
    x = obs[:, [0, 0]]
    lg = s(x)
    z = s.enc(torch.cat([obs[:, 0], obs[:, 0]], 1).float() / 255)
    assert torch.allclose(lg, s.head(z), atol=1e-5)
    g = tg.Guesser(tg.ARMS["SIGMA"])
    assert torch.allclose(torch.softmax(g.w, 0), torch.full((16,), 1 / 16))
    xs = obs                                      # rows 1..16 are offsets 1..16
    one = g(xs, keep={1})
    z1 = g.enc(torch.cat([obs[:, 0], obs[:, 1]], 1).float() / 255)
    assert torch.allclose(one, g.head(z1), atol=1e-4)
    with pytest.raises(ValueError):
        g(obs[:, :5])


# --------------------------------------------------------------------------
# training on the CPU, a toy task
# --------------------------------------------------------------------------
def toy_corpus(n=600, seed=0):
    """Moves random; the frame after each step shows the move as a bright
    block. S (no future) cannot beat chance; F_1 can."""
    rng = np.random.default_rng(seed)
    moves = rng.integers(0, 9, n)
    frames = np.zeros((2 * n, 3, 84, 84), np.uint8)
    win = np.zeros((n, 17, 4), np.int64)
    for i in range(n):
        frames[2 * i] = rng.integers(0, 40, (3, 84, 84))
        frames[2 * i + 1] = frames[2 * i]
        r, c = divmod(int(moves[i]), 3)
        frames[2 * i + 1, :, 20 + 15 * r:30 + 15 * r, 20 + 15 * c:30 + 15 * c] = 255
        win[i, 0] = 2 * i
        win[i, 1:] = 2 * i + 1
    return SimpleNamespace(frames=frames, window=win, move=moves.astype(np.int64))


def test_toy_task_f1_learns_s_cannot_and_resume(tmp_path):
    torch.set_num_threads(2)
    c = toy_corpus()
    fit, val = np.arange(500), np.arange(500, 600)
    t = tg.Trainer(c, fit, val, "cpu", batch=32)
    old = tg.EVAL_EVERY
    tg.EVAL_EVERY = 50
    try:
        t.train("F1", 0, 1e-3, 150, tmp_path / "f1")
        _, nll_f, _ = tg.best_within(tmp_path / "f1", 150)
        t.train("S", 0, 1e-3, 150, tmp_path / "s")
        _, nll_s, _ = tg.best_within(tmp_path / "s", 150)
        assert nll_f < 0.5 * math.log(9) < nll_s
        # resume: a stamped model continues, history and bests extend
        stamp = json.loads((tmp_path / "f1" / "stamp.json").read_text())
        t.train("F1", 0, 1e-3, 250, tmp_path / "f1", resume=stamp)
        hist = json.loads((tmp_path / "f1" / "history.json").read_text())
        assert [h["step"] for h in hist] == [0, 50, 100, 150, 200, 250]
        state, step = tg.best_state(tmp_path / "f1", 250)
        assert (tmp_path / "f1" / "bests" / f"step{step:06d}.pt").exists()
        # the best within any cap is on disk
        for cap in (0, 50, 100, 150, 200, 250):
            tg.best_state(tmp_path / "f1", cap)
    finally:
        tg.EVAL_EVERY = old


def test_common_validation_excludes_fit_overlap():
    frames = np.stack([np.full((3, 84, 84), i, np.uint8) for i in range(10)])
    win = np.zeros((4, 17, 4), np.int64)
    win[0], win[1] = 0, 1                      # FIT
    win[2], win[3] = 2, 1                      # VAL: 3 shares FIT's stack
    win[3, 0] = 3
    c = SimpleNamespace(frames=frames, window=win)
    c.stack_hashes = tg.Corpus.stack_hashes.__get__(c)
    keep = tg.common_validation(c, np.array([0, 1]), np.array([2, 3]))
    assert list(keep) == [2]


# --------------------------------------------------------------------------
# the plan's rules on a stand-in trainer
# --------------------------------------------------------------------------
class FakeTrainer:
    """Writes a synthetic history: nll(step) from curves[(arm, lr, tag?)],
    evaluations every EVAL_EVERY, each new best saved, stamps as the real one."""

    def __init__(self, curves, accs=None, majority_acc=0.3):
        self.curves, self.accs = curves, accs or {}
        self.batch, self.majority, self.majority_acc = 256, 0, majority_acc
        self.calls = []

    def train(self, arm, seed, lr, steps, mdir, resume=None, save_all=False):
        self.calls.append((arm, seed, lr, steps, mdir.name, resume is not None))
        (mdir / "bests").mkdir(parents=True, exist_ok=True)
        if save_all:
            (mdir / "evals").mkdir(exist_ok=True)
        start = resume["steps"] if resume else 0
        hist = json.loads((mdir / "history.json").read_text()) if resume else []
        best = min((h["nll"] for h in hist), default=math.inf)
        f = self.curves.get((arm, seed, lr)) or self.curves[(arm, lr)]
        acc = self.accs.get((arm, seed, lr), 0.5)
        s = start if resume else 0
        while s <= steps:
            if not hist or s > hist[-1]["step"]:
                nll = f(s)
                hist.append({"step": s, "nll": nll, "acc": acc})
                if nll < best:
                    best = nll
                    torch.save({}, mdir / "bests" / f"step{s:06d}.pt")
                if save_all:
                    torch.save({}, mdir / "evals" / f"step{s:06d}.pt")
            s += tg.EVAL_EVERY
        torch.save({"step": steps}, mdir / "last.pt")
        (mdir / "history.json").write_text(json.dumps(hist))
        (mdir / "stamp.json").write_text(json.dumps(
            {"arm": arm, "seed": seed, "lr": lr, "steps": steps, "batch": 256,
             "source": tg.SOURCE, "data_source": mg.SOURCE}))


def vee(best_step, depth=1.0):
    return lambda s: 2.0 - depth + abs(s - best_step) / 1e5


def test_plan_rules(tmp_path):
    P = 4000                                   # provisional cap for the test
    curves = {}
    for arm in tg.ARMS:
        for lr in tg.RATES + tuple(tg.EDGE.values()):
            # S's other rates are worse than its 1e-3; every future arm's
            # default (min 0.9) beats S's best (1.0)
            curves[(arm, lr)] = vee(1000, depth=0.5 if arm == "S" else 1.1)
    # S: 1e-3 best, peaks at 1,500; SIGMA: 3e-3 best (an edge), 1e-2 better still,
    # peaking at 3,800 (the last 10% of 4,000: extended to 8,000, peak holds)
    curves[("S", 1e-3)] = vee(1500, depth=1.0)
    curves[("S", 1e-3 / 3)] = vee(1500, depth=1.0)
    curves[("SIGMA", 3e-3)] = vee(3800, depth=1.2)
    curves[("SIGMA", 1e-2)] = vee(3800, depth=1.3)
    # F1 seed 2 fails to train at its rate; its rerun succeeds
    curves[("F1", 1e-3)] = vee(1000, depth=1.15)
    curves[("F1", 2, 1e-3)] = lambda s: 3.0
    curves[("F1", 1e-3 / 3)] = vee(1000, depth=1.5)
    # F16 fails at every rate and in its rerun: VOID
    for lr in tg.RATES + tuple(tg.EDGE.values()):
        curves[("F16", lr)] = lambda s: 3.0
    curves[("F16", 3e-4 / 3)] = lambda s: 3.0
    accs = {("S", 3, 1e-3): 0.1}               # S seed 3 fails its sanity check once
    t = FakeTrainer(curves, accs)
    old = tg.EVAL_EVERY
    tg.EVAL_EVERY = 100
    try:
        plan = tg.Plan(t, tmp_path, provisional=P, log=lambda s: None)
        rec = plan.run()
    finally:
        tg.EVAL_EVERY = old
    assert rec["rates"]["S"]["chosen"] == 1e-3
    assert rec["rates"]["SIGMA"]["chosen"] == 1e-2           # the edge rate won
    assert set(rec["rates"]["SIGMA"]["grid"]) == {"0.0003", "0.001", "0.003", "0.01"}
    # SIGMA's chosen model peaked at 3,800 of 4,000: extended to 8,000, best 3,800
    assert rec["rates"]["SIGMA"]["grid"]["0.01"]["cap"] == 2 * P
    assert rec["cap_from"] == {"S": 1500, "SIGMA": 3800}
    assert rec["cap"] == 3 * 3800
    cap = rec["cap"]
    # every arm's seed 0 and the five seeds of S, F1, SIGMA are in the read
    keys = set(rec["models"])
    assert {f"{a}-s0" for a in tg.ARMS} <= keys
    assert {f"{a}-s{s}" for a in tg.SEEDED for s in tg.SEEDS} <= keys
    # S seed 3: sanity rerun at a third of the rate and twice the cap
    m = rec["models"]["S-s3"]
    assert m["rerun"] and m["lr"] == pytest.approx(1e-3 / 3) and m["cap"] in (2 * cap, 4 * cap)
    assert "S-s3: sanity" not in rec["void"]                 # the rerun passed it
    # F1 seed 2: TRAINING FAILED, the rerun passes
    m = rec["models"]["F1-s2"]
    assert m["rerun"] and m["lr"] == pytest.approx(1e-3 / 3)
    assert not any(v.startswith("F1-s2") for v in rec["void"])
    # F16 fails twice: VOID
    assert any(v.startswith("F16-s0") for v in rec["void"])
    assert json.loads((tmp_path / "plan.json").read_text())["cap"] == cap
    # seed 0 of S and SIGMA are the grid's models, continued in their own
    # directories to the final cap, with the final step's state on disk
    for arm, lr in (("S", 1e-3), ("SIGMA", 1e-2)):
        m = rec["models"][f"{arm}-s0"]
        assert m["dir"] == f"{arm}-s0-lr{lr:g}"
        assert m["final_state"] == f"evals/step{m['cap']:06d}.pt"
        assert (tmp_path / "models" / m["dir"] / m["final_state"]).exists()
        stamp = json.loads((tmp_path / "models" / m["dir"] / "stamp.json").read_text())
        assert stamp["steps"] >= cap
    assert "final_state" not in rec["models"]["SIGMA-s1"]
    # flags are those of the read's models as they stand
    assert rec["flags"] == sorted(k for k, m in rec["models"].items() if m["flagged"])


def test_plan_flags_a_model_still_late_after_its_extension(tmp_path):
    curves = {(arm, lr): vee(1000, depth=0.5 if arm == "S" else 1.1)
              for arm in tg.ARMS for lr in tg.RATES + tuple(tg.EDGE.values())}
    curves[("S", 1e-3)] = vee(1500, depth=1.0)
    curves[("SIGMA", 1e-3)] = vee(1500, depth=1.2)
    curves[("F1", 1e-3)] = lambda s: 2.0 - 1.15 - s / 1e7      # still improving
    t = FakeTrainer(curves)
    old = tg.EVAL_EVERY
    tg.EVAL_EVERY = 100
    try:
        rec = tg.Plan(t, tmp_path, provisional=4000, log=lambda s: None).run()
    finally:
        tg.EVAL_EVERY = old
    cap = rec["cap"]
    for s in tg.SEEDS:
        m = rec["models"][f"F1-s{s}"]
        assert m["cap"] == 2 * cap and m["flagged"]
        assert f"F1-s{s}" in rec["flags"]


def test_killed_model_is_cleared_and_retrained(tmp_path):
    t = FakeTrainer({("S", 1e-3): vee(200)})
    old = tg.EVAL_EVERY
    tg.EVAL_EVERY = 100
    try:
        plan = tg.Plan(t, tmp_path, provisional=1000, log=lambda s: None)
        d = plan.mdir("S", 0, 1e-3)
        (d / "bests").mkdir(parents=True)
        (d / "bests" / "step000700.pt").write_bytes(b"partial")
        plan.ensure("S", 0, 1e-3, 1000)
        assert t.calls[-1][-1] is False                     # from scratch
        assert not (d / "bests" / "step000700.pt").exists()
        # a stamped model short of the target continues, stale bests cleared
        (d / "bests" / "step001500.pt").write_bytes(b"from a killed extension")
        plan.ensure("S", 0, 1e-3, 2000)
        assert t.calls[-1][-1] is True and t.calls[-1][3] == 2000
        assert not (d / "bests" / "step001500.pt").exists()
        # a stamped model at the target is not retrained
        n = len(t.calls)
        plan.ensure("S", 0, 1e-3, 1500)
        assert len(t.calls) == n
        # stamp, history and last.pt disagreeing (killed between writes): cleared
        torch.save({"step": 999}, d / "last.pt")
        plan.ensure("S", 0, 1e-3, 1500)
        assert t.calls[-1][-1] is False and len(t.calls) == n + 1
        # other stage-A data: cleared and retrained
        st = json.loads((d / "stamp.json").read_text())
        st["data_source"] = "older"
        (d / "stamp.json").write_text(json.dumps(st))
        plan.ensure("S", 0, 1e-3, 1500)
        assert t.calls[-1][-1] is False and len(t.calls) == n + 2
        # another batch: refused, never deleted
        t.batch = 128
        with pytest.raises(SystemExit, match="batch"):
            plan.ensure("S", 0, 1e-3, 1500)
        assert (d / "stamp.json").exists()
    finally:
        tg.EVAL_EVERY = old


def test_resume_equals_an_uninterrupted_run(tmp_path):
    torch.set_num_threads(2)
    c = toy_corpus(n=200)
    fit, val = np.arange(160), np.arange(160, 200)
    t = tg.Trainer(c, fit, val, "cpu", batch=16)
    old = tg.EVAL_EVERY
    tg.EVAL_EVERY = 10
    try:
        t.train("F1", 0, 1e-3, 40, tmp_path / "a")
        t.train("F1", 0, 1e-3, 20, tmp_path / "b")
        stamp = json.loads((tmp_path / "b" / "stamp.json").read_text())
        t.train("F1", 0, 1e-3, 40, tmp_path / "b", resume=stamp)
    finally:
        tg.EVAL_EVERY = old
    a = torch.load(tmp_path / "a" / "last.pt", weights_only=False)
    b = torch.load(tmp_path / "b" / "last.pt", weights_only=False)
    for k in a["model"]:
        assert torch.allclose(a["model"][k], b["model"][k], atol=1e-6), k
    ha = json.loads((tmp_path / "a" / "history.json").read_text())
    hb = json.loads((tmp_path / "b" / "history.json").read_text())
    assert [h["step"] for h in ha] == [h["step"] for h in hb]
    assert all(abs(x["nll"] - y["nll"]) < 1e-5 for x, y in zip(ha, hb))


def test_windows_refuse_a_malformed_record():
    arr = _game([0, 2, 2, 2], [0, 0, 0, 0])       # a life with no end flag
    with pytest.raises(ValueError, match="without an end flag"):
        tg.decision_windows(arr)
