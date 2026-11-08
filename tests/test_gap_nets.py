"""Test G, the value nets (CURRICULUM "Next experiment (pre-registered) --
Test G", THE VALUE NETS and G2's first clause). Pins, on logs written by
ablation.OwnPlayLog itself: the rebuild (frames, base steps by the reward
tuples, a cut-short last token, every logged stack and its hash), its
refusals, the iteration cut and the log totals, the life index, the late
third, the per-frame discounting on both scales against hand-computed
returns and the pilot's head_returns, the clip against muzero.grown_reward,
the end of a life, the per-head standardisation round trip, the stratified
split and the hash exclusion, seed averaging through load / predict, R2 and
G2's verdict, the stamps and whole-or-nothing writes, the output refusals,
and a whole small run in-process that never opens experiment state nor
writes outside its output. In a subprocess: importing the probe changes
nothing process-wide. Slow: the real T2 and T0 logs against the
registration's counts."""
import json
import math
import pathlib
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROBES = ROOT / "benchmarks" / "probes"
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(PROBES))
import muzero  # noqa: E402
import ablation  # noqa: E402
import probe_action_census as pc  # noqa: E402
import probe_e3a_sequences as e3  # noqa: E402
import probe_gap_check as gc  # noqa: E402
import probe_gap_nets as gn  # noqa: E402

# every file opened while RECORDING is on: (path, mode)
OPENED = []
RECORDING = [False]


def _audit(event, args):
    if RECORDING[0] and event == "open":
        OPENED.append((str(args[0]), str(args[1])))


sys.addaudithook(_audit)


# --------------------------------------------------------------------------
# fake logs, in OwnPlayLog's own form
# --------------------------------------------------------------------------
def fake_life(frames, tokens, rewards):
    """(traj, inner) of a life from its frames (n + 4, 3, 84, 84), the base
    steps each decision played and the raw rewards per base step, as
    play_episode and play_token log them."""
    traj, inner, b, prev = [], [], 0, ()
    for i, k in enumerate(tokens):
        traj.append((frames[b:b + 4].reshape(12, 84, 84).copy(), i % 7, prev))
        inner.append([frames[4 + b + j].copy() for j in range(k - 1)])
        prev = tuple(float(r) for r in rewards[b:b + k])
        b += k
    traj.append((frames[b:b + 4].reshape(12, 84, 84).copy(), 0, prev))
    return traj, inner


def write_log(results, player, lives, seed=0, start=None):
    """`lives`: [(iteration, real_done, terminated, tokens)]. Lives whose
    real_done chain puts them first in a game share `start`'s frames when
    given. Returns [(frames, rewards)] as written."""
    rng = np.random.default_rng(seed)
    folder = results / gc.PLAYERS[player]["run"] / "own-play-fake"
    log = ablation.OwnPlayLog(folder)
    written, first = [], True
    for it, real_done, terminated, tokens in lives:
        n = sum(tokens)
        frames = rng.integers(0, 256, (n + 4, 3, 84, 84), dtype=np.uint8)
        if start is not None and first:
            frames[:len(start)] = start
        rewards = rng.choice([0.0, 0.0, 0.0, 10.0, 50.0, 200.0, 5.0], size=n)
        traj, inner = fake_life(frames, tokens, rewards)
        log.write(it, traj, {"inner": inner, "real_done": real_done,
                             "temperature": 0.5}, terminated, n, 0)
        written.append((frames, rewards))
        first = real_done
    return folder, written


TOKENS = (4, 4, 2, 1, 3)                       # the last one cut short


def games(iterations, per_iteration, lives_per_game=3):
    """Lives over iterations 1..iterations, `per_iteration` each, in games of
    `lives_per_game` (real_done on a game's last)."""
    out, k = [], 0
    for it in range(1, iterations + 1):
        for _ in range(per_iteration):
            k += 1
            out.append((it, k % lives_per_game == 0, True, TOKENS))
    return out


def prep_json(results, unit=10.0, ceiling=1.0):
    p = results / gn.T0_PREP / "vocab.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "master_snapshot": f"X:\\r\\{gc.PLAYERS['T0']['run']}\\w.pth",
        "reward_unit": unit, "master_ceiling": ceiling}))


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A fake results tree: T2 9 lives over iterations 1-3 (all used); T0 30
    lives over 1-5, cut at 4 (24 used, 6 later). Games of 3 lives. First
    lives share their first 6 frames (3 identical stacks), as the real first
    lives repeat."""
    results = tmp_path / "tree" / "benchmarks" / "results"
    start = np.random.default_rng(99).integers(0, 256, (6, 3, 84, 84),
                                               dtype=np.uint8)
    logs = {}
    for player, (iters, per, cut, seed) in {"T2": (3, 3, 3, 1),
                                            "T0": (5, 6, 4, 2)}.items():
        lives = games(iters, per)
        _, written = write_log(results, player, lives, seed, start)
        raw = float(sum(r.sum() for _, r in written))
        later = sum(1 for life in lives if life[0] > cut)
        monkeypatch.setitem(gn.LOGS, player, {
            "last": iters, "cut": cut, "lives": len(lives),
            "base_steps": sum(TOKENS) * len(lives), "raw": raw,
            "later_lives": later})
        logs[player] = {"lives": lives, "written": written}
    prep_json(results)
    monkeypatch.setattr(gc, "RESULTS", results)
    return {"results": results, "out": tmp_path / "out", **logs}


# --------------------------------------------------------------------------
# the rebuild
# --------------------------------------------------------------------------
def test_rebuild_matches_the_logged_life(tree):
    lives, totals = gn.read_log("T2")
    assert len(lives) == 9 and totals["lives"] == 9
    assert totals["base_steps"] == 9 * sum(TOKENS)
    folder = gn.log_folder("T2")
    logged = [life for _, life in gn.e1.iter_own_play(folder, 3)]
    for life, raw, (frames, rewards) in zip(lives, logged,
                                            tree["T2"]["written"]):
        assert np.array_equal(life["frames"], frames)
        assert np.array_equal(life["rewards"], rewards)
        assert len(life["rewards"]) == raw["steps"] == sum(TOKENS)
    data = gn.Data(lives)
    assert len(data) == totals["base_steps"]          # one stack a base step
    assert data.y.shape == (len(data), 8)
    # every logged decision stack is the rebuilt stack at its base step, and
    # its hash the pilot's digest of it
    k, checked = 0, 0
    for li, raw in enumerate(logged):
        b, traj = 0, raw["traj"]
        for j in range(len(traj) - 1):
            i = k + b
            assert data.life[i] == li and data.t[i] == b
            assert np.array_equal(data.stack(i), traj[j][0])
            assert data.digests[i] == pc._digest(traj[j][0].tobytes())
            b += len(traj[j + 1][2])
            checked += 1
        assert b == len(lives[li]["rewards"])
        k += b
    assert checked == 9 * len(TOKENS)
    # the stack before base step t is frames t..t+3, for every t
    frames = tree["T2"]["written"][0][0]
    for t in range(sum(TOKENS)):
        assert np.array_equal(data.stack(t), frames[t:t + 4].reshape(12, 84,
                                                                     84))
    assert data.stacks(np.arange(3)).dtype == np.uint8


def test_base_steps_follow_the_reward_tuples_not_the_tokens(tree):
    """A cut-short last token: 3 base steps logged, whatever the token's
    length; decision stacks sit at the reward tuples' cumulative counts."""
    folder = gn.log_folder("T2")
    life = next(l for _, l in gn.e1.iter_own_play(folder, 1))
    steps = [len(r[2]) for r in life["traj"][1:]]
    assert steps == list(TOKENS)
    frames, rewards = gn.life_frames(life)
    assert len(frames) == sum(TOKENS) + 4 and len(rewards) == sum(TOKENS)
    starts = np.cumsum([0] + steps[:-1])
    for rec, b in zip(life["traj"][:-1], starts):
        assert np.array_equal(frames[b:b + 4].reshape(12, 84, 84), rec[0])


def _a_life(tree):
    folder = gn.log_folder("T2")
    return next(l for _, l in gn.e1.iter_own_play(folder, 1))


@pytest.mark.parametrize("break_it", [
    "inner short", "next stack", "steps", "no reward tuple", "first rewards",
    "frame dtype"])
def test_rebuild_refuses_a_broken_life(tree, break_it):
    life = _a_life(tree)
    traj, inner = [list(r) for r in life["traj"]], life["inner"]
    if break_it == "inner short":
        inner[0] = inner[0][:-1]
    elif break_it == "next stack":
        traj[2][0] = traj[2][0].copy()
        traj[2][0][0, 0, 0] ^= 1
    elif break_it == "steps":
        life["steps"] += 1
    elif break_it == "no reward tuple":
        traj[1][2] = float(sum(traj[1][2]))
    elif break_it == "first rewards":
        traj[0][2] = (0.0,)
    else:
        inner[0][0] = inner[0][0].astype(np.int16)
    life["traj"] = [tuple(r) for r in traj]
    with pytest.raises(ValueError):
        gn.life_frames(life)


def test_read_log_refuses_a_broken_log_and_missing_files(tree):
    folder = gn.log_folder("T2")
    (folder / "iter-004.pkl").write_bytes(b"")
    with pytest.raises(SystemExit, match="not iterations 1-3"):
        gn.read_log("T2")
    (folder / "iter-004.pkl").unlink()
    (folder / "iter-002.pkl").rename(folder / "x.pkl")
    with pytest.raises(SystemExit, match="not iterations 1-3"):
        gn.read_log("T2")


def test_read_log_refuses_a_life_that_does_not_rebuild(tmp_path,
                                                       monkeypatch):
    results = tmp_path / "results"
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, (sum(TOKENS) + 4, 3, 84, 84),
                          dtype=np.uint8)
    traj, inner = fake_life(frames, TOKENS, np.zeros(sum(TOKENS)))
    inner[1] = inner[1][1:]
    log = ablation.OwnPlayLog(results / gc.PLAYERS["T2"]["run"] /
                              "own-play-x")
    log.write(1, traj, {"inner": inner, "real_done": True,
                        "temperature": 0.5}, True, sum(TOKENS), 0)
    monkeypatch.setattr(gc, "RESULTS", results)
    monkeypatch.setitem(gn.LOGS, "T2", {**gn.LOGS["T2"], "last": 1,
                                        "cut": 1})
    with pytest.raises(SystemExit, match="iteration 1, life 1"):
        gn.read_log("T2")


# --------------------------------------------------------------------------
# the cut, the totals, the life index, the late third
# --------------------------------------------------------------------------
def test_registered_logs_and_cut():
    assert gn.LOGS["T0"]["cut"] == 45 == gc.PLAYERS["T0"]["iteration"]
    assert gn.LOGS["T0"]["last"] == 47
    assert (gn.LOGS["T0"]["lives"], gn.LOGS["T0"]["base_steps"],
            gn.LOGS["T0"]["raw"], gn.LOGS["T0"]["later_lives"]) == \
        (141, 33616, 34430.0, 6)
    assert gn.LOGS["T2"]["cut"] == gn.LOGS["T2"]["last"] == 42 == \
        gc.PLAYERS["T2"]["iteration"]
    assert (gn.LOGS["T2"]["lives"], gn.LOGS["T2"]["base_steps"],
            gn.LOGS["T2"]["raw"], gn.LOGS["T2"]["later_lives"]) == \
        (126, 33021, 66850.0, 0)
    assert {n: s["player"] for n, s in gn.NETS.items()} == \
        {"master": "T2", "twin": "T0", "late": "T0"}
    assert gn.GATED == ("master", "twin") and gn.SEEDS == (0, 1, 2)
    assert gn.HEADS == tuple((s, g) for s in ("clip", "raw")
                             for g in (0.997, 0.99, 0.98, 0.95))
    assert gn.HEADS[gn.PRIMARY] == ("clip", 0.997)
    assert (gc.CLIP_UNIT, gc.CLIP_CEILING, gc.BASE) == (10.0, 1.0, 4)


def test_the_iteration_cut(tree):
    lives, totals = gn.read_log("T0")
    assert {life["iteration"] for life in lives} == {1, 2, 3, 4}
    assert len(lives) == 24 and totals["lives"] == 30
    assert totals["later_lives"] == 6
    written = tree["T0"]["written"]
    assert totals["raw"] == pytest.approx(sum(r.sum() for _, r in written))
    for life, (frames, rewards) in zip(lives, written):
        assert np.array_equal(life["frames"], frames)
    gn.check_totals("T0", lives, totals)            # the fixture's numbers


@pytest.mark.parametrize("field,delta", [("lives", 1), ("base_steps", -1),
                                         ("raw", 10.0), ("later_lives", 1)])
def test_check_totals_refuses_other_counts(tree, monkeypatch, field, delta):
    lives, totals = gn.read_log("T0")
    monkeypatch.setitem(gn.LOGS["T0"], field, gn.LOGS["T0"][field] + delta)
    with pytest.raises(SystemExit, match="registration"):
        gn.check_totals("T0", lives, totals)


def test_check_totals_refuses_another_clip(tree):
    lives, totals = gn.read_log("T0")
    prep_json(tree["results"], ceiling=7.26)
    with pytest.raises(SystemExit, match="clip"):
        gn.check_totals("T0", lives, totals)
    prep_json(tree["results"], unit=1.0)
    with pytest.raises(SystemExit, match="clip"):
        gn.check_totals("T0", lives, totals)


def test_life_index_follows_the_games(tree):
    lives, _ = gn.read_log("T0")
    assert [life["index"] for life in lives] == [0, 1, 2] * 8
    lives, _ = gn.read_log("T2")
    assert [life["index"] for life in lives] == [0, 1, 2] * 3


def test_the_late_third():
    assert gn.select("late", list(range(135))) == list(range(90, 135))
    assert gn.select("twin", list(range(135))) == list(range(135))
    assert gn.select("master", list(range(126))) == list(range(126))
    assert gn.select("late", list(range(10))) == [7, 8, 9]


# --------------------------------------------------------------------------
# the targets
# --------------------------------------------------------------------------
def test_returns_by_hand():
    raw = [0.0, 10.0, 0.0, 50.0, 200.0, 5.0]
    clip = [0.0, 1.0, 0.0, 1.0, 1.0, 0.5]
    y = gn.life_returns(raw)
    assert y.shape == (6, 8)
    for h, (scale, gamma) in enumerate(gn.HEADS):
        g = gamma ** 4                            # per frame, 4 a base step
        x = clip if scale == "clip" else raw
        for t in range(6):
            want = sum(x[j] * g ** (j - t) for j in range(t, 6))
            assert y[t, h] == pytest.approx(want, rel=1e-12)
    g = 0.997 ** 4
    assert y[0, 0] == pytest.approx(g + g ** 3 + g ** 4 + 0.5 * g ** 5)
    g = 0.95 ** 4
    assert y[1, 7] == pytest.approx(10 + 50 * g ** 2 + 200 * g ** 3
                                    + 5 * g ** 4)
    assert y[5, 0] == 0.5 and y[5, 4] == 5.0      # the last base step


def test_returns_are_the_pilots_branch_returns():
    """Each base step's return is probe_gap_check.head_returns on the
    rewards from it: the real gap's definition."""
    rng = np.random.default_rng(3)
    raw = rng.choice([0.0, 10.0, 50.0, 200.0, 1600.0], size=40)
    y = gn.life_returns(raw)
    for t in range(40):
        assert np.allclose(y[t], gc.head_returns(list(raw[t:])).ravel(),
                           rtol=1e-12)


def test_the_clip_is_the_repos(monkeypatch):
    monkeypatch.setattr(muzero, "REWARD_UNIT", 10.0)
    for r in (0.0, 5.0, 10.0, 50.0, 200.0, 1600.0, -10.0, -3.0):
        assert e3.clipped(r, gc.CLIP_UNIT, gc.CLIP_CEILING) == \
            muzero.grown_reward([r], 1.0)
        assert gn.life_returns([r])[0, 0] == muzero.grown_reward([r], 1.0)


def test_returns_end_with_the_life(tree):
    """A life's returns never reach into the next life's rewards."""
    lives, _ = gn.read_log("T2")
    data = gn.Data(lives)
    k = 0
    for life in lives:
        n = len(life["rewards"])
        assert np.array_equal(data.y[k:k + n],
                              gn.life_returns(life["rewards"]))
        last = life["rewards"][-1]
        assert data.y[k + n - 1, 4] == last
        assert data.y[k + n - 1, 0] == e3.clipped(last, 10.0, 1.0)
        k += n


# --------------------------------------------------------------------------
# standardisation, the split, the hash exclusion
# --------------------------------------------------------------------------
def test_standardisation_round_trip():
    rng = np.random.default_rng(0)
    y = rng.normal(size=(500, 8)) * np.arange(1, 9) * 20 + np.arange(8)
    mean, sd = gn.standardiser(y)
    z = (y - mean) / sd
    assert np.allclose(z.mean(axis=0), 0) and np.allclose(z.std(axis=0), 1)
    assert np.allclose(z * sd + mean, y)
    y[:, 3] = 2.0
    with pytest.raises(SystemExit, match="spread"):
        gn.standardiser(y)


def test_fit_standardises_on_the_training_stacks_only(tree):
    lives, _ = gn.read_log("T2")
    data, info, train, held, scored, _ = gn.net_data("master", lives)
    model, mean, sd, history, best = gn.fit(data, train, scored, 0, 0,
                                            epochs=1)
    assert np.allclose(mean, data.y[train].mean(axis=0))
    assert np.allclose(sd, data.y[train].std(axis=0))
    assert not np.allclose(mean, data.y.mean(axis=0))
    assert best == 1 and len(history) == 1
    assert len(history[0]["held_mse"]) == 8


def test_fit_keeps_the_best_epoch_and_stops_after_patience(tree):
    lives, _ = gn.read_log("T0")
    data, info, train, held, scored, _ = gn.net_data("twin", lives)
    model, mean, sd, history, best = gn.fit(data, train, scored, 1, 1,
                                            epochs=8, patience=2)
    primary = [h["held_mse"][gn.PRIMARY] for h in history]
    assert best == 1 + int(np.argmin(primary))
    assert len(history) == min(8, best + 2)
    z = (data.y[scored] - mean) / sd
    again = ((gn.data_z(model, data, scored) - z) ** 2).mean(axis=0)
    assert again[gn.PRIMARY] == pytest.approx(primary[best - 1], rel=1e-5)


@pytest.mark.parametrize("offsets,best,epochs", [
    ((1.0, 0.5, 0.8, 0.9, 0.2, 0.1, 0.1, 0.1), 2, 4),   # stops 2 after it
    ((1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3), 8, 8),   # the cap
    ((0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5), 1, 3)])  # ties: the first
def test_early_stopping_on_scripted_held_out_losses(tree, monkeypatch,
                                                    offsets, best, epochs):
    """The held-out primary error per epoch scripted (prediction = target +
    offset, so the error is offset ** 2): the best epoch is the lowest, the
    first on ties, and training stops PATIENCE (here 2) epochs after it."""
    lives, _ = gn.read_log("T2")
    data, _, train, _, scored, _ = gn.net_data("master", lives)
    mean, sd = gn.standardiser(data.y[train])
    z = (data.y[scored] - mean) / sd
    script = iter(offsets)
    monkeypatch.setattr(gn, "data_z", lambda model, d, idx: z + next(script))
    _, _, _, history, got = gn.fit(data, train, scored, 0, 0, epochs=8,
                                   patience=2)
    assert (got, len(history)) == (best, epochs)
    assert [h["held_mse"][gn.PRIMARY] for h in history] == pytest.approx(
        [o * o for o in offsets[:epochs]])
    assert gn.best_at_cap(got, 8) is (best == 8)


def test_the_split_is_stratified_by_life_index():
    held = gn.split_lives([0, 1, 2] * 42, 0)
    assert len(held) == 24
    for k in range(3):
        assert sum(1 for p in held if p % 3 == k) == 8
    held = gn.split_lives([0, 1, 2] * 45, 1)
    assert len(held) == 27 and all(
        sum(1 for p in held if p % 3 == k) == 9 for k in range(3))
    assert held == gn.split_lives([0, 1, 2] * 45, 1)       # fixed seed
    assert held != gn.split_lives([0, 1, 2] * 45, 2)       # a net's own
    # floor(n / 5 + 1/2) a group
    for n, want in ((1, 0), (2, 0), (3, 1), (7, 1), (8, 2), (12, 2), (13, 3)):
        assert len(gn.split_lives([5] * n, 0)) == want
    assert len(gn.split_lives([0] * 13 + [1] * 2, 0)) == 3
    assert gn.split_lives([0] * 13 + [1] * 2, 0) == sorted(
        gn.split_lives([0] * 13 + [1] * 2, 0))


def test_held_out_stacks_in_training_are_not_scored(tree):
    """The first lives share their first 6 frames: stacks 0-2 of every first
    life are one stack; held-out copies of it are not scored."""
    lives, _ = gn.read_log("T2")
    data, info, train, held, scored, seen = gn.net_data("master", lives)
    firsts = [p for p, life in enumerate(lives) if life["index"] == 0]
    assert [p for p in info["held"] if p in firsts]       # one is held out
    shared = {i for i in held if data.life[i] in firsts and data.t[i] < 3}
    assert len(shared) == 3
    assert set(scored.tolist()) == set(held.tolist()) - shared
    assert seen == {data.digests[i] for i in train}
    assert all(data.digests[i] not in seen for i in scored)
    assert all(data.digests[i] in seen for i in shared)


def test_net_data_counts(tree):
    lives, _ = gn.read_log("T0")
    data, info, train, held, scored, _ = gn.net_data("late", lives)
    # the last third of 24 lives: positions 16-23, life indices 1 2 0 1 2 0
    # 1 2
    assert info["lives"] == 8 and info["iterations"] == [3, 4]
    assert info["by_index"] == {"0": [2, 0], "1": [3, 1], "2": [3, 1]}
    assert len(train) + len(held) == len(data) == 8 * sum(TOKENS)


# --------------------------------------------------------------------------
# R2 and G2
# --------------------------------------------------------------------------
def test_r2():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert gn.r2(y, y) == 1.0
    assert gn.r2(np.full(4, 2.5), y) == 0.0
    assert gn.r2(np.full(4, 2.0), y) < 0
    assert gn.r2(y, y, constant=0.0) == 1.0
    assert gn.r2(np.zeros(4), y, constant=0.0) == 0.0
    assert gn.r2(y, np.ones(4)) is None


def test_g2_verdict_is_the_first_clause_on_all_held_out_stacks():
    def reads(m, t, late, scored=-1.0):
        return {n: {"r2": v, "r2_scored": scored}
                for n, v in (("master", m), ("twin", t), ("late", late))}
    # the verdict names say they are G2's first clause, never all of G2
    assert (gn.G2_FIRST_PASS, gn.G2_FIRST_VOID) == ("G2_FIRST_CLAUSE_PASS",
                                                    "G2_FIRST_CLAUSE_VOID")
    # the scored-set R2 is descriptive: negative there does not VOID
    assert gn.g2_verdict(reads(0.2, 0.01, -1.0)) == (gn.G2_FIRST_PASS, [])
    v, why = gn.g2_verdict(reads(0.2, -0.01, 0.5, scored=0.9))
    assert v == gn.G2_FIRST_VOID and len(why) == 1
    assert why[0].startswith("twin") and "all held-out" in why[0]
    v, why = gn.g2_verdict(reads(0.0, 0.3, 0.5))
    assert v == gn.G2_FIRST_VOID and why[0].startswith("master")
    assert gn.g2_verdict(reads(None, 0.3, 0.5))[0] == gn.G2_FIRST_VOID


def test_best_epoch_at_the_cap():
    assert gn.best_at_cap(30, 30) and not gn.best_at_cap(29, 30)
    assert not gn.best_at_cap(1, 30)
    assert gn.best_at_cap(gn.EPOCHS) and not gn.best_at_cap(gn.EPOCHS - 1)


# --------------------------------------------------------------------------
# files: seeds averaged, stamps, writes
# --------------------------------------------------------------------------
def _seed_file(out, net, seed, bias, mean, sd, digests=(b"a" * 16,)):
    model = gn.Torso()
    with torch.no_grad():
        model.fc[2].weight.zero_()
        model.fc[2].bias.copy_(torch.as_tensor(bias, dtype=torch.float32))
    gn.write(gn.net_file(out, net, seed), {
        "net": net, "seed": seed, "state_dict": model.state_dict(),
        "mean": [float(v) for v in mean], "sd": [float(v) for v in sd],
        "train_digests": gn._digest_tensor(set(digests)),
        "split": {"held": [1]}})


def test_seeds_are_averaged(tmp_path):
    out = tmp_path
    mean, sd = np.arange(8.0), np.arange(1.0, 9.0)
    biases = [np.full(8, 1.0), np.full(8, -2.0), np.arange(8.0) / 4]
    for seed, b in zip(gn.SEEDS, biases):
        _seed_file(out, "twin", seed, b, mean, sd)
    stacks = np.random.default_rng(0).integers(0, 256, (5, 12, 84, 84),
                                               dtype=np.uint8)
    nets = gn.load("twin", out)
    per = nets.predict_seeds(stacks)
    assert per.shape == (3, 5, 8)
    for s, b in enumerate(biases):
        assert np.allclose(per[s], b * sd + mean, atol=1e-5)
    want = np.mean([b * sd + mean for b in biases], axis=0)
    assert np.allclose(gn.predict(stacks, "twin", out), want, atol=1e-5)
    assert gn.predict(stacks, "twin", out).dtype == np.float64
    assert nets.train_digests == frozenset({b"a" * 16})
    with pytest.raises(ValueError):
        nets.predict(stacks.astype(np.float32))
    with pytest.raises(ValueError):
        nets.predict(stacks[:, :9])
    with pytest.raises(ValueError):
        nets.predict(str(tmp_path))                  # stacks, never a path


def test_seeds_must_share_their_training_stacks(tmp_path):
    for seed in gn.SEEDS:
        _seed_file(tmp_path, "master", seed, np.zeros(8), np.zeros(8),
                   np.ones(8), digests=(bytes([seed]) * 16,))
    with pytest.raises(SystemExit, match="different stacks"):
        gn.load("master", tmp_path)


def test_digests_round_trip():
    ds = {bytes([i]) * 16 for i in range(5)}
    t = gn._digest_tensor(ds)
    assert t.dtype == torch.uint8 and t.shape == (5, 16)
    assert gn._digest_set(t) == ds
    stack = np.zeros((12, 84, 84), np.uint8)
    assert gn.stack_digest(stack) == pc._digest(stack.tobytes())
    with pytest.raises(ValueError):
        gn.stack_digest(stack.astype(np.int64))


@pytest.mark.parametrize("suffix", [".pt", ".json"])
def test_a_file_made_otherwise_is_refused(tmp_path, monkeypatch, suffix):
    path = tmp_path / f"x{suffix}"
    gn.write(path, {"a": 1})
    assert gn.read(path)["a"] == 1
    monkeypatch.setattr(gn, "EPOCHS", gn.EPOCHS + 1)
    with pytest.raises(SystemExit, match="never mixed"):
        gn.read(path)
    monkeypatch.undo()
    monkeypatch.setattr(gn, "SOURCE", "another")
    with pytest.raises(SystemExit, match="never mixed"):
        gn.read(path)
    monkeypatch.undo()
    monkeypatch.setattr(gn, "SMOKE", not gn.SMOKE)
    with pytest.raises(SystemExit, match="never mixed"):
        gn.read(path)


def test_writes_are_whole_or_nothing(tmp_path, monkeypatch):
    def broken(obj, f, *a, **k):
        pathlib.Path(f).write_bytes(b"half")
        raise RuntimeError("disk full")
    monkeypatch.setattr(torch, "save", broken)
    with pytest.raises(RuntimeError):
        gn.write(tmp_path / "net-x-s0.pt", {"a": 1})
    assert not (tmp_path / "net-x-s0.pt").exists()


def test_outputs_never_land_in_experiment_state(tmp_path):
    tree = gc.RESULTS.resolve().parents[1]
    for bad in (tree, tree / "x", gc.RESULTS / "y",
                PROBES / "probe_gap_check-pilot",
                PROBES / "probe_gap_check-pilot" / "sub",
                PROBES / "probe_gap_check-pilot-results",
                PROBES / "probe_gap_stage2a",
                PROBES / "probe_gap_stage2a-smoke"):
        with pytest.raises(SystemExit, match="never written"):
            gn.check_out(bad)
    assert gn.check_out(tmp_path / "o") == (tmp_path / "o").resolve()
    assert gn.default_out().name.startswith("probe_gap_nets-out")


# --------------------------------------------------------------------------
# a whole small run, in-process
# --------------------------------------------------------------------------
def test_a_whole_run_reads_only_its_logs(tree, monkeypatch, capsys):
    monkeypatch.setattr(gn, "EPOCHS", 2)
    out = gn.check_out(tree["out"])
    out.mkdir()
    OPENED.clear()
    RECORDING[0] = True
    try:
        for net in gn.NETS:
            for seed in gn.SEEDS:
                gn.stage_train(out, net, seed)
        gn.stage_g2(out)
    finally:
        RECORDING[0] = False
    printed = capsys.readouterr().out
    g2 = gn.read(out / "g2.json")
    assert set(g2["reads"]) == set(gn.NETS)
    assert g2["clause"] == "first"
    assert g2["verdict"] in (gn.G2_FIRST_PASS, gn.G2_FIRST_VOID)
    assert (g2["verdict"], g2["reasons"]) == gn.g2_verdict(g2["reads"])
    assert f"G2 FIRST CLAUSE: {g2['verdict']}" in printed
    assert g2["reads"]["late"]["gated"] is False
    # the cap flag: every pass file's and g2.json's, against the best epoch
    flagged = []
    for net in gn.NETS:
        for seed in gn.SEEDS:
            f = gn.read(gn.net_file(out, net, seed))
            assert f["epoch_cap"] == 2
            assert f["best_at_cap"] is (f["best_epoch"] == 2)
            if f["best_at_cap"]:
                flagged.append(f"{net} s{seed}")
                assert f"FLAG {net} seed {seed}" in printed
        assert g2["reads"][net]["best_at_cap"] == [
            s for s in gn.SEEDS if f"{net} s{s}" in flagged]
    assert g2["best_at_cap"] == flagged and g2["epoch_cap"] == 2
    # G2's R2 is on ALL held-out stacks, seed-averaged through predict; the
    # scored stacks' R2 is beside it
    for player, nets in (("T2", ("master",)), ("T0", ("twin", "late"))):
        lives, _ = gn.read_log(player)
        for net in nets:
            data, _, train, held, scored, _ = gn.net_data(net, lives)
            r = g2["reads"][net]
            pred = gn.predict(data.stacks(held), net, out)
            y = data.y[held]
            assert r["r2"] == pytest.approx(gn.r2(pred[:, 0], y[:, 0]),
                                            rel=1e-6, abs=1e-9)
            assert r["pass"] is (r["r2"] > 0)
            assert r["r2_vs_train_mean"] == pytest.approx(gn.r2(
                pred[:, 0], y[:, 0], data.y[train, 0].mean()), rel=1e-6,
                abs=1e-9)
            assert r["r2_per_head"] == pytest.approx(
                [gn.r2(pred[:, h], y[:, h]) for h in range(8)], rel=1e-6,
                abs=1e-9)
            keep = np.isin(held, scored)
            assert r["r2_scored"] == pytest.approx(
                gn.r2(pred[keep, 0], data.y[scored, 0]), rel=1e-6, abs=1e-9)
            if net != "late":                    # 3 or 6 shared stacks
                assert r["r2"] != pytest.approx(r["r2_scored"], abs=1e-6)
            assert len(r["r2_scored_per_seed"]) == 3
    # held-out first lives' 3 shared stacks: master 1 such life, twin 2, the
    # late twin none (its held lives are index 1 and 2)
    for net, shared in {"master": 3, "twin": 6, "late": 0}.items():
        r = g2["reads"][net]
        assert r["held_in_training"] == shared
        assert r["scored"] + shared == r["held_stacks"]
        assert len(r["r2_per_head"]) == 8 and len(r["r2_per_seed"]) == 3
    assert g2["reads"]["twin"]["held_lives"] == 6
    assert g2["reads"]["late"]["held_lives"] == 2
    master = gn.read(gn.net_file(out, "master", 0))
    assert master["split"]["held"] == gn.split_lives([0, 1, 2] * 3, 0)
    assert len(master["history"]) <= 2 and master["best_epoch"] >= 1
    # early stopping scored the stacks G2 scores: the best epoch's held-out
    # error is the saved predictions' error on them
    lives, _ = gn.read_log("T2")
    data, _, _, _, scored, _ = gn.net_data("master", lives)
    mean, sd = np.array(master["mean"]), np.array(master["sd"])
    z = (data.y[scored] - mean) / sd
    mse = (((master["held_pred"].numpy() - mean) / sd - z) ** 2).mean(axis=0)
    assert mse[gn.PRIMARY] == pytest.approx(
        master["history"][master["best_epoch"] - 1]["held_mse"][gn.PRIMARY],
        rel=1e-5)
    # what was opened: the fake logs and prep (read only), the outputs
    results = str(tree["results"])
    for path, mode in OPENED:
        assert "probe_gap_check-pilot" not in path
        assert "probe_gap_stage2" not in path
        assert "SwitchPlayer-testb2-run" not in path
        if path.startswith(results):
            assert "w" not in mode and "a" not in mode and "+" not in mode
        if any(c in mode for c in "wax+"):
            assert path.startswith(str(out)), (path, mode)
    assert any(p.startswith(results) for p, _ in OPENED)
    # reused, never overwritten, refused when made otherwise
    gn.stage_train(out, "master", 0)
    assert "reused" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="never overwritten"):
        gn.stage_g2(out)
    monkeypatch.setattr(gn, "EPOCHS", 3)
    with pytest.raises(SystemExit, match="never mixed"):
        gn.stage_train(out, "twin", 1)
    with pytest.raises(SystemExit, match="never mixed"):
        gn.load("master", out)


def test_g2_refuses_saved_predictions_that_do_not_come_back(tree,
                                                            monkeypatch):
    monkeypatch.setattr(gn, "EPOCHS", 1)
    out = tree["out"]
    out.mkdir()
    for net in gn.NETS:
        for seed in gn.SEEDS:
            gn.stage_train(out, net, seed)
    obj = torch.load(gn.net_file(out, "twin", 2), weights_only=True)
    obj["held_pred"] = obj["held_pred"] + 1.0
    torch.save(obj, gn.net_file(out, "twin", 2))
    with pytest.raises(SystemExit, match="do not come back"):
        gn.stage_g2(out)
    assert not (out / "g2.json").exists()


def test_a_best_epoch_at_the_cap_is_flagged(tree, monkeypatch, capsys):
    """One epoch: every best epoch is the cap, printed and recorded."""
    monkeypatch.setattr(gn, "EPOCHS", 1)
    out = tree["out"]
    out.mkdir()
    gn.stage_train(out, "master", 1)
    f = gn.read(gn.net_file(out, "master", 1))
    assert f["best_epoch"] == 1 and f["best_at_cap"] is True
    assert f["epoch_cap"] == 1
    assert "FLAG master seed 1" in capsys.readouterr().out
    monkeypatch.setattr(gn, "EPOCHS", 4)          # the cap is the knob's
    (out / "x").mkdir()
    gn.stage_train(out / "x", "master", 1)
    f = gn.read(gn.net_file(out / "x", "master", 1))
    assert f["epoch_cap"] == 4
    assert f["best_at_cap"] is (f["best_epoch"] == 4)
    assert ("FLAG master seed 1" in capsys.readouterr().out) is \
        f["best_at_cap"]


def test_importing_the_probe_changes_nothing_process_wide():
    code = ("import os, sys; sys.path[:0] = [{p!r}, {e!r}, {r!r}]; "
            "import torch; threads = torch.get_num_threads(); "
            "cuda = os.environ.get('CUDA_VISIBLE_DEVICES'); "
            "import probe_gap_nets; "
            "assert torch.get_num_threads() == threads; "
            "assert os.environ.get('CUDA_VISIBLE_DEVICES') == cuda; "
            "assert 'probe_exhaustive_eval' not in sys.modules; "
            "print('ok')").format(p=str(PROBES), e=str(ROOT / "experiments"),
                                  r=str(ROOT))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, cwd=str(ROOT), timeout=300)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr[-800:]


# --------------------------------------------------------------------------
# the real logs (slow: about 2 GB read)
# --------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.skipif(not gc.RESULTS.exists(), reason="no Test B2 run tree")
def test_the_real_logs_match_the_registration():
    OPENED.clear()
    RECORDING[0] = True
    try:
        lives, totals = gn.read_log("T2", max_lives=0)
        gn.check_totals("T2", lives, totals)
        assert len(lives) == 126
        assert sum(len(life["rewards"]) for life in lives) == 33021
        assert sum(float(life["rewards"].sum()) for life in lives) == 66850
        del lives
        lives, totals = gn.read_log("T0", max_lives=0)
        gn.check_totals("T0", lives, totals)
        assert len(lives) == 135 and totals["later_lives"] == 6
        assert max(life["iteration"] for life in lives) == 45
        late = gn.select("late", lives)
        assert len(late) == 45 and late[0] is lives[90]
        idx = [life["index"] for life in lives]
        assert idx[0] == 0 and set(idx) <= {0, 1, 2, 3}
    finally:
        RECORDING[0] = False
    for path, mode in OPENED:
        assert "w" not in mode and "a" not in mode and "+" not in mode
        assert "probe_gap_check-pilot" not in path
        assert "probe_gap_stage2" not in path
