"""Test H, stage B, the truth (CURRICULUM "Next experiment (pre-registered) --
Test H", METHOD: TRUTH, COUNTERFACTUAL ITEMS, REACTION-FREE ITEMS). Pins: the
recorded window (the step's own digests kept apart from the later steps', the
regression the first smoke found; life ends, caps, a full window, a malformed
record refused); future and raw keys, the partition, the per-step digests
padded past a life's end; and, on the real game with a stand-in policy (no
snapshot), a game recorded by stage A's loop then put through the truth: no
failures, the played move's class holds it, CRN continuations reproduce the
record across a lost life, the NOOP continuation plays NOOPs, the classes match
the one-step merge, the blocks pack as registered; and that the checks catch a
corrupted frame, a corrupted digest and a wrong uniform key. Slow, needing the
snapshots: stage A writes one OF game and stage B finds no failure on it."""
import hashlib
import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROBES = ROOT / "benchmarks" / "probes"
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(PROBES))
import probe_move_guesser as mg  # noqa: E402
import probe_move_guesser_truth as tr  # noqa: E402

HAVE_SNAPSHOTS = all(
    (pathlib.Path(s["tree"]) / "benchmarks" / "results" / s["run"]).exists()
    for s in mg.RUNS.values())


# --------------------------------------------------------------------------
# synthetic
# --------------------------------------------------------------------------
def _arr(kinds, ends, moves=None):
    n = len(kinds)
    dig = np.zeros((n, 34, mg.DIGEST), np.uint8)
    for i in range(n):
        for k in range(4):
            dig[i, k] = i * 4 + k
    return {"kind": np.array(kinds, np.int8),
            "move": np.array(moves or [i % 9 for i in range(n)], np.int8),
            "end": np.array(ends, np.int8),
            "reward": np.arange(n, dtype=np.float32),
            "frames": np.stack([np.full((3, 84, 84), i % 256, np.uint8)
                                for i in range(n)]),
            "digests": dig, "ndigests": np.full(n, 4, np.int16),
            "decision": np.array([i if k == mg.KIND_MOVE else -1
                                  for i, k in enumerate(kinds)], np.int32)}


def test_recorded_window_keeps_the_steps_own_digests_apart():
    arr = _arr([0] + [2] * 30, [0] * 31)
    arr["end"][-1] = 1
    rec = tr.recorded_window(arr, 3)
    assert rec["step_digests"] == [bytes(arr["digests"][3][k]) for k in range(4)]
    assert rec["digests"][0] == [bytes(arr["digests"][4][k]) for k in range(4)]
    assert len(rec["moves"]) == 15 and rec["end"] == tr.END_NONE
    assert rec["moves"] == [int(m) for m in arr["move"][4:19]]
    assert all(np.array_equal(a, b) for a, b in zip(rec["frames"], arr["frames"][4:19]))


def test_recorded_window_life_end_cap_and_malformed():
    arr = _arr([0, 2, 2, 2, 2, 0, 2, 2], [0, 0, 0, 1, 0, 0, 2, 0])
    rec = tr.recorded_window(arr, 1)
    assert rec["moves"] == [int(arr["move"][2]), int(arr["move"][3])]
    assert rec["end"] == tr.END_LIFE
    rec = tr.recorded_window(arr, 3)            # the step itself lost the life
    assert rec["ended"] and rec["moves"] == [] and rec["end"] == tr.END_LIFE
    rec = tr.recorded_window(arr, 6)            # the step reached the cap
    assert not rec["ended"] and rec["moves"] == [] and rec["end"] == tr.END_CAP
    bad = _arr([0, 2, 2, 0, 2], [0, 0, 0, 0, 1])
    with pytest.raises(ValueError, match="without an end flag"):
        tr.recorded_window(bad, 1)


def test_keys_partition_and_padding():
    f = [np.full((3, 84, 84), i, np.uint8) for i in range(5)]
    k1 = tr.future_key(f[0], False, f[1:3], tr.END_LIFE)
    k2 = tr.future_key(f[0], False, f[1:3], tr.END_NONE)
    k3 = tr.future_key(f[0], False, f[1:3], tr.END_LIFE)
    assert k1 == k3 and k1 != k2 and k1[1] == 3
    assert tr.future_key(f[0], True, [], tr.END_NONE)[2] == tr.END_LIFE
    assert tr.partition(["a", "b", "a", "c", "b"]) == [0, 1, 0, 2, 1]
    ds = tr.step_digests(f[0], f[1:3])
    assert len(ds) == tr.W and ds[2] == ds[3] == ds[-1] != ds[1]


# --------------------------------------------------------------------------
# the real game with a stand-in policy
# --------------------------------------------------------------------------
def standin_prior(stack):
    h = hashlib.blake2b(np.ascontiguousarray(stack).tobytes(), digest_size=36).digest()
    x = np.frombuffer(h, np.uint8)[:mg.MOVES].astype(np.float32)
    p = np.exp(x / 40.0).astype(np.float32)
    return p / p.sum()


T = 0.5


def standin_choose(key):
    def choose(stack, j, decision):
        if decision is None:
            return mg.opening_move(key, j), None, None, None
        p = standin_prior(stack)
        return mg.sample(mg.tempered(p, T), mg.uniform(key, decision)), \
            mg.uniform(key, decision), p, T
    return choose


def standin_act(key):
    def act(stack, index):
        return mg.sample(mg.tempered(standin_prior(stack), T), mg.uniform(key, index))
    return act


@pytest.fixture(scope="module")
def game():
    env, tap, noop, skip = mg.make_env()
    key = mg.game_key("standin", 1, 0)
    env.reset(seed=11)
    with mg.fixed_noops(noop, 9):
        arr, states = mg.play_game(env, tap, skip, standin_choose(key), opening=80,
                                   cap=700, keep_states=True)
    st = {d: (blob, buf) for d, blob, buf, lives in states}
    yield env, tap, skip, arr, st, key
    env.close()


def _decisions_near_life_end(arr, span=tr.W):
    out = []
    for i in range(len(arr["kind"])):
        if int(arr["kind"][i]) == mg.KIND_MOVE:
            k = i
            while k < len(arr["kind"]) and int(arr["end"][k]) == 0 and k - i < span:
                k += 1
            if k < len(arr["kind"]) and int(arr["end"][k]) == 1 and k - i < span:
                out.append(int(arr["decision"][i]))
    return out


def test_truth_on_a_standin_game(game):
    env, tap, skip, arr, st, key = game
    near = _decisions_near_life_end(arr)
    assert near, "the stand-in game lost no life inside a window"
    pick = sorted(set(list(st)[:25]) | set(near[:20]))
    sub = {d: st[d] for d in pick}
    results, failures = tr.game_truth(env, tap, skip, arr, sub, standin_act(key),
                                      cap=700)
    # game_truth reports every decision without a state as a failure: keep those apart
    real = [f for f in failures if "no saved state" not in f]
    assert real == []
    by_d = {r["decision"]: r for r in results}
    assert set(by_d) == set(pick)
    for d, r in by_d.items():
        cls = r["classes"]["crn"]
        # classes never split a one-step merge group
        for m in range(mg.MOVES):
            for m2 in range(mg.MOVES):
                if r["groups"][m] == r["groups"][m2]:
                    assert cls[m] == cls[m2]
                    assert r["classes"]["noop"][m] == r["classes"]["noop"][m2]
        for b in r["noop"]:
            assert all(m == 0 for m in b["moves"])
        assert r["contested"] == (len(set(cls)) > 1)
    ended = [by_d[d] for d in near[:20] if d in by_d]
    assert any(r["crn"][r["groups"][r["played"]]]["end"] == tr.END_LIFE for r in ended)
    packed = tr.pack(results)
    want = sum((len(set(r["classes"]["crn"])) - 1) + len(set(r["classes"]["noop"]))
               for r in results if r["contested"])
    assert len(packed["blk_dec"]) == want
    for b in range(len(packed["blk_dec"])):
        n = int(packed["blk_len"][b])
        assert 1 <= n <= tr.W
        assert not packed["blk_frames"][b, n:].any()


def test_checks_catch_a_corrupted_record_and_a_wrong_key(game):
    env, tap, skip, arr, st, key = game
    d = sorted(st)[5]
    i = {int(arr["decision"][k]): k for k in range(len(arr["kind"]))
         if int(arr["kind"][k]) == mg.KIND_MOVE}[d]
    stacks = mg.stacks_from_frames(arr)
    blob, buf = st[d]
    life_steps = tr.steps_in_life_before(arr, i)
    rec = tr.recorded_window(arr, i)
    assert rec["moves"], "pick a decision whose window has later steps"
    _, fail = tr.decision_truth(env, tap, skip, blob, buf, stacks[d], life_steps,
                                standin_act(key), d, rec)
    assert fail == []
    bad = dict(rec, frame=rec["frame"].copy())
    bad["frame"][0, 0, 0] ^= 1
    _, fail = tr.decision_truth(env, tap, skip, blob, buf, stacks[d], life_steps,
                                standin_act(key), d, bad)
    assert any("step differs" in f for f in fail)
    bad = dict(rec, step_digests=[b"\x00" * mg.DIGEST] + rec["step_digests"][1:])
    _, fail = tr.decision_truth(env, tap, skip, blob, buf, stacks[d], life_steps,
                                standin_act(key), d, bad)
    assert any("step differs" in f for f in fail)
    act = standin_act(key)
    _, fail = tr.decision_truth(env, tap, skip, blob, buf, stacks[d], life_steps,
                                lambda s, k: (act(s, k) + 1) % mg.MOVES, d, rec)
    assert any("continuation differs" in f for f in fail)


# --------------------------------------------------------------------------
# slow: a real snapshot
# --------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.skipif(not HAVE_SNAPSHOTS, reason="needs OF's snapshots")
def test_stage_truth_on_one_of_game(tmp_path):
    data = tmp_path / "data"
    mg.stage_data(data, ["OF"], iterations=(159,), games=1, cap=400)
    name = mg.game_name("OF", 159, 0)
    assert (data / f"{name}-states.npz").exists()
    out = tmp_path / "truth"
    tr.stage_truth(data, out, only={name}, max_decisions=40)
    info = json.loads((out / f"{name}-truth.json").read_text())
    assert info["failures"] == [] and info["decisions"] == 40
    with np.load(out / f"{name}-truth.npz") as z:
        assert z["class_crn"].shape == (40, mg.MOVES)
        assert int(z["contested"].sum()) == info["contested"]
