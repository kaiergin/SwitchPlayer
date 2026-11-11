"""Test H, stage B, the truth (CURRICULUM "Next experiment (pre-registered) --
Test H", METHOD: TRUTH, COUNTERFACTUAL ITEMS, REACTION-FREE ITEMS). Pins: the
recorded window (the step's own digests kept apart from the later steps', the
regression the first smoke found; life ends, caps, a full window, a malformed
record refused); future and raw keys, the partition, the per-step digests
padded past a life's end; and, on the real game with a stand-in policy (no
snapshot), a game recorded by stage A's loop then put through the truth: no
failures, CRN continuations reproduce the record across a lost life, the NOOP
continuation plays NOOPs, the classes match the one-step merge, the blocks pack
as registered; branches at the real game over (the last life), one replayed on
a fresh emulator to the same frames and the game's end; that the checks catch a
corrupted frame, a corrupted digest and a policy shifted off the record; that
written() refuses a failed or partial truth; the floors on a hand case. Slow,
needing the snapshots: stage A writes one OF game, stage B finds no failure on
it and its floors count it."""
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
import testh_windows as tw  # noqa: E402

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


@pytest.fixture(scope="module")
def game_over_game():
    """A stand-in game played to the real game over (no cap reached)."""
    env, tap, noop, skip = mg.make_env()
    key = mg.game_key("standin", 2, 0)
    env.reset(seed=12)
    with mg.fixed_noops(noop, 3):
        arr, states = mg.play_game(env, tap, skip, standin_choose(key), opening=80,
                                   cap=3000, keep_states=True)
    st = {d: (blob, buf) for d, blob, buf, lives in states}
    yield env, tap, skip, arr, st, key
    env.close()


def test_branches_at_the_real_game_over(game_over_game):
    env, tap, skip, arr, st, key = game_over_game
    assert int(arr["end"][-1]) == 1 and env.was_real_done      # the game ended
    moves = np.flatnonzero(arr["kind"] == mg.KIND_MOVE)
    last_life = int(arr["life"][-1])
    final = [int(arr["decision"][i]) for i in moves if int(arr["life"][i]) == last_life]
    sub = {d: st[d] for d in final[-250:]}
    results, failures = tr.game_truth(env, tap, skip, arr, sub, standin_act(key))
    assert [f for f in failures if "no saved state" not in f] == []
    by_d = {r["decision"]: r for r in results}
    # the played line reaches the game over in every window that holds it, and
    # reproduces the record byte for byte there (failures == [] above): the
    # step that ends the game breaks MaxAndSkip's loop early, so its frame
    # reads the restored buffer. In Ms. Pac-Man the game ends some 30 base
    # steps after the fatal touch, so inside these windows every move is
    # already in the same death: other branches merge with the played one.
    for d in final[-tr.W:]:
        r = by_d[d]
        g = r["groups"][r["played"]]
        assert r["crn"][g]["end"] == tr.END_LIFE
    # the final decision's step, replayed on a fresh emulator: the game over
    # itself, the recorded frame and digests
    d = final[-1]
    i = int(np.flatnonzero((arr["kind"] == mg.KIND_MOVE) & (arr["decision"] == d))[0])
    blob, buf = st[d]
    env2, tap2, noop2, skip2 = mg.make_env()
    try:
        env2.reset(seed=0)
        tr.restore(env2, skip2, blob, buf)
        f, _, ended, dg = tr.step(env2, tap2, int(arr["move"][i]))
        assert ended and env2.unwrapped.ale.game_over() and env2.was_real_done
        assert np.array_equal(f, arr["frames"][i])
        assert dg == [bytes(x) for x in arr["digests"][i][:int(arr["ndigests"][i])]]
    finally:
        env2.close()
    # the death is long: decisions on the last life whose windows end before
    # the game over keep their classes apart (life ends are not merged away)
    assert any(len(set(by_d[d]["classes"]["crn"])) > 1 for d in final[:-tr.W])


def _stage_a_file(path, name, frames_fill):
    """A stage-A game file: a start, then moves (one life, ended)."""
    n = len(frames_fill)
    kinds = [mg.KIND_START] + [mg.KIND_MOVE] * (n - 1)
    ends = [0] * n
    ends[-1] = 1
    arr = {"kind": np.array(kinds, np.int8), "end": np.array(ends, np.int8),
           "move": np.zeros(n, np.int8),
           "decision": np.array([-1] + list(range(n - 1)), np.int32),
           "frames": np.stack([np.full((3, 84, 84), v, np.uint8) for v in frames_fill])}
    np.savez(path / f"{name}.npz", **arr)
    return n - 1


def _truth_file(path, name, n, contested):
    np.savez(path / f"{name}-truth.npz", decision=np.arange(n),
             contested=np.array(contested, bool))


def test_floors_fit_and_cross_game_exclusions(tmp_path):
    data, out = tmp_path / "data", tmp_path / "truth"
    data.mkdir(), out.mkdir()
    # FIT game frames 1..9; held A frames 5,20,21,...; held B frames 40,20,41...
    # A's decision 0 obs_t = tile(5) is not in FIT (FIT tiles 1); make A's first
    # stack equal FIT's by giving A the same start frame as FIT
    _stage_a_file(data, "FIT", [1] + list(range(2, 30)))
    na = _stage_a_file(data, "A", [1] + list(range(100, 128)))     # shares FIT's start
    nb = _stage_a_file(data, "B", [7] + list(range(150, 178)))
    nc = _stage_a_file(data, "C", [7] + list(range(200, 228)))     # shares B's start
    for name, n in (("A", na), ("B", nb), ("C", nc)):
        _truth_file(out, name, n, [True] * n)
    rep = tr.floors(data, out, held=[("O0", "A"), ("O0", "B"), ("OF", "C")], fit=["FIT"])
    # A: decision 0's obs_t (tile of frame 1) occurs in FIT -> out; B and C:
    # decision 0's obs_t (tile of 7) recurs across held-out games -> out of both
    assert rep["O0"]["per_game"] == [na - 1, nb - 1]
    assert rep["OF"]["per_game"] == [nc - 1]
    assert rep["windows_source"] == tw.SOURCE
    assert json.loads((out / "floors.json").read_text())["O0"]["contested"] == na + nb - 2


def test_before_gpu_gate(tmp_path, monkeypatch):
    import train_move_guesser as tg
    data, truth = tmp_path / "data", tmp_path / "truth"
    data.mkdir(), truth.mkdir()
    with pytest.raises(SystemExit, match="diversity"):
        tg.before_gpu(data, truth)
    (data / "diversity.json").write_text(json.dumps({"passed": True, "source": "older"}))
    with pytest.raises(SystemExit, match="for this stage A"):
        tg.before_gpu(data, truth)
    (data / "diversity.json").write_text(json.dumps({"passed": True, "source": mg.SOURCE}))
    monkeypatch.setattr(tr, "held_games", lambda: [("O0", 10, 8, "O0-g"), ("OF", 10, 0, "OF-g")])
    with pytest.raises(SystemExit, match="O0-g"):
        tg.before_gpu(data, truth)
    for n in ("O0-g", "OF-g"):
        (truth / f"{n}-truth.npz").write_bytes(b"x")
        (truth / f"{n}-truth.json").write_text(json.dumps(
            {"source": tr.SOURCE, "failures": [], "max_decisions": None}))
    with pytest.raises(SystemExit, match="floors"):
        tg.before_gpu(data, truth)
    fl = {"source": tr.SOURCE, "windows_source": tw.SOURCE, "of_void": False,
          "O0": {"passed": False}}
    (truth / "floors.json").write_text(json.dumps(fl))
    with pytest.raises(SystemExit, match="floors not passed"):
        tg.before_gpu(data, truth)
    fl["O0"]["passed"] = True
    (truth / "floors.json").write_text(json.dumps(fl))
    tg.before_gpu(data, truth)                                  # passes
    # OF's truth failed: blocked, unless OF_VOID.json records the decision
    (truth / "OF-g-truth.json").write_text(json.dumps(
        {"source": tr.SOURCE, "failures": ["decision 3: x"], "max_decisions": None}))
    with pytest.raises(SystemExit, match="OF-g"):
        tg.before_gpu(data, truth)
    (truth / "OF_VOID.json").write_text(json.dumps({"reason": "cannot be fixed: ..."}))
    # floors computed before the decision no longer match it
    with pytest.raises(SystemExit, match="OF_VOID state"):
        tg.before_gpu(data, truth)
    fl["of_void"] = True
    (truth / "floors.json").write_text(json.dumps(fl))
    tg.before_gpu(data, truth)
    (truth / "OF_VOID.json").write_text(json.dumps({}))
    with pytest.raises(SystemExit, match="reason"):
        tr.of_void(truth)


def test_written_refuses_failed_or_partial_truth(tmp_path):
    (tmp_path / "g-truth.npz").write_bytes(b"x")
    def put(**kw):
        info = {"source": tr.SOURCE, "failures": [], "max_decisions": None, **kw}
        (tmp_path / "g-truth.json").write_text(json.dumps(info))
    put()
    assert tr.written(tmp_path, "g") is not None
    put(failures=["decision 3: ..."])
    assert tr.written(tmp_path, "g") is None
    put(max_decisions=150)
    assert tr.written(tmp_path, "g") is None
    assert tr.written(tmp_path, "g", 150) is not None
    put(source="other")
    assert tr.written(tmp_path, "g") is None


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
    # the floors on this one game, no FIT: every contested decision counts
    rep = tr.floors(data, out, held=[("OF", name)], fit=[])
    assert rep["OF"]["contested"] == info["contested"]
    assert rep["OF"]["effective_games"] == pytest.approx(1.0)
    assert rep["OF"]["passed"] is False
