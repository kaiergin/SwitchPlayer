"""Curriculum P3's distillation step (Kai, 2026-10-21; CURRICULUM "P3's
distillation step"). Pins: the two student arms' deltas and DS's end-state
schedules; the student vocabulary (validation, --token-vocab, the gate, the
resume recipe); the master recorder's raw-step hook (play_token's
INNER_FRAMES) and its stack check; Kai's drop rule; the re-tokenization (a
dropped token's base steps, rewards kept exactly once); the masked master
targets; the set file; run_one's wiring -- and, on real Ms. Pac-Man (CPU),
a whole prep from a master snapshot, then the distilled student's prefill
and offline phase with its snapshots."""
import inspect
import json
import pathlib
import pickle
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] /
                       "experiments"))
import ablation
import distill
import muzero
from tests.test_prioract2 import restore_rngs  # noqa: F401
from tests.test_tokens import _Steps

TK_ARM = "effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-tokens-rgrow"
DS_ARM = TK_ARM + "-student-distill"
FS_ARM = TK_ARM + "-student-fresh"
ENV_ID = "MsPacmanNoFrameskip-v4"
MASTER_VOCAB = ablation.ARMS[TK_ARM]["TOKEN_VOCAB"]
END_STATE = {"REWARD_GROWTH_START": 160.0, "PRIOR_ACT_STAGED_TEMP": False,
             "ACTING_TEMPERATURE": 0.25, "TOKEN_LONG_BIAS": 0.0,
             "PER_BETA": 1.0}


def _student_vocab(drop=((0, 8), (0, 32), (5, 16))):
    return tuple(t for t in MASTER_VOCAB if t not in drop)


def _vocab_file(tmp_path, vocab=None, name="vocab.json", set_path=None):
    p = tmp_path / name
    doc = {"vocab": [list(t) for t in (vocab or _student_vocab())]}
    if set_path is not None:
        doc["set_sha256"] = ablation.file_sha256(set_path)
    p.write_text(json.dumps(doc))
    return p


@pytest.fixture
def student(monkeypatch):
    """ARMS restored after a test applies --token-vocab."""
    for arm in (DS_ARM, FS_ARM, TK_ARM):
        monkeypatch.setitem(ablation.ARMS, arm, ablation.ARMS[arm])


# --------------------------------------------------------------------------
# the arms
# --------------------------------------------------------------------------
def test_ds_arm_delta_is_exactly_the_end_state_and_the_offline_phase():
    tk, ds = ablation.ARMS[TK_ARM], ablation.ARMS[DS_ARM]
    delta = {k for k in set(tk) | set(ds) if tk.get(k) != ds.get(k)}
    assert delta == set(END_STATE) | {"_student", "_distill_offline_steps",
                                      "_distill_snapshot_every", "_doc"}
    assert {k: ds[k] for k in END_STATE} == END_STATE
    assert (ds["_student"], ds["_distill_offline_steps"],
            ds["_distill_snapshot_every"]) == ("distill", 20000, 4000)
    assert ds["TOKEN_VOCAB"] == MASTER_VOCAB    # replaced by --token-vocab


def test_fs_arm_is_tk_over_the_student_vocabulary():
    tk, fs = ablation.ARMS[TK_ARM], ablation.ARMS[FS_ARM]
    delta = {k for k in set(tk) | set(fs) if tk.get(k) != fs.get(k)}
    assert delta == {"_student", "_doc"} and fs["_student"] == "fresh"


def test_ds_runs_the_masters_end_state_from_step_0():
    """DS's schedules are constant and equal TK's at the end of its run."""
    with ablation.arm_config(TK_ARM), ablation._run_length_globals(90, 2000):
        end = 180000
        tk_end = (muzero.reward_ceiling(end),
                  muzero.prior_acting_temperature(end),
                  muzero.per_beta(end))
    assert tk_end == (160.0, 0.125, 1.0)
    with ablation.arm_config(DS_ARM), ablation._run_length_globals(90, 2000):
        for ts in (0, 1, 20000, 90000, 135000, 200000):
            assert (muzero.reward_ceiling(ts),
                    muzero.prior_acting_temperature(ts),
                    muzero.per_beta(ts)) == tk_end
            assert muzero.token_long_bias(ts) == 0.0
    with ablation.arm_config(FS_ARM), ablation._run_length_globals(90, 2000):
        # FS keeps TK's beginning-of-learning schedules
        assert muzero.reward_ceiling(0) == 1.0
        assert muzero.prior_acting_temperature(0) == 0.5
        assert muzero.token_long_bias(0) == 0.5


@pytest.mark.parametrize("arm", [DS_ARM, FS_ARM])
def test_every_startup_gate_passes_for_the_students(arm, monkeypatch,
                                                    tmp_path, student):
    for k in ablation.ARMS[arm]:
        if hasattr(muzero, k):
            monkeypatch.setattr(muzero, k, getattr(muzero, k))
    for k in ("ENV_NAME", "FRAMESKIP", "ACTIONS"):
        monkeypatch.setattr(ablation, k, getattr(ablation, k))
    ablation.ENV_NAME, ablation.FRAMESKIP, ablation.ACTIONS = ENV_ID, 4, 9
    dset = None
    if arm == DS_ARM:
        dset = tmp_path / "set.pkl"
        distill.write_set(dset, _header(), [])
    ablation.student_arm(arm, _vocab_file(tmp_path, set_path=dset))
    with ablation.arm_config(arm) as spec, \
            ablation._run_length_globals(90, 2000), ablation.token_actions():
        assert ablation.ACTIONS == 33
        ablation.ray_scope_gate(spec)
        ablation.launch_gate(spec, True, True)
        ablation.eager_collect_gate(True)
        ablation.prior_temp_gate()
        ablation.bn_recal_gate(True)
        ablation.search_act_birth_gate()
        ablation.birth_search_gate()
        ablation.sve_gate()
        ablation.token_gate(spec, True, True, 0)
        ablation.reward_growth_gate(True, True, 0)
        ablation.student_gate(spec, dset)


def _header(vocab=None, master=None):
    return {"format": distill.FORMAT,
            "vocab": [list(t) for t in (vocab or _student_vocab())],
            "master_vocab": [list(t) for t in (master or MASTER_VOCAB)],
            "master_snapshot": "m.pth"}


# --------------------------------------------------------------------------
# the student vocabulary
# --------------------------------------------------------------------------
def test_validate_student_vocab_accepts_an_ordered_subset():
    v = ablation.validate_student_vocab([list(t) for t in _student_vocab()],
                                        MASTER_VOCAB)
    assert v == _student_vocab() and isinstance(v[0], tuple)
    # every non-base token may go
    base = tuple(t for t in MASTER_VOCAB if t[1] == 4)
    assert ablation.validate_student_vocab(base, MASTER_VOCAB) == base


@pytest.mark.parametrize("vocab, match", [
    (_student_vocab() + ((9, 4),), "outside the master"),
    (_student_vocab()[::-1], "slot order"),
    (_student_vocab()[:3] + _student_vocab()[2:], "slot order"),
    (tuple(t for t in MASTER_VOCAB if t != (3, 4)), "drops a 4-frame"),
    ("nonsense", "pairs"),
])
def test_validate_student_vocab_refusals(vocab, match):
    with pytest.raises(SystemExit, match=match):
        ablation.validate_student_vocab(vocab, MASTER_VOCAB)


def test_slot_0_must_stay_the_placeholder():
    with pytest.raises(SystemExit, match="slot 0|drops a 4-frame"):
        ablation.validate_student_vocab(MASTER_VOCAB[1:], MASTER_VOCAB)


def test_student_arm_applies_the_file_and_refuses_other_arms(tmp_path,
                                                             student):
    f = _vocab_file(tmp_path)
    spec = ablation.student_arm(FS_ARM, f)
    assert ablation.ARMS[FS_ARM] is spec
    assert spec["TOKEN_VOCAB"] == _student_vocab()
    assert spec["_token_vocab_sha256"] == ablation.file_sha256(f)
    with ablation.arm_config(FS_ARM):
        assert muzero.TOKEN_VOCAB == _student_vocab()
    with pytest.raises(SystemExit, match="student arms only"):
        ablation.student_arm(TK_ARM, f)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"tokens": []}))
    with pytest.raises(SystemExit, match="no 'vocab'"):
        ablation.student_arm(DS_ARM, bad)


def test_student_gate_refusals(tmp_path, student):
    good = tmp_path / "set.pkl"
    distill.write_set(good, _header(), [])
    other = tmp_path / "other.pkl"
    distill.write_set(other, _header(vocab=_student_vocab(((7, 8),))), [])
    foreign = tmp_path / "foreign.pkl"
    distill.write_set(foreign, _header(master=MASTER_VOCAB[:-1]), [])
    twin = tmp_path / "twin.pkl"          # same header, other bytes
    distill.write_set(twin, _header(), [{"traj": [], "terminated": True}])

    def gate(arm, dset, applied=True, snaps=True):
        if applied:
            ablation.student_arm(arm, _vocab_file(tmp_path, set_path=good))
        with ablation.arm_config(arm) as spec:
            ablation.student_gate(spec, dset, snaps)

    with pytest.raises(SystemExit, match="needs --token-vocab"):
        gate(DS_ARM, good, applied=False)
    with pytest.raises(SystemExit, match="needs --distill-set"):
        gate(DS_ARM, None)
    with pytest.raises(SystemExit, match="needs --snapshot-weights"):
        gate(DS_ARM, good, snaps=False)                       # r1 A-L2
    with pytest.raises(SystemExit, match="differs from --token-vocab"):
        gate(DS_ARM, other)
    with pytest.raises(SystemExit, match="other than the TK arm"):
        gate(DS_ARM, foreign)
    with pytest.raises(SystemExit, match="set_sha256"):
        gate(DS_ARM, twin)                                    # r1 A-NIT3
    with pytest.raises(SystemExit, match="takes no --distill-set"):
        gate(FS_ARM, good)
    with pytest.raises(SystemExit, match="-student-distill arm only"):
        with ablation.arm_config(TK_ARM) as spec:
            ablation.student_gate(spec, good)
    gate(DS_ARM, good)
    gate(FS_ARM, None)
    gate(FS_ARM, None, snaps=False)       # FS's snapshots are the usual ones
    with ablation.arm_config(TK_ARM) as spec:
        ablation.student_gate(spec, None)            # other arms: no-op
    # a vocab.json without the set's hash refuses DS
    ablation.student_arm(DS_ARM, _vocab_file(tmp_path))
    with ablation.arm_config(DS_ARM) as spec:
        with pytest.raises(SystemExit, match="set_sha256"):
            ablation.student_gate(spec, good)


def test_student_gate_refuses_a_growing_ceiling_or_bad_offline_steps(
        tmp_path, student, monkeypatch):
    dset = tmp_path / "set.pkl"
    distill.write_set(dset, _header(), [])
    ablation.student_arm(DS_ARM, _vocab_file(tmp_path, set_path=dset))
    with ablation.arm_config(DS_ARM) as spec:
        muzero.REWARD_GROWTH_START = 1.0
        with pytest.raises(SystemExit, match="end-state ceiling"):
            ablation.student_gate(spec, dset)
    applied = ablation.ARMS[DS_ARM]
    for steps, every in ((20000, 3000), (0, 0), (4000, 8000)):
        monkeypatch.setitem(ablation.ARMS, DS_ARM, {
            **applied, "_distill_offline_steps": steps,
            "_distill_snapshot_every": every})
        with ablation.arm_config(DS_ARM) as spec:
            with pytest.raises(SystemExit, match="multiple"):
                ablation.student_gate(spec, dset)


def test_the_resume_recipe_carries_the_vocabulary_and_the_set(tmp_path,
                                                              student):
    plain = ablation.run_expect(TK_ARM, 0, 90, 2000, 3, 5)
    assert set(plain) == {"arm", "seed", "env", "iterations",
                          "train_steps_per_iter", "episodes_per_iter",
                          "episode_cap", "eval_every", "ray_workers"}
    with ablation.arm_config(TK_ARM) as spec:
        assert ablation.student_expect(spec, None) is None
    dset = tmp_path / "set.pkl"
    distill.write_set(dset, _header(), [])
    ablation.student_arm(DS_ARM, _vocab_file(tmp_path))
    with ablation.arm_config(DS_ARM) as spec:
        extra = ablation.student_expect(spec, dset)
    assert extra == {"token_vocab": [list(t) for t in _student_vocab()],
                     "distill_set_sha256": ablation.file_sha256(dset)}
    full = ablation.run_expect(DS_ARM, 0, 90, 2000, 3, 5, student=extra)
    assert {k: full[k] for k in plain} == {**plain, "arm": DS_ARM}
    assert full["token_vocab"] == extra["token_vocab"]


def test_run_one_wires_the_students():
    src = inspect.getsource(ablation.run_one)
    assert "student_gate(spec, distill_set, snapshot_weights)" in src
    assert src.index("reward_growth_gate(") < src.index("student_gate(")
    assert src.index("student_gate(") < src.index(
        "refuse_conflicting_state(")
    assert "student=student_expect(spec, distill_set)" in src
    hook = 'if spec.get("_student") == "distill" and start_it == 0:'
    assert src.count(hook) == 1
    i = src.index(hook)
    # after the resume and the warm start, before the pipeline and the loop
    assert src.index("load_run_checkpoint(") < i
    assert src.index("WARMSTART_PREFILL_EPISODES") < i
    assert i < src.index("pipeline = (RayPipeline(")
    assert i < src.index("for it in range(start_it + 1, iterations + 1):")
    block = src[i:src.index("def write_row(")]
    assert "distill_prefill(run, lives)" in block
    assert block.index("distill_prefill(") < block.index("distill_offline(")
    main = inspect.getsource(ablation.main)
    assert "student_arm(arm, args.token_vocab)" in main
    assert "distill_set=args.distill_set" in main


def test_main_parses_the_student_flags(monkeypatch, tmp_path, student):
    seen = {}
    monkeypatch.setattr(ablation, "run_one",
                        lambda *a, **k: seen.update(k, arm=a[0]))
    monkeypatch.setattr(ablation, "plot", lambda *a: None)
    monkeypatch.setattr(ablation, "RESULTS", tmp_path)
    for k in ("ENV_NAME", "TAG", "FRAMESKIP", "EPISODE_CAP", "ACTIONS"):
        monkeypatch.setattr(ablation, k, getattr(ablation, k))
    f = _vocab_file(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "ablation.py", "--arms", FS_ARM, "--seed", "0", "--env", "Pong-v4",
        "--frameskip", "4", "--token-vocab", str(f)])
    ablation.main()
    assert seen["arm"] == FS_ARM and seen["distill_set"] is None
    assert ablation.ARMS[FS_ARM]["TOKEN_VOCAB"] == _student_vocab()
    monkeypatch.setattr(sys, "argv", [
        "ablation.py", "--arms", TK_ARM, "--seed", "0", "--env", "Pong-v4",
        "--frameskip", "4", "--token-vocab", str(f)])
    with pytest.raises(SystemExit, match="student arms only"):
        ablation.main()


# --------------------------------------------------------------------------
# the recorder's hook
# --------------------------------------------------------------------------
@pytest.fixture
def tokens4(monkeypatch):
    monkeypatch.setattr(muzero, "TOKEN_VOCAB",
                        ((0, 4), (2, 8), (3, 16), (0, 32)))
    monkeypatch.setattr(ablation, "INNER_FRAMES", None)


def _stack0(env):
    obs, _ = env.reset()
    f = muzero.preprocess_frame(obs, ablation.IMAGE_RESHAPE)
    return muzero.initial_frame_stack(f)


def test_play_token_records_its_inner_frames_only_when_asked(tokens4):
    plain_env, rec_env = _Steps(), _Steps()
    a = ablation.play_token(plain_env, 2, _stack0(plain_env))   # (3, 16)
    ablation.INNER_FRAMES = []
    b = ablation.play_token(rec_env, 2, _stack0(rec_env))
    assert a[1:4] + (a[5],) == b[1:4] + (b[5],)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[4], b[4])
    assert len(ablation.INNER_FRAMES) == 1
    frames = ablation.INNER_FRAMES[0]
    assert len(frames) == 3                       # 4 base steps, 3 pushed
    want = [muzero.preprocess_frame(rec_env.frame(t), ablation.IMAGE_RESHAPE)
            for t in (1, 2, 3)]
    assert all(np.array_equal(x, y) for x, y in zip(frames, want))
    s = _stack0(_Steps())
    for f in frames:
        s = muzero.push_frame(s, f)
    assert np.array_equal(s, b[4])                 # exactly what was pushed
    ablation.play_token(rec_env, 0, b[4])          # a base token: none
    assert ablation.INNER_FRAMES[1] == []


def test_play_token_records_done_minus_one_at_a_terminal(tokens4):
    env = _Steps(end=2)
    ablation.INNER_FRAMES = []
    out = ablation.play_token(env, 3, _stack0(env))
    assert out[5] == 2 and len(ablation.INNER_FRAMES[0]) == 1


# --------------------------------------------------------------------------
# the drop rule, re-tokenization, targets, the set file
# --------------------------------------------------------------------------
def test_student_vocabulary_drops_below_the_share_never_a_base_token():
    master = ((0, 4), (0, 8), (0, 16), (1, 4), (1, 16), (1, 32))
    shares = [0.0, 0.0099, 0.01, 0.5, 0.48, 0.0001]
    vocab, dropped = distill.student_vocabulary(master, shares)
    assert vocab == ((0, 4), (0, 16), (1, 4), (1, 16))
    assert dropped == ((0, 8), (1, 32))
    assert distill.DROP_SHARE == 0.01


def test_token_shares_count_acted_records_only():
    lives = [{"traj": [(None, 2, ()), (None, 2, (0.0,)), (None, 0, (0.0,))]},
             {"traj": [(None, 1, ()), (None, 0, (0.0,))]}]
    shares, counts = distill.token_shares(lives, 3)
    assert counts.tolist() == [0, 1, 2]
    assert shares.tolist() == pytest.approx([0, 1 / 3, 2 / 3])


def _frame(v):
    return np.full((3, 4, 4), v, dtype=np.uint8)


def _life():
    """A recorded life over MASTER = ((0,4),(0,8),(0,16),(1,4),(1,16)):
    d0 (0,16) plays 4 base steps (frames 1-4, rewards 1-4); d1 (1,16)
    plays 4 (frames 5-8, rewards 5-8); d2 (0,8) is cut short after 1 base
    step (frame 9, reward 9) by a life loss; the final record."""
    s = muzero.initial_frame_stack(_frame(0))
    traj, inner, t = [], [], 0
    plan = [(2, 4), (4, 4), (1, 1)]
    incoming = ()
    for slot, m in plan:
        traj.append((s, slot, incoming))
        pushed = []
        for j in range(1, m + 1):
            t += 1
            if j < m:
                pushed.append(_frame(t))
            s = muzero.push_frame(s, _frame(t))
        inner.append(pushed)
        incoming = tuple(float(x) for x in range(t - m + 1, t + 1))
    traj.append((s, 0, incoming))
    return traj, inner


MASTER5 = ((0, 4), (0, 8), (0, 16), (1, 4), (1, 16))
STUDENT5 = ((0, 4), (1, 4), (1, 16))          # (0,8), (0,16) dropped


def test_check_life_passes_a_recording_and_catches_a_wrong_frame():
    traj, inner = _life()
    distill.check_life(traj, inner)
    bad = [list(x) for x in inner]
    bad[0][1] = _frame(99)
    with pytest.raises(AssertionError):
        distill.check_life(traj, bad)
    with pytest.raises(AssertionError):
        distill.check_life(traj, inner[:-1])
    short = [list(x) for x in inner]
    short[1] = short[1][:-1]
    with pytest.raises(AssertionError):
        distill.check_life(traj, short)


def test_retokenize_splits_dropped_tokens_into_base_steps():
    traj, inner = _life()
    out = distill.retokenize(traj, inner, MASTER5, STUDENT5)
    # d0 (0,16) dropped -> 4 x (0,4); d1 (1,16) kept; d2 (0,8) cut short
    # after 1 step -> 1 x (0,4); then the final record
    assert [r[1] for r in out] == [0, 0, 0, 0, 2, 0, 0]
    assert [r[2] for r in out] == [(), (1.0,), (2.0,), (3.0,), (4.0,),
                                   (5.0, 6.0, 7.0, 8.0), (9.0,)]
    # each split record sits on the stack at its base step
    for j in range(4):
        assert out[j][0][-3:].max() == j            # newest frame = step j
    assert out[0][0] is traj[0][0]
    assert out[4][0] is traj[1][0] and out[5][0] is traj[2][0]
    assert out[6][0] is traj[3][0]
    # every raw reward and base step exactly once
    assert sum(sum(r[2]) for r in out) == sum(sum(r[2]) for r in traj)
    split = [r for r in out[:-1]]
    frames = sum(STUDENT5[r[1]][1] for r in split)
    assert frames == 4 * 4 + 16 + 4                # the cut-short token: 1


def test_retokenize_keeps_a_life_without_dropped_tokens_as_is():
    traj, inner = _life()
    out = distill.retokenize(traj, inner, MASTER5, MASTER5)
    assert [(r[1], r[2]) for r in out] == [(r[1], r[2]) for r in traj]
    assert all(a[0] is b[0] for a, b in zip(out, traj))


def test_masked_targets_renormalize_over_the_student():
    pols = [np.array([0.1, 0.2, 0.3, 0.4, 0.0]),
            np.array([0.0, 0.5, 0.5, 0.0, 0.0])]
    out, fb, gone = distill.masked_targets(pols, MASTER5, STUDENT5)
    assert out[0] == pytest.approx([0.2, 0.8, 0.0])
    assert out[1] == pytest.approx([1 / 3] * 3) and fb == 1
    assert gone == pytest.approx([0.5, 1.0])     # the discarded visit mass


class _ReplayEnv:
    """Deterministic frames and rewards per step; terminates at `end`."""

    def __init__(self, end=None):
        self.end, self.t, self.was_real_done = end, 0, True

    def frame(self, t):
        return np.random.default_rng(1000 + t).integers(
            0, 256, (84, 84, 3), dtype=np.uint8)

    def reset(self, seed=None, options=None):
        self.t = 0
        return self.frame(0), {}

    def step(self, a):
        self.t += 1
        term = self.end is not None and self.t >= self.end
        return self.frame(self.t), float((self.t * 37) % 11), term, False, {}


def _replay_shell(env, fn):
    from types import SimpleNamespace
    run = object.__new__(ablation.Run)
    run.env, run.env_steps = env, 0
    nm = SimpleNamespace(training_steps=0,
                         initial_inference=lambda stack: (fn(stack), 0.0))
    run.searcher = SimpleNamespace(network_manager=nm)
    return run


@pytest.mark.parametrize("trial", range(24))
def test_a_student_replaying_the_retokenized_actions_sees_the_same_records(
        trial, monkeypatch):
    """r1 A-L3 (the auditor's check, kept): a master plays tokens through
    the real play_episode with the hook on; retokenize; a student replaying
    those actions on the same env under the student vocabulary records
    exactly the same stacks, slots and raw reward tuples -- life losses
    inside a token, cap truncation and the first record included."""
    master = tuple((a, f) for a in range(3) for f in (4, 8, 16, 32))
    for k, v in (("REWARD_GROWTH", True), ("USE_SEARCH_ACTING", False),
                 ("PAPER_TARGETS", True), ("BIRTH_SEARCH", True),
                 ("PRIOR_ACT_STAGED_TEMP", False), ("ACTING_TEMPERATURE", 1.0),
                 ("PRIOR_ACT_TEMP_SCALE", 1.0), ("TOKEN_LONG_BIAS", 0.0),
                 ("TOKEN_VOCAB", master)):
        monkeypatch.setattr(muzero, k, v)
    for k in ("ACTIONS", "EPISODE_CAP", "INNER_FRAMES"):
        monkeypatch.setattr(ablation, k, getattr(ablation, k))
    rng = np.random.default_rng(trial)
    end = int(rng.integers(5, 200)) if trial % 3 else None
    drop = tuple(t for t in master if t[1] != 4 and rng.random() < 0.5)
    student = tuple(t for t in master if t not in drop)
    ablation.ACTIONS, ablation.EPISODE_CAP = len(master), int(
        rng.integers(20, 300))
    p = rng.dirichlet(np.ones(len(master)))
    ablation.INNER_FRAMES = []
    np.random.seed(trial)
    traj, _ret, steps, term = ablation.Run.play_episode(
        _replay_shell(_ReplayEnv(end), lambda s: p.copy()))
    inner, ablation.INNER_FRAMES = ablation.INNER_FRAMES, None
    distill.check_life(traj, inner)
    out = distill.retokenize(traj, inner, master, student)
    muzero.TOKEN_VOCAB, ablation.ACTIONS = student, len(student)
    ablation.EPISODE_CAP = steps          # the student stops where it did
    acts, k = [r[1] for r in out[:-1]], [0]

    def fn(stack):
        q = np.zeros(len(student))
        q[acts[k[0]]] = 1.0
        k[0] += 1
        return q

    straj, _r, ssteps, sterm = ablation.Run.play_episode(
        _replay_shell(_ReplayEnv(end), fn))
    assert (len(straj), ssteps, sterm) == (len(out), steps, term)
    for a, b in zip(straj, out):
        assert np.array_equal(a[0], b[0]) and a[1] == b[1]
        assert tuple(a[2]) == tuple(b[2])


def test_the_set_file_round_trips_and_refuses_damage(tmp_path):
    lives = [{"traj": [(np.zeros((2, 2), np.uint8), 1, (), np.ones(3) / 3,
                        0.5)], "terminated": True}]
    p = tmp_path / "set.pkl"
    distill.write_set(p, _header(), lives)
    h, got = distill.read_set(p)
    assert h["lives"] == 1 and distill.read_header(p) == h
    assert got[0]["terminated"] and got[0]["traj"][0][1:3] == (1, ())
    data = p.read_bytes()
    (tmp_path / "cut.pkl").write_bytes(data[:len(data) - 20])
    with pytest.raises(SystemExit, match="truncated|not a distillation"):
        distill.read_set(tmp_path / "cut.pkl")
    (tmp_path / "junk.pkl").write_bytes(b"junk" + data)
    with pytest.raises(SystemExit, match="not a distillation set"):
        distill.read_header(tmp_path / "junk.pkl")
    old = tmp_path / "old.pkl"
    with open(old, "wb") as f:
        f.write(distill.SET_MAGIC)
        pickle.dump({"format": 0, "lives": 0}, f)
    with pytest.raises(SystemExit, match="format"):
        distill.read_set(old)
    assert not list(tmp_path.glob("*.tmp"))


def test_prep_refuses_a_master_that_is_not_a_final_acting_snapshot(
        tmp_path, monkeypatch):
    blob = {"arm": TK_ARM, "spec": dict(ablation.ARMS[TK_ARM]),
            "training_steps": 4000, "acting_states": None}
    p = tmp_path / "m.pth"
    torch.save(blob, p)
    with pytest.raises(SystemExit, match="acting_states"):
        distill.prep(p, 2, tmp_path / "out")
    blob["acting_states"] = {"representation": {}}
    torch.save(blob, p)
    with pytest.raises(SystemExit, match="FINAL snapshot"):
        distill.prep(p, 3, tmp_path / "out")
    blob["spec"] = {**blob["spec"], "TOKEN_VOCAB": MASTER_VOCAB[:-1]}
    torch.save(blob, p)
    with pytest.raises(SystemExit, match="not its arm's"):
        distill.prep(p, 2, tmp_path / "out")
    blob["arm"] = "no-such-arm"
    torch.save(blob, p)
    with pytest.raises(SystemExit, match="unknown"):
        distill.prep(p, 2, tmp_path / "out")


# --------------------------------------------------------------------------
# real Ms. Pac-Man (CPU): a whole prep, then the distilled student
# --------------------------------------------------------------------------
MASTER_ARM = "test-distill-master"


def _pin(monkeypatch):
    for k in set(ablation.ARMS[DS_ARM]) | {"TEMPERATURE_TOTAL_STEPS",
                                          "OFFPOLICY_DECAY_STEPS",
                                          "ACT_SEARCH_SIMS",
                                          "TRAINING_BATCHES"}:
        if hasattr(muzero, k):
            monkeypatch.setattr(muzero, k, getattr(muzero, k))
    for k in ("ACTIONS", "EPISODE_CAP", "ENV_NAME", "FRAMESKIP", "DEVICE",
              "RESULTS", "TAG", "INNER_FRAMES"):
        monkeypatch.setattr(ablation, k, getattr(ablation, k))
    ablation.DEVICE = torch.device("cpu")
    torch.set_num_threads(2)


@pytest.fixture
def prepped(monkeypatch, tmp_path, restore_rngs, student):  # noqa: F811
    """A master snapshot (random TK-arm nets, a 4-sim search for speed)
    and its prep over one recorded game."""
    _pin(monkeypatch)
    monkeypatch.setitem(ablation.ARMS, MASTER_ARM,
                        {**ablation.ARMS[TK_ARM], "ACT_SEARCH_SIMS": 4})
    ablation.ENV_NAME, ablation.FRAMESKIP, ablation.ACTIONS = ENV_ID, 4, 9
    with ablation.arm_config(MASTER_ARM) as spec, ablation.token_actions():
        torch.manual_seed(3)
        nets = ablation.build_networks(spec)
    blob = {"arm": MASTER_ARM, "seed": 0, "spec": dict(spec),
            "training_steps": 4000, "iteration": 2,
            **{n: nets[n].state_dict() for n in ablation.NET_NAMES},
            "acting_states": {n: nets[n].state_dict()
                              for n in ("representation", "prediction")}}
    master = tmp_path / "weights-iter002.pth"
    torch.save(blob, master)
    out = tmp_path / "prep"
    logs = []
    header = distill.prep(master, 2, out, base_steps=40, log=logs.append)
    return header, out, logs


@pytest.mark.slow
def test_prep_records_retokenizes_and_targets_a_real_master(prepped):
    header, out, logs = prepped
    assert ablation.INNER_FRAMES is None              # the hook is off again
    h, lives = distill.read_set(out / "set.pkl")
    assert h == header and h["lives"] == len(lives) >= 1
    doc = json.loads((out / "vocab.json").read_text())
    assert doc["set_sha256"] == ablation.file_sha256(out / "set.pkl")
    vocab = ablation.validate_student_vocab(doc["vocab"], MASTER_VOCAB)
    assert [list(t) for t in vocab] == h["vocab"]
    assert h["acting_temperature"] == 0.125 and h["long_token_bias"] == 0.0
    assert h["games"] == len(h["game_returns"]) >= 1
    assert h["base_steps"] >= 40 and h["decisions"] > 0
    shares = {tuple(map(int, k.split(","))): v for k, v in h["shares"].items()}
    assert all(shares[tuple(t)] < distill.DROP_SHARE for t in h["dropped"])
    assert all(shares[t] >= distill.DROP_SHARE or t[1] == 4 for t in vocab)
    assert len(vocab) + len(h["dropped"]) == 36
    S = len(vocab)
    base_steps, raw_total = 0, 0.0
    for life in lives:
        traj = life["traj"]
        assert traj[0][2] == ()
        for i, (obs, slot, raw, pol, val) in enumerate(traj):
            assert obs.shape == (12, 84, 84) and obs.dtype == np.uint8
            assert 0 <= slot < S and isinstance(raw, tuple)
            assert pol.shape == (S,) and pol.sum() == pytest.approx(1.0)
            assert np.isfinite(val)
            raw_total += sum(raw)
        base_steps += sum(len(r[2]) for r in traj)
    assert base_steps == h["base_steps"]               # every base step once
    assert h["records"] == sum(len(lf["traj"]) for lf in lives)


@pytest.mark.slow
def test_the_distilled_student_prefills_and_trains_offline(
        prepped, monkeypatch, tmp_path):
    header, out, _ = prepped
    monkeypatch.setitem(ablation.ARMS, DS_ARM, {
        **ablation.ARMS[DS_ARM], "_distill_offline_steps": 4,
        "_distill_snapshot_every": 2})
    ablation.student_arm(DS_ARM, out / "vocab.json")
    results = tmp_path / "mspacman-student-distill-s0-test"
    results.mkdir()
    ablation.RESULTS, ablation.TAG = results, "-test"
    _, lives = distill.read_set(out / "set.pkl")
    with ablation.arm_config(DS_ARM) as spec, \
            ablation._run_length_globals(2, 3), ablation.token_actions():
        muzero.TRAINING_BATCHES = 8
        ablation.student_gate(spec, out / "set.pkl")
        run = ablation.Run(DS_ARM, 0, spec)
        try:
            ablation.distill_prefill(run, lives)
            S = len(muzero.TOKEN_VOCAB)
            assert run._reward_ceiling == 160.0
            assert run._tok_window is None and run._tok_totals is None
            n_rec = 0
            for traj, life in zip(run.replay, lives):
                assert traj[-1]["truncated"] == (not life["terminated"])
                for t, rec in zip(traj, life["traj"]):
                    assert t["born_step"] == 0 and t["action"] == rec[1]
                    assert t["target_policy"].shape == (S,)
                    assert t["raw_reward"] == rec[2]
                    assert t["reward"] == muzero.grown_reward(rec[2], 160.0)
                    assert np.isfinite(t["priority"]) and t["priority"] > 0
                    n_rec += 1
            assert n_rec == sum(len(lf["traj"]) for lf in lives)
            ablation.distill_offline(run, DS_ARM, 0, spec, True)
            assert run.trainer.training_steps == 4
            assert run.searcher.network_manager.training_steps == 4
            assert run._step_window is None
            assert not run.trainer.representation.training   # eval mode
        finally:
            run.env.close()
    snaps = results / f"snapshots-test-{DS_ARM}-seed0"
    row0 = torch.load(snaps / "weights-iter000.pth", weights_only=False)
    assert row0["iteration"] == 0 and row0["training_steps"] == 4
    assert tuple(map(tuple, row0["spec"]["TOKEN_VOCAB"])) == tuple(
        map(tuple, header["vocab"]))
    assert set(row0["acting_states"]) == {"representation", "prediction"}
    curve = (results / f"{results.name}-offline" /
             f"snapshots-test-{DS_ARM}-seed0" / "weights-iter001.pth")
    c1 = torch.load(curve, weights_only=False)
    assert c1["iteration"] == 1 and c1["training_steps"] == 2
    assert c1["acting_states"]
    # the probe files the curve under its own run name
    assert curve.parent.parent.name == results.name + "-offline"
    assert sorted(p.name for p in snaps.iterdir()) == ["weights-iter000.pth"]


@pytest.mark.slow
def test_the_prefill_evicts_the_oldest_lives_and_says_so(
        prepped, monkeypatch, tmp_path):
    header, out, _ = prepped
    ablation.student_arm(DS_ARM, out / "vocab.json")
    _, lives = distill.read_set(out / "set.pkl")
    with ablation.arm_config(DS_ARM) as spec, \
            ablation._run_length_globals(2, 3), ablation.token_actions():
        units = [ablation.replay_units([{"action": r[1]} for r in lf["traj"]])
                 for lf in lives]
        assert len(lives) >= 2
        muzero.MAX_REPLAY_TRANSITIONS = sum(units[1:])   # the first must go
        run = ablation.Run(DS_ARM, 0, spec)
        try:
            kept, gone, gone_units = ablation.distill_prefill(run, lives)
        finally:
            run.env.close()
    assert (kept, gone, gone_units) == (len(lives) - 1, 1, units[0])
    assert len(run.replay) == kept


@pytest.mark.slow
def test_run_one_distills_once_and_resumes_past_it(prepped, monkeypatch,
                                                   tmp_path, capsys):
    """r1 A-L3 (the auditor's smoke, kept): the real run_one, sequential on
    the CPU with the Ray-only gates stubbed -- prefill + offline phase,
    a crash at iteration 3 after the iteration-2 checkpoint, a resume that
    skips the prefill, a resume with another set refused."""
    header, out, _ = prepped
    results = tmp_path / "ds-run"
    results.mkdir()
    ablation.RESULTS, ablation.TAG, ablation.EPISODE_CAP = results, "-t", 60
    monkeypatch.setitem(ablation.ARMS, DS_ARM, {
        **ablation.ARMS[DS_ARM], "ACT_SEARCH_SIMS": 4, "TRAINING_BATCHES": 8,
        "_distill_offline_steps": 4, "_distill_snapshot_every": 2})
    ablation.student_arm(DS_ARM, out / "vocab.json")
    for g in ("token_gate", "reward_growth_gate", "bn_recal_gate",
              "launch_gate", "eager_collect_gate"):
        monkeypatch.setattr(ablation, g, lambda *a, **k: None)
    calls = []
    real_prefill, real_tb = ablation.distill_prefill, ablation.train_block
    monkeypatch.setattr(ablation, "distill_prefill", lambda run, lives: (
        calls.append(len(lives)), real_prefill(run, lives))[1])
    crash = [True]

    def tb(run, n, pipeline=None, init_name=None, consistency_base=None,
           it=None):
        if it == 3 and crash[0]:
            raise RuntimeError("crash at iteration 3")
        return real_tb(run, n, pipeline=pipeline, init_name=init_name,
                       consistency_base=consistency_base, it=it)

    monkeypatch.setattr(ablation, "train_block", tb)
    jsonl = results / "ablation-t.jsonl"
    kw = dict(checkpoint_every=2, use_ray=False, snapshot_weights=True,
              distill_set=str(out / "set.pkl"))
    with pytest.raises(RuntimeError, match="iteration 3"):
        ablation.run_one(DS_ARM, 0, 4, 1, 3, 2, jsonl, **kw)
    ck = ablation.resume_path(DS_ARM, 0)
    st = ablation._read_checkpoint(ck)
    assert st["iteration"] == 2 and st["training_steps"] == 4 + 2 * 3
    assert st["token_vocab"] == header["vocab"]
    assert st["distill_set_sha256"] == ablation.file_sha256(out / "set.pkl")
    assert calls == [header["lives"]]
    crash[0] = False
    ablation.run_one(DS_ARM, 0, 4, 1, 3, 2, jsonl, resume=True, **kw)
    assert calls == [header["lives"]]                  # no second prefill
    rows = [json.loads(l) for l in open(jsonl)]
    assert [r["iteration"] for r in rows] == [2, 4]
    assert all(r["reward_ceiling"] == 160.0 for r in rows)
    assert rows[0]["train_steps"] == 4 + 2 * 3
    snaps = results / f"snapshots-t-{DS_ARM}-seed0"
    assert sorted(x.name for x in snaps.iterdir()) == [
        "weights-iter000.pth", "weights-iter002.pth", "weights-iter004.pth"]


@pytest.mark.slow
def test_a_resume_under_another_set_is_refused(prepped, monkeypatch,
                                               tmp_path):
    header, out, _ = prepped
    results = tmp_path / "ds-run2"
    results.mkdir()
    ablation.RESULTS, ablation.TAG, ablation.EPISODE_CAP = results, "-t", 60
    monkeypatch.setitem(ablation.ARMS, DS_ARM, {
        **ablation.ARMS[DS_ARM], "ACT_SEARCH_SIMS": 4, "TRAINING_BATCHES": 8,
        "_distill_offline_steps": 2, "_distill_snapshot_every": 2})
    for g in ("token_gate", "reward_growth_gate", "bn_recal_gate",
              "launch_gate", "eager_collect_gate"):
        monkeypatch.setattr(ablation, g, lambda *a, **k: None)
    ablation.student_arm(DS_ARM, out / "vocab.json")
    jsonl = results / "ablation-t.jsonl"
    real_tb = ablation.train_block

    def tb(run, n, pipeline=None, init_name=None, consistency_base=None,
           it=None):
        if it == 3:
            raise RuntimeError("crash at iteration 3")
        return real_tb(run, n, pipeline=pipeline, init_name=init_name,
                       consistency_base=consistency_base, it=it)

    monkeypatch.setattr(ablation, "train_block", tb)
    kw = dict(checkpoint_every=2, use_ray=False, snapshot_weights=True)
    with pytest.raises(RuntimeError):
        ablation.run_one(DS_ARM, 0, 4, 1, 3, 2, jsonl,
                         distill_set=str(out / "set.pkl"), **kw)
    # a different set (one life fewer), with a vocab.json written for it
    _, lives = distill.read_set(out / "set.pkl")
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other = other_dir / "set.pkl"
    distill.write_set(other, {**header, "lives": len(lives) - 1}, lives[1:]
                      if len(lives) > 1 else lives + lives)
    doc = json.loads((out / "vocab.json").read_text())
    doc["set_sha256"] = ablation.file_sha256(other)
    (other_dir / "vocab.json").write_text(json.dumps(doc))
    monkeypatch.setitem(ablation.ARMS, DS_ARM, {
        **ablation.ARMS[DS_ARM]})
    ablation.student_arm(DS_ARM, other_dir / "vocab.json")
    with pytest.raises(ValueError, match="resume mismatch.*distill_set_sha256"):
        ablation.run_one(DS_ARM, 0, 4, 1, 3, 2, jsonl, resume=True,
                         distill_set=str(other), **kw)
