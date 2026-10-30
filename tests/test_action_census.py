"""E0, the action-equivalence census (CURRICULUM "Queued experiment
(pre-registered) -- E0", Kai 2026-10-30). Pins: the census's equivalence
(identical frames over the common window; strict = the same emulator state
at its end), the breadth-first sequence counts against brute force, every
branch restoring the saved state, the reads (1)-(6) and their verdict, the
hook's restore, the snapshot refusals, the main-line check against the
exact eval, and -- in a subprocess, on the real game -- that a censused
game is the exact eval's game."""
import itertools
import json
import os
import pathlib
import subprocess
import sys

import gymnasium as gym
import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROBES = ROOT / "benchmarks" / "probes"
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(PROBES))
import ablation  # noqa: E402
import probe_action_census as pc  # noqa: E402

ENV_ID = "MsPacmanNoFrameskip-v4"


def test_importing_the_census_changes_nothing_process_wide():
    """The script-mode settings stay out of a module import (the exact
    eval's import would set ablation's env globals)."""
    assert "probe_exhaustive_eval" not in pc.__dict__
    code = ("import os, sys; sys.path[:0] = [{p!r}, {e!r}, {r!r}]; "
            "import ablation; before = (ablation.ENV_NAME, ablation.ACTIONS); "
            "import probe_action_census; "
            "assert (ablation.ENV_NAME, ablation.ACTIONS) == before; "
            "assert 'probe_exhaustive_eval' not in sys.modules; "
            "assert os.environ.get('PROBE_ACTING_STATES') is None; "
            "print('ok')").format(p=str(PROBES), e=str(ROOT / "experiments"),
                                  r=str(ROOT))
    env = {k: v for k, v in os.environ.items()
           if k not in ("PROBE_ACTING_STATES", "PROBE_ENV")}
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=env,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-2000:]


# --------------------------------------------------------------------------
# a scripted emulator: actions move a position; action 4 bumps a counter
# the screen shows only from frame `reveal` on (hidden state)
# --------------------------------------------------------------------------
class _State:
    def __init__(self, t):
        self.t = t

    def serialize(self):
        return repr(self.t).encode()


class _ALE:
    EFFECT = {0: 0, 1: 0, 2: 1, 3: -1, 4: 0, 5: 1, 6: 1, 7: 1, 8: 1}

    def __init__(self, reveal=10 ** 9):
        self.pos, self.hidden, self.frame, self.reveal = 0, 0, 0, reveal
        self.acts = 0

    def act(self, a):
        self.pos += self.EFFECT[a]
        self.hidden += int(a == 4)
        self.frame += 1
        self.acts += 1

    def getScreenRGB(self):
        shown = self.hidden if self.frame >= self.reveal else 0
        return np.array([self.pos, shown], dtype=np.int64)

    def cloneState(self):
        return _State((self.pos, self.hidden, self.frame))

    def restoreState(self, s):
        self.pos, self.hidden, self.frame = s.t


ACTIONS = list(range(9))
SMALL_GRID = tuple((a, f) for a in range(9) for f in (4, 8))


def test_tokens_with_the_same_frames_share_a_class_and_hidden_state_is_strict():
    ale = _ALE()
    start = ale.cloneState()
    frames, strict = pc.token_census(ale, ACTIONS, start, grid=SMALL_GRID,
                                     window=8 + 3)
    cls = dict(zip(SMALL_GRID, frames))
    # NOOP, action 1 and the hidden action 4 never move: one frame class
    still = {cls[(a, f)] for a in (0, 1, 4) for f in (4, 8)}
    assert len(still) == 1
    # the right-movers 2, 5-8 share a class per duration; 3 has its own
    for f in (4, 8):
        assert len({cls[(a, f)] for a in (2, 5, 6, 7, 8)}) == 1
        assert cls[(3, f)] not in {cls[(2, f)]} | still
    assert cls[(2, 4)] != cls[(2, 8)]          # they stop at different places
    assert len(set(frames)) == 5
    # strict: action 4's counter separates it (4 and 8 bumps), the rest as
    # the frames
    s = dict(zip(SMALL_GRID, strict))
    assert s[(4, 4)] != s[(0, 4)] and s[(4, 8)] != s[(4, 4)]
    assert s[(0, 4)] == s[(1, 8)]
    assert len(set(strict)) == 7
    assert ale.cloneState().t == start.t         # restored


def test_shown_hidden_state_separates_the_frames_too():
    ale = _ALE(reveal=0)
    frames, strict = pc.token_census(ale, ACTIONS, ale.cloneState(),
                                     grid=SMALL_GRID, window=11)
    assert len(set(frames)) == len(set(strict)) == 7


def _brute(ale, start, depth, base, window):
    """Every sequence played on its own: distinct frame windows and end
    states per length."""
    out_f, out_s = [], []
    for length in range(1, depth + 1):
        fk, sk = set(), set()
        for seq in itertools.product(ACTIONS, repeat=length):
            inputs = [a for a in seq for _ in range(base)]
            f, s = pc.play_window(ale, start, inputs, window, ACTIONS[0])
            fk.add(f)
            sk.add(s)
        out_f.append(len(fk))
        out_s.append(len(sk))
    ale.restoreState(start)
    return out_f, out_s


@pytest.mark.parametrize("reveal", [10 ** 9, 3, 0])
def test_the_breadth_first_counts_equal_brute_force(reveal):
    ale = _ALE(reveal=reveal)
    start = ale.cloneState()
    got = pc.sequence_census(ale, ACTIONS, start, depth=3, base=2, window=9)
    assert got == _brute(ale, start, 3, 2, 9)
    assert ale.cloneState().t == start.t


def test_same_end_state_by_different_paths_is_one_strict_class_two_frame_classes():
    ale = _ALE()
    kf, ks = pc.sequence_census(ale, ACTIONS, ale.cloneState(), depth=2,
                                base=1, window=4)
    # (2, 3) and (3, 2) end where (0, 0) does, through different frames
    assert ks[1] < kf[1]
    # depth 1: still {0,1,4} / right / left = 3 frame classes; 4 strict
    assert kf[0] == 3 and ks[0] == 4


def test_merged_nodes_are_expanded_once():
    ale = _ALE()
    pc.sequence_census(ale, ACTIONS, ale.cloneState(), depth=3, base=1,
                       window=3)
    merged = ale.acts
    ale2 = _ALE()
    _brute(ale2, ale2.cloneState(), 3, 1, 3)
    assert merged < ale2.acts / 5


# --------------------------------------------------------------------------
# the records and the reads
# --------------------------------------------------------------------------
def _part(classes_of_action):
    """A 36-token partition from a class per action (all its durations
    alike)."""
    return [classes_of_action[a] for a, _ in pc.GRID]


def test_a_decision_record():
    part = _part([0, 0, 1, 2, 0, 1, 1, 1, 1])
    strict = list(range(36))
    vocab = [(0, 4), (1, 4), (2, 4), (3, 4)]
    vg = [pc.GRID.index(t) for t in vocab]
    prior = [0.5, 0.5, 0.0, 0.0]               # NOOP and UP: one class
    rec = pc.decision_record(2, 7, 1, vg, part, strict, prior,
                             seq=([3, 9, 27], [4, 12, 30]))
    assert rec["classes_vocab"] == 3 and rec["classes_grid"] == 3
    assert rec["strict_vocab"] == 4 and rec["strict_grid"] == 36
    assert rec["by_duration"] == {"4": 3, "8": 3, "16": 3, "32": 3}
    assert rec["durations_per_action"] == [1] * 9
    assert rec["chosen_class_size"] == 12       # actions 0, 1, 4 x 4
    assert rec["h_tokens"] == pytest.approx(np.log(2), abs=1e-5)
    assert rec["h_classes"] == pytest.approx(0.0, abs=1e-9)
    assert rec["seq"] == {"frames": [3, 9, 27], "strict": [4, 12, 30]}
    assert rec["partition"] == part


def _recs_flat(classes):
    """Records whose first k grid tokens are k classes (the rest join the
    last): the vocabulary is the first 18 grid tokens."""
    out = []
    for i, k in enumerate(classes):
        part = [min(j, k - 1) for j in range(36)]
        out.append(pc.decision_record(1, i, 0, list(range(18)), part, part,
                                      [1 / 18] * 18))
    return out


def _recs(classes, seq=True):
    """Records of k <= 9 classes, one per action (all its durations
    alike); the vocabulary is each action at 4 and 16 frames."""
    out = []
    for i, k in enumerate(classes):
        part = _part([min(a, k - 1) for a in range(9)])
        out.append(pc.decision_record(
            1, i, 0, list(range(0, 36, 2)), part, part, [1 / 18] * 18,
            seq=([3, 9, 27], [3, 9, 27]) if seq and i % 2 == 0 else None))
    return out


@pytest.mark.parametrize("classes,verdict", [
    ([9, 9], "E1"),                          # 9 of 18 = 0.5
    ([14, 14], "KAI'S CALL (does not pay)"),  # 0.78
    ([12, 12], "KAI'S CALL (in between)")])  # 0.67
def test_the_verdict_follows_the_primary_ratio(classes, verdict):
    s = pc.summarize(_recs_flat(classes), 18)
    assert s["verdict"] == verdict
    assert s["ratio_vocab"] == pytest.approx(classes[0] / 18, abs=1e-4)


def test_the_summary_reads():
    recs = _recs([1, 3, 9])
    s = pc.summarize(recs, 18)
    assert s["decisions"] == 3 and s["classes_grid"]["median"] == 3
    assert s["free_vocab"] == pytest.approx(1 / 3, abs=1e-4)
    assert s["at_most_two_vocab"] == pytest.approx(1 / 3, abs=1e-4)
    # every action's durations share its class: duration never matters here
    assert set(s["same_class_by_durations"].values()) == {1.0}
    assert s["durations_per_action"] == 1.0
    assert s["seq_decisions"] == 2
    assert s["seq_frames_mean"] == [3, 9, 27] and s["seq_growth"] == [3, 3]
    assert s["seq_nominal"] == [9, 81, 729]


def test_the_summary_is_the_same_from_the_decisions_file():
    recs = _recs([2, 5, 9, 4])
    back = [json.loads(json.dumps(r)) for r in recs]
    assert pc.summarize(back, 18) == pc.summarize(recs, 18)


# --------------------------------------------------------------------------
# the hook, the refusals, the main-line check
# --------------------------------------------------------------------------
def test_the_hook_restores_play_token(monkeypatch):
    orig = ablation.play_token
    census = pc.Census(None, [(0, 4)], None)
    with pytest.raises(RuntimeError):
        with census.hooked():
            assert ablation.play_token is not orig
            raise RuntimeError
    assert ablation.play_token is orig


def test_the_hook_censuses_then_plays_the_agents_token(monkeypatch):
    seen = []
    monkeypatch.setattr(ablation, "play_token",
                        lambda env, token, stack, inner=None:
                        seen.append(("play", token)) or "played")
    census = pc.Census(None, [(0, 4)], None)
    monkeypatch.setattr(census, "decision",
                        lambda env, token, stack: seen.append(("census",
                                                               token)))
    monkeypatch.setattr(pc, "MAX_DECISIONS", 0)
    with census.hooked():
        assert ablation.play_token("env", 3, "stack") == "played"
    assert seen == [("census", 3), ("play", 3)]


def test_smoke_stops_the_census_not_the_game(monkeypatch):
    plays = []
    monkeypatch.setattr(ablation, "play_token",
                        lambda env, token, stack, inner=None:
                        plays.append(token))
    census = pc.Census(None, [(0, 4)], None)

    def decide(env, token, stack):
        census.index += 1
    monkeypatch.setattr(census, "decision", decide)
    monkeypatch.setattr(pc, "MAX_DECISIONS", 2)
    with census.hooked():
        for t in range(5):
            ablation.play_token(None, t, None)
    assert census.index == 2 and plays == [0, 1, 2, 3, 4]


def test_the_snapshot_refusals():
    with pytest.raises(SystemExit, match="PROBE_ENV"):
        pc.check_snapshot(((0, 4),), "Pong-v4")
    with pytest.raises(SystemExit, match="TOKEN_VOCAB"):
        pc.check_snapshot(None, ENV_ID)
    with pytest.raises(SystemExit, match="outside the 36-token grid"):
        pc.check_snapshot(((0, 4), (2, 12)), ENV_ID)
    pc.check_snapshot(((0, 4), (8, 32)), ENV_ID)


def test_the_exact_eval_row_is_found(tmp_path, monkeypatch):
    run = tmp_path / "benchmarks" / "results" / "run-t" / "snaps"
    run.mkdir(parents=True)
    snap = run / "weights-iter042.pth"
    probes = tmp_path / "benchmarks" / "probes"
    probes.mkdir()
    rows = [{"iteration": 42, "exact_mean": 5.0, "max_noop": 2,
             "env": ENV_ID},                           # a smoke row
            {"iteration": 42, "exact_mean": 6.0, "env": "Pong-v4"},
            {"iteration": 40, "exact_mean": 7.0, "env": ENV_ID},
            {"iteration": 42, "exact_mean": 1699.0, "env": ENV_ID}]
    (probes / "probe_exhaustive_eval-run-t-acting.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")
    monkeypatch.setattr(pc, "HERE", tmp_path / "elsewhere")
    assert pc.exact_eval_mean(snap) == 1699.0
    assert pc.exact_eval_mean(run / "weights-iter041.pth") is None


# --------------------------------------------------------------------------
# the real emulator
# --------------------------------------------------------------------------
@pytest.fixture
def mspacman():
    env = gym.make(ENV_ID, frameskip=1)
    env.reset(seed=0)
    for _ in range(300):                       # past the start jingle
        env.step(0)
    yield env
    env.close()


def test_the_real_census_restores_and_repeats(mspacman):
    ale = mspacman.unwrapped.ale
    actions = list(mspacman.unwrapped._action_set)
    import ale_py
    assert actions[0] == ale_py.Action.NOOP        # the branches' no input
    start = ale.cloneState()
    before = start.serialize()
    grid = ((0, 4), (0, 8), (3, 8), (3, 8), (2, 4))
    a = pc.token_census(ale, actions, start, grid=grid, window=20)
    assert ale.cloneState().serialize() == before
    b = pc.token_census(ale, actions, start, grid=grid, window=20)
    assert a == b                                  # deterministic
    frames = a[0]
    assert frames[0] == frames[1]                  # NOOP x4 + NOOP = NOOP x8
    assert frames[2] == frames[3]                  # the same token twice


def test_the_game_continues_identically_after_a_census(mspacman):
    ale = mspacman.unwrapped.ale
    actions = list(mspacman.unwrapped._action_set)
    start = ale.cloneState()
    plan = [3] * 12 + [1] * 12

    def obs_after():
        return [mspacman.step(a)[0].tobytes() for a in plan]

    clean = obs_after()
    ale.restoreState(start)
    pc.token_census(ale, actions, start, grid=pc.GRID[:8], window=30)
    pc.sequence_census(ale, actions, start, depth=1)
    assert obs_after() == clean


def test_the_four_noop_tokens_are_always_one_class(mspacman):
    ale = mspacman.unwrapped.ale
    actions = list(mspacman.unwrapped._action_set)
    for _ in range(3):
        start = ale.cloneState()
        frames, _ = pc.token_census(ale, actions, start)
        assert len({frames[pc.GRID.index((0, f))]
                    for f in pc.DURATIONS}) == 1
        for _ in range(40):
            mspacman.step(3)


# --------------------------------------------------------------------------
# end to end: a censused game is the exact eval's game
# --------------------------------------------------------------------------
_E2E = r"""
import json, pathlib, sys
sys.path[:0] = [{probes!r}, {exp!r}, {root!r}]
import torch
import ablation, muzero
import probe_exhaustive_eval as pe
import probe_action_census as pc
out = pathlib.Path({out!r})
vocab = tuple(tuple(t) for t in {vocab!r})
snap = out / "run-c" / "snaps" / "weights-iter007.pth"
snap.parent.mkdir(parents=True)
with ablation.arm_config(pe.ARM), ablation._run_length_globals(90, 2000), \
        pe.snapshot_tokens(vocab):
    torch.manual_seed(5)
    run = ablation.Run(pe.ARM, 0, ablation.ARMS[pe.ARM])
    nets = {{n: getattr(run.trainer, n).state_dict()
            for n in ablation.NET_NAMES}}
    torch.save({{**nets, "training_steps": 0,
                "spec": {{"TOKEN_VOCAB": [list(t) for t in vocab]}},
                "acting_states": {{"representation": nets["representation"],
                                  "prediction": nets["prediction"]}}}}, snap)
pc.HERE = out
pc.GAMES, pc.MAX_DECISIONS, pc.SEQ_EVERY, pc.SMOKE = 1, 2, 1, True
pc.main([str(snap)])
with ablation.arm_config(pe.ARM), ablation._run_length_globals(90, 2000), \
        pe.snapshot_tokens(vocab):
    run = ablation.Run(pe.ARM, 0, ablation.ARMS[pe.ARM])
    d = pe.load(run, snap)
    pe.apply_acting_states(run, d)
    plain = pe.play(run, 1)
print("PLAIN", json.dumps(list(plain)))
"""


@pytest.mark.slow
def test_a_censused_game_is_the_exact_evals_game(tmp_path):
    vocab = [[0, 4], [1, 4], [2, 4], [3, 4], [3, 16], [3, 32], [4, 4],
             [4, 32], [5, 4], [5, 16], [5, 32], [6, 4], [7, 4], [7, 32],
             [8, 4], [8, 8], [8, 16], [8, 32]]          # T2's 18
    env = {**os.environ, "PROBE_ENV": ENV_ID, "PROBE_ACTING_STATES": "1",
           "PROBE_THREADS": "2", "CUDA_VISIBLE_DEVICES": "-1"}
    code = _E2E.format(probes=str(PROBES), exp=str(ROOT / "experiments"),
                       root=str(ROOT), out=str(tmp_path), vocab=vocab)
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=env,
                       capture_output=True, text=True, timeout=1500)
    assert r.returncode == 0, r.stderr[-3000:]
    plain = json.loads(next(l for l in r.stdout.splitlines()
                            if l.startswith("PLAIN "))[6:])
    summ = json.loads((tmp_path / "probe_action_census-run-c-smoke.jsonl")
                      .read_text().splitlines()[-1])
    assert summ["exact_mean"] == pytest.approx(plain[0])   # the same game
    assert summ["main_line_ok"] is None                    # a smoke run
    decs = [json.loads(l) for l in (tmp_path / "probe_action_census-run-c-"
                                    "smoke-decisions.jsonl")
            .read_text().splitlines()]
    assert len(decs) == 2 and all(len(d["partition"]) == 36 for d in decs)
    assert all("seq" in d for d in decs)                   # SEQ_EVERY 1
    for d in decs:
        assert len({d["partition"][pc.GRID.index((0, f))]
                    for f in pc.DURATIONS}) == 1
        k = d["seq"]["frames"]
        assert 1 <= k[0] <= 9 and k[0] <= k[1] <= 81 and k[1] <= k[2] <= 729
        assert d["classes_vocab"] <= 18 and d["classes_grid"] <= 36
