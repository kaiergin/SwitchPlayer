"""E0, the action-equivalence census (CURRICULUM "Queued experiment
(pre-registered) -- E0", Kai 2026-10-30). Pins: the census's equivalence
(identical frames over the common window; strict = identical frames AND
the same emulator state at its end), the windows, the breadth-first
sequence counts against brute force, every branch restoring the saved
state, the reads (1)-(6) and (1b) and their verdict, one decision end to
end on a scripted emulator, the hook's restore, the refusals, the validity
checks (A3), and -- in a subprocess, on the real game -- that a censused
game is the exact eval's game."""
import io
import itertools
import json
import os
import pathlib
import subprocess
import sys
from types import SimpleNamespace

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
            "import torch; threads = torch.get_num_threads(); "
            "import ablation; before = (ablation.ENV_NAME, ablation.ACTIONS); "
            "import probe_action_census; "
            "assert (ablation.ENV_NAME, ablation.ACTIONS) == before; "
            "assert torch.get_num_threads() == threads; "
            "assert 'probe_exhaustive_eval' not in sys.modules; "
            "assert os.environ.get('PROBE_ACTING_STATES') is None; "
            "assert os.environ.get('PROBE_ENV') is None; "
            "assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'; "
            "print('ok')").format(p=str(PROBES), e=str(ROOT / "experiments"),
                                  r=str(ROOT))
    env = {k: v for k, v in os.environ.items()
           if k not in ("PROBE_ACTING_STATES", "PROBE_ENV")}
    env["CUDA_VISIBLE_DEVICES"] = "-1"        # never the live run's GPU
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=env,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-2000:]


def test_the_pre_registered_windows():
    assert pc.WINDOW_TOKENS == 32 + 60 and pc.WINDOW_SEQ == 12 + 60
    assert len(pc.GRID) == 36 and pc.GRID[:4] == ((0, 4), (0, 8), (0, 16),
                                                  (0, 32))


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


class _LeakyALE(_ALE):
    """The screen also shows how many frames were ever emulated -- a
    branch-order dependence restoreState does not undo."""

    def getScreenRGB(self):
        return np.array([self.pos, self.acts], dtype=np.int64)


class _WrapALE(_ALE):
    """Positions wrap at 4: RIGHT for 4 frames ends in NOOP's state through
    different frames."""

    def act(self, a):
        super().act(a)
        self.pos %= 4


ACTIONS = list(range(9))
SMALL_GRID = tuple((a, f) for a in range(9) for f in (4, 8))


def test_strict_tokens_carry_the_frames():
    """RIGHT x4 and NOOP x4 end in one emulator state through different
    frames: the end state alone would merge them (audit r2b LOW-3)."""
    ale = _WrapALE()
    frames, strict, _ = pc.token_census(ale, ACTIONS, ale.cloneState(),
                                        grid=((0, 4), (2, 4)), window=6)
    assert frames[0] != frames[1] and strict[0] != strict[1]


def test_tokens_with_the_same_frames_share_a_class_and_hidden_state_is_strict():
    ale = _ALE()
    start = ale.cloneState()
    frames, strict, digests = pc.token_census(ale, ACTIONS, start,
                                              grid=SMALL_GRID, window=8 + 3)
    assert ale.cloneState().t == start.t         # restored
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
    # strict refines frames: action 4's counter separates it (4 and 8 bumps)
    s = dict(zip(SMALL_GRID, strict))
    assert s[(4, 4)] != s[(0, 4)] and s[(4, 8)] != s[(4, 4)]
    assert s[(0, 4)] == s[(1, 8)]
    assert len(set(strict)) == 7
    assert all(frames[i] == frames[j] for i in range(18) for j in range(18)
               if strict[i] == strict[j])
    assert len(digests) == 18 and digests[0] == digests[1]


def test_shown_hidden_state_separates_the_frames_too():
    ale = _ALE(reveal=0)
    frames, strict, _ = pc.token_census(ale, ACTIONS, ale.cloneState(),
                                        grid=SMALL_GRID, window=11)
    assert len(set(frames)) == len(set(strict)) == 7


def _brute(ale, start, depth, base, window):
    """Every sequence played on its own: distinct frame windows and
    (frame window, end state) pairs per length."""
    out_f, out_s = [], []
    for length in range(1, depth + 1):
        fk, sk = set(), set()
        for seq in itertools.product(ACTIONS, repeat=length):
            inputs = [a for a in seq for _ in range(base)]
            f, s = pc.play_window(ale, start, inputs, window, ACTIONS[0])
            fk.add(f)
            sk.add((f, s))
        out_f.append(len(fk))
        out_s.append(len(sk))
    ale.restoreState(start)
    return out_f, out_s


# reveal 9 / 10: the hidden counter shows on the window's last frame / just
# after it -- a wrong NOOP tail length changes the counts (audit r1b L1)
@pytest.mark.parametrize("reveal", [10 ** 9, 3, 0, 9, 10])
def test_the_breadth_first_counts_equal_brute_force(reveal):
    ale = _ALE(reveal=reveal)
    start = ale.cloneState()
    got = pc.sequence_census(ale, ACTIONS, start, depth=3, base=2, window=9)
    assert ale.cloneState().t == start.t         # restored, before _brute
    assert got == _brute(ale, start, 3, 2, 9)


def test_one_end_state_by_different_paths_is_two_frame_classes():
    """(2, 3) and (0, 0) end in the same emulator state through different
    frames: the end state alone would merge them, so strict keys carry the
    frames too (audit r1a L3)."""
    ale = _ALE()
    s = ale.cloneState()
    f1, e1 = pc.play_window(ale, s, [2, 3], 4, 0)
    f2, e2 = pc.play_window(ale, s, [0, 0], 4, 0)
    assert e1 == e2 and f1 != f2
    kf, ks = pc.sequence_census(ale, ACTIONS, s, depth=2, base=1, window=4)
    # depth 1: still / right / left; with the hidden counter: 4 strict.
    # depth 2: the 9 (e1, e2) motion pairs; x the counter's values: 15
    assert kf == [3, 9] and ks == [4, 15]


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
    # mass on (0, 4) and (2, 4): two classes (a prior read through the
    # wrong grid entries would find one -- audit r1b M3)
    prior = [0.5, 0.0, 0.5, 0.0]
    rec = pc.decision_record(2, 7, 2, vg, part, strict, prior,
                             seq=([3, 9, 27], [4, 12, 30]))
    assert rec["classes_vocab"] == 3 and rec["classes_grid"] == 3
    assert rec["strict_vocab"] == 4 and rec["strict_grid"] == 36
    assert rec["by_duration"] == {"4": 3, "8": 3, "16": 3, "32": 3}
    assert rec["durations_per_action"] == [1] * 9
    assert rec["chosen_class_size"] == 20       # (2, 4): actions 2, 5-8 x 4
    assert rec["h_tokens"] == pytest.approx(np.log(2), abs=1e-5)
    assert rec["h_classes"] == pytest.approx(np.log(2), abs=1e-5)
    assert rec["seq"] == {"frames": [3, 9, 27], "strict": [4, 12, 30]}
    assert rec["partition"] == part
    # the same mass inside one class: no class uncertainty
    rec = pc.decision_record(2, 7, 1, vg, part, strict, [0.5, 0.5, 0, 0])
    assert rec["h_classes"] == pytest.approx(0.0, abs=1e-9)
    assert rec["chosen_class_size"] == 12       # (1, 4): actions 0, 1, 4 x 4


def test_the_vocabulary_can_miss_a_grid_class():
    """(1) counts the agent's tokens' classes, not the grid's (audit r1b
    M2)."""
    part = _part([0, 0, 1, 2, 0, 1, 1, 1, 1])
    vg = [pc.GRID.index(t) for t in [(0, 4), (1, 4), (2, 4), (2, 8)]]
    rec = pc.decision_record(1, 0, 0, vg, part, part, [1, 0, 0, 0])
    assert rec["classes_vocab"] == 2 and rec["classes_grid"] == 3


def test_the_duration_read_on_a_scripted_game():
    """The full grid on the scripted emulator: the still actions 0, 1, 4
    are one class at every duration, the movers four (audit r1b L2); the
    summary's duration reads leave NOOP out (audit r1a L2)."""
    ale = _ALE()
    frames, strict, _ = pc.token_census(ale, ACTIONS, ale.cloneState(),
                                        window=32 + 3)
    rec = pc.decision_record(1, 0, 0, list(range(36)), frames, strict,
                             [1.0] + [0.0] * 35)
    assert rec["by_duration"] == {"4": 3, "8": 3, "16": 3, "32": 3}
    assert rec["durations_per_action"] == [1, 1, 4, 4, 1, 4, 4, 4, 4]
    s = pc.summarize([rec], 36)
    # actions 1-8: the still 1 and 4 keep a class across durations
    assert set(s["same_class_by_durations"].values()) == {0.25}
    assert s["durations_per_action"] == pytest.approx(26 / 8)
    assert s["classes_grid"]["mean"] == 1 + 4 + 4


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
    ([13, 14], "KAI'S CALL (in between)"),   # 0.75 exactly
    ([14, 14], "KAI'S CALL (does not pay)"),  # 0.78
    ([12, 12], "KAI'S CALL (in between)")])  # 0.67
def test_the_verdict_follows_the_primary_ratio(classes, verdict):
    s = pc.summarize(_recs_flat(classes), 18)
    assert s["verdict"] == s["verdict_1"] == verdict
    assert s["ratio_vocab"] == pytest.approx(np.mean(classes) / 18, abs=1e-4)


def test_the_verdict_rule_on_the_unrounded_ratio():
    assert pc.verdict(0.50004) == "KAI'S CALL (in between)"
    assert pc.verdict(0.5) == "E1" and pc.verdict(0.75) != pc.verdict(0.7501)
    # summarize applies it before any rounding: 9 / 17.9999 = 0.500003
    s = pc.summarize(_recs_flat([9, 9]), 17.9999)
    assert s["ratio_vocab"] == 0.5
    assert s["verdict_1"] == "KAI'S CALL (in between)"


def test_the_vocabulary_ratio_is_not_the_grids():
    s = pc.summarize(_recs_flat([20, 20]), 18)
    assert s["classes_vocab"]["mean"] == 18 and s["classes_grid"]["mean"] == 20
    assert s["ratio_vocab"] == 1.0 and s["ratio_grid"] == pytest.approx(20 / 36,
                                                                        abs=1e-4)


def test_read_1b_over_the_decisions_where_input_matters():
    """A6: free decisions (one class) pull (1) toward E1; (1b) leaves them
    out, and a disagreement is Kai's call."""
    recs = _recs_flat([1, 1, 12, 12])
    s = pc.summarize(recs, 18)
    assert s["live_share"] == 0.5
    assert s["ratio_vocab"] == pytest.approx(26 / 72, abs=1e-4)   # E1 alone
    assert s["ratio_vocab_live"] == pytest.approx(12 / 18, abs=1e-4)
    assert s["verdict_1"] == "E1"
    assert s["verdict_1b"] == "KAI'S CALL (in between)"
    assert s["verdict"] == ("KAI'S CALL ((1) E1; (1b) KAI'S CALL "
                            "(in between))")
    s = pc.summarize(_recs_flat([1, 1]), 18)          # nothing live
    assert "verdict_1b" not in s and s["verdict"] == "E1"


def test_read_1b_counts_the_vocabulary_on_the_live_decisions():
    """A live decision (grid classes > 1) whose vocabulary sits in one
    class: (1b) is the vocabulary's classes, and "live" is the grid's
    (audit r2b LOW-1)."""
    one = pc.decision_record(1, 0, 0, [0, 1], _part([0, 0, 1, 2, 0, 1, 1, 1,
                                                     1]),
                             list(range(36)), [1.0, 0.0])
    assert one["classes_vocab"] == 1 and one["classes_grid"] == 3
    twelve = _recs_flat([12])[0]
    s = pc.summarize([one, twelve], 18)
    assert s["live_share"] == 1.0
    assert s["ratio_vocab_live"] == pytest.approx(13 / 36, abs=1e-4)


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
# one decision end to end on the scripted emulator
# --------------------------------------------------------------------------
def _fake(ale, prior):
    env = SimpleNamespace(unwrapped=SimpleNamespace(ale=ale,
                                                    _action_set=ACTIONS))
    nm = SimpleNamespace(initial_inference=lambda stack: (np.array(prior),
                                                          None))
    run = SimpleNamespace(searcher=SimpleNamespace(network_manager=nm))
    return env, run


def test_a_decision_censuses_restores_and_writes(monkeypatch):
    """Census.decision: the A1 phase (a sequence census at decisions 0, 3,
    6 under SEQ_EVERY 3), the restore, one line per decision, the checks
    (audit r1b L3)."""
    monkeypatch.setattr(pc, "SEQ_EVERY", 3)
    ale = _ALE()
    vocab = [(0, 4), (2, 8), (3, 4)]
    prior = [0.1, 0.7, 0.2]                     # argmax: token 1
    env, run = _fake(ale, prior)
    out = io.StringIO()
    census = pc.Census(run, vocab, out)
    start = ale.cloneState().t
    for _ in range(7):
        census.decision(env, 1, None)
        assert ale.cloneState().t == start
    lines = [json.loads(l) for l in out.getvalue().splitlines()]
    assert [l["d"] for l in lines] == list(range(7))
    assert [("seq" in l) for l in lines] == [i % 3 == 0 for i in range(7)]
    assert lines[0]["seq"]["frames"][0] == 3
    assert census.argmax_mismatches == 0 and census.replay_mismatches == 0
    # (2, 8): the five right-movers at 8 frames
    assert lines[0]["classes_vocab"] == 3 and lines[0]["chosen_class_size"] == 5
    census.decision(env, 2, None)               # token 2 is not the argmax
    assert census.argmax_mismatches == 1


def test_a_branch_order_dependence_is_caught():
    ale = _LeakyALE()
    env, run = _fake(ale, [1.0, 0.0])
    census = pc.Census(run, [(0, 4), (2, 4)], io.StringIO())
    census.decision(env, 0, None)
    assert census.replay_mismatches == 1


# --------------------------------------------------------------------------
# the hook, the refusals, the validity checks
# --------------------------------------------------------------------------
def test_the_hook_restores_play_token(monkeypatch):
    monkeypatch.setattr(ablation, "play_token", ablation.play_token)
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


def test_a_full_census_is_never_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "HERE", tmp_path)
    p = pc.decisions_file("run-x", smoke=False)
    p.write_text("{}\n")
    with pytest.raises(SystemExit, match="never overwritten"):
        pc.decisions_file("run-x", smoke=False)
    assert pc.decisions_file("run-x", smoke=True).name.endswith(
        "-smoke-decisions.jsonl")


def _row_tree(tmp_path, rows):
    snaps = tmp_path / "benchmarks" / "results" / "run-t" / "snaps"
    snaps.mkdir(parents=True)
    probes = tmp_path / "benchmarks" / "probes"
    probes.mkdir()
    (probes / "probe_exhaustive_eval-run-t-acting.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n\n")
    return snaps


def test_the_exact_eval_row_is_found(tmp_path, monkeypatch):
    games = {"1": [100.0, 700, True, 150]}
    rows = [{"iteration": 42, "exact_mean": 5.0, "max_noop": 2,
             "env": ENV_ID, "games": games},              # a smoke row
            {"iteration": 42, "exact_mean": 6.0, "env": "Pong-v4",
             "games": games},
            {"iteration": 40, "exact_mean": 7.0, "env": ENV_ID,
             "games": games},
            {"iteration": 42, "exact_mean": 1500.0, "env": ENV_ID,
             "games": games},                             # superseded
            {"iteration": 42, "exact_mean": 1699.0, "env": ENV_ID,
             "games": games}]
    snaps = _row_tree(tmp_path, rows)
    monkeypatch.setattr(pc, "HERE", tmp_path / "elsewhere")
    assert pc.exact_eval_row(snaps / "weights-iter042.pth")["exact_mean"] \
        == 1699.0
    assert pc.exact_eval_row(snaps / "weights-iter041.pth") is None
    # a short path has no run tree above it: None, not an IndexError
    assert pc.exact_eval_row(pathlib.Path("weights-iter042.pth")) is None


def test_the_validity_checks(monkeypatch):
    monkeypatch.setattr(pc, "MAX_DECISIONS", 0)
    ref = {"games": {"1": [100.0, 700, True, 150], "2": [80.0, 600, True,
                                                         120]}}
    games = {1: (100.0, 700, True, 150), 2: (80.0, 600, True, 120)}
    censused = {1: 150, 2: 120}
    assert pc.validity(games, ref, censused, 0, 0) == (True, [])
    assert pc.validity(games, None, censused, 0, 0) == (None, [])
    ok, why = pc.validity({**games, 2: (80.0, 604, True, 121)}, ref,
                          {1: 150, 2: 121}, 0, 0)
    assert ok is False and "game 2" in why[0]
    ok, why = pc.validity(games, ref, {1: 150, 2: 119}, 0, 0)
    assert ok is False and "119 decisions censused of 120" in why[0]
    ok, why = pc.validity(games, ref, censused, 1, 2)
    assert ok is False and len(why) == 2
    ok, why = pc.validity({1: games[1]}, ref, {1: 150}, 0, 0, n_games=2)
    assert ok is False and why == ["stopped after 1 of 2 games"]
    monkeypatch.setattr(pc, "MAX_DECISIONS", 5)    # a cut census: <= 5
    assert pc.validity(games, ref, {1: 5, 2: 5}, 0, 0) == (True, [])
    ok, why = pc.validity(games, ref, {1: 5, 2: 4}, 0, 0)
    assert ok is False and "4 decisions censused of 5" in why[0]


# --------------------------------------------------------------------------
# the real emulator
# --------------------------------------------------------------------------
@pytest.fixture
def mspacman():
    env = gym.make(ENV_ID, frameskip=1)
    env.reset(seed=0)
    yield env
    env.close()


def _controllable(env, window=20):
    """Step NOOP until LEFT and RIGHT are not both NOOP's window (past the
    start jingle); the saved state there."""
    ale = env.unwrapped.ale
    actions = list(env.unwrapped._action_set)
    for _ in range(60):
        start = ale.cloneState()
        f, _, _ = pc.token_census(ale, actions, start,
                                  grid=((0, 4), (3, 8), (2, 8)),
                                  window=window)
        if f[1] != f[0] or f[2] != f[0]:
            return start
        for _ in range(50):
            env.step(0)
    raise AssertionError("no input ever changed the frames")


def test_the_real_census_sees_inputs_restores_and_repeats(mspacman):
    import ale_py
    ale = mspacman.unwrapped.ale
    actions = list(mspacman.unwrapped._action_set)
    assert actions[0] == ale_py.Action.NOOP        # the branches' no input
    start = _controllable(mspacman)
    before = start.serialize()
    grid = ((0, 4), (0, 8), (3, 8), (3, 8), (2, 8))
    a = pc.token_census(ale, actions, start, grid=grid, window=20)
    assert ale.cloneState().serialize() == before
    b = pc.token_census(ale, actions, start, grid=grid, window=20)
    assert a == b                                  # deterministic
    frames = a[0]
    assert frames[0] == frames[1]                  # NOOP x4 + NOOP = NOOP x8
    assert frames[2] == frames[3]                  # the same token twice
    assert frames[2] != frames[0] or frames[4] != frames[0]   # inputs show


def test_the_game_continues_identically_after_a_census(mspacman):
    ale = mspacman.unwrapped.ale
    actions = list(mspacman.unwrapped._action_set)
    start = _controllable(mspacman)
    plan = [3] * 12 + [1] * 12

    def obs_after():
        return [mspacman.step(a)[0].tobytes() for a in plan]

    ale.restoreState(start)
    clean = obs_after()
    ale.restoreState(start)
    pc.token_census(ale, actions, start, grid=pc.GRID[:8], window=30)
    pc.sequence_census(ale, actions, start, depth=1)
    assert obs_after() == clean


def test_the_four_noop_tokens_are_always_one_class(mspacman):
    ale = mspacman.unwrapped.ale
    actions = list(mspacman.unwrapped._action_set)
    _controllable(mspacman)
    for _ in range(3):
        start = ale.cloneState()
        frames, _, _ = pc.token_census(ale, actions, start)
        assert len({frames[pc.GRID.index((0, f))]
                    for f in pc.DURATIONS}) == 1
        for _ in range(40):
            mspacman.step(3)


# --------------------------------------------------------------------------
# end to end: a censused game is the exact eval's game
# --------------------------------------------------------------------------
_E2E = r"""
import json, pathlib, shutil, sys
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
    # the policy head is zero at init (a uniform prior: always token 0);
    # random weights make the argmax depend on the frames (audit r1b M1),
    # and the 4-frame tokens biased out carry the census's 60 decisions
    # past the start jingle (audit r2b M1)
    fc = run.trainer.prediction.policy_fc
    torch.nn.init.normal_(fc.weight, std=0.5)
    with torch.no_grad():
        for i, t in enumerate(vocab):
            fc.bias[i] = -20.0 if t[1] == 4 else 0.0
    nets = {{n: getattr(run.trainer, n).state_dict()
            for n in ablation.NET_NAMES}}
    torch.save({{**nets, "training_steps": 0,
                "spec": {{"TOKEN_VOCAB": [list(t) for t in vocab]}},
                "acting_states": {{"representation": nets["representation"],
                                  "prediction": nets["prediction"]}}}}, snap)
pc.HERE = out
# (a) a full run without its exact-eval row refuses before anything plays
pc.GAMES, pc.MAX_DECISIONS, pc.SEQ_EVERY, pc.SMOKE = 30, 0, 10, False
try:
    pc.main([str(snap)])
    print("REFUSAL", json.dumps(None))
except SystemExit as e:
    print("REFUSAL", json.dumps(str(e)))
print("FULLFILE", json.dumps((out / "probe_action_census-run-c-decisions"
                              ".jsonl").exists()))
# (b) a smoke census without a row
pc.GAMES, pc.MAX_DECISIONS, pc.SEQ_EVERY, pc.SMOKE = 1, 60, 30, True
pc.main([str(snap)])
shutil.copy(out / "probe_action_census-run-c-smoke-decisions.jsonl",
            out / "b-decisions.jsonl")
# (c) the plain game, every token recorded
tokens = []
orig = ablation.play_token
def recorded(env, token, stack, inner=None):
    tokens.append(int(token))
    return orig(env, token, stack, inner)
ablation.play_token = recorded
with ablation.arm_config(pe.ARM), ablation._run_length_globals(90, 2000), \
        pe.snapshot_tokens(vocab):
    run = ablation.Run(pe.ARM, 0, ablation.ARMS[pe.ARM])
    d = pe.load(run, snap)
    pe.apply_acting_states(run, d)
    plain = pe.play(run, 1)
ablation.play_token = orig
print("PLAIN", json.dumps(list(plain)))
print("TOKENS", json.dumps(tokens))
# (d) the exact eval's row is the plain game: the main line checks out
row = {{"iteration": 7, "exact_mean": plain[0], "env": pc.ENV_ID,
       "games": {{"1": list(plain)}}}}
rows = out / "probe_exhaustive_eval-run-c-acting.jsonl"
rows.write_text(json.dumps(row) + "\n")
pc.MAX_DECISIONS = 3
pc.main([str(snap)])
# (e) a newer row that differs: VOID (the last matching row wins)
row["games"]["1"][0] = plain[0] + 10
rows.write_text(rows.read_text() + json.dumps(row) + "\n")
pc.main([str(snap)])
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

    def tagged(tag):
        return json.loads(next(l for l in r.stdout.splitlines()
                               if l.startswith(tag + " "))[len(tag) + 1:])

    # (a) the refusal came before anything played
    assert "no deployed exact-eval row" in tagged("REFUSAL")
    assert tagged("FULLFILE") is False
    plain, tokens = tagged("PLAIN"), tagged("TOKENS")
    assert len(tokens) == plain[3] and len(set(tokens)) > 1
    runs = [json.loads(l) for l in (tmp_path / "probe_action_census-run-c-"
                                    "smoke.jsonl").read_text().splitlines()]
    assert len(runs) == 3
    # (b) the censused game is the plain game, decision by decision
    b = runs[0]
    assert pc.same_game(b["games"]["1"], plain)
    assert b["main_line_ok"] is None and b["void_reasons"] == []
    decs = [json.loads(l) for l in (tmp_path / "b-decisions.jsonl")
            .read_text().splitlines()]
    assert len(decs) == min(60, plain[3])
    assert [d["token"] for d in decs] == tokens[:len(decs)]
    assert any(d["classes_grid"] > 1 for d in decs)        # past the jingle
    assert [("seq" in d) for d in decs] == [d["d"] % 30 == 0 for d in decs]
    for d in decs:
        assert len(d["partition"]) == 36
        assert len({d["partition"][pc.GRID.index((0, f))]
                    for f in pc.DURATIONS}) == 1
        assert d["classes_vocab"] <= 18 and d["classes_grid"] <= 36
        assert d["strict_grid"] >= d["classes_grid"]
        if "seq" in d:
            k = d["seq"]["frames"]
            assert 1 <= k[0] <= 9 and k[0] <= k[1] <= 81 and k[1] <= k[2] <= 729
    # (d) with the matching row: the main line checks out
    assert runs[1]["main_line_ok"] is True and runs[1]["void_reasons"] == []
    assert not runs[1]["verdict"].startswith("VOID")
    # (e) with a differing (newer) row: VOID, everywhere
    e = runs[2]
    assert e["main_line_ok"] is False and e["verdict"].startswith("VOID")
    assert e["verdict_1"] == "VOID" and "verdict_if_valid" in e
    assert "game 1" in e["void_reasons"][0]
