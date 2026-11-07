"""Test Q (Kai, 2026-11-07; CURRICULUM.md "Test Q", Q-A1, Q-A2, Q-A5): mining
over long training on Q*bert. Pins: holds kept out of the candidates (Q-A1);
the played-path weights and what they do to a record's target (Q-A2); the
ceiling held at its cap and never below the master's (Q-A5), with 25 units
exactly at the supports' edge for a 16-base-step record; the arms (24 tokens,
reward unit 25, the mined arm its control but for mining and the master); a
student's refusals of a prep without the rules or the cap; the supervisor's
two lineages (the game's names, 300,000 env steps, 5 lives per iteration, the
prep's flags, the near-random bar, the external q0 in this tree and not frozen)
-- every Ms. Pac-Man lineage's names as before -- and, slow, on real Q*bert
(CPU): a mined prep from a logged master and its student's arm."""
import json
import pathlib
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] /
                       "experiments"))
import ablation
import distill
import muzero
from tests.test_e3 import MX, S6, V6, _plan_life
from tests.test_prioract2 import restore_rngs  # noqa: F401
from tests.test_supervise_testb import SUPERVISORS, _load
from tests.test_testb import _pin

Q = [f"{ablation.QT_ARM}-q{k}" for k in range(9)]
P = [None] + [f"{ablation.QM_ARM}-p{k}" for k in range(1, 9)]
QBERT = "QbertNoFrameskip-v4"


# --------------------------------------------------------------------------
# Q-A1: holds are not candidates
# --------------------------------------------------------------------------
def _slots(vocab, tokens):
    return [vocab.index(t) for t in tokens]


def test_holds_are_kept_out_only_with_holds_out():
    v = ablation.QBERT_DURATION_TOKENS
    played = [(3, 32), (3, 32), (5, 32), (3, 16), (3, 16)]
    lives = [{"slots": _slots(v, played), "full": [True] * 5, "offset": 0}]
    high = {0, 1, 2, 3}
    old, _, old_stats = distill.mine_candidates(lives, v, high, v)
    new, _, stats = distill.mine_candidates(lives, v, high, v, holds_out=True)
    assert (3,) * 16 in [c["inputs"] for c in old]
    assert "hold" not in old_stats
    keys = [c["inputs"] for c in new]
    assert all(len(set(k)) > 1 for k in keys) and keys
    assert stats["hold"] >= 1
    assert (3,) * 8 + (5,) * 8 in keys
    assert stats["counted"] + stats["hold"] + stats.get("vocab_equal", 0) \
        + stats.get("over_length", 0) + stats.get("cut_short", 0) \
        == stats["high_runs"]


def test_a_hold_equal_to_a_token_counts_as_a_hold_under_the_rule():
    v = ablation.QBERT_DURATION_TOKENS
    lives = [{"slots": _slots(v, [(3, 16), (3, 16)]), "full": [True] * 2,
              "offset": 0}]
    cands, dropped, stats = distill.mine_candidates(lives, v, {0}, v,
                                                    holds_out=True)
    assert cands == [] and dropped == {} and stats["hold"] == 1


# --------------------------------------------------------------------------
# Q-A2: played paths
# --------------------------------------------------------------------------
PATHS = [(1, 2), (1, 3), (4, 2)]


def test_played_path_is_certain_and_its_slots_other_paths_get_nothing():
    w = np.array([[0.2, 0.3, 0.4], [0.2, 0.3, 0.4], [0.2, 0.3, 0.4]])
    out = distill.played_path_weights(w, {1: (1, 3)}, PATHS)
    assert out[0].tolist() == [0.2, 0.3, 0.4]         # another record
    assert out[2].tolist() == [0.2, 0.3, 0.4]
    assert out[1].tolist() == [0.0, 1.0, 0.4]         # slot 4's path stays
    assert w[1].tolist() == [0.2, 0.3, 0.4]           # a copy


def test_a_played_path_outside_the_spellings_raises():
    with pytest.raises(ValueError, match="no spelling"):
        distill.played_path_weights(np.ones((1, 3)), {0: (9, 9)}, PATHS)


def test_a_played_record_moves_its_first_slots_whole_mass():
    # master tokens 0..2, student keeps 0 and 2, token 0 dropped? no: all
    # kept; one mined token spelled (1, 2)
    master = ((0, 4), (1, 4), (2, 4))
    student = master + (((1, 4), (2, 4)),)
    q = np.array([0.2, 0.6, 0.2])
    prior = np.array([[0.25]])
    plain, _, _ = distill.masked_targets([q], master, student, mined=[[1]],
                                         weights=prior)
    assert plain[0][3] == pytest.approx(0.6 * 0.25)
    assert plain[0][1] == pytest.approx(0.6 * 0.75)
    w = distill.played_path_weights(prior, {0: (1, 2)}, [(1, 2)])
    got, _, _ = distill.masked_targets([q], master, student, mined=[[1]],
                                       weights=w)
    assert got[0][3] == pytest.approx(0.6) and got[0][1] == 0.0
    assert got[0].sum() == pytest.approx(1.0)


def test_retokenize_reports_what_each_relabeled_record_replaced():
    traj, inner = _plan_life(V6, [(3, 2), (2, 1), (5, 4), (1, 1), (1, 1),
                                  (2, 1), (3, 1)])
    mined = {muzero.token_inputs(MX): 5}
    plain = distill.retokenize(traj, inner, V6, S6, mined=mined)
    played = {}
    out = distill.retokenize(traj, inner, V6, S6, mined=mined, played=played)
    assert [r[1:] for r in out] == [r[1:] for r in plain]
    assert played == {0: (3, 2), 5: (1, 1, 2)}
    assert all(out[i][1] == 5 for i in played)


# --------------------------------------------------------------------------
# Q-A5: the ceiling
# --------------------------------------------------------------------------
@pytest.mark.parametrize("halved, c0, cap, want", [
    (7.0, 1.0, None, (7.0, False)),
    (7.0, 1.0, 25.0, (7.0, False)),
    (25.0, 7.0, 25.0, (25.0, False)),
    (40.0, 7.0, 25.0, (25.0, True)),
    (40.0, 25.0, 25.0, (25.0, True)),
    (40.0, 30.0, 25.0, (30.0, True)),           # never below the master's
])
def test_the_ceiling_is_held_at_its_cap(halved, c0, cap, want):
    assert distill.held_ceiling(halved, c0, cap) == want


def test_25_units_is_the_supports_edge_for_a_16_step_record():
    with ablation.arm_config(Q[1]):
        assert ablation.ceiling_fits_supports(25.0, 16)
        assert not ablation.ceiling_fits_supports(25.5, 16)
        assert ablation.ceiling_fits_supports(25.5, 1)


# --------------------------------------------------------------------------
# the arms
# --------------------------------------------------------------------------
def test_qbert_tokens_and_reward_unit():
    v = ablation.QBERT_DURATION_TOKENS
    assert len(v) == 24 and {a for a, _ in v} == set(range(6))
    assert {f for _, f in v} == {4, 8, 16, 32}
    assert ablation.REWARD_UNITS["qbertnoframeskip"] == 25.0
    assert ablation.REWARD_UNITS["mspacmannoframeskip"] == 10.0
    for arm in Q + P[1:]:
        spec = ablation.ARMS[arm]
        assert spec["TOKEN_VOCAB"] == v and spec["REWARD_UNIT"] == 25.0
        assert spec["_ceiling_cap"] == 25.0


def test_q0_is_t0_on_qbert():
    t0 = ablation.ARMS[ablation.TWO_ARM + "-t0"]
    q0 = ablation.ARMS[Q[0]]
    skip = ("TOKEN_VOCAB", "REWARD_UNIT", "_ceiling_cap", "_doc")
    assert {k: v for k, v in q0.items() if k not in skip} == \
        {k: v for k, v in t0.items() if k not in skip}
    assert "_student" not in q0


@pytest.mark.parametrize("k", range(1, 9))
def test_students_are_t1_on_qbert_and_mined_is_control_plus_mining(k):
    t1 = ablation.ARMS[ablation.TWO_ARM + "-t1"]
    qt, qm = ablation.ARMS[Q[k]], ablation.ARMS[P[k]]
    skip = ("TOKEN_VOCAB", "REWARD_UNIT", "_ceiling_cap", "_doc",
            "_master_arm", "_vocab_parent_arm")
    assert {a: b for a, b in qt.items() if a not in skip} == \
        {a: b for a, b in t1.items() if a not in skip}
    skip = ("_master_arm", "_mine", "_mine_q", "_doc")
    assert {a: b for a, b in qt.items() if a not in skip} == \
        {a: b for a, b in qm.items() if a not in skip}
    assert "_mine" not in qt and "_mine_q" not in qt
    assert qm["_mine"] is True and qm["_mine_q"] is True
    assert qt["_master_arm"] == Q[k - 1]
    assert qm["_master_arm"] == (Q[0] if k == 1 else P[k - 1])
    assert qt["_vocab_parent_arm"] == qm["_vocab_parent_arm"] == Q[0]


def _vocab_doc(tmp_path, **kw):
    v = ablation.QBERT_DURATION_TOKENS
    doc = {"vocab": [list(t) for t in v[:-2]],
           "master_vocab": [list(t) for t in v],
           "dropped": [list(t) for t in v[-2:]], "student_ceiling": 2.0,
           "master_arm": Q[0], "master_iterations": 50,
           "student_discount": 0.997 ** 4, "discount_rule": "keep",
           "set_sha256": "x", **kw}
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps(doc))
    return str(path)


def test_a_student_refuses_a_prep_without_the_cap(tmp_path, monkeypatch):
    monkeypatch.setitem(ablation.ARMS, Q[1], ablation.ARMS[Q[1]])
    with pytest.raises(SystemExit, match="ceiling_cap"):
        ablation.student_arm(Q[1], _vocab_doc(tmp_path))


def test_a_mined_student_refuses_a_prep_without_the_rules(tmp_path,
                                                          monkeypatch):
    monkeypatch.setitem(ablation.ARMS, P[1], ablation.ARMS[P[1]])
    with pytest.raises(SystemExit, match="not a Test Q prep"):
        ablation.student_arm(P[1], _vocab_doc(tmp_path, mine=True,
                                              ceiling_cap=25.0))


# --------------------------------------------------------------------------
# the supervisor
# --------------------------------------------------------------------------
@pytest.fixture
def sq():
    return _load(SUPERVISORS / "supervise_testb.py", "supervise_testb_q")


def test_the_lineages_bind_the_game(sq):
    sq.configure("q")
    assert sq.ENV_ID == QBERT and sq.TAG == "-qbertnoframeskip-skip4"
    assert sq.NEAR_RANDOM == pytest.approx(2 * 163.9)
    assert sq.TOTAL_ENV_STEPS == 300000 and sq.GENERATIONS == 9
    assert sq.target("q0") == 33333 and sq.target("q8") == 300000
    assert sq.REWARD_UNIT == 25.0 and sq.N_TOKENS == 24
    assert sq.VARIANTS["q3"] == Q[3]
    sq.configure("p")
    assert sq.ENV_ID == QBERT and sq.target("p8") == 300000
    assert sq.VARIANTS["p3"] == P[3] and sq.VARIANTS["q0"] == Q[0]
    assert sq.master_of("p1") == "q0" and sq.master_of("p2") == "p1"
    assert sq.is_external("q0") and not sq.is_frozen("q0")


@pytest.mark.parametrize("letter", ["f", "t", "m", "w"])
def test_every_ms_pacman_lineage_keeps_its_names(sq, letter):
    sq.configure("q")
    sq.configure(letter)
    assert sq.ENV_ID == "MsPacmanNoFrameskip-v4"
    assert sq.TAG == "-mspacmannoframeskip-skip4"
    assert sq.NEAR_RANDOM == 502.08 and sq.TOTAL_ENV_STEPS == 100000
    assert sq.REWARD_UNIT == 10.0 and sq.N_TOKENS == 36
    assert sq.EPISODES_PER_ITER == 3
    assert not sq.is_frozen("q0") and (
        letter != "m" or sq.is_frozen("t0"))


def test_the_run_command_is_the_games(sq, monkeypatch):
    sq.configure("q")
    cmd = sq.command("q0", 0, stop=33333)
    assert cmd[cmd.index("--env") + 1] == QBERT
    assert cmd[cmd.index("--episodes-per-iter") + 1] == "5"
    assert cmd[cmd.index("--arms") + 1] == Q[0]
    sq.configure("t")
    cmd = sq.command("t0", 0, stop=33333)
    assert cmd[cmd.index("--episodes-per-iter") + 1] == "3"


def test_the_prep_command_carries_the_rules_and_the_game(sq):
    pick = {"snapshot": "s.pth", "iterations": 50}
    sq.configure("p")
    cmd = sq.prep_command("p2", 0, pick)
    assert "--mine" in cmd and "--mine-q" in cmd
    assert cmd[cmd.index("--env") + 1] == QBERT
    sq.configure("q")
    cmd = sq.prep_command("q2", 0, pick)
    assert "--mine" not in cmd and "--mine-q" not in cmd
    assert cmd[cmd.index("--env") + 1] == QBERT
    sq.configure("m")
    cmd = sq.prep_command("m2", 0, pick)
    assert "--mine" in cmd and "--mine-q" not in cmd and "--env" not in cmd


def test_this_suite_is_a_preflight_suite(sq):
    assert "tests/test_testq.py" in sq.SUITES


# --------------------------------------------------------------------------
# real Q*bert (CPU)
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_a_mined_prep_on_real_qbert(monkeypatch, tmp_path,
                                    restore_rngs):  # noqa: F811
    master_arm = "testq-master"
    _pin(monkeypatch)
    for k in ("SCHEDULE_ENV_TARGET", "SCHEDULE_ENV_STEPS", "OWN_PLAY_LOG"):
        monkeypatch.setattr(muzero, k, getattr(muzero, k))
    monkeypatch.setitem(ablation.ARMS, master_arm,
                        {**ablation.ARMS[Q[0]], "ACT_SEARCH_SIMS": 4})
    monkeypatch.setitem(ablation.ARMS, P[1], {**ablation.ARMS[P[1]],
                                              "_master_arm": master_arm})
    saved = ablation.ENV_NAME, ablation.FRAMESKIP, ablation.ACTIONS
    ablation.ENV_NAME, ablation.FRAMESKIP, ablation.ACTIONS = QBERT, 4, 6
    log = ablation.OwnPlayLog(tmp_path / "own-play")
    try:
        with ablation.arm_config(master_arm) as spec, \
                ablation._run_length_globals(2, 2000), \
                ablation.token_actions():
            torch.manual_seed(5)
            run = ablation.Run(master_arm, 0, spec)
            try:
                n = 0
                for it in (1, 2):
                    for _ in range(4):
                        ablation.INNER_FRAMES = []
                        try:
                            traj, _, steps, term = run.play_episode(
                                greedy=False)
                            inner = ablation.INNER_FRAMES
                        finally:
                            ablation.INNER_FRAMES = None
                        n += steps
                        log.write(it, traj,
                                  {"inner": inner,
                                   "real_done": bool(run.env.was_real_done),
                                   "temperature": 0.5}, term, steps, n)
                nets = {k: getattr(run.trainer, k) for k in ablation.NET_NAMES}
                run.trainer.training_steps = 4000
                blob = {"arm": master_arm, "seed": 0, "spec": dict(spec),
                        "training_steps": 4000, "iteration": 2,
                        **{k: v.state_dict() for k, v in nets.items()},
                        "acting_states": {k: nets[k].state_dict()
                                          for k in ("representation",
                                                    "prediction")}}
            finally:
                run.env.close()
        master = tmp_path / "weights-iter002.pth"
        torch.save(blob, master)
        out = tmp_path / "prep"
        h = distill.prep(master, 2, out, log=lambda m: None,
                         from_log=tmp_path / "own-play", mine=True,
                         mine_q=True, env=QBERT, drop_share=0.03)
    finally:
        ablation.ENV_NAME, ablation.FRAMESKIP, ablation.ACTIONS = saved
    assert h["env"] == QBERT
    assert h["mine_rules"] == {"holds_out": True, "played_paths": True}
    assert h["ceiling_cap"] == 25.0 and h["student_ceiling"] <= 25.0
    assert h["reward_unit"] == 25.0
    assert h["mining"]["hold"] >= 1
    for e in h["mined"]:
        assert len(set(e["inputs"])) > 1 and e["frames"] <= 64
    doc = json.loads((out / "vocab.json").read_text())
    assert doc["mine_rules"] == h["mine_rules"]
    # a played record's target is at least its product's
    _hh, lives = distill.read_set(out / "set.pkl")
    S = len(h["vocab"])
    n_kept = S - len(h["mined"])
    at = [r[3][r[1]] for lf in lives for r in lf["traj"][:-1]
          if r[1] >= n_kept]
    assert len(at) == sum(h["relabeled"].values())
    # the arm takes the prep, its ceiling and its vocabulary
    ablation.student_arm(P[1], str(out / "vocab.json"))
    spec = ablation.ARMS[P[1]]
    assert spec["REWARD_GROWTH_START"] == h["student_ceiling"]
    assert muzero.canon_vocab(spec["TOKEN_VOCAB"]) == \
        muzero.canon_vocab(h["vocab"])
