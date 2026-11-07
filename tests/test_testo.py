"""Test O's O0 (Kai, 2026-11-07; CURRICULUM.md "Next experiment
(pre-registered) -- Test O: OptionZero's option head in place of the token
vocabulary", ARMS, O0 PRE-CHECK, BUILD ORDER). Pins: the arm, C0's but for the
listed knobs; what C0's token knobs mean or do not do without a vocabulary; the
gate; the plain search path; the pre-check's statistic on hand-computed
trajectories, life ends and cut lives included, the learner's argmax and not
the actor's, its row fields and its window; the flag off leaving every other
arm's search, RNG streams and rows alone; the supervisor's lineage; and, end to
end on real Ms. Pac-Man (CPU), a run's rows. CPU only."""
import contextlib
import json
import pathlib
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] /
                       "experiments"))
import ablation
import muzero
import muzero_vtree
from tests.test_prioract2 import _fake_pipe, restore_rngs  # noqa: F401
from tests.test_supervise_testb import _py_each, real_sup, sup  # noqa
from tests.test_testc import make_run

O0 = ablation.PLAIN_ARM + "-o0"
C0 = ablation.LONG_ARM + "-n0"
ENV_ID = "MsPacmanNoFrameskip-v4"
A = 9
PC_FIELDS = {"pc_positions", "pc_hold2", "pc_hold3", "pc_hold2_share",
             "pc_hold3_share", "pc_pi_m", "pc_pi_played", "pc_hit_rate"}
# read (6)'s buffer size, a token arm's fields, on O0 too
REPLAY_FIELDS = {"replay_trajs", "replay_records", "replay_base_steps"}
# C0's knobs O0 changes (_doc apart); every other key is C0's
DIFFERENT = {"TOKEN_VOCAB", "TOKEN_LONG_BIAS", "OWN_PLAY_LOG",
             "OPTION_PRECHECK", "VTREE_REFRESH_BATCH", "VTREE_COMMUNAL_TRAJS"}


@pytest.fixture(autouse=True)
def no_leak():
    """Torn down last: no test leaves the flag or a vocabulary set."""
    yield
    assert muzero.OPTION_PRECHECK is False
    assert muzero.TOKEN_VOCAB is None


@pytest.fixture
def in_arm(restore_rngs):  # noqa: F811
    """Inside an arm's config at 9 env moves; yields a function entering an
    arm (O0 by default) and returning its spec. As test_testc's, no
    monkeypatch here: a test's own monkeypatch (requested AFTER in_arm) is
    then undone BEFORE the arm is left -- the other order would restore an
    arm-set knob into the module after arm_config restored it."""
    saved = {k: getattr(ablation, k)
             for k in ("ACTIONS", "ENV_NAME", "FRAMESKIP", "DEVICE")}
    ablation.ACTIONS, ablation.ENV_NAME = A, ENV_ID
    ablation.FRAMESKIP, ablation.DEVICE = 4, torch.device("cpu")
    with contextlib.ExitStack() as stack:
        def enter(arm=O0):
            spec = stack.enter_context(ablation.arm_config(arm))
            stack.enter_context(ablation.token_actions())
            return spec
        yield enter
    for k, v in saved.items():
        setattr(ablation, k, v)


# --------------------------------------------------------------------------
# the arm
# --------------------------------------------------------------------------
def test_o0_is_c0_but_for_the_listed_knobs():
    c0, o0 = ablation.ARMS[C0], ablation.ARMS[O0]
    assert ablation.PLAIN_ARM == ("effzero-repro-faithful-prioract-noredo-"
                                  "temp05-bnrecal-mspacman-plain")
    d = {k for k in set(c0) | set(o0) if c0.get(k, "-") != o0.get(k, "-")}
    assert d == DIFFERENT | {"_doc"}
    assert o0["TOKEN_VOCAB"] is None and o0["TOKEN_LONG_BIAS"] == 0.0
    assert o0["OWN_PLAY_LOG"] is False and o0["OPTION_PRECHECK"] is True
    assert len(c0["TOKEN_VOCAB"]) == 36 and "OPTION_PRECHECK" not in c0


def test_o0_holds_the_registered_recipe():
    o0 = ablation.ARMS[O0]
    assert o0["DISCOUNT"] == 0.997 ** 4
    assert o0["REWARD_GROWTH"] is True and o0["REWARD_UNIT"] == 10.0
    assert o0["REWARD_GROWTH_START"] == o0["REWARD_GROWTH_CEILING"] == 1.0
    assert o0["PREFIX_BN_PER_STEP"] is True
    assert o0["ACT_SEARCH_SIMS"] == 50 and o0["BIRTH_SEARCH"] is True
    assert o0["USE_SEARCH_ACTING"] is False and o0["USE_GUMBEL_SEARCH"] is False
    assert o0["REFRESH_VTREE"] is True and o0["BN_RECAL_ACTING"] is True
    assert o0["PRIOR_ACT_STAGED_TEMP"] is True
    assert o0["PRIOR_ACT_TEMP_SCALE"] == 0.5
    assert o0["MAX_REPLAY_TRANSITIONS"] == 31250 and o0["MAX_REPLAY"] == 255
    assert o0["_env_stop"] is True and o0["_nominal_iterations"] == 150
    assert o0["DORMANT_DIAGNOSTIC"] is True
    for gone in ("_student", "_master_arm", "_mine", "RESET_EVERY_STEPS",
                 "RAMP_STEPS"):
        assert gone not in o0


def test_only_o0_has_the_flag_on():
    assert muzero.OPTION_PRECHECK is False
    assert [n for n, s in ablation.ARMS.items()
            if s.get("OPTION_PRECHECK")] == [O0]


# --------------------------------------------------------------------------
# C0's knobs without a vocabulary
# --------------------------------------------------------------------------
def test_the_token_knobs_are_inert_and_the_steps_are_base_steps(in_arm):
    in_arm()
    assert ablation.ACTIONS == A                     # the env's, not 36
    assert muzero.token_edge_discounts() is None     # the scalar discount
    assert muzero.token_frames(5) == muzero.TOKEN_BASE_FRAMES == 4
    assert muzero.token_long_bias(0) == 0.0
    assert ablation.support_steps(muzero.TOKEN_VOCAB) == 1


def test_the_long_token_bias_is_inert_even_when_inherited(in_arm, monkeypatch):
    in_arm()
    monkeypatch.setattr(muzero, "TOKEN_LONG_BIAS", 0.5)
    assert muzero.token_long_bias(0) == 0.0


def test_the_replay_caps_count_records_which_are_base_steps(in_arm):
    in_arm()
    life = [{"action": a} for a in (1, 2, 3, 0)]
    assert ablation.replay_units(life) == 4
    assert muzero.MAX_REPLAY_TRANSITIONS == 31250
    replay = [[{"action": 0}] * 20000, [{"action": 0}] * 20000]
    ablation.evict_replay(replay)
    assert len(replay) == 1


def test_the_value_target_runs_five_base_steps_at_the_per_step_discount(in_arm):
    in_arm()
    g = 0.997 ** 4
    traj = [{"action": 1, "reward": 1.0, "root_value": 0.0} for _ in range(9)]
    total, gain, tail = muzero.paper_value_parts(traj, 0, muzero.TD_STEPS,
                                                 muzero.DISCOUNT)
    assert muzero.TD_STEPS == 5 and tail == 5
    assert total == pytest.approx(sum(g ** k for k in range(5)))
    assert gain == pytest.approx(g ** 5)


def test_reward_growth_at_ceiling_1_is_the_sign_clip_per_move(in_arm):
    in_arm()
    assert ablation.ceiling_fits_supports(1.0, 1)
    assert muzero.reward_ceiling(0) == muzero.reward_ceiling(10 ** 7) == 1.0
    assert muzero.grown_reward((), 1.0) == 0.0
    assert muzero.grown_reward((0.0,), 1.0) == 0.0
    assert muzero.grown_reward((10.0,), 1.0) == 1.0
    assert muzero.grown_reward((1600.0,), 1.0) == 1.0


def test_the_schedules_follow_env_steps_at_the_planned_length(in_arm):
    in_arm()
    with ablation._run_length_globals(150, 2000), \
            ablation.env_schedule(100000):
        total = muzero.TEMPERATURE_TOTAL_STEPS
        assert total == 300000
        assert muzero.schedule_steps(5) == 0
        muzero.SCHEDULE_ENV_STEPS = 50000
        assert muzero.schedule_steps(5) == total // 2
        muzero.SCHEDULE_ENV_STEPS = 200000
        assert muzero.schedule_steps(5) == total


def test_no_own_play_log_is_made(in_arm):
    spec = in_arm()
    assert muzero.OWN_PLAY_LOG is False
    ablation.env_stop_gate(spec, 100000, True, True, 300, True)
    with pytest.raises(SystemExit, match="--stop-env-steps"):
        ablation.env_stop_gate(spec, None, True, True, 300, True)


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------
def test_the_gates_pass_the_arm(in_arm):
    spec = in_arm()
    ablation.token_gate(spec, True, True, 0)         # no vocabulary: nothing
    ablation.option_gate(spec, True, True, 0)
    ablation.reward_growth_gate(True, True, 0)
    ablation.student_gate(spec, None, True)
    ablation.reset_gate(spec, True, True, 0, 2000, None)
    ablation.ramp_gate(spec)


@pytest.mark.parametrize("change", [
    {"USE_SEARCH_ACTING": True}, {"REFRESH_VTREE": False},
    {"VALUE_TARGET_MODE": "mixed"}, {"USE_GUMBEL_SEARCH": True},
    {"BIRTH_SEARCH": False}, {"USE_TARGET_MODEL": False},
    {"ASYNC_LOOP": True}, {"REFRESH_OFFLOAD": True},
    {"REFRESH_SATURATE": True}, {"USE_ATARI_WRAPPERS": False},
    {"ACTOR_GUMBEL_SEARCH": True}, {"PAPER_TARGETS": False}])
def test_the_gate_refuses_other_paths(in_arm, monkeypatch, change):
    spec = in_arm()
    for k, v in change.items():
        monkeypatch.setattr(muzero, k, v)
    with pytest.raises(SystemExit, match="tested path"):
        ablation.option_gate(spec, True, True, 0)


def test_the_gate_refuses_a_vocabulary_launch_modes_and_action_spaces(
        in_arm, monkeypatch):
    spec = in_arm()
    for args in ((False, False, 0), (True, False, 0), (True, True, 2)):
        with pytest.raises(SystemExit, match="--ray --cpu-actor"):
            ablation.option_gate(spec, *args)
    monkeypatch.setattr(ablation, "FRAMESKIP", 30)
    with pytest.raises(SystemExit, match="frameskip 4"):
        ablation.option_gate(spec, True, True, 0)
    monkeypatch.setattr(ablation, "FRAMESKIP", 4)
    monkeypatch.setattr(ablation, "ACTIONS", 5)
    with pytest.raises(SystemExit, match="whole action space"):
        ablation.option_gate(spec, True, True, 0)
    monkeypatch.setattr(muzero, "TOKEN_VOCAB", ((0, 4), (1, 4)))
    with pytest.raises(SystemExit, match="refuses a TOKEN_VOCAB"):
        ablation.option_gate(spec, True, True, 0)


def test_the_gate_ignores_every_other_arm(monkeypatch):
    # (a state it would refuse, were the flag on)
    monkeypatch.setattr(muzero, "USE_SEARCH_ACTING", True)
    ablation.option_gate({}, False, False, 2)


def test_run_one_wires_the_gate():
    import inspect
    src = inspect.getsource(ablation.run_one)
    assert "option_gate(spec, use_ray, cpu_actor, ray_workers)" in src
    assert src.index("token_actions()") < src.index("option_gate(")


# --------------------------------------------------------------------------
# the plain search path
# --------------------------------------------------------------------------
def _frames(n, seed=0):
    g = np.random.default_rng(seed)
    return g.integers(0, 255, (n, 12, *muzero.IMAGE_RESHAPE), dtype=np.uint8)


def _traj(n, seed=0):
    """n acted records and the closing one, 3 fields, as the actor's."""
    return [(o, i % A, ()) for i, o in enumerate(_frames(n + 1, seed))]


def test_o0_runs_the_plain_engine_on_real_nets(in_arm, monkeypatch):
    spec = in_arm()
    monkeypatch.setattr(muzero, "ACT_SEARCH_SIMS", 6)
    torch.set_num_threads(2)
    seen = []
    real = muzero_vtree.VectorizedPUCT.search

    def spy(self, obs, num_simulations, c_puct, discount, root_noise):
        seen.append((type(self), self.A, num_simulations, discount, root_noise,
                     muzero.token_edge_discounts()))
        return real(self, obs, num_simulations=num_simulations, c_puct=c_puct,
                    discount=discount, root_noise=root_noise)

    monkeypatch.setattr(muzero_vtree.VectorizedPUCT, "search", spy)
    run = make_run(spec, O0)
    out = run._birth_search(_traj(3), False)
    assert [len(r) for r in out] == [5] * 4
    assert all(r[3].shape == (A,) for r in out)
    assert seen == [(muzero_vtree.VectorizedPUCT, A, 6, 0.997 ** 4, True, None),
                    (muzero_vtree.VectorizedPUCT, A, 6, 0.997 ** 4, False,
                     None)]
    assert run.trainer.value_prefix.per_step == muzero.PREFIX_LSTM_HORIZON


# --------------------------------------------------------------------------
# the statistic, by hand
# --------------------------------------------------------------------------
def _policy(m, pm, played=(), other=None):
    """Birth targets with mass pm[t] on m[t]; the rest on the move after it,
    or `other` (t: mass) on the played move where that differs from m."""
    pi = np.zeros((len(m), A))
    for t, (a, p) in enumerate(zip(m, pm)):
        pi[t, a] = p
        rest = 1.0 - p
        if other and t in other:
            pi[t, played[t]] = other[t]
            rest -= other[t]
        pi[t, (a + 3) % A] += rest
    return pi


def _counts(m, pm, played, other=None):
    return ablation.option_precheck_counts(
        _policy(m, pm, played, other), np.array(played), np.array(m))


def test_five_positions_by_hand():
    # t         0    1    2    3    4
    m = (0, 1, 2, 0, 1)
    played = (0, 1, 0, 0, 1)            # t = 2 misses m
    pm = (0.4, 0.9, 0.7, 0.6, 0.9)
    got = _counts(m, pm, played, other={2: 0.2})
    # hold2: t = 0 (hit, 0.9), t = 1 (hit, 0.7), t = 2 no (a miss), t = 3
    # (hit, 0.9), t = 4 no (the life's last acted position)
    assert got["hold2"] == 3
    # hold3: t = 0 (hit, hit, 0.9 x 0.7 = 0.63); t = 1 no (t + 1 misses);
    # t = 2 no; t = 3 no (t + 2 is the closing record); t = 4 no
    assert got["hold3"] == 1
    assert got["n"] == 5 and got["hit"] == 4
    assert got["pi_m"] == pytest.approx(3.5)
    assert got["pi_played"] == pytest.approx(0.4 + 0.9 + 0.2 + 0.6 + 0.9)


def test_length_3_needs_the_product_not_each_term():
    # 0.8 x 0.6 = 0.48: both above 0.5, the product is not
    got = _counts((0, 0, 0, 0), (0.1, 0.8, 0.6, 0.1), (0, 0, 0, 0))
    assert got["hold2"] == 2            # t = 0 (0.8), t = 1 (0.6); t = 2 (0.1)
    assert got["hold3"] == 0


def test_the_threshold_is_strict():
    got = _counts((0, 0, 0), (1.0, 0.5, 1.0), (0, 0, 0))
    assert got["hold2"] == 1            # t = 1 (1.0); t = 0 sits at 0.5
    assert got["hold3"] == 0            # 0.5 x 1.0 is not above 0.5


def test_a_hit_is_needed_at_t_for_both_and_at_t_plus_1_for_length_3():
    m, pm = (0, 0, 0, 0), (0.9, 0.9, 0.9, 0.9)
    assert _counts(m, pm, (1, 0, 0, 0))["hold2"] == 2      # t = 1, 2
    assert _counts(m, pm, (1, 0, 0, 0))["hold3"] == 1      # t = 1
    assert _counts(m, pm, (0, 1, 0, 0))["hold2"] == 2      # t = 0, 2
    assert _counts(m, pm, (0, 1, 0, 0))["hold3"] == 0


@pytest.mark.parametrize("n, hold2, hold3", [(1, 0, 0), (2, 1, 0), (3, 2, 1),
                                             (4, 3, 2)])
def test_a_lifes_end_holds_nothing_and_stays_in_the_denominator(n, hold2,
                                                                hold3):
    got = _counts((0,) * n, (1.0,) * n, (0,) * n)
    assert (got["n"], got["hold2"], got["hold3"]) == (n, hold2, hold3)


# --------------------------------------------------------------------------
# through paper_targets: a stub search over stub nets
# --------------------------------------------------------------------------
class _Rep(torch.nn.Module):
    def forward(self, x):
        return x


class _Pred(torch.nn.Module):
    """Logits one-hot on moves[position], the position read from the frame
    (obs = position / 255 after preprocessing)."""

    def __init__(self, moves):
        super().__init__()
        self.moves, self.calls = moves, 0

    def forward(self, x):
        self.calls += 1
        pos = torch.round(x.reshape(len(x), -1)[:, 0] * 255).long().tolist()
        logits = torch.zeros(len(x), A)
        for i, p in enumerate(pos):
            logits[i, self.moves[p]] = 5.0
        return logits, None, None


def _nets(moves):
    return {"representation": _Rep(), "dynamics": _Rep(),
            "prediction": _Pred(moves), "value_prefix": _Rep()}


def _life(played, closing=0):
    """A life's 3-field records, each frame holding its position; `closing`
    its closing record's action, the actor's placeholder."""
    traj = [(np.full((1,), t, dtype=np.uint8), a, (0.0,))
            for t, a in enumerate(played)]
    return traj + [(np.full((1,), len(played), dtype=np.uint8), closing,
                    (0.0,))]


def _engine(monkeypatch, pols):
    class Stub:
        def __init__(self, nets, actions, device):
            pass

        def search(self, obs, num_simulations, c_puct, discount, root_noise):
            pos = [int(o.reshape(-1)[0]) for o in obs]
            return (np.stack([pols[p] for p in pos]),
                    np.zeros(len(pos)), None)

    monkeypatch.setattr(muzero_vtree, "VectorizedPUCT", Stub)


def _stub_run(spec, arm=O0):
    run = object.__new__(ablation.Run)
    run.arm, run.seed, run.spec = arm, 0, spec
    run.trainer = type("T", (), {"training_steps": 100,
                                 "device": torch.device("cpu")})()
    run.replay = []
    return run


def _rows(run):
    row = {}
    run._merge_row_stats(row)
    return {k: v for k, v in row.items() if k.startswith("pc_")}


M = (2, 2, 2, 2)
PI = _policy(M, (0.6, 0.9, 0.9, 0.9))
PLAYED = (2, 2, 2, 2)


def test_a_terminated_life_through_paper_targets(in_arm, monkeypatch):
    spec = in_arm()
    _engine(monkeypatch, list(PI))
    run = _stub_run(spec)
    run.paper_targets(_life(PLAYED), True, born_step=0, birth_nets=_nets(M))
    assert _rows(run) == {
        "pc_positions": 4, "pc_hold2": 3, "pc_hold3": 2,
        "pc_hold2_share": 0.75, "pc_hold3_share": 0.5,
        "pc_pi_m": 0.825, "pc_pi_played": 0.825, "pc_hit_rate": 1.0}


def test_a_cut_lifes_closing_search_counts_as_no_position(in_arm, monkeypatch):
    # a cut life's closing record carries a greedy search, here all on m: it
    # must not make the last acted position able to hold an option
    spec = in_arm()
    closing = np.eye(A)[2]
    _engine(monkeypatch, list(PI) + [closing])
    run = _stub_run(spec)
    run.paper_targets(_life(PLAYED), False, born_step=0,
                      birth_nets=_nets(M + (2,)))
    got = _rows(run)
    assert (got["pc_positions"], got["pc_hold2"], got["pc_hold3"]) == (4, 3, 2)


def test_the_window_pools_lives_and_never_crosses_one(in_arm, monkeypatch):
    # life A's last position meets life B's sure first position: no option
    spec = in_arm()
    a = _policy((1, 1), (0.9, 0.9))
    b = _policy((1, 1, 1), (1.0, 0.1, 0.1))
    run = _stub_run(spec)
    _engine(monkeypatch, list(a))
    run.paper_targets(_life((1, 1)), True, born_step=0,
                      birth_nets=_nets((1, 1)))
    _engine(monkeypatch, list(b))
    run.paper_targets(_life((1, 1, 1)), True, born_step=0,
                      birth_nets=_nets((1, 1, 1)))
    got = _rows(run)
    # A: t = 0 holds 2, nothing 3; B: 0.1 at t + 1 holds nothing
    assert (got["pc_positions"], got["pc_hold2"], got["pc_hold3"]) == (5, 1, 0)
    assert got["pc_hold2_share"] == 0.2
    assert got["pc_pi_m"] == pytest.approx((0.9 + 0.9 + 1.0 + 0.1 + 0.1) / 5)
    assert _rows(run) == {}             # the window restarts per row


def test_m_is_the_learners_argmax_not_the_actors(in_arm, monkeypatch):
    # the actor's recalibrated prediction prefers another move everywhere
    spec = in_arm()
    _engine(monkeypatch, list(PI))
    run = _stub_run(spec)
    learner, actor = _nets(M), _nets((5, 5, 5, 5))
    run.paper_targets(_life(PLAYED), True, born_step=0, birth_nets=learner,
                      birth_prior_nets=actor)
    assert run._birth_pmax.tolist() == list(M)
    assert run._birth_priors.argmax(1).tolist() == [5, 5, 5, 5]
    assert _rows(run)["pc_hit_rate"] == 1.0


def test_m_comes_in_the_searchs_chunks(in_arm, monkeypatch):
    spec = in_arm()
    monkeypatch.setattr(muzero, "VTREE_REFRESH_BATCH", 3)
    moves = (0, 1, 2, 3, 4, 5, 6)
    _engine(monkeypatch, [np.eye(A)[0]] * 8)
    run = _stub_run(spec)
    run._birth_search(_life(moves), True, nets=_nets(moves))
    assert run._birth_pmax.tolist() == list(moves)


def test_the_row_holds_the_buffers_size_in_records_and_base_steps(
        in_arm, monkeypatch):
    spec = in_arm()
    _engine(monkeypatch, list(PI))
    run = _stub_run(spec)
    for _ in range(2):
        run.replay.append(run.paper_targets(_life(PLAYED), True, born_step=0,
                                            birth_nets=_nets(M)))
    row = {}
    run._merge_row_stats(row)
    assert {k: row[k] for k in REPLAY_FIELDS} == {
        "replay_trajs": 2, "replay_records": 10, "replay_base_steps": 10}


def test_a_row_serializes(in_arm, monkeypatch):
    spec = in_arm()
    _engine(monkeypatch, list(PI))
    run = _stub_run(spec)
    run.paper_targets(_life(PLAYED), True, born_step=0, birth_nets=_nets(M))
    assert set(json.loads(json.dumps(_rows(run)))) == PC_FIELDS


# --------------------------------------------------------------------------
# the flag off
# --------------------------------------------------------------------------
def test_the_flag_off_takes_no_forward_and_writes_no_field(in_arm, monkeypatch):
    spec = in_arm(C0)
    assert muzero.OPTION_PRECHECK is False
    monkeypatch.setattr(muzero, "TOKEN_VOCAB", None)
    monkeypatch.setattr(ablation, "ACTIONS", A)
    _engine(monkeypatch, list(PI))
    run = _stub_run(spec, C0)
    learner, actor = _nets(M), _nets((5, 5, 5, 5))
    out = run.paper_targets(_life(PLAYED), True, born_step=0,
                            birth_nets=learner, birth_prior_nets=actor)
    assert learner["prediction"].calls == 0 and actor["prediction"].calls == 1
    assert not hasattr(run, "_birth_pmax")
    assert not hasattr(run, "_pc_window")
    row = {}
    run._merge_row_stats(row)
    assert not (PC_FIELDS | REPLAY_FIELDS) & set(row)
    assert "beh_act_prob" in row and len(out) == 5


def _one_life(flag, in_arm_spec, monkeypatch):
    """A real birth search and row on seeded nets and streams, the flag as
    given; returns the targets, the row and the streams after."""
    monkeypatch.setattr(muzero, "OPTION_PRECHECK", flag)
    torch.manual_seed(3)
    run = make_run(in_arm_spec, O0)
    run.seed = 0
    np.random.seed(5)
    torch.manual_seed(7)
    out = run.paper_targets(_traj(3), True, born_step=0)
    row = {}
    run._merge_row_stats(row)
    return (out, row, np.random.get_state(), torch.get_rng_state().clone())


def test_the_flag_changes_no_target_no_stream_and_no_other_field(
        in_arm, monkeypatch):
    spec = in_arm()
    monkeypatch.setattr(muzero, "ACT_SEARCH_SIMS", 6)
    torch.set_num_threads(2)
    on = _one_life(True, spec, monkeypatch)
    off = _one_life(False, spec, monkeypatch)
    for a, b in zip(on[0], off[0]):
        assert a["action"] == b["action"] and a["reward"] == b["reward"]
        assert np.array_equal(a["target_policy"], b["target_policy"])
        assert a["root_value"] == b["root_value"]
        assert a["priority"] == b["priority"]
    extra = PC_FIELDS | REPLAY_FIELDS
    assert extra <= set(on[1]) and not extra & set(off[1])
    keep = {k: v for k, v in on[1].items() if k not in extra
            and k != "birth_search_s_last"}
    assert keep == {k: v for k, v in off[1].items()
                    if k != "birth_search_s_last"}
    assert np.array_equal(on[2][1], off[2][1]) and on[2][2] == off[2][2]
    assert torch.equal(on[3], off[3])


# --------------------------------------------------------------------------
# the supervisor
# --------------------------------------------------------------------------
@pytest.fixture
def osup(sup):
    sup.configure("o")
    return sup


def test_the_lineage_is_n0s_pattern_alone(real_sup):  # noqa: F811
    s = real_sup
    n, o = s.LINEAGES["n"], s.LINEAGES["o"]
    assert s.PLAIN_ARM == ablation.PLAIN_ARM
    assert {k: v for k, v in o.items()
            if k not in ("arm", "run", "date", "plain")} == {
                k: v for k, v in n.items()
                if k not in ("arm", "run", "date", "after")}
    assert "after" not in o and o["plain"] is True
    s.configure("o")
    assert s.VARIANTS == {"o0": O0}
    assert s.RUNS == {"o0": "mspacman-plain-o0"} and s.NAMES == {"o0": "O0"}
    assert s.run_name("o0", 0) == "mspacman-plain-o0-s0-2026-11-07"
    assert s.DISCOUNTS == {"o0": 0.997 ** 4} and s.DISCOUNT_RULE == "keep"
    assert s.TOTAL_ENV_STEPS == 100000 and s.target("o0") == 100000
    assert s.upstream("o0") is None and s.master_of("o0") is None
    assert not s.is_student("o0") and s.PREP_PREFIX is None
    assert (s.ITERATIONS, s.NOMINAL_ITERATIONS) == (300, 150)
    assert s.ENV_ID == ENV_ID and s.EPISODES_PER_ITER == 3
    assert "tests/test_testo.py" in s.SUITES and "tests/test_testo.py" in s.AUDITED


def test_its_stop_and_command_are_n0s_with_its_arm(osup):
    sup = osup
    assert sup.stop_env_steps("o0", 0) == 100000
    c = sup.command("o0", 0)
    sup.configure("n")
    want = [x.replace(C0, O0) for x in sup.command("n0", 0, 100000)]
    assert c == want
    assert c[c.index("--arms") + 1] == O0
    assert "--token-vocab" not in c and "--distill-set" not in c


def test_main_starts_it_without_a_wait(osup, monkeypatch, capsys):
    sup = osup

    def stop(variant, seed):
        return None
    monkeypatch.setattr(sup, "settle_stop", stop)
    assert sup.main(["o0", "0"]) == 1
    assert sup.LINEAGE == "o"
    assert "o0 s0 starts the lineage" in capsys.readouterr().out
    for argv in (["o1", "0"], ["o0", "1"], ["O0", "0"], ["o", "0"]):
        assert sup.main(argv) == 2, argv


def test_its_quick_probe_passes_and_its_asserts_bite(osup):
    sup = osup
    code = sup.quick("o0", 0)
    a = "ablation.ARMS"
    anchor = f"spec = ablation.ARMS[{O0!r}]; "
    assert code.count(anchor) == 1
    breaks = [f"{a}[{O0!r}]['TOKEN_VOCAB'] = ((0, 4),)",
              f"{a}[{O0!r}]['OWN_PLAY_LOG'] = True",
              f"{a}[{O0!r}]['OPTION_PRECHECK'] = False",
              f"{a}[{O0!r}]['TOKEN_LONG_BIAS'] = 0.5",
              f"{a}[{O0!r}]['MAX_REPLAY_TRANSITIONS'] = 1000",
              f"{a}[{O0!r}]['VTREE_REFRESH_BATCH'] = 64",
              f"{a}[{O0!r}]['VTREE_COMMUNAL_TRAJS'] = 1",
              f"{a}[{O0!r}]['_nominal_iterations'] = 50",
              f"{a}[{C0!r}]['OPTION_PRECHECK'] = True"]
    broken = [code.replace(anchor, b + "; " + anchor) for b in breaks]
    got = _py_each(broken + [code])
    for b, g in zip(breaks, got):
        assert g.startswith("AssertionError"), (b, g)
    assert got[-1] == "ok"
    # the module default switched on: its own child (EACH restores no such
    # flag)
    head = "import muzero, ablation, muzero_vtree, distill; "
    flipped = code.replace(head, head + "muzero.OPTION_PRECHECK = True; ")
    assert flipped != code
    assert _py_each([flipped])[0].startswith("AssertionError")


def test_other_lineages_quick_probes_are_as_they_were(sup):
    sup.configure("n")
    code = sup.quick("n0", 0)
    assert "OPTION_PRECHECK" not in code
    assert "assert spec['OWN_PLAY_LOG'] is True; " in code
    assert "assert len(spec['TOKEN_VOCAB']) == 36; " in code


def test_a_record_is_one_base_step_for_the_reward_alert(osup):
    sup = osup
    assert sup.run_token_steps("o0") == 1
    r = {"iteration": 5, "reward_ceiling": 1.0, "reward_max": 1.0}
    assert sup.row_alerts(r, "n/a", 1.0, 1) == []
    r["reward_max"] = 2.0
    got = sup.row_alerts(r, "n/a", 1.0, 1)
    assert len(got) == 1 and "a plain record sums at most 1" in got[0]
    sup.configure("n")
    assert sup.run_token_steps("n0") == sup.MAX_TOKEN_STEPS


def _o_row(**extra):
    return {"iteration": 5, "train_steps": 10000, "env_steps": 3000,
            "return_mean": 300.0, "reward_ceiling": 1.0, "reward_kept": 0.5,
            "reward_max": 1.0, "elapsed_s": 60.0, **extra}


def test_each_row_prints_the_shares_and_a_row_without_them_alerts(
        osup, capsys):
    sup = osup
    path = sup.paths("o0", 0)[2]
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    good = _o_row(pc_positions=900, pc_hold2_share=0.0612,
                  pc_hold3_share=0.0211, pc_pi_m=0.41, pc_pi_played=0.33,
                  pc_hit_rate=0.7)
    with open(path, "a") as f:
        f.write(json.dumps(good) + "\n")
    rows = sup.Rows(path, "o0", 0)
    rows.emit()
    out = capsys.readouterr().out
    assert ("hold2 0.0612  hold3 0.0211  pi[m] 0.41  pi[played] 0.33  hit 0.7"
            "  positions 900") in out
    assert "ALERT" not in out and "frames/decision" not in out
    with open(path, "a") as f:
        f.write(json.dumps(_o_row(iteration=10)) + "\n")
    rows.emit()
    out = capsys.readouterr().out
    assert "ALERT: o0 s0 row 10: pc_positions None" in out


def test_other_lineages_rows_are_as_they_were(sup, capsys):
    sup.configure("n")
    assert sup.precheck_alert(_o_row()) is None
    path = sup.paths("n0", 0)[2]
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(_o_row()) + "\n")
    sup.Rows(path, "n0", 0).emit()
    out = capsys.readouterr().out
    assert "frames/decision" in out and "hold2" not in out


# --------------------------------------------------------------------------
# end to end: real Ms. Pac-Man, in-process pipeline, CPU
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_run_one_writes_the_shares_into_each_row(monkeypatch, tmp_path,
                                                 restore_rngs):  # noqa: F811
    torch.set_num_threads(2)
    fake = _fake_pipe([], 0)
    fake.snapshot_acting_states = lambda self: {}
    monkeypatch.setattr(ablation, "RayPipeline", fake)
    monkeypatch.setattr(ablation, "ENV_NAME", ENV_ID)
    monkeypatch.setattr(ablation, "FRAMESKIP", 4)
    monkeypatch.setattr(ablation, "ACTIONS", A)
    monkeypatch.setattr(ablation, "EPISODE_CAP", 8)
    monkeypatch.setattr(ablation, "RESULTS", tmp_path)
    monkeypatch.setattr(ablation, "TAG", "")
    monkeypatch.setattr(ablation, "DEVICE", torch.device("cpu"))
    monkeypatch.setitem(ablation.ARMS, "e2e-o0", {
        **ablation.ARMS[O0], "TRAINING_BATCHES": 16, "ACT_SEARCH_SIMS": 6})
    out = tmp_path / "a.jsonl"
    ablation.run_one("e2e-o0", 0, 300, 1, 4, 1, out, checkpoint_every=0,
                     use_ray=True, cpu_actor=True, snapshot_weights=True,
                     stop_env_steps=16)
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert [r["iteration"] for r in rows] == [1, 2]    # 8 env steps a life
    for r in rows:
        assert PC_FIELDS <= set(r) and r["pc_positions"] == 8
        assert 0 <= r["pc_hold3_share"] <= r["pc_hold2_share"] <= 1
        assert r["replay_records"] == r["replay_base_steps"] > 0
        assert r["replay_trajs"] == r["iteration"]
        assert r["reward_ceiling"] == 1.0 and r["reward_max"] <= 1.0
        assert not any(k.startswith("tok_") for k in r)
        assert r["beh_action_counts"] and len(r["beh_action_counts"]) == A
    assert not list(tmp_path.glob("own-play*"))
