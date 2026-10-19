"""Reward-clip growth (Kai, 2026-10-19; CURRICULUM "P2a on Ms. Pac-Man").
Pins: the defaults are off and both Ms. Pac-Man arms' deltas are exact; the
ceiling's log-linear schedule; the per-base-step clip (the sign clip at
ceiling 1); the actor records raw per-base-step rewards (a token keeps its
inner steps'); paper_targets clips them at the replay's ceiling and keeps
them; the whole replay is re-clipped once per iteration, at its top, and
after a resume; the row fields; the gate -- and the regression: the growth
arm with its ceiling held at 1 reproduces the Ms. Pac-Man arm bit for bit
(the real Ms. Pac-Man actor, a whole train + refresh slice)."""
import hashlib
import inspect
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] /
                       "experiments"))
import ablation
import muzero
from tests.test_prioract2 import restore_rngs  # noqa: F401
from tests.test_tokens import _Steps, _token_shell

MSP_ARM = "effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman"
RG_ARM = MSP_ARM + "-rgrow"
TK_ARM = MSP_ARM + "-tokens-rgrow"
P2A_ARM = "effzero-repro-faithful-prioract-noredo-temp05-bnrecal-tokens"
ENV_ID = "MsPacmanNoFrameskip-v4"
GROWTH_KEYS = ("REWARD_GROWTH", "REWARD_UNIT", "REWARD_GROWTH_START",
               "REWARD_GROWTH_CEILING", "REWARD_GROWTH_END")


# --------------------------------------------------------------------------
# defaults, the arms
# --------------------------------------------------------------------------
def test_defaults_are_off():
    assert muzero.REWARD_GROWTH is False
    assert (muzero.REWARD_UNIT, muzero.REWARD_GROWTH_START,
            muzero.REWARD_GROWTH_CEILING, muzero.REWARD_GROWTH_END) == \
        (1.0, 1.0, 1.0, 0.5)


def test_rg_arm_delta_is_exactly_the_growth():
    base, rg = ablation.ARMS[MSP_ARM], ablation.ARMS[RG_ARM]
    delta = {k for k in set(base) | set(rg) if base.get(k) != rg.get(k)}
    assert delta == set(GROWTH_KEYS) | {"_doc"}
    assert [rg[k] for k in GROWTH_KEYS] == [True, 10.0, 1.0, 160.0, 0.5]
    assert rg["_launch"] == {"ray": True, "cpu_actor": True}
    assert rg["REWARD_CLIP"] == 1.0           # the twin's sign clip, inherited


def test_tk_arm_delta_is_exactly_p2as_tokens():
    rg, tk = ablation.ARMS[RG_ARM], ablation.ARMS[TK_ARM]
    delta = {k for k in set(rg) | set(tk) if rg.get(k) != tk.get(k)}
    assert delta == {"TOKEN_VOCAB", "TOKEN_LONG_BIAS", "TOKEN_LONG_BIAS_END",
                     "_doc"}
    vocab = tk["TOKEN_VOCAB"]
    assert len(vocab) == 36 and len(set(vocab)) == 36
    assert vocab == tuple((a, f) for a in range(9) for f in (4, 8, 16, 32))
    assert vocab[0] == (0, 4)       # the placeholder action 0 is NOOP x4
    p2a = ablation.ARMS[P2A_ARM]
    # "the exact same setup": P2a's durations and bias
    assert {f for _, f in vocab} == {f for _, f in p2a["TOKEN_VOCAB"]}
    assert (tk["TOKEN_LONG_BIAS"], tk["TOKEN_LONG_BIAS_END"]) == \
        (p2a["TOKEN_LONG_BIAS"], p2a["TOKEN_LONG_BIAS_END"])


@pytest.mark.parametrize("arm", [RG_ARM, TK_ARM])
def test_every_startup_gate_passes_on_mspacman(arm, monkeypatch):
    for k in ablation.ARMS[arm]:
        if hasattr(muzero, k):
            monkeypatch.setattr(muzero, k, getattr(muzero, k))
    for k in ("ENV_NAME", "FRAMESKIP", "ACTIONS"):
        monkeypatch.setattr(ablation, k, getattr(ablation, k))
    ablation.ENV_NAME, ablation.FRAMESKIP, ablation.ACTIONS = ENV_ID, 4, 9
    with ablation.arm_config(arm) as spec, \
            ablation._run_length_globals(90, 2000), ablation.token_actions():
        assert ablation.ACTIONS == (36 if arm == TK_ARM else 9)
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


def test_run_one_wires_the_gate_and_the_iteration_top_reclip():
    src = inspect.getsource(ablation.run_one)
    assert "reward_growth_gate(use_ray, cpu_actor, ray_workers)" in src
    loop = src[src.index("for it in range(start_it + 1, iterations + 1):"):]
    # the re-clip comes first in the iteration: before the primed collect
    # and the sequential play, before training
    i = loop.index("run.apply_reward_ceiling()")
    assert i < loop.index("pipeline.collect()")
    assert i < loop.index("run.play_episode()")
    assert i < loop.index("train_block(")


# --------------------------------------------------------------------------
# the ceiling, the clip
# --------------------------------------------------------------------------
def test_ceiling_is_log_linear_to_its_end(monkeypatch):
    for k, v in (("REWARD_GROWTH_START", 1.0), ("REWARD_GROWTH_CEILING", 160.0),
                 ("REWARD_GROWTH_END", 0.5), ("TEMPERATURE_TOTAL_STEPS", 1000)):
        monkeypatch.setattr(muzero, k, v)
    c = [muzero.reward_ceiling(s) for s in (0, 100, 250, 499, 500, 999)]
    assert c[0] == 1.0 and c[4] == 160.0 and c[5] == 160.0
    assert c[1] == pytest.approx(160 ** 0.2, rel=1e-12)
    assert c[2] == pytest.approx(160 ** 0.5, rel=1e-12)    # geometric middle
    assert c[3] < 160.0
    monkeypatch.setattr(muzero, "REWARD_GROWTH_START", 2.0)
    monkeypatch.setattr(muzero, "REWARD_GROWTH_CEILING", 8.0)
    assert muzero.reward_ceiling(250) == pytest.approx(4.0, rel=1e-12)
    monkeypatch.setattr(muzero, "TEMPERATURE_TOTAL_STEPS", None)
    assert muzero.reward_ceiling(10 ** 6) == 2.0       # outside a run: start


def test_the_arms_schedule_matches_the_preregistration(monkeypatch):
    with ablation.arm_config(RG_ARM), ablation._run_length_globals(90, 2000):
        at = {row: muzero.reward_ceiling((row - 1) * 2000)
              for row in (1, 10, 23, 45, 46, 90)}
    assert at[1] == 1.0                         # the twin's sign clip
    assert at[10] == pytest.approx(160 ** 0.2, rel=1e-12)          # ~2.8
    assert 11.9 < at[23] < 12.7                                    # ~12.6
    assert at[45] < 160.0 and at[46] == 160.0 and at[90] == 160.0


MSP_REWARDS = (0.0, 10.0, 50.0, 60.0, 100.0, 200.0, 400.0, 800.0, 1600.0)


def test_ceiling_1_is_the_sign_clip_bit_for_bit(monkeypatch):
    monkeypatch.setattr(muzero, "REWARD_UNIT", 10.0)
    monkeypatch.setattr(muzero, "REWARD_CLIP", 1.0)
    for r in MSP_REWARDS:
        got = muzero.grown_reward((r,), 1.0)
        want = muzero.clip_reward(r)
        assert type(got) is float and got == want, r
    assert muzero.grown_reward((), 1.0) == 0.0 == muzero.clip_reward(0.0)


def test_the_clip_is_per_base_step_and_symmetric(monkeypatch):
    monkeypatch.setattr(muzero, "REWARD_UNIT", 10.0)
    token = (10.0, 10.0, 0.0, 200.0)             # 2 pellets, a ghost
    assert muzero.grown_reward(token, 1.0) == 3.0          # not 1: per step
    assert muzero.grown_reward(token, 12.5) == 14.5
    assert muzero.grown_reward(token, 160.0) == 22.0       # unclipped
    assert muzero.grown_reward((-30.0, 30.0), 2.0) == 0.0
    assert muzero.grown_reward((-30.0,), 2.0) == -2.0


# --------------------------------------------------------------------------
# recording raw rewards
# --------------------------------------------------------------------------
@pytest.fixture
def prior_play(monkeypatch):
    """Policy-only acting, no paper path, 4 actions."""
    for k, v in (("USE_SEARCH_ACTING", False), ("PAPER_TARGETS", False),
                 ("EVAL_PRIOR_ONLY", False), ("PRIOR_ACT_STAGED_TEMP", False),
                 ("ACTING_TEMPERATURE", 1.0), ("PRIOR_ACT_TEMP_SCALE", 1.0),
                 ("REWARD_CLIP", 1.0), ("TOKEN_VOCAB", None),
                 ("REWARD_GROWTH", False), ("REWARD_UNIT", 10.0)):
        monkeypatch.setattr(muzero, k, v)
    monkeypatch.setattr(ablation, "ACTIONS", 4)


def _pay(t):
    return {2: 10.0, 3: 50.0, 5: 200.0}.get(t, 0.0)


def test_plain_records_still_clip_at_record_time(prior_play):
    run = _token_shell(_Steps(reward=_pay), [0.1, 0.1, 0.7, 0.1])
    traj, total, steps, _ = run.play_episode(greedy=True, cap=6)
    assert [r[2] for r in traj] == [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]
    assert total == 260.0


def test_growth_records_carry_raw_per_step_rewards(prior_play, monkeypatch):
    monkeypatch.setattr(muzero, "REWARD_GROWTH", True)
    run = _token_shell(_Steps(reward=_pay), [0.1, 0.1, 0.7, 0.1])
    traj, total, steps, _ = run.play_episode(greedy=True, cap=6)
    assert [r[2] for r in traj] == [(), (0.0,), (10.0,), (50.0,), (0.0,),
                                    (200.0,), (0.0,)]
    assert all(type(x) is float for r in traj for x in r[2])
    assert total == 260.0 and steps == 6           # the game score stays raw


def test_growth_token_records_keep_their_inner_steps(prior_play, monkeypatch):
    monkeypatch.setattr(muzero, "REWARD_GROWTH", True)
    monkeypatch.setattr(muzero, "TOKEN_VOCAB",
                        ((0, 4), (2, 8), (3, 16), (0, 32)))
    env = _Steps(reward=_pay, end=7)
    run = _token_shell(env, [0.1, 0.1, 0.7, 0.1])   # greedy: DOWN x16
    traj, total, steps, term = run.play_episode(greedy=True, cap=100)
    # 4 base steps, then a terminal 3 steps into the second token
    assert term and steps == 7 and len(traj) == 3
    assert traj[1][2] == (0.0, 10.0, 50.0, 0.0)
    assert traj[2][2] == (200.0, 0.0, 0.0)
    assert total == 260.0


def test_play_token_collects_inner_rewards_only_when_asked(monkeypatch):
    monkeypatch.setattr(muzero, "TOKEN_VOCAB", ((2, 16),))
    env = _Steps(reward=lambda t: float(t))
    obs, _ = env.reset()
    stack = muzero.initial_frame_stack(
        muzero.preprocess_frame(obs, muzero.IMAGE_RESHAPE))
    inner = []
    out = ablation.play_token(env, 0, stack, inner)
    assert inner == [1.0, 2.0, 3.0, 4.0] and out[1] == 10.0
    assert len(ablation.play_token(env, 0, stack)) == 6   # signature unchanged


# --------------------------------------------------------------------------
# the learner: clip at the replay's ceiling, re-clip per iteration
# --------------------------------------------------------------------------
@pytest.fixture
def growth(monkeypatch):
    for k, v in (("REWARD_GROWTH", True), ("REWARD_UNIT", 10.0),
                 ("REWARD_GROWTH_START", 1.0),
                 ("REWARD_GROWTH_CEILING", 160.0),
                 ("REWARD_GROWTH_END", 0.5), ("TEMPERATURE_TOTAL_STEPS", 1000),
                 ("TOKEN_VOCAB", None), ("BIRTH_SEARCH", True),
                 ("VALUE_TARGET_MODE", "td")):
        monkeypatch.setattr(muzero, k, v)
    monkeypatch.setattr(ablation, "ACTIONS", 2)


def _learner(steps=0, replay=None):
    run = object.__new__(ablation.Run)
    run.trainer = SimpleNamespace(training_steps=steps)
    run.replay = [] if replay is None else replay
    return run


def _five(rewards):
    """5-field records (a searched trajectory) with the given reward slots."""
    return [(np.zeros(1), i % 2, r, np.array([0.5, 0.5]), 0.25 * i)
            for i, r in enumerate(rewards)]


RAW = [(), (10.0,), (200.0,), (0.0,), (50.0, 10.0)]


def test_paper_targets_clip_at_the_replays_ceiling_and_keep_raw(growth):
    run = _learner(steps=250)                      # ceiling sqrt(160)
    out = run.paper_targets(_five(RAW), terminated=True)
    c = 160 ** 0.5
    assert run._reward_ceiling == pytest.approx(c, rel=1e-12)
    assert [t["raw_reward"] for t in out] == RAW
    assert [t["reward"] for t in out] == pytest.approx(
        [0.0, 1.0, c, 0.0, 5.0 + 1.0], rel=1e-12)
    # the insertion priorities read the clipped rewards
    z0 = muzero.paper_value_target(out, 0, muzero.TD_STEPS, muzero.DISCOUNT)
    assert out[0]["priority"] == abs(out[0]["root_value"] - z0) + \
        muzero.PER_EPS


def test_paper_targets_refuse_a_record_clipped_at_record_time(growth):
    with pytest.raises(ValueError, match="raw reward tuples"):
        _learner().paper_targets(_five([0.0, 1.0]), terminated=True)


def test_plain_paper_targets_store_no_raw_rewards(growth, monkeypatch):
    monkeypatch.setattr(muzero, "REWARD_GROWTH", False)
    out = _learner().paper_targets(_five([0.0, 1.0, 1.0]), terminated=True)
    assert [t["reward"] for t in out] == [0.0, 1.0, 1.0]
    assert not any("raw_reward" in t for t in out)


def test_the_whole_replay_is_reclipped_when_the_ceiling_moves(growth):
    run = _learner(steps=0)
    run.replay.append(run.paper_targets(_five(RAW), terminated=True))
    run.replay.append(run.paper_targets(_five(RAW[::-1]), terminated=False))
    assert run._reward_ceiling == 1.0
    assert [t["reward"] for t in run.replay[0]] == [0.0, 1.0, 1.0, 0.0, 2.0]
    run.trainer.training_steps = 100          # same iteration, no new apply
    run.replay[0][1]["reward"] = -7.0
    run.apply_reward_ceiling()                # the ceiling moved: re-clip all
    c = 160 ** 0.2
    for traj in run.replay:
        for t in traj:
            assert t["reward"] == muzero.grown_reward(t["raw_reward"], c)
    run.replay[0][1]["reward"] = -7.0
    run.apply_reward_ceiling()                # unchanged ceiling: no pass
    assert run.replay[0][1]["reward"] == -7.0
    run.trainer.training_steps = 900          # past the end: 160
    run.apply_reward_ceiling()
    assert [t["reward"] for t in run.replay[0]] == [0.0, 1.0, 20.0, 0.0, 6.0]


def test_apply_is_a_no_op_without_growth(growth, monkeypatch):
    monkeypatch.setattr(muzero, "REWARD_GROWTH", False)
    run = _learner(replay=[[{"reward": 1.0}]])
    run.apply_reward_ceiling()
    assert run.replay == [[{"reward": 1.0}]]
    assert getattr(run, "_reward_ceiling", None) is None


def test_a_resume_reclips_the_loaded_buffer(tmp_path, growth):
    """The ceiling is not checkpointed: load resets it, and the next apply
    re-clips the loaded records at the resumed step's ceiling."""
    from tests.test_checkpoint_resume import EXPECT, META, make_run
    a = make_run()
    a.trainer.training_steps = META["training_steps"]
    a.replay = [[{"obs": np.zeros(1), "action": 0, "reward": 1.0,
                  "raw_reward": (200.0,)}]]
    a._reward_ceiling = 1.0
    path = tmp_path / "resume.ckpt"
    ablation.save_run_checkpoint(a, path, META)
    b = make_run()
    b._reward_ceiling = 3.0                # whatever it held before the load
    ablation.load_run_checkpoint(b, path, EXPECT)
    assert b._reward_ceiling is None
    assert b.replay[0][0]["raw_reward"] == (200.0,)
    b.apply_reward_ceiling()
    c = muzero.reward_ceiling(b.trainer.training_steps)
    assert b.replay[0][0]["reward"] == muzero.grown_reward((200.0,), c)


# --------------------------------------------------------------------------
# row fields
# --------------------------------------------------------------------------
def test_growth_rows_report_the_ceiling_the_kept_share_and_the_max(growth):
    run = _learner(steps=1000)                    # ceiling 160
    run.replay = [[{"reward": 0.0, "raw_reward": ()},
                   {"reward": 1.0, "raw_reward": (10.0,)},
                   {"reward": 2.0, "raw_reward": (10.0, 3000.0)}]]
    run._reward_ceiling = 1.0
    errs = {}
    run._merge_row_stats(errs)
    # raw units 1 + 1 + 300 = 302; kept at ceiling 1: 1 + 1 + 1
    assert errs["reward_ceiling"] == 1.0
    assert errs["reward_kept"] == round(3 / 302, 4)
    assert errs["reward_max"] == 2.0


def test_plain_rows_gain_no_reward_fields():
    run = _learner()
    run.replay = [[{"reward": 1.0}]]
    errs = {}
    run._merge_row_stats(errs)
    assert not any(k.startswith("reward_") for k in errs)


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------
@pytest.fixture
def rg_arm(monkeypatch):
    for k in ablation.ARMS[RG_ARM]:
        if hasattr(muzero, k):
            monkeypatch.setattr(muzero, k, getattr(muzero, k))
    with ablation.arm_config(RG_ARM) as spec:
        yield spec


def test_gate_passes_the_arm_and_ignores_plain_arms(rg_arm, monkeypatch):
    ablation.reward_growth_gate(True, True, 0)
    monkeypatch.setattr(muzero, "REWARD_GROWTH", False)
    monkeypatch.setattr(muzero, "ASYNC_LOOP", True)
    ablation.reward_growth_gate(False, False, 2)     # off: no checks


@pytest.mark.parametrize("change,match", [
    (("PAPER_TARGETS", False), "PAPER_TARGETS"),
    (("BIRTH_SEARCH", False), "BIRTH_SEARCH"),
    (("USE_SEARCH_ACTING", True), "USE_SEARCH_ACTING"),
    (("REFRESH_VTREE", False), "REFRESH_VTREE"),
    (("REFRESH_OFFLOAD", True), "REFRESH_OFFLOAD"),
    (("REFRESH_SATURATE", True), "REFRESH_SATURATE"),
    (("ASYNC_LOOP", True), "ASYNC_LOOP"),
    (("VALUE_TARGET_MODE", "mixed"), "VALUE_TARGET_MODE"),
    (("REWARD_UNIT", 0.0), "REWARD_UNIT"),
    (("REWARD_GROWTH_START", 0.0), "START"),
    (("REWARD_GROWTH_START", 200.0), "START"),
    (("REWARD_GROWTH_END", 0.0), "REWARD_GROWTH_END"),
    (("REWARD_GROWTH_END", 1.5), "REWARD_GROWTH_END"),
    (("REWARD_GROWTH_CEILING", 1e6), "past the reward support"),
])
def test_gate_refusals(rg_arm, monkeypatch, change, match):
    monkeypatch.setattr(muzero, *change)
    with pytest.raises(SystemExit, match=match):
        ablation.reward_growth_gate(True, True, 0)


@pytest.mark.parametrize("use_ray,cpu_actor,workers", [
    (False, False, 0), (True, False, 0), (True, True, 2)])
def test_gate_refuses_other_launch_modes(rg_arm, use_ray, cpu_actor, workers):
    with pytest.raises(SystemExit, match="--ray --cpu-actor"):
        ablation.reward_growth_gate(use_ray, cpu_actor, workers)


# --------------------------------------------------------------------------
# the regression: ceiling 1 IS the Ms. Pac-Man arm; the real actors
# --------------------------------------------------------------------------
def _pin_globals(monkeypatch, keys):
    for k in keys:
        if hasattr(muzero, k):
            monkeypatch.setattr(muzero, k, getattr(muzero, k))
    for k in ("ACTIONS", "EPISODE_CAP", "ENV_NAME", "FRAMESKIP", "DEVICE"):
        monkeypatch.setattr(ablation, k, getattr(ablation, k))


SCHEDULE = {"TEMPERATURE_TOTAL_STEPS": 180000, "OFFPOLICY_DECAY_STEPS": 54000}


def _arm(name):
    return {k: v for k, v in ablation.ARMS[name].items()
            if not k.startswith("_")}


@pytest.mark.slow
def test_ceiling_1_reproduces_the_plain_actor_on_real_mspacman(
        monkeypatch, restore_rngs):  # noqa: F811
    """The real Ms. Pac-Man actor (eval games + two self-play lives), the
    growth arm with its ceiling held at 1 against the twin's arm: actions,
    stacks, lengths, returns, RNG state equal; each raw reward tuple clips
    to the twin's stored reward."""
    import ray_actors
    base, rg = _arm(MSP_ARM), _arm(RG_ARM)
    _pin_globals(monkeypatch, list(rg) + list(SCHEDULE))
    torch.set_num_threads(2)

    def trace(overrides, clip):
        actor = ray_actors.SelfPlayActor(overrides, 9, 5000, SCHEDULE,
                                         ENV_ID, 4, 150, server=None,
                                         server_set="eval", force_cpu=True)
        log = [actor.eval_episodes([500000])]
        for s in (7, 8):
            t, term, n = actor.play(s)
            digest = hashlib.sha1(b"".join(r[0].tobytes() for r in t))
            log.append((term, n, [int(r[1]) for r in t],
                        [clip(r[2]) for r in t], digest.hexdigest()))
        log.append((float(np.random.random()), float(torch.rand(1))))
        return log

    plain = trace(base, float)
    muzero.REWARD_UNIT = 10.0
    grown = trace({**rg, "REWARD_GROWTH_CEILING": 1.0},
                  lambda raw: muzero.grown_reward(raw, 1.0))
    assert plain == grown
    assert any(r > 0 for life in plain[1:3] for r in life[3])   # it scored


def _slice(arm, ceiling=None, steps_after=0):
    """A sequential slice on real Ms. Pac-Man: two lives, birth search,
    three training steps, a communal refresh."""
    with ablation.arm_config(arm), ablation._run_length_globals(2, 3):
        if ceiling is not None:
            muzero.REWARD_GROWTH_CEILING = ceiling
        muzero.ACT_SEARCH_SIMS = 6
        muzero.TRAINING_BATCHES = 8
        with ablation.token_actions():
            run = ablation.Run(arm, 0, ablation.ARMS[arm])
            try:
                for _ in range(2):
                    run.apply_reward_ceiling()
                    traj, _, _, term = run.play_episode(cap=150)
                    run.replay.append(run.reanalyse(traj, term))
                ablation.train_block(run, 3)
                run.trainer.training_steps += steps_after
                run.apply_reward_ceiling()
                run.refresh_targets_vtree(1)
                nets = {n: {k: v.clone() for k, v in
                            getattr(run.trainer, n).state_dict().items()}
                        for n in ablation.NET_NAMES}
                stored = [[(t["action"], t["reward"], t["root_value"],
                            t["priority"], t["target_policy"].tolist())
                           for t in tr] for tr in run.replay]
                raw = [[t.get("raw_reward") for t in tr] for tr in run.replay]
                return (nets, stored, run.env_steps, float(torch.rand(1)),
                        raw, getattr(run, "_reward_ceiling", None))
            finally:
                run.env.close()


@pytest.fixture
def mspacman_slice(monkeypatch):
    keys = set(_arm(TK_ARM)) | set(SCHEDULE) | {
        "ACT_SEARCH_SIMS", "TRAINING_BATCHES", "TOKEN_VOCAB"}
    _pin_globals(monkeypatch, keys)
    ablation.ENV_NAME, ablation.FRAMESKIP, ablation.ACTIONS = ENV_ID, 4, 9
    ablation.DEVICE = torch.device("cpu")
    torch.set_num_threads(2)


@pytest.mark.slow
def test_ceiling_1_reproduces_a_plain_training_step(mspacman_slice,
                                                    restore_rngs):  # noqa: F811
    """Play, birth search, training, a refresh: the growth arm held at
    ceiling 1 equals the twin's arm -- every net's weights, the stored
    rewards, targets and priorities, env steps, the RNG."""
    plain = _slice(MSP_ARM)
    grown = _slice(RG_ARM, ceiling=1.0)
    assert plain[1:4] == grown[1:4]
    for n in ablation.NET_NAMES:
        for k, v in plain[0][n].items():
            assert torch.equal(v, grown[0][n][k]), (n, k)
    assert any(r for tr in plain[1] for (_, r, *_rest) in tr)     # scored
    assert all(r is None for tr in plain[4] for r in tr)
    assert all(isinstance(r, tuple) for tr in grown[4] for r in tr)


@pytest.mark.slow
def test_the_growth_arm_reclips_its_buffer_in_a_real_slice(
        mspacman_slice, restore_rngs):  # noqa: F811
    """The arm's own schedule, past its end (6 planned steps; + 100): every
    stored reward is its raw tuple clipped at 160."""
    _, stored, _, _, raw, c = _slice(RG_ARM, steps_after=100)
    assert c == 160.0
    muzero.REWARD_UNIT = 10.0
    for tr, rw in zip(stored, raw):
        for rec, r in zip(tr, rw):
            assert rec[1] == muzero.grown_reward(r, 160.0)


@pytest.mark.slow
def test_the_token_arm_plays_36_tokens_on_real_mspacman(
        monkeypatch, restore_rngs):  # noqa: F811
    """The TK arm's overrides reach the actor: tokens in 0..35, each record's
    raw tuple one reward per base step of the token played before it (a
    terminal can cut the last), base steps = frames / 4."""
    import ray_actors
    tk = _arm(TK_ARM)
    _pin_globals(monkeypatch, list(tk) + list(SCHEDULE))
    torch.set_num_threads(2)
    actor = ray_actors.SelfPlayActor(tk, 36, 0, SCHEDULE, ENV_ID, 4, 200,
                                     server=None, force_cpu=True)
    assert actor.get_constant("TOKEN_VOCAB") == tk["TOKEN_VOCAB"]
    assert actor.get_constant("REWARD_GROWTH") is True
    traj, term, steps = actor.play(3)
    acts = [int(r[1]) for r in traj[:-1]]
    assert set(acts) <= set(range(36)) and len(set(acts)) > 3
    assert traj[0][2] == ()
    n = [tk["TOKEN_VOCAB"][a][1] // 4 for a in acts]
    lens = [len(r[2]) for r in traj[1:]]
    assert lens[:-1] == n[:-1] and 1 <= lens[-1] <= n[-1]
    assert steps == sum(lens)


# --------------------------------------------------------------------------
# the supervisor
# --------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parents[1]
SUPERVISOR = ROOT / "benchmarks" / "supervisors" / "supervise_rgrow.py"


def _load(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def sup():
    return _load(SUPERVISOR, "supervise_rgrow")


def test_supervisor_recipe_is_the_twins_with_the_growth_arms(sup):
    assert sup.VARIANTS == {"rg": RG_ARM, "tk": TK_ARM}
    assert sup.SEEDS == {0: "2026-10-19"}
    twin = _load(ROOT / "benchmarks" / "supervisors" / "supervise_mspacman.py",
                 "supervise_mspacman")
    assert twin.VARIANTS["temp"] == MSP_ARM
    for v, arm in sup.VARIANTS.items():
        assert sup.command(v, 0)[1:] == twin.command(arm, 0)[1:]
        assert sup.command(v, 0)[1:5] == ["experiments/ablation.py", "--arms",
                                          arm, "--seed"]
    rg, tk = sup.paths("rg", 0), sup.paths("tk", 0)
    assert rg[0] == "benchmarks/results/mspacman-rgrow-s0-2026-10-19"
    assert tk[0] == "benchmarks/results/mspacman-tokens-s0-2026-10-19"
    for (rel, base, jsonl, ckpt, deaths), arm in ((rg, RG_ARM), (tk, TK_ARM)):
        assert jsonl.endswith("ablation-mspacmannoframeskip-skip4.jsonl")
        assert ckpt.endswith(
            f"resume-mspacmannoframeskip-skip4-{arm}-seed0.ckpt")
        assert pathlib.Path(base).parent == ROOT / "benchmarks" / "results"
    assert pathlib.Path(sup.REPO) == ROOT          # runs from its own tree


def test_supervisor_pins_what_the_runs_and_preflight_use(sup):
    import re
    for rel in sup.AUDITED:
        assert (ROOT / rel).exists(), rel
    assert set(sup.SUITES) <= set(sup.AUDITED)
    for rel in ("muzero.py", "muzero_vtree.py", "networks.py",
                "EpisodicLifeEnv.py", "atari_wrappers.py",
                "experiments/ablation.py", "experiments/ray_actors.py",
                "benchmarks/probes/probe_exhaustive_eval.py",
                "benchmarks/supervisors/supervise_rgrow.py",
                "tests/test_rgrow.py", "tests/test_tokens.py",
                "tests/test_mspacman.py"):
        assert rel in sup.AUDITED, rel
    # every tests.* module a suite imports, every supervisor a suite reads
    for rel in sup.SUITES:
        src = (ROOT / rel).read_text(encoding="utf-8")
        for mod in re.findall(r"from tests\.(\w+) import", src):
            assert f"tests/{mod}.py" in sup.AUDITED, (rel, mod)
        for name in re.findall(r"(supervise_\w+)\.py", src):
            assert f"benchmarks/supervisors/{name}.py" in sup.AUDITED, \
                (rel, name)


def test_supervisor_manifest_catches_an_edit(sup, monkeypatch, tmp_path):
    import json
    import shutil
    for rel in sup.AUDITED:
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, dst)
    monkeypatch.setattr(sup, "REPO", str(tmp_path))
    monkeypatch.setattr(sup, "MANIFEST", str(tmp_path / "m.json"))
    (tmp_path / "m.json").write_text(json.dumps(sup.file_hashes(str(tmp_path))))
    assert sup.manifest_check() == (True, "")
    with open(tmp_path / "experiments" / "ablation.py", "a",
              encoding="utf-8") as f:
        f.write("\n# edit\n")
    ok, why = sup.manifest_check()
    assert not ok and "experiments/ablation.py" in why
    p = tmp_path / "experiments" / "ablation.py"       # CRLF vs LF: same
    shutil.copyfile(ROOT / "experiments" / "ablation.py", p)
    p.write_bytes(p.read_bytes().replace(b"\r\n", b"\n"))
    assert sup.manifest_check() == (True, "")


def test_supervisor_write_manifest_refuses_a_dirty_tree(sup, monkeypatch,
                                                         tmp_path):
    monkeypatch.setattr(sup, "MANIFEST", str(tmp_path / "m.json"))
    monkeypatch.setattr(sup.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout=" M muzero.py\n", stderr=""))
    assert sup.main(["--write-manifest"]) == 1
    assert not (tmp_path / "m.json").exists()


def test_supervisor_refuses_bad_arguments(sup, capsys):
    for argv in ([], ["rg"], ["0"], ["rg", "1"], ["xx", "0"], ["rg", "x"],
                 ["rg", "0", "1"], ["0", "rg"]):
        assert sup.main(argv) == 2, argv
    assert "usage" in capsys.readouterr().out


def test_supervisor_finds_live_children_of_its_own_arm_and_seed(
        sup, monkeypatch):
    procs = [
        {"pid": 1, "cmdline": ["py", r"Y:\run\experiments\ablation.py",
                               "--arms", RG_ARM, "--seed", "0"]},
        {"pid": 2, "cmdline": ["py", "experiments/ablation.py", "--arms",
                               TK_ARM, "--seed", "0"]},
        {"pid": 3, "cmdline": ["py", "experiments/ablation.py", "--arms",
                               RG_ARM, "--seed", "1"]},
        {"pid": 4, "cmdline": ["py", "experiments/ablation.py", "--arms",
                               MSP_ARM, "--seed", "0"]},
        {"pid": 5, "cmdline": None}]
    import psutil
    monkeypatch.setattr(psutil, "process_iter", lambda attrs: [
        SimpleNamespace(info=p) for p in procs])
    assert sup.live_children("rg", 0) == [1]
    assert sup.live_children("tk", 0) == [2]


@pytest.mark.parametrize("variant", ["rg", "tk"])
def test_supervisor_quick_probe_passes(sup, variant):
    import subprocess
    r = subprocess.run([sys.executable, "-c", sup.quick(variant)], cwd=ROOT,
                       capture_output=True, text=True,
                       env={**__import__("os").environ,
                            "CUDA_VISIBLE_DEVICES": "-1"})
    assert r.returncode == 0, r.stderr[-2000:]


def test_supervisor_rows_print_the_growth_fields(sup, tmp_path, capsys):
    import json
    jsonl = tmp_path / "a.jsonl"
    rows = sup.Rows(str(jsonl), "tk", 0)
    jsonl.write_text(json.dumps({
        "iteration": 5, "return_mean": 500.0, "reward_ceiling": 1.8,
        "reward_kept": 0.42, "reward_max": 3.6,
        "tok_frames_per_decision": 9.5, "elapsed_s": 100.0}) + "\n")
    rows.emit()
    out = capsys.readouterr().out
    assert out.startswith("EVAL tk s0 iter 5")
    assert "ceiling 1.8  kept 0.42  max 3.6" in out
    assert "frames/decision 9.5" in out
