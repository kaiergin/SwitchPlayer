"""Policy-only acting on the paper data path (BIRTH_SEARCH,
PRIOR_ACT_STAGED_TEMP), ReDo (redo.py), and the
effzero-repro-faithful-prioract arm (2026-09-26).

Every flag defaults off; these tests pin the arm's exact delta, the
birth search's act_search contract (online nets, eval mode, root noise on
acted steps, greedy truncated tail), the staged prior temperature, and
ReDo's recycling arithmetic (dead units found, outputs unmoved, optimizer
state zeroed, live BN statistics and global RNG untouched)."""
import asyncio
import sys

import numpy as np
import pytest
import torch

import muzero
from tests.test_checkpoint_resume import make_run
from tests.test_paper_mode import ACTIONS, make_trainer
from tests.test_reanalyze_refresh import obs_traj

sys.path.insert(0, str(__import__("pathlib").Path(
    __file__).resolve().parent.parent / "experiments"))
import ablation
import redo

ARM = "effzero-repro-faithful-prioract"
FAITHFUL = "effzero-repro-refresh5x-communal-edge-faithful"
DELTA = {"REFRESH_VTREE", "VTREE_COMMUNAL_TRAJS", "USE_SEARCH_ACTING",
         "BIRTH_SEARCH", "PRIOR_ACT_STAGED_TEMP", "REDO_EVERY", "REDO_TAU",
         "REDO_NETS"}


# --------------------------------------------------------------------------
# The arm and the gates
# --------------------------------------------------------------------------
def test_arm_is_faithful_plus_exactly_the_new_flags():
    base, arm = ablation.ARMS[FAITHFUL], ablation.ARMS[ARM]
    delta = {k for k in set(base) | set(arm)
             if k != "_doc" and base.get(k, "<absent>") != arm.get(k, "<absent>")}
    assert delta == DELTA
    assert arm["REFRESH_VTREE"] is True and arm["VTREE_COMMUNAL_TRAJS"] == 8
    assert arm["USE_SEARCH_ACTING"] is False and arm["BIRTH_SEARCH"] is True
    assert arm["PRIOR_ACT_STAGED_TEMP"] is True
    assert arm["REDO_EVERY"] == 1000 and arm["REDO_TAU"] == 0.025
    # The communal topology and every faithful flag survive.
    assert arm["USE_COMMUNAL_REANALYSIS"] and arm["COMMUNAL_EDGE_VISITS"]
    assert arm["LATENT_RENORMALIZE"] is False and arm["OPTIMIZER"] == "sgd"
    for key in arm:
        if not key.startswith("_"):
            assert hasattr(muzero, key), key


def test_new_flags_default_off():
    assert muzero.BIRTH_SEARCH is False
    assert muzero.PRIOR_ACT_STAGED_TEMP is False
    assert muzero.REDO_EVERY == 0


def test_birth_search_gate(monkeypatch):
    ablation.birth_search_gate()                        # off: no checks
    with ablation.arm_config(ARM):
        ablation.birth_search_gate()                    # the arm is legal
    monkeypatch.setattr(muzero, "BIRTH_SEARCH", True)
    for key, bad in (("PAPER_TARGETS", False), ("USE_SEARCH_ACTING", True),
                     ("USE_GUMBEL_SEARCH", True)):
        with monkeypatch.context() as m:
            m.setattr(muzero, "PAPER_TARGETS", True)
            m.setattr(muzero, "USE_SEARCH_ACTING", False)
            m.setattr(muzero, "USE_GUMBEL_SEARCH", False)
            m.setattr(muzero, key, bad)
            with pytest.raises(SystemExit):
                ablation.birth_search_gate()


# --------------------------------------------------------------------------
# Prior acting temperature
# --------------------------------------------------------------------------
def test_prior_acting_temperature_follows_the_staged_schedule(monkeypatch):
    monkeypatch.setattr(muzero, "TEMPERATURE_TOTAL_STEPS", 1000)
    monkeypatch.setattr(muzero, "ACTING_TEMPERATURE", 1.0)
    assert muzero.prior_acting_temperature(900) == 1.0       # flag off
    monkeypatch.setattr(muzero, "PRIOR_ACT_STAGED_TEMP", True)
    assert muzero.prior_acting_temperature(100) == 1.0
    assert muzero.prior_acting_temperature(600) == 0.5
    assert muzero.prior_acting_temperature(900) == 0.25


class _OneStepEnv:
    """Minimal env: one decision, then a real terminal."""
    def reset(self, seed=None):
        return np.zeros((210, 160, 3), dtype=np.uint8), {}

    def step(self, action):
        return np.zeros((210, 160, 3), dtype=np.uint8), 0.0, True, False, {}


def test_play_episode_samples_the_prior_at_the_staged_temperature(monkeypatch):
    prior = np.array([0.4, 0.3, 0.2, 0.05, 0.05])
    nm = type("NM", (), {})()
    nm.training_steps = 900
    nm.initial_inference = lambda stack: (prior.copy(), 0.0)
    run = object.__new__(ablation.Run)
    run.env, run.env_steps = _OneStepEnv(), 0
    run.searcher = type("S", (), {"network_manager": nm})()
    seen = []
    real_choice = np.random.choice

    def spy(n, p=None):
        seen.append(np.asarray(p))
        return real_choice(n, p=p)

    monkeypatch.setattr(ablation, "ACTIONS", 5)
    monkeypatch.setattr(muzero, "USE_SEARCH_ACTING", False)
    monkeypatch.setattr(muzero, "TEMPERATURE_TOTAL_STEPS", 1000)
    monkeypatch.setattr(muzero, "PRIOR_ACT_STAGED_TEMP", True)
    monkeypatch.setattr(np.random, "choice", spy)
    run.play_episode(greedy=False, cap=5)
    want = prior ** 4.0                                  # T = 0.25 at 90%
    assert np.allclose(seen[0], want / want.sum())


# --------------------------------------------------------------------------
# Birth search
# --------------------------------------------------------------------------
def _policy_only_traj(n_acted):
    """3-field records as play_episode writes them without search."""
    return [(np.full((12, *muzero.IMAGE_RESHAPE), (i * 7) % 250,
                     dtype=np.uint8), i % ACTIONS, 0.0)
            for i in range(n_acted + 1)]                # distinct per step


class _SpyEngine:
    calls = []

    def __init__(self, nets, action_space, device):
        self.nets = nets
        _SpyEngine.calls.append({"nets": nets, "ctor_modes":
                                 {k: n.training for k, n in nets.items()}})

    def search(self, obs, num_simulations, c_puct, discount, root_noise):
        _SpyEngine.calls.append({
            "n": len(obs), "sims": num_simulations, "noise": root_noise,
            "c_puct": c_puct, "discount": discount,
            "modes": {k: n.training for k, n in self.nets.items()}})
        # Row-identifying outputs: policy one-hot on (obs mean % A), value =
        # obs mean, so each record's landing slot is checkable.
        keys = [int(o.mean()) for o in obs]
        pols = np.stack([np.eye(ACTIONS)[k % ACTIONS] for k in keys])
        return pols, np.array(keys, dtype=np.float64), None


def test_birth_search_is_the_act_search_contract_on_online_nets(monkeypatch):
    import muzero_vtree
    monkeypatch.setattr(ablation, "ACTIONS", ACTIONS)
    monkeypatch.setattr(muzero, "ACT_SEARCH_SIMS", 7)
    monkeypatch.setattr(muzero, "VTREE_REFRESH_BATCH", 3)
    monkeypatch.setattr(muzero_vtree, "VectorizedPUCT", _SpyEngine)
    run = make_run()
    run.trainer._ensure_target_model()
    for net in (run.trainer.representation, run.trainer.prediction):
        net.train()                                     # must be restored
    for terminated in (False, True):
        _SpyEngine.calls = []
        traj = _policy_only_traj(7)
        out = run._birth_search(traj, terminated)
        ctor = _SpyEngine.calls[0]
        assert ctor["nets"]["representation"] is run.trainer.representation
        assert ctor["nets"]["dynamics"] is run.trainer.dynamics
        assert ctor["nets"]["prediction"] is run.trainer.prediction
        assert ctor["nets"]["value_prefix"] is run.trainer.value_prefix
        searches = _SpyEngine.calls[1:]
        assert all(not any(c["modes"].values()) for c in searches)
        assert all(c["sims"] == 7 for c in searches)
        assert all(c["c_puct"] == muzero.C_PUCT
                   and c["discount"] == muzero.DISCOUNT for c in searches)
        acted = [c for c in searches if c["noise"]]
        assert sum(c["n"] for c in acted) == 7           # every acted step
        assert max(c["n"] for c in acted) <= 3           # batched waves
        tail = [c for c in searches if not c["noise"]]
        if terminated:
            assert not tail
            assert np.allclose(out[-1][3], 1.0 / ACTIONS) and out[-1][4] == 0.0
        else:
            assert len(tail) == 1 and tail[0]["n"] == 1  # greedy tail
        assert all(len(r) == 5 for r in out)
        searched = len(traj) - (1 if terminated else 0)
        for i in range(searched):                       # each row in its slot
            k = int(traj[i][0].mean())
            assert out[i][4] == k
            assert np.array_equal(out[i][3], np.eye(ACTIONS)[k % ACTIONS])
        assert [r[0] is t[0] for r, t in zip(out, traj)] == [True] * len(traj)
    assert run.trainer.representation.training and run.trainer.prediction.training
    assert run._birth_stats["birth_positions_last"] == 7


def test_paper_targets_builds_real_records_from_a_policy_only_trajectory(monkeypatch):
    monkeypatch.setattr(ablation, "ACTIONS", ACTIONS)
    monkeypatch.setattr(muzero, "BIRTH_SEARCH", True)
    monkeypatch.setattr(muzero, "USE_GUMBEL_SEARCH", False)   # paper arms: PUCT
    monkeypatch.setattr(muzero, "ACT_SEARCH_SIMS", 4)
    monkeypatch.setattr(muzero, "MAX_DEPTH", 4)
    np.random.seed(0)
    torch.manual_seed(0)
    run = make_run()
    run.trainer.training_steps = 17
    out = run.paper_targets(_policy_only_traj(4), terminated=True)
    assert len(out) == 5
    assert [t["has_policy_target"] for t in out] == [True] * 4 + [False]
    for t in out[:4]:
        assert t["target_policy"].sum() == pytest.approx(1.0)
        assert np.isfinite(t["root_value"]) and np.isfinite(t["priority"])
        assert t["born_step"] == 17
    assert out[-1]["truncated"] is False
    assert run._fresh_target_H is not None
    # Behaviour-vs-search diagnostics, recomputed by hand from the records
    # and the priors the birth search recorded.
    acted = out[:-1]
    agree = np.mean([int(np.argmax(t["target_policy"])) == t["action"]
                     for t in acted])
    prob = np.mean([t["target_policy"][t["action"]] for t in acted])
    q = run._birth_priors
    assert q.shape == (4, ACTIONS)
    prior_agree = np.mean([int(np.argmax(qi)) == int(np.argmax(t["target_policy"]))
                           for qi, t in zip(q, acted)])
    row = {}
    run._merge_row_stats(row)
    assert row["beh_act_agree"] == pytest.approx(agree, abs=1e-4)
    assert row["beh_act_prob"] == pytest.approx(prob, abs=1e-4)
    assert row["beh_prior_agree"] == pytest.approx(prior_agree, abs=1e-4)
    assert row["beh_T"] == 1.0 and sum(row["beh_action_counts"]) == 4
    assert row["beh_T_values"] == [1.0]
    qc = np.clip(q.astype(np.float64), 1e-12, None)
    kl = np.mean([np.sum(t["target_policy"] * (np.log(np.clip(
        t["target_policy"], 1e-12, None)) - np.log(qi)))
        for qi, t in zip(qc, acted)])
    assert row["beh_kl_target_prior"] == pytest.approx(kl, abs=1e-4)
    assert row["beh_prior_maxp"] == pytest.approx(qc.max(1).mean(), abs=1e-4)
    assert run._beh_window is None                       # window restarts
    # Without the flag, a policy-only trajectory is still refused loudly.
    monkeypatch.setattr(muzero, "BIRTH_SEARCH", False)
    with pytest.raises(ValueError, match="USE_SEARCH_ACTING"):
        run.paper_targets(_policy_only_traj(2), terminated=True)


# --------------------------------------------------------------------------
# ReDo
# --------------------------------------------------------------------------
def _np_state_equal(a, b):
    """Full MT19937 state: the key array AND the position (a few draws move
    only the position -- comparing the keys alone misses them)."""
    return (a[0] == b[0] and np.array_equal(a[1], b[1])
            and tuple(a[2:]) == tuple(b[2:]))


def _nets(t):
    return {n: getattr(t, n) for n in redo.ALL_NETS}


def _arm_nets():
    """The five nets and SGD optimizers exactly as the arm builds them."""
    with ablation.arm_config(ARM):
        nets = ablation.build_networks(ablation.ARMS[ARM])
        opts = [muzero.make_optimizer(nets[n].parameters())
                for n in redo.ALL_NETS]
    return nets, opts


def test_recyclable_units_are_single_consumer_layers_only():
    units = redo.recyclable_units(_nets(make_trainer()))
    prods = {(u.net, u.producer) for u in units}
    for must in (("representation", "conv1"), ("representation", "conv3"),
                 ("representation", "block1.conv1"),
                 ("prediction", "block.conv1"), ("prediction", "policy_conv"),
                 ("prediction", "value_conv"), ("value_prefix", "conv"),
                 ("projector", "projection.0"), ("projector", "projection.3"),
                 ("projector", "prediction.0")):
        assert must in prods, must
    for never in (("representation", "conv2"), ("representation", "conv4"),
                  ("dynamics", "conv1"), ("projector", "projection.6"),
                  ("prediction", "policy_fc"), ("dynamics", "reward_conv"),
                  ("value_prefix", "fc")):
        assert never not in prods, never
    assert not any(p.endswith(".conv2") for _, p in prods)   # residual adds
    t = make_trainer()
    assert sum(1 for u in units if u.net == "dynamics") == len(t.dynamics.blocks)
    vp = [u for u in units if u.net == "value_prefix" and u.producer == "conv"][0]
    assert vp.consumer == "lstm" and vp.consumer_attr == "weight_ih_l0"
    # REDO_NETS scoping: the arm drops the projector entirely.
    arm_units = redo.recyclable_units(
        _nets(t), include=ablation.ARMS[ARM]["REDO_NETS"])
    assert "projector" not in {u.net for u in arm_units}
    assert "projector" not in ablation.ARMS[ARM]["REDO_NETS"]
    assert "value_prefix" in {u.net for u in arm_units}


def _kill(nets):
    """Make specific units exactly dead (post-ReLU output 0 everywhere)."""
    with torch.no_grad():
        rep = nets["representation"]
        rep.conv1.weight[:3] = 0.0
        rep.bn1.weight[:3] = 0.0
        rep.bn1.bias[:3] = -1.0
        pred = nets["prediction"]
        pred.policy_conv.weight[1] = 0.0
        pred.policy_conv.bias[1] = -1.0
        pred.policy_fc.weight.normal_()                  # zero-init otherwise
        blk = nets["dynamics"].blocks[0]
        blk.conv1.weight[4] = 0.0
        blk.bn1.weight[4] = 0.0
        blk.bn1.bias[4] = -1.0
        vp = nets["value_prefix"]
        vp.conv.weight[2] = 0.0
        vp.norm_in.weight[2] = 0.0
        vp.norm_in.bias[2] = -1.0
        nets["projector"].projection[0].weight[5] = 0.0
        nets["projector"].projection[1].weight[5] = 0.0
        nets["projector"].projection[1].bias[5] = -1.0


def _outputs(nets, obs, acts, train):
    for n in nets.values():
        n.train(train)
    with torch.no_grad():
        lat = nets["representation"](obs)
        outs = list(nets["prediction"](lat)[::2])
        state = nets["value_prefix"].initial_state(obs.size(0), obs.device)
        for k in range(acts.shape[1]):
            lat, _, rl = nets["dynamics"](lat, acts[:, k])
            p, _, v = nets["prediction"](lat)
            _, pl, state = nets["value_prefix"](lat, state)
            outs += [lat, rl, p, v, pl, state[0],
                     nets["projector"](lat, with_prediction=True),
                     nets["projector"](lat, with_prediction=False)]
    return outs


@pytest.mark.parametrize("train", [False, True])
def test_redo_on_arm_nets_recycles_dead_units_without_moving_any_output(train):
    torch.manual_seed(0)
    nets, opts = _arm_nets()
    for net, opt in zip(nets.values(), opts):
        for p in net.parameters():
            p.grad = 1e-3 * torch.randn_like(p)          # realistic-size step
        opt.step()                                       # momentum buffers
    _kill(nets)                                          # after the step
    rep = nets["representation"]
    healthy_g = float(rep.bn1.weight[3:].abs().mean())
    healthy_b = float(rep.bn1.bias[3:].mean())
    mom_conv1 = opts[0].state[rep.conv1.weight]["momentum_buffer"].clone()
    w_conv1 = rep.conv1.weight.detach().clone()
    obs = torch.rand(32, 12, *muzero.IMAGE_RESHAPE)
    acts = torch.randint(0, ablation.ACTIONS, (32, muzero.UNROLL_STEPS))
    ref = _outputs(nets, obs, acts, train)
    units = redo.recyclable_units(nets)                  # all five nets
    scores = redo.dormancy_scores(nets, obs, acts, units)
    # tau 0: only units EXACTLY dead on the scored sample, for which
    # recycling is output-neutral on that sample (and on any data where they
    # stay dead -- true here by construction). At tau > 0 near-dormant but
    # live units are recycled too and outputs move slightly -- by design.
    plan = redo.recycle(nets, opts, scores, 0.0,
                        torch.Generator().manual_seed(1))
    by = {(u.net, u.producer): v for u, v in plan.items()}
    # BN state right after recycling (a train-mode forward below would
    # legitimately move the running statistics).
    rm, rv = rep.bn1.running_mean.clone(), rep.bn1.running_var.clone()
    assert {0, 1, 2} <= set(by[("representation", "conv1")])
    assert 1 in by[("prediction", "policy_conv")]
    assert 4 in by[("dynamics", "blocks.0.conv1")]
    assert 2 in by[("value_prefix", "conv")]
    assert 5 in by[("projector", "projection.0")]
    after = _outputs(nets, obs, acts, train)
    for a, b in zip(ref, after):
        assert torch.allclose(a, b, atol=1e-6)
    bound = 1.0 / np.sqrt(rep.conv1.weight[0].numel())
    for c in range(3):
        w = rep.conv1.weight[c]
        assert w.abs().max() > 0 and w.abs().max() <= bound
        # Matched to the survivors' mean |gamma| / mean beta, not the
        # identity (round-1/2 audits).
        assert float(rep.bn1.weight[c]) == pytest.approx(
            max(healthy_g, redo.GAMMA_FLOOR), rel=1e-5)
        assert float(rep.bn1.bias[c]) == pytest.approx(healthy_b, abs=1e-6)
        assert rm[c] == 0 and rv[c] == 1
        assert torch.all(rep.conv2.weight[:, c] == 0)
    mom = opts[0].state[rep.conv1.weight]["momentum_buffer"]
    assert torch.all(mom[:3] == 0)
    untouched = [c for c in range(rep.conv1.weight.shape[0])
                 if c not in by[("representation", "conv1")]]
    assert torch.equal(mom[untouched], mom_conv1[untouched])
    assert torch.equal(rep.conv1.weight[untouched], w_conv1[untouched])
    pred = nets["prediction"]
    hw = pred.pool[0] * pred.pool[1]
    assert torch.all(pred.policy_fc.weight[:, hw:2 * hw] == 0)  # NCHW block
    assert torch.all(pred.policy_fc.weight[:, :hw] != 0)        # neighbour
    vp = nets["value_prefix"]
    vhw = vp.pool[0] * vp.pool[1]
    assert torch.all(vp.lstm.weight_ih_l0[:, 2 * vhw:3 * vhw] == 0)  # 4 gates
    assert torch.all(nets["dynamics"].blocks[0].conv2.weight[:, 4] == 0)
    assert torch.all(nets["projector"].projection[3].weight[:, 5] == 0)


def test_recycle_zeroes_outgoing_after_all_incoming_reinits():
    """Two consecutive recyclable layers (conv -> hidden FC): a recycled
    consumer row must not refill a recycled producer's zeroed column."""
    from networks import PredictionNetwork
    torch.manual_seed(0)
    pred = PredictionNetwork(latent_channels=16, action_space=4, fc_hidden=8)
    hw = pred.pool[0] * pred.pool[1]
    u_conv = redo.Unit("prediction", "policy_conv", None, "policy_hidden", hw)
    u_fc = redo.Unit("prediction", "policy_hidden", None, "policy_fc", 1)
    scores = {u_conv: torch.tensor([1.0] * 2 + [0.0] + [1.0] * 29),
              u_fc: torch.tensor([0.0] + [1.0] * 7)}
    redo.recycle({"prediction": pred}, [], scores, 0.025,
                 torch.Generator().manual_seed(0))
    assert torch.all(pred.policy_hidden.weight[0, 2 * hw:3 * hw] == 0)
    assert pred.policy_hidden.weight[0, :2 * hw].abs().sum() > 0


def test_scoring_leaves_live_bn_statistics_and_global_rng_untouched():
    t = make_trainer()
    bufs = {n: {k: v.clone() for k, v in net.state_dict().items()}
            for n, net in _nets(t).items()}
    obs = torch.rand(8, 12, *muzero.IMAGE_RESHAPE)
    acts = torch.randint(0, ACTIONS, (8, muzero.UNROLL_STEPS))
    t_state, np_state = torch.get_rng_state(), np.random.get_state()
    redo.dormancy_scores(_nets(t), obs, acts, redo.recyclable_units(_nets(t)))
    for n, net in _nets(t).items():
        for k, v in net.state_dict().items():
            assert torch.equal(v, bufs[n][k]), (n, k)
    assert torch.equal(torch.get_rng_state(), t_state)
    assert _np_state_equal(np.random.get_state(), np_state)


def test_redo_step_samples_only_real_windows_and_is_rng_neutral(monkeypatch):
    K = muzero.UNROLL_STEPS
    run = make_run()
    run.seed = 0
    run.trainer.training_steps = 3000
    long_traj = obs_traj([0.0] * (K + 3), [0.0] * (K + 3))  # starts 0..2
    short = obs_traj([0.0] * K, [0.0] * K)                   # no full window
    for i, rec in enumerate(long_traj):
        rec["obs"] = np.full_like(rec["obs"], i)
        rec["action"] = i % ACTIONS
    run.replay = [short, long_traj]
    seen = {}

    def spy(nets, obs, acts, units):
        seen["obs"], seen["acts"] = obs, acts
        return {}, {}, {}

    monkeypatch.setattr(redo, "dormancy_stats", spy)
    monkeypatch.setattr(muzero, "REDO_BATCH", 64)
    t_state, np_state = torch.get_rng_state(), np.random.get_state()
    run.redo_step()
    starts = (seen["obs"][:, 0, 0, 0] * 255).round().long()
    assert set(starts.tolist()) <= {0, 1, 2}
    assert len(set(starts.tolist())) == 3                  # uniform, all hit
    for s, row in zip(starts.tolist(), seen["acts"].tolist()):
        assert row == [(s + j) % ACTIONS for j in range(K)]
    assert torch.equal(torch.get_rng_state(), t_state)
    assert _np_state_equal(np.random.get_state(), np_state)


def test_after_train_step_runs_redo_on_its_cadence(monkeypatch):
    run = make_run()
    run.replay = [obs_traj([0.0] * 3, [0.0] * 3)]
    calls = []
    monkeypatch.setattr(run, "redo_step", lambda: calls.append(
        run.trainer.training_steps))
    monkeypatch.setattr(muzero, "REDO_EVERY", 0)
    for s in range(1, 10):
        run.trainer.training_steps = s
        ablation.after_train_step(run)
    assert calls == []
    monkeypatch.setattr(muzero, "REDO_EVERY", 3)
    for s in range(1, 10):
        run.trainer.training_steps = s
        ablation.after_train_step(run)
    assert calls == [4, 7]                     # mid-period, after a full one
    # At the arm's settings no pass ever lands on an iteration boundary
    # (2000 steps: every eval row, checkpoint and the final step 180000).
    monkeypatch.setattr(muzero, "REDO_EVERY", 1000)
    due = [s for s in range(1, 180001) if ablation.redo_due(s)]
    assert due[0] == 1500 and due[-1] == 179500 and len(due) == 179
    assert all(s % 1000 == 500 for s in due)


# --------------------------------------------------------------------------
# Round-1 audit fixes
# --------------------------------------------------------------------------
class _FakeRay:
    @staticmethod
    def get(x):
        return x


class _FakeActor:
    """play.remote returns the episode itself (the fake ray.get is identity)."""
    def __init__(self, traj):
        self.play = type("P", (), {"remote": staticmethod(
            lambda seed: (traj, True, len(traj) - 1))})()


def _pipeline(run, traj):
    pipe = object.__new__(ablation.RayPipeline)
    pipe.run, pipe.server, pipe.actor = run, None, _FakeActor(traj)
    pipe.episodes_per_iter, pipe.seed, pipe.pending = 1, 0, None
    pipe._states_cache = (None, None)
    pipe._birth_states = pipe._birth_nets = None
    pipe.ray = _FakeRay()
    return pipe


def test_ray_birth_search_uses_the_weights_the_actor_played_with(monkeypatch):
    """HIGH-1: collect() runs after the next training block; the birth
    search must still see the request-time (actor-synced) weights."""
    monkeypatch.setattr(ablation, "ACTIONS", ACTIONS)
    monkeypatch.setattr(muzero, "PAPER_TARGETS", True)
    monkeypatch.setattr(muzero, "BIRTH_SEARCH", True)
    run = make_run()
    seen = {}

    def spy(traj, terminated, nets=None, prior_nets=None):
        seen["nets"] = nets
        assert prior_nets is None       # BN_RECAL_ACTING off: no acting nets
        seen["w"] = (nets or {}).get("representation",
                                     run.trainer.representation).conv1.weight.detach().clone()
        return [(o, a, r, np.full(ACTIONS, 1.0 / ACTIONS), 0.0)
                for (o, a, r) in traj]

    monkeypatch.setattr(run, "_birth_search", spy)
    pipe = _pipeline(run, _policy_only_traj(3))
    w_request = run.trainer.representation.conv1.weight.detach().clone()
    pipe.request(1)
    with torch.no_grad():                                 # "training" happens
        run.trainer.representation.conv1.weight.add_(1.0)
    run.trainer.training_steps += 4
    pipe.collect()
    assert seen["nets"] is not None
    assert seen["nets"]["representation"] is not run.trainer.representation
    assert torch.equal(seen["w"].cpu(), w_request.cpu())
    assert not any(n.training for n in seen["nets"].values())
    assert run.replay and run.replay[-1][0]["born_step"] == 0
    # Without BIRTH_SEARCH nothing is captured: live-net semantics.
    monkeypatch.setattr(muzero, "BIRTH_SEARCH", False)
    pipe.request(2)
    assert pipe._birth_states is None


def test_birth_and_redo_counters_survive_a_checkpoint(tmp_path):
    from tests.test_checkpoint_resume import EXPECT, META
    a = make_run()
    a.trainer.training_steps = META["training_steps"]
    a._redo_stats = {"redo_recycled_total": 42, "redo_recycled_window": 3}
    a._birth_stats = {"birth_positions_total": 900}
    path = tmp_path / "ck.ckpt"
    ablation.save_run_checkpoint(a, path, META)
    b = make_run()
    ablation.load_run_checkpoint(b, path, EXPECT)
    assert b._redo_stats == a._redo_stats and b._birth_stats == a._birth_stats
    # An arm with neither writes no extra head key (other arms unchanged).
    c = make_run()
    c.trainer.training_steps = META["training_steps"]
    path2 = tmp_path / "plain.ckpt"
    ablation.save_run_checkpoint(c, path2, META)
    assert "diag_stats" not in ablation._read_checkpoint(path2)


def test_row_windows_restart_after_each_row():
    run = make_run()
    run._redo_stats = {"redo_recycled_total": 10, "redo_recycled_window": 4,
                       "redo_recycled_last": 1}
    run._birth_stats = {"birth_positions_total": 50,
                        "birth_positions_window": 20}
    row = {}
    run._merge_row_stats(row)
    assert row["redo_recycled_window"] == 4 and row["birth_positions_window"] == 20
    assert run._redo_stats["redo_recycled_window"] == 0
    assert run._birth_stats["birth_positions_window"] == 0
    assert run._redo_stats["redo_recycled_total"] == 10


# --------------------------------------------------------------------------
# Search parity at the ARM's production settings (round-1 fidelity LOW)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("noise", [False, True])
def test_birth_engine_matches_act_search_at_the_arms_settings(monkeypatch, noise):
    """tests/test_vtree.py pins parity at small depths/sims with
    renormalisation on; this pins it where the arm runs: 50 sims, MAX_DEPTH
    50, horizon 5, the arm's discount and c_puct, NO latent renormalisation,
    value prefix in the tree -- greedy, and with injected root noise."""
    from tests.test_vtree import (dezero_heads, engine_per_tree,
                                  sequential_search)
    from muzero_vtree import VectorizedPUCT
    monkeypatch.setattr(muzero.random, "choice", lambda seq: seq[0])
    with ablation.arm_config(ARM):
        assert muzero.LATENT_RENORMALIZE is False and muzero.MAX_DEPTH == 50
        torch.manual_seed(4)
        helper = muzero.MCTSHelper(action_space=6, device=torch.device("cpu"),
                                   network_ref=None, storage_ref=None)
        nm = helper.network_manager
        nets = {n: getattr(nm, n) for n in ("representation", "dynamics",
                                            "prediction", "value_prefix")}
        assert nets["representation"].renormalize_output is False
        dezero_heads(nets, seed=21)
        for n in nets.values():
            n.eval()
        engine = VectorizedPUCT(nets, 6, torch.device("cpu"))
        obs = np.random.default_rng(11).integers(
            0, 255, (3, 12, *muzero.IMAGE_RESHAPE), dtype=np.uint8)
        kw = dict(num_simulations=50, c_puct=muzero.C_PUCT,
                  discount=muzero.DISCOUNT, max_depth=muzero.MAX_DEPTH)
        if not noise:
            seq = [sequential_search(helper, obs[t], 50, discount=kw["discount"],
                                     c_puct=kw["c_puct"])
                   for t in range(len(obs))]
            counts, vals = engine_per_tree(engine, obs, root_noise=False, **kw)
            assert counts.tolist() == [c.tolist() for c, _ in seq]
            assert np.allclose(vals, [v for _, v in seq], rtol=0, atol=1e-9)
        else:
            draws = np.random.default_rng(7).dirichlet([0.3] * 6, size=len(obs))
            calls = {"i": 0}

            def fixed_noise(prior, epsilon=0.25, alpha=0.25):
                n = draws[calls["i"]]
                calls["i"] += 1
                return ((1 - epsilon) * np.asarray(prior, dtype=np.float64)
                        + epsilon * n)

            monkeypatch.setattr(muzero, "add_dirichlet_noise", fixed_noise)
            seq = [sequential_search(helper, obs[t], 50, root_noise=True,
                                     discount=kw["discount"], c_puct=kw["c_puct"])
                   for t in range(len(obs))]
            counts = []
            for t in range(len(obs)):
                _, _, c = engine.search(obs[t:t + 1], root_noise=True,
                                        noise_override=draws[t:t + 1],
                                        tie_break="first", **kw)
                counts.append(c[0].astype(int))
            assert np.array(counts).tolist() == [c.tolist() for c, _ in seq]



# --------------------------------------------------------------------------
# Round-2 audit: production wiring that had no failing test
# --------------------------------------------------------------------------
def test_redo_step_in_production_scoping_momentum_and_rng(monkeypatch):
    """Run.redo_step as the arm runs it: REDO_NETS excludes the projector
    (fix f), optimizer momentum of recycled entries is zeroed, and the
    torch / numpy / python global RNG streams are untouched."""
    import random
    monkeypatch.setattr(ablation, "ACTIONS", ACTIONS)
    monkeypatch.setattr(muzero, "OPTIMIZER", "sgd")
    monkeypatch.setattr(muzero, "LEARNING_RATE", 0.1)
    monkeypatch.setattr(muzero, "REDO_NETS", ablation.ARMS[ARM]["REDO_NETS"])
    monkeypatch.setattr(muzero, "REDO_TAU", 0.025)
    monkeypatch.setattr(muzero, "REDO_BATCH", 16)
    torch.manual_seed(0)
    run = make_run()
    run.seed, run.trainer.training_steps = 0, 1000
    run.replay = [obs_traj([0.0] * 9, [0.0] * 9) for _ in range(2)]
    for net_name, opt_name in zip(ablation.NET_NAMES, ablation.OPT_NAMES):
        for p in getattr(run.trainer, net_name).parameters():
            p.grad = 1e-3 * torch.randn_like(p)
        getattr(run.trainer, opt_name).step()
    _kill(_nets(run.trainer))
    proj_before = {k: v.clone() for k, v in run.trainer.projector.state_dict().items()}
    real_recycle, seen = redo.recycle, {}

    def wrapped(*args, **kw):                 # production passes the scoring
        seen.update(kw)                       # inputs (activity matching)
        return real_recycle(*args, **kw)

    monkeypatch.setattr(redo, "recycle", wrapped)
    states = (torch.get_rng_state(), np.random.get_state(), random.getstate())
    run.redo_step()
    assert seen.get("raw") is not None and seen.get("inputs")
    assert torch.equal(torch.get_rng_state(), states[0])
    assert _np_state_equal(np.random.get_state(), states[1])
    assert random.getstate() == states[2]
    plan = run._redo_last_plan
    assert plan and not any(u.net == "projector" for u in plan)
    assert any(u.net == "value_prefix" for u in plan)
    for k, v in run.trainer.projector.state_dict().items():
        assert torch.equal(v, proj_before[k]), k        # projector untouched
    rep = run.trainer.representation
    conv1 = [u for u in plan if (u.net, u.producer) == ("representation", "conv1")][0]
    mom = run.trainer.rep_optimizer.state[rep.conv1.weight]["momentum_buffer"]
    assert torch.all(mom[plan[conv1]] == 0)
    assert run._redo_stats["redo_last_by_net"].get("projector", 0) == 0


def test_evaluate_diagnostics_path_merges_and_resets_the_new_counters():
    """M2: the row path itself calls _merge_row_stats."""
    import inspect
    src = inspect.getsource(ablation.Run.evaluate_diagnostics)
    assert "self._merge_row_stats(errs)" in src


def test_both_training_loops_call_after_train_step_after_the_increment():
    """M1: ReDo would silently never run if a loop dropped the hook. Both
    loops in run_one must call after_train_step(run) immediately after
    advancing the step counter."""
    import ast
    import inspect
    import textwrap
    run_src = textwrap.dedent(inspect.getsource(ablation.run_one))
    # The iteration loop's block lives in train_block (2026-10-01); run_one
    # must still call it.
    assert "train_block(run, steps_this_iter" in run_src
    tree = ast.Module(body=[ast.parse(run_src), ast.parse(textwrap.dedent(
        inspect.getsource(ablation.train_block)))], type_ignores=[])
    hits = 0
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        for a, b in zip(body, body[1:]):
            inc = (isinstance(a, ast.AugAssign)
                   and ast.unparse(a.target) == "run.trainer.training_steps")
            call = (isinstance(b, ast.Expr)
                    and ast.unparse(b.value) == "after_train_step(run)")
            hits += int(inc and call)
    assert hits == 2


@pytest.mark.parametrize("key,value,pattern", [
    ("USE_SEARCH_ACTING", True, "replaces the acting search"),
    ("USE_GUMBEL_SEARCH", True, "PUCT-only"),
])
def test_run_one_refuses_an_illegal_birth_search_mix(monkeypatch, key, value,
                                                     pattern):
    """M3: the gate must be reachable from run_one (and fire before any
    file work -- out=None means nothing can be touched)."""
    spec = dict(ablation.ARMS[ARM])
    spec[key] = value
    monkeypatch.setitem(ablation.ARMS, "prioract-variant", spec)
    with pytest.raises(SystemExit, match=pattern):
        ablation.run_one("prioract-variant", 0, 1, 1, 1, 1, None)


def test_run_one_validates_redo_before_touching_state(monkeypatch):
    """LOW-3: a stale ReDo layer map fails at startup, before
    refuse_conflicting_state could delete anything under --overwrite."""
    import redo
    calls = []
    monkeypatch.setattr(ablation, "refuse_conflicting_state",
                        lambda *a, **k: calls.append("refuse"))

    def stale(nets, include=redo.ALL_NETS):
        calls.append("validate")
        raise AssertionError("the layer map is stale")

    monkeypatch.setattr(redo, "recyclable_units", stale)
    with pytest.raises(AssertionError, match="stale"):
        ablation.run_one(ARM, 0, 1, 1, 1, 1, None)
    assert calls == ["validate"]


def test_redo_redormancy_baseline_survives_a_checkpoint(tmp_path):
    from tests.test_checkpoint_resume import EXPECT, META
    a = make_run()
    a.trainer.training_steps = META["training_steps"]
    u = redo.Unit("representation", "conv1", "bn1", "conv2", 1)
    a._redo_last_plan = {u: [0, 5]}
    a._redo_stats = {"redo_recycled_total": 2}
    path = tmp_path / "ck.ckpt"
    ablation.save_run_checkpoint(a, path, META)
    b = make_run()
    ablation.load_run_checkpoint(b, path, EXPECT)
    assert b._redo_last_plan == {u: [0, 5]}



def test_step_window_tracks_loss_grad_norm_and_redo_pre_post(monkeypatch):
    """Round-2 design audit: ReDo's effect must be attributable -- rows get
    the mean loss / pre-clip grad norm and the mean grad norm over the 20
    steps before vs after each pass."""
    run = make_run()
    run.replay = [obs_traj([0.0] * 3, [0.0] * 3)]
    monkeypatch.setattr(muzero, "REDO_EVERY", 25)
    passes = []
    monkeypatch.setattr(run, "redo_step", lambda: passes.append(
        run.trainer.training_steps))
    for s in range(1, 58):
        run.trainer.training_steps = s
        gn = 1.0 if s <= 37 else 10.0                    # burst after pass
        run.trainer._last_step = (0.5, torch.tensor(gn))
        ablation.after_train_step(run)
    assert passes == [37]                                # 25 // 2 = 12 mod 25
    row = {}
    run._merge_row_stats(row)
    assert row["step_loss_mean"] == pytest.approx(0.5)
    assert row["step_grad_norm_mean"] == pytest.approx((37 * 1 + 20 * 10) / 57,
                                                       abs=1e-4)
    assert row["redo_gn_pre20"] == pytest.approx(1.0)   # steps 18..37
    assert row["redo_gn_post20"] == pytest.approx(10.0)  # steps 38..57
    assert run._step_window is None



def _survivor_like_head(bias):
    """Arm nets whose policy head mimics the trained survivors: decayed
    rows with a learned bias (the round-3 audit: survivors fire on 83-99%
    of positions through positive biases), and unit 1 dead."""
    torch.manual_seed(0)
    nets, opts = _arm_nets()
    pc = nets["prediction"].policy_conv
    with torch.no_grad():
        pc.weight.mul_(0.3)
        pc.bias.fill_(bias)
        pc.weight[1] = 0.0
        pc.bias[1] = -1.0                                # dead
    obs = torch.rand(32, 12, *muzero.IMAGE_RESHAPE)
    acts = torch.randint(0, ablation.ACTIONS, (32, muzero.UNROLL_STEPS))
    units = [u for u in redo.recyclable_units(nets)
             if (u.net, u.producer) == ("prediction", "policy_conv")]
    return nets, opts, pc, obs, acts, units


@pytest.mark.parametrize("bias", [0.1, 0.0, -0.05])
def test_no_bn_units_get_a_solved_bias_at_the_default_scale(bias):
    """Round-2/3 ReDo MEDIUMs: a recycled head unit (no BN) must come back
    ALIVE, at the median survivor's activity, with the default init's
    UNSCALED row (rescaling it inflated rows 4-24x into sparse units).
    Checked by RE-SCORING through the real training-shaped forward."""
    nets, opts, pc, obs, acts, units = _survivor_like_head(bias)
    u = units[0]
    scores, raw, inputs = redo.dormancy_stats(nets, obs, acts, units)
    assert scores[u][1] == 0 and u in inputs
    only1 = torch.ones_like(scores[u])
    only1[1] = 0                                         # recycle unit 1 only
    keep = [c for c in range(len(raw[u])) if c != 1]
    target = float(raw[u][keep].median())
    gen = torch.Generator().manual_seed(3)
    redo.recycle(nets, opts, {u: only1}, 0.0, gen, raw=raw, inputs=inputs)
    # The row is exactly the generator's default-init draw (no rescale).
    fan_in = pc.weight[1].numel()
    ref = torch.empty(pc.weight[1].shape).uniform_(
        -1 / np.sqrt(fan_in), 1 / np.sqrt(fan_in),
        generator=torch.Generator().manual_seed(3))
    assert torch.equal(pc.weight[1], ref)
    # Re-scored through dormancy_stats: the target activity, alive.
    _, raw2, inputs2 = redo.dormancy_stats(nets, obs, acts, units)
    assert float(raw2[u][1]) == pytest.approx(target, rel=1e-4)
    assert float(raw2[u][1]) > 0
    # The solved bias is positive exactly when a bias-0 unit would be
    # quieter than the survivors (monotonicity).
    act0 = redo._unit_activity(pc, inputs[u], pc.weight[1])
    assert (float(pc.bias[1]) > 0) == (target > act0)


def test_solve_bias_hits_the_target_on_skewed_inputs():
    g = torch.Generator().manual_seed(0)
    for z in (torch.randn(5000, generator=g) * 3 - 2,
              torch.randn(5000, generator=g).abs() * -1,     # all <= 0
              torch.full((100,), 0.7)):
        for target in (1e-4, 0.05, 1.0, 10.0):
            b = redo._solve_bias(z, target)
            assert float(torch.relu(z.double() + b).mean()) == pytest.approx(
                target, rel=1e-9, abs=1e-12)


def test_recycle_without_stats_keeps_the_papers_bias_zero():
    nets, opts, pc, obs, acts, units = _survivor_like_head(0.1)
    u = units[0]
    scores = redo.dormancy_scores(nets, obs, acts, units)
    redo.recycle(nets, opts, {u: scores[u]}, 0.0, torch.Generator().manual_seed(3))
    assert pc.bias[1] == 0


def test_note_behaviour_at_other_temperatures(monkeypatch):
    """The T != 1 paths: H_act from the TEMPERED prior, rare_rate from the
    birth target, and a row spanning a temperature change reports the
    step-weighted mean T and every value applied (round-3 LOW-4/7)."""
    run = make_run()
    assert ablation.ACTIONS == 5
    q = np.array([[0.7, 0.1, 0.1, 0.07, 0.03],
                  [0.2, 0.2, 0.2, 0.2, 0.2],
                  [0.5, 0.5, 0.0, 0.0, 0.0]])
    tgt = [np.array([0.9, 0.04, 0.03, 0.02, 0.01]),
           np.array([0.1, 0.5, 0.2, 0.1, 0.1]),
           np.array([0.3, 0.3, 0.2, 0.1, 0.1])]
    acted = [{"target_policy": t, "action": a} for t, a in zip(tgt, (1, 1, 4))]
    Ts = iter([0.5, 0.25])
    monkeypatch.setattr(muzero, "prior_acting_temperature", lambda s: next(Ts))

    def expect(T):
        qc = np.clip(q, 1e-12, None) ** (1 / T)
        qc /= qc.sum(1, keepdims=True)
        return -(qc * np.log(np.clip(qc, 1e-12, None))).sum(1)

    run._birth_priors = q
    run._note_behaviour(acted, 0)
    run._birth_priors = q[:2]
    run._note_behaviour(acted[:2], 0)
    row = {}
    run._merge_row_stats(row)
    H = np.concatenate([expect(0.5), expect(0.25)[:2]])
    assert row["beh_H_act"] == pytest.approx(H.mean(), abs=1e-4)
    # p_act: 0.04 (rare), 0.5, 0.1 ; then 0.04 (rare), 0.5
    assert row["beh_rare_rate"] == pytest.approx(2 / 5, abs=1e-4)
    assert row["beh_T"] == pytest.approx((3 * 0.5 + 2 * 0.25) / 5, abs=1e-4)
    assert row["beh_T_values"] == [0.25, 0.5]
    assert row["beh_action_counts"] == [0, 4, 0, 0, 1]


def test_redo_nets_names_are_validated():
    t = make_trainer()
    with pytest.raises(AssertionError, match="REDO_NETS"):
        redo.recyclable_units(_nets(t), include="representation")  # bare str
    with pytest.raises(AssertionError, match="REDO_NETS"):
        redo.recyclable_units(_nets(t), include=("representaton",))


def test_nonfinite_steps_are_counted_not_averaged(monkeypatch):
    run = make_run()
    run.replay = [obs_traj([0.0] * 3, [0.0] * 3)]
    monkeypatch.setattr(muzero, "REDO_EVERY", 1000)
    for s, gn in enumerate([1.0, float("nan"), 3.0], start=1):
        run.trainer.training_steps = s
        run.trainer._last_step = (0.5, torch.tensor(gn))
        ablation.after_train_step(run)
    row = {}
    run._merge_row_stats(row)
    assert row["step_nonfinite"] == 1
    assert row["step_grad_norm_mean"] == pytest.approx(2.0)


def test_recycled_gamma_is_floored_when_survivors_have_collapsed():
    t = make_trainer()
    rep = t.representation
    with torch.no_grad():
        rep.bn1.weight[:] = 1e-4                         # every survivor ~dead
    u = redo.Unit("representation", "conv1", "bn1", "conv2", 1)
    s = torch.ones(rep.conv1.weight.shape[0])
    s[0] = 0.0
    redo.recycle(_nets(t), [], {u: s}, 0.025, torch.Generator().manual_seed(0))
    assert float(rep.bn1.weight[0]) == pytest.approx(redo.GAMMA_FLOOR)



def test_real_train_step_feeds_the_step_diagnostics(monkeypatch):
    """Round-4 LOW-A: the step_* / redo_gn_* fields depend on train_step
    storing (loss, pre-clip grad norm); a dropped muzero.py hunk would blank
    them silently (after_train_step tolerates a missing _last_step)."""
    from tests.test_paper_mode import fake_paper_traj
    trainer = make_trainer()
    trainer.training_steps = 0
    traj = fake_paper_traj([1.0] * 6, [5.0] * 6)
    for t in traj:
        t["obs"] = np.random.randint(0, 255, (12, *muzero.IMAGE_RESHAPE),
                                     dtype=np.uint8)
    clipped = []
    real_clip = torch.nn.utils.clip_grad_norm_

    def spy_clip(*a, **k):
        clipped.append(real_clip(*a, **k))
        return clipped[-1]

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy_clip)
    np.random.seed(1)
    batch = trainer.prepare_batch_paper([traj], 4, muzero.UNROLL_STEPS, ACTIONS)
    loss = asyncio.run(trainer.train_step(batch))
    last = trainer._last_step
    assert last[0] == loss and np.isfinite(loss)
    assert len(clipped) == 1 and last[1] is clipped[0]
    assert np.isfinite(float(last[1])) and float(last[1]) > 0
    run = make_run()
    run.trainer, run.replay = trainer, [traj]
    monkeypatch.setattr(muzero, "REDO_EVERY", 1000)
    trainer.training_steps = 1
    ablation.after_train_step(run)
    row = {}
    run._merge_row_stats(row)
    assert row["step_loss_mean"] == pytest.approx(loss, abs=1e-4)
    assert row["step_grad_norm_mean"] == pytest.approx(float(last[1]), abs=1e-4)



def test_skipped_nonfinite_step_is_counted_not_double_reported(monkeypatch):
    """Round-5 LOW-2: the arm skips non-finite steps; train_step's skip
    branch must store ITS OWN (loss, NaN norm), or the row would re-count
    the previous finite step and step_nonfinite would read 0."""
    from tests.test_paper_mode import fake_paper_traj
    monkeypatch.setattr(muzero, "SKIP_NONFINITE_GRAD_STEPS", True)
    monkeypatch.setattr(muzero, "REDO_EVERY", 1000)
    trainer = make_trainer()
    trainer.training_steps = 0
    traj = fake_paper_traj([1.0] * 6, [5.0] * 6)
    for t in traj:
        t["obs"] = np.random.randint(0, 255, (12, *muzero.IMAGE_RESHAPE),
                                     dtype=np.uint8)
    run = make_run()
    run.trainer, run.replay = trainer, [traj]
    np.random.seed(1)
    batch = trainer.prepare_batch_paper([traj], 4, muzero.UNROLL_STEPS, ACTIONS)
    loss1 = asyncio.run(trainer.train_step(batch))
    gn1 = float(trainer._last_step[1])
    trainer.training_steps = 1
    ablation.after_train_step(run)
    trainer.representation.conv1.weight.register_hook(
        lambda g: g * float("nan"))
    np.random.seed(2)
    batch = trainer.prepare_batch_paper([traj], 4, muzero.UNROLL_STEPS, ACTIONS)
    asyncio.run(trainer.train_step(batch))
    assert trainer.nonfinite_skips == 1
    assert not np.isfinite(float(trainer._last_step[1]))
    trainer.training_steps = 2
    ablation.after_train_step(run)
    row = {}
    run._merge_row_stats(row)
    assert row["step_nonfinite"] == 1
    assert row["step_loss_mean"] == pytest.approx(loss1, abs=1e-4)
    assert row["step_grad_norm_mean"] == pytest.approx(gn1, abs=1e-4)
