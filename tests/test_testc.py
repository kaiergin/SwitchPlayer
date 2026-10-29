"""Test C (Kai, 2026-10-29; CURRICULUM.md "Next experiment (pre-registered)
-- Test C: BBF-style resets in one 100k run, against a no-reset twin"): T0's
recipe run from scratch to 100,000 env steps without (C0, ...-long-n0) and
with (C, ...-reset-r0) BBF's periodic resets. Pins the arms, the warmup's
origin (off: bit for bit), the reset gate, the reset schedule, the reset
itself (shrink and perturb, re-init, the target model, the optimizer state,
the warmup, the actor's hold, the caches, the RNG streams), the pipeline's
hold, the checkpoint's reset state, the row fields and the plasticity read
(dormant units, first-layer filters), and ReDo's sampler after its
extraction. CPU only."""
import contextlib
import copy
import inspect
import pathlib
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] /
                       "experiments"))
import ablation
import muzero
import redo
from tests.test_prioract2 import restore_rngs  # noqa: F401

PREFIX = "effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-"
T0 = PREFIX + "tokens-two-t0"
C0 = PREFIX + "tokens-long-n0"
C = PREFIX + "tokens-reset-r0"
KNOBS = {"RESET_EVERY_STEPS": 40000, "RESET_UNTIL_ENV_STEPS": 80000,
         "RESET_SHRINK_NETS": ("representation", "dynamics"),
         "RESET_REINIT_NETS": ("prediction", "value_prefix", "projector"),
         "RESET_SHRINK_FACTOR": 0.5, "RESET_PERTURB_FACTOR": 0.5,
         "RESET_HOLD_STEPS": 10000}
SHRINK, REINIT = KNOBS["RESET_SHRINK_NETS"], KNOBS["RESET_REINIT_NETS"]


@pytest.fixture(autouse=True)
def no_leak():
    """Torn down last: every test leaves the module's Test C knobs at their
    defaults."""
    yield
    assert (muzero.RESET_EVERY_STEPS, muzero.RESET_UNTIL_ENV_STEPS,
            muzero.RESET_SHRINK_NETS, muzero.RESET_REINIT_NETS,
            muzero.RESET_HOLD_STEPS, muzero.DORMANT_DIAGNOSTIC,
            muzero.LR_WARMUP_ORIGIN) == (0, 0, (), (), 0, False, 0)


@pytest.fixture
def origin(monkeypatch):
    monkeypatch.setattr(muzero, "LR_WARMUP_ORIGIN", muzero.LR_WARMUP_ORIGIN)


@pytest.fixture
def in_arm(restore_rngs):  # noqa: F811
    """Inside an arm's config and its 36-token action space; yields a
    function entering an arm (C by default) and returning its spec. No
    monkeypatch here: a test's own monkeypatch (requested AFTER in_arm) is
    then undone BEFORE the arm is left -- the other order would restore an
    arm-set knob into the module after arm_config restored it."""
    saved = muzero.LR_WARMUP_ORIGIN
    with contextlib.ExitStack() as stack:
        def enter(arm=C):
            spec = stack.enter_context(ablation.arm_config(arm))
            stack.enter_context(ablation.token_actions())
            return spec
        yield enter
    muzero.LR_WARMUP_ORIGIN = saved


def make_run(spec, arm=C, steps=40000, env=13000):
    """A Run (object.__new__) with the arm's real nets and optimizers on the
    CPU, at `steps` training steps and `env` env steps; no replay."""
    run = object.__new__(ablation.Run)
    run.arm, run.seed, run.spec = arm, 0, spec
    nets = ablation.build_networks(spec)
    tr = object.__new__(muzero.TrainingHelper)
    tr.device = ablation.DEVICE
    tr.action_space = ablation.ACTIONS
    tr.training_steps = steps
    for n, net in nets.items():
        setattr(tr, n, net)
    for n, o in zip(ablation.NET_NAMES, ablation.OPT_NAMES):
        setattr(tr, o, muzero.make_optimizer(getattr(tr, n).parameters()))
    run.trainer = tr
    run.replay, run.env_steps = [], env
    run.reset_log, run.hold, run.held_ships = [], None, []
    return run


def trained(run):
    """Give the run a trained-looking state: momentum in every optimizer,
    moved BN statistics, a target model synced 100 steps BEFORE the run's
    step -- less than the 200-step interval, so the reset's own
    _ensure_target_model() does not sync and only bbf_reset's lines can
    set the target's stamp and snapshot (audit r1a TEST-2, r2a LOW-1) --
    that the online nets then left."""
    tr = run.trainer
    for n, o in zip(ablation.NET_NAMES, ablation.OPT_NAMES):
        net, opt = getattr(tr, n), getattr(tr, o)
        opt.zero_grad()
        sum((p ** 2).sum() for p in net.parameters()).backward()
        opt.step()
    with torch.no_grad():
        for n in ablation.NET_NAMES:
            for m in getattr(tr, n).modules():
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                    m.running_mean.normal_()
                    m.running_var.uniform_(0.5, 2.0)
    steps = tr.training_steps
    tr.training_steps = steps - 100
    tr._ensure_target_model()
    tr.training_steps = steps
    with torch.no_grad():
        for n in ablation.NET_NAMES:
            for p in getattr(tr, n).parameters():
                p.add_(0.01 * torch.randn_like(p))


def states(net):
    return {k: v.detach().clone() for k, v in net.state_dict().items()}


def equal(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


class FakePipe:
    """What bbf_reset reads and writes on the pipeline."""

    def __init__(self, pending=True):
        self.pending = ("futures", 38000) if pending else None
        self.payload = {"representation": {"w": torch.ones(2)}}
        self._states_cache = (40000, "S")
        self._acting_cache = (40000, "A")
        self._target_states_cache = (40000, "T")
        self._refresh_shipped_stamp = 40000
        self._birth_states = "OLD"
        self.calls = []

    def _acting_states(self):
        self.calls.append("acting")
        return self.payload

    def _states(self):
        self.calls.append("states")
        return {"post": True}


# --------------------------------------------------------------------------
# the arms
# --------------------------------------------------------------------------
def test_c0_is_t0_run_to_100k_with_the_plasticity_read():
    t0, c0 = ablation.ARMS[T0], ablation.ARMS[C0]
    d = {k for k in set(t0) | set(c0) if t0.get(k) != c0.get(k)}
    assert d == {"_nominal_iterations", "DORMANT_DIAGNOSTIC", "_doc"}
    assert c0["_nominal_iterations"] == 150
    assert c0["DORMANT_DIAGNOSTIC"] is True
    # T0's recipe: the sign clip, the 36 tokens, 0.997^4, the env stop
    assert c0["DISCOUNT"] == 0.997 ** 4 and len(c0["TOKEN_VOCAB"]) == 36
    assert c0["REWARD_GROWTH_START"] == c0["REWARD_GROWTH_CEILING"] == 1.0
    assert c0["_env_stop"] is True and c0["PREFIX_BN_PER_STEP"] is True
    assert "_student" not in c0 and "RESET_EVERY_STEPS" not in c0
    assert c0["REDO_EVERY"] == 0 and c0["USE_TARGET_MODEL"] is True


def test_c_is_c0_plus_bbfs_reset():
    c0, c = ablation.ARMS[C0], ablation.ARMS[C]
    d = {k for k in set(c0) | set(c) if c0.get(k) != c.get(k)}
    assert d == set(KNOBS) | {"_doc"}
    assert {k: c[k] for k in KNOBS} == KNOBS
    # BBF's: every 40k gradient steps (20 iterations of 2,000), halfway
    assert c["RESET_EVERY_STEPS"] % 2000 == 0
    assert c["RESET_HOLD_STEPS"] % 2000 == 0


def test_every_reset_knob_is_off_by_default():
    assert muzero.RESET_EVERY_STEPS == 0
    assert muzero.RESET_UNTIL_ENV_STEPS == 0
    assert muzero.RESET_SHRINK_NETS == () == muzero.RESET_REINIT_NETS
    assert muzero.RESET_HOLD_STEPS == 0
    assert muzero.DORMANT_DIAGNOSTIC is False
    assert muzero.LR_WARMUP_ORIGIN == 0
    # no other arm carries a reset knob or the plasticity read
    for name, spec in ablation.ARMS.items():
        if name not in (C0, C):
            assert not any(k.startswith("RESET_") for k in spec), name
            assert not spec.get("DORMANT_DIAGNOSTIC"), name


# --------------------------------------------------------------------------
# the warmup's origin
# --------------------------------------------------------------------------
def _lr_after(steps, start=7.0):
    opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=start)
    muzero.apply_lr_warmup((opt,), steps)
    return opt.param_groups[0]["lr"]


def test_the_warmup_at_origin_0_is_the_old_function(monkeypatch, origin):
    monkeypatch.setattr(muzero, "LEARNING_RATE", 0.2)
    monkeypatch.setattr(muzero, "LR_WARMUP_STEPS", 1000)
    for s in (0, 1, 250, 999, 1000):
        assert _lr_after(s) == 0.2 * min(1.0, s / 1000), s
    assert _lr_after(1001) == 7.0               # past it: untouched
    monkeypatch.setattr(muzero, "LR_WARMUP_STEPS", 0)
    assert _lr_after(5) == 7.0                  # off


def test_a_reset_restarts_the_warmup_from_its_step(monkeypatch, origin):
    monkeypatch.setattr(muzero, "LEARNING_RATE", 0.2)
    monkeypatch.setattr(muzero, "LR_WARMUP_STEPS", 1000)
    monkeypatch.setattr(muzero, "LR_WARMUP_ORIGIN", 40000)
    assert _lr_after(40000) == 0.0
    assert _lr_after(40500) == pytest.approx(0.1)
    assert _lr_after(41000) == pytest.approx(0.2)
    assert _lr_after(41001) == 7.0
    assert _lr_after(39999) == 7.0              # before it: untouched


def test_the_warmup_origin_starts_at_0_per_run_and_is_restored(origin):
    muzero.LR_WARMUP_ORIGIN = 123
    with ablation.warmup_origin():
        assert muzero.LR_WARMUP_ORIGIN == 0
        muzero.LR_WARMUP_ORIGIN = 80000
    assert muzero.LR_WARMUP_ORIGIN == 123
    src = inspect.getsource(ablation.run_one)
    assert "env_schedule(stop_env_steps), warmup_origin():" in src


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------
def test_both_arms_pass_their_gates_at_the_launch_recipe(in_arm):
    for arm in (C, C0):
        with ablation.arm_config(arm) as spec:
            ablation.reset_gate(spec, True, True, 0, 2000, None)
            ablation.env_stop_gate(spec, 100000, True, True, 300, True)


def test_search_acting_with_birth_targets_passes_the_gate(in_arm,
                                                          monkeypatch):
    """The birth search still makes every target under
    SEARCH_ACT_BIRTH_TARGETS: only the action comes from the actor."""
    spec = in_arm(C)
    monkeypatch.setattr(muzero, "USE_SEARCH_ACTING", True)
    monkeypatch.setattr(muzero, "SEARCH_ACT_BIRTH_TARGETS", True)
    ablation.reset_gate(spec, True, True, 0, 2000, None)


GATE_CASES = [
    ({}, dict(use_ray=False), "pipelined --ray --cpu-actor"),
    ({}, dict(cpu_actor=False), "pipelined --ray --cpu-actor"),
    ({}, dict(ray_workers=2), "pipelined --ray --cpu-actor"),
    ({"ASYNC_LOOP": True}, {}, "no ASYNC_LOOP"),
    ({"REFRESH_OFFLOAD": True}, {}, "REFRESH_OFFLOAD"),
    ({"REFRESH_SATURATE": True}, {}, "REFRESH_SATURATE"),
    ({}, dict(init_weights="w.pth"), "from scratch"),
    ({}, dict(spec={"_student": "distill"}), "from scratch"),
    ({"REDO_EVERY": 1000}, {}, "ReDo are exclusive"),
    ({"USE_TARGET_MODEL": False}, {}, "USE_TARGET_MODEL"),
    # audit r1a LOW-2
    ({"LR_DECAY_AT": 0.75}, {}, "undo LR_DECAY_AT"),
    ({"LR_WARMUP_STEPS": 0}, {}, "needs LR_WARMUP_STEPS"),
    ({"BIRTH_SEARCH": False}, {}, "needs BIRTH_SEARCH"),
    ({"USE_SEARCH_ACTING": True}, {}, "needs BIRTH_SEARCH"),
    ({"RESET_SHRINK_NETS": ("representation",)}, {}, "must split"),
    ({"RESET_SHRINK_NETS": ("representation", "dynamics", "prediction")},
     {}, "must split"),
    ({"RESET_REINIT_NETS": ("prediction", "prediction", "value_prefix",
                            "projector")}, {}, "must split"),
    ({"RESET_EVERY_STEPS": 41000}, {}, "whole iterations"),
    ({"RESET_EVERY_STEPS": True}, {}, "whole iterations"),
    ({"RESET_HOLD_STEPS": 10500}, {}, "whole iterations"),
    ({"RESET_HOLD_STEPS": -2000}, {}, "whole iterations"),
    ({"RESET_HOLD_STEPS": 40000}, {}, "the hold shorter"),
    ({"RESET_UNTIL_ENV_STEPS": 0}, {}, "positive env-step"),
    ({"RESET_UNTIL_ENV_STEPS": True}, {}, "positive env-step"),
    ({"RESET_SHRINK_FACTOR": 0.0}, {}, "RESET_SHRINK_FACTOR"),
    ({"RESET_PERTURB_FACTOR": 1.5}, {}, "RESET_PERTURB_FACTOR"),
    ({"RESET_PERTURB_FACTOR": True}, {}, "RESET_PERTURB_FACTOR"),
]


@pytest.mark.parametrize("flags,kw,match", GATE_CASES)
def test_the_reset_gate_refuses(in_arm, monkeypatch, flags, kw, match):
    spec = in_arm(C)
    for k, v in flags.items():
        monkeypatch.setattr(muzero, k, v)
    args = dict(spec=spec, use_ray=True, cpu_actor=True, ray_workers=0,
                train_steps=2000, init_weights=None)
    args.update(kw)
    with pytest.raises(SystemExit, match=match):
        ablation.reset_gate(**args)


@pytest.mark.parametrize("knob,value", [
    ("RESET_UNTIL_ENV_STEPS", 80000), ("RESET_SHRINK_NETS", ("dynamics",)),
    ("RESET_REINIT_NETS", ("projector",)), ("RESET_HOLD_STEPS", 2000)])
def test_a_reset_knob_without_the_period_is_refused(in_arm, monkeypatch,
                                                    knob, value):
    spec = in_arm(C0)
    monkeypatch.setattr(muzero, knob, value)
    with pytest.raises(SystemExit, match="need RESET_EVERY_STEPS"):
        ablation.reset_gate(spec, True, True, 0, 2000, None)


def test_run_one_gates_before_any_deletion():
    src = inspect.getsource(ablation.run_one)
    assert (src.index("env_stop_gate(spec") < src.index("reset_gate(spec")
            < src.index("refuse_conflicting_state("))


# --------------------------------------------------------------------------
# the schedule
# --------------------------------------------------------------------------
@pytest.mark.parametrize("steps,env,done,last,want", [
    (38000, 12000, 0, False, False),
    (40000, 13000, 0, False, True),
    (40000, 13000, 0, True, False),        # the run's last iteration
    (40000, 13000, 1, False, False),       # already done (a resume)
    (78000, 60000, 1, False, False),
    (80000, 60000, 1, False, True),
    (240000, 80000, 5, False, True),       # at the cutoff: still due
    (240000, 80001, 5, False, False),      # past it
    (280000, 93000, 6, False, False),
])
def test_reset_due(monkeypatch, steps, env, done, last, want):
    for k, v in KNOBS.items():
        monkeypatch.setattr(muzero, k, v)
    run = SimpleNamespace(trainer=SimpleNamespace(training_steps=steps),
                          env_steps=env, reset_log=[{}] * done)
    assert ablation.reset_due(run, last) is want


def test_no_reset_is_due_without_the_period():
    run = SimpleNamespace(trainer=SimpleNamespace(training_steps=40000),
                          env_steps=1, reset_log=[])
    assert ablation.reset_due(run, False) is False


def test_the_reset_lands_after_the_rows_eval_before_the_checkpoint():
    src = inspect.getsource(ablation.run_one)
    loop = src[src.index("for it in range(start_it + 1, iterations + 1):"):]
    i = loop.index("if reset_due(run, last):")
    # the iteration's sync and its next game's request come first: the
    # hold keeps what that sync shipped (audit r2a NIT-2)
    assert (loop.index("pipeline.sync()")
            < loop.index("pipeline.request(it + 1)") < i)
    assert loop.index("pipeline.request_eval(it, 8,") < i
    assert loop.index("save_weight_snapshot(") < i
    assert i < loop.index("save_run_checkpoint(run, ck, {")
    assert loop[i:].index("bbf_reset(run, pipeline, it)") < 60


# --------------------------------------------------------------------------
# the reset
# --------------------------------------------------------------------------
def test_the_reset_is_bbfs(in_arm, capsys):
    spec = in_arm(C)
    run = make_run(spec)
    trained(run)
    tr = run.trainer
    before = {n: states(getattr(tr, n)) for n in ablation.NET_NAMES}
    params = {n: {k: p.detach().clone()
                  for k, p in getattr(tr, n).named_parameters()}
              for n in SHRINK}
    opt_state = {o: copy.deepcopy(getattr(tr, o).state_dict())
                 for o in ablation.OPT_NAMES}
    fresh = ablation.fresh_networks(spec, (0, 1, 0xB8F))
    pipe = FakePipe()
    rng = (torch.get_rng_state(), np.random.get_state()[1].copy(),
           random.getstate())
    ablation.bbf_reset(run, pipe, 20)
    # the global streams are untouched
    assert torch.equal(torch.get_rng_state(), rng[0])
    assert np.array_equal(np.random.get_state()[1], rng[1])
    assert random.getstate() == rng[2]
    # shrink and perturb: 0.5 x itself + 0.5 x a fresh draw, every
    # parameter (BN affines included); running statistics kept
    for n in SHRINK:
        new = dict(fresh[n].named_parameters())
        now = dict(getattr(tr, n).named_parameters())
        for k, old in params[n].items():
            assert torch.allclose(now[k], 0.5 * old + 0.5 * new[k],
                                  rtol=0, atol=1e-6), (n, k)
            assert not torch.equal(now[k], old) or torch.equal(new[k], old)
        for k, v in getattr(tr, n).state_dict().items():
            if k not in now:                       # a buffer
                assert torch.equal(v, before[n][k]), (n, k)
    # re-initialized: the fresh draw, statistics included
    for n in REINIT:
        assert equal(states(getattr(tr, n)), states(fresh[n])), n
    # the optimizers: the re-initialized nets' state cleared, the shrunk
    # nets' kept
    for n, o in zip(ablation.NET_NAMES, ablation.OPT_NAMES):
        sd = getattr(tr, o).state_dict()
        if n in REINIT:
            assert sd["state"] == {}, n
        else:
            assert sd["state"] and all(
                torch.equal(sd["state"][i]["momentum_buffer"],
                            opt_state[o]["state"][i]["momentum_buffer"])
                for i in sd["state"]), n
    # the target model, both buffers, is the reset online nets at this step
    pairs = (("representation", tr.target_representation),
             ("prediction", tr.target_prediction),
             ("dynamics", tr.target_dynamics),
             ("value_prefix", tr.target_value_prefix))
    for (n, target), snap in zip(pairs, tr._target_snapshot):
        assert equal(states(target), states(getattr(tr, n))), n
        assert equal(snap, states(getattr(tr, n))), n
        assert not target.training
    assert tr._target_synced_at == 40000
    # the warmup restarts here
    assert muzero.LR_WARMUP_ORIGIN == 40000
    # the hold: the pre-reset acting states (taken BEFORE the reset), to
    # the hold's end; the caches dropped; the in-flight game birth-searched
    # by the reset nets
    assert pipe.calls == ["acting", "states"]
    assert set(run.hold) == {"until", "from", "held"}
    assert run.hold["until"] == 50000 and run.hold["from"] == 40000
    assert run.hold["held"] is pipe.payload
    assert pipe._states_cache == pipe._acting_cache == (None, None)
    assert pipe._target_states_cache == (None, None)
    assert pipe._refresh_shipped_stamp is None
    assert pipe._birth_states == {"post": True}
    assert run.reset_log == [{"index": 1, "iteration": 20,
                              "training_steps": 40000, "env_steps": 13000}]
    out = capsys.readouterr().out
    assert ("RESET 1 after iter 20 (step 40000, 13000 env steps): "
            "representation+dynamics shrunk x0.5 + 0.5 x fresh; "
            "prediction+value_prefix+projector re-initialized; target "
            "reset; warmup restarts; the actor holds the pre-reset weights "
            "until step 50000") in out


def test_the_shrink_and_perturb_factors_are_not_swapped(in_arm,
                                                        monkeypatch):
    """Audit r3a NIT-3: the registered factors are both 0.5; unequal ones
    show which weighs the old parameters and which the fresh draw."""
    spec = in_arm(C)
    monkeypatch.setattr(muzero, "RESET_SHRINK_FACTOR", 0.8)
    monkeypatch.setattr(muzero, "RESET_PERTURB_FACTOR", 0.2)
    run = make_run(spec)
    trained(run)
    old = {k: p.detach().clone()
           for k, p in run.trainer.representation.named_parameters()}
    fresh = ablation.fresh_networks(spec, (0, 1, 0xB8F))
    new = dict(fresh["representation"].named_parameters())
    ablation.bbf_reset(run, None, 20)
    for k, p in run.trainer.representation.named_parameters():
        assert torch.allclose(p, 0.8 * old[k] + 0.2 * new[k], rtol=0,
                              atol=1e-6), k


def test_the_reset_is_keyed_on_seed_and_index(in_arm):
    spec = in_arm(C)
    a, b = make_run(spec), make_run(spec)
    trained(a)
    trained(b)
    ablation.bbf_reset(a, None, 20)
    ablation.bbf_reset(b, None, 20)
    # different pre-reset states, the same fresh draw
    for n in REINIT:
        assert equal(states(getattr(a.trainer, n)),
                     states(getattr(b.trainer, n))), n
    # the second reset draws another
    a.trainer.training_steps = 80000
    ablation.bbf_reset(a, None, 40)
    assert not equal(states(a.trainer.prediction),
                     states(b.trainer.prediction))
    assert [r["index"] for r in a.reset_log] == [1, 2]
    assert muzero.LR_WARMUP_ORIGIN == 80000


def test_no_hold_without_a_pipeline_or_a_hold_length(in_arm, monkeypatch):
    spec = in_arm(C)
    run = make_run(spec)
    trained(run)
    ablation.bbf_reset(run, None, 20)
    assert run.hold is None and len(run.reset_log) == 1
    monkeypatch.setattr(muzero, "RESET_HOLD_STEPS", 0)
    run = make_run(spec)
    trained(run)
    pipe = FakePipe(pending=False)
    ablation.bbf_reset(run, pipe, 20)
    assert run.hold is None
    # nothing in flight: the birth states stay as they were
    assert pipe._birth_states == "OLD"


def test_a_reset_before_the_first_target_sync_builds_the_target(in_arm):
    spec = in_arm(C)
    run = make_run(spec)
    assert getattr(run.trainer, "target_representation", None) is None
    ablation.bbf_reset(run, None, 20)
    assert equal(states(run.trainer.target_prediction),
                 states(run.trainer.prediction))


def test_the_fresh_draw_never_seeds_cuda(in_arm, monkeypatch):
    """Audit r1a LOW-1: torch.manual_seed would reseed every CUDA
    generator, which fork_rng(devices=[]) does not restore."""
    spec = in_arm(C)
    calls = []
    for name in ("manual_seed_all", "manual_seed"):
        monkeypatch.setattr(torch.cuda, name,
                            lambda *a, name=name: calls.append(name))
    a = ablation.fresh_networks(spec, (0, 1, 0xB8F))
    b = ablation.fresh_networks(spec, (0, 1, 0xB8F))
    assert calls == []
    for n in ablation.NET_NAMES:
        assert equal(states(a[n]), states(b[n])), n


def _perturb(run, scale):
    with torch.no_grad():
        for n in ablation.NET_NAMES:
            for p in getattr(run.trainer, n).parameters():
                p.add_(scale * torch.randn_like(p))


@pytest.mark.parametrize("synced_at", [39900, 40000])
def test_the_target_double_buffer_runs_on_from_the_reset(in_arm, synced_at):
    """Audit r1a TEST-2 / r2a LOW-1: the reset's stamp and snapshot, then
    the 200-step syncs -- the target stays the reset nets until the sync
    after next. Two orders: a sync 100 steps before the reset (only
    bbf_reset's stamp keeps the next sync at 40,200), and the run's own at
    an even iteration -- the refresh synced at 40,000 just before the
    reset, its snapshot the PRE-reset nets, which only bbf_reset's
    snapshot keeps out of the target at 40,200."""
    spec = in_arm(C)
    run = make_run(spec)
    trained(run)                               # synced at 39,900
    tr = run.trainer
    assert tr._target_synced_at == 39900
    if synced_at == 40000:
        # the run's even iteration: the last in-training sync at 39,800,
        # then the refresh's at 40,000 -- the target takes the earlier
        # snapshot, the snapshot becomes the PRE-reset nets (audit r3a
        # MEDIUM-1: from the helper's 39,900 no sync is due at 40,000)
        tr._target_synced_at = 39800
        tr._ensure_target_model()
        assert tr._target_synced_at == 40000
    ablation.bbf_reset(run, None, 20)
    assert tr._target_synced_at == 40000
    reset = states(tr.prediction)
    seq = []
    for ts in (40000, 40100, 40200, 40400):
        tr.training_steps = ts
        tr._ensure_target_model()
        seq.append((ts, tr._target_synced_at,
                    equal(states(tr.target_prediction), reset)))
        _perturb(run, 0.001)                   # "training"
    assert seq == [(40000, 40000, True), (40100, 40000, True),
                   (40200, 40200, True), (40400, 40400, False)], seq


def test_the_hold_on_the_real_pipeline(in_arm, monkeypatch):
    """Audit r1a TEST-1: the pipeline's own _states / _acting_states /
    _actor_states / sync / request on real nets (a fake actor), in
    run_one's order at a reset iteration: the hold carries EXACTLY the
    pre-reset states shipped at 40,000 -- a reset that ran before taking
    them would hold the reset nets -- the in-flight game is birth-searched
    by the reset nets, the ships at 42k-48k are held, from 50k the
    learner's; held_ships records each held ship once."""
    spec = in_arm(C)
    monkeypatch.setattr(muzero, "BN_RECAL_FRAMES", 64)
    run = make_run(spec)
    run.replay = _replay()
    trained(run)
    pipe = object.__new__(ablation.RayPipeline)
    pipe.run, pipe.server, pipe.seed, pipe.episodes_per_iter = run, None, 0, 3
    pipe.ray = SimpleNamespace(get=lambda x: x)
    shipped = []
    pipe.actor = SimpleNamespace(
        set_weights=Remote(lambda st, steps, *env:
                           shipped.append((st, steps))),
        play=Remote(lambda seed: f"game{seed}"))
    pipe.pending = None
    pipe._states_cache = pipe._acting_cache = (None, None)
    pipe._target_states_cache = (None, None)
    pipe._refresh_shipped_stamp = None
    pipe._birth_states = pipe._birth_acting = None
    pre = {n: states(getattr(run.trainer, n))
           for n in ("dynamics", "value_prefix")}
    # iteration 20's end: sync, request(21), (the row), the reset
    pipe.sync()
    pipe.request(21)
    s20 = shipped[-1][0]
    assert pipe._birth_acting is s20
    ablation.bbf_reset(run, pipe, 20)
    assert run.hold["held"] is s20 and run.hold["until"] == 50000
    for n in ("dynamics", "value_prefix"):
        assert equal(run.hold["held"][n], pre[n]), n
        assert not equal(states(getattr(run.trainer, n)), pre[n]), n
    assert equal(pipe._birth_states["prediction"],
                 states(run.trainer.prediction))
    assert run.held_ships == []
    # iterations 21-26: train, sync, request
    kinds = []
    for it in range(21, 27):
        _perturb(run, 0.001)
        run.trainer.training_steps = it * 2000
        pipe.sync()
        pipe.request(it + 1)
        st = shipped[-1][0]
        # (the learner's acting states: raw nets + recalibrated BN stats)
        live = st is pipe._acting_states() and equal(
            st["dynamics"], states(run.trainer.dynamics))
        kinds.append("HELD" if st is s20 else "LIVE" if live else "??")
        assert pipe._birth_acting is (s20 if st is s20 else st)
        assert equal(pipe._birth_states["prediction"],
                     states(run.trainer.prediction))
    assert kinds == ["HELD"] * 4 + ["LIVE"] * 2, kinds
    assert run.hold is None
    assert run.held_ships == [42000, 44000, 46000, 48000]


# --------------------------------------------------------------------------
# the hold in the pipeline
# --------------------------------------------------------------------------
class Remote:
    def __init__(self, f):
        self.f = f

    def remote(self, *a, **k):
        return self.f(*a, **k)


def _pipe(monkeypatch, run, shipped):
    pipe = object.__new__(ablation.RayPipeline)
    pipe.run, pipe.server, pipe.seed, pipe.episodes_per_iter = run, None, 0, 1
    pipe.ray = SimpleNamespace(get=lambda x: x)
    pipe.actor = SimpleNamespace(
        set_weights=Remote(lambda st, steps, *env:
                           shipped.append((st, steps, env))),
        play=Remote(lambda seed: f"game{seed}"))
    monkeypatch.setattr(pipe, "_acting_states", lambda: "LIVE")
    monkeypatch.setattr(pipe, "_states", lambda: "STATES")
    return pipe


def test_the_actor_keeps_the_held_weights_until_the_holds_end(monkeypatch):
    monkeypatch.setattr(muzero, "SCHEDULE_ENV_TARGET", 100000)
    tr = SimpleNamespace(training_steps=42000)
    run = SimpleNamespace(trainer=tr, env_steps=14000,
                          hold={"until": 50000, "from": 40000,
                                "held": "HELD"})
    shipped = []
    pipe = _pipe(monkeypatch, run, shipped)
    pipe.sync()
    pipe.sync()                    # a resume's re-ship at the same step
    tr.training_steps = 48000
    pipe.sync()
    assert run.hold is not None
    tr.training_steps = 50000
    pipe.sync()
    assert run.hold is None                   # dropped at the hold's end
    tr.training_steps = 52000
    pipe.sync()
    assert shipped == [("HELD", 42000, (14000,)), ("HELD", 42000, (14000,)),
                       ("HELD", 48000, (14000,)), ("LIVE", 50000, (14000,)),
                       ("LIVE", 52000, (14000,))]
    # each held ship's step, once
    assert run.held_ships == [42000, 48000]


def test_a_resumes_reship_at_the_resets_own_step_is_no_held_ship(
        monkeypatch):
    """Audit r2b MEDIUM-1: a resume from the reset iteration's own
    checkpoint constructs the pipeline at the reset's step; its sync ships
    the held states -- what the run shipped there before the reset -- but
    counts no held ship (else every later row reads 5 per reset)."""
    monkeypatch.setattr(muzero, "SCHEDULE_ENV_TARGET", None)
    tr = SimpleNamespace(training_steps=40000)
    run = SimpleNamespace(trainer=tr, env_steps=13000, held_ships=[],
                          hold={"until": 50000, "from": 40000,
                                "held": "HELD"})
    shipped = []
    pipe = _pipe(monkeypatch, run, shipped)
    pipe.sync()
    assert shipped == [("HELD", 40000, ())] and run.held_ships == []
    for steps in (42000, 44000, 46000, 48000, 50000):
        tr.training_steps = steps
        pipe.sync()
    assert run.held_ships == [42000, 44000, 46000, 48000]
    assert [s for _st, s, _e in shipped] == [40000, 42000, 44000, 46000,
                                             48000, 50000]


def test_a_run_without_a_hold_ships_its_acting_states(monkeypatch):
    monkeypatch.setattr(muzero, "SCHEDULE_ENV_TARGET", None)
    run = SimpleNamespace(trainer=SimpleNamespace(training_steps=5),
                          env_steps=1)          # no hold attribute at all
    shipped = []
    _pipe(monkeypatch, run, shipped).sync()
    assert shipped == [("LIVE", 5, ())]
    assert not hasattr(run, "held_ships")


def test_the_behaviour_prior_follows_the_hold_the_birth_search_does_not(
        monkeypatch):
    monkeypatch.setattr(muzero, "BIRTH_SEARCH", True)
    monkeypatch.setattr(muzero, "BN_RECAL_ACTING", True)
    tr = SimpleNamespace(training_steps=44000)
    run = SimpleNamespace(trainer=tr, env_steps=14000,
                          hold={"until": 50000, "from": 40000,
                                "held": "HELD"})
    pipe = _pipe(monkeypatch, run, [])
    pipe.request(23)
    assert pipe._birth_acting == "HELD" and pipe._birth_states == "STATES"
    tr.training_steps = 50000
    pipe.request(26)
    assert pipe._birth_acting == "LIVE"


def test_evals_and_snapshots_take_the_learners_states():
    for f in (ablation.RayPipeline.request_eval,
              ablation.RayPipeline.snapshot_acting_states):
        src = inspect.getsource(f)
        assert "self._acting_states()" in src and "_actor_states" not in src


# --------------------------------------------------------------------------
# the checkpoint
# --------------------------------------------------------------------------
def test_the_checkpoint_carries_the_resets_and_the_hold(
        monkeypatch, origin, tmp_path, restore_rngs):  # noqa: F811
    from tests.test_checkpoint_resume import EXPECT, META, make_run as mk
    monkeypatch.setattr(muzero, "RESET_EVERY_STEPS", 40000)
    a = mk()
    a.trainer.training_steps = META["training_steps"]
    a.reset_log = [{"index": 1, "iteration": 20, "training_steps": 40000,
                    "env_steps": 13000}]
    held = {"representation": {"conv1.weight": torch.randn(3, 2)}}
    a.hold = {"until": 50000, "from": 40000, "held": held}
    a.held_ships = [42000, 44000]
    path = tmp_path / "resume.ckpt"
    ablation.save_run_checkpoint(a, path, META)
    b = mk()
    muzero.LR_WARMUP_ORIGIN = 0
    ablation.load_run_checkpoint(b, path, EXPECT)
    assert b.reset_log == a.reset_log and b.reset_log is not a.reset_log
    assert b.hold["until"] == 50000 and b.hold["from"] == 40000
    assert torch.equal(b.hold["held"]["representation"]["conv1.weight"],
                       held["representation"]["conv1.weight"])
    assert b.held_ships == [42000, 44000]
    assert muzero.LR_WARMUP_ORIGIN == 40000
    # past the hold: none; no reset yet: the warmup's origin is 0
    a.hold, a.reset_log = None, []
    ablation.save_run_checkpoint(a, path, META)
    c = mk()
    muzero.LR_WARMUP_ORIGIN = 7
    ablation.load_run_checkpoint(c, path, EXPECT)
    assert c.hold is None and c.reset_log == []
    assert muzero.LR_WARMUP_ORIGIN == 0


def test_no_other_arms_checkpoint_changes(monkeypatch, origin, tmp_path,
                                          restore_rngs):  # noqa: F811
    from tests.test_checkpoint_resume import EXPECT, META, make_run as mk
    a = mk()                                  # no reset_log / hold at all
    path = tmp_path / "resume.ckpt"
    ablation.save_run_checkpoint(a, path, META)
    assert "resets" not in ablation._read_checkpoint(path)
    muzero.LR_WARMUP_ORIGIN = 5
    b = mk()
    ablation.load_run_checkpoint(b, path, EXPECT)
    assert muzero.LR_WARMUP_ORIGIN == 5
    assert not hasattr(b, "reset_log")


# --------------------------------------------------------------------------
# the rows: the reset fields and the plasticity read
# --------------------------------------------------------------------------
def test_a_reset_runs_rows_count_its_resets_and_held_ships(in_arm):
    spec = in_arm(C)
    run = make_run(spec, steps=42000)
    run.reset_log = [{"index": 1}]
    run.hold = {"until": 50000, "from": 40000, "held": {}}
    run.held_ships = [42000]
    row = run.evaluate_diagnostics()
    assert row["resets"] == 1 and row["held_ships"] == 1
    assert "acting_held" not in row
    del run.held_ships                        # a run built before the field
    assert run.evaluate_diagnostics()["held_ships"] == 0
    # the plasticity read, here without a replay: the filters only
    assert "conv1_weak_frac" in row and "dormant_025" not in row


def test_the_controls_rows_carry_the_read_and_no_reset_fields(in_arm):
    spec = in_arm(C0)
    run = make_run(spec, arm=C0)
    row = run.evaluate_diagnostics()
    assert "resets" not in row and "held_ships" not in row
    assert "conv1_weak_frac" in row


def _replay(n_traj=3, length=12):
    rng = np.random.default_rng(0)
    return [[{"obs": rng.integers(0, 255, (12, *muzero.IMAGE_RESHAPE),
                                  dtype=np.uint8),
              "action": int(rng.integers(0, ablation.ACTIONS))}
             for _ in range(length)] for _ in range(n_traj)]


def test_the_dormant_read(in_arm, monkeypatch):
    spec = in_arm(C0)
    monkeypatch.setattr(muzero, "REDO_BATCH", 16)
    run = make_run(spec, arm=C0)
    run.replay = _replay()
    tr = run.trainer
    before = {n: states(getattr(tr, n)) for n in ablation.NET_NAMES}
    rng = (torch.get_rng_state(), np.random.get_state()[1].copy())
    out = run.dormant_stats()
    assert torch.equal(torch.get_rng_state(), rng[0])
    assert np.array_equal(np.random.get_state()[1], rng[1])
    for n in ablation.NET_NAMES:              # nets and statistics as they were
        assert equal(states(getattr(tr, n)), before[n]), n
    assert set(out) == {"dormant_025", "dormant_025_all", "dormant_10",
                        "dormant_10_all", "conv1_weak_frac"}
    for key in ("dormant_025", "dormant_10"):
        assert set(out[key]) == set(redo.ALL_NETS)
        assert all(0.0 <= v <= 1.0 for v in out[key].values())
    for n in redo.ALL_NETS:
        assert out["dormant_10"][n] >= out["dormant_025"][n]
    assert out["dormant_10_all"] >= out["dormant_025_all"]
    assert out == run.dormant_stats()         # keyed on (seed, step)
    # a fresh net's filters are all strong
    assert out["conv1_weak_frac"] == 0.0
    # dead units read as dormant: the representation's bn1 zeroed kills
    # every conv1 unit
    with torch.no_grad():
        tr.representation.bn1.weight.zero_()
        tr.representation.bn1.bias.zero_()
        half = tr.representation.conv1.weight.shape[0] // 2
        tr.representation.conv1.weight[:half].zero_()
    nets = {n: getattr(tr, n) for n in redo.ALL_NETS}
    units = redo.recyclable_units(nets, include=redo.ALL_NETS)
    rep = [u for u in units if u.net == "representation"]
    n_rep = sum(dict(tr.representation.named_modules())[u.producer]
                .weight.shape[0] for u in rep)
    n_conv1 = tr.representation.conv1.weight.shape[0]
    dead = run.dormant_stats()
    assert dead["dormant_025"]["representation"] >= n_conv1 / n_rep
    assert dead["conv1_weak_frac"] == 0.5


def test_redos_sample_is_unchanged_by_the_extraction(in_arm):
    """uniform_windows (ReDo's sampler, extracted for the dormant read)
    draws exactly what redo_step drew inline before."""
    spec = in_arm(C0)
    run = make_run(spec, arm=C0)
    run.replay = _replay(n_traj=4, length=9) + [[{"obs": None,
                                                  "action": 0}] * 3]
    K = muzero.UNROLL_STEPS

    def old(rng, n):
        starts = [(ti, len(t) - 1 - K) for ti, t in enumerate(run.replay)
                  if len(t) - 1 - K >= 0]
        sizes = np.array([m + 1 for _, m in starts], dtype=np.int64)
        picks = rng.choice(int(sizes.sum()), size=n)
        bounds = np.cumsum(sizes)
        obs, acts = [], []
        for p in picks:
            k = int(np.searchsorted(bounds, p, side="right"))
            ti = starts[k][0]
            i = int(p - (bounds[k] - sizes[k]))
            obs.append(run.replay[ti][i]["obs"])
            acts.append([run.replay[ti][i + j]["action"] for j in range(K)])
        return (torch.tensor(np.stack(obs), dtype=torch.float32) / 255.0,
                torch.tensor(np.asarray(acts), dtype=torch.long))

    got = run.uniform_windows(np.random.default_rng(5), 40)
    want = old(np.random.default_rng(5), 40)
    assert torch.equal(got[0].cpu(), want[0])
    assert torch.equal(got[1].cpu(), want[1])
    run.replay = [run.replay[-1]]                 # no whole window
    assert run.uniform_windows(np.random.default_rng(5), 4) is None
    src = inspect.getsource(ablation.Run.redo_step)
    assert "self.uniform_windows(rng, int(muzero.REDO_BATCH))" in src
