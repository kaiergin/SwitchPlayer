"""Test C2 (Kai, 2026-11-02; CURRICULUM.md "Next experiment (pre-registered)
-- Test C2: BBF's n-step and discount ramps on top of the resets"): Test C's
C (...-reset-r0) plus BBF's n-step and discount ramps, restarted at every
reset. Pins the knobs (off: every other arm bit for bit), the arm (C plus
the RAMP_* knobs), ramp_values against BBF's own scheduler (re-implemented
here from the released bbf/agents/spr_agent.py) and the pre-registration's
table, apply_ramps, the gate, every site that moves the step count or the
reset origin (the run's start, every gradient step, a reset, a checkpoint's
resume), the batch each step trains on, the restore after a run and the
row fields. CPU only."""
import contextlib
import inspect
import json
import pathlib
import re
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
from tests.test_testc import (C, C0, KNOBS as RESET_KNOBS, PREFIX, make_run,
                              trained)

C2 = PREFIX + "tokens-ramp-a0"
G_END = 0.997 ** 4
RAMP = {"RAMP_STEPS": 10000, "RAMP_N_START": 10, "RAMP_GAMMA_START": 0.97,
        "RAMP_N_END": 5, "RAMP_GAMMA_END": G_END}
# the pre-registration's schedule: steps since the reset, n, gamma (3 places)
TABLE = [(0, 10, 0.970), (2000, 9, 0.975), (4000, 8, 0.979),
         (6000, 7, 0.983), (8000, 6, 0.986), (10000, 5, 0.988)]
POINTS = (0, 1, 999, 1000, 2000, 5000, 9999, 10000, 10001, 40000)


@pytest.fixture(autouse=True)
def no_leak():
    """Torn down last: every test leaves the RAMP_* knobs at their defaults
    and TD_STEPS, DISCOUNT and the warmup's origin as it found them."""
    before = (muzero.TD_STEPS, muzero.DISCOUNT, muzero.LR_WARMUP_ORIGIN)
    yield
    assert ({k: getattr(muzero, k) for k in ablation.RAMP_OFF}
            == ablation.RAMP_OFF)
    assert (muzero.TD_STEPS, muzero.DISCOUNT,
            muzero.LR_WARMUP_ORIGIN) == before
    assert muzero.RESET_EVERY_STEPS == 0


@pytest.fixture
def in_arm(restore_rngs):  # noqa: F811
    """test_testc's: inside an arm's config (C2 by default) and its 36-token
    action space; the warmup's origin restored after. A test's own
    monkeypatch, requested AFTER in_arm, is undone before the arm is left."""
    saved = muzero.LR_WARMUP_ORIGIN
    with contextlib.ExitStack() as stack:
        def enter(arm=C2):
            spec = stack.enter_context(ablation.arm_config(arm))
            stack.enter_context(ablation.token_actions())
            return spec
        yield enter
    muzero.LR_WARMUP_ORIGIN = saved


@pytest.fixture
def ramp_on(monkeypatch):
    """C2's ramp knobs and end values on the module, outside any arm
    (TD_STEPS, DISCOUNT and the origin restored after)."""
    for k, v in {**RAMP, "TD_STEPS": 5, "DISCOUNT": G_END,
                 "LR_WARMUP_ORIGIN": 0}.items():
        monkeypatch.setattr(muzero, k, v)


def values():
    return muzero.TD_STEPS, muzero.DISCOUNT


# --------------------------------------------------------------------------
# BBF's scheduler, re-implemented from the released bbf/agents/spr_agent.py
# (onp -> np), and BBFAgent.__init__'s two schedules
# --------------------------------------------------------------------------
def bbf_exponential_decay_scheduler(decay_period, warmup_steps,
                                    initial_value, final_value,
                                    reverse=False):
    if reverse:
        initial_value = 1 - initial_value
        final_value = 1 - final_value

    start = np.log(initial_value)
    end = np.log(final_value)

    if decay_period == 0:
        return lambda x: initial_value if x < warmup_steps else final_value

    def scheduler(step):
        steps_left = decay_period + warmup_steps - step
        bonus_frac = steps_left / decay_period
        bonus = np.clip(bonus_frac, 0.0, 1.0)
        new_value = bonus * (start - end) + end

        new_value = np.exp(new_value)
        if reverse:
            new_value = 1 - new_value
        return new_value

    return scheduler


def bbf_schedules(cycle_steps=10000, update_horizon=5, max_update_horizon=10,
                  min_gamma=0.97, gamma=G_END):
    """update_horizon_scheduler and gamma_scheduler as BBFAgent builds them,
    at our window and end values (C's TD_STEPS and DISCOUNT)."""
    n_schedule = bbf_exponential_decay_scheduler(
        cycle_steps, 0, 1, update_horizon / max_update_horizon)

    def n(x):
        return int(np.round(n_schedule(x) * max_update_horizon))

    g = bbf_exponential_decay_scheduler(cycle_steps, 0, min_gamma, gamma,
                                        reverse=True)
    return n, g


# --------------------------------------------------------------------------
# the knobs and the arm
# --------------------------------------------------------------------------
def test_every_ramp_knob_is_off_by_default():
    assert ablation.RAMP_OFF == {"RAMP_STEPS": 0, "RAMP_N_START": 10,
                                 "RAMP_GAMMA_START": 0.97, "RAMP_N_END": None,
                                 "RAMP_GAMMA_END": None}
    assert ({k: getattr(muzero, k) for k in ablation.RAMP_OFF}
            == ablation.RAMP_OFF)
    # no other arm carries a ramp knob
    for name, spec in ablation.ARMS.items():
        if name != C2:
            assert not any(k.startswith("RAMP_") for k in spec), name


def test_c2_is_c_plus_the_ramps():
    c, c2 = ablation.ARMS[C], ablation.ARMS[C2]
    assert ablation.RAMP_ARM + "-a0" == C2
    d = {k for k in set(c) | set(c2) if c.get(k) != c2.get(k)}
    assert d == set(RAMP) | {"_doc"}
    assert {k: c2[k] for k in RAMP} == RAMP
    # every reset knob is C's
    assert {k: c2[k] for k in RESET_KNOBS} == RESET_KNOBS
    # the end values are C's own, exactly: n 5 base steps, 0.997^4
    assert c2["RAMP_N_END"] == c["TD_STEPS"] == 5
    assert c2["RAMP_GAMMA_END"] == c["DISCOUNT"] == 0.997 ** 4
    assert c2["TD_STEPS"] == 5 and c2["DISCOUNT"] == 0.997 ** 4
    # the window: 5 iterations of 2,000, the hold's
    assert c2["RAMP_STEPS"] == c["RESET_HOLD_STEPS"] == 5 * 2000


# --------------------------------------------------------------------------
# the schedule
# --------------------------------------------------------------------------
def test_ramp_values_are_bbfs_scheduler(ramp_on):
    """Bit for bit against BBF's own computation at every step of the
    window (the rounding included: n changes exactly where BBF's does), and
    past it; from the window's end the end values themselves."""
    n, g = bbf_schedules()
    for x in POINTS + tuple(range(10002)):
        got = muzero.ramp_values(x)
        assert type(got[0]) is int and type(got[1]) is float, x
        assert got[0] == n(x), x
        if x < 10000:
            assert got[1] == float(g(x)), x
        else:
            # BBF's exp / log round trip lands within an ulp; ours is the
            # arm's DISCOUNT exactly
            assert got == (5, G_END), x
            assert abs(float(g(x)) - G_END) < 1e-15, x
    assert [muzero.ramp_values(x)[0] for x in POINTS] == [
        10, 10, 9, 9, 9, 7, 5, 5, 5, 5]
    assert muzero.ramp_values(0)[1] == pytest.approx(0.97, abs=1e-15)
    # onp.round: half to even
    assert "int(np.round(" in inspect.getsource(muzero.ramp_values)


def test_n_changes_where_the_rounding_says(ramp_on):
    """10 x 0.5^f crosses 9.5 at f = log2(10 / 9.5), step 740.03..: 740
    is the last step at 10."""
    assert muzero.ramp_values(740)[0] == 10
    assert muzero.ramp_values(741)[0] == 9
    first = {}
    for x in range(10001):
        first.setdefault(muzero.ramp_values(x)[0], x)
    assert sorted(first, reverse=True) == [10, 9, 8, 7, 6, 5]
    for n, x in first.items():
        if n < 10:
            # the first step whose 10 x 0.5^f lies below n + 0.5
            assert 10 * 0.5 ** (x / 10000) < n + 0.5 <= (
                10 * 0.5 ** ((x - 1) / 10000)), (n, x)


@pytest.mark.parametrize("steps,n,gamma", TABLE)
def test_the_preregistered_schedule(ramp_on, steps, n, gamma):
    got_n, got_g = muzero.ramp_values(steps)
    f = min(steps / 10000, 1)
    assert got_n == n == int(np.round(10 * 0.5 ** f))
    assert round(got_g, 3) == gamma
    # 1 - gamma(f) = 0.03^(1-f) x (1 - 0.997^4)^f
    assert 1 - got_g == pytest.approx(
        (1 - 0.97) ** (1 - f) * (1 - G_END) ** f, rel=1e-12, abs=0)


def test_apply_ramps_is_a_no_op_while_off(monkeypatch):
    monkeypatch.setattr(muzero, "LR_WARMUP_ORIGIN", 40000)
    td, d = values()
    for s in (0, 1, 40000, 42000, 10 ** 6):
        muzero.apply_ramps(s)
        assert muzero.TD_STEPS is td and muzero.DISCOUNT is d


def test_apply_ramps_sets_the_globals_from_the_origin(ramp_on):
    muzero.apply_ramps(0)
    assert values() == muzero.ramp_values(0) and muzero.TD_STEPS == 10
    muzero.apply_ramps(2000)
    assert values() == muzero.ramp_values(2000) and muzero.TD_STEPS == 9
    muzero.LR_WARMUP_ORIGIN = 40000          # a reset at 40,000
    muzero.apply_ramps(42000)
    assert values() == muzero.ramp_values(2000)
    muzero.apply_ramps(40000)
    assert values() == muzero.ramp_values(0)
    for s in (50000, 50001, 79999):
        muzero.apply_ramps(s)
        assert values() == (5, G_END), s
        assert muzero.DISCOUNT == G_END


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------
def test_c2_passes_its_gates_at_the_launch_recipe(in_arm):
    spec = in_arm(C2)
    ablation.reset_gate(spec, True, True, 0, 2000, None)
    ablation.ramp_gate(spec)
    ablation.env_stop_gate(spec, 100000, True, True, 300, True)
    ablation.eager_collect_gate(True)
    ablation.sve_gate()
    ablation.birth_search_gate()
    # the gates run before anything applied a ramp: the arm's static values
    assert values() == (5, G_END)


def test_every_other_arm_passes_the_ramp_gate_off(in_arm):
    for arm in ablation.ARMS:
        if arm != C2:
            with ablation.arm_config(arm) as spec:
                ablation.ramp_gate(spec)


GATE_CASES = [
    ({"RESET_EVERY_STEPS": 0}, {}, "needs RESET_EVERY_STEPS"),
    ({"RAMP_STEPS": 50000}, {}, "at most RESET_EVERY_STEPS"),
    ({"RAMP_STEPS": -10000}, {}, "positive gradient-step count"),
    ({"RAMP_STEPS": 10000.0}, {}, "positive gradient-step count"),
    ({"RAMP_STEPS": True}, {}, "positive gradient-step count"),
    ({"ASYNC_LOOP": True}, {}, "no ASYNC_LOOP"),
    ({}, {"_student": "distill"}, "student's offline phase"),
    ({"TRAIN_PREFETCH": True}, {}, "TRAIN_PREFETCH"),
    ({"PAPER_TARGETS": False}, {}, "PAPER_TARGETS"),
    ({"VALUE_TARGET_MODE": "mixed"}, {}, "VALUE_TARGET_MODE 'td'"),
    ({"RAMP_N_START": 0}, {}, "RAMP_N_START 0 must be an int"),
    ({"RAMP_N_START": 10.0}, {}, "RAMP_N_START 10.0 must be an int"),
    ({"RAMP_N_START": True}, {}, "RAMP_N_START True must be an int"),
    ({"RAMP_N_END": None}, {}, "RAMP_N_END None must be an int"),
    ({"RAMP_GAMMA_START": 1.0}, {}, "RAMP_GAMMA_START 1.0 must be a float"),
    ({"RAMP_GAMMA_START": 0.0}, {}, "RAMP_GAMMA_START 0.0 must be a float"),
    ({"RAMP_GAMMA_START": 1}, {}, "RAMP_GAMMA_START 1 must be a float"),
    ({"RAMP_GAMMA_END": None}, {}, "RAMP_GAMMA_END None must be a float"),
    ({"RAMP_N_START": 4}, {}, "from the long horizon"),
    ({"RAMP_GAMMA_START": 0.995}, {}, "from the long horizon"),
    # the end values: the arm's own, exactly
    ({"RAMP_N_END": 3}, {}, "end on the arm's own"),
    ({"RAMP_GAMMA_END": 0.997}, {}, "end on the arm's own"),
    ({"RAMP_GAMMA_END": 0.98805}, {}, "end on the arm's own"),
    ({"TD_STEPS": 3}, {}, "end on the arm's own"),
    ({"DISCOUNT": 0.997}, {}, "end on the arm's own"),
    ({}, {"_drop": "TD_STEPS"}, "end on the arm's own"),
    ({}, {"_drop": "DISCOUNT"}, "end on the arm's own"),
]


@pytest.mark.parametrize("flags,edit,match", GATE_CASES)
def test_the_ramp_gate_refuses(in_arm, monkeypatch, flags, edit, match):
    spec = dict(in_arm(C2))
    for k, v in flags.items():
        monkeypatch.setattr(muzero, k, v)
    edit = dict(edit)
    drop = edit.pop("_drop", None)
    spec.update(edit)
    if drop:
        del spec[drop]
    with pytest.raises(SystemExit, match=match):
        ablation.ramp_gate(spec)


@pytest.mark.parametrize("knob,value", [
    ("RAMP_N_START", 3), ("RAMP_GAMMA_START", 0.9), ("RAMP_N_END", 5),
    ("RAMP_GAMMA_END", G_END)])
def test_a_ramp_knob_without_the_window_is_refused(in_arm, monkeypatch,
                                                   knob, value):
    spec = in_arm(C)
    monkeypatch.setattr(muzero, knob, value)
    with pytest.raises(SystemExit, match="need RAMP_STEPS"):
        ablation.ramp_gate(spec)


def test_run_one_gates_then_applies_the_ramp_before_any_reader():
    src = inspect.getsource(ablation.run_one)
    # the gate: after reset_gate, before any --overwrite deletion
    assert (src.index("reset_gate(spec") < src.index("ramp_gate(spec)")
            < src.index("refuse_conflicting_state("))
    # the run's start / resume: after the checkpoint's load (and the
    # warm-start and distillation blocks, which C2 refuses), before the
    # pipeline and the loop (the first collect's birth search)
    apply = src.index("        if muzero.RAMP_STEPS:\n"
                      "            muzero.apply_ramps(run.trainer.training_steps)")
    assert src.index("loaded = load_run_checkpoint(run, ck, expect)") < apply
    assert src.index("if init_weights and start_it == 0:") < apply
    assert src.index('if spec.get("_student") == "distill" and '
                     'start_it == 0:') < apply
    assert apply < src.index("pipeline = (RayPipeline(")
    assert apply < src.index("for it in range(start_it + 1, "
                             "iterations + 1):")
    # the restore: ramp_globals inside arm_config, around the whole run
    assert "with arm_config(arm) as spec, ramp_globals(), \\" in src


# --------------------------------------------------------------------------
# every site that moves the step count or the reset origin
# --------------------------------------------------------------------------
def test_every_site_that_moves_the_step_count_or_the_origin_applies_it():
    """The mutations in ablation.py: the step count at Run's start (0), a
    resume, and the two training loops (each calls after_train_step right
    after its increment: tests/test_prioract.py); the origin at
    warmup_origin's start (0) and restore, a resume and a reset. A new site
    fails this count."""
    src = inspect.getsource(ablation)
    assert len(re.findall(r"trainer\.training_steps \+= 1", src)) == 2
    assert len(re.findall(r"trainer\.training_steps = (?!=)", src)) == 2
    assert ("self.trainer.training_steps = 0"
            in inspect.getsource(ablation.Run.__init__))
    assert len(re.findall(r"LR_WARMUP_ORIGIN = ", src)) == 4
    assert inspect.getsource(ablation.warmup_origin).count(
        "LR_WARMUP_ORIGIN = ") == 2
    load = inspect.getsource(ablation.load_run_checkpoint)
    assert (load.index('run.trainer.training_steps = state["training_steps"]')
            < load.index("muzero.apply_ramps(run.trainer.training_steps)"))
    assert (load.index("muzero.LR_WARMUP_ORIGIN = (")
            < load.index("muzero.apply_ramps(run.trainer.training_steps)"))
    reset = inspect.getsource(ablation.bbf_reset)
    assert (reset.index("muzero.LR_WARMUP_ORIGIN = steps")
            < reset.index("muzero.apply_ramps(steps)"))
    assert ("muzero.apply_ramps(run.trainer.training_steps)"
            in inspect.getsource(ablation.after_train_step))
    tb = inspect.getsource(ablation.train_block)
    assert (tb.index("ramp_step(run, it,")
            < tb.index("run.trainer.train_step(next_batch())"))


class FakeTrainer:
    """train_block's trainer surface: each batch records the step it is
    built for and the module's n and gamma at that moment."""

    def __init__(self, steps):
        self.training_steps = steps
        self.batches = []

    async def prepare_batch(self, replay, batch_size, unroll, actions):
        self.batches.append((self.training_steps, muzero.TD_STEPS,
                             muzero.DISCOUNT))
        return {"step": self.training_steps}

    async def train_step(self, batch):
        assert batch["step"] == self.training_steps
        return 0.0


def test_each_steps_batch_takes_the_ramp_at_its_step(in_arm):
    """Iterations 21-26 after a reset at 40,000 (bbf_reset applied the
    ramp): every batch is built with the values at its own step, the
    iteration's first step is logged, and the end values hold from the
    window's end."""
    in_arm(C2)
    assert not muzero.TRAIN_PREFETCH
    run = SimpleNamespace(trainer=FakeTrainer(40000), replay=[])
    muzero.LR_WARMUP_ORIGIN = 40000
    muzero.apply_ramps(40000)
    for it in range(21, 27):
        ablation.train_block(run, 2000, it=it)
    seen = run.trainer.batches
    assert [s for s, _, _ in seen] == list(range(40000, 52000))
    for s, n, g in seen:
        assert (n, g) == muzero.ramp_values(s - 40000), s
    assert [n for s, n, _ in seen if s % 2000 == 0] == [10, 9, 8, 7, 6, 5]
    assert run.ramp_trace == [
        [it, c, *muzero.ramp_values(c)]
        for it, c in zip(range(21, 27), range(0, 12000, 2000))]
    assert [e[2] for e in run.ramp_trace] == [10, 9, 8, 7, 6, 5]
    assert values() == (5, G_END) and run.trainer.training_steps == 52000


def test_the_first_cycle_starts_at_the_runs_start(in_arm):
    """Before the first reset the ramp counts from step 0 (origin 0)."""
    in_arm(C2)
    run = SimpleNamespace(trainer=FakeTrainer(0), replay=[])
    muzero.apply_ramps(0)                     # run_one's start
    ablation.train_block(run, 2000, it=1)
    ablation.train_block(run, 2000, it=2)
    assert run.ramp_trace == [[1, 0, *muzero.ramp_values(0)],
                              [2, 2000, *muzero.ramp_values(2000)]]
    assert run.ramp_trace[0][2] == 10 and run.ramp_trace[1][2] == 9


class EagerPipe:
    """An in-flight game that is ready 35 steps into the block; collect
    records the values its birth search would read."""

    def __init__(self, run):
        self.run, self.start = run, run.trainer.training_steps
        self.pending = ("futures", self.start)
        self.collected = []

    def acting_ready(self):
        return self.run.trainer.training_steps - self.start >= 35

    def collect(self):
        self.pending = None
        self.collected.append((self.run.trainer.training_steps,
                               muzero.TD_STEPS, muzero.DISCOUNT))


def test_an_eager_collect_reads_the_ramp_at_its_step(in_arm, monkeypatch):
    in_arm(C2)
    monkeypatch.setattr(muzero, "RAY_EAGER_COLLECT", True)
    monkeypatch.setattr(muzero, "EAGER_POLL_STEPS", 10)
    run = SimpleNamespace(trainer=FakeTrainer(40000), replay=[])
    muzero.LR_WARMUP_ORIGIN = 40000
    muzero.apply_ramps(40000)
    pipe = EagerPipe(run)
    ablation.train_block(run, 2000, pipeline=pipe, it=21)
    # the game joined at step 40,040 (the poll grid) with the values there
    assert pipe.collected == [(40040, *muzero.ramp_values(40))]
    # the block reopened: still one trace entry, the iteration's first step
    assert run.ramp_trace == [[21, 0, *muzero.ramp_values(0)]]
    for s, n, g in run.trainer.batches:
        assert (n, g) == muzero.ramp_values(s - 40000), s


def test_a_missed_site_fails_loudly(in_arm):
    """The origin moved (a reset) without apply_ramps: the next step
    refuses before its batch is built."""
    in_arm(C2)
    run = SimpleNamespace(trainer=FakeTrainer(42000), replay=[])
    muzero.LR_WARMUP_ORIGIN = 40000
    assert values() == (5, G_END)              # stale: the ramp says 9
    with pytest.raises(RuntimeError, match="without muzero.apply_ramps"):
        ablation.train_block(run, 10, it=22)
    assert run.trainer.batches == []


def test_other_arms_train_on_their_static_values(in_arm):
    in_arm(C)
    td, d = values()
    run = SimpleNamespace(trainer=FakeTrainer(40000), replay=[])
    muzero.LR_WARMUP_ORIGIN = 40000
    ablation.train_block(run, 50, it=21)
    assert all(n is td and g is d for _, n, g in run.trainer.batches)
    assert muzero.TD_STEPS is td and muzero.DISCOUNT is d
    assert not hasattr(run, "ramp_trace")


# --------------------------------------------------------------------------
# a reset restarts the ramps
# --------------------------------------------------------------------------
def test_a_reset_restarts_the_ramps(in_arm):
    spec = in_arm(C2)
    run = make_run(spec, arm=C2, steps=40000)
    trained(run)
    muzero.apply_ramps(40000)                  # 40,000 since the start
    assert values() == (5, G_END)
    ablation.bbf_reset(run, None, 20)
    assert muzero.LR_WARMUP_ORIGIN == 40000
    assert values() == muzero.ramp_values(0) and muzero.TD_STEPS == 10
    assert muzero.DISCOUNT == pytest.approx(0.97, abs=1e-15)
    seen = []
    for s in range(42000, 54000, 2000):
        run.trainer.training_steps = s
        ablation.after_train_step(run)
        seen.append((muzero.TD_STEPS, round(muzero.DISCOUNT, 3)))
    assert seen == [(n, g) for _, n, g in TABLE[1:]] + [(5, 0.988)]
    assert muzero.DISCOUNT == G_END
    # the next reset restarts them again
    run.trainer.training_steps = 80000
    ablation.after_train_step(run)
    assert values() == (5, G_END)
    ablation.bbf_reset(run, None, 40)
    assert muzero.LR_WARMUP_ORIGIN == 80000
    assert values() == muzero.ramp_values(0)


def test_cs_reset_leaves_the_values_alone(in_arm):
    spec = in_arm(C)
    run = make_run(spec, arm=C, steps=40000)
    td, d = values()
    ablation.bbf_reset(run, None, 20)
    assert muzero.TD_STEPS is td and muzero.DISCOUNT is d


# --------------------------------------------------------------------------
# the batch: the value targets with the ramped n and gamma
# --------------------------------------------------------------------------
def _token_traj(rng, length, born):
    A = len(muzero.TOKEN_VOCAB)
    traj = [{"obs": rng.integers(0, 255, (12, *muzero.IMAGE_RESHAPE),
                                 dtype=np.uint8),
             "action": int(rng.integers(0, A)),
             "reward": float(rng.normal()),
             "target_policy": np.full(A, 1.0 / A),
             "root_value": float(rng.normal()),
             "target_value": 0.0,
             "has_policy_target": True,
             "born_step": born,
             "priority": 1.0} for _ in range(length)]
    traj[-1]["root_value"] = 0.0
    traj[-1]["has_policy_target"] = False
    traj[-1]["truncated"] = False
    return traj


def test_the_batch_value_targets_use_the_ramped_n_and_gamma(in_arm,
                                                            monkeypatch):
    """Two iterations after a reset at 40,000 (n 9, gamma at 2,000), with
    EfficientZero's age shrink on top (age 2,500 at 1,000 per step: n - 2),
    the paper batch (assemble + finish, the target-model bootstrap) gives
    every real position z = paper_value_parts(n 7, the ramped gamma) +
    g x v_target(tail) -- not the static values' (n 3, 0.997^4)."""
    spec = in_arm(C2)
    monkeypatch.setattr(muzero, "OFFPOLICY_DECAY_STEPS", 1000)
    run = make_run(spec, arm=C2, steps=42000)
    tr = run.trainer
    muzero.LR_WARMUP_ORIGIN = 40000
    muzero.apply_ramps(42000)
    n, g = muzero.ramp_values(2000)
    assert (n, values()) == (9, (9, g))
    assert muzero.paper_horizon(2500) == 7
    traj = _token_traj(np.random.default_rng(3), 30, born=42000 - 2500)
    np.random.seed(0)
    K = muzero.UNROLL_STEPS
    batch = tr.prepare_batch_paper([traj], 8, K, ablation.ACTIONS)

    def v_target(obs):
        with torch.no_grad():
            x = torch.tensor(obs[None], dtype=torch.float32) / 255.0
            _, v, _ = tr.target_prediction(tr.target_representation(x))
        return float(v.squeeze(-1)[0])

    checked = differs = 0
    for row, ref in enumerate(batch["sample_refs"]):
        start = next(i for i, t in enumerate(traj) if t is ref)
        for k in range(K + 1):
            pos = start + k
            if pos >= len(traj):
                continue
            want = []
            for horizon, gamma in ((7, g), (3, G_END)):
                rsum, gg, tail = muzero.paper_value_parts(traj, pos, horizon,
                                                          gamma)
                boot = 0.0 if tail == len(traj) - 1 else v_target(
                    traj[tail]["obs"])
                want.append(rsum + gg * boot)
            assert batch["target_value"][row][k] == pytest.approx(
                want[0], rel=1e-4, abs=1e-4), (row, k)
            checked += 1
            differs += abs(want[0] - want[1]) > 1e-3
    assert checked >= 8 and differs > 0


# --------------------------------------------------------------------------
# the checkpoint: a resume continues the ramp exactly
# --------------------------------------------------------------------------
def _ckpt_run(mk, steps, log):
    run = mk()
    run.trainer.training_steps = steps
    run.reset_log = [dict(r) for r in log]
    run.hold, run.held_ships = None, []
    return run


RESET1 = {"index": 1, "iteration": 20, "training_steps": 40000,
          "env_steps": 13000}
RESET2 = {"index": 2, "iteration": 40, "training_steps": 80000,
          "env_steps": 26000}


@pytest.mark.parametrize("log,steps,cycle", [
    ([], 6000, 6000),                    # before the first reset
    ([RESET1], 40000, 0),                # the reset iteration's checkpoint
    ([RESET1], 46000, 6000),             # mid-ramp
    ([RESET1, RESET2], 84000, 4000),
    ([RESET1], 60000, 20000),            # past the window
])
def test_a_resume_continues_the_ramp_exactly(ramp_on, monkeypatch, tmp_path,
                                             restore_rngs,  # noqa: F811
                                             log, steps, cycle):
    from tests.test_checkpoint_resume import EXPECT, META, make_run as mk
    monkeypatch.setattr(muzero, "RESET_EVERY_STEPS", 40000)
    origin = log[-1]["training_steps"] if log else 0
    a = _ckpt_run(mk, steps, log)
    muzero.LR_WARMUP_ORIGIN = origin
    muzero.apply_ramps(steps)
    want = values()
    assert want == muzero.ramp_values(cycle)
    a.ramp_trace = [[it, c, *muzero.ramp_values(c)]
                    for it, c in ((21, 0), (22, 2000), (23, 4000))]
    path = tmp_path / "resume.ckpt"
    ablation.save_run_checkpoint(a, path, {**META, "training_steps": steps})
    assert ablation._read_checkpoint(path)["ramp"] == {"trace": a.ramp_trace}
    # a fresh process: the arm's static values, the origin 0
    muzero.LR_WARMUP_ORIGIN = 0
    muzero.TD_STEPS, muzero.DISCOUNT = 5, G_END
    b = mk()
    ablation.load_run_checkpoint(b, path, EXPECT)
    assert b.trainer.training_steps == steps
    assert muzero.LR_WARMUP_ORIGIN == origin
    assert values() == want
    assert b.ramp_trace == a.ramp_trace and b.ramp_trace is not a.ramp_trace
    # and on, step by step, as the uninterrupted run
    for s in range(steps + 1, steps + 3001):
        b.trainer.training_steps = s
        ablation.after_train_step(b)
        assert values() == muzero.ramp_values(s - origin), s


def test_no_other_arms_checkpoint_changes(monkeypatch, tmp_path,
                                          restore_rngs):  # noqa: F811
    from tests.test_checkpoint_resume import EXPECT, META, make_run as mk
    a = mk()
    path = tmp_path / "resume.ckpt"
    ablation.save_run_checkpoint(a, path, META)
    assert "ramp" not in ablation._read_checkpoint(path)
    td, d = values()
    b = mk()
    ablation.load_run_checkpoint(b, path, EXPECT)
    assert muzero.TD_STEPS is td and muzero.DISCOUNT is d
    assert not hasattr(b, "ramp_trace")


# --------------------------------------------------------------------------
# the restore after a run
# --------------------------------------------------------------------------
def test_ramp_globals_restores_after_an_exception():
    td, d = values()
    with pytest.raises(RuntimeError):
        with ablation.ramp_globals():
            muzero.TD_STEPS, muzero.DISCOUNT = 10, 0.97
            raise RuntimeError("mid-run")
    assert muzero.TD_STEPS is td and muzero.DISCOUNT is d


@pytest.mark.parametrize("fail", [False, True])
def test_a_c2_run_leaves_the_module_as_it_found_it(fail):
    """run_one's contexts around a ramped run, ended normally or by an
    exception mid-ramp: TD_STEPS, DISCOUNT, the origin and the knobs are
    the pre-run values again (arm_config alone restores them too: the spec
    carries both)."""
    td, d = values()
    with contextlib.suppress(RuntimeError):
        with ablation.arm_config(C2), ablation.ramp_globals(), \
                ablation.warmup_origin():
            muzero.apply_ramps(0)              # the run's start
            assert values() == muzero.ramp_values(0)
            muzero.LR_WARMUP_ORIGIN = 40000    # a reset
            muzero.apply_ramps(42000)
            assert muzero.TD_STEPS == 9
            if fail:
                raise RuntimeError("mid-ramp")
    assert muzero.TD_STEPS is td and muzero.DISCOUNT is d
    with ablation.arm_config(C2):
        muzero.apply_ramps(0)
        assert muzero.TD_STEPS == 10
    assert muzero.TD_STEPS is td and muzero.DISCOUNT is d


# --------------------------------------------------------------------------
# the rows
# --------------------------------------------------------------------------
def test_a_ramp_runs_rows_carry_the_ramp_and_its_trace(in_arm):
    spec = in_arm(C2)
    run = make_run(spec, arm=C2, steps=50000)
    run.reset_log = [dict(RESET1)]
    muzero.LR_WARMUP_ORIGIN = 40000
    muzero.apply_ramps(50000)
    trace = [[it, c, *muzero.ramp_values(c)]
             for it, c in zip(range(21, 26), range(0, 10000, 2000))]
    run.ramp_trace = [list(e) for e in trace]
    row = run.evaluate_diagnostics()
    assert row["ramp"] == {"n": 5, "gamma": G_END, "cycle_steps": 10000}
    assert row["ramp_trace"] == trace
    assert [e[2] for e in row["ramp_trace"]] == [10, 9, 8, 7, 6]
    assert run.ramp_trace == []                # the window restarts
    assert json.loads(json.dumps(row["ramp"])) == row["ramp"]
    assert json.loads(json.dumps(row["ramp_trace"])) == trace
    # C's fields stay
    assert row["resets"] == 1 and "held_ships" in row
    # the next row: only what came since; mid-ramp values at a mid-ramp row
    run.trainer.training_steps = 44000
    muzero.apply_ramps(44000)
    row = run.evaluate_diagnostics()
    assert row["ramp"] == {"n": 8, "gamma": muzero.ramp_values(4000)[1],
                           "cycle_steps": 4000}
    assert row["ramp_trace"] == []


def test_the_controls_rows_carry_no_ramp_fields(in_arm):
    for arm in (C, C0):
        spec = in_arm(arm)
        row = make_run(spec, arm=arm).evaluate_diagnostics()
        assert "ramp" not in row and "ramp_trace" not in row, arm
