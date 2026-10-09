"""The sharper prior-acting temperature arm (Kai, 2026-10-04):
effzero-repro-faithful-prioract-noredo-temp05 = prioract-noredo with
PRIOR_ACT_TEMP_SCALE 0.5, and its supervisor's pre-registered trigger."""
import ast
import json
import os
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] /
                       "experiments"))
import ablation
import muzero

ARM = "effzero-repro-faithful-prioract-noredo-temp05"
BASE = "effzero-repro-faithful-prioract-noredo"
SUP = (pathlib.Path(__file__).resolve().parents[1] / "benchmarks" /
       "supervisors" / "supervise_tempscale.py")


@pytest.fixture(autouse=True)
def _restore_np_rng():
    """Round-3 INFO: the play_episode tests draw from the global numpy RNG
    (np.random.choice); leave it as found."""
    state = np.random.get_state()
    yield
    np.random.set_state(state)


def test_default_is_the_identity():
    assert muzero.PRIOR_ACT_TEMP_SCALE == 1.0


def test_arm_differs_from_prioract_noredo_only_in_the_scale():
    a, b = ablation.ARMS[BASE], ablation.ARMS[ARM]
    assert {k for k in set(a) | set(b) if a.get(k) != b.get(k)} == {
        "PRIOR_ACT_TEMP_SCALE", "_doc"}
    assert b["PRIOR_ACT_TEMP_SCALE"] == 0.5
    assert b["USE_SEARCH_ACTING"] is False and b["PRIOR_ACT_STAGED_TEMP"] is True
    assert b["_launch"] == {"ray": True, "cpu_actor": True}
    assert b["RAY_EAGER_COLLECT"] is True and b["REDO_EVERY"] == 0


@pytest.mark.parametrize("scale,staged,want", [
    (1.0, True, (1.0, 0.5, 0.25)),
    (0.5, True, (0.5, 0.25, 0.125)),
    (1.0, False, (1.0, 1.0, 1.0)),
    (0.5, False, (0.5, 0.5, 0.5)),
])
def test_schedule(monkeypatch, scale, staged, want):
    monkeypatch.setattr(muzero, "TEMPERATURE_TOTAL_STEPS", 1000)
    monkeypatch.setattr(muzero, "ACTING_TEMPERATURE", 1.0)
    monkeypatch.setattr(muzero, "PRIOR_ACT_STAGED_TEMP", staged)
    monkeypatch.setattr(muzero, "PRIOR_ACT_TEMP_SCALE", scale)
    got = tuple(muzero.prior_acting_temperature(s) for s in (100, 600, 900))
    assert got == want


def test_identity_is_bit_exact(monkeypatch):
    monkeypatch.setattr(muzero, "TEMPERATURE_TOTAL_STEPS", 1000)
    monkeypatch.setattr(muzero, "PRIOR_ACT_STAGED_TEMP", True)
    for s in range(0, 1001, 37):
        assert (muzero.prior_acting_temperature(s)
                == muzero.visit_softmax_temperature(s))


def test_the_arms_own_schedule_at_production_length():
    """At the production run length (90 x 2000 steps) the arm acts at 0.5
    until 90k, 0.25 until 135k, then 0.125."""
    with ablation.arm_config(ARM), ablation._run_length_globals(90, 2000):
        got = [muzero.prior_acting_temperature(s)
               for s in (0, 89_999, 90_000, 134_999, 135_000, 180_000)]
    assert got == [0.5, 0.5, 0.25, 0.25, 0.125, 0.125]


class _OneStepEnv:
    def reset(self, seed=None):
        return np.zeros((210, 160, 3), dtype=np.uint8), {}

    def step(self, action):
        return np.zeros((210, 160, 3), dtype=np.uint8), 0.0, True, False, {}


@pytest.mark.parametrize("steps,power", [(100, 2.0), (600, 4.0), (900, 8.0)])
def test_play_episode_samples_prior_at_the_scaled_temperature(monkeypatch,
                                                              steps, power):
    prior = np.array([0.4, 0.3, 0.2, 0.05, 0.05])
    nm = type("NM", (), {})()
    nm.training_steps = steps
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
    monkeypatch.setattr(muzero, "PRIOR_ACT_TEMP_SCALE", 0.5)
    monkeypatch.setattr(np.random, "choice", spy)
    run.play_episode(greedy=False, cap=5)
    want = prior ** power
    assert np.allclose(seen[0], want / want.sum())


def test_a_very_peaked_prior_stays_a_valid_distribution(monkeypatch):
    """T = 0.125 raises the prior to the 8th power: tiny entries underflow
    to 0, which np.random.choice accepts; the sum must stay 1."""
    prior = np.array([0.999, 1e-4, 1e-5, 1e-6, 1e-7, 1e-30], dtype=np.float32)
    prior = prior / prior.sum()
    nm = type("NM", (), {})()
    nm.training_steps = 900
    nm.initial_inference = lambda stack: (prior.copy(), 0.0)
    run = object.__new__(ablation.Run)
    run.env, run.env_steps = _OneStepEnv(), 0
    run.searcher = type("S", (), {"network_manager": nm})()
    monkeypatch.setattr(ablation, "ACTIONS", 6)
    monkeypatch.setattr(muzero, "USE_SEARCH_ACTING", False)
    monkeypatch.setattr(muzero, "TEMPERATURE_TOTAL_STEPS", 1000)
    monkeypatch.setattr(muzero, "PRIOR_ACT_STAGED_TEMP", True)
    monkeypatch.setattr(muzero, "PRIOR_ACT_TEMP_SCALE", 0.5)
    run.play_episode(greedy=False, cap=5)          # raises if p is invalid


@pytest.mark.parametrize("scale", [0, -0.5, float("nan"), float("inf"),
                                   True, "0.5", None])
def test_gate_refuses_a_bad_scale(monkeypatch, scale):
    monkeypatch.setattr(muzero, "PRIOR_ACT_TEMP_SCALE", scale)
    with pytest.raises(SystemExit, match="PRIOR_ACT_TEMP_SCALE"):
        ablation.prior_temp_gate()


def test_gate_refuses_the_inert_search_acting_mix(monkeypatch):
    monkeypatch.setattr(muzero, "PRIOR_ACT_TEMP_SCALE", 0.5)
    monkeypatch.setattr(muzero, "USE_SEARCH_ACTING", True)
    with pytest.raises(SystemExit, match="inert under USE_SEARCH_ACTING"):
        ablation.prior_temp_gate()


@pytest.mark.parametrize("actions,ok_scale,bad_scale", [
    (6, 0.084, 0.08),       # floor 0.0208: 0.021 ok, 0.020 refused
    (18, 0.14, 0.13)])      # floor 0.0336: 0.035 ok, 0.0325 refused
def test_gate_floor_scales_with_the_action_count(monkeypatch, actions,
                                                 ok_scale, bad_scale):
    """Round-1 L3 / round-2 L2: refuse a lowest temperature within 20% of
    the float32 underflow point ln(A)/103.28 (staged path: 0.25 x scale)."""
    monkeypatch.setattr(ablation, "ACTIONS", actions)
    monkeypatch.setattr(muzero, "PRIOR_ACT_STAGED_TEMP", True)
    monkeypatch.setattr(muzero, "PRIOR_ACT_TEMP_SCALE", ok_scale)
    ablation.prior_temp_gate()
    monkeypatch.setattr(muzero, "PRIOR_ACT_TEMP_SCALE", bad_scale)
    with pytest.raises(SystemExit, match="underflows"):
        ablation.prior_temp_gate()


def test_gate_floor_on_the_unstaged_path(monkeypatch):
    monkeypatch.setattr(ablation, "ACTIONS", 6)
    monkeypatch.setattr(muzero, "PRIOR_ACT_STAGED_TEMP", False)
    monkeypatch.setattr(muzero, "ACTING_TEMPERATURE", 0.1)
    monkeypatch.setattr(muzero, "PRIOR_ACT_TEMP_SCALE", 0.19)    # 0.019
    with pytest.raises(SystemExit, match="underflows"):
        ablation.prior_temp_gate()


def test_gate_floor_matches_float32_underflow():
    """The floor's premise, measured: a uniform float32 prior over A actions
    raised to 1/T sums to 0 just below ln(A)/103.28 and not at the floor."""
    import math
    for a in (5, 6, 9, 18):
        u = np.full(a, 1.0 / a, dtype=np.float32)
        t_under = math.log(a) / 103.28
        assert (u ** np.float32(1.0 / (t_under * 0.97))).sum() == 0
        assert (u ** np.float32(1.0 / (t_under * 1.2))).sum() > 0


def test_gate_passes_the_arm_and_the_defaults():
    ablation.prior_temp_gate()
    with ablation.arm_config(ARM):
        ablation.prior_temp_gate()
    with ablation.arm_config(BASE):
        ablation.prior_temp_gate()


def test_gate_is_wired_into_run_one():
    import inspect
    assert "prior_temp_gate()" in inspect.getsource(ablation.run_one)


PERT = 0.1     # prediction-head noise: non-uniform, not saturated


def test_the_cpu_actor_acts_at_the_scaled_temperature(monkeypatch):
    """Round-1 L1: the REAL acting path -- ray_actors.SelfPlayActor built
    in-process (force_cpu, no Ray) from the overrides RayPipeline ships,
    its training_steps mirrored through set_weights, a short real Pong
    episode on the float32 prior: every sampled p equals prior**(1/T)
    normalized at T = 0.5 / 0.25 / 0.125 across the breakpoints."""
    import random
    import torch
    import ray_actors
    overrides = {k: v for k, v in ablation.ARMS[ARM].items()
                 if not k.startswith("_")}
    schedule = {"TEMPERATURE_TOTAL_STEPS": 180000,
                "OFFPOLICY_DECAY_STEPS": 54000}
    for k in list(overrides) + list(schedule):
        if hasattr(muzero, k):
            monkeypatch.setattr(muzero, k, getattr(muzero, k))
    for k in ("ACTIONS", "EPISODE_CAP", "ENV_NAME", "FRAMESKIP"):
        monkeypatch.setattr(ablation, k, getattr(ablation, k))
    np_state, t_state = np.random.get_state(), torch.get_rng_state()
    py_state, threads = random.getstate(), torch.get_num_threads()
    try:
        actor = ray_actors.SelfPlayActor(overrides, 6, 5000, schedule,
                                         "Pong-v4", 4, 12, server=None,
                                         force_cpu=True)
        assert muzero.PRIOR_ACT_TEMP_SCALE == 0.5
        nm = actor.searcher.network_manager
        torch.manual_seed(5)          # a non-uniform prior, so T matters
        with torch.no_grad():
            for prm in nm.prediction.parameters():
                prm.add_(torch.randn_like(prm) * PERT)
        seen = []
        real_ii, real_choice = nm.initial_inference, np.random.choice

        def ii(stack):
            pr, v = real_ii(stack)
            seen.append(("prior", np.array(pr, dtype=np.float64)))
            return pr, v

        def spy(n, p=None, **kw):
            seen.append(("p", np.asarray(p, dtype=np.float64)))
            return real_choice(n, p=p, **kw)

        nm.initial_inference = ii
        monkeypatch.setattr(np.random, "choice", spy)
        for steps, T in ((89_999, 0.5), (90_000, 0.25), (135_000, 0.125)):
            actor.set_weights({}, steps)
            seen.clear()
            actor.play(123)
            priors = [x[1] for x in seen if x[0] == "prior"]
            ps = [x[1] for x in seen if x[0] == "p"]
            assert ps and len(ps) <= len(priors)
            # non-vacuity: a one-hot or uniform prior is unchanged by T
            assert 0.25 < max(pr.max() for pr in priors) < 0.95
            for pr, p in zip(priors, ps):
                want = pr ** (1.0 / T)
                assert np.abs(p - want / want.sum()).max() < 1e-5, (steps, T)
            # ...and discriminating: the unscaled T would not match.
            unscaled = priors[0] ** (0.5 / T)
            assert np.abs(ps[0] - unscaled / unscaled.sum()).max() > 1e-3
    finally:
        np.random.set_state(np_state)
        torch.set_rng_state(t_state)
        random.setstate(py_state)
        torch.set_num_threads(threads)


# --------------------------------------------------------------------------
# The supervisor's pre-registered trigger
# --------------------------------------------------------------------------
def _trigger_ns():
    src = SUP.read_text(encoding="utf-8")
    tree = ast.parse(src)
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef)
            and n.name in ("_glued_record", "read_records", "first_success", "triggered",
                           "_num", "trigger_decision")]
    keep += [n for n in tree.body if isinstance(n, ast.Assign)
             and getattr(n.targets[0], "id", "") in (
                 "REPO", "PROBES", "TRIGGER_RUN", "TRIGGER_ROWS",
                 "TRIGGER_FILE", "MEAN_ROWS", "WORSE_BELOW")]
    ns = {"json": json, "os": os, "math": __import__("math")}
    exec(compile(ast.Module(body=sorted(keep, key=lambda n: n.lineno),
                            type_ignores=[]), "trigger", "exec"), ns)
    return ns


def _write(path, run, rows):
    with open(path, "a") as f:
        for it, (m, won) in rows.items():
            f.write(json.dumps({"run": run, "iteration": it,
                                "exact_mean": m, "phases_won": won}) + "\n")


def _decide(ns, tmp_path, rows, run=None):
    path = tmp_path / "rec.jsonl"
    _write(path, run or ns["TRIGGER_RUN"], rows)
    return ns["trigger_decision"](str(path))


ROWS = tuple(range(45, 91, 5))


@pytest.mark.parametrize("success_at,want_code", [
    (None, 0), (45, 2), (65, 2), (70, 0), (90, 0)])
def test_trigger_with_seed1_not_worse(tmp_path, success_at, want_code):
    """Non-success rows at 0.0 (mean above the WORSE edge): RUN iff the
    first success is >= 70 or never."""
    ns = _trigger_ns()
    rows = {i: (0.0, 1) for i in ROWS}
    if success_at is not None:
        rows[success_at] = (12.0, 4)
    code, msg = _decide(ns, tmp_path, rows)
    assert code == want_code, msg
    assert ("first success at %s" % (success_at if success_at else "none")) in msg


def test_waits_while_any_row_is_missing(tmp_path):
    ns = _trigger_ns()
    code, msg = _decide(ns, tmp_path, {i: (0.0, 1) for i in ROWS if i != 90})
    assert code is None and "[90]" in msg


@pytest.mark.parametrize("success_at", [None, 45, 70])
def test_worse_blocks_the_launch_whatever_the_trigger(tmp_path, success_at):
    """Round-1 M1: the DIRECTION proviso -- seed 1 WORSE (mean 60-90 <
    -2.32) -> exit 3, even when the trigger alone would say RUN or SKIP."""
    ns = _trigger_ns()
    rows = {i: (-10.0, 0) for i in ROWS}
    if success_at is not None:
        rows[success_at] = (12.0, 4)
    code, msg = _decide(ns, tmp_path, rows)
    assert code == 3 and "WORSE" in msg


@pytest.mark.parametrize("value,want_code", [(-2.25, 0), (-2.375, 3)])
def test_the_worse_edge(tmp_path, value, want_code):
    """WORSE iff the mean over rows 60-90 is below -2.32 (values exactly
    representable in binary, so the mean is exact)."""
    ns = _trigger_ns()
    mean_rows = ns["MEAN_ROWS"]
    assert mean_rows == tuple(range(60, 91, 5)) and ns["WORSE_BELOW"] == -2.32
    rows = {i: (0.0, 0) for i in ROWS}
    rows.update({i: (value, 0) for i in mean_rows})
    code, _ = _decide(ns, tmp_path, rows)
    assert code == want_code


@pytest.mark.parametrize("exact,won,is_success", [
    (10.0, 2, True), (9.99, 2, False), (-3.0, 3, True), (-3.0, 2, False)])
def test_success_criterion(tmp_path, exact, won, is_success):
    """Exact >= +10 alone, or >= 3/4 phases alone (round-1 L1)."""
    ns = _trigger_ns()
    rows = {i: (0.0, 1) for i in ROWS}
    rows[50] = (exact, won)
    code, msg = _decide(ns, tmp_path, rows)
    assert (code == 2) is is_success, msg


def test_last_record_wins_and_other_runs_are_ignored(tmp_path):
    ns = _trigger_ns()
    path = tmp_path / "rec.jsonl"
    _write(path, ns["TRIGGER_RUN"], {i: (0.0, 1) for i in ROWS} | {50: (12.0, 4)})
    _write(path, ns["TRIGGER_RUN"], {50: (0.0, 1)})       # a re-run probe
    _write(path, "some-other-run", {i: (15.0, 4) for i in ROWS})   # written LAST
    code, msg = ns["trigger_decision"](str(path))
    assert code == 0 and "first success at none" in msg


@pytest.mark.parametrize("fragment", [
    '{"snapshot": "x", "run": "%s", "iteration": 90, "exa',
    '{"snapshot": "x", "run": "%s", "iteration": 90, "phase_means": {"0": 5.0, "1"',
    # cut AFTER closed nested dicts (round-3 L1): a prefix decoder would
    # return {"0": 5.0} here; only decode-to-end-of-line finds the record
    '{"snapshot": "x", "run": "%s", "iteration": 90, "phase_means": {"0": 5.0}, '
    '"phases_won": 4, "games": {"1": [-21.0, 764, true]'])
def test_torn_tail_waits_then_heals_when_the_probe_reruns(tmp_path, fragment):
    """Round-1 L2 / round-2 L1: a killed probe's torn write (no newline), then
    a re-run appending EXACTLY as the probe does (json + "\n", no leading
    newline -- glued onto the fragment): waits, then recovers row 90. The
    second fragment holds a nested dict, which must not be mistaken for the
    record."""
    ns = _trigger_ns()
    run = ns["TRIGGER_RUN"]
    path = tmp_path / "rec.jsonl"
    _write(path, run, {i: (0.0, 1) for i in ROWS if i != 90})
    with open(path, "a") as f:
        f.write(fragment % run)
    code, msg = ns["trigger_decision"](str(path))
    assert code is None and "1 torn/undecodable" in msg
    rec = {"snapshot": "y", "run": run, "iteration": 90, "exact_mean": 12.0,
           "phase_means": {"0": 1.0}, "phases_won": 4}
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")           # as probe_exhaustive_eval does
    code, msg = ns["trigger_decision"](str(path))
    assert code == 0 and "first success at 90" in msg


def test_a_glued_record_from_another_run_does_not_fill_the_row(tmp_path):
    """Round-3 L1: the run filter applies to recovered records too."""
    ns = _trigger_ns()
    run = ns["TRIGGER_RUN"]
    path = tmp_path / "rec.jsonl"
    _write(path, run, {i: (0.0, 1) for i in ROWS if i != 90})
    with open(path, "a") as f:
        f.write('{"snapshot": "x", "run": "%s", "iteration": 90, "exa' % run)
        f.write(json.dumps({"snapshot": "y", "run": "some-other-run",
                            "iteration": 90, "exact_mean": 12.0,
                            "phases_won": 4}) + "\n")
    code, msg = ns["trigger_decision"](str(path))
    assert code is None and "[90]" in msg


@pytest.mark.parametrize("record", [
    {"exact_mean": 0.0}, {"phases_won": 1}, {"exact_mean": "0", "phases_won": 1},
    {"exact_mean": 0.0, "phases_won": True}, {"exact_mean": float("nan"),
                                              "phases_won": 1},
    {"exact_mean": float("inf"), "phases_won": 1},
    {"exact_mean": 0.0, "phases_won": 1.0}])
def test_malformed_record_refuses(tmp_path, record):
    """Round-1 L2: a missing phases_won must not silently count as 0."""
    ns = _trigger_ns()
    path = tmp_path / "rec.jsonl"
    _write(path, ns["TRIGGER_RUN"], {i: (0.0, 1) for i in ROWS if i != 60})
    with open(path, "a") as f:
        f.write(json.dumps({"run": ns["TRIGGER_RUN"], "iteration": 60,
                            **record}) + "\n")
    code, msg = ns["trigger_decision"](str(path))
    assert code == 4 and "malformed" in msg


def test_trigger_file_is_seed1s_probe_record_in_the_main_tree():
    ns = _trigger_ns()
    assert ns["TRIGGER_RUN"] == "pong-prioract2-s1-2026-10-02"
    assert pathlib.Path(ns["TRIGGER_FILE"]) == (
        pathlib.Path(r"Y:\SwitchPlayer") / "benchmarks" / "probes" /
        "probe_exhaustive_eval-pong-prioract2-s1-2026-10-02.jsonl")


def test_seed0_records_would_have_triggered():
    """Seed 0's real records: first success @75, mean 60-90 +3.6431 (as
    in CURRICULUM) -> RUN."""
    ns = _trigger_ns()
    probes = pathlib.Path(__file__).resolve().parents[1] / "benchmarks" / "probes"
    code, msg = ns["trigger_decision"](
        str(probes / "probe_exhaustive_eval-pong-prioract2-s0-2026-10-02.jsonl"),
        run="pong-prioract2-s0-2026-10-02")
    assert code == 0
    assert "first success at 75, mean 60-90 +3.6431" in msg


def test_the_module_level_enforcement_exits_on_every_nonzero_code():
    """The decision's one call site: break out of the wait loop on a code,
    print, and sys.exit(code) whenever it is non-zero (round-1 L1)."""
    tree = ast.parse(SUP.read_text(encoding="utf-8"))
    body = tree.body
    calls = [n for n in body if isinstance(n, ast.While)
             and "trigger_decision(TRIGGER_FILE)" in ast.unparse(n)]
    assert len(calls) == 1
    # the wait loop leaves only on a decision (0 = RUN is a decision too)
    assert "if code is not None:\n        break" in ast.unparse(calls[0])
    i = body.index(calls[0])
    assert ast.unparse(body[i + 1]) == "print(msg, flush=True)"
    assert isinstance(body[i + 2], ast.If)
    assert ast.unparse(body[i + 2].test) == "code == 3 and OVERRIDE"
    assert ast.unparse(body[i + 3]) == "if code:\n    sys.exit(code)"
    # ...and nothing is launched or created before it.
    before = ast.unparse(ast.Module(body=body[:i], type_ignores=[]))
    assert "os.makedirs(BASE" not in before and "Popen" not in before


def test_supervisor_variant_matches_the_pin():
    tree = ast.parse(SUP.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and getattr(n.targets[0], "id", "") == "VARIANTS")
    variants = ast.literal_eval(node.value)
    assert set(variants) == {"ray"}
    arm, _stem, flags = variants["ray"]
    assert arm == ARM and flags == ["--ray", "--cpu-actor"]



@pytest.mark.parametrize("code,override,fired,ckpt,exits,note", [
    (0, False, True, False, None, None), (0, True, True, False, None, None),
    (2, False, True, False, 2, None), (2, True, True, False, 2, None),
    (3, False, True, False, 3, None),
    (3, False, True, True, 3, "HINT"),             # relaunch without the flag
    (3, True, True, False, None, "OVERRIDE:"),     # the override launches
    (3, True, False, False, 2, "OVERRIDE refused"),  # ...not over an unfired trigger
    (4, False, True, False, 4, None), (4, True, True, False, 4, None)])
def test_kai_override_semantics(code, override, fired, ckpt, exits, note):
    """The decision's post-processing, executed from the supervisor's own
    source: --kai-override converts exit 3 only, and only if the trigger
    fired; a relaunch without the flag over a checkpoint gets a hint."""
    tree = ast.parse(SUP.read_text(encoding="utf-8"))
    body = tree.body
    loop = next(n for n in body if isinstance(n, ast.While)
                and "trigger_decision(TRIGGER_FILE)" in ast.unparse(n))
    i = body.index(loop)
    consts = [n for n in body if isinstance(n, ast.Assign)
              and getattr(n.targets[0], "id", "") in (
                  "OVERRIDE_NOTE", "OVERRIDE_REFUSED", "RELAUNCH_HINT")]
    printed = []

    class _Exit(Exception):
        pass

    def fake_exit(c):
        raise _Exit(c)

    ns = {"code": code, "msg": "m", "OVERRIDE": override, "CKPT": "ck",
          "TRIGGER_FILE": "tf",
          "override_code": lambda path: 0 if fired else 2,
          "os": type("O", (), {"path": type("P", (), {
              "exists": staticmethod(lambda p: ckpt)})})(),
          "print": lambda *a, **k: printed.append(a[0]),
          "sys": type("S", (), {"exit": staticmethod(fake_exit)})}
    try:
        exec(compile(ast.Module(body=consts + body[i + 1:i + 4],
                                type_ignores=[]), "post", "exec"), ns)
        got = None
    except _Exit as e:
        got = e.args[0]
    assert got == exits
    notes = [p for p in printed if p != "m"]
    if note is None:
        assert notes == []
    else:
        assert len(notes) == 1 and notes[0].startswith(note)


def test_override_code_on_records(tmp_path):
    """override_code from the real records (seed 1: no first success ->
    trigger fired -> 0) and from a fixture whose trigger did not fire."""
    ns = _trigger_ns()
    tree = ast.parse(SUP.read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == "override_code")
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "oc", "exec"), ns)
    real = (pathlib.Path(__file__).resolve().parents[1] / "benchmarks" /
            "probes" / "probe_exhaustive_eval-pong-prioract2-s1-2026-10-02.jsonl")
    assert ns["trigger_decision"](str(real))[0] == 3
    assert "first success at none, mean 60-90 -10.4524" in ns[
        "trigger_decision"](str(real))[1]
    assert ns["override_code"](str(real)) == 0
    rows = {i: (-10.0, 0) for i in ROWS}
    rows[50] = (12.0, 4)                     # broke out early, then WORSE
    path = tmp_path / "rec.jsonl"
    _write(path, ns["TRIGGER_RUN"], rows)
    assert ns["trigger_decision"](str(path))[0] == 3
    assert ns["override_code"](str(path)) == 2


def test_override_is_bound_exactly_once():
    tree = ast.parse(SUP.read_text(encoding="utf-8"))
    stores = [n for n in ast.walk(tree) if isinstance(n, ast.Name)
              and n.id == "OVERRIDE" and isinstance(n.ctx, ast.Store)]
    assert len(stores) == 1


def test_preflight_runs_and_audits_the_seqref_suite():
    src = SUP.read_text(encoding="utf-8")
    tree = ast.parse(src)
    aud = ast.literal_eval(next(n for n in tree.body if isinstance(n, ast.Assign)
                                and getattr(n.targets[0], "id", "") == "AUDITED").value)
    assert "tests/test_seqref.py" in aud and "tests/test_tempscale.py" in aud
    pre = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
               and n.name == "preflight")
    suites = [e.value for n in ast.walk(pre) if isinstance(n, ast.List)
              for e in n.elts if isinstance(e, ast.Constant)
              and str(e.value).startswith("tests/")]
    assert {"tests/test_tempscale.py", "tests/test_seqref.py"} <= set(suites)
    assert set(suites) <= set(aud)


def test_override_flag_is_explicit_and_argv_only():
    tree = ast.parse(SUP.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and getattr(n.targets[0], "id", "") == "OVERRIDE")
    assert ast.unparse(node.value) == "'--kai-override' in sys.argv"
