"""SVE on the temperature + stabilizer config (Kai, 2026-10-10):
effzero-repro-faithful-prioract-noredo-temp05-bnrecal-sve. Pins that it is
the stabilizer arm plus exactly the 09-28 SVE flags, that every gate
admits it, that SVE switches on across T1 inside the --ray loop with eager
mid-block collects (stage-2 searches run, the stale share is reported),
and the supervisor."""
import ast
import json
import pathlib
import subprocess
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] /
                       "experiments"))
import ablation
import muzero
from tests.test_prioract2 import _fake_pipe, restore_rngs  # noqa: F401

ARM = "effzero-repro-faithful-prioract-noredo-temp05-bnrecal-sve"
TWIN = "effzero-repro-faithful-prioract-noredo-temp05-bnrecal"
OLD_SVE = "effzero-repro-faithful-prioract-sve"
ROOT = pathlib.Path(__file__).resolve().parents[1]
SUP = ROOT / "benchmarks" / "supervisors" / "supervise_sve2.py"
SVE_KEYS = ("VALUE_TARGET_MODE", "SVE_START_TRAIN_STEPS",
            "SVE_FRESH_TRANSITIONS", "SVE_SOURCE", "SVE_SEARCH_SIMS")


def test_arm_is_the_stabilizer_arm_plus_the_0928_sve_flags():
    a, b = ablation.ARMS[TWIN], ablation.ARMS[ARM]
    assert {k for k in set(a) | set(b) if a.get(k) != b.get(k)} == {
        *SVE_KEYS, "_doc"}
    old = ablation.ARMS[OLD_SVE]
    assert all(b[k] == old[k] for k in SVE_KEYS)       # as pre-registered
    assert b["SVE_SOURCE"] == "search" and b["SVE_SEARCH_SIMS"] == 16
    assert b["BN_RECAL_ACTING"] is True and b["REDO_EVERY"] == 0
    assert b["_launch"] == {"ray": True, "cpu_actor": True}


def test_every_startup_gate_passes():
    with ablation.arm_config(ARM) as spec:
        ablation.ray_scope_gate(spec)
        ablation.launch_gate(spec, True, True)
        ablation.eager_collect_gate(True)
        ablation.prior_temp_gate()
        ablation.bn_recal_gate(True)
        ablation.search_act_birth_gate()
        ablation.birth_search_gate()
        ablation.sve_gate()


@pytest.mark.parametrize("ready_after", [5, 10_000])
def test_sve_switches_on_across_T1_in_the_ray_loop(
        monkeypatch, tmp_path, restore_rngs, ready_after):  # noqa: F811
    """run_one --ray (in-process fake pipeline), 3 iterations x 30 steps,
    T1 = 30, T2 = 5 transitions: no SVE before T1; after it, stale window
    roots get stage-2 searches -- with games collected mid-block
    (ready_after 5) or at the iteration end."""
    import torch
    torch.set_num_threads(2)
    inval = []
    real_inval = muzero.TrainingHelper.invalidate_bootstrap_cache

    def spy_inval(self, *a, **k):
        inval.append(self.training_steps)
        return real_inval(self, *a, **k)

    monkeypatch.setattr(muzero.TrainingHelper, "invalidate_bootstrap_cache",
                        spy_inval)
    monkeypatch.setattr(ablation, "RayPipeline", _fake_pipe([], ready_after))
    monkeypatch.setattr(ablation, "ENV_NAME", "Pong-v4")
    monkeypatch.setattr(ablation, "FRAMESKIP", 4)
    monkeypatch.setattr(ablation, "EPISODE_CAP", 10)
    monkeypatch.setattr(ablation, "RESULTS", tmp_path)
    monkeypatch.setattr(ablation, "TAG", "")
    monkeypatch.setitem(ablation.ARMS, "e2e-sve2", {
        **ablation.ARMS[ARM], "TRAINING_BATCHES": 16,
        "SVE_START_TRAIN_STEPS": 30, "SVE_FRESH_TRANSITIONS": 5,
        "SVE_SEARCH_SIMS": 4})
    out = tmp_path / "a.jsonl"
    ablation.run_one("e2e-sve2", 1, 3, 1, 30, 1, out, checkpoint_every=1,
                     use_ray=True, cpu_actor=True)
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert [r["iteration"] for r in rows] == [1, 2, 3]
    assert rows[0].get("sve_buffer_frac", 0) == 0          # before T1
    after = rows[1:]
    assert any(r.get("sve_buffer_frac", 0) > 0 for r in after)
    assert any(r.get("sve_searches", 0) > 0 for r in after)  # stage 2 ran
    assert all(0 <= r.get("sve_frac", 0) <= 1 for r in after)
    if ready_after == 5:
        # sve2 audit r1 LOW-1: the games really joined MID-BLOCK after T1
        # (offset 10 of 30), and each reopen dropped the SVE/bootstrap
        # cache at that step
        assert rows[1]["eager_collect_steps"] == [[2, 10]]
        assert rows[2]["eager_collect_steps"] == [[3, 10]]
        assert {40, 70} <= set(inval)


def test_supervisor_variant_suites_and_seed():
    tree = ast.parse(SUP.read_text(encoding="utf-8"))
    variants = ast.literal_eval(next(
        n for n in tree.body if isinstance(n, ast.Assign)
        and getattr(n.targets[0], "id", "") == "VARIANTS").value)
    arm, stem, flags = variants["ray"]
    assert arm == ARM and stem == "pong-sve2"
    assert flags == ["--ray", "--cpu-actor"]
    aud = ast.literal_eval(next(
        n for n in tree.body if isinstance(n, ast.Assign)
        and getattr(n.targets[0], "id", "") == "AUDITED").value)
    pre = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
               and n.name == "preflight")
    suites = [e.value for n in ast.walk(pre) if isinstance(n, ast.List)
              for e in n.elts if isinstance(e, ast.Constant)
              and str(e.value).startswith("tests/")]
    for t in ("tests/test_sve2.py", "tests/test_sve.py",
              "tests/test_bnrecal.py", "tests/test_reanalyze_refresh.py"):
        assert t in suites and t in aud, t
    assert set(suites) <= set(aud)
    r = subprocess.run([sys.executable, str(SUP), "ray", "0"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    assert r.returncode != 0 and "pre-registered as SEED 1" in r.stderr



# --------------------------------------------------------------------------
# the stop rule (sve2 audit r1 MEDIUM-1)
# --------------------------------------------------------------------------
def _stop_ns():
    tree = ast.parse(SUP.read_text(encoding="utf-8"))
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef)
            and n.name in ("twin_values", "stop_rule_step")]
    keep += [n for n in tree.body if isinstance(n, ast.Assign)
             and getattr(n.targets[0], "id", "") in (
                 "STOP_FROM", "STOP_RATIO", "STOP_ROWS")]
    ns = {"json": json, "os": __import__("os"), "TWIN_JSONL": "unused"}
    exec(compile(ast.Module(body=sorted(keep, key=lambda n: n.lineno),
                            type_ignores=[]), "stop", "exec"), ns)
    return ns


def test_stop_rule_streak_and_alert():
    ns = _stop_ns()
    assert (ns["STOP_FROM"], ns["STOP_RATIO"], ns["STOP_ROWS"]) == (25, 2.0, 3)
    twin = {i: 0.5 for i in range(5, 91, 5)}
    step, streak = ns["stop_rule_step"], 0
    seq = [(20, 9.0), (25, 1.2), (30, 1.1), (35, 0.9), (40, 1.5)]
    alerts = []
    for it, v in seq:
        ratio, streak, alert = step(it, v, twin, streak)
        alerts.append((it, streak, alert))
    # @20 is before STOP_FROM (ignored despite 18x); 25/30 > 2x, 35 resets,
    # 40 starts again -- never 3 in a row
    assert alerts == [(20, 0, False), (25, 1, False), (30, 2, False),
                      (35, 0, False), (40, 1, False)]
    streak = 0
    for it in (45, 50, 55):
        _, streak, alert = step(it, 1.01, twin, streak)
    assert streak == 3 and alert is True            # exactly the rule


@pytest.mark.parametrize("value,twinv", [(None, 0.5), (True, 0.5), ("1", 0.5),
                                         (1.5, None), (1.5, 0.0)])
def test_stop_rule_ignores_unusable_rows(value, twinv):
    ns = _stop_ns()
    twin = {} if twinv is None else {30: twinv}
    assert ns["stop_rule_step"](30, value, twin, 2) == (None, 2, False)


def test_twin_values_from_the_real_twin_rows(tmp_path):
    ns = _stop_ns()
    real = (ROOT / "benchmarks" / "results" / "pong-bnrecal-s1-2026-10-08" /
            "ablation-pong-skip4.jsonl")
    if real.exists():
        tv = ns["twin_values"](str(real))
        assert set(range(5, 91, 5)) <= set(tv)
        assert abs(tv[5] - 0.0861) < 1e-3
    p = tmp_path / "t.jsonl"
    p.write_text('{"iteration": 5, "value": 0.25}\n{"iteration": 10}\n'
                 'not json\n{"iteration": 15, "value": true}\n')
    assert ns["twin_values"](str(p)) == {5: 0.25}
    assert ns["twin_values"](str(tmp_path / "missing.jsonl")) == {}


def test_supervisor_prints_and_replays_the_stop_rule():
    src = SUP.read_text(encoding="utf-8")
    tree = ast.parse(src)
    emit = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "emit_new_rows")
    body = ast.unparse(emit)
    assert "stop_rule_step(" in body and "TD-referenced value" in body
    assert "ALERT: STOP RULE" in body
    # the relaunch replay over already-written rows
    loops = [n for n in tree.body if isinstance(n, ast.For)
             and "rows()" in ast.unparse(n.iter)]
    assert any("stop_rule_step(" in ast.unparse(n) for n in loops)
    assert "emit_new_rows.streak = 0" in src
