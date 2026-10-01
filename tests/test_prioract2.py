"""Policy-only acting without ReDo (2026-10-01): effzero-repro-faithful-
prioract-noredo (--ray --cpu-actor) and its sequential twin. Pins each
arm's delta against the run it is compared with."""
import sys

import pytest

sys.path.insert(0, str(__import__("pathlib").Path(
    __file__).resolve().parents[1] / "experiments"))
import ablation
import muzero

RAY = "effzero-repro-faithful-prioract-noredo"
SEQ = "effzero-repro-faithful-prioract-noredo-seq"
PRIOR = "effzero-repro-faithful-prioract"
NOREDO = "effzero-repro-faithful-searchact-noredo"
SEQVTREE = "effzero-repro-faithful-seqvtree"


def _eff_delta(a, b):
    a, b = ablation.ARMS[a], ablation.ARMS[b]

    def eff(spec, k):
        return spec[k] if k in spec else getattr(muzero, k, None)
    return {k for k in set(a) | set(b) if eff(a, k) != eff(b, k)}


def test_ray_arm_is_prioract_with_redo_off():
    a, b = ablation.ARMS[PRIOR], ablation.ARMS[RAY]
    raw = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert raw == {"REDO_EVERY", "STEP_DIAGNOSTICS", "_launch", "_doc"}
    assert b["REDO_EVERY"] == 0 and b["STEP_DIAGNOSTICS"] is True
    assert b["USE_SEARCH_ACTING"] is False and b["BIRTH_SEARCH"] is True


def test_the_two_variants_differ_only_in_launch_mode():
    assert _eff_delta(RAY, SEQ) == {"_launch", "_doc"}
    assert ablation.ARMS[RAY]["_launch"] == {"ray": True, "cpu_actor": True}
    assert ablation.ARMS[SEQ]["_launch"] == {"ray": False}


ACTING = {"USE_SEARCH_ACTING", "BIRTH_SEARCH", "PRIOR_ACT_STAGED_TEMP",
          "EVAL_PRIOR_ONLY"}


def test_ray_arm_differs_from_noredo_only_in_who_acts():
    """vs the search-acting run on the same stack and launch mode: only the
    acting mode (EVAL_PRIOR_ONLY is inert without search acting)."""
    assert _eff_delta(RAY, NOREDO) <= ACTING | {"_launch", "_doc"}
    assert _eff_delta(RAY, NOREDO) >= {"USE_SEARCH_ACTING", "BIRTH_SEARCH"}


def test_seq_arm_differs_from_seqvtree_only_in_who_acts():
    """...and the sequential twin vs seqvtree: the acting mode, plus ReDo
    knobs that are inert with REDO_EVERY 0."""
    d = _eff_delta(SEQ, SEQVTREE)
    assert d <= ACTING | {"REDO_NETS", "REDO_TAU", "_launch", "_doc"}
    assert d >= {"USE_SEARCH_ACTING", "BIRTH_SEARCH"}


@pytest.mark.parametrize("arm,use_ray,cpu_actor,ok", [
    (RAY, True, True, True), (RAY, False, False, False),
    (SEQ, False, False, True), (SEQ, True, True, False)])
def test_launch_pins(arm, use_ray, cpu_actor, ok):
    if ok:
        ablation.launch_gate(ablation.ARMS[arm], use_ray, cpu_actor)
    else:
        with pytest.raises(SystemExit, match="pins its launch mode"):
            ablation.launch_gate(ablation.ARMS[arm], use_ray, cpu_actor)


def test_supervisor_variants_match_the_pins():
    import ast
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1] / "benchmarks" /
           "supervisors" / "supervise_prioract2.py").read_text(encoding="utf-8")
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.Assign)
                and getattr(n.targets[0], "id", "") == "VARIANTS")
    for arm, _stem, flags in ast.literal_eval(node.value).values():
        want = ablation.ARMS[arm]["_launch"]
        assert ("--ray" in flags) == want["ray"], arm
        assert ("--cpu-actor" in flags) == want.get("cpu_actor", False), arm



def _rule_funcs():
    """exact_means / launch_decision from the supervisor's source, without
    running its top-level loop."""
    import ast
    import json
    import os
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1] / "benchmarks" /
           "supervisors" / "supervise_prioract2.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef)
            and n.name in ("exact_means", "launch_decision")]
    keep += [n for n in tree.body if isinstance(n, ast.Assign)
             and getattr(n.targets[0], "id", "") in ("RULE_ROWS", "RULE_RUNS")]
    ns = {"os": os, "json": json}
    exec(compile(ast.Module(body=sorted(keep, key=lambda n: n.lineno),
                            type_ignores=[]), "rule", "exec"), ns)
    return ns


def _write(d, run, vals):
    import json
    with open(d / f"probe_exhaustive_eval-{run}.jsonl", "a") as f:
        for it, v in vals.items():
            f.write(json.dumps({"run": run, "iteration": it, "exact_mean": v}) + "\n")


@pytest.mark.parametrize("seq_mean,want", [(12.0, "ray"), (12.3763, "ray"),
                                           (12.3764, "seq"), (20.0, "seq")])
def test_launch_rule_from_the_records(tmp_path, seq_mean, want):
    """noredo 5.8620 + |12.1953 - 5.6810| = 12.3763 (one operative number);
    the rule waits (None) until every row 60-90 exists."""
    ns = _rule_funcs()
    rows = range(60, 91, 5)
    for name, m in (("noredo", 5.8620), ("s0", 12.1953), ("s1", 5.6810)):
        _write(tmp_path, ns["RULE_RUNS"][name], {i: m for i in rows})
    _write(tmp_path, ns["RULE_RUNS"]["seqvtree"], {i: seq_mean for i in rows if i < 90})
    assert ns["launch_decision"](ns["exact_means"](str(tmp_path), ns["RULE_RUNS"])) == (None, None)
    _write(tmp_path, ns["RULE_RUNS"]["seqvtree"], {90: seq_mean})
    chosen, thr = ns["launch_decision"](ns["exact_means"](str(tmp_path), ns["RULE_RUNS"]))
    assert thr == pytest.approx(12.3763, abs=1e-9) and chosen == want


def test_launch_rule_on_todays_records():
    """The controls' and noredo's real records reproduce the CURRICULUM
    numbers (seqvtree's rows 60-90 do not exist yet)."""
    import pathlib
    ns = _rule_funcs()
    probes = pathlib.Path(__file__).resolve().parents[1] / "benchmarks" / "probes"
    m = ns["exact_means"](str(probes), ns["RULE_RUNS"])
    assert m["noredo"] == pytest.approx(5.862, abs=1e-3)
    assert m["s0"] == pytest.approx(12.1953, abs=1e-3)
    assert m["s1"] == pytest.approx(5.6810, abs=1e-3)
