"""Test G, stage 2b: the value nets on the kept states and reads (1)-(6)
(CURRICULUM "Next experiment (pre-registered) -- Test G"). Pins: the
registered numbers and the script's environment (stage 2a's); g = master minus
twin head by head, matched to G's (scale, discount) axes; every VOID route
(stage 2a's verdict or a failed gate inside it, G2's first clause, every pass
on its own) forms no g and writes a VOID summary, while missing inputs are
errors; G2's first clause as amended (the file says it is the first clause,
a PASS verdict, and the master's and twin's own reads decide); the
training-overlap exclusion (master or twin for the primary, the late twin's
own for read (3)) and the read beside it; G2's inside clause; each baseline
(the deployed prior's
entropy, the twin's value, T0's value head on the trainer's BN, base steps)
and its one orientation on the sample; the best baseline inside each draw;
HOLDS and NOT SHOWN on constructed worlds both ways and the verdict's logic
alone; read (4)'s 30 states; the bootstrap's unit (the sample's clusters from
gap.pkl, whole, the draws reproduced by hand with ranks recomputed) and an
undefined draw; reads (3), (4), (5), (6) by hand; the position table from
the logs read as probe_gap_nets reads them; the stamps (knobs, inputs),
resume, the summary never overwritten, the read-only inputs. Whole stages
run in-process on stub inputs (real net files, stand-in players and logs).
Slow: T0's real snapshot through both loaders, on made-up stacks."""
import contextlib
import hashlib
import json
import os
import pathlib
import stat
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROBES = ROOT / "benchmarks" / "probes"
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(PROBES))
import ablation  # noqa: E402
import probe_gap_check as gc  # noqa: E402
import probe_gap_nets as gn  # noqa: E402
import probe_gap_stage2a as s2  # noqa: E402
import probe_gap_stage2b as s2b  # noqa: E402

DRAWS = 300               # the unit tests' draws; the registered 10,000 run real


# --------------------------------------------------------------------------
# constructed worlds for the reads
# --------------------------------------------------------------------------
def world(n_strata=20, per=12, seed=0, **over):
    """The reads' data: strongly positioned returns (each stratum its own
    level), G = T2's mean return minus T0's, nets that track their players,
    g = master - twin, noise baselines, the twin's value = the twin."""
    rng = np.random.default_rng(seed)
    n = n_strata * per
    sid = np.repeat(np.arange(n_strata), per)
    level = rng.normal(0, 3, n_strata)[sid]
    t0 = level + rng.normal(size=n)
    t2 = level + 0.5 + rng.normal(size=n)
    master = t2 + 0.2 * rng.normal(size=n)
    twin = t0 + 0.2 * rng.normal(size=n)
    G = t2 - t0
    G_all = rng.normal(size=(n, 2, 4))
    g_all = rng.normal(size=(n, 2, 4))
    G_all[:, 0, 0], g_all[:, 0, 0] = G, master - twin
    d = {"G": G, "g": master - twin, "G_all": G_all, "g_all": g_all,
         "master": master, "twin": twin, "late": t0 + rng.normal(size=n),
         "t2_mean": t2, "t0_mean": t0,
         "per_delay": G[:, None] + rng.normal(size=(n, 16)),
         "prior_entropy": rng.normal(size=n), "twin_value": twin,
         "value_head": rng.normal(size=n),
         "steps": (20 * sid + rng.integers(0, 20, n)).astype(float),
         "table": level + rng.normal(size=n), "has_table": np.ones(n, bool),
         "sid": sid, "clusters": rng.integers(0, n // 3, n),
         "late_ok": np.ones(n, bool)}
    d["g_late"] = d["master"] - d["late"]
    d.update(over)
    return d


def primary(d, draws=DRAWS):
    signs = s2b.orientation(d)
    res = s2b.boot(s2b.read_primary, d, d["clusters"], signs, draws)
    return res, signs, s2b.primary_verdict(res["intervals"])


def noise(n, seed):
    return np.random.default_rng(seed).normal(size=n)


# --------------------------------------------------------------------------
# the module and its numbers
# --------------------------------------------------------------------------
def test_the_registered_numbers():
    assert s2b.DRAWS == 10_000 and not s2b.SMOKE
    assert s2.INTERVAL == (2.5, 97.5)
    assert gn.HEADS[s2b.head(*s2b.PRIMARY)] == ("clip", 0.997)
    assert s2b.BASELINES == ("prior_entropy", "twin_value", "value_head",
                             "steps")
    assert s2b.EXCLUDED_BY == ("master", "twin")
    assert s2b.NETS == tuple(gn.NETS)
    assert s2b.READ4_MIN == 30
    # its own bootstrap stream: not the pilot's (1-3) nor stage 2a's (4)
    assert s2b.BOOT_SEED == (gc.SEED, 5) and s2b.BOOT_SEED != s2.BOOT_SEED


def test_the_script_runs_in_stage_2as_environment():
    assert s2b.ENV == s2.ENV


def test_importing_the_stage_changes_nothing_process_wide():
    code = ("import os, sys; sys.path[:0] = [{p!r}, {e!r}, {r!r}]; "
            "import torch; threads = torch.get_num_threads(); "
            "import ablation; before = (ablation.ENV_NAME, ablation.ACTIONS, "
            "str(ablation.DEVICE)); "
            "import probe_gap_stage2b; "
            "assert (ablation.ENV_NAME, ablation.ACTIONS, "
            "str(ablation.DEVICE)) == before; "
            "assert torch.get_num_threads() == threads; "
            "assert 'probe_exhaustive_eval' not in sys.modules; "
            "assert 'probe_prefix_bn' not in sys.modules; "
            "assert os.environ.get('PROBE_ACTING_STATES') is None; "
            "assert os.environ.get('PROBE_ENV') is None; "
            "assert os.environ.get('PROBE_THREADS') is None; "
            "print('ok')").format(p=str(PROBES), e=str(ROOT / "experiments"),
                                  r=str(ROOT))
    env = {k: v for k, v in os.environ.items()
           if k not in ("PROBE_ACTING_STATES", "PROBE_ENV", "PROBE_THREADS")}
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=env,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-2000:]


# --------------------------------------------------------------------------
# g
# --------------------------------------------------------------------------
def test_g_is_master_minus_twin_head_by_head():
    n = 5
    master = 100.0 + np.arange(8)[None, :] * 10 + np.arange(n)[:, None]
    twin = np.tile(np.arange(8.0), (n, 1))
    g = s2b.gap_of(master, twin)
    assert g.shape == (n, len(gc.SCALES), len(gc.DISCOUNTS))
    for s, sc in enumerate(gc.SCALES):
        for k, dc in enumerate(gc.DISCOUNTS):
            h = gn.HEADS.index((sc, dc))
            assert np.allclose(g[:, s, k], master[:, h] - twin[:, h])
    assert (g > 0).all()                         # master above twin: g > 0
    assert np.allclose(s2b.gap_of(twin, master), -g)
    # the raw heads are the last four, the clip's the first four
    assert np.allclose(g[:, 1, 2], master[:, 6] - twin[:, 6])


def test_assemble_pairs_g_and_G_on_the_same_head(tmp_path):
    d = stub_inputs(tmp_path, scenario="holds")
    data, _ = s2b.assemble(d.states, d.keys, d.gap, d.nets_rec, d.t0_rec,
                           d.table_rec)
    pm, pt = d.nets_rec["pred"]["master"], d.nets_rec["pred"]["twin"]
    assert np.allclose(data["g"], pm[:, 0] - pt[:, 0])
    assert np.allclose(data["g_late"],
                       pm[:, 0] - d.nets_rec["pred"]["late"][:, 0])
    assert np.allclose(data["G"], d.gap["G"][:, 0, 0])
    assert np.allclose(data["G_all"], d.gap["G"])
    assert np.allclose(data["g_all"], s2b.gap_of(pm, pt))
    t0r, t2r = d.gap["returns"]["T0"], d.gap["returns"]["T2"]
    assert np.allclose(data["t0_mean"], t0r[:, :, 0, 0].mean(axis=1))
    assert np.allclose(data["t2_mean"], t2r[:, :, 0, 0].mean(axis=1))
    assert np.allclose(data["per_delay"], t2r[:, :, 0, 0] - t0r[:, :, 0, 0])
    assert np.allclose(data["twin_value"], pt[:, 0])
    assert np.array_equal(data["steps"], [s["steps"] for s in d.states])
    assert np.allclose(data["value_head"], d.t0_rec["value_head"])
    assert np.allclose(data["prior_entropy"], d.t0_rec["prior_entropy"])


def test_assemble_refuses_inputs_for_other_states_or_heads(tmp_path):
    d = stub_inputs(tmp_path, scenario="holds")
    args = (d.states, d.keys, d.gap, d.nets_rec, d.t0_rec, d.table_rec)
    s2b.assemble(*args)
    bad = dict(d.nets_rec, states=d.nets_rec["states"][::-1])
    with pytest.raises(SystemExit, match="nets.pkl"):
        s2b.assemble(d.states, d.keys, d.gap, bad, d.t0_rec, d.table_rec)
    bad = dict(d.nets_rec, heads=d.nets_rec["heads"][::-1])
    with pytest.raises(SystemExit, match="other heads"):
        s2b.assemble(d.states, d.keys, d.gap, bad, d.t0_rec, d.table_rec)
    bad = dict(d.gap, G=-d.gap["G"])
    with pytest.raises(SystemExit, match="G is not"):
        s2b.assemble(d.states, d.keys, bad, d.nets_rec, d.t0_rec,
                     d.table_rec)
    bad = dict(d.gap, keys=d.gap["keys"][::-1])
    with pytest.raises(SystemExit, match="strata"):
        s2b.assemble(d.states, d.keys, bad, d.nets_rec, d.t0_rec,
                     d.table_rec)


def test_read_5_reads_each_head_against_its_own_G():
    d = world()
    n = len(d["G"])
    G_all = np.random.default_rng(5).normal(size=(n, 2, 4))
    g_all = np.random.default_rng(6).normal(size=(n, 2, 4))
    g_all[:, 1, 2] = G_all[:, 1, 2]              # raw 0.98: g ranks G exactly
    g_all[:, 0, 3] = -G_all[:, 0, 3]             # clip 0.95: reversed
    g_all[:, 1, 0] = g_all[:, 1, 1]              # raw: two heads agree
    d.update(G_all=G_all, g_all=g_all)
    out = s2b.read_5(d, np.arange(n))
    assert out["raw 0.98"] == pytest.approx(1.0)
    assert out["clip 0.95"] == pytest.approx(-1.0)
    assert abs(out["clip 0.997"]) < 0.3
    assert out["g agree raw 0.997-0.99"] == pytest.approx(1.0)
    assert abs(out["G agree raw 0.997-0.99"]) < 0.3
    assert len([k for k in out if k.startswith("g agree")]) == 12


# --------------------------------------------------------------------------
# the gates
# --------------------------------------------------------------------------
def gates_rec(verdict="PASS-SO-FAR", reasons=(), g3=True, g5=True,
              g4=("T0", "T2"), bad_g4=()):
    return {"verdict": verdict, "reasons": list(reasons),
            "G3": {"mean": 0.4, "interval": [0.2, 0.6], "ok": g3},
            "G4": {n: {"ok": n not in bad_g4, "reasons": []} for n in g4},
            "G5": {"rho_half": 0.7, "reliability": 0.64, "bar": 0.5,
                   "ok": g5}}


def g2_rec(verdict=None, reasons=(), passes=(True, True, True),
           clause="first", r2=(0.3, 0.3, 0.3)):
    """g2.json as probe_gap_nets writes it since its audit fixes: it names
    its clause and its verdict is G2_FIRST_PASS or G2_FIRST_VOID."""
    rec = {"verdict": gn.G2_FIRST_PASS if verdict is None else verdict,
           "reasons": list(reasons), "gated": list(gn.GATED),
           "reads": {n: {"r2": r, "pass": p}
                     for n, p, r in zip(gn.NETS, passes, r2)}}
    if clause is not None:
        rec["clause"] = clause
    return rec


def test_g2s_verdict_names_are_the_nets():
    assert gn.G2_FIRST_PASS == "G2_FIRST_CLAUSE_PASS"
    assert gn.G2_FIRST_VOID == "G2_FIRST_CLAUSE_VOID"
    assert tuple(gn.GATED) == ("master", "twin")
    # the nets' own verdict function and this stage's reader agree
    for r2s in ((0.3, 0.2), (0.3, -0.1), (None, 0.4), (0.0, 0.5)):
        reads = {n: {"r2": r, "pass": r is not None and r > 0}
                 for n, r in zip(("master", "twin"), r2s)}
        reads["late"] = {"r2": -1.0, "pass": False}
        verdict, reasons = gn.g2_verdict(reads)
        rec = {"clause": "first", "verdict": verdict, "reasons": reasons,
               "gated": list(gn.GATED), "reads": reads}
        assert (s2b.g2_first_clause(rec) == []) == (
            verdict == gn.G2_FIRST_PASS)


def test_g2s_first_clause_passes_on_the_nets_pass_verdict():
    assert s2b.g2_first_clause(g2_rec()) == []
    # the late twin gates nothing
    assert s2b.g2_first_clause(g2_rec(passes=(True, True, False),
                                      r2=(0.3, 0.3, -0.5))) == []


@pytest.mark.parametrize("kw, says", [
    ({"clause": None}, "clause is None"),             # the old file
    ({"clause": "second"}, "clause is 'second'"),
    ({"verdict": "G2_FIRST_CLAUSE_VOID", "reasons": ["twin: R2 -0.1"]},
     "twin: R2 -0.1"),
    ({"verdict": "G2_FIRST_CLAUSE_VOID"}, "verdict G2_FIRST_CLAUSE_VOID"),
    ({"verdict": "PASS"}, "verdict PASS, not G2_FIRST_CLAUSE_PASS"),  # old
    ({"verdict": "G2_VOID"}, "verdict G2_VOID"),                      # old
    ({"verdict": "g2_first_clause_pass"}, "verdict g2_first_clause_pass"),
    ({"passes": (False, True, True)}, "master's read is pass False"),
    ({"passes": (True, None, True)}, "twin's read is pass None"),
    ({"r2": (0.3, 0.0, 0.3)}, "twin's read is pass True, R2 0.0"),
    ({"r2": (-0.2, 0.3, 0.3)}, "master's read is pass True, R2 -0.2"),
    ({"r2": (None, 0.3, 0.3)}, "R2 None"),
    ({"r2": (True, 0.3, 0.3)}, "R2 True"),
])
def test_g2s_first_clause_refuses(kw, says):
    """Whatever the verdict says, the master's and the twin's own reads
    decide; a file that is not the amended first clause is refused."""
    r = s2b.g2_first_clause(g2_rec(**kw))
    assert len(r) == 1 and r[0].startswith("G2: ") and says in r[0], r


def test_g2s_first_clause_needs_both_reads():
    rec = g2_rec()
    del rec["reads"]["twin"]
    r = s2b.g2_first_clause(rec)
    assert len(r) == 1 and "twin's read is pass None" in r[0]
    assert s2b.g2_first_clause({"clause": "first",
                                "verdict": gn.G2_FIRST_PASS})
    rec = g2_rec()
    del rec["verdict"]
    assert s2b.g2_first_clause(rec) == [
        f"G2: verdict None, not {gn.G2_FIRST_PASS}"]


def test_gate_reasons():
    assert s2b.gate_reasons(gates_rec()) == []
    assert s2b.gate_reasons(gates_rec(), g2_rec()) == []
    r = s2b.gate_reasons(gates_rec("VOID", ["G3: no"]), g2_rec())
    assert r == ["stage 2a: G3: no"]
    for kw in ({"g3": False}, {"g5": False}, {"bad_g4": ("T2",)},
               {"g4": ("T0",)}):
        r = s2b.gate_reasons(gates_rec(**kw), g2_rec())
        assert len(r) == 1 and r[0].startswith("stage 2a: PASS-SO-FAR")
    r = s2b.gate_reasons(gates_rec(), g2_rec(gn.G2_FIRST_VOID,
                                             ["twin: R2 -0.1"]))
    assert r == ["G2: twin: R2 -0.1"]
    r = s2b.gate_reasons(gates_rec(), g2_rec(passes=(True, False, True)))
    assert r == ["G2: verdict G2_FIRST_CLAUSE_PASS, but twin's read is pass "
                 "False, R2 0.3"]
    # the late twin gates nothing
    assert s2b.gate_reasons(gates_rec(), g2_rec(passes=(True, True, False))
                            ) == []
    # a VOID stage 2a is reported alone: g2.json is not consulted
    assert s2b.gate_reasons(gates_rec("VOID", ["G3: no"]),
                            g2_rec(clause=None)) == ["stage 2a: G3: no"]


def test_a_void_stage_2a_forms_no_g(route, monkeypatch, capsys):
    route.build(verdict="VOID", reasons=["G5: reliability 0.41 under 0.5"])
    monkeypatch.setattr(gn, "load", _forbidden)
    rec = route.run()
    assert rec["verdict"] == "VOID"
    assert rec["reasons"] == ["stage 2a: G5: reliability 0.41 under 0.5"]
    assert route.passes.calls == []
    assert "STAGE 2b: VOID" in capsys.readouterr().out
    saved = s2.read(route.out / s2b.SUMMARY,
                    s2b.stamp(route.states, None, None,
                              files=[route.gap / "gates.json"]))
    assert saved["verdict"] == "VOID" and saved["G2"] is None
    assert sorted(p.name for p in route.out.iterdir()) == [s2b.SUMMARY]
    with pytest.raises(SystemExit, match="never overwritten"):
        route.run()


def test_stage_2a_void_needs_no_g2(route):
    route.build(verdict="VOID", reasons=["G3: no"], g2=None)
    assert route.run()["verdict"] == "VOID"


@pytest.mark.parametrize("kw", [{"g3": False}, {"g5": False},
                                {"bad_g4": ("T0",)}])
def test_a_failed_gate_under_pass_so_far_is_void(route, kw):
    route.build(**kw)
    rec = route.run()
    assert rec["verdict"] == "VOID" and route.passes.calls == []


def test_g2s_first_clause_voids_the_stage(route, monkeypatch):
    route.build(g2=g2_rec(gn.G2_FIRST_VOID,
                          ["master: R2 -0.02 at the primary head on all "
                           "held-out stacks, not above 0"],
                          passes=(False, True, True), r2=(-0.02, 0.3, 0.3)))
    monkeypatch.setattr(gn, "load", _forbidden)
    rec = route.run()
    assert rec["verdict"] == "VOID" and route.passes.calls == []
    assert rec["reasons"] == ["G2: master: R2 -0.02 at the primary head on "
                              "all held-out stacks, not above 0"]
    assert rec["G2"] == {"clause": "first", "verdict": gn.G2_FIRST_VOID,
                         "reasons": ["master: R2 -0.02 at the primary head "
                                     "on all held-out stacks, not above 0"]}


def test_missing_inputs_are_errors_not_void(route):
    route.build(g2=None)
    with pytest.raises(SystemExit, match="nets have not finished"):
        route.run()
    (route.gap / "gates.json").chmod(stat.S_IREAD | stat.S_IWRITE)
    (route.gap / "gates.json").unlink()
    with pytest.raises(SystemExit, match="stage 2a has not finished"):
        route.run()
    assert not (route.out / s2b.SUMMARY).exists()


@pytest.mark.parametrize("name", s2b.PASSES)
def test_every_pass_refuses_on_a_void(route, monkeypatch, name):
    route.build(verdict="VOID", reasons=["G3: no"])
    monkeypatch.setattr(gn, "load", _forbidden)
    with pytest.raises(SystemExit, match="VOID"):
        route.pass_fn(name)
    assert not s2b.pass_file(route.out, name).exists()


def _forbidden(*a, **k):
    raise AssertionError("a net was loaded on a VOID")


# --------------------------------------------------------------------------
# G2's inside clause and the verdict
# --------------------------------------------------------------------------
def test_g2s_inside_clause_needs_both_nets():
    d = world()
    n = len(d["G"])
    res, _, (verdict, reasons) = primary(d)
    assert verdict == "HOLDS", reasons
    assert res["intervals"]["g2_master"][0] > 0
    assert res["intervals"]["g2_twin"][0] > 0
    # a twin that cannot rank T0's returns: NOT SHOWN, whatever g does
    bad = world(twin=noise(n, 1), twin_value=noise(n, 1))
    res, _, (verdict, reasons) = primary(bad)
    assert res["intervals"]["g"][0] > 0.5
    assert verdict == "NOT SHOWN"
    assert len(reasons) == 1 and "twin, T0's mean return" in reasons[0]
    bad = world(master=noise(n, 2))
    _, _, (verdict, reasons) = primary(bad)
    assert verdict == "NOT SHOWN"
    assert len(reasons) == 1 and "master net, T2's mean return" in reasons[0]


def test_g2s_inside_clause_reads_each_net_against_its_own_player():
    d = world()
    n = len(d["G"])
    # each net tracks the other player: both clauses fail
    swapped = world(master=d["t0_mean"] + 0.01 * noise(n, 3),
                    twin=d["t2_mean"] + 0.01 * noise(n, 4))
    out = s2b.read_primary(swapped, np.arange(n), s2b.orientation(swapped))
    assert out["g2_master"] < 0.3 and out["g2_twin"] < 0.3
    out = s2b.read_primary(d, np.arange(n), s2b.orientation(d))
    assert out["g2_master"] > 0.8 and out["g2_twin"] > 0.8


def test_holds_when_g_ranks_G_beyond_every_baseline():
    res, signs, (verdict, reasons) = primary(world())
    assert verdict == "HOLDS" and reasons == []
    assert res["intervals"]["g"][0] > 0
    assert res["intervals"]["g_minus_best"][0] > 0


def test_not_shown_when_a_baseline_ranks_as_well():
    d = world()
    d["steps"] = d["G"].copy()
    res, _, (verdict, reasons) = primary(d)
    assert verdict == "NOT SHOWN"
    assert res["intervals"]["g"][0] > 0          # g alone would pass
    assert any("best baseline" in r for r in reasons)


def test_not_shown_when_a_reversed_baseline_ranks_as_well():
    """Orientation: a baseline that ranks G backwards is turned round."""
    d = world()
    d["prior_entropy"] = -d["G"]
    res, signs, (verdict, _) = primary(d)
    assert signs["prior_entropy"] == -1.0
    assert res["point"]["b:prior_entropy"] == pytest.approx(1.0)
    assert verdict == "NOT SHOWN"


def test_not_shown_when_g_does_not_rank_G():
    d = world()
    d["g"] = noise(len(d["G"]), 9)
    _, _, (verdict, reasons) = primary(d)
    assert verdict == "NOT SHOWN"
    assert any(r.startswith("(1): rho_w(g, G) lower") for r in reasons)


def test_the_verdicts_logic():
    ok = {"g2_master": [0.1, 0.5], "g2_twin": [0.1, 0.5], "g": [0.2, 0.6],
          "g_minus_best": [0.01, 0.3]}
    assert s2b.primary_verdict(ok) == ("HOLDS", [])
    for name in ok:
        for lo in (0.0, -0.1):                   # above 0, strictly
            bad = dict(ok, **{name: [lo, 0.5]})
            v, r = s2b.primary_verdict(bad)
            assert v == "NOT SHOWN" and len(r) == 1
        v, r = s2b.primary_verdict(dict(ok, **{name: None}))
        assert v == "NOT SHOWN" and len(r) == 1
    # the upper ends never matter
    assert s2b.primary_verdict(
        {k: [v[0], 9.0] for k, v in ok.items()})[0] == "HOLDS"


# --------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------
def test_each_baseline_is_oriented_once_on_the_whole_sample():
    d = world()
    n = len(d["G"])
    d["value_head"] = -d["G"] + 0.5 * noise(n, 11)
    d["steps"] = noise(n, 12)                    # no relation at all
    signs = s2b.orientation(d)
    sid = list(d["sid"])
    for b in s2b.BASELINES:
        r = gc.rho_w(d[b], d["G"], sid)
        assert signs[b] == (-1.0 if r < 0 else 1.0)
    assert signs["value_head"] == -1.0 and signs["twin_value"] == -1.0
    res = s2b.boot(s2b.read_primary, d, d["clusters"], signs, DRAWS)
    j = res["names"].index("b:steps")
    col = res["draws"][:, j]
    # one sign for every draw: a null baseline's draws fall on both sides
    assert (col < 0).any() and (col > 0).any()
    for k, idx in enumerate(s2b.draws_of(d["clusters"], 20)):
        want = gc.rho_w(signs["steps"] * d["steps"][idx], d["G"][idx],
                        list(d["sid"][idx]))
        assert col[k] == pytest.approx(want, abs=1e-12)


def test_an_undefined_orientation_is_plus_one():
    d = world()
    d["steps"] = np.zeros(len(d["G"]))
    assert s2b.orientation(d)["steps"] == 1.0


def test_prior_entropy_by_hand():
    p = np.array([[0.5, 0.5, 0.0], [1.0, 0.0, 0.0], [0.25] * 2 + [0.5]])
    h = s2b.prior_entropy(p)
    assert h == pytest.approx([np.log(2), 0.0,
                               -(0.5 * np.log(0.25) + 0.5 * np.log(0.5))])


def test_the_t0_pass_reads_the_deployed_prior_and_the_trainers_value(route):
    route.build()
    route.pass_fn("t0")
    st = s2b.stamp(route.states, route.gap, route.nets)
    rec = s2.read(s2b.pass_file(route.out, "t0"), st)
    stacks = [s["stack"] for s in route.states]
    acting = np.array([softmax(logits(x, b"acting")) for x in stacks])
    assert np.allclose(rec["prior_entropy"], s2b.prior_entropy(acting))
    assert np.array_equal(rec["prior_argmax"], acting.argmax(axis=1))
    trainer = [logits(x, b"trainer-value")[0] for x in stacks]
    assert np.allclose(rec["value_head"], trainer)       # not the acting one
    assert rec["cross_check"] <= s2b.CROSS_TOL
    assert route.loaded == ["T0.pth"]
    assert rec["states"] == s2.ids(route.states)


def test_the_cross_check_refuses_another_prior(route, monkeypatch):
    route.build()
    monkeypatch.setattr(s2b, "trainer_run", trainer_stand_in(b"other"))
    with pytest.raises(SystemExit, match="not T0 as the pilot plays it"):
        route.pass_fn("t0")
    assert not s2b.pass_file(route.out, "t0").exists()


# --------------------------------------------------------------------------
# the best baseline inside each draw
# --------------------------------------------------------------------------
def test_the_best_baseline_is_taken_inside_each_draw():
    d = world()
    n = len(d["G"])
    d["prior_entropy"] = d["G"] + 1.2 * noise(n, 21)
    d["value_head"] = d["G"] + 1.2 * noise(n, 22)
    d["twin_value"], d["steps"] = noise(n, 23), noise(n, 24)
    signs = s2b.orientation(d)
    res = s2b.boot(s2b.read_primary, d, d["clusters"], signs, DRAWS)
    col = {k: res["draws"][:, res["names"].index(k)] for k in res["names"]}
    b = np.stack([col["b:" + k] for k in s2b.BASELINES], axis=1)
    assert np.allclose(col["best"], b.max(axis=1))
    assert np.allclose(col["g_minus_best"], col["g"] - b.max(axis=1))
    # the best changes between draws, so the full sample's best would differ
    won = set(b.argmax(axis=1).tolist())
    assert {0, 2} <= won
    full = s2b.BASELINES.index(max(s2b.BASELINES,
                                   key=lambda k: res["point"]["b:" + k]))
    assert (col["g_minus_best"] < col["g"] - b[:, full] - 1e-12).any()


# --------------------------------------------------------------------------
# read (4)
# --------------------------------------------------------------------------
def test_read_4_needs_30_states_at_g_at_most_0():
    g = np.r_[-np.ones(28), 0.0, np.ones(50)]    # 0 counts as g <= 0
    assert s2b.read4_counts(g) == {"low": 29, "min": 30, "read": False}
    g = np.r_[-np.ones(29), 0.0, np.ones(50)]
    assert s2b.read4_counts(g) == {"low": 30, "min": 30, "read": True}


def test_read_4_by_hand():
    d = {"g": np.array([-1.0, 0.0, 1.0, 2.0]),
         "G": np.array([-1.0, -2.0, 3.0, 4.0])}
    out = s2b.read_4(d, np.arange(4))
    assert out == {"n_low": 2.0, "mean_G_low": -1.5, "mean_G_rest": 3.5,
                   "share_low": 1.0, "share_rest": 0.0, "mean_diff": -5.0,
                   "share_diff": 1.0}
    out = s2b.read_4(d, np.array([2, 3, 3]))     # a draw with no g <= 0
    assert np.isnan(out["mean_G_low"]) and out["mean_G_rest"] == pytest.approx(
        11 / 3)


def test_read_4_unread_under_30_in_the_summary(route):
    route.build()
    rec = route.run()
    assert rec["read4"]["low"] == int((route.data()["g"] <= 0).sum())
    assert (rec["reads"]["read4"] is None) == (rec["read4"]["low"] < 30)


# --------------------------------------------------------------------------
# the bootstrap
# --------------------------------------------------------------------------
def test_the_draws_are_whole_clusters_with_ranks_recomputed():
    d = world()
    signs = s2b.orientation(d)
    res = s2b.boot(s2b.read_primary, d, d["clusters"], signs, 25)
    labels = np.asarray(d["clusters"])
    draws = list(s2.cluster_draws(labels, 25, seed=s2b.BOOT_SEED))
    for k, idx in enumerate(draws):
        # every drawn cluster comes whole, each draw as many as there are
        counts = {c: int((labels[idx] == c).sum()) for c in set(labels[idx])}
        sizes = {c: int((labels == c).sum()) for c in counts}
        assert all(counts[c] % sizes[c] == 0 for c in counts)
        assert sum(counts[c] // sizes[c] for c in counts) == len(set(labels))
        sid = list(d["sid"][idx])
        assert res["draws"][k, res["names"].index("g")] == pytest.approx(
            gc.rho_w(d["g"][idx], d["G"][idx], sid), abs=1e-12)
        assert res["draws"][k, res["names"].index("g2_twin")] == \
            pytest.approx(gc.rho_w(d["twin"][idx], d["t0_mean"][idx], sid),
                          abs=1e-12)
    assert res["point"]["g"] == pytest.approx(
        gc.rho_w(d["g"], d["G"], list(d["sid"])))


def test_one_cluster_leaves_no_spread():
    d = world(clusters=np.zeros(240, int))
    res, _, _ = primary(d, 30)
    for name, iv in res["intervals"].items():
        assert iv[0] == pytest.approx(res["point"][name])
        assert iv[1] == pytest.approx(res["point"][name])


def test_the_interval_is_two_sided_95_and_undefined_with_any_nan():
    v = np.arange(1001, dtype=float)
    assert s2b.interval(v) == [25.0, 975.0]
    assert s2b.interval(np.r_[v, np.nan]) is None
    assert s2b.interval([]) is None


def test_the_reads_resample_the_samples_clusters_from_gap_pkl(route,
                                                              monkeypatch):
    route.build(train={"master": (0, 1), "twin": (2,)})
    seen = []
    real = s2b.boot

    def spy(read, d, clusters, signs, draws=None):
        seen.append((read.__name__, np.asarray(clusters).copy()))
        return real(read, d, clusters, signs, draws)

    monkeypatch.setattr(s2b, "boot", spy)
    route.run()
    labels = np.asarray(route.inputs.gap["clusters"])
    keep = np.ones(len(labels), bool)
    keep[[0, 1, 2]] = False
    names = [n for n, _ in seen]
    assert names.count("read_primary") == 2      # the sample, all states
    for name, c in seen[:len(s2b.GROUPS)]:
        assert np.array_equal(c, labels[keep])
    assert np.array_equal(seen[-1][1], labels)


# --------------------------------------------------------------------------
# reads (3) and (6), the table
# --------------------------------------------------------------------------
def test_rank_gap_by_hand():
    a = np.array([1.0, 2.0, 3.0, 7.0])
    b = np.array([3.0, 2.0, 1.0, 5.0])
    out = s2b.rank_gap(a, b, [0, 0, 0, 1])
    assert out == pytest.approx([-2 / 3, 0.0, 2 / 3, 0.0])
    out = s2b.rank_gap(np.array([1.0, 1.0, 2.0]), np.zeros(3), [4, 4, 4])
    assert out == pytest.approx([-1 / 6, -1 / 6, 1 / 3])   # ties averaged


def test_read_3():
    d = world()
    n = len(d["G"])
    out = s2b.read_3(d, np.arange(n))
    assert out["twin"] == pytest.approx(gc.rho_w(d["g"], d["G"],
                                                 list(d["sid"])))
    same = world(late=d["twin"], value_head=d["twin"])
    same["g_late"] = same["master"] - same["late"]
    out = s2b.read_3(same, np.arange(n))
    assert out["late"] == pytest.approx(out["twin"])
    assert out["value_head_ranks"] == pytest.approx(out["twin_ranks"])
    # ranks only: any increasing transform of the value head reads the same
    warped = world(value_head=np.exp(d["twin"]) * 1e3 - 7)
    assert s2b.read_3(warped, np.arange(n))["value_head_ranks"] == \
        pytest.approx(out["twin_ranks"])


def test_read_3_leaves_out_the_late_twins_training_stacks():
    """The late twin has its own split: all of read (3) reads only the
    states whose stack is not in its training stacks, in every draw."""
    d = world()
    n = len(d["G"])
    ok = np.arange(n) % 4 != 1
    d["late_ok"] = ok
    d["g_late"] = d["G"].copy()
    d["g_late"][~ok] = -d["G"][~ok] * 50       # in-sample: would spoil it
    out = s2b.read_3(d, np.arange(n))
    sub = {k: (v[ok] if isinstance(v, np.ndarray) and len(v) == n else v)
           for k, v in d.items()}
    sub["late_ok"] = np.ones(int(ok.sum()), bool)
    assert out == pytest.approx(s2b.read_3(sub, np.arange(int(ok.sum()))))
    assert out["late"] == pytest.approx(1.0)
    idx = np.r_[0, 1, 1, 2, 5, 5, 6]
    keep = idx[ok[idx]]
    want = gc.rho_w(d["g"][keep], d["G"][keep], list(d["sid"][keep]))
    assert s2b.read_3(d, idx)["twin"] == pytest.approx(want)


def test_read_6_is_plain_spearman_and_the_table_on_its_states():
    d = world()
    n = len(d["G"])
    d["table"] = d["G"] * 2 + 1                  # monotone in G
    d["has_table"] = np.arange(n) % 3 != 0
    d["table"][~d["has_table"]] = np.nan
    out = s2b.read_6(d, np.arange(n))
    assert out["table"] == pytest.approx(1.0)
    t = d["has_table"]
    assert out["g_on_table_states"] == pytest.approx(
        gc.rho_w(d["g"][t], d["G"][t], [0] * int(t.sum())))
    assert out["g"] == pytest.approx(gc.rho_w(d["g"], d["G"], [0] * n))
    # across positions, unlike rho_w inside strata
    assert out["g"] != pytest.approx(gc.rho_w(d["g"], d["G"],
                                              list(d["sid"])))


def test_the_position_table_by_hand():
    lives = [{"index": 0, "returns": np.arange(25.0)},
             {"index": 1, "returns": np.full(3, 10.0)},
             {"index": 0, "returns": np.full(21, 1.0)}]
    table = s2b.position_table(lives)
    assert table == {(0, 0): [(sum(range(20)) + 20) / 40, 40],
                     (0, 1): [(sum(range(20, 25)) + 1) / 6, 6],
                     (1, 0): [10.0, 3]}
    tables = {"T2": {(0, 0): [5.0, 1], (1, 0): [2.0, 1]},
              "T0": {(0, 0): [1.5, 1], (2, 0): [0.0, 1]}}
    gap = s2b.table_gap(tables, [(0, 0), (1, 0), (2, 0), (0, 0)])
    assert gap[0] == 3.5 and gap[3] == 3.5
    assert np.isnan(gap[1]) and np.isnan(gap[2])


def fake_life(frames, tokens, rewards):
    """(traj, inner) of a life, as play_episode and play_token log them
    (test_gap_nets' fake_life)."""
    traj, inner, b, prev = [], [], 0, ()
    for i, k in enumerate(tokens):
        traj.append((frames[b:b + 4].reshape(12, 84, 84).copy(), i % 7, prev))
        inner.append([frames[4 + b + j].copy() for j in range(k - 1)])
        prev = tuple(float(r) for r in rewards[b:b + k])
        b += k
    traj.append((frames[b:b + 4].reshape(12, 84, 84).copy(), 0, prev))
    return traj, inner


def test_the_logs_are_read_as_the_nets_read_them(tmp_path, monkeypatch):
    results = tmp_path / "tree" / "benchmarks" / "results"
    rng = np.random.default_rng(4)
    for player, (iters, cut) in {"T2": (2, 2), "T0": (3, 2)}.items():
        log = ablation.OwnPlayLog(results / gc.PLAYERS[player]["run"]
                                  / "own-play-fake")
        k = 0
        for it in range(1, iters + 1):
            for _ in range(4):
                k += 1
                tokens = (4, 3, 2) if k % 2 else (1, 4, 4, 1)
                n = sum(tokens)
                frames = rng.integers(0, 256, (n + 4, 3, 84, 84),
                                      dtype=np.uint8)
                rewards = rng.choice([0.0, 10.0, 50.0, 200.0], size=n)
                traj, inner = fake_life(frames, tokens, rewards)
                log.write(it, traj, {"inner": inner, "real_done": k % 3 == 0,
                                     "temperature": 0.5}, True, n, 0)
        monkeypatch.setitem(gn.LOGS, player, {
            "last": iters, "cut": cut, "lives": 4 * iters, "base_steps": 0,
            "raw": 0.0, "later_lives": 4 * (iters - cut)})
    monkeypatch.setattr(gc, "RESULTS", results)
    for player in ("T2", "T0"):
        want, want_totals = gn.read_log(player)
        got, totals = s2b.log_returns(player)
        assert totals == want_totals
        assert [x["index"] for x in got] == [x["index"] for x in want]
        assert len(got) == len(want) == 8
        for a, b in zip(got, want):
            assert np.allclose(a["returns"],
                               gn.life_returns(b["rewards"])[:, gn.PRIMARY])
    assert totals["later_lives"] == 4


def test_the_table_pass_checks_the_logs_totals(route):
    route.build()
    route.pass_fn("table")
    assert route.totals_checked == ["T2", "T0"]
    rec = s2.read(s2b.pass_file(route.out, "table"),
                  s2b.stamp(route.states, route.gap, route.nets))
    for p in ("T2", "T0"):
        assert rec["tables"][p] == s2b.position_table(fake_lives(p))


# --------------------------------------------------------------------------
# stub inputs: a pilot, stage 2a's files, real net files; stand-in players
# --------------------------------------------------------------------------
N = 80
A = 7


def stub_states(seed=3):
    """80 states, lives 0 and 1, 4 strata a life, 10 states a stratum."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(N):
        life, j = divmod(i, 40)
        steps = (80 if life == 0 else 0) + 20 * (j // 10) + j % 10
        out.append({"game": i, "decision": 3, "life": life, "steps": steps,
                    "state": b"s%d" % i, "merged": [],
                    "stack": rng.integers(0, 256, (12, 84, 84),
                                          dtype=np.uint8)})
    return out


def logits(stack, salt):
    h = hashlib.blake2b(salt + stack.tobytes(), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(h, "little")).normal(size=A)


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def stand_in_run(prior_salt, value_salt):
    def initial_inference(stack):
        return (softmax(logits(stack, prior_salt)).astype(np.float32),
                float(logits(stack, value_salt)[0]))
    return SimpleNamespace(searcher=SimpleNamespace(
        network_manager=SimpleNamespace(initial_inference=initial_inference)))


@contextlib.contextmanager
def player_stand_in(name):
    assert name == "T0"
    yield stand_in_run(b"acting", b"acting-value")


def trainer_stand_in(acting_salt=b"acting"):
    @contextlib.contextmanager
    def trainer_run(blob, acting=False):
        assert blob == "BLOB"
        yield (stand_in_run(acting_salt, b"acting-value-2") if acting else
               stand_in_run(b"trainer", b"trainer-value")), ["SOME_KEY"]
    return trainer_run


def fake_lives(player):
    rng = np.random.default_rng(1 if player == "T2" else 2)
    return [{"index": k % 2, "returns": rng.normal(size=170)}
            for k in range(6)]


def net_files(d, train, scale):
    """Three seeds a net, random init (outputs vary with the stack), SD
    `scale[net]` on every head; training hashes `train[net]`."""
    d.mkdir(parents=True, exist_ok=True)
    for k, net in enumerate(gn.NETS):
        for seed in gn.SEEDS:
            torch.manual_seed(100 * k + seed)
            model = gn.Torso()
            gn.write(gn.net_file(d, net, seed), {
                "net": net, "seed": seed, "state_dict": model.state_dict(),
                "mean": [0.0] * 8, "sd": [float(scale[net])] * 8,
                "train_digests": gn._digest_tensor(set(train[net])),
                "split": {"held": [0]}})


def stub_inputs(tmp, scenario="holds", train=None, verdict="PASS-SO-FAR",
                reasons=(), g2="pass", clusters=None, **gate_kw):
    """Every input stage 2b reads, written where stage 2a and the nets
    would write them, under tmp, and the records the passes would make."""
    tmp = pathlib.Path(tmp)
    states = stub_states()
    keys = [gc.stratum(s["life"], s["steps"]) for s in states]
    stacks = np.stack([s["stack"] for s in states])
    digests = [gn.stack_digest(x) for x in stacks]
    train = train or {}
    train = {n: [digests[i] for i in train.get(n, ())] or [b"\0" * 16]
             for n in gn.NETS}
    pilot, gap, nets = tmp / "pilot", tmp / "gap", tmp / "nets"
    pilot.mkdir(parents=True)
    gc.write(pilot / "select.pkl", {"opening": 80, "kept": states,
                                    "keys": keys, "chosen": [0, 1, 2],
                                    "counts": {"kept": N}})
    net_files(nets, train, {"master": 100.0, "twin": 1.0, "late": 1.0})
    pred = {n: gn.load(n, nets).predict(stacks) for n in gn.NETS}
    rng = np.random.default_rng(7)
    shape = (N, 16, 2, 4)
    if scenario == "holds":
        def returns(p):
            base = np.stack([np.stack([p[:, s2b.head(s, k)] for k in range(4)],
                                      1) for s in range(2)], 1)
            spread = base.std(axis=0, keepdims=True)
            return base[:, None] + 0.05 * spread[:, None] * rng.normal(
                size=shape)
        t0, t2 = returns(pred["twin"]), returns(pred["master"])
    else:
        t0, t2 = rng.normal(size=shape), rng.normal(size=shape)
    clusters = [i % 40 for i in range(N)] if clusters is None else clusters
    gap.mkdir()
    st = s2.stamp(states)
    s2.write(gap / "gap.pkl", {
        "states": s2.ids(states), "merged": [[] for _ in states],
        "keys": keys, "clusters": clusters, "delays": list(s2.DELAYS),
        "scales": list(gc.SCALES), "discounts": list(gc.DISCOUNTS),
        "returns": {"T0": t0, "T2": t2}, "G": s2.real_gap(t0, t2)}, st)
    s2.write(gap / "gates.json", gates_rec(verdict, reasons, **gate_kw), st)
    if g2 == "pass":
        g2 = g2_rec()
    if g2 is not None:
        gn.write(nets / "g2.json", g2)
    acting = np.array([softmax(logits(x, b"acting")) for x in stacks])
    nets_rec = {"states": s2.ids(states),
                "heads": [list(h) for h in gn.HEADS], "pred": pred,
                "in_training": {n: np.array([x in set(train[n])
                                             for x in digests])
                                for n in gn.NETS}}
    t0_rec = {"states": s2.ids(states),
              "prior_entropy": s2b.prior_entropy(acting),
              "value_head": np.array([logits(x, b"trainer-value")[0]
                                      for x in stacks])}
    table_rec = {"tables": {p: s2b.position_table(fake_lives(p))
                            for p in ("T2", "T0")}}
    return SimpleNamespace(
        tmp=tmp, pilot=pilot, gap=s2.read(gap / "gap.pkl", st),
        gap_dir=gap, nets=nets, states=states, keys=keys,
        nets_rec=nets_rec, t0_rec=t0_rec, table_rec=table_rec)


class Passes:
    """Stands in for run_pass: the pass, in this process."""

    def __init__(self):
        self.calls, self.die_at = [], None

    def __call__(self, out, gap, nets, pilot, name):
        self.calls.append(name)
        if name == self.die_at:
            raise SystemExit(f"pass {name}: rc 1")
        {"nets": s2b.stage_nets, "t0": s2b.stage_t0,
         "table": s2b.stage_table}[name](out, gap, nets, pilot)


def read_only(d, on=True):
    for p in pathlib.Path(d).iterdir():
        os.chmod(p, stat.S_IREAD if on else stat.S_IREAD | stat.S_IWRITE)


def listing(d):
    return {p.name: (p.stat().st_mtime_ns,
                     hashlib.blake2b(p.read_bytes(), digest_size=16).digest())
            for p in sorted(pathlib.Path(d).iterdir())}


@pytest.fixture
def route(tmp_path, monkeypatch):
    """Stub inputs and stand-ins; build(...) writes the inputs, run() runs
    the stage in-process, pass_fn(name) one pass."""
    monkeypatch.setattr(s2b, "DRAWS", 200)
    monkeypatch.setattr(gc, "player", player_stand_in)
    monkeypatch.setattr(s2b, "trainer_run", trainer_stand_in())
    ns = SimpleNamespace(out=tmp_path / "out", loaded=[], totals_checked=[],
                         passes=Passes(), inputs=None)

    def load_blob(path):
        ns.loaded.append(str(path))
        return "BLOB"

    monkeypatch.setattr(s2b, "load_blob", load_blob)
    monkeypatch.setattr(gc, "snapshot", lambda name: pathlib.Path("T0.pth"))
    monkeypatch.setattr(s2b, "log_returns",
                        lambda player: (fake_lives(player), {"lives": 6}))
    monkeypatch.setattr(gn, "check_totals",
                        lambda player, lives, totals:
                        ns.totals_checked.append(player))
    monkeypatch.setattr(s2b, "run_pass", ns.passes)

    def build(**kw):
        ns.inputs = stub_inputs(tmp_path / "in", **kw)
        ns.pilot, ns.gap, ns.nets = (ns.inputs.pilot, ns.inputs.gap_dir,
                                     ns.inputs.nets)
        ns.states = ns.inputs.states
        ns.out.mkdir(exist_ok=True)
        for d in (ns.pilot, ns.gap, ns.nets):
            read_only(d)
        return ns.inputs

    ns.build = build
    ns.run = lambda: s2b.stage2b(ns.out, ns.gap, ns.nets, ns.pilot)
    ns.pass_fn = lambda name: {"nets": s2b.stage_nets, "t0": s2b.stage_t0,
                               "table": s2b.stage_table}[name](
        ns.out, ns.gap, ns.nets, ns.pilot)
    ns.data = lambda: s2b.assemble(
        ns.states, ns.inputs.keys, ns.inputs.gap, ns.inputs.nets_rec,
        ns.inputs.t0_rec, ns.inputs.table_rec)[0]
    yield ns
    for d in ("pilot", "gap", "nets"):
        if hasattr(ns, d):
            read_only(getattr(ns, d), on=False)


# --------------------------------------------------------------------------
# whole stages on stub inputs
# --------------------------------------------------------------------------
def test_the_route_to_holds(route, capsys):
    route.build()
    before = {d: listing(getattr(route, d)) for d in ("pilot", "gap", "nets")}
    rec = route.run()
    assert route.passes.calls == list(s2b.PASSES)
    assert rec["verdict"] == "HOLDS", rec["reasons"]
    assert rec["left_out"]["either"] == 0 and rec["beside"] is None
    assert rec["sample"] == {"states": N, "strata": 8, "clusters": 40}
    assert rec["ceiling"] == pytest.approx(0.8)          # sqrt(0.64)
    assert set(rec["reads"]) == set(s2b.GROUPS)
    assert rec["t0"]["missing_spec_keys"] == ["SOME_KEY"]
    out = capsys.readouterr().out
    assert "STAGE 2b PRIMARY: HOLDS" in out and "SMOKE" not in out
    st = s2b.stamp(route.states, route.gap, route.nets)
    saved = s2.read(route.out / s2b.SUMMARY, st)
    assert saved["verdict"] == "HOLDS"
    assert {d: listing(getattr(route, d)) for d in ("pilot", "gap", "nets")
            } == before
    files = sorted(p.name for p in route.out.iterdir())
    assert files == sorted([s2b.SUMMARY] + [f"{n}.pkl" for n in s2b.PASSES]
                           + [f"reads-{g}.pkl" for g in s2b.GROUPS])


def test_the_route_to_not_shown(route, capsys):
    route.build(scenario="null")
    rec = route.run()
    assert rec["verdict"] == "NOT SHOWN"
    assert "STAGE 2b PRIMARY: NOT SHOWN" in capsys.readouterr().out


def test_training_overlap_is_left_out_and_read_beside(route, capsys):
    route.build(train={"master": (0, 1), "twin": (1, 2),
                       "late": (2, 3, 4)})
    rec = route.run()
    lo = rec["left_out"]
    assert (lo["master"], lo["twin"], lo["either"]) == (2, 2, 3)
    assert lo["late_in_sample"] == 2           # 2 is out of the sample already
    assert lo["states"] == s2.ids(route.states)[:3]
    assert rec["sample"]["states"] == N - 3
    assert rec["beside"] is not None
    data = route.data()
    sid = list(data["sid"])
    assert rec["beside"]["point"]["g"] == pytest.approx(
        gc.rho_w(data["g"], data["G"], sid))
    keep = np.arange(N) >= 3
    assert rec["reads"]["primary"]["point"]["g"] == pytest.approx(
        gc.rho_w(data["g"][keep], data["G"][keep], list(data["sid"][keep])))
    assert "beside, all 80 kept states" in capsys.readouterr().out
    assert (route.out / "reads-all-primary.pkl").exists()
    # read (3) also leaves out the late twin's own training stacks (3 and 4)
    assert rec["read3_states"] == N - 5
    k3 = np.arange(N) >= 5
    assert rec["reads"]["read3"]["point"]["late"] == pytest.approx(
        gc.rho_w(data["g_late"][k3], data["G"][k3], list(data["sid"][k3])))
    assert rec["reads"]["read3"]["point"]["twin"] == pytest.approx(
        gc.rho_w(data["g"][k3], data["G"][k3], list(data["sid"][k3])))
    assert rec["G2_first"]["clause"] == "first"


def test_the_nets_pass_applies_every_net_seed_averaged(route):
    route.build(train={"twin": (5,)})
    route.pass_fn("nets")
    rec = s2.read(s2b.pass_file(route.out, "nets"),
                  s2b.stamp(route.states, route.gap, route.nets))
    stacks = np.stack([s["stack"] for s in route.states])
    for n in gn.NETS:
        assert np.allclose(rec["pred"][n], gn.load(n, route.nets).predict(
            stacks))
        assert rec["pred"][n].shape == (N, 8)
    assert np.flatnonzero(rec["in_training"]["twin"]).tolist() == [5]
    assert not rec["in_training"]["master"].any()


def test_a_stopped_stage_resumes_where_it_stopped(route, monkeypatch):
    route.build()
    route.passes.die_at = "t0"
    with pytest.raises(SystemExit, match="pass t0"):
        route.run()
    route.passes.die_at = None
    calls = []
    real = s2b.boot
    monkeypatch.setattr(s2b, "boot", lambda read, *a, **k: (
        calls.append(read.__name__), real(read, *a, **k))[1])
    rec = route.run()
    assert route.passes.calls == ["nets", "t0", "t0", "table"]
    assert len(calls) == len(s2b.GROUPS)
    (route.out / s2b.SUMMARY).unlink()
    (route.out / "reads-read5.pkl").unlink()
    calls.clear()
    again = route.run()
    assert calls == ["read_5"]
    assert json.dumps(again["reads"], sort_keys=True) == json.dumps(
        rec["reads"], sort_keys=True)


def test_the_summary_is_never_overwritten(route, monkeypatch):
    route.build()
    route.run()
    before = listing(route.out)
    # refused at once, before any input is read or any pass or read runs
    monkeypatch.setattr(s2, "load_states", _forbidden)
    monkeypatch.setattr(s2b, "boot", _forbidden)
    with pytest.raises(SystemExit, match="exists: the summary is never "
                                         "overwritten"):
        route.run()
    assert listing(route.out) == before


def test_files_from_other_knobs_or_inputs_are_refused(route, monkeypatch):
    route.build()
    route.run()
    (route.out / s2b.SUMMARY).unlink()
    monkeypatch.setattr(s2b, "DRAWS", 201)
    with pytest.raises(SystemExit, match="move it aside"):
        route.run()
    monkeypatch.setattr(s2b, "DRAWS", 200)
    route.run()
    (route.out / s2b.SUMMARY).unlink()
    # gap.pkl made again: everything after it is refused
    gap = route.gap / "gap.pkl"
    gap.chmod(stat.S_IREAD | stat.S_IWRITE)
    obj = s2.read(gap, s2.stamp(route.states))
    obj["G"] = obj["G"] + 0.0
    obj["clusters"] = list(obj["clusters"])[::-1]
    s2.write(gap, {k: v for k, v in obj.items() if k != "stamp"},
             s2.stamp(route.states))
    with pytest.raises(SystemExit, match="move it aside"):
        route.run()


def test_no_output_inside_the_inputs(route, tmp_path):
    route.build()
    for d in (route.pilot, route.gap, route.nets, route.gap / "sub"):
        with pytest.raises(SystemExit, match="never writes"):
            s2b.stage2b(d, route.gap, route.nets, route.pilot)
    with pytest.raises(SystemExit, match="never writes"):
        s2b.guard(gc.RESULTS / "x")
    s2b.guard(tmp_path / "elsewhere", route.pilot, route.gap, route.nets)


def test_a_smoke_stage_says_so(route, monkeypatch, capsys):
    route.build()
    monkeypatch.setattr(s2b, "SMOKE", True)
    rec = route.run()
    assert rec["smoke"] and "SMOKE STAGE 2b PRIMARY" in capsys.readouterr().out


def test_one_cluster_gives_degenerate_intervals_end_to_end(route):
    route.build(clusters=[0] * N)
    rec = route.run()
    p = rec["reads"]["primary"]
    assert p["intervals"]["g"][0] == pytest.approx(p["point"]["g"])


# --------------------------------------------------------------------------
# slow: T0's real snapshot through both loaders, on made-up stacks
# --------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.skipif(not gc.RESULTS.exists(), reason="no Test B2 run tree")
def test_t0s_real_snapshot_through_both_loaders():
    code = """
import os, sys, json
sys.path[:0] = [{p!r}, {e!r}, {r!r}]
import probe_gap_stage2b as s2b
os.environ.update(s2b.ENV)
import numpy as np
import probe_gap_check as gc
stacks = list(np.random.default_rng(0).integers(0, 256, (6, 12, 84, 84),
                                                dtype=np.uint8))
with gc.player("T0") as run:
    p = s2b.deployed_priors(run, stacks)
    v_acting = s2b.value_heads(run, stacks)
blob = s2b.load_blob(gc.snapshot("T0"))
with s2b.trainer_run(blob) as (run, missing):
    v = s2b.value_heads(run, stacks)
    p_trainer = s2b.deployed_priors(run, stacks)
with s2b.trainer_run(blob, acting=True) as (run, _):
    again = s2b.deployed_priors(run, stacks)
print(json.dumps({{"shape": list(p.shape), "cross": float(abs(p - again).max()),
    "trainer_vs_acting": float(abs(p - p_trainer).max()),
    "v": v.tolist(), "v_acting": v_acting.tolist(), "missing": missing,
    "entropy": s2b.prior_entropy(p).tolist()}}))
""".format(p=str(PROBES), e=str(ROOT / "experiments"), r=str(ROOT))
    env = dict(os.environ, **s2b.ENV)
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=env,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["shape"] == [6, gc.PLAYERS["T0"]["tokens"]]
    assert out["cross"] <= s2b.CROSS_TOL
    # the value head is read without the acting overlay: the BN states differ
    assert out["trainer_vs_acting"] > 1e-6 or not np.allclose(
        out["v"], out["v_acting"])
    assert all(np.isfinite(out["entropy"])) and all(np.isfinite(out["v"]))
