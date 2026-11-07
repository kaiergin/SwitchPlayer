"""Test B (Kai, 2026-10-27; CURRICULUM.md "Next experiment (pre-registered)
-- Test B: a fast lineage from scratch"): supervise_testb.py runs FIVE
generations F0-F4, one supervisor per variant, chained. Derived from
test_supervise_gen2.py: g2's tests adapted to the students f1-f4 (DS2's,
the old fixed master, to F{k-1}, in this same tree), plus the chain's own:
f0 (no wait, no prep, but the guard), each f{k} waiting for F{k-1}'s marker
and evals over its DYNAMIC rows (its ablation jsonl's in-run rows, its
final iteration K among them; row 0 and the offline curve only after a
student), THE STOP (target(): 20,000 x (k + 1), less the earlier
generations' final env steps: an overshoot never accumulates; a spent
budget stops the chain), the master F{k-1}'s final snapshot at K, the NEAR
RANDOM gate on its last two deployed rows, the prep's --from-log and
--master-iterations K, set_intact's master / own-play log / discount
checks, command() and the quick probe per variant, the guard / manifest /
prep-integrity order before each launch, the child's environment, the
stray-state rules (a student's own pre-checkpoint snapshots excepted), the
singleton per variant, NO PROGRESS and the row alerts at each run's
ceiling. CPU only, no launches.

Test B2 (Kai, 2026-10-29; the last section): the same supervisor bound to
the lineage "t" (configure; main() binds a variant's) -- three generations
T0-T2 of ~33,333 env steps at the constant discount 0.997^4, the prep's
rule "keep": its constants, stops, command, paths, quick probe and
set_intact's rule check; Test B's own constants come back unchanged.

Test C2 (Kai, 2026-11-02; the last section): the lineage "a" (a0, C plus
BBF's ramps, one run alone): its names, command and quick probe (the ramp
knobs pinned; every other lineage's pinned off), the standard-library
ramp_values against muzero.ramp_values, ramp_alert row by row (the ramp,
its trace and its windows: off the schedule, malformed, across a resume,
through Rows.emit with a rewritten row), the reset checks as r0's, and the
smokes counted as trainers.

E3 (Kai, 2026-11-05; the last section): the lineage "m" (m1, m2; generation
0 Test B2's T0, external, in its frozen tree): its names, paths and arms,
the external master's wait / gate / snapshot / own-play log / stop in that
tree, the prep's --mine and absolute --from-log, set_intact's "mine", the
reward_max bound at the mined vocabulary's longest token (m only), the
MINED line's nested tokens, and the quick probe passing with each mining
pin biting.

No test reads LIVE state (the distillation step's audit r5 HIGH): the `sup`
fixture points REPO (every run folder, probe jsonl, prep and snapshot of
the five runs) and E3's external tree into tmp, the manifest away and the
process table at an empty stand-in, and traps an unstubbed sleep; tests
that need a lineage build it in tmp (_finish). Only the constants tests read
the modules as deployed (`real_sup`, `g2sup`), and only their names -- but
for E3's two reads of T0's FROZEN run in Test B2's tree (read only; skipped
where that tree is absent)."""
import hashlib
import inspect
import json
import os
import pathlib
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]
                       / "experiments"))
import ablation  # noqa: E402
import distill  # noqa: E402
import muzero  # noqa: E402

FAST = ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-"
        "tokens-fast")
F = [f"{FAST}-f{k}" for k in range(5)]
V = [f"f{k}" for k in range(5)]
TK_ARM = ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-"
          "tokens-rgrow")
G0_ARM = ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-"
          "tokens-gen0")
DS2_ARM = ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-"
           "tokens-gen1-distill")
G2_ARM = ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-"
          "tokens-gen2-distill")
ROOT = pathlib.Path(__file__).resolve().parents[1]
SUPERVISORS = ROOT / "benchmarks" / "supervisors"
ENV_ID = "MsPacmanNoFrameskip-v4"
DATE = "2026-10-28"
NL = chr(10)
# The pre-registered discounts, F0-F4 (the gap 1 - g halved, capped).
DISCOUNTS = [0.97, 0.985, 0.9925, 0.99625, 0.997]
TK_VOCAB = ablation.ARMS[TK_ARM]["TOKEN_VOCAB"]
# A lineage's vocabularies: each generation drops one more token (never a
# 4-frame one, never slot 0).
DROP = ((0, 8), (0, 32), (5, 16), (1, 16))
VOCABS = [TK_VOCAB] + [tuple(t for t in TK_VOCAB if t not in DROP[:j + 1])
                       for j in range(4)]


def _load(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Stop(Exception):
    pass


def _stop(*a, **k):
    raise Stop


def _bounded(effect, n=5):
    """A time.sleep stand-in that runs `effect` and records each call --
    and raises Stop past `n` calls, so a broken wait fails instead of
    looping forever."""
    calls = []

    def sleep(s):
        calls.append(s)
        if len(calls) > n:
            raise Stop("the wait never returned")
        effect(len(calls))

    sleep.calls = calls
    return sleep


def _py(code):
    """`code` in a CPU-only child python, from this tree."""
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                          capture_output=True, text=True,
                          env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"})


# Several probes in ONE child python (an import of the trainer costs
# seconds): each exec'd alone, its outcome recorded ("ok" or the exception
# type and text), then ablation.ARMS and the module flags a probe asserts
# on restored -- a prelude that edits them never leaks into the next.
EACH = ("import json, sys" + NL
        + "sys.path[:0] = ['.', 'experiments']" + NL
        + "import muzero, ablation" + NL
        + "arms = {k: dict(v) for k, v in ablation.ARMS.items()}" + NL
        + "flags = {k: getattr(muzero, k) for k in ('PREFIX_BN_PER_STEP', "
          "'SCHEDULE_ENV_TARGET', 'OWN_PLAY_LOG')}" + NL
        + "out = []" + NL
        + "for code in json.loads(sys.stdin.read()):" + NL
        + "    try:" + NL
        + "        exec(code, {})" + NL
        + "        out.append('ok')" + NL
        + "    except BaseException as e:" + NL
        + "        out.append(type(e).__name__ + ': ' + str(e)[:800])" + NL
        + "    ablation.ARMS.clear()" + NL
        + "    ablation.ARMS.update({k: dict(v) for k, v in arms.items()})"
        + NL
        + "    for k, v in flags.items():" + NL
        + "        setattr(muzero, k, v)" + NL
        + "print('EACH ' + json.dumps(out))" + NL)


def _py_each(codes):
    r = subprocess.run([sys.executable, "-c", EACH], cwd=ROOT,
                       input=json.dumps(codes), capture_output=True,
                       text=True,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"})
    assert r.returncode == 0, r.stderr[-2000:]
    line = [l for l in r.stdout.splitlines() if l.startswith("EACH ")][-1]
    got = json.loads(line[5:])
    assert len(got) == len(codes)
    return got


def _no_sleep(s):
    raise AssertionError("a test reached an unstubbed time.sleep")


@pytest.fixture
def real_sup():
    """The module as deployed: its REPO is this tree. Only the constants
    tests read it -- as names, never the run files behind them."""
    return _load(SUPERVISORS / "supervise_testb.py", "supervise_testb")


@pytest.fixture
def sup(real_sup, monkeypatch, tmp_path):
    """The module with REPO (the five run folders, their probe jsonls, the
    preps, the snapshots and own-play logs) an empty temp tree, the
    manifest absent, the process table empty, host memory unknown, the real
    preflight / death snapshot disarmed and time.sleep trapped (a stop
    that is never settled would retry for minutes): no test may depend on
    this tree's own run state -- the preflight runs this suite in the run
    worktree beside the live runs. Tests that need a lineage build it in
    tmp (_finish)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(real_sup, "REPO", str(repo))
    monkeypatch.setattr(real_sup, "MANIFEST", str(tmp_path / "no-m.json"))
    # E3: the external T0's frozen tree, into tmp too -- and checked (audit
    # r2): a lapsed redirect must never let a test touch Test B2's tree
    monkeypatch.setitem(real_sup.LINEAGES["m"]["external"], "tree",
                        str(tmp_path / "external"))
    real_sup.configure("m")
    assert real_sup.tree_of("t0").startswith(str(tmp_path))
    assert real_sup.paths("t0", 0)[1].startswith(str(tmp_path))
    real_sup.configure("f")
    monkeypatch.setattr(real_sup, "_procs", lambda: iter(()))
    monkeypatch.setattr(real_sup, "commit_free_gb", lambda: None)

    def disarmed(*a, **k):
        raise AssertionError("a test reached the real preflight without "
                             "stubbing it")

    monkeypatch.setattr(real_sup, "preflight", disarmed)
    monkeypatch.setattr(real_sup, "snapshot", lambda *a: None)
    monkeypatch.setattr(real_sup.time, "sleep", _no_sleep)
    return real_sup


@pytest.fixture
def g2sup():
    return _load(SUPERVISORS / "supervise_gen2.py", "supervise_gen2")


def _rows(path, iterations, value=700.0, extra=()):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for i in iterations:
            f.write(json.dumps({"iteration": i, "env": ENV_ID,
                                "exact_mean": value}) + NL)
        for r in extra:
            f.write(json.dumps(r) + NL)


def _jsonl(sup, v, rows, arm=None, seed=0):
    """In-run rows {iteration: env_steps} appended to `v`'s own ablation
    jsonl (as the trainer writes them: its arm and seed)."""
    path = pathlib.Path(sup.paths(v, 0)[2])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for i, e in rows.items():
            f.write(json.dumps({"arm": arm or sup.VARIANTS[v], "seed": seed,
                                "env": ENV_ID, "iteration": i,
                                "env_steps": e, "return_mean": 500.0}) + NL)


def _in_run(final):
    """A generation's in-run rows: every 5 iterations, and its final one."""
    return sorted(set(range(5, final + 1, 5)) | {final})


def _probe(sup, v, suffix="-acting", offline=False):
    run = sup.run_name(v, 0) + ("-offline" if offline else "")
    return pathlib.Path(sup.probe_path(run, suffix))


def _log_dir(sup, v):
    return pathlib.Path(sup.REPO, *sup.own_play_rel(v, 0).split("/"))


def _finish(sup, v, final=27, env=20400, value=700.0, evals=True,
            files=True):
    """Generation `v` complete in tmp: its in-run rows (the last, at its
    final iteration, at `env` env steps), its completion marker, its final
    snapshot and own-play log iterations 1..final (`files`), and both exact
    evals holding every row (`evals`; a student's row 0 and offline curve
    too). Returns its in-run rows."""
    rows = _in_run(final)
    _jsonl(sup, v, {i: env if i == final else env * i // (final + 1)
                    for i in rows})
    pathlib.Path(sup.paths(v, 0)[3]).write_bytes(b"done")
    if files:
        snap = pathlib.Path(sup.snapshot_path(v, 0, final))
        snap.parent.mkdir(parents=True, exist_ok=True)
        snap.write_bytes(b"x")
        log = _log_dir(sup, v)
        log.mkdir(parents=True, exist_ok=True)
        for i in range(1, final + 1):
            (log / f"iter-{i:03d}.pkl").write_bytes(b"x")
    if evals:
        want = ([0] if sup.is_student(v) else []) + rows
        for s in ("-acting", ""):
            _rows(_probe(sup, v, s), want, value)
            if sup.is_student(v):
                _rows(_probe(sup, v, s, offline=True), (1, 2, 3, 4), value)
    return rows


def _master_like(path, k, final, arms=F, discounts=DISCOUNTS, **change):
    """F{k}'s final snapshot as distill.master_checks reads it (no nets:
    its arm, spec, steps and acting states): F0 as its arm after `final` x
    2,000 steps; a student as ablation.student_arm leaves its arm -- a
    student vocabulary (of its master's, kept as _master_vocab when that
    master is a student too), its prep's constant ceiling and discount --
    after 20,000 offline + `final` x 2,000 steps. `arms` / `discounts`: the
    lineage's (Test B's F0-F4 by default; Test B2's T0-T2)."""
    import torch
    spec = dict(ablation.ARMS[arms[k]])
    steps = final * 2000
    if k:
        c = 2.0 * k + 3.25
        spec.update({"TOKEN_VOCAB": VOCABS[k],
                     "_token_vocab_sha256": "a" * 64,
                     "_vocab_set_sha256": "b" * 64,
                     "REWARD_GROWTH_START": c, "REWARD_GROWTH_CEILING": c,
                     "_student_ceiling": c, "DISCOUNT": discounts[k],
                     "_student_discount": discounts[k]})
        if k >= 2:
            spec["_master_vocab"] = VOCABS[k - 1]
        steps += 20000
    blob = {"arm": arms[k], "spec": spec, "training_steps": steps,
            "iteration": final, "acting_states": {"representation": {}}}
    for key, val in change.items():
        if key in blob:
            blob[key] = val
        else:
            spec[key] = val
    torch.save(blob, path)
    return str(path)


def _master_ready(sup, v, final=27, **change):
    """`v`'s master F{k-1} complete in tmp (_finish), its final snapshot a
    master-like one (_master_like)."""
    m = sup.master_of(v)
    _finish(sup, m, final=final)
    return _master_like(sup.snapshot_path(m, 0, final), sup.gen(m), final,
                        arms=list(sup.VARIANTS.values()),
                        discounts=list(sup.DISCOUNTS.values()), **change)


def test_the_sup_fixture_redirects_every_path(sup, tmp_path):
    """Every path the supervisor builds lies under REPO, which the fixture
    points into tmp; a module-level absolute path other than the
    interpreter would have to be redirected too."""
    seen = 0
    for name in dir(sup):
        val = getattr(sup, name)
        if name.startswith("__") or not isinstance(val, str):
            continue
        if os.path.isabs(val) and name != "PY":
            assert val.startswith(str(tmp_path)), name
            seen += 1
    assert seen == 2                                 # REPO, MANIFEST
    for v in V:
        assert all(p.startswith(str(tmp_path)) for p in sup.paths(v, 0)[1:])
        assert sup.prep_dir(v).startswith(str(tmp_path))
        assert sup.snapshot_path(v, 0, 27).startswith(str(tmp_path))
        assert all(p.startswith(str(tmp_path))
                   for p in sup.probe_jsonls(v, 0))
    for v in V[1:]:
        _n, marker, probes, suplog, _s, _p, flag = sup.predecessor(v, 0)
        for p in [marker, suplog, flag] + [p for p, _r in probes]:
            assert p.startswith(str(tmp_path)), p
    # E3 (audit r2): m1 / m2 and the external T0's every path too
    sup.configure("m")
    for v in ("m1", "m2", "t0"):
        assert all(p.startswith(str(tmp_path)) for p in sup.paths(v, 0)[1:])
        assert sup.snapshot_path(v, 0, 47).startswith(str(tmp_path))
        assert sup.own_play_arg(v, 0).startswith(
            str(tmp_path)) or not os.path.isabs(sup.own_play_arg(v, 0))
    for v in ("m1", "m2"):
        assert sup.prep_dir(v).startswith(str(tmp_path))
        assert all(p.startswith(str(tmp_path))
                   for p in sup.probe_jsonls(v, 0))
        _n, marker, probes, suplog, _s, _p, flag = sup.predecessor(v, 0)
        for p in [marker, suplog, flag] + [p for p, _r in probes]:
            assert p.startswith(str(tmp_path)), p
    assert sup.own_play_arg("t0", 0).startswith(str(tmp_path / "external"))
    sup.configure("f")


# --------------------------------------------------------------------------
# the recipe, the constants
# --------------------------------------------------------------------------
def test_recipe_is_gen2s_with_the_fast_arms_the_cap_and_the_stop(sup,
                                                                 g2sup):
    assert sup.VARIANTS == dict(zip(V, F))
    assert sup.RUNS == {v: f"mspacman-fast-{v}" for v in V}
    assert sup.NAMES == {v: f"F{k}" for k, v in enumerate(V)}
    assert sup.SEEDS == {0: DATE}
    assert (sup.ITERATIONS, sup.NOMINAL_ITERATIONS, sup.target("f0")) == (
        60, 30, 20000)
    assert sup.TRAIN_STEPS == g2sup.TRAIN_STEPS == 2000
    _finish(sup, "f0", final=27, env=20400, evals=False, files=False)
    g2 = g2sup.command("g2", 0)
    for v, stop in (("f0", 20000), ("f1", 19600)):
        cmd = sup.command(v, 0)
        # G2's command (supervise_gen2.py g2) with the arm and the prep
        # moved, the --iterations cap, the stop -- and no prep for f0
        want = [sup.PY] + [x.replace(G2_ARM, sup.VARIANTS[v])
                            .replace(g2sup.PREP_REL, sup.prep_rel(v))
                           for x in g2[1:]]
        want[want.index("--iterations") + 1] = "60"
        i = want.index("--token-vocab")
        want = want[:i] + ["--stop-env-steps", str(stop)] + (
            want[i:] if v != "f0" else [])
        assert cmd == want, v
        assert sup.command(v, 0, 12345) == [
            "12345" if x == str(stop) else x for x in cmd]
        assert cmd[cmd.index("--arms") + 1] in ablation.ARMS
        assert "--resume" not in cmd and "--snapshot-weights" in cmd
        for flag, val in (("--iterations", "60"), ("--train-steps", "2000"),
                          ("--env", ENV_ID), ("--frameskip", "4"),
                          ("--episode-cap", "3000"),
                          ("--episodes-per-iter", "3"), ("--seed", "0"),
                          ("--eval-every", "5"), ("--checkpoint-every", "2"),
                          ("--stop-env-steps", str(stop))):
            assert cmd[cmd.index(flag) + 1] == val, (v, flag)
        j = cmd.index("--ray")
        assert cmd[j:j + 2] == ["--ray", "--cpu-actor"]
    f0 = sup.command("f0", 0)
    assert "--token-vocab" not in f0 and "--distill-set" not in f0
    f1 = sup.command("f1", 0)
    prep = f"benchmarks/results/fast-prep-f1-s0-{DATE}"
    assert f1[f1.index("--token-vocab") + 1] == prep + "/vocab.json"
    assert f1[f1.index("--distill-set") + 1] == prep + "/set.pkl"
    assert f1.count("--distill-set") == 1


def test_paths_the_run_folders_and_the_preps(real_sup, monkeypatch):
    sup = real_sup
    assert sup.TAG == ("-" + ablation.env_tag_fragment(ENV_ID) + "-skip4")
    src = inspect.getsource(ablation.save_weight_snapshot)
    assert 'f"snapshots{TAG}-{arm}-seed{seed}"' in src
    assert 'f"weights-iter{it:03d}.pth"' in src
    for k, v in enumerate(V):
        rel, base, jsonl, ckpt, deaths = sup.paths(v, 0)
        assert rel == f"benchmarks/results/mspacman-fast-{v}-s0-{DATE}"
        assert sup.run_name(v, 0) == rel.split("/")[-1]
        assert jsonl.endswith(os.path.join(
            rel.split("/")[-1], "ablation-mspacmannoframeskip-skip4.jsonl"))
        assert ckpt.endswith(f"resume-mspacmannoframeskip-skip4-{F[k]}"
                             f"-seed0.ckpt")
        assert deaths.endswith("deaths.log")
        assert pathlib.Path(base).parent == ROOT / "benchmarks" / "results"
        assert sup.prep_rel(v) == f"benchmarks/results/fast-prep-{v}-s0-{DATE}"
        assert pathlib.Path(sup.prep_dir(v)) == ROOT.joinpath(
            *sup.prep_rel(v).split("/"))
        # ablation's own names under the run folder (ABLATION_RESULTS =
        # rel, the trainer's cwd REPO): the own-play log, the snapshots
        monkeypatch.setattr(ablation, "RESULTS", pathlib.Path(rel))
        monkeypatch.setattr(ablation, "TAG", sup.TAG)
        assert ablation.own_play_dir(F[k], 0) == pathlib.Path(
            sup.own_play_rel(v, 0))
        assert sup.own_play_rel(v, 0) == (
            f"{rel}/own-play-mspacmannoframeskip-skip4-{F[k]}-seed0")
        assert pathlib.Path(sup.snapshot_path(v, 0, 27)) == ROOT.joinpath(
            *rel.split("/"), f"snapshots{sup.TAG}-{F[k]}-seed0",
            "weights-iter027.pth")
    assert pathlib.Path(sup.REPO) == ROOT           # runs from its own tree
    assert pathlib.Path(sup.MANIFEST) == SUPERVISORS / "testb_manifest.json"
    assert pathlib.Path(sup.probe_path("x", "-acting")) == (
        ROOT / "benchmarks" / "probes"
        / "probe_exhaustive_eval-x-acting.jsonl")
    assert (sup.MAX_ATTEMPTS, sup.PREP_ATTEMPTS, sup.PREP_RETRY_S) == (40, 3,
                                                                       900)
    assert (sup.NO_PROGRESS_DEATHS, sup.STALL_MIN_S) == (3, 300)
    assert sup.PACE_ALERT_MIN == 25.0 and sup.COMMIT_ALERT_GB == 4.0


def test_the_interpreter_override_is_testbs(real_sup):
    src = inspect.getsource(real_sup)
    assert 'os.environ.get("TESTB_PY", ' in src and "GEN2_PY" not in src


def test_the_arms_are_test_bs(real_sup):
    sup = real_sup
    for k, v in enumerate(V):
        spec = ablation.ARMS[sup.VARIANTS[v]]
        assert spec["_env_stop"] is True and spec["OWN_PLAY_LOG"] is True
        assert spec["PREFIX_BN_PER_STEP"] is True
        assert spec["_nominal_iterations"] == sup.NOMINAL_ITERATIONS
        assert sup.NOMINAL_ITERATIONS < sup.ITERATIONS        # only a cap
        assert sup.gen(v) == k and sup.is_student(v) == (k > 0)
        assert sup.master_of(v) == (V[k - 1] if k else None)
        if not k:
            continue
        assert spec["_master_arm"] == sup.VARIANTS[V[k - 1]]
        assert spec["_student"] == "distill"
        assert spec["_distill_offline_steps"] == sup.OFFLINE_STEPS == \
            spec["SCHEDULE_OFFSET_STEPS"] == 20000
        # the offline curve: a snapshot every 4,000 steps but the last (row
        # 0)
        assert tuple(range(1, spec["_distill_offline_steps"]
                           // spec["_distill_snapshot_every"])) == \
            sup.CURVE_ROWS == (1, 2, 3, 4)
        assert sup.master_offline(v) == distill.master_offline_steps(
            ablation.ARMS[F[k - 1]]) == (20000 if k > 1 else 0)
    f0 = ablation.ARMS[F[0]]
    assert f0["DISCOUNT"] == sup.DISCOUNTS["f0"] == 0.97
    assert f0["REWARD_GROWTH_START"] == f0["REWARD_GROWTH_CEILING"] == \
        sup.F0_CEILING == 1.0
    # the discount schedule is the prep's rule, step by step as a lineage
    # computes it (each from the previous one's float)
    assert [sup.DISCOUNTS[v] for v in V] == DISCOUNTS
    g = sup.DISCOUNTS["f0"]
    for v in V[1:]:
        g = distill.student_discount_rule(g)
        assert abs(g - sup.DISCOUNTS[v]) <= sup.DISCOUNT_TOL, v
    assert sup.DISCOUNT_CAP == ablation.STUDENT_DISCOUNT_CAP == \
        sup.DISCOUNTS["f4"]


def test_the_alert_bounds(real_sup):
    sup = real_sup
    with ablation.arm_config(F[0]):
        assert max(f for _, f in muzero.TOKEN_VOCAB) // 4 == \
            sup.MAX_TOKEN_STEPS == 8
    assert sup.NEAR_RANDOM == pytest.approx(sup.G_RANDOM + 0.2 * sup.G_HUMAN,
                                            abs=1e-9)
    assert (sup.G_RANDOM, sup.G_HUMAN, sup.NEAR_RANDOM) == (307.3, 973.9,
                                                            502.08)
    assert sup.REWARD_MAX_SLACK < 1e-3
    assert sup.REWARD_SUPPORT_EDGE == muzero.REWARD_CLIP == 400
    # far below the 3-decimal probe means' 1/2000 step in a 2-row mean
    assert sup.NEAR_RANDOM_TOL <= 1e-6 < 0.001 / 2


def _imports(src):
    """Every module name an import statement names at any depth: each name
    of `import a, b.c` (a, b), the module of `from m import x` (m) and its
    imported names (x: a submodule perhaps; audit r1 B-L1)."""
    import ast
    out = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            out.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                out.add(node.module.split(".")[0])
            out.update(a.name for a in node.names)
    return out


def test_imports_sees_every_name():
    src = ("import a, b.c" + NL + "from d.e import f, g as h" + NL
           + "def x():" + NL + "    import i" + NL)
    assert _imports(src) == {"a", "b", "d", "f", "g", "i"}


def test_supervisor_pins_what_the_runs_and_preflight_use(real_sup, g2sup):
    import re
    sup = real_sup
    for rel in sup.AUDITED:
        assert (ROOT / rel).exists(), rel
    assert len(set(sup.AUDITED)) == len(sup.AUDITED)
    assert set(sup.SUITES) <= set(sup.AUDITED)
    assert len(set(sup.SUITES)) == len(sup.SUITES) == 29
    # every suite supervise_gen2.py ran, after this supervisor's, Test B's,
    # Test C's, Test C2's, Test B2W's, E3's and Test Q's (first: -x stops on
    # them before the slow suites)
    assert sup.SUITES[:7] == ["tests/test_supervise_testb.py",
                              "tests/test_testb.py", "tests/test_testc.py",
                              "tests/test_testc2.py", "tests/test_testb2w.py",
                              "tests/test_e3.py", "tests/test_testq.py"]
    assert sup.SUITES[7:] == g2sup.SUITES
    assert set(g2sup.AUDITED) <= set(sup.AUDITED)
    assert set(sup.AUDITED) - set(g2sup.AUDITED) == {
        "tests/test_supervise_testb.py", "tests/test_testb.py",
        "tests/test_testc.py", "tests/test_testc2.py",
        "tests/test_testb2w.py", "tests/test_e3.py", "tests/test_testq.py",
        # E3's suite reads E3a's probe (and the probes it imports)
        "benchmarks/probes/probe_e3a_sequences.py",
        "benchmarks/probes/probe_action_census.py",
        "benchmarks/probes/probe_e1_equivalence.py",
        "benchmarks/supervisors/supervise_testb.py"}
    assert [r for r in sup.AUDITED if not r.endswith(".py")] == [
        "pytest.ini", "benchmarks/probes/step0_prefix_bn/states.npz"]
    for rel in ("experiments/ablation.py", "experiments/distill.py",
                "experiments/ray_actors.py",
                "benchmarks/probes/probe_exhaustive_eval.py",
                "benchmarks/probes/probe_prefix_bn.py",
                "benchmarks/supervisors/supervise_gen1.py",
                "benchmarks/supervisors/supervise_gen2.py"):
        assert rel in sup.AUDITED
    homes = ("", "experiments/", "benchmarks/probes/", "tests/")
    # the suites: what they import (at any depth) or read
    for rel in sup.SUITES:
        src = (ROOT / rel).read_text(encoding="utf-8")
        for mod in re.findall(r"from tests\.(\w+) import", src):
            assert f"tests/{mod}.py" in sup.AUDITED, (rel, mod)
        for name in re.findall(r"(supervise_\w+)\.py", src):
            assert f"benchmarks/supervisors/{name}.py" in sup.AUDITED, \
                (rel, name)
        for name in re.findall(r"(probe_\w+)\.py", src):
            if (ROOT / "benchmarks" / "probes" / f"{name}.py").exists():
                assert f"benchmarks/probes/{name}.py" in sup.AUDITED, \
                    (rel, name)
        for name in _imports(src):
            for home in homes:
                if (ROOT / f"{home}{name}.py").exists():
                    assert f"{home}{name}.py" in sup.AUDITED, (rel, name)
    # the code the runs, the preps and the probes execute: every repo
    # module it imports, transitively
    todo = ["experiments/ablation.py", "experiments/distill.py",
            "benchmarks/probes/probe_exhaustive_eval.py",
            "benchmarks/probes/probe_prefix_bn.py"]
    seen = set()
    while todo:
        rel = todo.pop()
        if rel in seen:
            continue
        seen.add(rel)
        assert rel in sup.AUDITED, rel
        for name in _imports((ROOT / rel).read_text(encoding="utf-8")):
            for home in homes[:3]:
                if (ROOT / f"{home}{name}.py").exists():
                    todo.append(f"{home}{name}.py")
    assert {"muzero.py", "experiments/ray_actors.py"} <= seen


def test_supervisor_manifest_catches_an_edit(sup, monkeypatch, tmp_path):
    import shutil
    tree = tmp_path / "tree"
    for rel in sup.AUDITED:
        dst = tree / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, dst)
    monkeypatch.setattr(sup, "REPO", str(tree))
    monkeypatch.setattr(sup, "MANIFEST", str(tmp_path / "m.json"))
    (tmp_path / "m.json").write_text(json.dumps(sup.file_hashes(str(tree))))
    assert sup.manifest_check() == (True, "")
    for rel in ("experiments/distill.py", "experiments/ablation.py",
                "benchmarks/supervisors/supervise_testb.py",
                "benchmarks/supervisors/supervise_gen2.py",
                "tests/test_supervise_testb.py", "tests/test_testb.py"):
        p = tree / rel
        orig = p.read_bytes()
        p.write_bytes(orig + b"\n# edit\n")
        ok, why = sup.manifest_check()
        assert not ok and rel in why
        # CRLF vs LF: the same. Flip whichever way differs -- the run
        # worktree is CRLF (autocrlf), the dev tree LF (the distillation
        # step's audit r6 HIGH: LF -> CRLF on a CRLF file wrote \r\r\n).
        lf = orig.replace(b"\r\n", b"\n")
        flipped = lf if lf != orig else lf.replace(b"\n", b"\r\n")
        assert flipped != orig
        p.write_bytes(flipped)
        assert sup.manifest_check() == (True, "")
    (tmp_path / "m.json").write_text("[1, 2]")
    assert sup.manifest_check() == (False, "manifest check: not a JSON "
                                           "object")


def test_supervisor_write_manifest_refuses_a_dirty_tree(sup, monkeypatch,
                                                         tmp_path):
    monkeypatch.setattr(sup, "MANIFEST", str(tmp_path / "m.json"))
    monkeypatch.setattr(sup.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout=" M experiments/distill.py\n", stderr=""))
    assert sup.main(["--write-manifest"]) == 1
    assert not (tmp_path / "m.json").exists()


def test_supervisor_refuses_bad_arguments(sup, monkeypatch, capsys):
    def no(*a, **k):
        raise AssertionError("a bad argv got past the usage check")

    for name in ("other_supervisors", "run_done", "wait_external",
                 "preflight", "live_scripts", "settle_stop"):
        monkeypatch.setattr(sup, name, no)
    for argv in ([], ["f0"], ["0"], ["f0", "1"], ["f5", "0"], ["g2", "0"],
                 ["ds2", "0"], ["F0", "0"], ["f", "0"], ["f01", "0"],
                 ["f-1", "0"], ["f0", "x"], ["f0", "0", "1"], ["0", "f0"],
                 ["f1", "0", "--no-wait"], ["f1", "0", "--stop-env-steps"],
                 # Test B2's lineage: three generations, seed 0
                 ["t3", "0"], ["t", "0"], ["t0", "1"], ["T0", "0"],
                 ["t01", "0"], ["t1", "0", "1"],
                 # Test C's: one run each, seed 0
                 ["r1", "0"], ["n1", "0"], ["r0", "1"], ["n0", "1"],
                 ["R0", "0"], ["r", "0"], ["n", "0"],
                 # Test C2's: one run, seed 0
                 ["a1", "0"], ["a0", "1"], ["A0", "0"], ["a", "0"],
                 ["a01", "0"], ["a0", "0", "1"],
                 # Test B2W's: three generations, seed 0
                 ["w3", "0"], ["w", "0"], ["w0", "1"], ["W0", "0"],
                 ["w01", "0"], ["w1", "0", "1"],
                 # E3's: m1 and m2 only (its generation 0 is Test B2's T0,
                 # never run here), seed 0
                 ["m0", "0"], ["m3", "0"], ["m", "0"], ["m1", "1"],
                 ["M1", "0"], ["m01", "0"], ["m1", "0", "1"]):
        assert sup.main(argv) == 2, argv
    out = capsys.readouterr().out
    assert ("usage: supervise_testb.py <variant: ['f0', 'f1', 'f2', 'f3', "
            "'f4', 't0', 't1', 't2', 'r0', 'n0', 'a0', 'w0', 'w1', 'w2', "
            "'m1', 'm2', 'q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', "
            "'q8', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8']> "
            "<seed: [0]>") in out


def test_no_generation_2_leftovers(real_sup):
    """No fixed master in another tree, no fixed rows, no fallback, no
    per-variant equality branch (the chain is master_of / is_student), and
    no torch in the long-lived supervisor."""
    src = inspect.getsource(real_sup)
    for word in ("GEN2_PY", "MASTER_TREE", "MASTER_RUN", "MASTER_SNAPSHOT",
                 "MASTER_ITERATIONS", "MASTER_ABANDONED", "gen1b-run",
                 "gen2-prep", "gen2_manifest", "master_mean_60_90",
                 "window_mean", "60-90", "DS2", "no_wait", "NO-WAIT",
                 "vocab_is_ds", "DS_ROWS", "variant ==", "torch"):
        assert word not in src, word


# --------------------------------------------------------------------------
# the quick probe
# --------------------------------------------------------------------------
@pytest.mark.parametrize("k", range(5))
def test_the_quick_probe_passes(sup, k):
    """f0's on the real arms alone; a student's ends with the master checks
    on a master-like final snapshot of F{k-1} in tmp (the preflight's real
    probe reads the real one)."""
    if k:
        _master_ready(sup, V[k])
    r = _py(sup.quick(V[k], 0))
    assert r.returncode == 0, r.stderr[-2000:]


def test_the_quick_probe_ends_with_the_master_checks_on_its_final_snapshot(
        sup):
    with pytest.raises(SystemExit, match="F1 has no completion marker"):
        sup.quick("f2", 0)
    _finish(sup, "f1", final=31, env=19950, evals=False)
    probe = sup.quick("f2", 0)
    snap = sup.snapshot_path("f1", 0, 31)
    assert probe.endswith(f"distill.master_checks({snap!r}, 31)")
    assert snap.endswith(os.path.join(
        f"mspacman-fast-f1-s0-{DATE}",
        f"snapshots-mspacmannoframeskip-skip4-{F[1]}-seed0",
        "weights-iter031.pth"))
    assert "master_checks(" not in sup.quick("f0", 0)


# A master the prep would refuse, on f1's (F0 the master) or f2's (F1, a
# student master) probe: (k, the change to F{k-1}'s final snapshot, why).
REFUSED = [
    # an edit to F1's arm in this tree (its _doc included)
    (2, {"_doc": "edited"}, "beyond the student keys: ['_doc']"),
    (2, {"PER_BETA_END": 0.5}, "beyond the student keys"),
    # not F1's final snapshot
    (2, {"training_steps": 27 * 2000}, "after 20000 offline steps"),
    (2, {"iteration": 26}, "after 20000 offline steps"),
    # its prep's ceiling or discount not held
    (2, {"DISCOUNT": 0.99}, "_student_discount"),
    (2, {"REWARD_GROWTH_START": 1.0}, "constant _student_ceiling"),
    (2, {"acting_states": {}}, "no acting_states"),
    (2, {"arm": "nope"}, "arm 'nope' unknown"),
    # F0 (from scratch): no offline steps, its arm's vocabulary
    (1, {"training_steps": 20000 + 27 * 2000},
     "is not the end of a 27-iteration run"),
    (1, {"iteration": 30}, "is not the end of a 27-iteration run"),
    (1, {"TOKEN_VOCAB": VOCABS[1]}, "its TOKEN_VOCAB is not its arm's"),
]


def test_the_quick_probe_fails_on_a_master_the_prep_would_refuse(sup,
                                                                 tmp_path):
    probes = {k: (_master_ready(sup, V[k]), sup.quick(V[k], 0))
              for k in (1, 2)}
    codes = []
    for n, (k, change, _why) in enumerate(REFUSED):
        snap, probe = probes[k]
        assert snap == sup.snapshot_path(V[k - 1], 0, 27)
        assert probe.count(repr(snap)) == 1
        case = _master_like(tmp_path / f"case{n}.pth", k - 1, 27, **change)
        codes.append(probe.replace(repr(snap), repr(case)))
    got = _py_each(codes + [probes[1][1], probes[2][1]])
    for g, (k, change, why) in zip(got, REFUSED):
        assert g.startswith("SystemExit: ") and why in g, (change, g)
    assert got[-2:] == ["ok", "ok"]                  # the controls


def test_the_quick_probe_tells_the_arms_apart(sup):
    """As the student, a neighbour's arm (F{k-1}'s, F{k+1}'s: another
    master), Generation 2's or TK's fail; as the master, F0 for a student
    master (and a student for F0), G0's and DS2's (no flags) fail; f0's
    probe takes no student and no older arm."""
    for k in (1, 2):
        _master_ready(sup, V[k])
    cases = []
    p0 = sup.quick("f0", 0)
    s0 = f"ARMS[{F[0]!r}]"
    assert p0.count(s0) == 1
    for other in (F[1], G0_ARM, DS2_ARM, TK_ARM):
        cases.append(p0.replace(s0, f"ARMS[{other!r}]"))
    probes = [p0]
    for k, swaps in ((1, (("s", F[0]), ("s", F[2]), ("s", DS2_ARM),
                          ("m", G0_ARM), ("m", F[1]))),
                     (2, (("s", F[1]), ("s", F[3]), ("s", G2_ARM),
                          ("m", F[0]), ("m", DS2_ARM)))):
        p = sup.quick(V[k], 0)
        probes.append(p)
        student, master = f"ARMS[{F[k]!r}]", f"ARMS[{F[k - 1]!r}]"
        assert p.count(student) == 1 and p.count(master) == 1
        for which, other in swaps:
            old = student if which == "s" else master
            swapped = p.replace(old, f"ARMS[{other!r}]")
            assert swapped != p
            cases.append(swapped)
    got = _py_each(cases + probes)
    for g in got[:len(cases)]:
        assert g.startswith(("AssertionError", "KeyError")), g
    assert got[len(cases):] == ["ok"] * 3            # the controls


def test_each_assert_of_the_quick_probe_bites_alone(sup):
    """Audit r2 A-NIT-1 (Test 2's), for Test B: with everything else the
    real arms, each flag alone fails the probe -- the arm's per-step BN,
    own-play log, env stop or planned length off, its master's flag off,
    a module default on, a student's discount not from its prep, its
    master's offline steps, F0's discount, student-ness or sign clip."""
    for k in (1, 2):
        _master_ready(sup, V[k])
    a = "ablation.ARMS"
    breaks = {
        "f0": [f"{a}[{F[0]!r}]['DISCOUNT'] = 0.997",
               f"{a}[{F[0]!r}]['_student'] = 'fresh'",
               f"{a}[{F[0]!r}]['REWARD_GROWTH_CEILING'] = 2.0",
               f"{a}[{F[0]!r}]['OWN_PLAY_LOG'] = False",
               f"{a}[{F[0]!r}]['_nominal_iterations'] = 60",
               "muzero.OWN_PLAY_LOG = True",
               "muzero.SCHEDULE_ENV_TARGET = 20000"],
        "f1": [f"{a}[{F[1]!r}]['PREFIX_BN_PER_STEP'] = False",
               f"{a}[{F[0]!r}]['PREFIX_BN_PER_STEP'] = False",
               "muzero.PREFIX_BN_PER_STEP = True",
               f"{a}[{F[1]!r}]['_env_stop'] = False",
               f"{a}[{F[1]!r}]['_discount_from_prep'] = False",
               f"{a}[{F[1]!r}]['SCHEDULE_OFFSET_STEPS'] = 0"],
        "f2": [f"{a}[{F[1]!r}]['_distill_offline_steps'] = 4000",
               f"{a}[{F[1]!r}]['OWN_PLAY_LOG'] = False",
               f"{a}[{F[2]!r}]['_master_iterations'] = 27"]}
    codes, controls = [], []
    for v, preludes in breaks.items():
        probe = sup.quick(v, 0)
        anchor = f"spec = ablation.ARMS[{sup.VARIANTS[v]!r}]; "
        assert probe.count(anchor) == 1
        for prelude in preludes:
            codes.append(probe.replace(anchor, prelude + "; " + anchor))
        controls.append(probe)
    got = _py_each(codes + controls)
    for code, g in zip(codes, got):
        assert g.startswith("AssertionError"), (code[-300:], g)
    assert got[len(codes):] == ["ok"] * 3            # the controls


def test_the_quick_probe_is_bounded(monkeypatch, tmp_path):
    """Audit r1 B-N3 (PB's r1 B-N5): a hung quick probe is a PREFLIGHT
    not-ready reason (the loop retries), never a preflight stalled for
    good; the suites are not run after it -- nor anything while a
    student's master has no final row. The real preflight, REPO in tmp,
    every subprocess stubbed."""
    pre = _load(SUPERVISORS / "supervise_testb.py", "supervise_testb_pre")
    monkeypatch.setattr(pre, "REPO", str(tmp_path / "repo"))
    monkeypatch.setattr(pre, "manifest_check", lambda: (True, ""))
    calls, outcome = [], ["hang"]

    def run(args, **k):
        calls.append((list(args), k))
        if args[1] == "-c" and outcome[0] == "hang":
            raise subprocess.TimeoutExpired(args, k["timeout"])
        if args[1] == "-c" and outcome[0] == "fail":
            return SimpleNamespace(returncode=1, stdout="",
                                   stderr="x" + NL + "AssertionError")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pre.subprocess, "run", run)
    assert pre.preflight("f0", 0) == (False, "quick probe timed out")
    assert len(calls) == 1                         # no suites after it
    args, k = calls[0]
    assert args == [pre.PY, "-c", pre.quick("f0", 0)]
    assert k["timeout"] == pre.QUICK_TIMEOUT_S == 900
    assert k["env"]["CUDA_VISIBLE_DEVICES"] == "-1"
    outcome[0], calls[:] = "fail", []
    assert pre.preflight("f0", 0) == (False, "quick probe: AssertionError")
    assert len(calls) == 1
    outcome[0], calls[:] = "ok", []
    assert pre.preflight("f0", 0) == (True, "")
    assert [a[1:3] for a, _k in calls] == [["-c", pre.quick("f0", 0)],
                                           ["-m", "pytest"]]
    # the suites: all of them, this file first, CPU only, in this tree
    args, k = calls[1]
    assert args[-len(pre.SUITES):] == pre.SUITES
    assert args[-len(pre.SUITES)] == "tests/test_supervise_testb.py"
    assert k["env"]["CUDA_VISIBLE_DEVICES"] == "-1" and k["cwd"] == pre.REPO
    # a student whose master is not complete: a reason, nothing run
    calls[:] = []
    assert pre.preflight("f1", 0) == (
        False, "quick probe: F0 has no completion marker -- its final row "
               "is not settled")
    assert calls == []


# --------------------------------------------------------------------------
# the chain: f0 waits for nothing; f{k} for F{k-1}
# --------------------------------------------------------------------------
def test_f0_has_no_predecessor_and_no_prep(sup, monkeypatch, capsys):
    assert sup.master_of("f0") is None and not sup.is_student("f0")
    assert sup.predecessor("f0", 0) is None
    assert sup.wait_external("f0", 0) == "ready"     # at once: no sleep
    monkeypatch.setattr(sup, "wait_external", lambda *a: (_ for _ in (
        )).throw(AssertionError("f0 waits for no predecessor")))
    monkeypatch.setattr(sup, "run_prep", lambda *a, **k: (_ for _ in (
        )).throw(AssertionError("f0 has no prep")))
    events = []
    monkeypatch.setattr(sup, "wait_no_trainer",
                        lambda *a: events.append("guard"))

    def preflight(v, s):
        events.append(("preflight", v, s))
        raise Stop

    monkeypatch.setattr(sup, "preflight", preflight)
    with pytest.raises(Stop):
        sup.main(["f0", "0"])
    # the guard before the preflight all the same
    assert events == ["guard", ("preflight", "f0", 0)]
    out = capsys.readouterr().out
    assert "f0 s0 starts the lineage: no predecessor to wait for" in out
    assert "STOP f0 s0: --stop-env-steps 20000 = 20000 - 0" in out


def test_f0_is_held_back_by_any_trainer(sup, monkeypatch, capsys):
    """Generation 2 may still train when Test B is launched: f0's guard
    (before its preflight) waits it out -- its suites never run beside
    it."""
    alive = [7]
    monkeypatch.setattr(sup, "live_trainers", lambda: list(alive))
    sleep = _bounded(lambda k: alive.clear() if k == 2 else None)
    monkeypatch.setattr(sup.time, "sleep", sleep)
    seen = []

    def preflight(v, s):
        seen.append((v, list(alive)))
        raise Stop

    monkeypatch.setattr(sup, "preflight", preflight)
    with pytest.raises(Stop):
        sup.main(["f0", "0"])
    assert seen == [("f0", [])] and sleep.calls == [60, 60]
    out = capsys.readouterr().out
    assert "trainer is already running (pids [7])" in out
    assert out.index("STOP f0 s0") < out.index("already running")


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_each_student_waits_for_the_previous_generation(sup, k):
    m = V[k - 1]
    _rel, base, _j, ck, _d = sup.paths(m, 0)
    name, marker, probes, suplog, script, pvar, flag = sup.predecessor(V[k],
                                                                       0)
    assert (name, marker, suplog, script, pvar, flag) == (
        f"F{k - 1}", ck, os.path.join(base, "supervisor.log"),
        "supervise_testb.py", m, os.path.join(base, "ABANDONED"))
    run = sup.run_name(m, 0)
    want = [f"probe_exhaustive_eval-{run}-acting.jsonl",
            f"probe_exhaustive_eval-{run}.jsonl"]
    if k > 1:                     # a student master: its offline curve too
        want += [f"probe_exhaustive_eval-{run}-offline-acting.jsonl",
                 f"probe_exhaustive_eval-{run}-offline.jsonl"]
    assert [os.path.basename(p) for p, _r in probes] == want
    assert all(os.path.dirname(p) == os.path.join(sup.REPO, "benchmarks",
                                                  "probes")
               for p, _r in probes)


def test_the_wait_rows_are_the_predecessors_in_run_rows(sup):
    # F0 (from scratch): every in-run row of its jsonl, its final 27
    # included; no row 0, no offline curve
    _jsonl(sup, "f0", {5: 3000, 10: 7000, 15: 11000, 20: 14500, 25: 18500,
                       27: 20400})
    assert [r for _p, r in sup.predecessor("f1", 0)[2]] == [
        (5, 10, 15, 20, 25, 27)] * 2
    # F1 (a student): row 0 first, then its rows (a final row on a
    # multiple of 5); its offline curve's rows 1-4
    _jsonl(sup, "f1", {i: 650 * i for i in (5, 10, 15, 20, 25, 30)})
    want = [(0, 5, 10, 15, 20, 25, 30)] * 2 + [(1, 2, 3, 4)] * 2
    assert [r for _p, r in sup.predecessor("f2", 0)[2]] == want
    # another arm's or seed's rows, a jsonl row 0, a torn line: not F1's
    _jsonl(sup, "f1", {35: 1}, arm=F[2])
    _jsonl(sup, "f1", {40: 1}, seed=1)
    with open(sup.paths("f1", 0)[2], "a") as f:
        f.write(json.dumps({"arm": F[1], "seed": 0, "iteration": 0,
                            "env_steps": 0}) + NL)
        f.write('{"arm": "' + F[1] + '", "seed": 0, "iter' + NL)
    assert [r for _p, r in sup.predecessor("f2", 0)[2]] == want
    assert sup.run_rows("f1", 0) == {i: 650 * i for i in (5, 10, 15, 20, 25,
                                                          30)}


def test_f1_waits_for_f0s_marker_and_every_row_of_both_evals(sup):
    marker = pathlib.Path(sup.paths("f0", 0)[3])
    marker.parent.mkdir(parents=True)
    missing = sup.pred_missing("f1", 0)
    assert any("no checkpoint" in m for m in missing)
    assert "in-run rows (its ablation jsonl holds none)" in missing
    marker.write_bytes(b"x" * 1_000_000)
    assert any("full size" in m for m in sup.pred_missing("f1", 0))
    marker.write_bytes(b"done")
    _jsonl(sup, "f0", {i: 700 * i for i in (5, 10, 15, 20, 25, 28)})
    acting, raw = _probe(sup, "f0", "-acting"), _probe(sup, "f0", "")
    _rows(acting, (5, 10, 15, 20, 25))
    _rows(raw, (5, 10, 15, 20, 25), extra=[
        {"iteration": 28, "env": ENV_ID, "max_noop": 2},    # a smoke row
        {"iteration": 28, "env": "Pong-v4"},
        {"iteration": 0, "env": ENV_ID, "max_noop": 2}])
    # the final iteration 28 is waited for in both
    assert sup.pred_missing("f1", 0) == [f"{acting.name} rows [28]",
                                         f"{raw.name} rows [28]"]
    _rows(acting, [28])
    _rows(raw, [28])
    assert sup.pred_missing("f1", 0) == []
    # F0 has no row 0 and no offline curve: neither is waited for
    assert not any(set(rows) & {0, 1, 2, 3, 4}
                   for _p, rows in sup.predecessor("f1", 0)[2])


def test_f2_waits_for_f1s_row_0_and_its_offline_curve(sup):
    _finish(sup, "f1", final=30, evals=False)
    run = sup.run_name("f1", 0)
    assert sup.pred_missing("f2", 0) == [
        f"probe_exhaustive_eval-{run}-acting.jsonl rows "
        f"[0, 5, 10, 15, 20, 25, 30]",
        f"probe_exhaustive_eval-{run}.jsonl rows [0, 5, 10, 15, 20, 25, 30]",
        f"probe_exhaustive_eval-{run}-offline-acting.jsonl rows "
        f"[1, 2, 3, 4]",
        f"probe_exhaustive_eval-{run}-offline.jsonl rows [1, 2, 3, 4]"]
    for s in ("-acting", ""):
        _rows(_probe(sup, "f1", s), _in_run(30))
    assert len(sup.pred_missing("f2", 0)) == 4          # row 0, the curve
    for s in ("-acting", ""):
        _rows(_probe(sup, "f1", s), [0])
    assert [m.split(" rows ")[1] for m in sup.pred_missing("f2", 0)] == [
        "[1, 2, 3, 4]"] * 2
    for s in ("-acting", ""):
        _rows(_probe(sup, "f1", s, offline=True), (1, 2, 3, 4))
    assert sup.pred_missing("f2", 0) == []


def test_wait_external_blocks_and_alerts(sup, monkeypatch, capsys):
    rows = _finish(sup, "f0", final=27, evals=False, files=False)
    marker = pathlib.Path(sup.paths("f0", 0)[3])
    os.utime(marker, (1, 1))                           # done long ago
    asked = []
    monkeypatch.setattr(sup, "live_scripts",
                        lambda *a: asked.append(a) or [])

    def rows_land(k):
        if k == 2:
            _rows(_probe(sup, "f0", "-acting"), rows)
            _rows(_probe(sup, "f0", ""), rows)

    sleep = _bounded(rows_land)
    monkeypatch.setattr(sup.time, "sleep", sleep)
    assert sup.wait_external("f1", 0) == "ready"
    out = capsys.readouterr().out
    assert "WAITING for F0 before f1 s0" in out
    assert "ALERT: F0 completed" in out and "evals still lack rows" in out
    # the line the hand-off echoes
    assert "F0 complete and evaluated; f1 s0 may start" in out
    # the master's snapshot and own-play log are checked while waiting
    assert "master files missing: " + sup.snapshot_path("f0", 0, 27) in out
    assert f"{_log_dir(sup, 'f0')} iterations [1, 2, 3, 4, 5, 6, 7, 8] ..." \
        in out
    assert "keep F0's run folder" in out
    assert sleep.calls == [60, 60]
    # unfinished and undriven
    marker.write_bytes(b"x" * 1_000_000)
    monkeypatch.setattr(sup.time, "sleep", _stop)
    with pytest.raises(Stop):
        sup.wait_external("f1", 0)
    out = capsys.readouterr().out
    assert "nothing drives it: no live supervise_testb.py f0" in out
    assert asked[-1] == ("supervise_testb.py", "f0")
    assert "master files missing" not in out        # no marker yet
    # driven, but its log silent for hours
    monkeypatch.setattr(sup, "live_scripts", lambda *a: [5])
    suplog = pathlib.Path(sup.paths("f0", 0)[1]) / "supervisor.log"
    suplog.write_text("x")
    os.utime(suplog, (1, 1))
    with pytest.raises(Stop):
        sup.wait_external("f1", 0)
    assert "is alive but its log is silent" in capsys.readouterr().out
    # its supervisor gave up
    suplog.write_text("LAUNCHED f0 s0 attempt 3" + NL + "NO PROGRESS: 3 x"
                      + NL)
    with pytest.raises(Stop):
        sup.wait_external("f1", 0)
    assert "F0's supervisor gave up -- f1 keeps waiting" in \
        capsys.readouterr().out


def test_wait_external_stops_when_this_run_is_abandoned(sup, monkeypatch):
    base = pathlib.Path(sup.paths("f1", 0)[1])
    base.mkdir(parents=True)
    monkeypatch.setattr(sup.time, "sleep", _bounded(
        lambda k: (base / "ABANDONED").write_text("")))
    monkeypatch.setattr(sup, "live_scripts", lambda *a: [1])
    assert sup.wait_external("f1", 0) == "abandoned"


@pytest.mark.parametrize("complete", [False, True])
def test_an_abandoned_predecessor_is_master_abandoned(sup, monkeypatch,
                                                      capsys, complete):
    """No fallback: f1 stops (PREP FAILED, the hand-off's failed state)
    whether F0 was abandoned before or after completing."""
    if complete:
        _finish(sup, "f0")
    flag = pathlib.Path(sup.paths("f0", 0)[1]) / "ABANDONED"
    flag.parent.mkdir(parents=True, exist_ok=True)
    sleep = _bounded(lambda k: flag.write_text(""))
    monkeypatch.setattr(sup.time, "sleep", sleep)
    monkeypatch.setattr(sup, "live_scripts", lambda *a: [])
    if complete:
        flag.write_text("")
    assert sup.wait_external("f1", 0) == "failed"
    out = capsys.readouterr().out
    line = [l for l in out.splitlines() if "MASTER ABANDONED" in l]
    # the message, never the flag's path
    assert line == ["PREP FAILED: " + sup.master_abandoned_msg("f1")]
    assert len(sleep.calls) == (0 if complete else 1)
    # main stops there: no stop, no preflight, no prep, no launch
    monkeypatch.setattr(sup, "settle_stop", _stop)
    monkeypatch.setattr(sup, "preflight", _stop)
    monkeypatch.setattr(sup, "run_prep", _stop)
    monkeypatch.setattr(sup.subprocess, "Popen", _stop)
    assert sup.main(["f1", "0"]) == 1
    assert "PREP FAILED: MASTER ABANDONED" in capsys.readouterr().out


def test_master_abandoned_msg_names_the_generations(real_sup):
    sup = real_sup
    for k in (1, 2, 3, 4):
        assert sup.master_abandoned_msg(V[k]) == (
            f"MASTER ABANDONED: F{k - 1}'s run folder holds an ABANDONED "
            f"file -- no F{k} on it and no fallback master (Kai's call)")
    # every stop prints the message; the existence checks read the flag
    src = inspect.getsource(sup)
    assert "raise SystemExit(master_abandoned_msg(variant))" in src
    assert 'print(f"PREP FAILED: {master_abandoned_msg(variant)}"' in src


def test_an_undriven_predecessor_is_named_and_its_stops_are_kais_call(
        sup, monkeypatch, capsys):
    """F1's own supervisor has a master gate and a stop: stopped on either
    it is Kai's call, not "rerun its launcher" (Test 2's audit r1 B-NIT2)."""
    monkeypatch.setattr(sup.time, "sleep", _stop)
    with pytest.raises(Stop):
        sup.wait_external("f2", 0)
    assert "no live supervise_testb.py f1 -- rerun its launcher" in \
        capsys.readouterr().out
    suplog = pathlib.Path(sup.paths("f1", 0)[1]) / "supervisor.log"
    suplog.parent.mkdir(parents=True)
    for stop in ("PREP FAILED: MASTER NEAR RANDOM: F0's deployed mean ...",
                 "PREP FAILED: MASTER ABANDONED: F0's run folder ...",
                 "PREP FAILED: STOP NOT POSITIVE: F1's --stop-env-steps ...",
                 "PREP FAILED: PREP CHECK: the value support (5000 ..."):
        suplog.write_text("F0 complete and evaluated; f1 s0 may start" + NL
                          + stop + NL)
        with pytest.raises(Stop):
            sup.wait_external("f2", 0)
        out = capsys.readouterr().out
        assert "F1's supervisor stopped on" in out and stop[:30] in out
        assert "Kai's call" in out and "-- rerun its launcher" not in out
        assert "F1's supervisor gave up -- f2 keeps waiting" in out
    # a later preflight (a relaunch after the fix) clears it
    suplog.write_text("PREP FAILED: MASTER NEAR RANDOM: ..." + NL
                      + "preflight ok (...)" + NL)
    with pytest.raises(Stop):
        sup.wait_external("f2", 0)
    assert "no live supervise_testb.py f1" in capsys.readouterr().out
    # another give-up keeps the rerun advice
    suplog.write_text("preflight ok (...)" + NL
                      + "PREP FAILED after 3 attempts -- see ..." + NL)
    with pytest.raises(Stop):
        sup.wait_external("f2", 0)
    assert "no live supervise_testb.py f1" in capsys.readouterr().out
    # F1's flag: wait_external stops on it first (MASTER ABANDONED);
    # pred_unattended names it too
    (suplog.parent / "ABANDONED").write_text("")
    assert sup.pred_unattended("f2", 0).startswith("F1 is ABANDONED")


# --------------------------------------------------------------------------
# THE STOP: 20,000 x (k + 1) less the earlier generations' final env steps
# --------------------------------------------------------------------------
def test_f0s_stop_is_the_budget(sup):
    assert sup.stop_env_steps("f0", 0) == sup.target("f0") == 20000
    cmd = sup.command("f0", 0)                   # nothing on disk needed
    assert cmd[cmd.index("--stop-env-steps") + 1] == "20000"


def test_a_stop_unknown_is_a_plain_give_up(sup, monkeypatch, capsys):
    """Audit r1 B-L4: STOP UNKNOWN (a predecessor's final row not settled
    yet) clears on a rerun -- not Kai's call."""
    monkeypatch.setattr(sup.time, "sleep", _stop)
    suplog = pathlib.Path(sup.paths("f1", 0)[1]) / "supervisor.log"
    suplog.parent.mkdir(parents=True)
    suplog.write_text("F0 complete and evaluated; f1 s0 may start" + NL
                      + "PREP FAILED: STOP UNKNOWN: F0 has no completion "
                      "marker" + NL)
    with pytest.raises(Stop):
        sup.wait_external("f2", 0)
    out = capsys.readouterr().out
    assert "no live supervise_testb.py f1 -- rerun its launcher" in out
    assert "Kai's call" not in out


def test_an_overshoot_never_accumulates(sup):
    """Each generation passes its stop by up to an iteration's env steps;
    the next one's stop absorbs it, so after F{k} the lineage stands at
    20,000 x (k + 1) plus F{k}'s OWN overshoot only (a fixed 20,000 per
    generation would carry 1,650 extra env steps into F4)."""
    overshoot = [400, 550, 450, 250]
    lineage = 0
    for k in range(4):
        stop = sup.stop_env_steps(V[k], 0)
        assert stop == 20000 * (k + 1) - lineage
        own = stop + overshoot[k]
        _finish(sup, V[k], final=26 + k, env=own, evals=False, files=False)
        lineage += own
        assert lineage == 20000 * (k + 1) + overshoot[k]
    assert [sup.stop_env_steps(v, 0) for v in V] == [20000, 19600, 19450,
                                                     19550, 19750]
    assert 100000 - lineage == sup.stop_env_steps("f4", 0)
    # an undershoot (a generation that hit the --iterations cap short of
    # its stop) is made up the same way
    for v in V[1:4]:
        _rel, base, jsonl, ck, _d = sup.paths(v, 0)
        os.remove(jsonl)
        os.remove(ck)
    pathlib.Path(sup.paths("f0", 0)[2]).write_text("")
    _jsonl(sup, "f0", {60: 18000})
    assert sup.stop_env_steps("f1", 0) == 22000


def test_the_stop_reads_the_final_row(sup):
    _jsonl(sup, "f0", {27: 20400, 5: 3500, 10: 7400, 25: 18600})
    pathlib.Path(sup.paths("f0", 0)[3]).write_bytes(b"done")
    # the last ITERATION is the final row, whatever the file order
    assert sup.final_row("f0", 0) == (27, 20400)
    # another arm's or seed's rows past it, a torn line: not F0's
    _jsonl(sup, "f0", {30: 99999}, arm=F[1])
    _jsonl(sup, "f0", {31: 99999}, seed=1)
    with open(sup.paths("f0", 0)[2], "a") as f:
        f.write('{"arm": "' + F[0] + '", "seed": 0, "iteration": 32, "en'
                + NL)
    assert sup.final_row("f0", 0) == (27, 20400)
    assert sup.stop_env_steps("f1", 0) == 19600
    assert sup.master_final("f1", 0) == (27, sup.snapshot_path("f0", 0, 27))


@pytest.mark.parametrize("env", [None, True, 0, -5, 20400.0, "20400"])
def test_a_final_row_without_env_steps_is_no_stop(sup, env):
    _jsonl(sup, "f0", {5: 3500, 27: env})
    pathlib.Path(sup.paths("f0", 0)[3]).write_bytes(b"done")
    with pytest.raises(SystemExit) as e:
        sup.stop_env_steps("f1", 0)
    assert str(e.value) == (f"STOP UNKNOWN: F0's final row 27 records no "
                            f"env steps ({env!r})")


def test_the_stop_needs_complete_predecessors(sup):
    with pytest.raises(SystemExit) as e:
        sup.stop_env_steps("f1", 0)
    assert str(e.value) == ("STOP UNKNOWN: F0 has no completion marker -- "
                            "its final row is not settled")
    ck = pathlib.Path(sup.paths("f0", 0)[3])
    ck.parent.mkdir(parents=True)
    ck.write_bytes(b"x" * 1_000_000)               # still training
    with pytest.raises(SystemExit, match="F0 has no completion marker"):
        sup.stop_env_steps("f1", 0)
    ck.write_bytes(b"done")
    with pytest.raises(SystemExit, match="STOP UNKNOWN: F0's ablation jsonl "
                                         "holds no in-run row"):
        sup.stop_env_steps("f1", 0)
    _jsonl(sup, "f0", {27: 20400})
    assert sup.stop_env_steps("f1", 0) == 19600
    # every earlier generation counts: F1 unfinished stops f2's
    with pytest.raises(SystemExit, match="STOP UNKNOWN: F1 has no "
                                         "completion marker"):
        sup.stop_env_steps("f2", 0)
    with pytest.raises(SystemExit, match="STOP UNKNOWN: F1"):
        sup.command("f4", 0)


@pytest.mark.parametrize("env, stop", [(40000, 0), (40001, -1)])
def test_a_spent_budget_stops_the_chain(sup, monkeypatch, capsys, env, stop):
    _finish(sup, "f0", final=55, env=env)
    with pytest.raises(SystemExit) as e:
        sup.stop_env_steps("f1", 0)
    assert str(e.value) == (
        f"STOP NOT POSITIVE: F1's --stop-env-steps would be {stop} = 40000 "
        f"- {env} (the final env steps of F0: [{env}]) -- the lineage spent "
        f"its budget (Kai's call)")
    # main: F0 is complete and evaluated (no wait), then the stop -- final
    # at once, before any suite or prep: no retry, so the fixture's sleep
    # trap stays unsprung (and a wait that never readies fails on it
    # instead of spinning)
    for name in ("preflight", "run_prep", "wait_no_trainer"):
        monkeypatch.setattr(sup, name, _stop)
    monkeypatch.setattr(sup.subprocess, "Popen", _stop)
    assert sup.main(["f1", "0"]) == 1
    out = capsys.readouterr().out
    assert any(l.startswith("PREP FAILED: STOP NOT POSITIVE: F1's")
               for l in out.splitlines()), out


def test_settle_stop_retries_a_transient_failure(sup, monkeypatch, capsys):
    plan = [SystemExit("STOP UNKNOWN: F0's ablation jsonl holds no in-run "
                       "row"), 19600]

    def stop(v, s):
        r = plan.pop(0)
        if isinstance(r, BaseException):
            raise r
        return r

    monkeypatch.setattr(sup, "stop_env_steps", stop)
    sleeps = []
    monkeypatch.setattr(sup.time, "sleep", sleeps.append)
    assert sup.settle_stop("f1", 0) == 19600 and sleeps == [60]
    out = capsys.readouterr().out
    assert ("STOP f1 s0: --stop-env-steps 19600 = 40000 - 20400 (the "
            "lineage's target after F1, less the earlier generations' final "
            "env steps)") in out
    # still unknown after three reads: PREP FAILED (the hand-off's failed
    # state; the next generation's wait calls it Kai's call)
    monkeypatch.setattr(sup, "stop_env_steps", lambda v, s: (
        _ for _ in ()).throw(SystemExit("STOP UNKNOWN: F0 has no completion "
                                        "marker")))
    sleeps.clear()
    assert sup.settle_stop("f1", 0) is None and sleeps == [60, 60]
    assert "PREP FAILED: STOP UNKNOWN: F0 has no completion marker" in \
        capsys.readouterr().out.splitlines()


def test_the_stop_is_the_same_on_every_launch_and_resume(sup, monkeypatch):
    """The trainer's resume recipe holds the stop: every attempt and every
    later supervisor (a resume) passes the same one, from F0's final
    row."""
    _finish(sup, "f0", final=27, env=20400, evals=False, files=False)
    events, launched = [], []
    _loop(sup, monkeypatch, events)                    # the real stop
    ck = pathlib.Path(sup.paths("f1", 0)[3])

    def popen(args, **k):
        launched.append(list(args))
        ck.write_bytes(b"x" * 1_000_001)
        return Dead()

    monkeypatch.setattr(sup.subprocess, "Popen", popen)
    assert sup.main(["f1", "0"]) == 1                 # 5 quick deaths
    assert sup.main(["f1", "0"]) == 1                 # a resume: the same
    assert len(launched) == 10
    assert {a[a.index("--stop-env-steps") + 1] for a in launched} == {
        "19600"}
    assert [a[-1] == "--resume" for a in launched] == [False] + [True] * 9
    assert events.count("wait") == 1                  # the resume: no wait


# --------------------------------------------------------------------------
# the master: F{k-1}'s final snapshot, after the NEAR RANDOM rule
# --------------------------------------------------------------------------
def test_choose_master_is_the_predecessors_final_snapshot(sup):
    with pytest.raises(SystemExit, match="F0 has no completion marker"):
        sup.choose_master("f1", 0)
    ck = pathlib.Path(sup.paths("f0", 0)[3])
    ck.parent.mkdir(parents=True)
    ck.write_bytes(b"done")
    with pytest.raises(SystemExit, match="holds no in-run row"):
        sup.choose_master("f1", 0)
    _jsonl(sup, "f0", {i: 700 * i for i in (5, 10, 15, 20, 25, 28)})
    acting = _probe(sup, "f0", "-acting")
    with pytest.raises(SystemExit) as e:
        sup.choose_master("f1", 0)
    assert "F0's last two deployed rows [25, 28] are incomplete" in str(
        e.value)
    _rows(acting, (5, 10, 15, 20, 25), 900.0)
    _rows(_probe(sup, "f0", ""), (5, 10, 15, 20, 25, 28), 900.0)   # raw
    with pytest.raises(SystemExit, match=r"rows \[25, 28\] are incomplete"):
        sup.choose_master("f1", 0)
    _rows(acting, [28], 900.0)
    with pytest.raises(SystemExit) as e:
        sup.choose_master("f1", 0)
    snap = sup.snapshot_path("f0", 0, 28)
    assert str(e.value).startswith("F0's master files are missing: " + snap)
    pathlib.Path(snap).parent.mkdir(parents=True)
    pathlib.Path(snap).write_bytes(b"x")
    with pytest.raises(SystemExit, match="iterations"):    # the own-play log
        sup.choose_master("f1", 0)
    _log_dir(sup, "f0").mkdir(parents=True)
    for i in range(1, 29):
        (_log_dir(sup, "f0") / f"iter-{i:03d}.pkl").write_bytes(b"x")
    assert sup.choose_master("f1", 0) == {
        "which": "F0", "snapshot": snap, "iterations": 28,
        "master_mean_last2": pytest.approx(900.0)}


@pytest.mark.parametrize("last2, raw, ok", [
    ((502.08, 502.08), 502.08, True),     # exactly g = 0.2: passes
    # a decimal 502.08 mean whose float sum lands a hair below
    ((502.08, 502.08 - 1e-12), 0.0, True),
    ((502.08, 502.079), 9999.0, False),   # 5e-4 below in the mean: fails
    ((502.07, 502.07), 9999.0, False),    # the DEPLOYED mean decides
    ((1000.0, 4.16), 0.0, True),          # one poor row alone is no stop
    ((1000.0, 4.0), 9999.0, False),
    ((9999.0, 9999.0), 100.0, True),
    ((307.3, 307.3), 307.3, False)])
def test_the_near_random_rule_on_the_last_two_rows(sup, last2, raw, ok):
    rows = _finish(sup, "f0", final=27, value=9999.0)   # earlier: unread
    acting = _probe(sup, "f0", "-acting")
    _rows(acting, [25], last2[0])                 # the latest record wins
    _rows(acting, [27], last2[1])
    _rows(_probe(sup, "f0", ""), rows, raw)
    mean = sum(last2) / 2
    if ok:
        assert sup.choose_master("f1", 0)["master_mean_last2"] == \
            pytest.approx(mean)
    else:
        with pytest.raises(SystemExit) as e:
            sup.choose_master("f1", 0)
        assert str(e.value).startswith("MASTER NEAR RANDOM: F0's deployed "
                                       "mean over its last two rows "
                                       "[25, 27]")
        assert f"{mean:.2f}" in str(e.value) and "502.08" in str(e.value)
        assert "Kai's call" in str(e.value)


def test_the_last_two_rows_are_the_latest_in_run_iterations(sup):
    # a final iteration on a multiple of 5: rows 25 and 30
    _finish(sup, "f0", final=30, value=9999.0)
    acting = _probe(sup, "f0", "-acting")
    _rows(acting, [25], 800.0)
    _rows(acting, [30], 1000.0)
    _rows(acting, [31], 1.0)       # no in-run row 31 (a stale snapshot's)
    pick = sup.choose_master("f1", 0)
    assert pick["master_mean_last2"] == pytest.approx(900.0)
    assert pick["iterations"] == 30
    # a student master: its row 0 and offline curve are no in-run rows
    _finish(sup, "f1", final=27, value=9999.0)
    acting = _probe(sup, "f1", "-acting")
    _rows(acting, [25], 600.0)
    _rows(acting, [27], 700.0)
    _rows(acting, [0], 1.0)
    _rows(_probe(sup, "f1", "-acting", offline=True), (1, 2, 3, 4), 1.0)
    assert sup.choose_master("f2", 0) == {
        "which": "F1", "snapshot": sup.snapshot_path("f1", 0, 27),
        "iterations": 27, "master_mean_last2": pytest.approx(650.0)}


def test_an_abandoned_predecessor_is_never_the_master(sup):
    _finish(sup, "f0", value=2000.0)
    (pathlib.Path(sup.paths("f0", 0)[1]) / "ABANDONED").write_text("")
    with pytest.raises(SystemExit) as e:
        sup.choose_master("f1", 0)
    assert str(e.value) == sup.master_abandoned_msg("f1")


def test_master_files_missing(sup):
    assert sup.master_files_missing("f1", 0) == []          # F0 unfinished
    _finish(sup, "f0", final=12, evals=False, files=False)
    log = _log_dir(sup, "f0")
    assert sup.master_files_missing("f1", 0) == [
        sup.snapshot_path("f0", 0, 12),
        f"{log} iterations [1, 2, 3, 4, 5, 6, 7, 8] ..."]
    snap = pathlib.Path(sup.snapshot_path("f0", 0, 12))
    snap.parent.mkdir(parents=True)
    snap.write_bytes(b"x")
    log.mkdir(parents=True)
    for i in range(1, 12):
        (log / f"iter-{i:03d}.pkl").write_bytes(b"x")
    # another row's snapshot or a later iteration's log is no stand-in
    (log / "iter-013.pkl").write_bytes(b"x")
    assert sup.master_files_missing("f1", 0) == [f"{log} iterations [12]"]
    (log / "iter-012.pkl").write_bytes(b"x")
    assert sup.master_files_missing("f1", 0) == []


# --------------------------------------------------------------------------
# the prep
# --------------------------------------------------------------------------
PICK = {"which": "F0", "snapshot": "m.pth", "iterations": 27,
        "master_mean_last2": 700.0}
GONE = "<gone>"         # a vocab.json key _write_prep leaves out


@pytest.fixture
def prep_env(sup, monkeypatch):
    """f1's prep, its gate stubbed (PICK), beside a complete F0 (final
    iteration 27: what set_intact checks the prep against); no guard, an
    audited tree, no sleeps."""
    _finish(sup, "f0", final=27, evals=False, files=False)
    monkeypatch.setattr(sup, "wait_no_trainer", lambda *a: None)
    monkeypatch.setattr(sup, "manifest_check", lambda: (True, ""))
    monkeypatch.setattr(sup.time, "sleep", lambda s: None)
    calls = []
    monkeypatch.setattr(sup, "choose_master",
                        lambda v, s: calls.append(("choose", v))
                        or dict(PICK))
    return calls


def _write_prep(sup, v="f1", good=True, k=27, **doc):
    """`v`'s prep as distill.py --from-log writes it from its master's
    final snapshot at iteration `k`; `doc` changes (GONE drops a key)."""
    d = pathlib.Path(sup.prep_dir(v))
    d.mkdir(parents=True, exist_ok=True)
    (d / "set.pkl").write_bytes(b"set")
    sha = hashlib.sha256(b"set" if good else b"other").hexdigest()
    m = sup.master_of(v)
    # the master's final snapshot (as _finish writes it) and its sha256
    snap = pathlib.Path(sup.snapshot_path(m, 0, k))
    if not snap.exists():
        snap.parent.mkdir(parents=True, exist_ok=True)
        snap.write_bytes(b"x")
    full = {"set_sha256": sha, "master_arm": sup.VARIANTS[m],
            "master_sha256": hashlib.sha256(snap.read_bytes()).hexdigest(),
            "master_iterations": k,
            "master_offline_steps": sup.master_offline(v),
            "master_snapshot": sup.snapshot_path(m, 0, k),
            "source": "own-play log", "log_dir": sup.own_play_rel(m, 0),
            "log_iterations": k, "student_ceiling": 7.5,
            "master_discount": sup.DISCOUNTS[m],
            "student_discount": distill.swap_discount(
                sup.DISCOUNTS[m], sup.DISCOUNT_RULE),
            "discount_rule": sup.DISCOUNT_RULE,
            "vocab": [[0, 4], [1, 4]], **doc}
    (d / "vocab.json").write_text(json.dumps(
        {key: val for key, val in full.items() if val != GONE}))


class FakeProc:
    """A finished Popen: runs `effect` (writing the prep's files) at start."""

    def __init__(self, rc, effect=None):
        self.returncode, self.pid = rc, 4321
        if effect:
            effect()

    def poll(self):
        return self.returncode


def _popen(monkeypatch, sup, plan, seen=None):
    """subprocess.Popen -> FakeProc per call, from `plan`: a list of (rc,
    effect); `seen` collects (args, kwargs)."""
    def popen(args, **k):
        if seen is not None:
            seen.append((args, k))
        rc, effect = plan.pop(0)
        return FakeProc(rc, effect)

    monkeypatch.setattr(sup.subprocess, "Popen", popen)
    monkeypatch.setattr(sup.subprocess, "run", _stop)


def test_the_prep_runs_once_and_logs_its_master(sup, prep_env, monkeypatch,
                                                capsys):
    calls = prep_env
    runs = []
    _popen(monkeypatch, sup, [(1, None), (0, lambda: _write_prep(sup))], runs)
    env = {"X": "1"}
    assert sup.run_prep("f1", 0, env) is True
    out = capsys.readouterr().out
    assert ("MASTER F0: m.pth (iteration 27; deployed mean of its last two "
            "rows 700.00, g 0.403)") in out
    assert "PREP attempt 1 failed" in out and "PREP done (attempt 2)" in out
    assert calls == [("choose", "f1")] and len(runs) == 2
    args, k = runs[0]
    # the master's own-play log, no recording; its final iteration K
    assert args == [sup.PY, "experiments/distill.py", "--master", "m.pth",
                    "--master-iterations", "27", "--out", sup.prep_rel("f1"),
                    "--seed", "0", "--from-log",
                    f"benchmarks/results/mspacman-fast-f0-s0-{DATE}/"
                    f"own-play-mspacmannoframeskip-skip4-{F[0]}-seed0"]
    assert args[-1] == sup.own_play_rel("f0", 0)
    assert k["cwd"] == sup.REPO and k["env"] is env
    master = json.loads(pathlib.Path(sup.prep_dir("f1"), "master.json")
                        .read_text())
    assert master == PICK
    assert not list(pathlib.Path(sup.prep_dir("f1")).glob("*.tmp"))
    assert sup.run_prep("f1", 0, {}) is True         # done: nothing runs
    assert len(runs) == 2 and calls == [("choose", "f1")]


def test_the_real_gate_picks_the_masters_final_snapshot_into_master_json(
        sup, monkeypatch, capsys):
    """The real choose_master inside f2's run_prep: master.json carries
    F1's pick at its final iteration 31, and distill.py gets F1's snapshot,
    31 and F1's own-play log."""
    _finish(sup, "f1", final=31, env=19950, value=1231.0)
    monkeypatch.setattr(sup, "wait_no_trainer", lambda *a: None)
    monkeypatch.setattr(sup, "manifest_check", lambda: (True, ""))
    seen = []
    _popen(monkeypatch, sup, [(0, lambda: _write_prep(sup, "f2", k=31))],
           seen)
    assert sup.run_prep("f2", 0, {}) is True
    snap = sup.snapshot_path("f1", 0, 31)
    out = capsys.readouterr().out
    assert (f"MASTER F1: {snap} (iteration 31; deployed mean of its last two "
            f"rows 1231.00, g 0.948)") in out
    pick = json.loads(pathlib.Path(sup.prep_dir("f2"), "master.json")
                      .read_text())
    assert pick == {"which": "F1", "snapshot": snap, "iterations": 31,
                    "master_mean_last2": 1231.0}
    args = seen[0][0]
    assert args[args.index("--master") + 1] == snap
    assert args[args.index("--master-iterations") + 1] == "31"
    assert args[args.index("--from-log") + 1] == sup.own_play_rel("f1", 0)
    assert args[args.index("--out") + 1] == sup.prep_rel("f2")


def test_read_pick_needs_the_master_key(sup, tmp_path):
    """master_mean_last2 is the pick's key: Generation 2's
    master_mean_60_90 pick is no Test B pick."""
    path = str(tmp_path / "master.json")
    sup.write_pick(path, PICK)
    assert sup.read_pick(path) == PICK
    old = {**{k: v for k, v in PICK.items() if k != "master_mean_last2"},
           "which": "DS2", "master_mean_60_90": 700.0}
    sup.write_pick(path, old)
    assert sup.read_pick(path) is None
    for k in PICK:
        sup.write_pick(path, {j: v for j, v in PICK.items() if j != k})
        assert sup.read_pick(path) is None, k
    sup.write_pick(path, [PICK])
    assert sup.read_pick(path) is None


def test_a_finished_prep_with_another_master_json_is_refused(
        sup, prep_env, monkeypatch, capsys):
    _write_prep(sup)
    sup.write_pick(os.path.join(sup.prep_dir("f1"), "master.json"), {
        "which": "DS2", "snapshot": "m.pth", "iterations": 90,
        "master_mean_60_90": 700.0})
    monkeypatch.setattr(sup.subprocess, "Popen", _stop)
    assert sup.run_prep("f1", 0, {}) is False
    assert "PREP FAILED: a finished prep without a readable master.json" in \
        capsys.readouterr().out


def test_a_restarted_prep_runs_the_master_gate_again(sup, prep_env,
                                                     monkeypatch):
    """The master is fixed: a prep (re)started by a new supervisor runs the
    gate again and rewrites master.json; a torn one is no obstacle."""
    calls = prep_env
    d = pathlib.Path(sup.prep_dir("f1"))
    d.mkdir(parents=True)
    (d / "master.json").write_text('{"which": "F0", "snap')    # torn
    seen = []
    _popen(monkeypatch, sup, [(0, lambda: _write_prep(sup))], seen)
    assert sup.run_prep("f1", 0, {}) is True and calls == [("choose", "f1")]
    assert sup.read_pick(str(d / "master.json")) == PICK
    assert seen[0][0][3] == "m.pth" and seen[0][0][5] == "27"


@pytest.mark.parametrize("why, start", [
    ("MASTER NEAR RANDOM: F0's deployed mean ...", "PREP FAILED: MASTER "
                                                   "NEAR RANDOM"),
    ("MASTER ABANDONED: x", "PREP FAILED: MASTER ABANDONED"),
    ("F0's last two deployed rows [25, 27] are incomplete",
     "PREP FAILED: F0's"),
    ("F0's master files are missing: x", "PREP FAILED: F0's master files")])
def test_a_master_gate_failure_is_prep_failed(sup, prep_env, monkeypatch,
                                              capsys, why, start):
    monkeypatch.setattr(sup, "choose_master", lambda v, s: (
        _ for _ in ()).throw(SystemExit(why)))
    monkeypatch.setattr(sup.subprocess, "Popen", _stop)
    assert sup.run_prep("f1", 0, {}) is False
    out = capsys.readouterr().out
    assert any(l.startswith(start) for l in out.splitlines()), out
    assert not pathlib.Path(sup.prep_dir("f1"), "master.json").exists()


def test_near_random_stops_the_real_gate(sup, monkeypatch, capsys):
    """The real choose_master inside run_prep: the line the hand-off reads
    as a failed state, with the mean; nothing starts."""
    _finish(sup, "f0", value=480.5)
    monkeypatch.setattr(sup, "wait_no_trainer", _stop)
    monkeypatch.setattr(sup.subprocess, "Popen", _stop)
    assert sup.run_prep("f1", 0, {}) is False
    out = capsys.readouterr().out
    line = [l for l in out.splitlines() if l.startswith("PREP FAILED")]
    assert line and line[0].startswith("PREP FAILED: MASTER NEAR RANDOM: "
                                       "F0's")
    assert "480.50" in line[0] and "Kai's call" in line[0]
    assert "before F1" in line[0]
    (pathlib.Path(sup.paths("f0", 0)[1]) / "ABANDONED").write_text("")
    assert sup.run_prep("f1", 0, {}) is False
    assert ("PREP FAILED: " + sup.master_abandoned_msg("f1")) in \
        capsys.readouterr().out.splitlines()


def test_the_prep_gives_up_after_its_attempts(sup, prep_env, monkeypatch,
                                              capsys):
    _popen(monkeypatch, sup, [(1, None)] * 3)
    assert sup.run_prep("f1", 0, {}) is False
    out = capsys.readouterr().out
    assert out.count("failed (rc=1") == sup.PREP_ATTEMPTS == 3
    assert "PREP FAILED after 3 attempts" in out


def test_a_failed_attempt_names_its_reason(sup, prep_env, monkeypatch,
                                           capsys):
    # audit r1 B-L1: the reason in the supervisor's log, not only prep.log
    def exits():
        d = pathlib.Path(sup.prep_dir("f1"))
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "prep.log", "a", encoding="utf-8") as f:
            f.write("own-play log: x\nRuntimeError: CUDA error: out of "
                    "memory\n\n")
    _popen(monkeypatch, sup, [(1, exits)] * 3)
    assert sup.run_prep("f1", 0, {}) is False
    out = capsys.readouterr().out
    assert out.count("its last line: RuntimeError: CUDA error: out of "
                     "memory") == 3
    assert "PREP FAILED after 3 attempts" in out
    assert "PREP CHECK" not in out
    assert sup.prep_log_tail(os.path.join(sup.prep_dir("f1"), "nope")) == ""


@pytest.mark.parametrize("why", [
    "SystemExit: the value support (5000 units) does not hold 2.0 x the "
    "largest discounted return 2600.0 under the student's ceiling 40.1 and "
    "discount 0.997",
    "SystemExit: the halved-clip ceiling 401.0 lies past the reward "
    "supports",
    "SystemExit: own-play log x: iteration 12 is missing",
    "SystemExit: own-play log x: no lives",
    # Test B2's discount rule (audit r3 NIT-1)
    "SystemExit: _discount_rule 'x' is not one of ('halve', 'keep')",
    "SystemExit: a kept discount 0.998 past the cap 0.997",
    # E3's target rule (audit r3): the structural and the numeric overlap
    # checks, non-finite path weights
    "ValueError: overlapping mined paths: (1,) is a slot-prefix of (1, 2)",
    "ValueError: overlapping mined paths at record 3: path products sum to "
    "1.5 at slot 2 (at most 1)",
    "ValueError: non-finite mined path weights at record 0: [nan, 0.5]"])
def test_a_deterministic_prep_check_stops_at_once(sup, prep_env, monkeypatch,
                                                  capsys, why):
    """Audit r1 B-L1: distill.py's deterministic refusals (A2's value
    check among them) are not retried -- PREP FAILED: PREP CHECK at the
    first attempt (one Popen only), Kai's call."""
    def exits():
        d = pathlib.Path(sup.prep_dir("f1"))
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "prep.log", "a", encoding="utf-8") as f:
            f.write("x\n" + why + "\n")
    _popen(monkeypatch, sup, [(1, exits)])
    assert sup.run_prep("f1", 0, {}) is False
    out = capsys.readouterr().out
    assert f"PREP FAILED: PREP CHECK: {why[:150]}" in out
    assert "PREP FAILED after" not in out
    assert any(l.startswith(sup.KAIS_CALL_STOPS) for l in out.splitlines()
               if l.startswith("PREP FAILED"))


def test_the_preps_own_progress_line_is_retried(sup, prep_env, monkeypatch,
                                                capsys):
    """Audit r2 NIT-4: a prep killed while reading the log leaves its own
    progress line ("own-play log: <dir>, ...") last -- not a check of the
    prep's: retried like any other failure."""
    def exits():
        d = pathlib.Path(sup.prep_dir("f1"))
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "prep.log", "a", encoding="utf-8") as f:
            f.write("x\n10-28 06:00 own-play log: benchmarks/results/r/"
                    "own-play-a-seed0, iterations 1-27 of master "
                    "weights-iter027.pth (a, 54000 steps); temperatures "
                    "{}\n")
    _popen(monkeypatch, sup, [(1, exits)] * 3)
    assert sup.run_prep("f1", 0, {}) is False
    out = capsys.readouterr().out
    assert "PREP CHECK" not in out
    assert "PREP FAILED after 3 attempts" in out


def test_set_intact_needs_the_master_snapshot(sup):
    """Audit r2 NIT-4: set_intact's read of the master snapshot (its
    sha256) fails closed."""
    _finish(sup, "f0", final=27, evals=False, files=False)
    _write_prep(sup)
    assert sup.set_intact("f1", 0) == (True, "")
    os.remove(sup.snapshot_path("f0", 0, 27))
    ok, why = sup.set_intact("f1", 0)
    assert not ok and why.startswith("the master snapshot: "), why


def test_the_prep_retries_15_min_apart(sup, prep_env, monkeypatch):
    sleeps = []
    monkeypatch.setattr(sup.time, "sleep", sleeps.append)
    _popen(monkeypatch, sup, [(1, None)] * 3)
    assert sup.run_prep("f1", 0, {}) is False
    assert sleeps.count(sup.PREP_RETRY_S) == 2 and sup.PREP_RETRY_S == 900


def test_a_silent_prep_is_an_alert(sup, prep_env, monkeypatch, capsys):
    class Slow(FakeProc):
        polls = 0

        def poll(self):
            Slow.polls += 1
            return None if Slow.polls < 3 else self.returncode

    import time as real_time
    clock = [real_time.time()]      # prep.log's real mtime is "now"

    def sleep(s):
        clock[0] += 3600

    monkeypatch.setattr(sup, "time", SimpleNamespace(
        time=lambda: clock[0], sleep=sleep, strftime=real_time.strftime))
    monkeypatch.setattr(sup.subprocess, "Popen", lambda args, **k: Slow(
        0, lambda: _write_prep(sup)))
    assert sup.run_prep("f1", 0, {}) is True
    assert "ALERT: the prep (pid 4321) has been silent" in \
        capsys.readouterr().out


def test_a_damaged_prep_is_moved_aside_and_redone(sup, prep_env, monkeypatch,
                                                  capsys):
    calls = prep_env
    _write_prep(sup, good=False)
    sup.write_pick(os.path.join(sup.prep_dir("f1"), "master.json"), PICK)
    _popen(monkeypatch, sup, [(0, lambda: _write_prep(sup))])
    assert sup.run_prep("f1", 0, {}) is True
    out = capsys.readouterr().out
    assert "PREP damaged" in out and "moved aside" in out
    d = pathlib.Path(sup.prep_dir("f1"))
    assert len(list(d.glob("set.pkl.damaged-*"))) == 1
    assert len(list(d.glob("vocab.json.damaged-*"))) == 1
    assert sup.set_intact("f1", 0) == (True, "")
    assert calls == [("choose", "f1")]


def test_a_prep_of_another_iteration_is_moved_aside_and_redone(
        sup, prep_env, monkeypatch, capsys):
    """A prep of F0 at another iteration than its final one (a stale
    timeline's, or the planned 30) is not intact: before a launch,
    redone."""
    calls = prep_env
    _write_prep(sup, k=30)
    sup.write_pick(os.path.join(sup.prep_dir("f1"), "master.json"), PICK)
    _popen(monkeypatch, sup, [(0, lambda: _write_prep(sup))])
    assert sup.run_prep("f1", 0, {}) is True
    out = capsys.readouterr().out
    assert "PREP damaged" in out and "after 30 iterations" in out
    assert calls == [("choose", "f1")] and sup.set_intact("f1", 0) == (True,
                                                                       "")


def test_a_prep_that_never_checks_out_fails(sup, prep_env, monkeypatch,
                                            capsys):
    _popen(monkeypatch, sup,
           [(0, lambda: _write_prep(sup, good=False))] * 3)
    assert sup.run_prep("f1", 0, {}) is False
    assert "does not match" in capsys.readouterr().out


@pytest.mark.parametrize("doc, why", [
    ({}, ""),
    # the master: F0's arm, its final iteration 27, no offline steps
    ({"master_arm": G0_ARM}, "not F0's final snapshot (iteration 27)"),
    ({"master_arm": F[1]}, "not F0's final snapshot (iteration 27)"),
    ({"master_arm": GONE}, "vocab.json's master is None after 27"),
    ({"master_iterations": 26}, "after 26 iterations"),
    ({"master_iterations": 30}, "after 30 iterations"),     # the planned
    ({"master_iterations": 27.0}, "after 27.0 iterations"),
    ({"master_iterations": "27"}, "after '27' iterations"),
    ({"master_iterations": GONE}, "after None iterations"),
    ({"master_offline_steps": 20000}, "and 20000 offline steps, not F0's"),
    ({"master_offline_steps": False}, "and False offline steps"),
    ({"master_offline_steps": 0.0}, "and 0.0 offline steps"),
    ({"master_offline_steps": GONE}, "and None offline steps"),
    # the master's own snapshot (Generation 2's audit r1 B-L3)
    ({"master_snapshot": "other.pth"},
     "master snapshot is 'other.pth', not "),
    ({"master_snapshot": GONE}, "master snapshot is None"),
    # its own-play log, no recording
    ({"source": "recording"}, "source is 'recording' ("),
    ({"source": GONE}, "source is None ("),
    ({"log_dir": "elsewhere"}, "('elsewhere'), not F0's own-play log"),
    ({"log_dir": GONE}, "(None), not F0's own-play log"),
    ({"student_ceiling": None}, "no valid student_ceiling"),
    ({"student_ceiling": 0.0}, "no valid student_ceiling"),
    ({"student_ceiling": True}, "no valid student_ceiling"),
    ({"student_ceiling": "7.5"}, "no valid student_ceiling"),
    # the pre-registered discount: F0's 0.97 halved to 0.985
    ({"student_discount": None}, "student_discount is None, not F1's 0.985"),
    ({"student_discount": GONE}, "student_discount is None"),
    ({"student_discount": 0.0}, "student_discount is 0.0"),
    ({"student_discount": 1.0}, "student_discount is 1.0"),
    ({"student_discount": 0.998}, "student_discount is 0.998"),
    ({"student_discount": True}, "student_discount is True"),
    ({"student_discount": "0.985"}, "student_discount is '0.985'"),
    ({"student_discount": 0.97}, "student_discount is 0.97"),   # unhalved
    ({"student_discount": 0.9925}, "student_discount is 0.9925"),
    ({"student_discount": 0.98501}, "student_discount is 0.98501"),
    # the lineage's rule (Test B2): a prep without one is Test B's "halve"
    ({"discount_rule": GONE}, ""),
    ({"discount_rule": "keep"},
     "discount_rule is 'keep', not the lineage's 'halve'"),
    ({"discount_rule": None}, "discount_rule is None, not the lineage's"),
    # the master snapshot's bytes (audit r1 B-L2)
    ({"master_sha256": "0" * 64}, "master_sha256 is not "),
    ({"master_sha256": GONE}, "a prep of another timeline's snapshot"),
    ({"student_discount": 0.985}, "")])
def test_set_intact_checks_the_master_the_log_the_ceiling_and_the_discount(
        sup, doc, why):
    _finish(sup, "f0", final=27, evals=False, files=False)
    _write_prep(sup, **doc)
    ok, got = sup.set_intact("f1", 0)
    assert ok == (not why) and why in got, got
    pathlib.Path(sup.prep_dir("f1"), "set.pkl").unlink()
    assert sup.set_intact("f1", 0)[1].startswith("prep files:")


def test_set_intact_of_a_student_masters_prep(sup):
    """f2's master F1 is a student: its offline steps and its discount
    0.985 halved to 0.9925; its final iteration is its own (31)."""
    _finish(sup, "f1", final=31, evals=False, files=False)
    _write_prep(sup, "f2", k=31)
    assert sup.set_intact("f2", 0) == (True, "")
    _write_prep(sup, "f2", k=31, master_offline_steps=0)
    assert sup.set_intact("f2", 0) == (False, (
        f"vocab.json's master is {F[1]!r} after 31 iterations and 0 offline "
        f"steps, not F1's final snapshot (iteration 31)"))
    _write_prep(sup, "f2", k=31, student_discount=0.985)
    assert sup.set_intact("f2", 0) == (
        False, "vocab.json's student_discount is 0.985, not F2's 0.9925")
    _write_prep(sup, "f2", k=31, log_dir=sup.own_play_rel("f0", 0))
    assert "not F1's own-play log" in sup.set_intact("f2", 0)[1]


def test_set_intact_names_the_master_it_found(sup):
    _finish(sup, "f0", final=27, evals=False, files=False)
    _write_prep(sup, master_arm=G0_ARM, master_iterations=90,
                master_offline_steps=0)
    assert sup.set_intact("f1", 0) == (False, (
        f"vocab.json's master is {G0_ARM!r} after 90 iterations and 0 "
        f"offline steps, not F0's final snapshot (iteration 27)"))
    pathlib.Path(sup.prep_dir("f1"), "vocab.json").write_text("[1]")
    assert sup.set_intact("f1", 0) == (False, "vocab.json is not a JSON "
                                              "object")


def test_set_intact_reads_the_masters_final_row(sup):
    _finish(sup, "f0", final=27, evals=False, files=False)
    _write_prep(sup)
    assert sup.set_intact("f1", 0) == (True, "")
    ck = pathlib.Path(sup.paths("f0", 0)[3])
    ck.write_bytes(b"x" * 1_000_000)            # F0 not complete (moved?)
    assert sup.set_intact("f1", 0) == (False, (
        "the master: F0 has no completion marker -- its final row is not "
        "settled"))
    ck.write_bytes(b"done")
    # a later final row (another timeline): the prep is not of it
    _jsonl(sup, "f0", {28: 21000})
    ok, why = sup.set_intact("f1", 0)
    assert not ok and "after 27 iterations" in why and "(iteration 28)" in why


def test_a_launched_run_never_redoes_its_prep(sup, prep_env, monkeypatch,
                                             capsys):
    """A launched student is bound to the set; a transient lock or real
    damage blocks the launch (the integrity gate), never a redo."""
    calls = prep_env
    _write_prep(sup, good=False)
    _popen(monkeypatch, sup, [])                     # any launch fails
    assert sup.run_prep("f1", 0, {}, redo_ok=False) is True
    out = capsys.readouterr().out
    assert "never redone" in out and calls == []
    assert not list(pathlib.Path(sup.prep_dir("f1")).glob("*.damaged-*"))
    # a transient read failure that clears is not damage
    _write_prep(sup)
    sup.write_pick(os.path.join(sup.prep_dir("f1"), "master.json"), PICK)
    real = sup.set_intact
    flaky = [(False, "prep files: [Errno 13] locked"), None]

    def once(v, s):
        r = flaky.pop(0)
        return r if r else real(v, s)

    monkeypatch.setattr(sup, "set_intact", once)
    assert sup.run_prep("f1", 0, {}) is True
    assert "damaged" not in capsys.readouterr().out and calls == []


def test_a_launched_run_never_starts_a_prep(sup, prep_env, monkeypatch,
                                            capsys):
    """vocab.json missing after the student was launched: no prep at
    all."""
    calls = prep_env
    d = pathlib.Path(sup.prep_dir("f1"))
    d.mkdir(parents=True)
    (d / "set.pkl").write_bytes(b"bound")
    monkeypatch.setattr(sup.subprocess, "Popen", _stop)
    assert sup.run_prep("f1", 0, {}, redo_ok=False) is True
    assert "never redone" in capsys.readouterr().out and calls == []
    assert (d / "set.pkl").read_bytes() == b"bound"


def test_the_prep_needs_the_audited_tree(sup, prep_env, monkeypatch, capsys):
    monkeypatch.setattr(sup, "manifest_check",
                        lambda: (False, "not the audited revision: x"))
    monkeypatch.setattr(sup.subprocess, "Popen", _stop)
    assert sup.run_prep("f1", 0, {}) is False
    assert "PREP FAILED: not the audited revision" in capsys.readouterr().out


def test_an_orphaned_prep_that_finished_counts(sup, prep_env, monkeypatch,
                                               capsys):
    """The next attempt's guard waited out an orphaned prep that wrote the
    prep: no second prep."""
    monkeypatch.setattr(sup, "wait_no_trainer", lambda *a: _write_prep(sup))
    monkeypatch.setattr(sup.subprocess, "Popen", _stop)
    assert sup.run_prep("f1", 0, {}) is True
    assert "PREP done (by an earlier prep)" in capsys.readouterr().out


def test_stale_prep_tmp_files_are_swept(sup, prep_env, monkeypatch):
    d = pathlib.Path(sup.prep_dir("f1"))
    d.mkdir(parents=True)
    (d / "vocab.json.4242.tmp").write_bytes(b"x")
    (d / "keep.txt").write_bytes(b"x")
    _popen(monkeypatch, sup, [(0, lambda: _write_prep(sup))])
    assert sup.run_prep("f1", 0, {}) is True
    assert not list(d.glob("*.tmp")) and (d / "keep.txt").exists()


def test_a_finished_prep_without_its_master_json_is_refused(
        sup, prep_env, monkeypatch, capsys):
    _write_prep(sup)
    monkeypatch.setattr(sup.subprocess, "Popen", _stop)
    assert sup.run_prep("f1", 0, {}) is False
    assert "PREP FAILED: a finished prep without a readable master.json" in \
        capsys.readouterr().out
    assert sup.run_prep("f1", 0, {}, redo_ok=False) is True   # launched


def test_each_student_has_its_own_prep_and_f0_none(sup):
    assert len({sup.prep_dir(v) for v in V[1:]}) == 4
    for v, c in (("f1", 7.5), ("f2", 12.0)):
        pathlib.Path(sup.prep_dir(v)).mkdir(parents=True)
        pathlib.Path(sup.prep_dir(v), "vocab.json").write_text(json.dumps(
            {"student_ceiling": c}))
    assert (sup.prep_ceiling("f1"), sup.prep_ceiling("f2"),
            sup.prep_ceiling("f3")) == (7.5, 12.0, None)
    assert (sup.run_ceiling("f1"), sup.run_ceiling("f3")) == (7.5, None)
    # F0's is its sign clip, whatever a folder of its name holds
    pathlib.Path(sup.prep_dir("f0")).mkdir(parents=True)
    pathlib.Path(sup.prep_dir("f0"), "vocab.json").write_text(json.dumps(
        {"student_ceiling": 9.0}))
    assert sup.run_ceiling("f0") == sup.F0_CEILING == 1.0


# --------------------------------------------------------------------------
# the trainer guard, one supervisor per run
# --------------------------------------------------------------------------
def _psutil(monkeypatch, procs):
    import psutil
    monkeypatch.setattr(psutil, "process_iter", lambda attrs: [
        SimpleNamespace(info=p) for p in procs])


GUARD_PROCS = [
    (1, "python.exe", ["py", "Y:" + chr(92) + "run" + chr(92) + "experiments"
                       + chr(92) + "ablation.py", "--arms", F[1]]),
    # Generation 2's run, live when Test B is launched
    (2, "python.exe", ["py", "experiments/ablation.py", "--arms", G2_ARM,
                       "--seed", "0"]),
    (3, "Python3.12", ["py", "experiments/ablation.py", "--arms", "pong-x"]),
    (4, "python.exe", ["py", "experiments" + chr(92) + "distill.py",
                       "--master", "m.pth", "--from-log", "x"]),  # a prep
    (5, "python.exe", ["py", "benchmarks/probes/probe_exhaustive_eval.py",
                       "x.pth"]),
    (6, "python.exe", ["py", "-m", "pytest", "tests/test_testb.py"]),
    (7, "python.exe", None),                # unreadable: never counted
    (8, "Code.exe", ["code", "experiments/ablation.py"]),
    (9, "git.exe", ["git", "diff", "--", "experiments/distill.py"]),
    (10, None, ["experiments/ablation.py"]),
    (11, "python.exe", ["py", "tests/test_distill.py"]),
    (12, "python.exe", ["py", "ablation.py", "--arms", F[0]]),
    (13, "python.exe", ["py", "-m", "experiments.distill", "--master", "m"]),
    (14, "python.exe", ["py", "-m", "ablation"]),
    (15, "python.exe", ["py", "-m", "pytest", "ablation"]),
    (16, "python.exe", ["py", "experiments/ablation.pyc"]),
    (17, "python.exe", ["py", "benchmarks/supervisors/supervise_testb.py",
                        "f1", "0"]),
    (18, "python.exe", ["py", "-m", "distill"])]


def test_live_trainers_finds_any_run_or_prep(sup, g2sup, monkeypatch):
    """is_trainer as supervise_gen2.py's: the same verdict on every command
    line; the prep counts; python only; unreadable not counted."""
    fresh = _load(SUPERVISORS / "supervise_testb.py", "supervise_testb_procs")
    monkeypatch.setattr(sup, "_procs", fresh._procs)
    _psutil(monkeypatch, [{"pid": p, "name": n, "cmdline": c}
                          for p, n, c in GUARD_PROCS])
    assert sup.live_trainers() == [1, 2, 3, 4, 12, 13, 14, 18]
    for _p, _n, cmd in GUARD_PROCS:
        if cmd is not None:
            assert sup.is_trainer(cmd) == g2sup.is_trainer(cmd), cmd
    assert sup.TRAINER_SCRIPTS == g2sup.TRAINER_SCRIPTS
    assert sup.TRAINER_MODULES == g2sup.TRAINER_MODULES


def _procs(monkeypatch, sup, procs):
    monkeypatch.setattr(sup, "_procs", lambda: iter(procs))


def test_live_scripts_match_the_exact_name_and_variant(sup, monkeypatch):
    procs = [(1, ["py", r"C:\x\supervise_testb.py", "f1", "0"]),
             (2, ["py", "benchmarks/supervisors/supervise_testb.py", "f1",
                  "0"]),
             (3, ["py", "supervise_testb.py.bak", "f1", "0"]),
             (4, ["py", "benchmarks/supervisors/supervise_gen2.py", "g2",
                  "0"]),
             (5, ["py", "supervise_testb.py", "f0", "0"]),
             (6, ["py", "experiments/ablation.py", "--arms", "x"]),
             (7, ["py", "supervise_gen2.py", "f1", "0"]),
             (8, ["py", "supervise_testb.py", "f2", "0"])]
    _procs(monkeypatch, sup, procs)
    assert sup.live_scripts("supervise_testb.py") == [1, 2, 5, 8]
    assert sup.live_scripts("supervise_testb.py", "f1") == [1, 2]
    assert sup.live_scripts("supervise_testb.py", "f0") == [5]
    assert sup.live_scripts("supervise_gen2.py") == [4, 7]
    assert sup.live_trainers() == [6]
    monkeypatch.setattr(sup, "_mine", lambda: {1})
    assert sup.other_supervisors("f1") == [2]
    monkeypatch.setattr(sup, "_mine", lambda: {1, 2})
    assert sup.other_supervisors("f1") == []


def test_one_supervisor_per_run(sup, monkeypatch, capsys):
    def no(*a, **k):
        raise AssertionError("a second supervisor must stop at once")

    _procs(monkeypatch, sup, [
        (20, ["py", "benchmarks/supervisors/supervise_testb.py", "f1", "0"])])
    monkeypatch.setattr(sup, "_mine", lambda: {99})
    for name in ("run_done", "wait_external", "preflight", "settle_stop"):
        monkeypatch.setattr(sup, name, no)
    assert sup.main(["f1", "0"]) == 2
    out = capsys.readouterr().out
    assert "refusing: another supervise_testb.py f1 is alive (pids [20])" in \
        out
    # the other generations' supervisors (all five run side by side),
    # Generation 2's or a backup copy are no obstacle
    _procs(monkeypatch, sup, [
        (21, ["py", "benchmarks/supervisors/supervise_testb.py", "f0", "0"]),
        (22, ["py", "benchmarks/supervisors/supervise_testb.py", "f2", "0"]),
        (23, ["py", r"Y:\SwitchPlayer-gen2-run\benchmarks\supervisors"
                    r"\supervise_gen2.py", "g2", "0"]),
        (24, ["py", "supervise_testb.py.bak", "f1", "0"])])
    monkeypatch.setattr(sup, "run_done", lambda v, s: True)
    assert sup.main(["f1", "0"]) == 0
    assert "RUN COMPLETE f1 s0" in capsys.readouterr().out


def test_wait_no_trainer_waits_out_any_trainer_and_streams_rows(
        sup, monkeypatch, tmp_path, capsys):
    jsonl = tmp_path / "a.jsonl"
    rows = sup.Rows(str(jsonl), "f0", 0)
    seen = [[7], [7], []]
    monkeypatch.setattr(sup, "live_trainers",
                        lambda: seen.pop(0) if seen else [])
    sleeps = []

    def sleep(s):
        sleeps.append(s)
        jsonl.write_text(json.dumps({"iteration": 25, "reward_ceiling": 99.0,
                                     "elapsed_s": 1.0}) + NL)

    monkeypatch.setattr(sup.time, "sleep", sleep)
    sup.wait_no_trainer(rows)
    out = capsys.readouterr().out
    assert out.count("ALERT: an ablation.py or distill.py trainer is "
                     "already running (pids [7])") == 1     # told once
    assert sleeps == [60, 60]
    assert "EVAL f0 s0 iter 25" in out            # an orphan stays watched
    assert "ALERT: f0 s0 row 25: reward_ceiling 99.0" in out


def test_the_preflight_waits_for_no_trainer(sup, monkeypatch, capsys):
    """Audit r1 B-L1 (PB's r1 B-L2): a RESUMING relaunch beside an
    orphaned trainer or prep -- the preflight (the suites, Ray-spinning
    slow tests among them) is not called until none is alive; the orphan's
    rows stream meanwhile (primed first: an old row is not re-printed).
    Sleeps bounded: a guard that never returns fails instead of hanging."""
    _finish(sup, "f0", final=27, env=20400, evals=False, files=False)
    _rel, base, jsonl, ck, _d = sup.paths("f1", 0)
    pathlib.Path(base).mkdir(parents=True)
    pathlib.Path(ck).write_bytes(b"x" * 1_000_001)   # a resume: no wait
    _write_prep(sup)

    def row(it):
        with open(jsonl, "a") as f:
            f.write(json.dumps({"iteration": it, "elapsed_s": float(it),
                                "reward_ceiling": 7.5}) + NL)

    row(3)                                           # before the prime
    alive = [7]
    monkeypatch.setattr(sup, "live_trainers", lambda: list(alive))

    def effect(k):
        if k == 1:
            row(4)                                   # the orphan writes
        if k == 3:
            alive.clear()                            # the trainer exits

    sleep = _bounded(effect, n=5)
    monkeypatch.setattr(sup.time, "sleep", sleep)
    seen = []

    def preflight(v, s):
        seen.append((v, list(alive)))
        raise Stop

    monkeypatch.setattr(sup, "preflight", preflight)
    with pytest.raises(Stop):
        sup.main(["f1", "0"])
    assert seen == [("f1", [])]
    assert sleep.calls == [60, 60, 60]
    out = capsys.readouterr().out
    assert "RESUMING f1 s0" in out
    assert "STOP f1 s0: --stop-env-steps 19600" in out
    assert "trainer is already running (pids [7])" in out
    assert "EVAL f1 s0 iter 4" in out
    assert "EVAL f1 s0 iter 3" not in out


# --------------------------------------------------------------------------
# main: the order, the launch gate, the child
# --------------------------------------------------------------------------
@pytest.mark.parametrize("v", ["f0", "f4"])
def test_supervisor_exits_at_once_when_its_run_is_complete(sup, monkeypatch,
                                                           capsys, v):
    monkeypatch.setattr(sup, "wait_external", _stop)
    monkeypatch.setattr(sup, "settle_stop", _stop)
    ck = pathlib.Path(sup.paths(v, 0)[3])
    ck.parent.mkdir(parents=True)
    ck.write_bytes(b"done")
    assert sup.main([v, "0"]) == 0
    assert f"RUN COMPLETE {v} s0" in capsys.readouterr().out


def test_a_resume_skips_the_wait(sup, monkeypatch, capsys):
    _finish(sup, "f0", final=27, env=20400, evals=False, files=False)
    _finish(sup, "f1", final=26, env=19800, evals=False, files=False)
    monkeypatch.setattr(sup, "wait_external",
                        lambda *a: (_ for _ in ()).throw(
                            AssertionError("a resume must not wait")))
    monkeypatch.setattr(sup, "preflight", _stop)
    ck = pathlib.Path(sup.paths("f2", 0)[3])
    ck.parent.mkdir(parents=True)
    ck.write_bytes(b"x" * 1_000_001)
    with pytest.raises(Stop):
        sup.main(["f2", "0"])
    out = capsys.readouterr().out
    assert "RESUMING f2 s0" in out
    # the stop all the same: 60,000 less F0's and F1's final env steps
    assert "STOP f2 s0: --stop-env-steps 19800 = 60000 - 40200" in out
    ck.unlink()
    # a LAUNCHED line in its own log is a resume too
    (ck.parent / "supervisor.log").write_text(
        "LAUNCHED f2 s0 attempt 1 (fresh), pid 1" + NL)
    with pytest.raises(Stop):
        sup.main(["f2", "0"])
    assert "RESUMING f2 s0" in capsys.readouterr().out
    (ck.parent / "supervisor.log").unlink()
    monkeypatch.setattr(sup, "wait_external", _stop)
    with pytest.raises(Stop):
        sup.main(["f2", "0"])


def test_launched(sup, tmp_path):
    base, ck = tmp_path / "b", tmp_path / "b" / "c.ckpt"
    base.mkdir()
    assert not sup.launched(str(base), str(ck))
    (base / "supervisor.log").write_text("STOP f1 s0: --stop-env-steps 1"
                                         + NL)
    assert not sup.launched(str(base), str(ck))
    (base / "supervisor.log").write_text("x LAUNCHED f1" + NL)
    assert not sup.launched(str(base), str(ck))
    (base / "supervisor.log").write_text("LAUNCHED f1 s0 attempt 1" + NL)
    assert sup.launched(str(base), str(ck))
    (base / "supervisor.log").write_text("")
    ck.write_bytes(b"x")
    assert sup.launched(str(base), str(ck))


class Dead:
    pid, returncode = 99, 1

    def poll(self):
        return 1


def _loop(sup, monkeypatch, events, preps=None, stop=None):
    """main's collaborators stubbed into `events`; `stop` (not None) stubs
    the stop too ("stop"), else the real one is read from the tmp
    lineage."""
    monkeypatch.setattr(sup, "wait_external",
                        lambda v, s: events.append("wait") or "ready")
    if stop is not None:
        monkeypatch.setattr(sup, "stop_env_steps",
                            lambda v, s: events.append("stop") or stop)
    monkeypatch.setattr(sup, "preflight",
                        lambda v, s: events.append("preflight") or (True, ""))

    def run_prep(v, s, env, redo_ok):
        events.append(("prep", redo_ok))
        if preps is not None:
            preps.append(env)
        return True

    monkeypatch.setattr(sup, "run_prep", run_prep)
    monkeypatch.setattr(sup, "wait_no_trainer",
                        lambda *a: events.append("guard"))
    monkeypatch.setattr(sup, "manifest_check",
                        lambda: events.append("manifest") or (True, ""))
    monkeypatch.setattr(sup, "set_intact",
                        lambda v, s: events.append("intact") or (True, ""))
    monkeypatch.setattr(sup.time, "sleep", lambda s: None)


def test_f0_runs_preflight_then_each_launch_guarded_without_a_prep(
        sup, monkeypatch, capsys):
    events, launched = [], []
    _loop(sup, monkeypatch, events)                  # f0's stop: 20,000
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    ck = pathlib.Path(sup.paths("f0", 0)[3])

    def popen(args, **k):
        events.append("popen")
        launched.append((list(args), k))
        if len(launched) == 1:          # the first attempt leaves a checkpoint
            ck.write_bytes(b"x" * 1_000_001)
        return Dead()

    monkeypatch.setattr(sup.subprocess, "Popen", popen)
    assert sup.main(["f0", "0"]) == 1                 # 5 quick deaths
    out = capsys.readouterr().out
    assert "CRASH LOOP" in out
    assert ("preflight ok (manifest + quick probe + 29 suites); starting f0 "
            f"s0 ({F[0]}, --stop-env-steps 20000, at most 60 iterations)") \
        in out
    # no wait, no prep, no prep integrity: f0 has none
    assert events == (["guard", "preflight"]
                      + ["guard", "manifest", "popen"] * 5)
    rel = sup.paths("f0", 0)[0]
    for i, (args, k) in enumerate(launched):
        assert args == sup.command("f0", 0, 20000) + (["--resume"] if i
                                                      else [])
        assert args[args.index("--stop-env-steps") + 1] == "20000"
        assert "--token-vocab" not in args
        assert k["cwd"] == sup.REPO
        assert k["env"]["ABLATION_RESULTS"] == rel
        assert k["env"]["PYTHONFAULTHANDLER"] == "1"
        assert "CUDA_VISIBLE_DEVICES" not in k["env"]


def test_a_student_runs_wait_stop_preflight_prep_then_each_launch_guarded(
        sup, monkeypatch, capsys):
    events, preps, launched = [], [], []
    _loop(sup, monkeypatch, events, preps, stop=19600)
    # the preflight's CPU-only env must never reach the trainer or the prep
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    ck = pathlib.Path(sup.paths("f1", 0)[3])

    def popen(args, **k):
        events.append("popen")
        launched.append((list(args), k))
        if len(launched) == 1:          # the first attempt leaves a checkpoint
            ck.write_bytes(b"x" * 1_000_001)
        return Dead()

    monkeypatch.setattr(sup.subprocess, "Popen", popen)
    assert sup.main(["f1", "0"]) == 1                 # 5 quick deaths
    assert "CRASH LOOP" in capsys.readouterr().out
    assert events == (["wait", "stop", "guard", "preflight", ("prep", True)]
                      + ["guard", "manifest", "intact", "popen"] * 5)
    rel = sup.paths("f1", 0)[0]
    for i, (args, k) in enumerate(launched):
        assert args == sup.command("f1", 0, 19600) + (["--resume"] if i
                                                      else [])
        assert k["cwd"] == sup.REPO
        assert k["env"]["ABLATION_RESULTS"] == rel
        assert k["env"]["PYTHONFAULTHANDLER"] == "1"
        assert "CUDA_VISIBLE_DEVICES" not in k["env"]
    # the prep's env: neither the run folder nor the CPU-only flag
    assert "ABLATION_RESULTS" not in preps[0]
    assert "CUDA_VISIBLE_DEVICES" not in preps[0]
    # --token-vocab / --distill-set name the prep's files, from the child's
    # cwd
    args, k = launched[0]
    for flag, name in (("--token-vocab", "vocab.json"),
                       ("--distill-set", "set.pkl")):
        assert os.path.normpath(os.path.join(
            k["cwd"], args[args.index(flag) + 1])) == os.path.normpath(
            os.path.join(sup.prep_dir("f1"), name))


def test_a_failed_prep_launches_nothing(sup, monkeypatch):
    events = []
    _loop(sup, monkeypatch, events, stop=19600)
    monkeypatch.setattr(sup, "run_prep", lambda v, s, env, redo_ok: False)
    monkeypatch.setattr(sup.subprocess, "Popen", _stop)
    assert sup.main(["f1", "0"]) == 1


def test_a_broken_prep_blocks_the_launch_until_intact(sup, monkeypatch,
                                                      capsys):
    events = []
    _loop(sup, monkeypatch, events, stop=19600)
    checks = [(False, "set.pkl does not match vocab.json's set_sha256"),
              (False, "set.pkl does not match vocab.json's set_sha256"),
              (True, "")]
    monkeypatch.setattr(sup, "set_intact",
                        lambda v, s: events.append("check") or checks.pop(0))
    monkeypatch.setattr(sup.subprocess, "Popen", _stop)
    with pytest.raises(Stop):
        sup.main(["f1", "0"])
    pre = events.index("preflight")
    assert events[pre - 1] == "guard"            # the guard before it
    gate = events[events.index("guard", pre):]
    assert gate == ["guard", "manifest", "check"] * 3
    out = capsys.readouterr().out
    assert out.count("not relaunching until") == 1          # told once


@pytest.mark.parametrize("v, head, gate", [
    ("f0", ["guard", "preflight"], ["popen"]),
    ("f1", ["wait", "stop", "guard", "preflight", ("prep", True)],
     ["intact", "popen"])])
def test_a_failing_manifest_blocks_the_launch(sup, monkeypatch, v, head,
                                              gate):
    events = []
    _loop(sup, monkeypatch, events, stop=None if v == "f0" else 19600)
    checks = [(False, "not the audited revision: x"), (True, "")]
    monkeypatch.setattr(sup, "manifest_check",
                        lambda: events.append("manifest") or checks.pop(0))
    monkeypatch.setattr(sup.subprocess, "Popen",
                        lambda *a, **k: events.append("popen") or _stop())
    with pytest.raises(Stop):
        sup.main([v, "0"])
    assert events[:len(head)] == head
    assert events[len(head):] == ["guard", "manifest", "guard",
                                  "manifest"] + gate


def test_stray_state(sup, monkeypatch, capsys):
    _finish(sup, "f0", final=27, env=20400, evals=False, files=False)
    monkeypatch.setattr(sup, "wait_external", _stop)
    rel, base, _j, ckpt, _d = sup.paths("f1", 0)
    base = pathlib.Path(base)
    off = base / f"{base.name}-offline" / "snapshots-x"
    off.mkdir(parents=True)
    (off / "weights-iter001.pth").write_bytes(b"x")
    own = base / "snapshots-x"
    own.mkdir()
    (own / "weights-iter000.pth").write_bytes(b"x")
    assert len(sup.stray_snapshots(str(base), ckpt)) == 2
    # never launched: no excuse, even for exactly row 0 and the curve
    assert sup.main(["f1", "0"]) == 2
    assert "snapshots but no checkpoint" in capsys.readouterr().out
    # F1's own death before its first checkpoint: launched, and exactly
    # row 0 + the offline curve excused
    (base / "supervisor.log").write_text("LAUNCHED f1 s0 attempt 1 (fresh)"
                                         + NL)
    assert sup.stray_snapshots(str(base), ckpt, own_ok=True) == []
    monkeypatch.setattr(sup, "preflight", _stop)
    with pytest.raises(Stop):
        sup.main(["f1", "0"])
    (own / "weights-iter005.pth").write_bytes(b"x")  # a later row: stray
    assert sup.main(["f1", "0"]) == 2
    (own / "weights-iter005.pth").unlink()
    capsys.readouterr()
    # probe rows without a checkpoint: refused even when launched
    _rows(_probe(sup, "f1", "-acting", offline=True), [1])
    assert sup.main(["f1", "0"]) == 2
    assert "probe rows" in capsys.readouterr().out
    pathlib.Path(ckpt).write_bytes(b"x" * 1_000_001)
    assert sup.stray_snapshots(str(base), ckpt) == []
    assert sup.stray_probe_rows("f1", 0, ckpt) == []


def test_f0_gets_no_own_death_excuse(sup, monkeypatch, capsys):
    """F0 writes no row 0 and no offline curve (from scratch): a
    weights-iter000 in its folder is another timeline's, launched or
    not."""
    monkeypatch.setattr(sup, "settle_stop", _stop)
    base = pathlib.Path(sup.paths("f0", 0)[1])
    (base / "snapshots-x").mkdir(parents=True)
    (base / "snapshots-x" / "weights-iter000.pth").write_bytes(b"x")
    (base / "supervisor.log").write_text("LAUNCHED f0 s0 attempt 1 (fresh)"
                                         + NL)
    assert sup.main(["f0", "0"]) == 2
    assert "snapshots but no checkpoint" in capsys.readouterr().out


def test_the_attempt_loop_refuses_stray_probe_rows(sup, monkeypatch, capsys):
    events = []
    _loop(sup, monkeypatch, events, stop=19600)
    monkeypatch.setattr(sup.subprocess, "Popen", _stop)

    def rows_appear(*a):
        _rows(_probe(sup, "f1", ""), [5])

    monkeypatch.setattr(sup, "wait_no_trainer", rows_appear)
    assert sup.main(["f1", "0"]) == 2
    assert "stray snapshots / probe jsonls" in capsys.readouterr().out


def test_the_attempt_loop_refuses_stray_snapshots_of_an_unlaunched_run(
        sup, monkeypatch, capsys):
    """Snapshots appearing while the first launch waited (no LAUNCHED line
    yet): no excuse in the loop either."""
    events = []
    _loop(sup, monkeypatch, events, stop=19600)
    monkeypatch.setattr(sup.subprocess, "Popen", _stop)
    base = pathlib.Path(sup.paths("f1", 0)[1])

    def snap_appears(*a):
        (base / "snapshots-x").mkdir(parents=True, exist_ok=True)
        (base / "snapshots-x" / "weights-iter000.pth").write_bytes(b"x")

    monkeypatch.setattr(sup, "wait_no_trainer", snap_appears)
    assert sup.main(["f1", "0"]) == 2
    assert "weights-iter000.pth" in capsys.readouterr().out


def test_a_leftover_read5_jsonl_is_refused_like_the_exact_evals(sup,
                                                                capsys):
    """As Test 2's rebuild (PB's audit r1 B-M1): read (5)'s probe_prefix_bn
    jsonl is one of the run's probe jsonls -- rows there without a
    checkpoint refuse a fresh start (the hand-off would never re-probe
    those snapshots on the new timeline)."""
    run = sup.run_name("f2", 0)
    probes = pathlib.Path(sup.REPO) / "benchmarks" / "probes"
    assert [os.path.basename(p) for p in sup.probe_jsonls("f2", 0)] == [
        f"probe_exhaustive_eval-{run}-acting.jsonl",
        f"probe_exhaustive_eval-{run}.jsonl",
        f"probe_exhaustive_eval-{run}-offline-acting.jsonl",
        f"probe_exhaustive_eval-{run}-offline.jsonl",
        f"probe_prefix_bn-{run}.jsonl"]
    ckpt = sup.paths("f2", 0)[3]
    mech = probes / f"probe_prefix_bn-{run}.jsonl"
    mech.parent.mkdir(parents=True, exist_ok=True)
    mech.write_text(json.dumps({"iteration": 0, "p1_ratio": 0.5}) + NL)
    assert sup.stray_probe_rows("f2", 0, ckpt) == [mech.name]
    assert sup.main(["f2", "0"]) == 2
    assert f"probe rows ({mech.name})" in capsys.readouterr().out
    pathlib.Path(ckpt).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(ckpt).write_bytes(b"x" * 1_000_001)
    assert sup.stray_probe_rows("f2", 0, ckpt) == []


def test_completion_and_abandon_rechecked_before_each_launch(
        sup, monkeypatch, capsys):
    events = []
    _loop(sup, monkeypatch, events)
    monkeypatch.setattr(sup.subprocess, "Popen", _stop)
    base = pathlib.Path(sup.paths("f0", 0)[1])
    ckpt = pathlib.Path(sup.paths("f0", 0)[3])
    monkeypatch.setattr(sup, "wait_no_trainer",
                        lambda *a: (base / "ABANDONED").write_text(""))
    assert sup.main(["f0", "0"]) == 0
    assert "ABANDONED f0 s0" in capsys.readouterr().out
    (base / "ABANDONED").unlink()
    monkeypatch.setattr(sup, "wait_no_trainer",
                        lambda *a: ckpt.write_bytes(b"done"))
    assert sup.main(["f0", "0"]) == 0
    assert "RUN COMPLETE f0 s0" in capsys.readouterr().out


def test_a_resumed_student_never_redoes_its_prep(sup, monkeypatch):
    events = []
    _loop(sup, monkeypatch, events, stop=19550)
    monkeypatch.setattr(sup.subprocess, "Popen", _stop)
    ck = pathlib.Path(sup.paths("f3", 0)[3])
    ck.parent.mkdir(parents=True)
    ck.write_bytes(b"x" * 1_000_001)                  # a resume
    with pytest.raises(Stop):
        sup.main(["f3", "0"])
    assert events == ["stop", "guard", "preflight", ("prep", False),
                      "guard", "manifest", "intact"]


def test_an_abandoned_student_runs_no_prep(sup, monkeypatch, capsys):
    events = []
    _loop(sup, monkeypatch, events, stop=19600)
    base = pathlib.Path(sup.paths("f1", 0)[1])

    def preflight(v, s):
        base.mkdir(parents=True, exist_ok=True)
        (base / "ABANDONED").write_text("")
        return True, ""

    monkeypatch.setattr(sup, "preflight", preflight)
    assert sup.main(["f1", "0"]) == 0
    assert not any(isinstance(e, tuple) for e in events)
    assert "no prep" in capsys.readouterr().out


def test_main_returns_when_abandoned_while_waiting(sup, monkeypatch, capsys):
    events = []
    _loop(sup, monkeypatch, events, stop=19600)
    monkeypatch.setattr(sup, "wait_external", lambda v, s: "abandoned")
    monkeypatch.setattr(sup, "preflight", _stop)
    assert sup.main(["f1", "0"]) == 0
    assert "ABANDONED f1 s0" in capsys.readouterr().out
    monkeypatch.setattr(sup, "wait_external", lambda v, s: "failed")
    assert sup.main(["f1", "0"]) == 1
    assert "stop" not in events                       # never reached


def test_main_source_order(sup):
    src = inspect.getsource(sup.main)
    assert src.index("other_supervisors(") < src.index("run_done(") < \
        src.index("abandoned(base)") < src.index("stray_snapshots(") < \
        src.index("wait_external(") < src.index("settle_stop(")
    # the stop before the preflight; the guard once before the preflight
    # (audit r1 B-L1), after the prime
    assert src.index("settle_stop(") < src.index("rows.prime()") < \
        src.index("wait_no_trainer(rows)") < src.index(
            "preflight(variant, seed)")
    assert src.index("preflight(variant, seed)") < src.index("run_prep(")
    assert src.index("run_prep(") < src.index("for attempt in range(")
    loop = src[src.index("for attempt in range("):]
    assert loop.index("wait_no_trainer(rows)") < loop.index(
        "manifest_check()") < loop.index("set_intact(variant, seed)") < \
        loop.index("subprocess.Popen(")
    assert src.count("wait_no_trainer(") == 2       # before the preflight
    assert "redo_ok=not launched(base, ckpt)" in src
    # every launch: the stop computed once, the same each attempt
    assert "command(variant, seed, stop)" in src
    assert src.count("stop_env_steps(") == 0 and src.count("settle_stop(") \
        == 1
    # the chain's branches read is_student (no per-variant equality); the
    # prep, its integrity gate and the own-death excuse are a student's
    assert "variant ==" not in src
    assert src.count("if student:") == 1
    assert "if ok and student:" in src
    assert src.count("stray_snapshots(base, ckpt,") == 2
    assert src.count("student and launched(base, ckpt)") == 2
    assert src.count("launched(base, ckpt)") == 4


# --------------------------------------------------------------------------
# CRASH LOOP / NO PROGRESS (supervise_gen0.py's rule)
# --------------------------------------------------------------------------
def _clock(sup, monkeypatch, t0=1_000_000.0):
    import time as real_time
    clock = [t0]

    def sleep(s):
        clock[0] += s                       # backoffs take clock time

    monkeypatch.setattr(sup, "time", SimpleNamespace(
        time=lambda: clock[0], sleep=sleep, strftime=real_time.strftime))
    return clock


@pytest.mark.parametrize("life, stop, n", [
    (240, "CRASH LOOP", 5),          # quick deaths: CRASH LOOP's
    (480, "NO PROGRESS", 3),         # Test 1 audit r2 LOW-1: the old gap
    (1200, "NO PROGRESS", 3),
    (3 * 3600, "NO PROGRESS", 3)])
def test_every_repeating_death_without_a_checkpoint_stops(
        sup, monkeypatch, capsys, life, stop, n):
    events = []
    _loop(sup, monkeypatch, events, stop=19600)
    clock = _clock(sup, monkeypatch)
    launches = []

    class Dies:
        pid, returncode = 99, 1

        def poll(self):
            clock[0] += life
            return 1

    monkeypatch.setattr(sup.subprocess, "Popen",
                        lambda args, **k: launches.append(args) or Dies())
    assert sup.main(["f1", "0"]) == 1
    out = capsys.readouterr().out
    assert stop in out and "MAX ATTEMPTS" not in out
    assert len(launches) == n
    assert sup.STALL_MIN_S == 300 < 465


def test_a_checkpoint_resets_the_no_progress_count(sup, monkeypatch,
                                                   capsys):
    events = []
    _loop(sup, monkeypatch, events, stop=19600)
    clock = _clock(sup, monkeypatch)
    ck = pathlib.Path(sup.paths("f1", 0)[3])
    launches = []

    class Slow:
        pid, returncode = 99, 1

        def poll(self):
            clock[0] += 1200                # each attempt lives 20 min
            return 1

    def popen(args, **k):
        launches.append(args)
        if len(launches) == 2:              # attempt 2 writes a checkpoint
            ck.write_bytes(b"x" * 1_000_001)
            os.utime(ck, (clock[0] + 600, clock[0] + 600))
        return Slow()

    monkeypatch.setattr(sup.subprocess, "Popen", popen)
    assert sup.main(["f1", "0"]) == 1
    out = capsys.readouterr().out
    assert "NO PROGRESS: 3 attempts in a row" in out
    assert "CRASH LOOP" not in out
    assert len(launches) == 5                # 1 stall, reset, then 3 stalls
    assert launches[2][-1] == "--resume" and "--resume" not in launches[0]


# --------------------------------------------------------------------------
# rows: each run's constant ceiling
# --------------------------------------------------------------------------
C = 7.5


@pytest.mark.parametrize("row, per, want", [
    ({"iteration": 1, "reward_ceiling": C, "reward_max": C}, "n/a", []),
    ({"iteration": 30, "reward_ceiling": C, "reward_max": 8 * C}, "24.9", []),
    ({"iteration": 9, "reward_ceiling": 7.4999}, "n/a", []),   # 4 decimals
    ({"iteration": 9, "reward_ceiling": 7.4}, "n/a", ["reward_ceiling"]),
    ({"iteration": 5, "reward_ceiling": 1.0}, "n/a", ["reward_ceiling"]),
    ({"iteration": 27, "reward_ceiling": 160.0}, "n/a", ["reward_ceiling"]),
    ({"iteration": 5}, "n/a", ["reward_ceiling"]),
    ({"iteration": 0, "reward_ceiling": None}, "n/a", []),
    ({"iteration": 20, "reward_ceiling": C}, "25.5", ["min/iteration"]),
    ({"iteration": 20, "reward_ceiling": C, "reward_max": 8 * C + 0.01},
     "n/a", ["reward_max"]),
    ({"iteration": 20, "reward_ceiling": 160.0, "reward_max": 400.0}, "30",
     ["reward_ceiling", "min/iteration", "reward_max"]),
    # past the support's edge (both reward_max alerts)
    ({"iteration": 20, "reward_ceiling": C, "reward_max": 400.5}, "n/a",
     ["8 x the ceiling", "support's edge"]),
])
def test_row_alerts(sup, row, per, want):
    got = sup.row_alerts(row, per, C)
    assert len(got) == len(want), got
    for g, w in zip(got, want):
        assert w in g


def test_row_alerts_at_the_rounded_bound(sup):
    c = 7.123457
    bound = round(8 * c, 4)              # what the row logs at the bound
    assert bound > 8 * c
    assert sup.row_alerts({"iteration": 3, "reward_ceiling": round(c, 4),
                           "reward_max": bound}, "n/a", c) == []


def test_row_alerts_without_a_ceiling(sup):
    got = sup.row_alerts({"iteration": 5, "reward_ceiling": 7.5,
                          "reward_max": 1e6}, "n/a", None)
    assert len(got) == 2 and "no valid student_ceiling" in got[0]
    assert "support's edge" in got[1]          # checked without a ceiling


def test_rows_check_against_the_preps_ceiling(sup, tmp_path, capsys):
    jsonl = tmp_path / "a.jsonl"
    rows = sup.Rows(str(jsonl), "f1", 0)

    def row(**r):
        with open(jsonl, "a") as f:
            f.write(json.dumps({"elapsed_s": 1.0, **r}) + NL)

    row(iteration=1, reward_ceiling=7.5)
    rows.emit()                                  # no prep yet
    assert "no valid student_ceiling" in capsys.readouterr().out
    os.makedirs(sup.prep_dir("f1"))
    vocab = pathlib.Path(sup.prep_dir("f1"), "vocab.json")
    vocab.write_text(json.dumps({"student_ceiling": 7.5}))
    row(iteration=2, reward_ceiling=7.5, reward_max=60.0, policy_maxp=0.4,
        reward_kept=0.9, tok_frames_per_decision=12.0, env_steps=1400)
    rows.emit()
    out = capsys.readouterr().out
    assert out.startswith("EVAL f1 s0 iter 2") and "ALERT" not in out
    assert "env 1400" in out
    assert "maxp 0.4" in out and "ceiling 7.5  kept 0.9  max 60.0" in out
    assert "frames/decision 12.0" in out
    # Test 2's rebuild's live reads (audit r1 B-N3, PB's r1 B-N4)
    assert "freshH None | prefix None  prefix_step1 None (control None)" \
        in out
    row(iteration=4, reward_ceiling=7.5, fresh_target_entropy=1.25,
        prefix=0.5, prefix_step1=0.125, prefix_step1_control=0.75)
    rows.emit()
    out = capsys.readouterr().out
    assert ("freshH 1.25 | prefix 0.5  prefix_step1 0.125 (control 0.75) "
            "| ceiling 7.5") in out
    row(iteration=3, reward_ceiling=1.0, reward_max=60.1)
    rows.emit()
    out = capsys.readouterr().out
    assert "ALERT: f1 s0 row 3: reward_ceiling 1.0" in out
    assert "ALERT: f1 s0 row 3: reward_max 60.1 > 8 x the ceiling" in out
    # read once: constant for the whole run
    vocab.write_text(json.dumps({"student_ceiling": 1.0}))
    row(iteration=4, reward_ceiling=7.5)
    rows.emit()
    assert "ALERT" not in capsys.readouterr().out


def test_f0s_rows_check_against_the_sign_clip(sup, tmp_path, capsys):
    """F0 has no prep: its ceiling is the sign clip, 1, from its first row
    (a vocab.json in a folder of its name is never read)."""
    pathlib.Path(sup.prep_dir("f0")).mkdir(parents=True)
    pathlib.Path(sup.prep_dir("f0"), "vocab.json").write_text(json.dumps(
        {"student_ceiling": 9.0}))
    jsonl = tmp_path / "a.jsonl"
    rows = sup.Rows(str(jsonl), "f0", 0)

    def row(**r):
        with open(jsonl, "a") as f:
            f.write(json.dumps({"elapsed_s": 1.0, **r}) + NL)

    row(iteration=5, reward_ceiling=1.0, reward_max=8.0)
    rows.emit()
    out = capsys.readouterr().out
    assert out.startswith("EVAL f0 s0 iter 5") and "ALERT" not in out
    row(iteration=10, reward_ceiling=9.0)
    rows.emit()
    assert ("ALERT: f0 s0 row 10: reward_ceiling 9.0, the run's constant "
            "ceiling is 1.0000") in capsys.readouterr().out
    row(iteration=15, reward_ceiling=1.0, reward_max=8.01)
    rows.emit()
    assert "ALERT: f0 s0 row 15: reward_max 8.01 > 8 x the ceiling 1.0000" \
        in capsys.readouterr().out


def test_rows_alert_on_low_host_commit(sup, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sup, "commit_free_gb", lambda: 3.2)
    os.makedirs(sup.prep_dir("f1"))
    pathlib.Path(sup.prep_dir("f1"), "vocab.json").write_text(
        json.dumps({"student_ceiling": 7.5}))
    jsonl = tmp_path / "a.jsonl"
    rows = sup.Rows(str(jsonl), "f1", 0)
    jsonl.write_text(json.dumps({"iteration": 20, "reward_ceiling": 7.5,
                                 "elapsed_s": 1.0}) + NL)
    rows.emit()
    out = capsys.readouterr().out
    assert "ALERT: f1 s0 row 20: host commit free 3.2 GB < 4 GB" in out


@pytest.mark.parametrize("text, want", [
    ('{"student_ceiling": 7.5}', 7.5),
    ('{"student_ceiling": 8}', 8.0),
    ('{"student_ceiling": 0}', None),
    ('{"student_ceiling": -1.0}', None),
    ('{"student_ceiling": true}', None),
    ('{"student_ceiling": "7.5"}', None),
    ('{"student_ceiling": NaN}', None),
    ('{"student_ceiling": Infinity}', None),
    ('{}', None), ('[7.5]', None), ('{"student_ceil', None), (None, None)])
def test_prep_ceiling(sup, text, want):
    if text is not None:
        os.makedirs(sup.prep_dir("f1"))
        pathlib.Path(sup.prep_dir("f1"), "vocab.json").write_text(text)
    assert sup.prep_ceiling("f1") == want
    assert sup.run_ceiling("f1") == want



# --------------------------------------------------------------------------
# Test B2 (Kai, 2026-10-29): the same chain over three generations t0-t2
# --------------------------------------------------------------------------
TWO = ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-"
       "tokens-two")
T = [f"{TWO}-t{k}" for k in range(3)]
TV = [f"t{k}" for k in range(3)]
T_DATE = "2026-10-29"
KEPT = 0.997 ** 4                 # the token arms' 0.997 per frame


@pytest.fixture
def tsup(sup):
    """`sup` bound to Test B2's lineage, as main() binds it for t0-t2."""
    sup.configure("t")
    return sup


LINEAGE_NAMES = ("LINEAGE", "GENERATIONS", "VARIANTS", "RUNS", "NAMES",
                 "ITERATIONS", "NOMINAL_ITERATIONS", "DISCOUNTS",
                 "DISCOUNT_RULE", "SEEDS", "PREP_PREFIX")


def test_test_b2s_lineage_and_test_bs_unchanged(real_sup):
    sup = real_sup
    assert set(sup.LINEAGES) == {"f", "t", "r", "n", "a", "w", "m", "q", "p"}
    assert sup.TOTAL_ENV_STEPS == 100000
    loaded = {k: getattr(sup, k) for k in LINEAGE_NAMES}
    assert loaded["LINEAGE"] == "f"         # the module loads as Test B's
    sup.configure("t")
    assert (sup.LINEAGE, sup.GENERATIONS) == ("t", 3)
    assert sup.VARIANTS == dict(zip(TV, T))
    assert sup.RUNS == {v: f"mspacman-two-{v}" for v in TV}
    assert sup.NAMES == {"t0": "T0", "t1": "T1", "t2": "T2"}
    assert sup.SEEDS == {0: T_DATE} and sup.PREP_PREFIX == "two-prep"
    assert (sup.ITERATIONS, sup.NOMINAL_ITERATIONS) == (100, 50)
    assert sup.DISCOUNTS == {v: KEPT for v in TV}
    assert sup.DISCOUNT_RULE == "keep"
    assert [sup.target(v) for v in TV] == [33333, 66667, 100000]
    with pytest.raises(SystemExit, match="f0 is not of the bound lineage"):
        sup.target("f0")
    for k, v in enumerate(TV):
        spec = ablation.ARMS[sup.VARIANTS[v]]
        assert spec["_env_stop"] is True and spec["OWN_PLAY_LOG"] is True
        assert spec["_nominal_iterations"] == sup.NOMINAL_ITERATIONS
        assert spec["DISCOUNT"] == sup.DISCOUNTS[v] == KEPT
        assert distill.discount_rule(spec) == sup.DISCOUNT_RULE
        assert sup.gen(v) == k and sup.is_student(v) == (k > 0)
        assert sup.master_of(v) == (TV[k - 1] if k else None)
        if k:
            assert spec["_master_arm"] == T[k - 1]
            assert sup.master_offline(v) == distill.master_offline_steps(
                ablation.ARMS[T[k - 1]]) == (20000 if k > 1 else 0)
    # Test B's own names and numbers come back exactly
    sup.configure("f")
    assert {k: getattr(sup, k) for k in LINEAGE_NAMES} == loaded
    assert [sup.target(v) for v in V] == [20000 * (k + 1) for k in range(5)]
    for v in V:
        assert distill.discount_rule(
            ablation.ARMS[sup.VARIANTS[v]]) == sup.DISCOUNT_RULE == "halve"


def test_main_binds_the_variants_lineage(sup, monkeypatch, capsys):
    monkeypatch.setattr(sup, "other_supervisors", lambda v: [])
    monkeypatch.setattr(sup, "run_done", lambda v, s: True)
    assert sup.main(["t1", "0"]) == 0
    assert sup.LINEAGE == "t" and sup.VARIANTS == dict(zip(TV, T))
    assert "RUN COMPLETE t1 s0" in capsys.readouterr().out
    assert sup.main(["f2", "0"]) == 0
    assert sup.LINEAGE == "f" and sup.VARIANTS == dict(zip(V, F))


def test_test_b2s_stops_fill_its_100k(tsup):
    sup = tsup
    assert sup.stop_env_steps("t0", 0) == 33333
    _finish(sup, "t0", final=48, env=33600, evals=False, files=False)
    assert sup.stop_env_steps("t1", 0) == 66667 - 33600
    _finish(sup, "t1", final=47, env=33200, evals=False, files=False)
    assert sup.stop_env_steps("t2", 0) == 100000 - 66800


def test_a_spent_test_b2_budget_names_its_generations(tsup):
    _finish(tsup, "t0", final=48, env=60000, evals=False, files=False)
    _finish(tsup, "t1", final=47, env=40000, evals=False, files=False)
    with pytest.raises(SystemExit, match=(
            r"STOP NOT POSITIVE: T2's --stop-env-steps would be 0 = 100000 "
            r"- 100000 \(the final env steps of T0, T1: \[60000, 40000\]\)")):
        tsup.stop_env_steps("t2", 0)


def test_test_b2s_command_is_test_bs_at_its_cap(tsup):
    sup = tsup
    fsup = _load(SUPERVISORS / "supervise_testb.py", "supervise_testb_f")
    for k, v in enumerate(TV):
        f = V[k]
        want = [x.replace(F[k], sup.VARIANTS[v])
                .replace(fsup.prep_rel(f), sup.prep_rel(v))
                for x in fsup.command(f, 0, 123)]
        want[want.index("--iterations") + 1] = "100"
        assert sup.command(v, 0, 123) == want, v
    c0 = sup.command("t0", 0)
    assert c0[c0.index("--stop-env-steps") + 1] == "33333"
    assert c0[c0.index("--arms") + 1] == T[0] and "--token-vocab" not in c0
    prep = f"benchmarks/results/two-prep-t1-s0-{T_DATE}"
    c1 = sup.command("t1", 0, 123)
    assert c1[c1.index("--token-vocab") + 1] == prep + "/vocab.json"
    assert c1[c1.index("--distill-set") + 1] == prep + "/set.pkl"


def test_test_b2s_paths_and_chain(tsup):
    sup = tsup
    for k, v in enumerate(TV):
        rel, base, jsonl, ckpt, deaths = sup.paths(v, 0)
        assert rel == f"benchmarks/results/mspacman-two-{v}-s0-{T_DATE}"
        assert ckpt.endswith(f"resume-mspacmannoframeskip-skip4-{T[k]}"
                             f"-seed0.ckpt")
        assert sup.own_play_rel(v, 0) == (
            f"{rel}/own-play-mspacmannoframeskip-skip4-{T[k]}-seed0")
        assert sup.prep_rel(v) == (
            f"benchmarks/results/two-prep-{v}-s0-{T_DATE}")
    assert sup.predecessor("t0", 0) is None
    name, marker, probes, _log, script, pvar, _flag = sup.predecessor("t2",
                                                                     0)
    assert (name, script, pvar) == ("T1", "supervise_testb.py", "t1")
    assert marker == sup.paths("t1", 0)[3]
    assert [os.path.basename(p) for p, _r in probes] == [
        f"probe_exhaustive_eval-mspacman-two-t1-s0-{T_DATE}{s}.jsonl"
        for s in ("-acting", "", "-offline-acting", "-offline")]
    _finish(sup, "t0", final=48, env=33600)
    _finish(sup, "t1", final=47, env=33200)
    assert sup.pred_missing("t2", 0) == []
    assert sup.master_final("t2", 0) == (47, sup.snapshot_path("t1", 0, 47))


@pytest.mark.parametrize("k", range(3))
def test_test_b2s_quick_probe_passes(tsup, k):
    if k:
        _master_ready(tsup, TV[k], final=48)
    r = _py(tsup.quick(TV[k], 0))
    assert r.returncode == 0, r.stderr[-2000:]


def test_each_test_b2_discount_assert_of_the_quick_probe_bites(tsup):
    """The constant discount and the "keep" rule, each alone: T0's
    DISCOUNT, its rule or its planned length; a master's or a student's
    _discount_rule; a prep rule that halves."""
    sup = tsup
    for k in (1, 2):
        _master_ready(sup, TV[k], final=48)
    a = "ablation.ARMS"
    breaks = {
        "t0": [f"{a}[{T[0]!r}]['DISCOUNT'] = 0.97",
               f"{a}[{T[0]!r}]['_discount_rule'] = 'halve'",
               f"{a}[{T[0]!r}]['_nominal_iterations'] = 30"],
        "t1": [f"{a}[{T[0]!r}]['_discount_rule'] = 'halve'",
               f"{a}[{T[1]!r}]['_discount_rule'] = 'halve'"],
        "t2": [f"{a}[{T[1]!r}]['_discount_rule'] = 'halve'"]}
    codes, controls = [], []
    for v, preludes in breaks.items():
        probe = sup.quick(v, 0)
        anchor = f"spec = ablation.ARMS[{sup.VARIANTS[v]!r}]; "
        assert probe.count(anchor) == 1
        for prelude in preludes:
            codes.append(probe.replace(anchor, prelude + "; " + anchor))
        controls.append(probe)
    got = _py_each(codes + controls)
    # a prep rule that halves: it patches distill itself, which EACH does
    # not restore, so it runs in a child of its own
    anchor = f"spec = ablation.ARMS[{T[1]!r}]; "
    halving = controls[1].replace(
        anchor, "distill.swap_discount = lambda g, r: "
        "distill.student_discount_rule(g); " + anchor)
    assert halving != controls[1]
    for code, g in zip(codes + [halving], got[:len(codes)]
                       + _py_each([halving])):
        # an assert of the probe's own, never the master checks' SystemExit
        # (audit r1b NIT-4)
        assert g.startswith("AssertionError"), (code[-300:], g)
    assert got[len(codes):] == ["ok"] * 3            # the controls


def test_test_b2s_prep_reads_t0s_final_snapshot_and_log(tsup, monkeypatch,
                                                        capsys):
    """run_prep under Test B2's lineage, the real master gate: T0's final
    snapshot and own-play log, Test B2's prep folder (audit r1b NIT-7)."""
    sup = tsup
    _finish(sup, "t0", final=48, env=33600)
    monkeypatch.setattr(sup, "wait_no_trainer", lambda *a: None)
    monkeypatch.setattr(sup, "manifest_check", lambda: (True, ""))
    monkeypatch.setattr(sup.time, "sleep", lambda s: None)
    runs = []
    _popen(monkeypatch, sup, [(0, lambda: _write_prep(sup, "t1", k=48))],
           runs)
    assert sup.run_prep("t1", 0, {}) is True
    out = capsys.readouterr().out
    want = ("MASTER T0: " + sup.snapshot_path("t0", 0, 48)
            + " (iteration 48; deployed mean of its last two rows 700.00")
    assert want in out
    assert "PREP done (attempt 1): " + sup.prep_rel("t1") in out
    args, _k = runs[0]
    assert args == [sup.PY, "experiments/distill.py", "--master",
                    sup.snapshot_path("t0", 0, 48), "--master-iterations",
                    "48", "--out",
                    f"benchmarks/results/two-prep-t1-s0-{T_DATE}", "--seed",
                    "0", "--from-log", sup.own_play_rel("t0", 0)]
    pick = json.loads(pathlib.Path(sup.prep_dir("t1"), "master.json")
                      .read_text())
    assert pick["which"] == "T0" and pick["iterations"] == 48


def test_test_b2s_prep_must_keep_the_discount(tsup):
    sup = tsup
    _finish(sup, "t0", final=48, evals=False, files=False)
    _write_prep(sup, "t1", k=48)
    assert sup.set_intact("t1", 0) == (True, "")
    halved = distill.student_discount_rule(KEPT)
    _write_prep(sup, "t1", k=48, student_discount=halved)
    assert sup.set_intact("t1", 0) == (False, (
        f"vocab.json's student_discount is {halved!r}, not T1's {KEPT}"))
    for rule, shown in (("halve", "'halve'"), (GONE, "'halve'"),
                        (None, "None")):
        _write_prep(sup, "t1", k=48, discount_rule=rule)
        assert sup.set_intact("t1", 0) == (False, (
            f"vocab.json's discount_rule is {shown}, not the lineage's "
            f"'keep'"))



# --------------------------------------------------------------------------
# Test C (Kai, 2026-10-29): r0 (BBF-style resets), then n0 (the control)
# --------------------------------------------------------------------------
RESET = ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-"
         "tokens-reset")
LONG = ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-"
        "tokens-long")
C_DATE = "2026-10-30"


@pytest.fixture
def nsup(sup):
    """`sup` bound to Test C's control lineage, as main() binds it for
    n0."""
    sup.configure("n")
    return sup


def test_test_cs_lineages(real_sup):
    sup = real_sup
    loaded = {k: getattr(sup, k) for k in LINEAGE_NAMES}
    sup.configure("r")
    assert (sup.LINEAGE, sup.GENERATIONS) == ("r", 1)
    assert sup.VARIANTS == {"r0": RESET + "-r0"}
    assert sup.RUNS == {"r0": "mspacman-reset-r0"}
    assert sup.NAMES == {"r0": "R0"}
    assert sup.SEEDS == {0: C_DATE} and sup.PREP_PREFIX is None
    assert (sup.ITERATIONS, sup.NOMINAL_ITERATIONS) == (300, 150)
    assert sup.DISCOUNTS == {"r0": KEPT} and sup.DISCOUNT_RULE == "keep"
    assert sup.target("r0") == 100000
    assert sup.upstream("r0") is None and not sup.is_student("r0")
    assert sup.run_name("r0", 0) == f"mspacman-reset-r0-s0-{C_DATE}"
    sup.configure("n")
    # the "after" run's names are bound too: n0's wait reads R0's files
    assert sup.VARIANTS == {"n0": LONG + "-n0", "r0": RESET + "-r0"}
    assert sup.RUNS == {"n0": "mspacman-long-n0", "r0": "mspacman-reset-r0"}
    assert sup.NAMES == {"n0": "N0", "r0": "R0"}
    assert sup.DISCOUNTS == {"n0": KEPT}
    assert sup.target("n0") == 100000
    with pytest.raises(SystemExit, match="r0 is not of the bound lineage"):
        sup.target("r0")
    assert sup.upstream("n0") == "r0" and sup.master_of("n0") is None
    assert not sup.is_student("n0")
    assert sup.run_name("r0", 0) == f"mspacman-reset-r0-s0-{C_DATE}"
    assert sup.run_name("n0", 0) == f"mspacman-long-n0-s0-{C_DATE}"
    for arm in (RESET + "-r0", LONG + "-n0"):
        spec = ablation.ARMS[arm]
        assert spec["_env_stop"] is True and spec["OWN_PLAY_LOG"] is True
        assert spec["_nominal_iterations"] == 150
        assert spec["DISCOUNT"] == KEPT and "_student" not in spec
        assert spec["DORMANT_DIAGNOSTIC"] is True
    assert {k: ablation.ARMS[RESET + "-r0"][k]
            for k in sup.RESET_KNOBS} == sup.RESET_KNOBS
    assert sup.LINEAGES["r"]["resets"] is sup.RESET_KNOBS
    assert sup.LINEAGES["n"]["resets"] == {"RESET_EVERY_STEPS": 0}
    # Test B's and Test B2's names and numbers come back exactly
    sup.configure("f")
    assert {k: getattr(sup, k) for k in LINEAGE_NAMES} == loaded
    for v in V:
        assert sup.upstream(v) == sup.master_of(v)
    sup.configure("t")
    assert sup.VARIANTS == dict(zip(TV, T))
    for v in TV:
        assert sup.upstream(v) == sup.master_of(v)


def test_test_cs_stops_and_commands_are_test_bs_recipe(sup):
    fsup = _load(SUPERVISORS / "supervise_testb.py", "supervise_testb_f")
    for letter, arm in (("r", RESET + "-r0"), ("n", LONG + "-n0")):
        sup.configure(letter)
        v = letter + "0"
        assert sup.stop_env_steps(v, 0) == 100000
        want = [x.replace(F[0], arm) for x in fsup.command("f0", 0, 100000)]
        want[want.index("--iterations") + 1] = "300"
        c = sup.command(v, 0)
        assert c == want, v
        assert c[c.index("--arms") + 1] == arm
        assert "--token-vocab" not in c and "--distill-set" not in c
    # n0's stop is its own 100,000, never less R0's
    _finish(sup, "r0", final=140, env=100500, evals=False, files=False)
    assert sup.stop_env_steps("n0", 0) == 100000


def test_n0_waits_for_r0_complete_and_evaluated(nsup):
    sup = nsup
    name, marker, probes, suplog, script, pvar, flag = sup.predecessor("n0",
                                                                      0)
    base = sup.paths("r0", 0)[1]
    assert (name, script, pvar) == ("R0", "supervise_testb.py", "r0")
    assert marker == sup.paths("r0", 0)[3]
    assert suplog == os.path.join(base, "supervisor.log")
    assert flag == os.path.join(base, "ABANDONED")
    assert [os.path.basename(p) for p, _r in probes] == [
        f"probe_exhaustive_eval-mspacman-reset-r0-s0-{C_DATE}{s}.jsonl"
        for s in ("-acting", "")]
    assert sup.pred_missing("n0", 0) == [
        "completion marker (no checkpoint)",
        "in-run rows (its ablation jsonl holds none)"]
    rows = _finish(sup, "r0", final=143, env=100500, evals=False,
                   files=False)
    assert rows[-1] == 143 and rows[0] == 5
    missing = sup.pred_missing("n0", 0)
    assert len(missing) == 2 and all(
        f"mspacman-reset-r0-s0-{C_DATE}" in m for m in missing), missing
    _rows(_probe(sup, "r0", "-acting"), rows)
    assert len(sup.pred_missing("n0", 0)) == 1
    _rows(_probe(sup, "r0", ""), rows)
    assert sup.pred_missing("n0", 0) == []
    # no master files (no prep): none can be missing
    assert sup.master_files_missing("n0", 0) == []
    assert sup.wait_external("n0", 0) == "ready"


def test_n0_stops_on_an_abandoned_r0(nsup, capsys):
    sup = nsup
    base = pathlib.Path(sup.paths("r0", 0)[1])
    base.mkdir(parents=True)
    (base / "ABANDONED").write_text("")
    assert sup.wait_external("n0", 0) == "failed"
    out = capsys.readouterr().out
    assert ("AFTER ABANDONED: R0's run folder holds an ABANDONED file -- N0 "
            "does not start (Kai's call)") in out
    assert "PREP FAILED" not in out and "MASTER" not in out


def test_n0_keeps_waiting_on_an_unfinished_r0(nsup, monkeypatch, capsys):
    sup = nsup

    def effect(n):
        if n == 2:
            _finish(sup, "r0", final=140, env=100500, files=False)

    monkeypatch.setattr(sup.time, "sleep", _bounded(effect))
    assert sup.wait_external("n0", 0) == "ready"
    out = capsys.readouterr().out
    assert ("WAITING for R0 before n0 s0 [0 h]: missing completion marker"
            in out)
    assert "R0 complete and evaluated; n0 s0 may start" in out
    # (nothing drives R0 here: no live supervise_testb.py r0)
    assert ("ALERT: R0 is unfinished and nothing drives it: no live "
            "supervise_testb.py r0 -- rerun its launcher") in out


def test_main_starts_r0_at_once_and_waits_n0_on_r0(sup, monkeypatch,
                                                   capsys):
    calls = []
    monkeypatch.setattr(sup, "other_supervisors", lambda v: [])
    monkeypatch.setattr(sup, "settle_stop",
                        lambda v, s: calls.append(("stop", v)) or None)
    monkeypatch.setattr(sup, "wait_external",
                        lambda v, s: calls.append(("wait", v)) or "ready")
    assert sup.main(["r0", "0"]) == 1          # (the stub stop: None)
    assert ("r0 s0 starts the lineage: no predecessor to wait for"
            in capsys.readouterr().out)
    assert calls == [("stop", "r0")]
    calls.clear()
    assert sup.main(["n0", "0"]) == 1
    assert calls == [("wait", "n0"), ("stop", "n0")]
    assert sup.LINEAGE == "n"
    calls.clear()
    monkeypatch.setattr(sup, "wait_external",
                        lambda v, s: calls.append(("wait", v)) or "failed")
    assert sup.main(["n0", "0"]) == 1
    assert calls == [("wait", "n0")]           # no stop, no launch


def test_test_cs_quick_probes_pass_and_their_reset_asserts_bite(sup):
    """r0's probe pins the reset knobs, n0's their absence; both the
    plasticity read and the module defaults off; Test B's and Test B2's
    probes carry none of it."""
    codes = []
    for letter in ("r", "n"):
        sup.configure(letter)
        codes.append(sup.quick(letter + "0", 0))
    for letter, v in (("f", "f0"), ("t", "t0")):
        sup.configure(letter)
        q = sup.quick(v, 0)
        assert "RESET_" not in q and "DORMANT" not in q
    a = "ablation.ARMS"
    r0, n0 = RESET + "-r0", LONG + "-n0"
    breaks = [(0, f"{a}[{r0!r}]['RESET_EVERY_STEPS'] = 20000"),
              (0, f"{a}[{r0!r}]['RESET_HOLD_STEPS'] = 0"),
              (0, f"{a}[{r0!r}]['RESET_SHRINK_FACTOR'] = 0.2"),
              (0, f"del {a}[{r0!r}]['RESET_REINIT_NETS']"),
              (0, f"{a}[{r0!r}]['DORMANT_DIAGNOSTIC'] = False"),
              (1, f"{a}[{n0!r}]['RESET_EVERY_STEPS'] = 40000"),
              (1, f"{a}[{n0!r}]['DORMANT_DIAGNOSTIC'] = False"),
              (1, f"{a}[{n0!r}]['_nominal_iterations'] = 50")]
    broken = []
    for i, prelude in breaks:
        anchor = f"spec = ablation.ARMS[{(r0, n0)[i]!r}]; "
        assert codes[i].count(anchor) == 1
        broken.append(codes[i].replace(anchor, prelude + "; " + anchor))
    got = _py_each(broken + codes)
    for code, g in zip(broken, got):
        assert g.startswith("AssertionError"), (code[-300:], g)
    assert got[len(broken):] == ["ok", "ok"]
    # a module default switched on: its own child (EACH restores no reset
    # flag)
    head = "import muzero, ablation, muzero_vtree, distill; "
    flipped = codes[1].replace(head, head + "muzero.RESET_EVERY_STEPS = "
                               "40000; ")
    assert flipped != codes[1]
    assert _py_each([flipped])[0].startswith("AssertionError")


def test_each_runs_name_is_dated_by_its_own_lineage(sup, monkeypatch):
    """Audit r1b NIT-3: r and n share a date, so the lineages test alone
    would pass with the bound lineage's date too."""
    monkeypatch.setitem(sup.LINEAGES["r"], "date", "2026-11-01")
    sup.configure("n")
    assert sup.SEEDS == {0: C_DATE}
    assert sup.run_name("r0", 0) == "mspacman-reset-r0-s0-2026-11-01"
    assert sup.run_name("n0", 0) == f"mspacman-long-n0-s0-{C_DATE}"
    assert sup.paths("r0", 0)[0].endswith("-2026-11-01")
    with pytest.raises(KeyError):
        sup.run_name("r0", 1)                 # SEEDS still rejects seed 1


def _row(it, env, **extra):
    return {"iteration": it, "env_steps": env, **extra}


def test_the_reset_schedule_is_checked_row_by_row(sup):
    """Audit r1b LOW-1: a reset after every 20th iteration whose row's env
    steps are <= 80,000 -- counted in the LATER rows -- and its 4 held
    weight ships (the syncs after iterations j + 1 .. j + 4)."""
    sup.configure("r")
    envs = {}
    seq = [(5, 3000, 0), (20, 13000, 0), (25, 16500, 1), (40, 26000, 1),
           (45, 29500, 2), (60, 40000, 2), (80, 53000, 3), (100, 66000, 4),
           (120, 80000, 5), (125, 83400, 6), (140, 93000, 6),
           (143, 100500, 6)]
    for it, env, n in seq:
        row = _row(it, env, resets=n, held_ships=4 * n)
        assert sup.reset_alert(row, envs) is None, it
        envs[it] = env
    # past the cutoff at row 120: no 7th reset (nor its ships)
    envs[120] = 80001
    assert sup.reset_alert(_row(125, 83400, resets=6, held_ships=24),
                           envs) == (
        "resets 6, the schedule's 5 (a reset after every 20th iteration "
        "while env steps <= 80000); held_ships 24, the schedule's 20 (4 "
        "held weight ships after each reset)")
    # a reset that never fired, a missing field, a bool
    assert "the schedule's 1" in sup.reset_alert(
        _row(25, 16500, resets=0, held_ships=4), {20: 13000})
    assert "resets None" in sup.reset_alert(_row(25, 16500, held_ships=4),
                                            {20: 13000})
    assert sup.reset_alert(_row(25, 16500, resets=True, held_ships=4),
                           {20: 13000})
    # a hold that shipped the learner's weights (no held ship)
    assert sup.reset_alert(_row(25, 16500, resets=1, held_ships=0),
                           {20: 13000}) == (
        "held_ships 0, the schedule's 4 (4 held weight ships after each "
        "reset)")
    # mid-hold (a resume's rows are every 5, but the count is per sync)
    assert sup.reset_alert(_row(22, 14000, resets=1, held_ships=2),
                           {20: 13000}) is None
    # the control and Test B / B2 are never checked
    for letter in ("n", "f", "t"):
        sup.configure(letter)
        assert sup.reset_alert(_row(25, 16500, resets=7), {20: 1}) is None


def test_reset_rows_print_and_alert(sup, monkeypatch, tmp_path, capsys):
    sup.configure("r")
    monkeypatch.setattr(sup, "run_ceiling", lambda v: 1.0)
    jsonl = tmp_path / "a.jsonl"
    rows = sup.Rows(str(jsonl), "r0", 0)
    plastic = {"resets": 0, "held_ships": 0, "dormant_025_all": 0.12,
               "dormant_10_all": 0.3, "conv1_weak_frac": 0.0,
               "reward_ceiling": 1.0}
    jsonl.write_text(json.dumps(_row(20, 13000, **plastic)) + NL)
    rows.prime()                       # a supervisor restart: row 20 known
    with open(jsonl, "a") as f:
        f.write(json.dumps(_row(25, 16500, **plastic)) + NL)
    rows.emit()
    out = capsys.readouterr().out
    assert ("| resets 0  held ships 0  dormant 0.12 / 0.3  conv1 weak 0.0 "
            "|") in out
    assert ("ALERT: r0 s0 row 25: resets 0, the schedule's 1 (a reset "
            "after every 20th iteration while env steps <= 80000); "
            "held_ships 0, the schedule's 4") in out
    # a row without the read prints as before
    sup.configure("f")
    rows = sup.Rows(str(tmp_path / "b.jsonl"), "f0", 0)
    (tmp_path / "b.jsonl").write_text(json.dumps(
        _row(5, 3000, reward_ceiling=1.0)) + NL)
    monkeypatch.setattr(sup, "run_ceiling", lambda v: 1.0)
    rows.emit()
    out = capsys.readouterr().out
    assert "EVAL f0 s0 iter 5" in out and "resets" not in out
    assert "ALERT" not in out


def test_the_trainers_reset_lines_are_checked_exactly(sup):
    """Audit r3b LOW-1: rows see every 5th iteration only; the RESET line
    shows the reset's own iteration, step and env steps."""
    sup.configure("r")
    assert sup.reset_line_alert(1, 20, 40000, 13000) is None
    assert sup.reset_line_alert(4, 80, 160000, 80000) is None
    assert sup.reset_line_alert(1, 21, 42000, 13700) == (
        "RESET 1 after iteration 21 (step 42000, 13700 env steps); the "
        "schedule's reset 1 lands after iteration 20 (step 40000) with env "
        "steps <= 80000")
    assert sup.reset_line_alert(2, 40, 80001, 26000)       # off step
    assert sup.reset_line_alert(6, 120, 240000, 80001)     # past the cutoff
    sup.configure("n")
    assert "has no resets" in sup.reset_line_alert(1, 20, 40000, 13000)


def test_reset_lines_are_echoed_as_the_log_grows(sup, tmp_path, capsys):
    sup.configure("r")
    log = tmp_path / "supv2_1.log"
    lines = sup.ResetLines("r0", 0)
    good = ("  [arm s0] RESET 1 after iter 20 (step 40000, 13000 env steps): "
            "representation+dynamics shrunk ...")
    bad = ("  [arm s0] RESET 2 after iter 41 (step 82000, 27000 env steps): "
           "...")
    log.write_text("iter 19 ..." + NL + good + NL + "half a li")
    lines.emit(str(log))
    out = capsys.readouterr().out
    assert out == ("RESET r0 s0 1 after iter 20 (step 40000, 13000 env "
                   "steps)" + NL)
    with open(log, "a") as f:
        f.write("ne" + NL + bad + NL)
    lines.emit(str(log))
    lines.emit(str(log))                       # nothing new: nothing printed
    out = capsys.readouterr().out
    assert out.count("RESET r0 s0 2 after iter 41") == 1
    assert ("ALERT: r0 s0: RESET 2 after iteration 41 (step 82000, 27000 "
            "env steps); the schedule's reset 2 lands after iteration 40 "
            "(step 80000)") in out
    assert "RESET r0 s0 1 " not in out          # never re-read
    lines.emit(str(tmp_path / "missing.log"))   # a missing log: nothing
    # the control and Test B: inert
    for letter, v in (("n", "n0"), ("f", "f0")):
        sup.configure(letter)
        sup.ResetLines(v, 0).emit(str(log))
    assert capsys.readouterr().out == ""


# --------------------------------------------------------------------------
# Test C2 (Kai, 2026-11-02): a0 (C plus BBF's n-step and discount ramps),
# one run alone
# --------------------------------------------------------------------------
RAMP = ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-"
        "tokens-ramp")
C2_DATE = "2026-11-02"
RAMP_KNOBS = {"RAMP_STEPS": 10000, "RAMP_N_START": 10,
              "RAMP_GAMMA_START": 0.97, "RAMP_N_END": 5,
              "RAMP_GAMMA_END": 0.997 ** 4}
# the pre-registration's schedule: steps since the reset, n, gamma (3
# places)
RAMP_TABLE = [(0, 10, 0.970), (2000, 9, 0.975), (4000, 8, 0.979),
              (6000, 7, 0.983), (8000, 6, 0.986), (10000, 5, 0.988)]


@pytest.fixture
def asup(sup):
    """`sup` bound to Test C2's lineage, as main() binds it for a0."""
    sup.configure("a")
    return sup


@pytest.fixture
def ramp_knobs(monkeypatch):
    """C2's ramp knobs on muzero (restored after), so muzero.ramp_values
    -- the trainer's own schedule -- builds the expected rows."""
    for k, v in RAMP_KNOBS.items():
        monkeypatch.setattr(muzero, k, v)


def test_test_c2s_lineage(real_sup):
    sup = real_sup
    loaded = {k: getattr(sup, k) for k in LINEAGE_NAMES}
    sup.configure("a")
    assert (sup.LINEAGE, sup.GENERATIONS) == ("a", 1)
    assert sup.VARIANTS == {"a0": RAMP + "-a0"}
    assert sup.RUNS == {"a0": "mspacman-ramp-a0"}
    assert sup.NAMES == {"a0": "A0"}
    assert sup.SEEDS == {0: C2_DATE} and sup.PREP_PREFIX is None
    assert (sup.ITERATIONS, sup.NOMINAL_ITERATIONS) == (300, 150)
    assert sup.DISCOUNTS == {"a0": KEPT} and sup.DISCOUNT_RULE == "keep"
    assert sup.target("a0") == 100000
    # alone: no "after" run, no master, no prep
    assert "after" not in sup.LINEAGES["a"]
    assert sup.upstream("a0") is None and sup.master_of("a0") is None
    assert not sup.is_student("a0") and sup.predecessor("a0", 0) is None
    assert sup.run_name("a0", 0) == f"mspacman-ramp-a0-s0-{C2_DATE}"
    assert sup.paths("a0", 0)[0] == (
        f"benchmarks/results/mspacman-ramp-a0-s0-{C2_DATE}")
    assert ablation.RAMP_ARM == sup.RAMP_ARM == RAMP
    spec = ablation.ARMS[RAMP + "-a0"]
    assert spec["_env_stop"] is True and spec["OWN_PLAY_LOG"] is True
    assert spec["_nominal_iterations"] == 150
    assert spec["DISCOUNT"] == KEPT and "_student" not in spec
    assert spec["DORMANT_DIAGNOSTIC"] is True
    # C's resets unchanged, plus the ramps: quick() pins both
    assert sup.LINEAGES["a"]["resets"] is sup.RESET_KNOBS
    assert {k: spec[k] for k in sup.RESET_KNOBS} == sup.RESET_KNOBS
    assert sup.LINEAGES["a"]["ramps"] is sup.RAMP_KNOBS
    assert sup.RAMP_KNOBS == RAMP_KNOBS
    assert {k: spec[k] for k in RAMP_KNOBS} == RAMP_KNOBS
    # the end values are the arm's own TD_STEPS and DISCOUNT
    assert (spec["TD_STEPS"], spec["DISCOUNT"]) == (5, 0.997 ** 4)
    # every other lineage: no ramps (RAMP_OFF: muzero's defaults, the
    # trainer's own off set), none in its arms
    assert sup.RAMP_OFF == ablation.RAMP_OFF == {
        k: getattr(muzero, k) for k in ablation.RAMP_OFF}
    for letter, lin in sup.LINEAGES.items():
        if letter != "a":
            assert "ramps" not in lin, letter
            # (E3's "m" runs generations 1-2 only: "first")
            for k in range(lin.get("first", 0), lin["generations"]):
                arm = ablation.ARMS[f"{lin['arm']}-{letter}{k}"]
                assert not any(x.startswith("RAMP_") for x in arm), letter
    # the tolerance sits far below one step of the ramp (~1e-6 in gamma)
    assert sup.RAMP_GAMMA_TOL == 1e-12
    # Test B's, Test B2's and Test C's names and numbers come back
    sup.configure("f")
    assert {k: getattr(sup, k) for k in LINEAGE_NAMES} == loaded
    sup.configure("r")
    assert sup.VARIANTS == {"r0": RESET + "-r0"} and sup.SEEDS == {0: C_DATE}
    sup.configure("n")
    assert sup.upstream("n0") == "r0"


def test_c2s_stop_and_command_are_cs_recipe(asup):
    """a0's command is r0's (Test C's C) with the arm moved, its stop its
    own 100,000."""
    sup = asup
    rsup = _load(SUPERVISORS / "supervise_testb.py", "supervise_testb_r")
    rsup.configure("r")
    assert sup.stop_env_steps("a0", 0) == 100000
    want = [x.replace(RESET + "-r0", RAMP + "-a0")
            for x in rsup.command("r0", 0, 100000)]
    c = sup.command("a0", 0)
    assert c == want and c != rsup.command("r0", 0, 100000)
    assert c[c.index("--arms") + 1] == RAMP + "-a0"
    for flag, val in (("--iterations", "300"), ("--train-steps", "2000"),
                      ("--eval-every", "5"), ("--checkpoint-every", "2"),
                      ("--stop-env-steps", "100000"), ("--seed", "0")):
        assert c[c.index(flag) + 1] == val, flag
    assert "--token-vocab" not in c and "--distill-set" not in c


def test_main_starts_a0_at_once_and_alone(sup, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(sup, "other_supervisors", lambda v: [])
    monkeypatch.setattr(sup, "settle_stop",
                        lambda v, s: calls.append(("stop", v)) or None)
    monkeypatch.setattr(sup, "wait_external",
                        lambda v, s: calls.append(("wait", v)) or "ready")
    assert sup.main(["a0", "0"]) == 1          # (the stub stop: None)
    assert ("a0 s0 starts the lineage: no predecessor to wait for"
            in capsys.readouterr().out)
    assert calls == [("stop", "a0")] and sup.LINEAGE == "a"
    monkeypatch.setattr(sup, "run_done", lambda v, s: True)
    assert sup.main(["a0", "0"]) == 0
    assert "RUN COMPLETE a0 s0" in capsys.readouterr().out


def test_c2s_quick_probe_pins_the_ramps_and_every_other_lineage_off(sup):
    """a0's probe pins the five ramp knobs (and C's reset knobs and
    plasticity read), every other lineage's pins RAMP_OFF; each break of
    one alone fails its probe, the controls pass."""
    probes, arms = {}, {"a": RAMP + "-a0", "r": RESET + "-r0",
                        "n": LONG + "-n0", "f": F[0], "t": T[0]}
    for letter in arms:
        sup.configure(letter)
        probes[letter] = sup.quick(letter + "0", 0)
    for k, v in RAMP_KNOBS.items():
        assert f"assert spec.get({k!r}, muzero.{k}) == {v!r}; " in probes["a"]
    for letter in "rnft":
        q = probes[letter]
        for k, v in sup.RAMP_OFF.items():
            assert f"assert spec.get({k!r}, muzero.{k}) == {v!r}; " in q
        assert "hasattr(ablation, 'ramp_gate')" not in q
    assert "hasattr(ablation, 'ramp_gate')" in probes["a"]
    assert "RESET_" not in probes["f"] and "RESET_" not in probes["t"]
    a, a0 = "ablation.ARMS", RAMP + "-a0"
    breaks = [("a", f"{a}[{a0!r}]['RAMP_STEPS'] = 20000"),
              ("a", f"{a}[{a0!r}]['RAMP_N_START'] = 3"),
              ("a", f"{a}[{a0!r}]['RAMP_GAMMA_START'] = 0.9"),
              ("a", f"{a}[{a0!r}]['RAMP_N_END'] = 3"),
              ("a", f"del {a}[{a0!r}]['RAMP_GAMMA_END']"),
              ("a", f"{a}[{a0!r}]['TD_STEPS'] = 3"),
              ("a", f"{a}[{a0!r}]['DISCOUNT'] = 0.97"),
              ("a", f"{a}[{a0!r}]['RESET_EVERY_STEPS'] = 20000"),
              ("a", f"{a}[{a0!r}]['RESET_HOLD_STEPS'] = 0"),
              ("a", f"{a}[{a0!r}]['DORMANT_DIAGNOSTIC'] = False"),
              ("a", f"{a}[{a0!r}]['_nominal_iterations'] = 50")]
    for letter in "rnft":
        breaks += [(letter, f"{a}[{arms[letter]!r}]['RAMP_STEPS'] = 10000"),
                   (letter, f"{a}[{arms[letter]!r}]['RAMP_GAMMA_END'] = "
                            f"{0.997 ** 4!r}")]
    broken = []
    for letter, prelude in breaks:
        anchor = f"spec = ablation.ARMS[{arms[letter]!r}]; "
        assert probes[letter].count(anchor) == 1
        broken.append(probes[letter].replace(anchor, prelude + "; " + anchor))
    got = _py_each(broken + list(probes.values()))
    for code, g in zip(broken, got):
        assert g.startswith("AssertionError"), (code[-300:], g)
    assert got[len(broken):] == ["ok"] * len(probes)        # the controls
    # a module default switched on, or the trainer's off set changed: each
    # in its own child (EACH restores no ramp flag)
    head = "import muzero, ablation, muzero_vtree, distill; "
    flips = []
    for letter, flip in (("a", "muzero.RAMP_STEPS = 10000; "),
                         ("f", "muzero.RAMP_N_END = 5; "),
                         ("t", "ablation.RAMP_OFF = {}; ")):
        flipped = probes[letter].replace(head, head + flip)
        assert flipped != probes[letter]
        flips.append(flipped)
    for g in [_py_each([f])[0] for f in flips]:
        assert g.startswith("AssertionError"), g


def test_the_stdlib_ramp_values_is_muzeros(real_sup, monkeypatch):
    """The supervisor's ramp_values (the standard library) against the
    trainer's muzero.ramp_values at every step of the window and past it,
    at the run's knobs and at the smoke's (a 50-step window): n and the end
    values exactly, gamma within a last-bit difference -- far inside
    RAMP_GAMMA_TOL, which sits far below one step's change; and the
    pre-registered table."""
    sup = real_sup
    worst, step_min = 0.0, 1.0
    for knobs, xs in ((RAMP_KNOBS, list(range(10002)) + [10 ** 5]),
                      ({**RAMP_KNOBS, "RAMP_STEPS": 50}, range(120))):
        for k, v in knobs.items():
            monkeypatch.setattr(muzero, k, v)
        last = None
        for x in xs:
            want, got = muzero.ramp_values(x), sup.ramp_values(x, knobs)
            assert type(got[0]) is int and type(got[1]) is float, x
            assert got[0] == want[0], x
            worst = max(worst, abs(got[1] - want[1]))
            if x >= knobs["RAMP_STEPS"]:
                assert got == want == (5, 0.997 ** 4), x
            elif last is not None and knobs is RAMP_KNOBS:
                step_min = min(step_min, got[1] - last)
            last = got[1]
    assert worst <= 1e-15 < sup.RAMP_GAMMA_TOL
    assert step_min > 1e-7 > 1000 * sup.RAMP_GAMMA_TOL
    for steps, n, gamma in RAMP_TABLE:
        got = sup.ramp_values(steps, RAMP_KNOBS)
        assert got[0] == n and round(got[1], 3) == gamma, steps
    # onp.round's half to even is Python's round on a float
    assert round(8.5) == 8 and round(9.5) == 10
    assert "round(math.exp(" in inspect.getsource(sup.ramp_values)


def _c2_rows(final=134, cut=100, env_per_it=670):
    """A ramp run's rows as the trainer writes them -- the module's ramp
    knobs (the `ramp_knobs` fixture) through muzero.ramp_values, the
    trainer's own schedule: one every 5 iterations and the final one, each
    with its ramp at the row (before the row's reset), the trace since the
    previous row (at each iteration's first step) and the ramp windows
    since it (each gradient step as ablation.note_ramp_window counts it:
    "in" while its steps since the reset are below RAMP_STEPS; every step
    scored, made-up finite means); a reset after every 20th iteration up
    to `cut` (the rows' env steps keep it on the schedule: <= 80,000
    through `cut`). JSON round-tripped."""
    rows, origin, trace = [], 0, []
    wins = {"in": 0, "out": 0}
    for i in range(1, final + 1):
        c = (i - 1) * 2000 - origin
        trace.append([i, c, *muzero.ramp_values(c)])
        for s in range((i - 1) * 2000, i * 2000):
            wins["in" if s - origin < muzero.RAMP_STEPS else "out"] += 1
        if i % 5 == 0 or i == final:
            c = i * 2000 - origin
            n, g = muzero.ramp_values(c)
            rows.append(json.loads(json.dumps({
                "iteration": i, "env_steps": env_per_it * i,
                "ramp": {"n": n, "gamma": g, "cycle_steps": c},
                "ramp_trace": trace,
                "ramp_windows": {k: {
                    "steps": m, "scored": m,
                    "value_loss": 1.234567 if m else None,
                    "target_mean": 4.5 if m else None}
                    for k, m in wins.items()}})))
            trace, wins = [], {"in": 0, "out": 0}
        if i % 20 == 0 and i <= cut and i != final:
            origin = i * 2000
    return rows


def _stream_rows(final):
    """_c2_rows with the fields Rows' other checks read (the ceiling, the
    resets and held ships on the schedule, the plasticity read)."""
    rows = _c2_rows(final=final)
    for r in rows:
        n = min(5, (r["iteration"] - 1) // 20)
        r.update({"dormant_025_all": 0.0, "dormant_10_all": 0.0003,
                  "conv1_weak_frac": 0.0, "reward_ceiling": 1.0},
                 resets=n, held_ships=4 * n)
    return rows


def _checked(sup, rows):
    """Each row through ramp_alert against the rows before it, as Rows
    feeds it; the alerts by iteration (None: on schedule)."""
    envs, out = {}, {}
    for r in rows:
        out[r["iteration"]] = sup.ramp_alert(r, dict(envs))
        envs[r["iteration"]] = r["env_steps"]
    return out


def test_the_ramp_is_checked_row_by_row(asup, ramp_knobs):
    """A whole run on the schedule passes every row: the first cycle from
    the start, a restart after each reset (rows 20, 40, ..., 100), the end
    values past the cutoff (row 120's env steps > 80,000: no reset)."""
    sup = asup
    rows = _c2_rows()
    by = {r["iteration"]: r for r in rows}
    # the generator is the pre-registration's: n 10 -> 5 over 5 iterations
    for i in (5, 25, 45, 105):
        assert [e[2] for e in by[i]["ramp_trace"]] == [10, 9, 8, 7, 6], i
        assert by[i]["ramp"] == {"n": 5, "gamma": 0.997 ** 4,
                                 "cycle_steps": 10000}, i
    assert by[20]["ramp"]["cycle_steps"] == 40000     # before its reset
    assert by[125]["env_steps"] > 80000 >= by[100]["env_steps"]
    assert [e[1] for e in by[125]["ramp_trace"]] == list(
        range(40000, 50000, 2000))                    # no reset after 120
    assert [e[0] for e in by[134]["ramp_trace"]] == list(range(131, 135))
    # the windows: the ramp's 10,000 steps at the rows that end one, none
    # anywhere else (the final row: 4 iterations, all out)
    for i, r in by.items():
        want = ((10000, 0) if i in (5, 25, 45, 65, 85, 105) else
                (0, 8000) if i == 134 else (0, 10000))
        w = r["ramp_windows"]
        assert (w["in"]["steps"], w["out"]["steps"]) == want, i
    assert set(_checked(sup, rows).values()) == {None}


def _alert(sup, rows, it, edit):
    """ramp_alert on row `it` after `edit` (a function of the row), the
    rows before it as context."""
    import copy
    envs = {r["iteration"]: r["env_steps"] for r in rows
            if r["iteration"] < it}
    row = copy.deepcopy(next(r for r in rows if r["iteration"] == it))
    edit(row)
    return sup.ramp_alert(row, envs)


def test_the_ramp_alerts_off_its_schedule(asup, ramp_knobs):
    sup = asup
    rows = _c2_rows()
    g22 = muzero.ramp_values(2000)[1]

    def entry(i, k, v):
        def edit(r):
            next(e for e in r["ramp_trace"] if e[0] == i)[k] = v
        return edit

    # a trace entry's n or gamma off the schedule
    msg = _alert(sup, rows, 25, entry(22, 2, 10))
    assert msg.startswith(f"ramp_trace iteration 22: n 10 gamma {g22!r} at "
                          f"2000 steps since the reset, the schedule's n 9 "
                          f"gamma 0.97"), msg
    assert ";" not in msg                       # nothing else is off
    assert "iteration 23: n 8 gamma" in _alert(
        sup, rows, 25, entry(23, 3, muzero.ramp_values(4000)[1] + 1e-6))
    # ... gamma within the tolerance is on it
    assert _alert(sup, rows, 25, entry(23, 3, muzero.ramp_values(4000)[1]
                                       + 1e-14)) is None
    # a step count off (an iteration's first step after the reset is 0)
    assert "iteration 21: 2000 steps since the reset, the schedule's 0" in (
        _alert(sup, rows, 25, entry(21, 1, 2000)))
    # the ramp NOT restarted at the reset after 20: the trace and the row
    # still count from the start
    def no_restart(r):
        for e in r["ramp_trace"]:
            e[1:] = [(e[0] - 1) * 2000, 5, 0.997 ** 4]
        r["ramp"]["cycle_steps"] = 50000

    msg = _alert(sup, rows, 25, no_restart)
    assert msg.startswith("the row's ramp: 50000 steps since the reset, the "
                          "schedule's 10000; ramp_trace iteration 21: 40000 "
                          "steps since the reset, the schedule's 0"), msg
    assert "(+2 more)" in msg
    # the row's own ramp: n, gamma, the steps
    for key, val in (("n", 6), ("gamma", 0.98), ("cycle_steps", 8000)):
        def edit(r, key=key, val=val):
            r["ramp"][key] = val
        assert _alert(sup, rows, 25, edit).startswith("the row's ramp: "), key
    # a gap, a repeat, a lost window (a resume that kept no entries), a
    # foreign one
    for edit, its in ((lambda r: r["ramp_trace"].pop(2), [21, 22, 24, 25]),
                      (lambda r: r["ramp_trace"].insert(
                          2, list(r["ramp_trace"][1])), [21, 22, 22, 23, 24,
                                                         25]),
                      (lambda r: r["ramp_trace"].__delitem__(slice(0, 3)),
                       [24, 25]),
                      (lambda r: r.__setitem__("ramp_trace", []), [])):
        msg = _alert(sup, rows, 25, edit)
        assert msg.startswith(f"ramp_trace iterations {its!r}, the "
                              f"iterations since the previous row 21..25 "
                              f"once each"), msg
    # past the cutoff (no reset after 120): a restart at 121 is off it
    def restart(r):
        for e in r["ramp_trace"]:
            e[1:] = [(e[0] - 121) * 2000, *muzero.ramp_values(
                (e[0] - 121) * 2000)]

    assert "iteration 121: 0 steps since the reset, the schedule's 40000" in (
        _alert(sup, rows, 125, restart))
    # ... and a reset the schedule wants (row 120 at <= 80,000) that the
    # ramp missed
    envs = {r["iteration"]: r["env_steps"] for r in rows
            if r["iteration"] < 125}
    envs[120] = 80000
    assert "the schedule's 10000" in sup.ramp_alert(
        next(r for r in rows if r["iteration"] == 125), envs)


def test_a_malformed_ramp_is_an_alert_never_an_exception(asup, ramp_knobs):
    sup = asup
    rows = _c2_rows(final=30)
    for edit in (lambda r: r.pop("ramp"), lambda r: r.pop("ramp_trace"),
                 lambda r: r.__setitem__("ramp", [5, 0.98, 10000]),
                 lambda r: r.__setitem__("ramp", None),
                 lambda r: r.__setitem__("ramp_trace", "x"),
                 lambda r: r["ramp_trace"].append("x"),
                 lambda r: r["ramp_trace"].append([26, 0, 10]),
                 lambda r: r["ramp_trace"][0].__setitem__(3, None),
                 lambda r: r["ramp_trace"][0].__setitem__(3, "0.97"),
                 lambda r: r["ramp_trace"][0].__setitem__(3, float("nan")),
                 lambda r: r["ramp_trace"][0].__setitem__(2, True),
                 lambda r: r["ramp_trace"][0].__setitem__(2, 10.0),
                 lambda r: r["ramp_trace"][0].__setitem__(1, 0.0),
                 lambda r: r["ramp_trace"][0].__setitem__(0, "21"),
                 lambda r: r["ramp"].__setitem__("n", "5"),
                 lambda r: r["ramp"].__setitem__("cycle_steps", None),
                 lambda r: r["ramp"].pop("gamma"),
                 # the windows (audit r1b L1)
                 lambda r: r.pop("ramp_windows"),
                 lambda r: r.__setitem__("ramp_windows", [1, 2]),
                 lambda r: r["ramp_windows"].pop("out"),
                 lambda r: r["ramp_windows"].__setitem__("in", None),
                 lambda r: r["ramp_windows"]["in"].pop("steps"),
                 lambda r: r["ramp_windows"]["in"].__setitem__("steps",
                                                               10000.0),
                 lambda r: r["ramp_windows"]["in"].__setitem__("scored",
                                                               True),
                 lambda r: r["ramp_windows"]["in"].__setitem__(
                     "value_loss", "1.2"),
                 lambda r: r["ramp_windows"]["in"].__setitem__(
                     "target_mean", float("inf"))):
        msg = _alert(sup, rows, 25, edit)
        assert isinstance(msg, str) and msg, msg
    # rows that are not in-run rows are not checked
    for it in (0, None, True, "25"):
        assert sup.ramp_alert({"iteration": it}, {}) is None


def test_a_resumes_rows_check_against_the_rows_before_them(asup,
                                                            ramp_knobs):
    """A kill between row 25 and its checkpoint: the resumed run re-writes
    row 25 (its trace from the checkpoint at 24 plus 25). The stale row 25
    is among the earlier rows, so only the rows before 25 count."""
    sup = asup
    rows = _c2_rows(final=40)
    envs = {r["iteration"]: r["env_steps"] for r in rows
            if r["iteration"] <= 25}
    r25 = next(r for r in rows if r["iteration"] == 25)
    assert sup.ramp_alert(r25, envs) is None
    envs[25] = r25["env_steps"] + 300           # the stale copy's
    assert sup.ramp_alert(r25, envs) is None
    # a stale reset row: the re-written row 20's env steps decide
    envs[20] = 80001
    assert "the schedule's 50000" in sup.ramp_alert(r25, envs)


def test_the_ramp_check_is_c2s_alone(sup, ramp_knobs):
    """Every other lineage: a row without a ramp is not checked, a row WITH
    one is an ALERT (its arm is not its lineage's); a0 needs one."""
    row = _c2_rows(final=5)[0]
    for letter in "rnft":
        sup.configure(letter)
        assert sup.ramp_alert(_row(25, 16500, resets=1), {20: 1}) is None
        assert sup.ramp_alert(row, {}) == (
            f"a ramp in the row, but lineage {letter!r} has no ramps")
    sup.configure("a")
    assert sup.ramp_alert(row, {}) is None
    assert sup.ramp_alert(_row(5, 3000), {}) == (
        "the row's ramp is None; ramp_trace is not a list of [iteration, "
        "steps since the reset, n, gamma] entries; ramp_windows None is not "
        "an in / out pair of windows")
    # the windows alone in another lineage's row: an ALERT too
    sup.configure("r")
    assert "has no ramps" in sup.ramp_alert(
        _row(5, 3000, ramp_windows=row["ramp_windows"]), {})


def test_a0s_resets_are_checked_as_r0s(sup):
    """The reset schedule's checks apply to a0 exactly as to r0: the rows
    (reset_alert), the trainer's RESET lines (reset_line_alert,
    ResetLines)."""
    seq = [(5, 3000, 0, 0), (20, 13000, 0, 0), (22, 14000, 1, 2),
           (25, 16500, 1, 4), (40, 26000, 1, 4), (45, 29500, 2, 8),
           (120, 80000, 5, 20), (125, 83400, 6, 24), (125, 83400, 5, 20),
           (143, 100500, 6, 24), (25, 16500, 0, 4), (25, 16500, True, 4),
           (25, 16500, 1, 0)]
    lines = [(1, 20, 40000, 13000), (4, 80, 160000, 80000),
             (1, 21, 42000, 13700), (2, 40, 80001, 26000),
             (6, 120, 240000, 80001)]
    got = {}
    for letter in ("r", "a"):
        sup.configure(letter)
        envs = {5: 3000, 20: 13000, 40: 26000, 60: 40000, 80: 53000,
                100: 66000, 120: 80000}
        got[letter] = (
            [sup.reset_alert(_row(it, env, resets=n, held_ships=h),
                             {j: e for j, e in envs.items() if j < it})
             for it, env, n, h in seq],
            [sup.reset_line_alert(*x) for x in lines])
    assert got["a"] == got["r"]
    rows, ls = got["a"]
    # on the schedule (row 120 at 80,000 env steps: its reset counts), off
    # it: a 6th reset missing at 125, none at 25, a bool, no held ship
    assert [x is None for x in rows] == [True] * 8 + [False, True, False,
                                                      False, False]
    assert ls[:2] == [None, None] and all(ls[2:])


def test_a0s_reset_lines_are_echoed_as_r0s(asup, tmp_path, capsys):
    sup = asup
    log = tmp_path / "supv2_1.log"
    log.write_text("  [arm s0] RESET 1 after iter 20 (step 40000, 13000 env "
                   "steps): ..." + NL + "  [arm s0] RESET 2 after iter 41 "
                   "(step 82000, 27000 env steps): ..." + NL)
    sup.ResetLines("a0", 0).emit(str(log))
    out = capsys.readouterr().out
    assert out.startswith("RESET a0 s0 1 after iter 20 (step 40000, 13000 "
                          "env steps)" + NL)
    assert ("ALERT: a0 s0: RESET 2 after iteration 41 (step 82000, 27000 "
            "env steps)") in out


def test_ramp_rows_print_and_alert(asup, ramp_knobs, monkeypatch, tmp_path,
                                   capsys):
    sup = asup
    monkeypatch.setattr(sup, "run_ceiling", lambda v: 1.0)
    plastic = {"dormant_025_all": 0.0, "dormant_10_all": 0.0003,
               "conv1_weak_frac": 0.0, "reward_ceiling": 1.0}
    rows = _c2_rows(final=30)
    for r in rows:
        n = r["iteration"] // 21
        r.update(plastic, resets=n, held_ships=4 * n)
    jsonl = tmp_path / "a.jsonl"
    jsonl.write_text("".join(json.dumps(r) + NL for r in rows[:4]))
    stream = sup.Rows(str(jsonl), "a0", 0)
    stream.prime()                     # a supervisor restart: rows 5-20
    with open(jsonl, "a") as f:
        f.write(json.dumps(rows[4]) + NL)
    stream.emit()
    out = capsys.readouterr().out
    assert ("| resets 1  held ships 4  dormant 0.0 / 0.0003  conv1 weak 0.0 "
            "| ramp n 5  gamma 0.98805  cycle 10000  trace n 10/9/8/7/6  "
            "in/out 10000/0 | elapsed") in out, out
    assert "ALERT" not in out
    bad = dict(rows[5])
    bad["ramp_trace"] = bad["ramp_trace"][1:]          # iteration 26 lost
    with open(jsonl, "a") as f:
        f.write(json.dumps(bad) + NL)
    stream.emit()
    out = capsys.readouterr().out
    assert "trace n 5/5/5/5  in/out 0/10000 |" in out
    assert ("ALERT: a0 s0 row 30: ramp_trace iterations [27, 28, 29, 30], "
            "the iterations since the previous row 26..30 once each") in out
    # Test C's rows print as before: no ramp
    sup.configure("r")
    r = _row(5, 3000, resets=0, held_ships=0, **plastic)
    assert sup.ramp_text(r) == ""
    assert sup.ramp_text({"ramp_trace": [[1, 0, 10, 0.97], "x"]}) == (
        " | ramp n None  gamma None  cycle None  trace n 10/?  in/out ?/?")


def test_the_ramp_windows_are_checked(asup, ramp_knobs):
    """Audit r1b L1: each row's ramp windows against the steps since the
    previous row -- in the ramp while the steps since the reset are below
    10,000, out after -- every step scored, finite means (None where none
    was scored)."""
    sup = asup
    rows = _c2_rows()

    def win(k, **change):
        def edit(r):
            r["ramp_windows"][k].update(change)
        return edit

    # on schedule: row 25 (in 10,000 / out 0), row 30 (0 / 10,000)
    assert _alert(sup, rows, 25, lambda r: None) is None
    assert _alert(sup, rows, 30, lambda r: None) is None
    # the steps off: a window that lost or double-counted steps (a resume
    # that did not restore the checkpoint's windows, or replayed them)
    assert _alert(sup, rows, 25, win("in", steps=2000, scored=2000)) == (
        "ramp_windows in 2000 steps, the schedule's 10000")
    assert _alert(sup, rows, 25, win("in", steps=12000, scored=12000)) == (
        "ramp_windows in 12000 steps, the schedule's 10000")
    # steps in the wrong window (the ramp not restarted at the reset)
    msg = _alert(sup, rows, 25, lambda r: r.__setitem__("ramp_windows", {
        "in": {"steps": 0, "scored": 0, "value_loss": None,
               "target_mean": None},
        "out": {"steps": 10000, "scored": 10000, "value_loss": 1.0,
                "target_mean": 2.0}}))
    assert msg == ("ramp_windows in 0 steps, the schedule's 10000; out 10000 "
                   "steps, the schedule's 0"), msg
    # steps not scored (non-finite value losses), non-finite or missing
    # means, a mean without a scored step
    assert _alert(sup, rows, 25, win("in", scored=9990)) == (
        "ramp_windows in 9990 of 10000 steps scored (a non-finite value loss "
        "or target?)")
    assert _alert(sup, rows, 25, win("in", value_loss=float("nan"))) == (
        "ramp_windows in value_loss nan over 10000 scored steps")
    assert _alert(sup, rows, 25, win("in", target_mean=None)) == (
        "ramp_windows in target_mean None over 10000 scored steps")
    assert _alert(sup, rows, 30, win("in", value_loss=0.5)) == (
        "ramp_windows in value_loss 0.5 with no step scored")
    # the final row: 4 iterations past the cutoff, all out
    assert _alert(sup, rows, 134, lambda r: None) is None
    assert "out 10000 steps, the schedule's 8000" in _alert(
        sup, rows, 134, win("out", steps=10000, scored=10000))
    # the smoke's shape: 20 steps per iteration, a 50-step ramp, a reset
    # every 4 iterations, rows every 4 -> 50 in / 30 out per cycle row
    # (the cutoff at 8,000: row 16's reset falls past it)
    smoke = dict(sup.LINEAGES["a"])
    sup.LINEAGES["a"] = {**smoke, "ramps": {**smoke["ramps"],
                                            "RAMP_STEPS": 50},
                         "resets": {**smoke["resets"],
                                    "RESET_EVERY_STEPS": 80,
                                    "RESET_UNTIL_ENV_STEPS": 8000}}
    sup.TRAIN_STEPS = 20
    try:
        for it, envs, want in ((4, {}, (50, 30)), (8, {4: 2000}, (50, 30)),
                               (20, {12: 7000, 16: 9000}, (0, 80)),
                               (23, {12: 7000, 16: 9000, 20: 11000},
                                (0, 60))):
            wins = {k: {"steps": n, "scored": n,
                        "value_loss": 1.0 if n else None,
                        "target_mean": 1.0 if n else None}
                    for k, n in zip(("in", "out"), want)}
            got = sup.ramp_alert({"iteration": it, "ramp_windows": wins},
                                 envs)
            assert "ramp_windows" not in got, (it, got)
            wins["in"]["steps"] += 20
            assert "ramp_windows in" in sup.ramp_alert(
                {"iteration": it, "ramp_windows": wins}, envs), it
    finally:
        sup.LINEAGES["a"], sup.TRAIN_STEPS = smoke, 2000


def test_a_rewritten_row_streams_without_an_alert(asup, ramp_knobs,
                                                  monkeypatch, tmp_path,
                                                  capsys):
    """Audit r1b L1, through Rows.emit: a kill between row 25 and its
    checkpoint, then the resume. The supervisor primed rows 5-20, streamed
    the killed timeline's row 25, then the resumed run pruned it and wrote
    its own row 25 (its trace and windows from the checkpoint at 24 plus
    iteration 25), then row 30: no ALERT -- the stale copy is no "previous
    row" for the rewrite."""
    sup = asup
    monkeypatch.setattr(sup, "run_ceiling", lambda v: 1.0)
    rows = _stream_rows(30)
    by = {r["iteration"]: r for r in rows}
    jsonl = tmp_path / "a.jsonl"
    head = "".join(json.dumps(by[i]) + NL for i in (5, 10, 15, 20))
    jsonl.write_text(head)
    stream = sup.Rows(str(jsonl), "a0", 0)
    stream.prime()
    stale = dict(by[25], env_steps=by[25]["env_steps"] + 300)
    with open(jsonl, "a") as f:
        f.write(json.dumps(stale) + NL)
    stream.emit()
    jsonl.write_text(head + json.dumps(by[25]) + NL)   # pruned, rewritten
    stream.emit()
    with open(jsonl, "a") as f:
        f.write(json.dumps(by[30]) + NL)
    stream.emit()
    out = capsys.readouterr().out
    assert out.count("EVAL a0 s0 iter 25 ") == 2, out
    assert "EVAL a0 s0 iter 30 " in out and "in/out 0/10000" in out
    assert "ALERT" not in out, out
    # ... and a rewrite that lost the checkpoint's windows (only iteration
    # 25's 2,000 steps) is an ALERT
    lost = dict(by[25], ramp_windows={
        "in": {"steps": 2000, "scored": 2000, "value_loss": 1.0,
               "target_mean": 1.0},
        "out": {"steps": 0, "scored": 0, "value_loss": None,
                "target_mean": None}})
    jsonl.write_text(head + json.dumps(lost) + NL)
    stream.emit()
    assert ("ALERT: a0 s0 row 25: ramp_windows in 2000 steps, the schedule's "
            "10000") in capsys.readouterr().out


def test_rows_with_a_rewritten_row_in_the_file(asup, ramp_knobs,
                                               monkeypatch, tmp_path,
                                               capsys):
    """Audit r1b L1: a file holding a stale row 25 AND its rewrite (unpruned)
    streams without an ALERT, and so does a supervisor restarted (primed)
    after the stale row, before the rewrite replaced it."""
    sup = asup
    monkeypatch.setattr(sup, "run_ceiling", lambda v: 1.0)
    rows = _stream_rows(30)
    by = {r["iteration"]: r for r in rows}
    stale = dict(by[25], env_steps=by[25]["env_steps"] + 300)
    head = "".join(json.dumps(by[i]) + NL for i in (5, 10, 15, 20))
    jsonl = tmp_path / "a.jsonl"
    jsonl.write_text(head + json.dumps(stale) + NL + json.dumps(by[25]) + NL
                     + json.dumps(by[30]) + NL)
    sup.Rows(str(jsonl), "a0", 0).emit()
    out = capsys.readouterr().out
    assert out.count("EVAL a0 s0 iter 25 ") == 2 and "ALERT" not in out, out
    jsonl.write_text(head + json.dumps(stale) + NL)
    stream = sup.Rows(str(jsonl), "a0", 0)
    stream.prime()                     # restarted after the stale row
    jsonl.write_text(head + json.dumps(by[25]) + NL + json.dumps(by[30])
                     + NL)
    stream.emit()
    out = capsys.readouterr().out
    assert "EVAL a0 s0 iter 25 " in out and "EVAL a0 s0 iter 30 " in out
    assert "ALERT" not in out, out


def test_a_smoke_counts_as_a_trainer(sup, monkeypatch, capsys):
    """Audit r1b L4: the Test C / C2 smokes (their parents and their
    "child" trainers) hold the GPU without an ablation.py on the command
    line: live_trainers counts them, so a supervisor started by hand beside
    one waits. Test B2W's smoke (smoke_testb2w.py) too, and E3's
    (smoke_teste3.py: its parent and its "child" trainers)."""
    assert sup.SMOKE_SCRIPTS == ("smoke_testc.py", "smoke_testc2.py",
                                 "smoke_testb2w.py", "smoke_teste3.py",
                                 "smoke_testq.py")
    procs = [
        (1, ["py", "-u", "C:" + chr(92) + "sp" + chr(92) + "smoke_testc2.py",
             "C:/sp/smoke_testc2_r2"]),
        (2, ["py", "-u", "C:/sp/smoke_testc2.py", "child", "C:/sp/a0",
             "resume1"]),
        (3, ["py", "smoke_testc.py", "kill", "out"]),
        (8, ["py", "-u", "C:/sp/smoke_testb2w.py", "child", "w1a",
             "C:/sp/b2w_smoke"]),
        (11, ["py", "-u", "C:" + chr(92) + "sp" + chr(92) + "smoke_teste3.py",
              "C:/sp/e3_smoke"]),
        (12, ["py", "-u", "C:/sp/smoke_teste3.py", "child", "m1",
              "C:/sp/e3_smoke"]),
        (4, ["py", "smoke_testc2.py.bak"]),
        (5, ["py", "tests/test_smoke_testc2.py"]),
        (6, ["py", "benchmarks/supervisors/supervise_testb.py", "a0", "0"]),
        (7, ["py", "post_testc2.py", "Y:/run"]),
        (9, ["py", "smoke_testb2w.py.bak"]),
        (10, ["py", "post_testb2w.py", "Y:/run"]),
        (13, ["py", "smoke_teste3.py.bak"]),
        (14, ["py", "tests/test_smoke_teste3.py"])]
    _procs(monkeypatch, sup, procs)
    assert sup.live_trainers() == [1, 2, 3, 8, 11, 12]
    assert [p for p, c in procs if sup.is_trainer(c)] == [1, 2, 3, 8, 11,
                                                          12]

    def effect(n):
        if n == 2:
            _procs(monkeypatch, sup, procs[6:])

    monkeypatch.setattr(sup.time, "sleep", _bounded(effect))
    sup.wait_no_trainer()
    assert ("ALERT: an ablation.py or distill.py trainer is already running "
            "(pids [1, 2, 3, 8, 11, 12])") in capsys.readouterr().out


# --------------------------------------------------------------------------
# Test B2W (Kai, 2026-11-04): Test B2's chain, w0-w2, with BBF's AdamW
# --------------------------------------------------------------------------
ADAM = ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-"
        "tokens-adam")
W = [f"{ADAM}-w{k}" for k in range(3)]
WV = [f"w{k}" for k in range(3)]
W_DATE = "2026-11-04"
# the pre-registration's optimizers, written out here (not read from the
# supervisor) so a drift in either is caught
BBF_OPTIM = {"OPTIMIZER": "adamw", "LEARNING_RATE": 1e-4,
             "WEIGHT_DECAY": 0.1, "ADAM_EPS": 0.00015, "DECAY_RANK1": False,
             "LR_WARMUP_STEPS": 0}
SGD_OPTIM = {"OPTIMIZER": "sgd", "LEARNING_RATE": 0.2, "WEIGHT_DECAY": 1e-4,
             "SGD_MOMENTUM": 0.9, "ADAM_EPS": None, "DECAY_RANK1": True,
             "LR_WARMUP_STEPS": 1000}


@pytest.fixture
def wsup(sup):
    """`sup` bound to Test B2W's lineage, as main() binds it for w0-w2."""
    sup.configure("w")
    return sup


def _knob(spec, k):
    return spec.get(k, getattr(muzero, k))


def test_test_b2ws_lineage(real_sup):
    sup = real_sup
    loaded = {k: getattr(sup, k) for k in LINEAGE_NAMES}
    sup.configure("w")
    assert (sup.LINEAGE, sup.GENERATIONS) == ("w", 3)
    assert sup.VARIANTS == dict(zip(WV, W))
    assert sup.RUNS == {v: f"mspacman-adam-{v}" for v in WV}
    assert sup.NAMES == {"w0": "W0", "w1": "W1", "w2": "W2"}
    assert sup.SEEDS == {0: W_DATE} and sup.PREP_PREFIX == "adam-prep"
    assert (sup.ITERATIONS, sup.NOMINAL_ITERATIONS) == (100, 50)
    assert sup.DISCOUNTS == {v: KEPT for v in WV}
    assert sup.DISCOUNT_RULE == "keep"
    assert [sup.target(v) for v in WV] == [33333, 66667, 100000]
    assert ablation.ADAM_ARM == sup.ADAM_ARM == ADAM
    assert sup.OPTIM_BBF == ablation.BBF_ADAMW == BBF_OPTIM
    assert sup.OPTIM_SGD == SGD_OPTIM
    assert sup.LINEAGES["w"]["optim"] is sup.OPTIM_BBF
    for k, v in enumerate(WV):
        spec = ablation.ARMS[sup.VARIANTS[v]]
        assert {o: _knob(spec, o) for o in BBF_OPTIM} == BBF_OPTIM
        assert sup.gen(v) == k and sup.is_student(v) == (k > 0)
        assert sup.master_of(v) == (WV[k - 1] if k else None)
        assert spec.get("_master_arm") == (W[k - 1] if k else None)
        if k:
            assert sup.master_offline(v) == distill.master_offline_steps(
                ablation.ARMS[W[k - 1]]) == (20000 if k > 1 else 0)
        # T{k} but for the optimizer (its keys, the master and the doc)
        t = ablation.ARMS[T[k]]
        skip = set(BBF_OPTIM) | {"_master_arm", "_doc"}
        assert ({x: y for x, y in spec.items() if x not in skip}
                == {x: y for x, y in t.items() if x not in skip})
        assert {o: _knob(t, o) for o in SGD_OPTIM} == SGD_OPTIM
    # every other lineage: no "optim" (quick() pins OPTIM_SGD), and every
    # one of its arms on EfficientZero's SGD
    for letter, lin in sup.LINEAGES.items():
        if letter == "w":
            continue
        assert "optim" not in lin, letter
        # (E3's "m" runs generations 1-2 only: "first")
        for k in range(lin.get("first", 0), lin["generations"]):
            arm = ablation.ARMS[f"{lin['arm']}-{letter}{k}"]
            assert {o: _knob(arm, o) for o in SGD_OPTIM} == SGD_OPTIM, letter
    # the others' names and numbers come back
    sup.configure("f")
    assert {k: getattr(sup, k) for k in LINEAGE_NAMES} == loaded


def test_main_binds_the_w_lineage(sup, monkeypatch, capsys):
    monkeypatch.setattr(sup, "other_supervisors", lambda v: [])
    monkeypatch.setattr(sup, "run_done", lambda v, s: True)
    assert sup.main(["w1", "0"]) == 0
    assert sup.LINEAGE == "w" and sup.VARIANTS == dict(zip(WV, W))
    assert "RUN COMPLETE w1 s0" in capsys.readouterr().out


def test_test_b2ws_command_and_paths_are_test_b2s_with_the_arm_moved(wsup):
    sup = wsup
    tsup2 = _load(SUPERVISORS / "supervise_testb.py", "supervise_testb_t")
    tsup2.configure("t")
    for k, v in enumerate(WV):
        t = TV[k]
        want = [x.replace(T[k], W[k]).replace(tsup2.prep_rel(t),
                                              sup.prep_rel(v))
                for x in tsup2.command(t, 0, 123)]
        assert sup.command(v, 0, 123) == want, v
        rel = sup.paths(v, 0)[0]
        assert rel == f"benchmarks/results/mspacman-adam-{v}-s0-{W_DATE}"
        assert sup.prep_rel(v) == (
            f"benchmarks/results/adam-prep-{v}-s0-{W_DATE}")
    c0 = sup.command("w0", 0)
    assert c0[c0.index("--stop-env-steps") + 1] == "33333"
    assert c0[c0.index("--iterations") + 1] == "100"
    assert c0[c0.index("--arms") + 1] == W[0] and "--token-vocab" not in c0


@pytest.mark.parametrize("k", range(3))
def test_test_b2ws_quick_probe_passes(wsup, k):
    if k:
        _master_ready(wsup, WV[k], final=48)
    r = _py(wsup.quick(WV[k], 0))
    assert r.returncode == 0, r.stderr[-2000:]


def test_b2ws_quick_probe_pins_adamw_and_every_other_lineage_sgd(sup):
    """w0's probe pins BBF's AdamW, every other lineage's EfficientZero's
    SGD, both the module's AdamW defaults; each break of one knob alone
    fails its probe, the controls pass."""
    arms = {"w": W[0], "t": T[0], "f": F[0], "r": RESET + "-r0",
            "n": LONG + "-n0", "a": RAMP + "-a0"}
    probes = {}
    for letter in arms:
        sup.configure(letter)
        probes[letter] = sup.quick(letter + "0", 0)
    for letter, q in probes.items():
        want = BBF_OPTIM if letter == "w" else SGD_OPTIM
        for k, v in want.items():
            op = "is" if v is None or isinstance(v, bool) else "=="
            assert f"assert spec.get({k!r}, muzero.{k}) {op} {v!r}; " in q
        # (audit r1a L2) and it builds one net's optimizer under the arm
        assert "muzero.make_optimizer(ablation.build_networks(_sp)" in q
        assert ("'AdamW'" if letter == "w" else "'SGD'") in q
    a = "ablation.ARMS"
    w0 = W[0]
    breaks = [("w", f"{a}[{w0!r}]['OPTIMIZER'] = 'sgd'"),
              ("w", f"{a}[{w0!r}]['LEARNING_RATE'] = 3e-4"),
              ("w", f"{a}[{w0!r}]['WEIGHT_DECAY'] = 1e-4"),
              ("w", f"{a}[{w0!r}]['ADAM_EPS'] = 1e-8"),
              ("w", f"del {a}[{w0!r}]['ADAM_EPS']"),
              ("w", f"{a}[{w0!r}]['DECAY_RANK1'] = True"),
              ("w", f"{a}[{w0!r}]['LR_WARMUP_STEPS'] = 1000")]
    for letter in "tfrna":
        arm = arms[letter]
        breaks += [(letter, f"{a}[{arm!r}]['OPTIMIZER'] = 'adamw'"),
                   (letter, f"{a}[{arm!r}]['LEARNING_RATE'] = 1e-4"),
                   (letter, f"{a}[{arm!r}]['WEIGHT_DECAY'] = 0.1"),
                   (letter, f"{a}[{arm!r}]['SGD_MOMENTUM'] = 0.0"),
                   (letter, f"{a}[{arm!r}]['ADAM_EPS'] = 0.00015"),
                   (letter, f"{a}[{arm!r}]['DECAY_RANK1'] = False"),
                   (letter, f"{a}[{arm!r}]['LR_WARMUP_STEPS'] = 0")]
    broken = []
    for letter, prelude in breaks:
        anchor = f"spec = ablation.ARMS[{arms[letter]!r}]; "
        assert probes[letter].count(anchor) == 1
        broken.append(probes[letter].replace(anchor, prelude + "; " + anchor))
    got = _py_each(broken + list(probes.values()))
    for code, g in zip(broken, got):
        assert g.startswith("AssertionError"), (code[-300:], g)
    assert got[len(broken):] == ["ok"] * len(probes)        # the controls
    # the module's AdamW defaults flipped: each in its own child (EACH
    # restores neither)
    head = "import muzero, ablation, muzero_vtree, distill; "
    for letter, flip in (("w", "muzero.ADAM_EPS = 1e-8; "),
                         ("t", "muzero.DECAY_RANK1 = False; ")):
        flipped = probes[letter].replace(head, head + flip)
        assert flipped != probes[letter]
        g = _py_each([flipped])[0]
        assert g.startswith("AssertionError"), g
    # the built optimizer (audit r1a L2): a builder that ignores the arm's
    # knobs fails the probe though every spec pin passes -- each in its own
    # child (EACH does not restore make_optimizer)
    for letter, flip in (
            ("w", "muzero.make_optimizer = lambda p: "
                  "__import__('torch').optim.AdamW(p, lr=1e-4); "),
            ("w", "muzero.make_optimizer = lambda p: "
                  "__import__('torch').optim.SGD(p, lr=1e-4); "),
            ("t", "muzero.make_optimizer = lambda p: "
                  "__import__('torch').optim.AdamW(p); ")):
        flipped = probes[letter].replace(head, head + flip)
        g = _py_each([flipped])[0]
        assert g.startswith("AssertionError"), (letter, flip, g)


# --------------------------------------------------------------------------
# E3 (Kai, 2026-11-05): Test B2's chain with mining at both swaps -- m1, m2
# here; generation 0 is Test B2's T0 itself, EXTERNAL, in its frozen tree
# --------------------------------------------------------------------------
MINED = ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-"
         "tokens-mined")
M = {"m1": f"{MINED}-m1", "m2": f"{MINED}-m2"}
M_DATE = "2026-11-05"
FROZEN = pathlib.Path(r"Y:\SwitchPlayer-testb2-run")
T0_RUN = f"mspacman-two-t0-s0-{T_DATE}"
MINED_TOK = [[1, 32], [3, 32]]       # a 64-frame mined token: 16 base steps


@pytest.fixture
def msup(sup):
    """`sup` bound to E3's lineage, as main() binds it for m1 / m2 -- its
    external T0's tree in tmp (the `sup` fixture)."""
    sup.configure("m")
    return sup


def _finish_t0(sup, final=47, env=33616, value=900.0, evals=True,
               files=True):
    """T0 complete in the EXTERNAL tree (tmp): its in-run rows (the last at
    `env` env steps), its marker, its final snapshot and own-play log, both
    exact evals there (_finish, which writes into REPO, for the external
    run). Returns its rows."""
    rows = _in_run(final)
    _jsonl(sup, "t0", {i: env if i == final else env * i // (final + 1)
                       for i in rows})
    pathlib.Path(sup.paths("t0", 0)[3]).write_bytes(b"done")
    if files:
        snap = pathlib.Path(sup.snapshot_path("t0", 0, final))
        snap.parent.mkdir(parents=True, exist_ok=True)
        snap.write_bytes(b"x")
        log = pathlib.Path(sup.own_play_arg("t0", 0))
        log.mkdir(parents=True, exist_ok=True)
        for i in range(1, final + 1):
            (log / f"iter-{i:03d}.pkl").write_bytes(b"x")
    if evals:
        for s in ("-acting", ""):
            _rows(sup.probe_path(T0_RUN, s, sup.tree_of("t0")), rows, value)
    return rows


def _m1_prep(sup, **doc):
    """m1's prep, a mined one, of the external T0's final snapshot (k 47)
    from its own-play log (the absolute path the prep was given)."""
    _write_prep(sup, "m1", k=47, **{"log_dir": sup.own_play_arg("t0", 0),
                                    "mine": True, **doc})


def test_e3s_lineage(real_sup):
    sup = real_sup
    loaded = {k: getattr(sup, k) for k in LINEAGE_NAMES}
    sup.configure("m")
    assert (sup.LINEAGE, sup.GENERATIONS) == ("m", 3)
    assert sup.VARIANTS == {**M, "t0": T[0]}
    assert sup.RUNS == {"m1": "mspacman-mined-m1", "m2": "mspacman-mined-m2",
                        "t0": "mspacman-two-t0"}
    assert sup.NAMES == {"m1": "M1", "m2": "M2", "t0": "T0"}
    assert sup.SEEDS == {0: M_DATE} and sup.PREP_PREFIX == "mined-prep"
    assert (sup.ITERATIONS, sup.NOMINAL_ITERATIONS) == (100, 50)
    assert sup.DISCOUNTS == {"t0": KEPT, "m1": KEPT, "m2": KEPT}
    assert sup.DISCOUNT_RULE == "keep"
    # the budget is Test B2's: its first third spent by T0 itself
    assert [sup.target(v) for v in ("m1", "m2")] == [66667, 100000]
    assert [sup.lineage_variant("m", k) for k in range(3)] == ["t0", "m1",
                                                               "m2"]
    assert (sup.master_of("m1"), sup.master_of("m2")) == ("t0", "m1")
    assert sup.master_of("t0") is None and not sup.is_student("t0")
    assert sup.is_student("m1") and sup.is_student("m2")
    assert sup.is_external("t0") and not sup.is_external("m1")
    assert sup.tree_of("t0") == str(FROZEN) and sup.tree_of("m2") == sup.REPO
    assert sup.run_name("t0", 0) == T0_RUN
    rel, base, jsonl, ckpt, _d = sup.paths("t0", 0)
    assert pathlib.Path(base) == FROZEN / "benchmarks" / "results" / T0_RUN
    assert ckpt.endswith(f"resume-mspacmannoframeskip-skip4-{T[0]}-seed0.ckpt")
    assert pathlib.Path(sup.snapshot_path("t0", 0, 47)) == (
        FROZEN / rel / f"snapshots{sup.TAG}-{T[0]}-seed0"
        / "weights-iter047.pth")
    assert sup.paths("m1", 0)[0] == (
        f"benchmarks/results/mspacman-mined-m1-s0-{M_DATE}")
    assert pathlib.Path(sup.paths("m1", 0)[1]).parent == (
        ROOT / "benchmarks" / "results")
    assert sup.prep_rel("m2") == (
        f"benchmarks/results/mined-prep-m2-s0-{M_DATE}")
    assert (sup.master_offline("m1"), sup.master_offline("m2")) == (0, 20000)
    lin = sup.LINEAGES["m"]
    assert lin["mine"] is True and lin["twin"] == "t" and lin["first"] == 1
    assert lin["external"] == {"variant": "t0", "tree": str(FROZEN)}
    assert ablation.MINE_ARM == sup.MINE_ARM == MINED
    for k, v in ((1, "m1"), (2, "m2")):
        spec, t = ablation.ARMS[M[v]], ablation.ARMS[T[k]]
        skip = {"_master_arm", "_mine", "_doc"}
        assert ({x: y for x, y in spec.items() if x not in skip}
                == {x: y for x, y in t.items() if x not in skip}), v
        assert spec["_mine"] is True and "_mine" not in t
        assert spec["_master_arm"] == sup.VARIANTS[sup.master_of(v)]
        assert sup.master_offline(v) == distill.master_offline_steps(
            ablation.ARMS[spec["_master_arm"]])
    # every other lineage: no mining, no external generation (Test Q's
    # mined lineage has its own suite)
    for letter, other in sup.LINEAGES.items():
        if letter not in ("m", "p"):
            assert not {"mine", "external", "first", "twin"} & set(other), \
                letter
    sup.configure("f")
    assert {k: getattr(sup, k) for k in LINEAGE_NAMES} == loaded
    sup.configure("t")                       # Test B2's own t0 is in-tree
    assert not sup.is_external("t0") and sup.tree_of("t0") == sup.REPO


def test_main_binds_the_m_lineage(sup, monkeypatch, capsys):
    monkeypatch.setattr(sup, "other_supervisors", lambda v: [])
    monkeypatch.setattr(sup, "run_done", lambda v, s: True)
    assert sup.main(["m2", "0"]) == 0
    assert sup.LINEAGE == "m" and sup.VARIANTS == {**M, "t0": T[0]}
    assert "RUN COMPLETE m2 s0" in capsys.readouterr().out
    assert sup.main(["m0", "0"]) == 2          # T0 is never run here


def test_e3s_paths_and_the_external_master(msup, tmp_path):
    sup = msup
    ext = sup.LINEAGES["m"]["external"]["tree"]
    assert ext == str(tmp_path / "external")
    rel, base, jsonl, ckpt, _d = sup.paths("t0", 0)
    assert base == os.path.join(ext, *rel.split("/"))
    assert sup.snapshot_path("t0", 0, 47).startswith(ext)
    assert sup.own_play_arg("t0", 0) == os.path.join(
        ext, *sup.own_play_rel("t0", 0).split("/"))
    assert sup.own_play_arg("m1", 0) == sup.own_play_rel("m1", 0)
    name, marker, probes, log, script, pvar, flag = sup.predecessor("m1", 0)
    assert (name, script, pvar) == ("T0", "supervise_testb.py", "t0")
    assert (marker, log, flag) == (ckpt, os.path.join(base, "supervisor.log"),
                                   os.path.join(base, "ABANDONED"))
    # T0 is no student: its two exact evals, no row 0 and no offline curve
    assert [p for p, _r in probes] == [
        sup.probe_path(T0_RUN, s, ext) for s in ("-acting", "")]
    assert all(p.startswith(ext) for p, _r in probes)
    name, marker, probes, *_x = sup.predecessor("m2", 0)
    assert name == "M1" and marker == sup.paths("m1", 0)[3]
    assert [os.path.basename(p) for p, _r in probes] == [
        f"probe_exhaustive_eval-mspacman-mined-m1-s0-{M_DATE}{s}.jsonl"
        for s in ("-acting", "", "-offline-acting", "-offline")]
    assert all(p.startswith(sup.REPO) for p, _r in probes)
    # T0 complete and evaluated in ITS tree: m1 may start, on T0's final
    # snapshot and own-play log there
    assert sup.pred_missing("m1", 0)
    _finish_t0(sup)
    assert sup.pred_missing("m1", 0) == []
    assert sup.master_final("m1", 0) == (47, sup.snapshot_path("t0", 0, 47))
    assert sup.master_files_missing("m1", 0) == []
    assert sup.choose_master("m1", 0) == {
        "which": "T0", "snapshot": sup.snapshot_path("t0", 0, 47),
        "iterations": 47, "master_mean_last2": 900.0}
    assert not (pathlib.Path(sup.REPO) / "benchmarks" / "results"
                / T0_RUN).exists()            # nothing of T0's in this tree
    # an own-play iteration missing there, an ABANDONED T0: no master
    (pathlib.Path(sup.own_play_arg("t0", 0)) / "iter-047.pkl").unlink()
    assert "iterations [47]" in sup.master_files_missing("m1", 0)[0]
    pathlib.Path(base, "ABANDONED").write_text("")
    with pytest.raises(SystemExit, match="MASTER ABANDONED: T0's run folder"):
        sup.choose_master("m1", 0)


def test_e3s_stops_count_t0s_final_env_steps(msup):
    sup = msup
    with pytest.raises(SystemExit,
                       match="STOP UNKNOWN: T0 has no completion marker"):
        sup.stop_env_steps("m1", 0)
    _finish_t0(sup, final=47, env=33616, evals=False, files=False)
    assert sup.stop_env_steps("m1", 0) == 66667 - 33616 == 33051
    _finish(sup, "m1", final=46, env=33100, evals=False, files=False)
    assert sup.stop_env_steps("m2", 0) == 100000 - 33616 - 33100


def test_a_spent_e3_budget_names_t0_and_m1(msup):
    _finish_t0(msup, final=47, env=60000, evals=False, files=False)
    _finish(msup, "m1", final=46, env=40000, evals=False, files=False)
    with pytest.raises(SystemExit, match=(
            r"STOP NOT POSITIVE: M2's --stop-env-steps would be 0 = 100000 "
            r"- 100000 \(the final env steps of T0, M1: \[60000, 40000\]\)")):
        msup.stop_env_steps("m2", 0)


def test_e3s_master_and_stop_from_t0s_real_frozen_run(real_sup):
    """T0's REAL files in the frozen tree (read only, never written; frozen,
    so the numbers never move): its final row 47 at 33,616 env steps, both
    evals complete, its last two deployed rows 745.667 and 1061.667 (mean
    903.667 >= 502.08), its snapshot and own-play log there -- m1's stop
    33,051."""
    if not (FROZEN / "benchmarks" / "results" / T0_RUN).exists():
        pytest.skip("Test B2's frozen tree is not on this host")
    sup = real_sup
    sup.configure("m")
    assert sup.final_row("t0", 0) == (47, 33616)
    assert sup.stop_env_steps("m1", 0) == 66667 - 33616 == 33051
    assert sup.pred_missing("m1", 0) == []
    assert sup.master_files_missing("m1", 0) == []
    pick = sup.choose_master("m1", 0)
    assert (pick["which"], pick["iterations"]) == ("T0", 47)
    assert pick["snapshot"] == sup.snapshot_path("t0", 0, 47)
    assert pick["master_mean_last2"] == pytest.approx(
        (745.667 + 1061.667) / 2, abs=1e-9)
    assert pick["master_mean_last2"] >= sup.NEAR_RANDOM


def test_e3s_command_is_test_b2s_with_the_arm_moved(msup):
    sup = msup
    tsup2 = _load(SUPERVISORS / "supervise_testb.py", "supervise_testb_t2")
    tsup2.configure("t")
    for k, v in ((1, "m1"), (2, "m2")):
        t = TV[k]
        want = [x.replace(T[k], M[v]).replace(tsup2.prep_rel(t),
                                              sup.prep_rel(v))
                for x in tsup2.command(t, 0, 123)]
        got = sup.command(v, 0, 123)
        assert got == want and got != tsup2.command(t, 0, 123), v
        assert "--mine" not in got              # mining is the prep's
        assert got[got.index("--token-vocab") + 1] == (
            f"benchmarks/results/mined-prep-{v}-s0-{M_DATE}/vocab.json")


def test_e3s_prep_reads_t0_in_its_tree_and_mines(msup, monkeypatch, capsys):
    """run_prep for m1, the real master gate: T0's final snapshot and
    own-play log in the external tree (absolute paths), --mine; then the
    MINED line from vocab.json's nested tokens."""
    sup = msup
    _finish_t0(sup, final=47, env=33616, value=900.0)
    monkeypatch.setattr(sup, "wait_no_trainer", lambda *a: None)
    monkeypatch.setattr(sup, "manifest_check", lambda: (True, ""))
    monkeypatch.setattr(sup.time, "sleep", lambda s: None)
    runs = []
    doc = {"vocab": [[0, 4], [1, 4], MINED_TOK],
           "dropped": [[0, 8], [0, 32]],
           "mined": [{"token": MINED_TOK, "count": 5, "k": 2,
                      "consistency": 0.75, "score": 3.75}]}
    _popen(monkeypatch, sup, [(0, lambda: _m1_prep(sup, **doc))], runs)
    assert sup.run_prep("m1", 0, {}) is True
    out = capsys.readouterr().out
    snap = sup.snapshot_path("t0", 0, 47)
    assert (f"MASTER T0: {snap} (iteration 47; deployed mean of its last two "
            f"rows 900.00") in out
    assert "PREP done (attempt 1): " + sup.prep_rel("m1") in out
    assert ("MINED m1: 1 tokens appended: 1,32;3,32 (count 5, k 2, c 0.750, "
            "score 3.8); vocabulary 3 slots; dropped 2" + NL) in out
    args, _k = runs[0]
    assert args == [sup.PY, "experiments/distill.py", "--master", snap,
                    "--master-iterations", "47", "--out",
                    f"benchmarks/results/mined-prep-m1-s0-{M_DATE}",
                    "--seed", "0", "--from-log", sup.own_play_arg("t0", 0),
                    "--mine"]
    assert os.path.isabs(args[args.index("--from-log") + 1])
    assert sup.prep_command("m1", 0, {"snapshot": snap, "iterations": 47}) \
        == args
    # Test B2's prep: no --mine, its master's log REPO-relative (as ever)
    sup.configure("t")
    t = sup.prep_command("t1", 0, {"snapshot": "s.pth", "iterations": 48})
    assert "--mine" not in t
    assert t[t.index("--from-log") + 1] == sup.own_play_rel("t0", 0)


def test_e3s_set_intact_needs_a_mined_prep_of_t0_in_its_tree(msup):
    sup = msup
    _finish_t0(sup, final=47, evals=False, files=False)
    _m1_prep(sup)
    assert sup.set_intact("m1", 0) == (True, "")
    for mine, shown in ((GONE, "None"), (False, "False"), (1, "1")):
        _m1_prep(sup, mine=mine)
        assert sup.set_intact("m1", 0) == (False, (
            f"vocab.json's mine is {shown}, not the lineage's True "
            f"(distill.py --mine)"))
    # the log T0's prep read must be T0's own-play log in its tree
    _m1_prep(sup, log_dir=sup.own_play_rel("t0", 0))
    ok, why = sup.set_intact("m1", 0)
    assert not ok and why.endswith(
        f"not T0's own-play log {sup.own_play_arg('t0', 0)}")
    # ... and a Test B2 prep must not be a mined one
    sup.configure("t")
    _finish(sup, "t0", final=48, evals=False, files=False)
    _write_prep(sup, "t1", k=48)
    assert sup.set_intact("t1", 0) == (True, "")
    _write_prep(sup, "t1", k=48, mine=True)
    assert sup.set_intact("t1", 0) == (False, (
        "vocab.json's mine is True, not the lineage's False (distill.py "
        "--mine)"))


def test_the_token_bound_is_the_mined_vocabularys_longest_token(msup):
    """E3-A9: m1's bound is its prep's vocabulary's longest token (nested
    mined tokens read as JSON lists), MINED_TOKEN_STEPS (16) while it
    cannot be read; every other lineage's MAX_TOKEN_STEPS (8) whatever its
    prep holds."""
    sup = msup
    assert (sup.MAX_TOKEN_STEPS, sup.MINED_TOKEN_STEPS) == (8, 16)
    assert sup.MINED_TOKEN_STEPS == muzero.TOKEN_MINED_MAX_FRAMES // 4
    assert sup.run_token_steps("m1") == 16                    # no prep yet
    _finish_t0(sup, evals=False, files=False)
    for vocab, want in (([[0, 4], [1, 32], [[2, 8], [3, 4]]], 8),
                        ([[0, 4], [1, 32], MINED_TOK], 16),
                        ([[0, 4], [[2, 8], [3, 12]]], 5),
                        ("nonsense", 16), ([[0, 4], [[2, 8], "x"]], 16),
                        ([[0, 4], [2.0, 8]], 16), (GONE, 16),
                        # (audit r2) capped at MINED_TOKEN_STEPS: 96 frames
                        ([[0, 4], [[1, 64], [3, 32]]], 16)):
        _m1_prep(sup, vocab=vocab)
        assert sup.run_token_steps("m1") == want, vocab
    sup.configure("t")
    _finish(sup, "t0", final=48, evals=False, files=False)
    _write_prep(sup, "t1", k=48, vocab=[[0, 4], MINED_TOK])
    assert sup.run_token_steps("t1") == sup.run_token_steps("t0") == 8


def test_row_alerts_at_the_mined_bound(sup):
    row = {"iteration": 20, "reward_ceiling": C}
    assert sup.row_alerts({**row, "reward_max": 16 * C}, "n/a", C, 16) == []
    got = sup.row_alerts({**row, "reward_max": 16 * C + 0.01}, "n/a", C, 16)
    assert got == [f"reward_max {16 * C + 0.01} > 16 x the ceiling 7.5000 "
                   f"(the vocabulary's longest token sums at most 16 base "
                   f"steps at the ceiling: the clip did not apply)"]
    # the bound as ever (None or 8): the 32-frame token's text
    for steps in (None, 8):
        got = sup.row_alerts({**row, "reward_max": 8 * C + 0.01}, "n/a", C,
                             steps)
        assert got == [f"reward_max {8 * C + 0.01} > 8 x the ceiling 7.5000 "
                       f"(a 32-frame token sums at most 8 base steps at the "
                       f"ceiling: the clip did not apply)"]


def test_e3s_rows_check_reward_max_at_the_vocabularys_bound(
        msup, tmp_path, capsys):
    sup = msup
    _finish_t0(sup, evals=False, files=False)
    _m1_prep(sup, student_ceiling=C, vocab=[[0, 4], [1, 32], MINED_TOK])
    jsonl = tmp_path / "m1.jsonl"
    rows = sup.Rows(str(jsonl), "m1", 0)

    def row(path, **r):
        with open(path, "a") as f:
            f.write(json.dumps({"elapsed_s": 1.0, "reward_ceiling": C, **r})
                    + NL)

    row(jsonl, iteration=5, reward_max=16 * C)
    rows.emit()
    out = capsys.readouterr().out
    assert out.startswith("EVAL m1 s0 iter 5") and "ALERT" not in out
    row(jsonl, iteration=10, reward_max=16 * C + 0.5,
        tok_mined_play_share={"1,32;3,32": 0.25},
        tok_mined_search_share={"1,32;3,32": 0.5},
        tok_mined_overlap_search_share=0.125)
    rows.emit()
    out = capsys.readouterr().out
    assert ("mined play {'1,32;3,32': 0.25}  search {'1,32;3,32': 0.5}  "
            "overlap 0.125") in out
    assert ("ALERT: m1 s0 row 10: reward_max 120.5 > 16 x the ceiling 7.5000 "
            "(the vocabulary's longest token") in out
    # Test B2's t1 at the same kind of row: its 8-step bound, its text
    sup.configure("t")
    _finish(sup, "t0", final=48, evals=False, files=False)
    _write_prep(sup, "t1", k=48, student_ceiling=C)
    tjsonl = tmp_path / "t1.jsonl"
    rows = sup.Rows(str(tjsonl), "t1", 0)
    row(tjsonl, iteration=5, reward_max=8 * C + 0.5)
    rows.emit()
    out = capsys.readouterr().out
    assert "ALERT: t1 s0 row 5: reward_max 60.5 > 8 x the ceiling" in out
    assert "a 32-frame token" in out and " | mined play" not in out


def test_the_mined_line_reads_nested_tokens(msup):
    sup = msup
    _finish_t0(sup, evals=False, files=False)
    _m1_prep(sup, vocab=[[0, 4], [[1, 8], [3, 4]], MINED_TOK],
             dropped=[[0, 8], [[2, 4], [5, 8]]],
             mined=[{"token": MINED_TOK, "count": 3, "k": 3,
                     "consistency": 0.5, "score": 3.0},
                    {"token": [[1, 8], [3, 4]], "count": 2, "k": 2,
                     "consistency": 1.0, "score": 2.0}])
    assert sup.prep_tokens_line("m1") == (
        "MINED m1: 2 tokens appended: 1,32;3,32 (count 3, k 3, c 0.500, "
        "score 3.0); 1,8;3,4 (count 2, k 2, c 1.000, score 2.0); vocabulary "
        "3 slots; dropped 2 (mined tokens among them: 2,4;5,8)")
    for bad in ({"mined": GONE}, {"dropped": [[0, "8"]]},
                {"mined": [{"token": MINED_TOK}]},
                {"mined": [{"token": MINED_TOK, "count": 1, "k": 2,
                            "consistency": None, "score": 1.0}]}):
        _m1_prep(sup, **bad)
        assert sup.prep_tokens_line("m1").startswith(
            "MINED m1: vocab.json's mined tokens cannot be read"), bad
    assert sup.json_token_steps([[1, 8], [3, 4]]) == [(1, 8), (3, 4)]
    assert sup.json_token_steps([2, 16]) == [(2, 16)]
    for bad in ([], [1], [1, 2, 3], [[1, 2], [3]], [True, 4], "ab", None):
        with pytest.raises(ValueError):
            sup.json_token_steps(bad)


def _m_masters(sup):
    """m1's master T0 and m2's master M1, each complete and master-like
    (_master_like) in its tree: T0 in the external one (its arm after 47 x
    2,000 steps), M1 here (M1's arm as student_arm leaves it, 20,000 + 46 x
    2,000 steps)."""
    _finish_t0(sup, final=47, evals=False)
    _master_like(sup.snapshot_path("t0", 0, 47), 0, 47, arms=T,
                 discounts=[KEPT] * 3)
    _finish(sup, "m1", final=46, env=33100, evals=False)
    _master_like(sup.snapshot_path("m1", 0, 46), 1, 46,
                 arms=[T[0], M["m1"], M["m2"]], discounts=[KEPT] * 3)


def test_e3s_quick_probes_pass_and_end_with_their_masters_checks(msup):
    sup = msup
    _m_masters(sup)
    q1, q2 = sup.quick("m1", 0), sup.quick("m2", 0)
    # the external T0: its master checks, then its recorded spec key by key
    # against its arm (E3-A1: generation 0 shared exactly)
    assert (f"_mb = distill.master_checks("
            f"{sup.snapshot_path('t0', 0, 47)!r}, 47)[0]; ") in q1
    assert q1.endswith("'keys that differ', _diff)")
    assert sup.snapshot_path("t0", 0, 47).startswith(
        sup.LINEAGES["m"]["external"]["tree"])
    assert f"assert muzero.TOKEN_BASE_FRAMES == {sup.BASE_FRAMES}; " in q1
    assert sup.BASE_FRAMES == muzero.TOKEN_BASE_FRAMES == 4
    assert q2.endswith(
        f"distill.master_checks({sup.snapshot_path('m1', 0, 46)!r}, 46)")
    for q, twin in ((q1, T[1]), (q2, T[2])):
        assert "assert spec.get('_mine') is True; " in q
        assert f"_t = ablation.ARMS[{twin!r}]; " in q
    assert f"assert ablation.ARMS[{T[0]!r}].get('_mine') is None; " in q1
    assert f"assert ablation.ARMS[{M['m1']!r}].get('_mine') is True; " in q2
    assert _py_each([q1, q2]) == ["ok", "ok"]
    # every other lineage's probe: no mining pin (bit for bit)
    sup.configure("t")
    assert "_mine" not in sup.quick("t0", 0)


def test_each_mining_pin_of_the_quick_probe_bites(msup):
    sup = msup
    _m_masters(sup)
    a = "ablation.ARMS"
    probes = {"m1": sup.quick("m1", 0), "m2": sup.quick("m2", 0)}
    breaks = [("m1", f"{a}[{M['m1']!r}]['_mine'] = False"),
              ("m1", f"del {a}[{M['m1']!r}]['_mine']"),
              ("m1", f"{a}[{M['m1']!r}]['ACT_SEARCH_SIMS'] = 8"),
              ("m1", f"{a}[{T[1]!r}]['ACT_SEARCH_SIMS'] = 8"),
              ("m1", f"{a}[{T[1]!r}]['_mine'] = True"),
              ("m1", f"{a}[{T[0]!r}]['_mine'] = True"),
              ("m2", f"{a}[{M['m1']!r}]['_mine'] = None"),
              ("m2", f"{a}[{M['m2']!r}]['TD_STEPS'] = 3"),
              ("m2", f"{a}[{M['m2']!r}]['_master_arm'] = {T[1]!r}")]
    broken = []
    for v, prelude in breaks:
        anchor = f"spec = ablation.ARMS[{M[v]!r}]; "
        assert probes[v].count(anchor) == 1
        broken.append(probes[v].replace(anchor, prelude + "; " + anchor))
    got = _py_each(broken + list(probes.values()))
    for code, g in zip(broken, got):
        assert g.startswith("AssertionError"), (code[-300:], g)
    assert got[len(broken):] == ["ok", "ok"]                # the controls
    # the build's own names, each in a child of its own (EACH restores
    # neither distill nor these muzero globals)
    head = "import muzero, ablation, muzero_vtree, distill; "
    for flip in ("del distill.mine_tokens; ", "del ablation.support_steps; ",
                 "muzero.TOKEN_MINED_PER_SWAP = 4; ",
                 "muzero.TOKEN_MINED_MAX_FRAMES = 32; ",
                 "distill.prep = lambda master: None; "):
        flipped = probes["m1"].replace(head, head + flip)
        assert flipped != probes["m1"]
        g = _py_each([flipped])[0]
        assert g.startswith("AssertionError"), (flip, g)


def test_the_external_masters_spec_must_be_its_arm_key_by_key(msup):
    """(audit r2) T0's snapshot recorded a spec the prep's master checks do
    not compare beyond its vocabulary (a non-student master runs as its
    arm): m1's probe compares every key -- a value that differs, or a key
    in one only, fails it, by name."""
    sup = msup
    _m_masters(sup)
    snap = sup.snapshot_path("t0", 0, 47)
    for change, key in (({"ACT_SEARCH_SIMS": 8}, "ACT_SEARCH_SIMS"),
                        ({"_an_extra_key": 1}, "_an_extra_key")):
        _master_like(snap, 0, 47, arms=T, discounts=[KEPT] * 3, **change)
        g = _py_each([sup.quick("m1", 0)])[0]
        assert g.startswith("AssertionError") and key in g, g
        assert "the external master snapshot spec is not its arm" in g
    _master_like(snap, 0, 47, arms=T, discounts=[KEPT] * 3)
    assert _py_each([sup.quick("m1", 0)]) == ["ok"]


def test_e3s_quick_probe_on_t0s_real_final_snapshot(real_sup):
    """The preflight's real probe for m1: distill.master_checks on T0's
    REAL final snapshot in the frozen tree (read only), and its recorded
    spec key by key against T0's arm in this tree."""
    if not (FROZEN / "benchmarks" / "results" / T0_RUN).exists():
        pytest.skip("Test B2's frozen tree is not on this host")
    sup = real_sup
    sup.configure("m")
    q = sup.quick("m1", 0)
    assert (f"_mb = distill.master_checks("
            f"{sup.snapshot_path('t0', 0, 47)!r}, 47)[0]; ") in q
    assert _py_each([q]) == ["ok"]


def test_other_lineages_quick_probes_are_their_own(sup):
    """(audit r2) Binding "m" leaves nothing behind: every other lineage's
    generation-0 probe is the one a separately loaded copy bound to that
    lineage writes, with no mining pin."""
    for letter in ("f", "t", "r", "n", "a", "w"):
        fresh = _load(SUPERVISORS / "supervise_testb.py",
                      f"supervise_testb_{letter}_fresh")
        fresh.configure(letter)
        sup.configure("m")
        sup.configure(letter)
        q = sup.quick(letter + "0", 0)
        assert q == fresh.quick(letter + "0", 0), letter
        assert "_mine" not in q and "TOKEN_MINED" not in q, letter


def test_an_external_master_alerts_name_its_frozen_tree(msup, monkeypatch,
                                                        capsys):
    """(audit r2) T0 unreadable or incomplete in its frozen tree: the wait
    and the master gate say so -- never "rerun its launcher" or "is its
    hand-off alive?"."""
    sup = msup
    ext = sup.LINEAGES["m"]["external"]["tree"]
    frozen = (f"T0's frozen files in {ext} are unreadable or incomplete -- "
              f"run nothing in that tree; Kai's call")
    monkeypatch.setattr(sup.time, "sleep", _bounded(lambda n: None, n=1))
    with pytest.raises(Stop):
        sup.wait_external("m1", 0)
    out = capsys.readouterr().out
    assert "WAITING for T0 before m1 s0" in out
    assert f"ALERT: {frozen}" in out
    for word in ("rerun its launcher", "hand-off", "keep T0's run folder",
                 "nothing drives it"):
        assert word not in out, word
    # complete but its deployed rows missing: the gate says the same
    _finish_t0(sup, evals=False)
    with pytest.raises(SystemExit) as e:
        sup.choose_master("m1", 0)
    assert str(e.value).endswith(frozen) and "hand-off" not in str(e.value)
    # its own-play log incomplete there
    _finish_t0(sup)
    (pathlib.Path(sup.own_play_arg("t0", 0)) / "iter-003.pkl").unlink()
    with pytest.raises(SystemExit) as e:
        sup.choose_master("m1", 0)
    assert "iterations [3]" in str(e.value)
    assert str(e.value).endswith(frozen)
    # an in-tree master (M1 for m2) keeps its own texts
    with pytest.raises(SystemExit, match="land them"):
        _finish(sup, "m1", final=46, env=33100, evals=False)
        sup.choose_master("m2", 0)


def test_the_mined_line_and_its_alert_on_every_prep_found(msup,
                                                          monkeypatch,
                                                          capsys):
    """(audit r2) The MINED line also when an intact prep is found on a
    restart, and an ALERT (Kai's call) when the mined prep appended no
    token."""
    sup = msup
    _finish_t0(sup, final=47, env=33616, value=900.0)
    monkeypatch.setattr(sup.time, "sleep", lambda s: None)
    _m1_prep(sup, vocab=[[0, 4], MINED_TOK], dropped=[[0, 8]],
             mined=[{"token": MINED_TOK, "count": 4, "k": 2,
                     "consistency": 0.5, "score": 2.0}])
    sup.write_pick(os.path.join(sup.prep_dir("m1"), "master.json"),
                   sup.choose_master("m1", 0))
    assert sup.run_prep("m1", 0, {}) is True
    out = capsys.readouterr().out
    assert "MINED m1: 1 tokens appended: 1,32;3,32" in out
    assert "ALERT" not in out
    _m1_prep(sup, vocab=[[0, 4]], dropped=[[0, 8]], mined=[])
    assert sup.run_prep("m1", 0, {}) is True
    out = capsys.readouterr().out
    assert "MINED m1: 0 tokens appended; vocabulary 1 slots" in out
    assert ("ALERT: M1's mined prep appended NO mined token -- it would run "
            "as T1 with no mining (Kai's call)") in out
    # Test B2's: no MINED line at all
    sup.configure("t")
    _finish(sup, "t0", final=48, evals=False)
    _write_prep(sup, "t1", k=48)
    sup.write_pick(os.path.join(sup.prep_dir("t1"), "master.json"),
                   {"which": "T0", "snapshot": "s", "iterations": 48,
                    "master_mean_last2": 700.0})
    assert sup.run_prep("t1", 0, {}) is True
    assert "MINED" not in capsys.readouterr().out
