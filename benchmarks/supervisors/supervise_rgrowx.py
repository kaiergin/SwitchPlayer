"""The token arm on Ms. Pac-Man, 3x longer (Kai, 2026-10-20): ONE run at
SEED 0 --
  tkx: effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-tokens-rgrow
       (TK, unchanged) for 270 iterations instead of 90, fresh; every
       run-length schedule stretches with --iterations (_run_length_globals)
--ray --cpu-actor (pinned by the arm). Design and pre-registered reads:
CURRICULUM.md "Next experiment (pre-registered) -- the token arm on Ms.
Pac-Man, 3x longer".

    python benchmarks/supervisors/supervise_rgrowx.py tkx 0
    python benchmarks/supervisors/supervise_rgrowx.py tkx 0 --no-wait
    python benchmarks/supervisors/supervise_rgrowx.py --write-manifest

It starts only once the GPU and the CPUs are free: before its preflight it
waits (wait_external) for RG -- the 90-iteration clip-growth run, training
from the OTHER run worktree (RG_TREE) -- to have its completion marker AND
both of its probe jsonls (acting + raw) to hold every row 5-90, i.e. RG's
evals have landed; RG complete for over 2 h with its evals still missing
is an hourly ALERT (is RG's hand-off alive?). A RESUME (this run's own
checkpoint exists) skips that wait: its start was already sanctioned
(audit r1 A-L1). --no-wait (logged, Kai's call) skips it too; it is refused
while any supervise_rgrow.py process lives (an unfinished RG's supervisor
would interleave RG's attempts with this run's). PROCEDURE for --no-wait:
kill this run's waiting supervisor, RG's supervisor and trainer, and RG's
hand-off post_rgrow.py and its probes (by command line, exact script names,
never taskkill /T), then run `launch_rgrowx.ps1 -NoWait`; RG's missing
evals run after this run completes (or beside it with Kai's approval). Before
EVERY launch attempt it also waits while ANY python runs an
experiments/ablation.py (any tree, arm or seed: one ablation trainer on the
GPU at a time, never a second writer; other GPU users -- a GPU probe,
Kai's Switch code -- are not seen). One supervisor per run: a second
supervise_rgrowx.py refuses to start (audit r1 A-L3 / B-M2). A run already
complete exits at once (RUN COMPLETE), before any wait or preflight. Each
row also ALERTs when host commit falls below 4 GB (a 3x run's memory creep
is untested; audit r1 A-L6).

Audit r2: a RESUME is "the checkpoint exists, or this run's supervisor.log
already logged a LAUNCHED / NO-WAIT override line" (a --no-wait start that
died before its first checkpoint does not wait for RG again). An ABANDONED
file in the run folder (Kai's call: stop the supervisor and its trainer
first, then create it) stops every later launch. A run folder holding
snapshots but no checkpoint is refused: a fresh start there would mix two
timelines (move the run folder and its two probe jsonls aside, or use a new
folder). Each attempt waits for no trainer, THEN checks the manifest, then
launches; rows keep streaming while it waits (an orphaned trainer of this
run stays monitored).

Audit r3: a run folder with no checkpoint while this run's probe jsonls
hold records is refused too (a fresh restart that forgot to move them would
keep the dead timeline's scores). While RG has no marker and no live
supervise_rgrow.py (or its log is silent > 4 h) the wait ALERTs hourly:
after a reboot RG's own launcher must be rerun too. --no-wait is also
refused while RG's hand-off post_rgrow.py lives. RETRACTING a --no-wait:
follow the FRESH RESTART procedure (CURRICULUM amendments r3, steps 1-4:
stop the supervisor and trainer, stop the hand-off and its probes, move the
run folder and its two probe jsonls aside, relaunch). Completion and
ABANDONED are re-checked right before every launch, and a launch without a
checkpoint re-checks the stray snapshots / probe rows there too (audit r4
A-L1). Rows stream from before the preflight, also between its retries.

Create the run folder before a launch that redirects into it (`>>
<folder>/supervisor.log`): the shell opens the redirect before this file
runs.

The run executes from the RUN WORKTREE this file lives in (a checkout of
the audited commit that nobody edits; development continues on branch
curriculum-p2 elsewhere), so REPO is this file's own tree. Supervisor v2
pattern: a relaunch auto-resumes from the last checkpoint; before EVERY
launch attempt the tree's AUDITED files must hash to the manifest written
from the audited commit (rgrowx_manifest.json)."""
import json
import os
import re
import subprocess
import sys
import time

VARIANTS = {
    "tkx": "effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-"
           "tokens-rgrow",
}
RUNS = {"tkx": "mspacman-tokens-rgrow-x270"}
ITERATIONS = 270
# Row checks (ALERTs): the ceiling each row must log -- the arm's schedule,
# 1 -> 160 log-linear by half of 270 x 2000 steps, from the iteration's
# starting step (pinned against muzero.reward_ceiling in the tests) -- a
# pace past which the cost is Kai's call, and a stored record reward past
# the per-step support's raw ceiling (400 units).
CEILING, CEILING_END_STEPS, TRAIN_STEPS = 160.0, 270_000, 2000
PACE_ALERT_MIN = 25.0
REWARD_MAX_ALERT = 400.0
COMMIT_ALERT_GB = 4.0
RG_EVALS_ALERT_S = 2 * 3600     # RG done this long, its evals still missing
RG_SILENT_ALERT_S = 4 * 3600    # RG unfinished, its supervisor.log silent


def expected_ceiling(iteration):
    """The reward ceiling row `iteration` must log."""
    return CEILING ** min(1.0, (iteration - 1) * TRAIN_STEPS
                          / CEILING_END_STEPS)
MODE = ["--ray", "--cpu-actor"]
ENV_ID = "MsPacmanNoFrameskip-v4"
TAG = "-mspacmannoframeskip-skip4"        # ablation's TAG for ENV_ID, skip 4
SEEDS = {0: "2026-10-20"}                 # the pre-registered seed and date
REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
# The shared interpreter lives in the main tree (worktrees have no .venv).
PY = os.environ.get("RGROW_PY", r"Y:\SwitchPlayer\.venv\Scripts\python.exe")
MAX_ATTEMPTS = 80                         # a 3x run: 2x supervise_rgrow's 40
MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "rgrowx_manifest.json")
# What RG's completion means here (wait_external): its checkpoint shrunk to
# a marker, and both probe jsonls holding a full-range record of every row.
RG_TREE = r"Y:\SwitchPlayer-rgrow-run"
RG_RUN = "mspacman-rgrow-s0-2026-10-19"
RG_ARM = "effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-rgrow"
RG_MARKER = os.path.join(RG_TREE, "benchmarks", "results", RG_RUN,
                         f"resume{TAG}-{RG_ARM}-seed0.ckpt")
RG_SUPLOG = os.path.join(RG_TREE, "benchmarks", "results", RG_RUN,
                         "supervisor.log")
RG_PROBES = tuple(os.path.join(RG_TREE, "benchmarks", "probes",
                               f"probe_exhaustive_eval-{RG_RUN}{suffix}.jsonl")
                  for suffix in ("-acting", ""))
RG_ROWS = tuple(range(5, 95, 5))
# Every repo module the run executes, the probe the readout runs, the
# suites the preflight runs, and the supervisors (supervise_rgrow.py: the
# suites read it).
SUITES = ["tests/test_rgrowx.py", "tests/test_rgrow.py",
          "tests/test_tokens.py",
          "tests/test_mspacman.py", "tests/test_bnrecal.py",
          "tests/test_prioract.py", "tests/test_prioract2.py",
          "tests/test_tempscale.py", "tests/test_paper_mode.py",
          "tests/test_mcts.py", "tests/test_vtree.py",
          "tests/test_vtree_communal.py", "tests/test_reanalyze_refresh.py",
          "tests/test_value_prefix.py", "tests/test_checkpoint_resume.py",
          "tests/test_harness.py"]
AUDITED = ["muzero.py", "redo.py", "muzero_vtree.py", "networks.py",
           "scalar_categorical.py", "memory.py", "EpisodicLifeEnv.py",
           "atari_wrappers.py", "residual_block.py",
           "experiments/ablation.py", "experiments/ray_actors.py",
           "pytest.ini", "tests/test_gumbelact8.py",
           # imported by the suites
           "tests/test_reanalysis.py",
           # older supervisors the suites read or exec
           "benchmarks/supervisors/supervise_bnrecal.py",
           "benchmarks/supervisors/supervise_gumbelact8.py",
           "benchmarks/supervisors/supervise_prioract2.py",
           "benchmarks/supervisors/supervise_searchact8.py",
           "benchmarks/supervisors/supervise_tempscale.py",
           "benchmarks/supervisors/supervise_tokens.py",
           "benchmarks/supervisors/supervise_mspacman.py",
           "benchmarks/supervisors/supervise_rgrow.py",
           "benchmarks/probes/probe_exhaustive_eval.py",
           "benchmarks/supervisors/supervise_rgrowx.py"] + SUITES


def paths(variant, seed):
    """(REL, BASE, JSONL, CKPT, DEATHS) of `variant` at `seed`: one folder
    per run."""
    arm = VARIANTS[variant]
    rel = f"benchmarks/results/{RUNS[variant]}-s{seed}-{SEEDS[seed]}"
    base = os.path.join(REPO, *rel.split("/"))
    return (rel, base, os.path.join(base, f"ablation{TAG}.jsonl"),
            os.path.join(base, f"resume{TAG}-{arm}-seed{seed}.ckpt"),
            os.path.join(base, "deaths.log"))


def command(variant, seed):
    """The pre-registered recipe: TK's (supervise_rgrow.py tk), --iterations
    270 the only change."""
    return [PY, "experiments/ablation.py", "--arms", VARIANTS[variant],
            "--seed", str(seed),
            "--iterations", str(ITERATIONS), "--train-steps",
            str(TRAIN_STEPS), "--eval-every", "5",
            "--env", ENV_ID, "--frameskip", "4", "--episode-cap", "3000",
            "--episodes-per-iter", "3", "--checkpoint-every", "2",
            "--snapshot-weights"] + MODE


def file_hashes(root):
    """sha256 of every AUDITED file's text, CRLF normalized to LF (git's
    autocrlf round trip must not matter)."""
    import hashlib
    out = {}
    for rel in AUDITED:
        with open(os.path.join(root, *rel.split("/")), "rb") as f:
            out[rel] = hashlib.sha256(
                f.read().replace(b"\r\n", b"\n")).hexdigest()
    return out


def manifest_check():
    """(ok, reason): this tree's AUDITED files hash to the manifest. Run
    before EVERY launch attempt: a relaunch must never run an unaudited
    edit."""
    try:
        with open(MANIFEST, encoding="utf-8") as f:
            want = json.load(f)
        have = file_hashes(REPO)
    except (OSError, ValueError) as e:
        return False, f"manifest check: {e}"
    if not isinstance(want, dict):
        return False, "manifest check: not a JSON object"
    stale = sorted(k for k in AUDITED if want.get(k) != have.get(k))
    if stale:
        return False, "not the audited revision: " + ", ".join(stale)
    return True, ""


def write_manifest():
    """Written from a CLEAN tree (the audited commit) only: uncommitted
    edits to an AUDITED file refuse it. The commit it was written at is
    recorded under "_commit" (provenance; manifest_check reads only the
    AUDITED keys)."""
    r = subprocess.run(["git", "status", "--porcelain", "--", *AUDITED],
                       cwd=REPO, capture_output=True, text=True)
    if r.returncode or r.stdout.strip():
        print("refusing --write-manifest: AUDITED files differ from the "
              "commit (or git failed):\n" + (r.stdout or r.stderr), flush=True)
        return 1
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                          capture_output=True, text=True)
    out = file_hashes(REPO)
    out["_commit"] = head.stdout.strip() if head.returncode == 0 else "?"
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print(f"wrote {MANIFEST} from {REPO}", flush=True)
    return 0


def quick(variant):
    """The import probe for `variant`'s arm (TK's: 36 tokens)."""
    arm = VARIANTS[variant]
    return ("import sys; sys.path[:0] = ['.', 'experiments']; "
            "import muzero, ablation, muzero_vtree; "
            f"spec = ablation.ARMS[{arm!r}]; "
            "assert spec['REWARD_GROWTH'] is True and "
            "spec['REWARD_UNIT'] == 10.0 and "
            "spec['REWARD_GROWTH_CEILING'] == 160.0; "
            "assert spec['BN_RECAL_ACTING'] is True; "
            "assert spec['MAX_REPLAY'] == 255; "
            "assert len(spec['TOKEN_VOCAB']) == 36; "
            "assert hasattr(ablation, 'reward_growth_gate') and "
            "hasattr(ablation.Run, 'apply_reward_ceiling'); "
            "assert hasattr(muzero, 'grown_reward')")


def run_done(variant, seed):
    """The run's completion marker: its checkpoint shrunk to a marker."""
    try:
        return os.path.getsize(paths(variant, seed)[3]) < 1_000_000
    except OSError:
        return False


def lines_of(path):
    try:
        with open(path, "rb") as f:
            return f.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return []


def rg_missing():
    """What RG still lacks before this run may start: [] once its marker is
    there and both probe jsonls hold a full-range Ms. Pac-Man record (a
    dict without "max_noop" whose env is ENV_ID) of every row in
    RG_ROWS."""
    out = []
    try:
        if os.path.getsize(RG_MARKER) >= 1_000_000:
            out.append("completion marker (checkpoint still full size)")
    except OSError:
        out.append("completion marker (no checkpoint)")
    for path in RG_PROBES:
        got = set()
        for line in lines_of(path):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if (isinstance(r, dict) and "max_noop" not in r
                    and r.get("env") == ENV_ID):
                got.add(r.get("iteration"))
        lack = [i for i in RG_ROWS if i not in got]
        if lack:
            out.append(f"{os.path.basename(path)} rows {lack}")
    return out


def rg_marker_age():
    """Seconds since RG's checkpoint became the completion marker (its
    mtime), or None while it is not one."""
    try:
        if os.path.getsize(RG_MARKER) < 1_000_000:
            return time.time() - os.path.getmtime(RG_MARKER)
    except OSError:
        pass
    return None


def rg_unattended():
    """None while RG is finished or driven; else why not: no live
    supervise_rgrow.py (e.g. a reboot that relaunched only this run), or a
    live one whose supervisor.log is silent for RG_SILENT_ALERT_S."""
    try:
        if os.path.getsize(RG_MARKER) < 1_000_000:
            return None
    except OSError:
        pass
    alive = live_rgrow_supervisors()
    if not alive:
        return ("no live supervise_rgrow.py -- rerun RG's launcher "
                "(scratchpad launch_rgrow.ps1)")
    try:
        quiet = time.time() - os.path.getmtime(RG_SUPLOG)
    except OSError:
        return None
    if quiet > RG_SILENT_ALERT_S:
        # launch_rgrow.ps1 is NOT idempotent: never rerun it beside a live
        # RG supervisor (audit r4 A-L2)
        return (f"RG's supervisor (pids {alive}) is alive but its log is "
                f"silent for {quiet / 3600:.1f} h -- check it; do NOT rerun "
                f"launch_rgrow.ps1 beside it")
    return None


def rg_gave_up():
    """RG's supervisor logged CRASH LOOP / MAX ATTEMPTS after its last
    launch or preflight line."""
    lines = lines_of(RG_SUPLOG)
    fail = max((i for i, l in enumerate(lines) if l.startswith(
        ("CRASH LOOP", "MAX ATTEMPTS"))), default=-1)
    live = max((i for i, l in enumerate(lines) if l.startswith(
        ("LAUNCHED", "preflight ok", "PREFLIGHT"))), default=-1)
    return fail > live


def wait_external(variant, seed):
    """Block until RG has completed AND its evals have landed (rg_missing()
    empty): the GPU and the CPUs are then free. Logs a WAITING line hourly
    (the hand-off's stale-log ALERT reads the log's mtime) and ALERTs while
    RG's supervisor has given up."""
    first, polls = True, 0
    while True:
        missing = rg_missing()
        if not missing:
            if not first:
                print(f"RG ({RG_RUN}) complete and evaluated; {variant} "
                      f"s{seed} may start", flush=True)
            return
        if first or polls % 60 == 0:
            print(f"WAITING for RG ({RG_RUN}) before {variant} s{seed} "
                  f"[{polls // 60} h]: missing " + "; ".join(missing)[:400],
                  flush=True)
            if rg_gave_up():
                print("ALERT: RG's supervisor gave up (CRASH LOOP / MAX "
                      f"ATTEMPTS) -- {variant} keeps waiting", flush=True)
            why = rg_unattended()
            if why:
                print(f"ALERT: RG is unfinished and nothing drives it: {why}"
                      f"; or start {variant} with --no-wait (Kai's call)",
                      flush=True)
            age = rg_marker_age()
            if age is not None and age > RG_EVALS_ALERT_S:
                print(f"ALERT: RG completed {age / 3600:.1f} h ago but its "
                      "evals still lack rows -- is RG's hand-off (scratchpad "
                      "post_rgrow.py) alive? Rerun its missing probes, or "
                      f"start {variant} with --no-wait (Kai's call)",
                      flush=True)
        first = False
        time.sleep(60)
        polls += 1


def preflight(variant):
    """(ok, reason): the manifest, the quick import probe, then the suites
    IN this tree."""
    ok, why = manifest_check()
    if not ok:
        return False, why
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}
    r = subprocess.run([PY, "-c", quick(variant)], cwd=REPO,
                       capture_output=True, text=True, env=env)
    if r.returncode:
        return False, "quick probe: " + (r.stderr.strip().splitlines()
                                         or ["?"])[-1]
    try:
        r = subprocess.run([PY, "-m", "pytest", "-x", "-q", "-p",
                            "no:cacheprovider", *SUITES],
                           cwd=REPO, capture_output=True, text=True, env=env,
                           timeout=3 * 3600)
    except subprocess.TimeoutExpired:
        return False, "tests timed out"
    if r.returncode:
        lines = (r.stdout + r.stderr).strip().splitlines()
        bad = [l for l in lines if l.startswith(("FAILED", "ERROR"))][:3]
        # Stable across polls: no timings, no object addresses.
        clean = [re.sub(r" at 0x[0-9a-fA-F]+", "", re.sub(
            r" in [0-9.]+s( \(\d+:\d\d:\d\d\))?", "", l)) for l in lines[-2:]]
        return False, f"tests failed (rc={r.returncode}): " + " | ".join(
            [re.sub(r" at 0x[0-9a-fA-F]+", "", l) for l in bad] or clean)
    return True, ""


def live_trainers():
    """PIDs of ANY python running experiments/ablation.py (any tree, arm or
    seed): an orphan of an earlier supervisor (the child is detached and
    survives a supervisor kill), RG, a duplicate launch. Python processes
    only: an editor or a git command naming the file is not a trainer
    (audit r1 A-L5)."""
    import psutil
    out = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name = (p.info.get("name") or "").lower()
            cmd = p.info["cmdline"] or []
        except (psutil.Error, KeyError):
            continue
        if name.startswith("python") and any(
                c.replace(chr(92), "/").endswith("experiments/ablation.py")
                for c in cmd):
            out.append(p.info["pid"])
    return out


def _mine():
    """This process and its ancestors (the venv's python.exe is a launcher
    whose child runs the script with the same command line)."""
    import psutil
    out = {os.getpid()}
    try:
        out.update(a.pid for a in psutil.Process().parents())
    except psutil.Error:
        pass
    return out


def other_supervisors():
    """PIDs of OTHER python processes running supervise_rgrowx.py (never
    this process or its ancestors)."""
    import psutil
    mine = _mine()
    out = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name = (p.info.get("name") or "").lower()
            cmd = p.info["cmdline"] or []
        except (psutil.Error, KeyError):
            continue
        if p.info["pid"] in mine or not name.startswith("python"):
            continue
        if any(c.replace(chr(92), "/").endswith("supervise_rgrowx.py")
               for c in cmd):
            out.append(p.info["pid"])
    return out


def commit_free_gb():
    """Free host commit (page file + RAM), GB; None off Windows."""
    import ctypes

    class MEMSTAT(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    try:
        m = MEMSTAT()
        m.dwLength = ctypes.sizeof(MEMSTAT)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return None
        return m.ullAvailPageFile / 2**30
    except (AttributeError, OSError):
        return None


def live_rgrow_supervisors():
    """PIDs of PYTHON processes running supervise_rgrow.py (either variant,
    any seed; never supervise_rgrowx.py; an editor naming the file is not
    one -- audit r4 A-L4)."""
    import psutil
    out = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name = (p.info.get("name") or "").lower()
            cmd = p.info["cmdline"] or []
        except (psutil.Error, KeyError):
            continue
        if name.startswith("python") and any(
                c.replace(chr(92), "/").endswith("/supervise_rgrow.py")
                or c == "supervise_rgrow.py" for c in cmd):
            out.append(p.info["pid"])
    return out


def live_rgrow_handoffs():
    """PIDs of python processes running RG's hand-off post_rgrow.py (exact
    name: never post_rgrowx.py)."""
    import psutil
    out = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name = (p.info.get("name") or "").lower()
            cmd = p.info["cmdline"] or []
        except (psutil.Error, KeyError):
            continue
        if name.startswith("python") and any(
                os.path.basename(c.replace(chr(92), "/")) == "post_rgrow.py"
                for c in cmd):
            out.append(p.info["pid"])
    return out


class Rows:
    """Stream new jsonl rows as EVAL lines (the run's in-run eval, the
    reward-growth fields and the token diagnostics)."""

    def __init__(self, jsonl, variant, seed):
        self.jsonl, self.variant, self.seed = jsonl, variant, seed
        self.seen, self.last = set(), None

    def lines(self):
        try:
            with open(self.jsonl, "rb") as f:
                return [l for l in f.read().decode("utf-8", "replace")
                        .splitlines() if l.strip()]
        except OSError:
            return []

    def prime(self):
        self.seen.update(l.strip() for l in self.lines())

    def emit(self):
        for line in self.lines():
            key = line.strip()
            if key in self.seen:
                continue
            self.seen.add(key)
            try:
                r = json.loads(line)
            except ValueError:
                continue
            el = r.get("elapsed_s")
            per = "n/a"
            if (isinstance(el, (int, float)) and self.last
                    and r.get("iteration", 0) > self.last[1]):
                per = "%.1f" % ((el - self.last[0]) / 60.0
                                / (r["iteration"] - self.last[1]))
            if isinstance(el, (int, float)):
                self.last = (el, r.get("iteration", 0))
            # decisions per TRAJECTORY = per life on Ms. Pac-Man
            tok = (" | frames/decision %s  decisions/life %s  play share "
                   "%s  search share %s" % (
                       r.get("tok_frames_per_decision"),
                       r.get("tok_decisions_per_game"),
                       r.get("tok_duration_play_share"),
                       r.get("tok_duration_search_share")))
            print("EVAL %s s%d iter %s  train %s  env %s  return %s  H(pi) %s "
                  " erank %s | ceiling %s  kept %s  max %s%s | elapsed %s s, "
                  "%s min/iter" % (
                      self.variant, self.seed, r.get("iteration"),
                      r.get("train_steps"), r.get("env_steps"),
                      r.get("return_mean"), r.get("policy_entropy"),
                      r.get("proj_erank"), r.get("reward_ceiling"),
                      r.get("reward_kept"), r.get("reward_max"), tok, el,
                      per),
                  flush=True)
            for alert in row_alerts(r, per):
                print(f"ALERT: {self.variant} s{self.seed} row "
                      f"{r.get('iteration')}: {alert}", flush=True)
            free = commit_free_gb()
            if free is not None and free < COMMIT_ALERT_GB:
                print(f"ALERT: {self.variant} s{self.seed} row "
                      f"{r.get('iteration')}: host commit free {free:.1f} GB "
                      f"< {COMMIT_ALERT_GB:g} GB -- memory creep?", flush=True)


def row_alerts(r, per):
    """What a row shows that needs a human: a ceiling that is not the
    schedule's (the re-clip did not run), a pace past PACE_ALERT_MIN
    min/iteration (pre-registered: Kai's call), a stored record reward past
    REWARD_MAX_ALERT units."""
    out = []
    it, c = r.get("iteration"), r.get("reward_ceiling")
    if isinstance(it, int) and it >= 1:
        want = expected_ceiling(it)
        if not isinstance(c, (int, float)) or abs(c - want) > 1e-3 * want:
            out.append(f"reward_ceiling {c!r}, the schedule says "
                       f"{want:.4f}")
    try:
        if float(per) > PACE_ALERT_MIN:
            out.append(f"{per} min/iteration > {PACE_ALERT_MIN:g} -- the "
                       f"pre-registered pace rule: Kai's call")
    except ValueError:
        pass
    m = r.get("reward_max")
    if isinstance(m, (int, float)) and m > REWARD_MAX_ALERT:
        out.append(f"reward_max {m} > {REWARD_MAX_ALERT:g} units (past the "
                   f"per-step support's raw ceiling)")
    return out


def snapshot(deaths_log, variant, seed, attempt, rc, lifetime):
    """Append the death-time environment to deaths.log; one line to stdout."""
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"=== death attempt={attempt} rc={rc} lifetime={lifetime:.0f}s "
             f"at {stamp}"]
    for cmd in (["nvidia-smi", "--query-gpu=memory.used,memory.total,"
                 "utilization.gpu,temperature.gpu", "--format=csv,noheader"],
                ["powershell", "-NoProfile", "-Command",
                 "$o=Get-CimInstance Win32_OperatingSystem; "
                 "'ram_free_gb: ' + [math]::Round($o.FreePhysicalMemory/1MB,1)"
                 " + ' commit_free_gb: ' + "
                 "[math]::Round($o.FreeVirtualMemory/1MB,1)"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=30).stdout.strip()
        except Exception as e:
            out = f"<{type(e).__name__}>"
        lines.append(out)
    with open(deaths_log, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    gpu = lines[1].split(",")[0].strip() if len(lines) > 1 else "?"
    print(f"DEATH {variant} s{seed} rc={rc} after {lifetime:.0f}s; snapshot "
          f"-> deaths.log (gpu used {gpu})", flush=True)


def wait_no_trainer(rows=None):
    """Block while ANY python runs experiments/ablation.py (an orphan of an
    earlier supervisor, RG, a duplicate): one ablation trainer on the GPU at
    a time, and never a second writer. `rows` keep streaming meanwhile (an
    orphan of THIS run keeps its EVAL lines and ALERTs; audit r2 A-L4)."""
    polls, told = 0, None
    while True:
        if rows is not None:
            rows.emit()
        pids = live_trainers()
        if not pids:
            return
        msg = (f"ALERT: an experiments/ablation.py is already running (pids "
               f"{pids}) -- one run on the GPU at a time, and never a second "
               f"writer")
        if msg != told or polls % 60 == 0:
            print(f"{msg} [{polls} min]", flush=True)
            told = msg
        time.sleep(60)
        polls += 1


def abandoned(base):
    """Kai's call: an ABANDONED file in the run folder stops every launch."""
    return os.path.exists(os.path.join(base, "ABANDONED"))


def sanctioned(base, ckpt):
    """This run already started once: its checkpoint exists, or its own
    supervisor.log logged a launch or the --no-wait override."""
    if os.path.exists(ckpt):
        return True
    return any(l.startswith(("LAUNCHED", "NO-WAIT override"))
               for l in lines_of(os.path.join(base, "supervisor.log")))


def stray_snapshots(base, ckpt):
    """Snapshots in the run folder with no checkpoint beside them (a fresh
    start there would mix two timelines; audit r2 A-M1)."""
    import glob
    if os.path.exists(ckpt):
        return []
    return sorted(glob.glob(os.path.join(base, "snapshots*", "*.pth")))


def stray_probe_rows(variant, seed, ckpt):
    """This run's probe jsonls holding records while no checkpoint exists
    (probe rows only exist for settled snapshots, i.e. after a checkpoint;
    audit r3 A-M1)."""
    if os.path.exists(ckpt):
        return []
    run = f"{RUNS[variant]}-s{seed}-{SEEDS[seed]}"
    out = []
    for suffix in ("-acting", ""):
        path = os.path.join(REPO, "benchmarks", "probes",
                            f"probe_exhaustive_eval-{run}{suffix}.jsonl")
        if any(l.strip() for l in lines_of(path)):
            out.append(os.path.basename(path))
    return out


def main(argv):
    try:
        # cp1252 under the launcher's redirect: a non-ASCII character in a
        # PREFLIGHT line must not kill the supervisor (audit r3 A-L6)
        sys.stdout.reconfigure(errors="backslashreplace")
    except (AttributeError, ValueError):
        pass
    if "--write-manifest" in argv:
        return write_manifest()
    no_wait = argv[2:] == ["--no-wait"]
    if (len(argv) != (3 if no_wait else 2) or argv[0] not in VARIANTS
            or not argv[1].isdigit() or int(argv[1]) not in SEEDS):
        print(f"usage: supervise_rgrowx.py <variant: {sorted(VARIANTS)}> "
              f"<seed: {sorted(SEEDS)}> [--no-wait] | --write-manifest",
              flush=True)
        return 2
    variant, seed = argv[0], int(argv[1])
    arm = VARIANTS[variant]
    rel, base, jsonl, ckpt, deaths_log = paths(variant, seed)
    others = other_supervisors()
    if others:
        print(f"refusing: another supervise_rgrowx.py is alive (pids "
              f"{others}) -- one supervisor per run", flush=True)
        return 2
    if run_done(variant, seed):
        # Before any wait or preflight: a relaunch after a reboot must not
        # run the CPU suites for nothing.
        print(f"RUN COMPLETE {variant} s{seed} (checkpoint is a marker)",
              flush=True)
        return 0
    if abandoned(base):
        print(f"ABANDONED {variant} s{seed}: the run folder holds an "
              f"ABANDONED file -- not launching", flush=True)
        return 0
    stray = stray_snapshots(base, ckpt)
    if stray:
        print(f"refusing: {len(stray)} snapshots but no checkpoint in "
              f"{rel} -- a fresh start would mix two timelines; move the run "
              f"folder and its two probe jsonls aside", flush=True)
        return 2
    stray = stray_probe_rows(variant, seed, ckpt)
    if stray:
        print(f"refusing: probe rows ({', '.join(stray)}) but no checkpoint "
              f"-- a fresh start would keep the old timeline's scores; move "
              f"them (and the run folder) aside", flush=True)
        return 2
    if no_wait:
        alive = live_rgrow_supervisors()
        if alive:
            print(f"refusing --no-wait: a supervise_rgrow.py is alive (pids "
                  f"{alive}) -- it would relaunch RG beside this run; kill it "
                  f"first", flush=True)
            return 2
        alive = live_rgrow_handoffs()
        if alive:
            print(f"refusing --no-wait: RG's hand-off post_rgrow.py is alive "
                  f"(pids {alive}) -- its probes would run beside this run; "
                  f"kill it first", flush=True)
            return 2
        print(f"NO-WAIT override: {variant} s{seed} starts without RG's "
              f"completion marker and evals", flush=True)
    elif sanctioned(base, ckpt):
        print(f"RESUMING {variant} s{seed}: it started before (checkpoint or "
              f"launch logged) -- no wait for RG", flush=True)
    else:
        wait_external(variant, seed)

    os.makedirs(base, exist_ok=True)
    # Before the preflight: rows an orphan of this run writes meanwhile are
    # still printed and checked (audit r3 A-L3).
    rows = Rows(jsonl, variant, seed)
    rows.prime()
    polls, told = 0, None
    while True:
        ok, why = preflight(variant)
        if ok:
            break
        if why != told or polls % 6 == 0:
            print(f"PREFLIGHT not ready ({polls * 10} min): {why}", flush=True)
            told = why
        time.sleep(600)
        polls += 1
        rows.emit()          # an orphan's rows between retries (r4 A-L3)
    print(f"preflight ok (manifest + quick probe + {len(SUITES)} suites); "
          f"starting {variant} s{seed} ({arm}, {ITERATIONS} iterations)",
          flush=True)

    # CUDA_VISIBLE_DEVICES is dropped: inherited from a shell that set -1 it
    # would silently train on the CPU.
    env = {**{k: v for k, v in os.environ.items()
              if k != "CUDA_VISIBLE_DEVICES"},
           "ABLATION_RESULTS": rel, "PYTHONFAULTHANDLER": "1"}
    stall_s = 2 * 3600    # the child logs at least every checkpoint (2 iters)
    deaths, quick_deaths = [], 0

    def completed():
        try:
            return os.path.getsize(ckpt) < 1_000_000
        except OSError:
            return False

    for attempt in range(1, MAX_ATTEMPTS + 1):
        if completed():
            print(f"RUN COMPLETE {variant} s{seed} (checkpoint is a marker)",
                  flush=True)
            return 0
        if abandoned(base):
            print(f"ABANDONED {variant} s{seed}: the run folder holds an "
                  f"ABANDONED file -- not relaunching", flush=True)
            return 0
        # The guard first, THEN the manifest (the guard can block for hours;
        # the manifest is checked right before the launch; audit r2 A-L3).
        polls, told = 0, None
        while True:
            wait_no_trainer(rows)
            ok, why = manifest_check()
            if ok:
                break
            if why != told or polls % 60 == 0:
                print(f"ALERT: {why} -- not relaunching until the tree is the "
                      f"audited revision [{polls} min]", flush=True)
                told = why
            time.sleep(60)
            polls += 1
        # The waits above can last hours: completion (an orphan finished)
        # and ABANDONED are re-checked right before the launch (r3 A-L2).
        if completed():
            print(f"RUN COMPLETE {variant} s{seed} (checkpoint is a marker)",
                  flush=True)
            return 0
        if abandoned(base):
            print(f"ABANDONED {variant} s{seed}: the run folder holds an "
                  f"ABANDONED file -- not relaunching", flush=True)
            return 0
        # A launch without a checkpoint: no old timeline may sit beside it
        # (a retraction done while this supervisor waited; r4 A-L1).
        stray = (stray_snapshots(base, ckpt)
                 + stray_probe_rows(variant, seed, ckpt))
        if stray:
            names = ", ".join(os.path.basename(x) for x in stray[:4])
            print(f"refusing: {len(stray)} stray snapshots / probe jsonls "
                  f"but no checkpoint ({names}) -- not launching; follow the "
                  f"fresh-restart procedure", flush=True)
            return 2
        args = command(variant, seed) + (["--resume"] if os.path.exists(ckpt)
                                         else [])
        log = open(os.path.join(base, f"supv2_{attempt}.log"), "ab")
        born = time.time()
        p = subprocess.Popen(args, cwd=REPO, env=env, stdout=log,
                             stderr=subprocess.STDOUT, creationflags=0x08000000)
        print(f"LAUNCHED {variant} s{seed} attempt {attempt} "
              f"({'resume' if '--resume' in args else 'fresh'}), pid {p.pid}",
              flush=True)
        alerted = 0.0
        while p.poll() is None:
            rows.emit()
            try:
                idle = time.time() - max(os.path.getmtime(log.name), born)
            except OSError:
                idle = 0.0
            if idle > stall_s and time.time() - alerted > 3600:
                print(f"ALERT: attempt {attempt} (pid {p.pid}) log silent for "
                      f"{idle / 3600:.1f} h -- possible hang", flush=True)
                alerted = time.time()
            time.sleep(30)
        log.close()
        rows.emit()
        rc = p.returncode
        lifetime = time.time() - born
        snapshot(deaths_log, variant, seed, attempt, rc, lifetime)
        if completed():
            print(f"RUN COMPLETE {variant} s{seed}", flush=True)
            return 0
        deaths.append(time.time())
        if len([t for t in deaths if time.time() - t < 3600]) >= 5:
            print("CRASH LOOP: 5 deaths within 60 min -- stopping for "
                  "diagnosis", flush=True)
            return 1
        if lifetime < 600:
            quick_deaths += 1
            wait = min(900, 60 * (2 ** quick_deaths))
            print(f"quick death #{quick_deaths}; backing off {wait}s",
                  flush=True)
            time.sleep(wait)
        else:
            quick_deaths = 0
            time.sleep(15)
    print("MAX ATTEMPTS REACHED", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
