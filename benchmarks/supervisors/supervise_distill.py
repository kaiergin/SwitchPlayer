"""Curriculum P3's distillation step (Kai, 2026-10-21): TWO runs at SEED 0,
one after the other --
  ds: effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-tokens-
      rgrow-student-distill (the distilled student: the master's re-tokenized
      games pre-filled, 20,000 offline steps, the master's end-state
      schedules)
  fs: ...-student-fresh (the same student vocabulary, from scratch, TK's
      schedules)
90 iterations each, the TK recipe, --ray --cpu-actor (pinned by the arms).
Design and pre-registered reads: CURRICULUM.md "Next experiment
(pre-registered) -- P3's distillation step".

    python benchmarks/supervisors/supervise_distill.py ds 0
    python benchmarks/supervisors/supervise_distill.py ds 0 --no-wait
    python benchmarks/supervisors/supervise_distill.py fs 0
    python benchmarks/supervisors/supervise_distill.py --write-manifest

ds waits (wait_external) for TKX -- the 270-iteration token run training
from ITS run worktree (TKX_TREE) -- to have its completion marker AND both
of its probe jsonls to hold every row 5-270 (its evals have landed); then
its preflight (manifest, quick probe, the suites); then THE PREP, once:
choose_master picks the master (the final snapshot of whichever of TKX and
TK-90 has the higher 7-row deployed mean; TK-90 when TKX has no complete
row window), logged in PREP/master.json, and experiments/distill.py
records it, derives the student vocabulary, re-tokenizes and searches
(GPU, ~0.5 h; log PREP/prep.log; up to PREP_ATTEMPTS tries 15 min apart;
vocab.json is its completion signal). Then DS launches. fs waits for DS's completion
marker AND both DS probe jsonls holding rows 0 and 5-90, both offline-curve
jsonls holding rows 1-4 (and the prep's vocab.json), then its preflight,
then FS launches. A RESUME (this run's checkpoint exists, or its
supervisor.log logged a LAUNCHED line -- or a NO-WAIT line while TKX has
its completion marker or ABANDONED file) skips the wait. --no-wait (ds only; logged, Kai's call: TKX abandoned or
stuck) skips it too and is refused (unless this run was already
launched: then it just resumes) while any supervise_rgrowx.py,
post_rgrowx.py or trainer lives, and unless TKX has its completion marker
or its ABANDONED file; the master is then TK-90 when TKX has no marker, and the
prep stops (PREP FAILED) when TKX completed without rows 240-270. Before
EVERY launch attempt (and before the prep) it waits while ANY python runs
experiments/ablation.py or experiments/distill.py. One supervisor per run: a second
supervise_distill.py of the same variant refuses to start. A run already
complete exits at once (RUN COMPLETE). ABANDONED in the run folder stops
every launch; a run folder holding snapshots (its own or its offline
curve's) or probe rows but no checkpoint is refused (move them aside).

Audit r1 (B): the prep (experiments/distill.py) counts as a trainer for
the any-trainer guard, so an orphaned prep (its supervisor killed) blocks a
second one; a prep is never redone once this run was launched (a
checkpoint, or a LAUNCHED line; r2: nor started at all) -- the integrity
gate then ALERTs and waits; before that a transient read error is retried before the files are
judged damaged, and damaged files are moved aside, not overwritten.
master.json is written atomically; a TKX that completed with rows 240-270
incomplete stops the prep (PREP FAILED: the rule needs them) instead of
falling back to TK-90; a tie keeps TK-90 (TKX must be strictly higher). fs
also waits for DS's offline-curve rows 1-4, and before every launch checks
that vocab.json is the vocabulary DS trained with (DS's completion marker).
Only ds, and only for weights-iter000 and the offline curve, may start over
its own pre-checkpoint snapshots. The prep runs under Popen with an hourly
ALERT while prep.log is silent past 30 min. ABANDONED is also checked while
waiting and before the prep. --no-wait needs TKX's completion marker or its
ABANDONED file. Missing master snapshots are an hourly ALERT while ds waits.
PAUSE (a Switch session): kill the supervisor, then its trainer
(experiments/ablation.py) or prep (experiments/distill.py), by command
line. RETRACTING a --no-wait = the fresh-restart procedure: stop the ds
supervisor, its trainer or prep and the hand-off, move the DS run folder
(its supervisor.log holds the NO-WAIT line) AND the prep folder (its
master.json holds the first choice) aside, relaunch. After a REBOOT rerun
BOTH launchers (launch_rgrowx.ps1 while TKX is unfinished, and
launch_distill.ps1).

Audit r2 (B, nothing above LOW; its LOWs): an orphaned prep that finishes
while the next attempt waits on the guard counts (no second prep); once DS
was LAUNCHED (a checkpoint or a LAUNCHED line) no prep is ever started
again, whatever is missing; a NO-WAIT line alone sanctions a skip of the
wait only while the --no-wait preconditions still hold; fs checks DS's
vocabulary in a child process (no torch in the long-lived supervisor);
3 consecutive long-lived attempts (>= 10 min) that die without writing a
new checkpoint stop the supervisor (NO PROGRESS: DS's attempts each cost
the ~1 h offline phase until its first checkpoint); stale prep tmp files are swept before a prep
attempt.

The runs execute from the RUN WORKTREE this file lives in (a checkout of
the audited commit that nobody edits), so REPO is this file's own tree.
Supervisor v2 pattern: a relaunch auto-resumes from the last checkpoint;
before EVERY launch attempt and the prep the tree's AUDITED files must hash
to the manifest written from the audited commit (distill_manifest.json)."""
import json
import os
import re
import subprocess
import sys
import time

TK_ARM = "effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-tokens-rgrow"
VARIANTS = {"ds": TK_ARM + "-student-distill", "fs": TK_ARM + "-student-fresh"}
RUNS = {"ds": "mspacman-student-distill", "fs": "mspacman-student-fresh"}
ITERATIONS = 90
TRAIN_STEPS = 2000
# Row checks (ALERTs): DS's ceiling is the master's end state, 160 at every
# row; FS's is TK's schedule over 90 iterations (1 -> 160 log-linear by half
# of 90 x 2000 steps, from the iteration's starting step).
CEILING, CEILING_END_STEPS = 160.0, 90_000
PACE_ALERT_MIN = 25.0
REWARD_MAX_ALERT = 400.0
COMMIT_ALERT_GB = 4.0
PRED_EVALS_ALERT_S = 2 * 3600    # predecessor done this long, evals missing
PRED_SILENT_ALERT_S = 4 * 3600   # predecessor unfinished, its log silent
PREP_ATTEMPTS = 3
PREP_RETRY_S = 900
NO_PROGRESS_DEATHS = 3


def expected_ceiling(variant, iteration):
    """The reward ceiling row `iteration` of `variant` must log."""
    if variant == "ds":
        return CEILING
    return CEILING ** min(1.0, (iteration - 1) * TRAIN_STEPS
                          / CEILING_END_STEPS)


MODE = ["--ray", "--cpu-actor"]
ENV_ID = "MsPacmanNoFrameskip-v4"
TAG = "-mspacmannoframeskip-skip4"        # ablation's TAG for ENV_ID, skip 4
SEEDS = {0: "2026-10-21"}                 # the pre-registered seed and date
REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
PY = os.environ.get("RGROW_PY", r"Y:\SwitchPlayer\.venv\Scripts\python.exe")
MAX_ATTEMPTS = 40
MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "distill_manifest.json")
# The prep's folder (REPO-relative): set.pkl, vocab.json (written last: the
# prep's completion signal), master.json, prep.log.
PREP_REL = "benchmarks/results/distill-prep-s0-2026-10-21"


def prep_dir():
    return os.path.join(REPO, *PREP_REL.split("/"))


# The masters: TKX (live in its worktree until it completes) and TK-90.
SNAP_DIR = f"snapshots{TAG}-{TK_ARM}-seed0"
TKX_TREE = r"Y:\SwitchPlayer-rgrowx-run"
TKX_RUN = "mspacman-tokens-rgrow-x270-s0-2026-10-20"
TKX_MARKER = os.path.join(TKX_TREE, "benchmarks", "results", TKX_RUN,
                          f"resume{TAG}-{TK_ARM}-seed0.ckpt")
TKX_SUPLOG = os.path.join(TKX_TREE, "benchmarks", "results", TKX_RUN,
                          "supervisor.log")
TKX_PROBES = tuple(os.path.join(TKX_TREE, "benchmarks", "probes",
                                f"probe_exhaustive_eval-{TKX_RUN}{s}.jsonl")
                   for s in ("-acting", ""))
TKX_ROWS = tuple(range(5, 275, 5))
TKX_SNAPSHOT = os.path.join(TKX_TREE, "benchmarks", "results", TKX_RUN,
                            SNAP_DIR, "weights-iter270.pth")
TKX_ABANDONED = os.path.join(TKX_TREE, "benchmarks", "results", TKX_RUN,
                             "ABANDONED")
TK90_RUN = "mspacman-tokens-rgrow-s0-2026-10-19"
TK90_SNAPSHOT = os.path.join(r"Y:\SwitchPlayer-rgrow-run", "benchmarks",
                             "results", TK90_RUN, SNAP_DIR,
                             "weights-iter090.pth")


def tk90_probe():
    """TK-90's deployed probe rows, committed on the branch (this tree)."""
    return os.path.join(REPO, "benchmarks", "probes",
                        f"probe_exhaustive_eval-{TK90_RUN}-acting.jsonl")


DS_ROWS = (0,) + tuple(range(5, 95, 5))
DS_CURVE_ROWS = (1, 2, 3, 4)
PREP_SILENT_ALERT_S = 1800
SUITES = ["tests/test_supervise_distill.py", "tests/test_distill.py",
          "tests/test_rgrowx.py", "tests/test_rgrow.py",
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
           "experiments/distill.py",
           "pytest.ini", "tests/test_gumbelact8.py",
           "tests/test_reanalysis.py",
           "benchmarks/supervisors/supervise_bnrecal.py",
           "benchmarks/supervisors/supervise_gumbelact8.py",
           "benchmarks/supervisors/supervise_prioract2.py",
           "benchmarks/supervisors/supervise_searchact8.py",
           "benchmarks/supervisors/supervise_tempscale.py",
           "benchmarks/supervisors/supervise_tokens.py",
           "benchmarks/supervisors/supervise_mspacman.py",
           "benchmarks/supervisors/supervise_rgrow.py",
           "benchmarks/supervisors/supervise_rgrowx.py",
           "benchmarks/probes/probe_exhaustive_eval.py",
           "benchmarks/supervisors/supervise_distill.py"] + SUITES


def run_name(variant, seed):
    return f"{RUNS[variant]}-s{seed}-{SEEDS[seed]}"


def paths(variant, seed):
    """(REL, BASE, JSONL, CKPT, DEATHS) of `variant` at `seed`: one folder
    per run."""
    arm = VARIANTS[variant]
    rel = f"benchmarks/results/{run_name(variant, seed)}"
    base = os.path.join(REPO, *rel.split("/"))
    return (rel, base, os.path.join(base, f"ablation{TAG}.jsonl"),
            os.path.join(base, f"resume{TAG}-{arm}-seed{seed}.ckpt"),
            os.path.join(base, "deaths.log"))


def command(variant, seed):
    """The pre-registered recipe: TK's (supervise_rgrow.py tk) with the
    student vocabulary (both) and the distillation set (ds)."""
    vocab = PREP_REL + "/vocab.json"
    extra = ["--token-vocab", vocab]
    if variant == "ds":
        extra += ["--distill-set", PREP_REL + "/set.pkl"]
    return [PY, "experiments/ablation.py", "--arms", VARIANTS[variant],
            "--seed", str(seed),
            "--iterations", str(ITERATIONS), "--train-steps",
            str(TRAIN_STEPS), "--eval-every", "5",
            "--env", ENV_ID, "--frameskip", "4", "--episode-cap", "3000",
            "--episodes-per-iter", "3", "--checkpoint-every", "2",
            "--snapshot-weights"] + MODE + extra


def file_hashes(root):
    """sha256 of every AUDITED file's text, CRLF normalized to LF."""
    import hashlib
    out = {}
    for rel in AUDITED:
        with open(os.path.join(root, *rel.split("/")), "rb") as f:
            out[rel] = hashlib.sha256(
                f.read().replace(b"\r\n", b"\n")).hexdigest()
    return out


def manifest_check():
    """(ok, reason): this tree's AUDITED files hash to the manifest."""
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
    """Written from a CLEAN tree (the audited commit) only."""
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
    """The import probe for `variant`'s arm and the prep module."""
    arm = VARIANTS[variant]
    kind = "distill" if variant == "ds" else "fresh"
    return ("import sys; sys.path[:0] = ['.', 'experiments']; "
            "import muzero, ablation, muzero_vtree, distill; "
            f"spec = ablation.ARMS[{arm!r}]; "
            f"assert spec['_student'] == {kind!r}; "
            "assert spec['REWARD_GROWTH'] is True and "
            "spec['REWARD_GROWTH_CEILING'] == 160.0; "
            "assert spec['BN_RECAL_ACTING'] is True; "
            "assert spec['MAX_REPLAY'] == 255; "
            "assert len(spec['TOKEN_VOCAB']) == 36; "
            "assert hasattr(ablation, 'student_gate') and "
            "hasattr(ablation, 'distill_offline') and "
            "hasattr(distill, 'prep') and distill.DROP_SHARE == 0.01")


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


def probe_rows(path):
    """{iteration: exact_mean} of a probe jsonl's full-range Ms. Pac-Man
    records (dicts without "max_noop" whose env is ENV_ID)."""
    out = {}
    for line in lines_of(path):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if (isinstance(r, dict) and "max_noop" not in r
                and r.get("env") == ENV_ID
                and isinstance(r.get("iteration"), int)):
            out[r["iteration"]] = r.get("exact_mean")
    return out


def marker_missing(marker):
    try:
        if os.path.getsize(marker) >= 1_000_000:
            return ["completion marker (checkpoint still full size)"]
    except OSError:
        return ["completion marker (no checkpoint)"]
    return []


def predecessor(variant, seed):
    """What `variant` waits for: (name, marker, [(probe jsonl, rows)],
    supervisor.log, supervisor script, its variant arg or None, its
    ABANDONED file)."""
    if variant == "ds":
        return ("TKX", TKX_MARKER, [(p, TKX_ROWS) for p in TKX_PROBES],
                TKX_SUPLOG, "supervise_rgrowx.py", None, TKX_ABANDONED)
    _, base, _, ck, _ = paths("ds", seed)
    run = run_name("ds", seed)
    # DS's rows 0 and 5-90 and its offline curve's rows 1-4 (audit r1 B-L2:
    # FS must not start while DS's final pass still probes).
    probes = [(os.path.join(REPO, "benchmarks", "probes",
                            f"probe_exhaustive_eval-{name}{s}.jsonl"), rows)
              for name, rows in ((run, DS_ROWS),
                                 (run + "-offline", DS_CURVE_ROWS))
              for s in ("-acting", "")]
    return ("DS", ck, probes, os.path.join(base, "supervisor.log"),
            "supervise_distill.py", "ds", os.path.join(base, "ABANDONED"))


def pred_missing(variant, seed):
    """What the predecessor still lacks before `variant` may start: [] once
    its marker is there and its probe jsonls hold every row (fs: and the
    prep's vocab.json)."""
    name, marker, probes, _, _, _, _ = predecessor(variant, seed)
    out = marker_missing(marker)
    for path, rows in probes:
        got = probe_rows(path)
        lack = [i for i in rows if i not in got]
        if lack:
            out.append(f"{os.path.basename(path)} rows {lack[:8]}"
                       + (" ..." if len(lack) > 8 else ""))
    if variant == "fs" and not os.path.exists(os.path.join(prep_dir(),
                                                           "vocab.json")):
        out.append("the prep's vocab.json")
    return out


def pred_marker_age(variant, seed):
    marker = predecessor(variant, seed)[1]
    try:
        if os.path.getsize(marker) < 1_000_000:
            return time.time() - os.path.getmtime(marker)
    except OSError:
        pass
    return None


def pred_unattended(variant, seed):
    """None while the predecessor is finished or driven; else why not."""
    name, marker, _, suplog, script, pvar, flag = predecessor(variant, seed)
    if not marker_missing(marker):
        return None
    if os.path.exists(flag):
        # (audit r1 B-L8: rerunning its launcher would not help)
        return (f"{name} is ABANDONED (its run folder's flag) -- no "
                f"verdict needing it runs; Kai's call")
    alive = live_scripts(script, pvar)
    if not alive:
        return f"no live {script}{' ' + pvar if pvar else ''} -- rerun its launcher"
    try:
        quiet = time.time() - os.path.getmtime(suplog)
    except OSError:
        return None
    if quiet > PRED_SILENT_ALERT_S:
        return (f"{name}'s supervisor (pids {alive}) is alive but its log is "
                f"silent for {quiet / 3600:.1f} h -- check it; do NOT rerun "
                f"its launcher beside it")
    return None


def pred_gave_up(variant, seed):
    """The predecessor's supervisor logged CRASH LOOP / MAX ATTEMPTS / PREP
    FAILED / NO PROGRESS after its last launch or preflight line."""
    lines = lines_of(predecessor(variant, seed)[3])
    fail = max((i for i, l in enumerate(lines) if l.startswith(
        ("CRASH LOOP", "MAX ATTEMPTS", "PREP FAILED", "NO PROGRESS"))),
               default=-1)
    live = max((i for i, l in enumerate(lines) if l.startswith(
        ("LAUNCHED", "preflight ok", "PREFLIGHT"))), default=-1)
    return fail > live


def wait_external(variant, seed):
    """Block until the predecessor has completed AND its evals have landed
    (returns True), or this run's ABANDONED file appears (returns False);
    hourly WAITING lines and ALERTs."""
    name = predecessor(variant, seed)[0]
    base = paths(variant, seed)[1]
    first, polls = True, 0
    while True:
        if abandoned(base):              # (audit r1 B-L9)
            return False
        missing = pred_missing(variant, seed)
        if not missing:
            if not first:
                print(f"{name} complete and evaluated; {variant} s{seed} may "
                      f"start", flush=True)
            return True
        if first or polls % 60 == 0:
            print(f"WAITING for {name} before {variant} s{seed} "
                  f"[{polls // 60} h]: missing " + "; ".join(missing)[:400],
                  flush=True)
            if pred_gave_up(variant, seed):
                print(f"ALERT: {name}'s supervisor gave up -- {variant} keeps "
                      f"waiting", flush=True)
            why = pred_unattended(variant, seed)
            if why:
                print(f"ALERT: {name} is unfinished and nothing drives it: "
                      f"{why}" + ("; or start ds with --no-wait (Kai's call)"
                                  if variant == "ds" else ""), flush=True)
            age = pred_marker_age(variant, seed)
            if age is not None and age > PRED_EVALS_ALERT_S:
                print(f"ALERT: {name} completed {age / 3600:.1f} h ago but its "
                      f"evals still lack rows -- is its hand-off alive? (a "
                      f"hand-off log line '... rows missing' means rerun "
                      f"them)", flush=True)
            if variant == "ds":
                gone = master_files_missing()
                if gone:
                    print(f"ALERT: master snapshot(s) missing: "
                          f"{'; '.join(gone)} -- keep both run trees",
                          flush=True)
        first = False
        time.sleep(60)
        polls += 1


def window_mean(rows, lo, hi):
    """The mean of rows lo..hi (step 5) when every one is there, else None."""
    want = range(lo, hi + 1, 5)
    if any(not isinstance(rows.get(i), (int, float)) for i in want):
        return None
    return sum(rows[i] for i in want) / len(want)


def master_files_missing():
    """The master snapshots that should exist and do not: TK-90's always,
    TKX's once TKX has its completion marker (audit r1 B-L10)."""
    want = [TK90_SNAPSHOT] + ([] if marker_missing(TKX_MARKER)
                              else [TKX_SNAPSHOT])
    return [p for p in want if not os.path.exists(p)]


def choose_master():
    """The pre-registered master: the final snapshot of whichever of TKX
    (deployed rows 240-270) and TK-90 (rows 60-90) has the higher 7-row
    deployed mean (TKX only when strictly higher); TK-90 when TKX has no
    completion marker (abandoned / --no-wait). A TKX that completed with
    its rows 240-270 incomplete raises: the rule needs them (audit r1
    B-L7). Returns a dict (snapshot, iterations, which, both means)."""
    tk90 = window_mean(probe_rows(tk90_probe()), 60, 90)
    if tk90 is None:
        raise SystemExit(f"TK-90's deployed rows 60-90 are missing from "
                         f"{tk90_probe()}")
    tkx = None
    if not marker_missing(TKX_MARKER):
        tkx = window_mean(probe_rows(TKX_PROBES[0]), 240, 270)
        if tkx is None:
            raise SystemExit("TKX completed but its deployed rows 240-270 "
                             "are incomplete -- land them (its hand-off) "
                             "first; the pre-registered rule needs them")
    if tkx is not None and tkx > tk90:
        pick = {"which": "TKX", "snapshot": TKX_SNAPSHOT, "iterations": 270}
    else:
        pick = {"which": "TK-90", "snapshot": TK90_SNAPSHOT, "iterations": 90}
    return {**pick, "tkx_mean_240_270": tkx, "tk90_mean_60_90": tk90}


def prep_done():
    return os.path.exists(os.path.join(prep_dir(), "vocab.json"))


def read_pick(path):
    """master.json's choice, or None when it is unreadable / incomplete."""
    try:
        with open(path, encoding="utf-8") as f:
            pick = json.load(f)
    except (OSError, ValueError):
        return None
    keys = ("which", "snapshot", "iterations", "tkx_mean_240_270",
            "tk90_mean_60_90")
    if not isinstance(pick, dict) or any(k not in pick for k in keys):
        return None
    return pick


def write_pick(path, pick):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(pick, f, indent=1)
    os.replace(tmp, path)


def set_intact():
    """(ok, reason): vocab.json's set_sha256 is set.pkl's."""
    try:
        with open(os.path.join(prep_dir(), "vocab.json"), encoding="utf-8") as f:
            doc = json.load(f)
        import hashlib
        h = hashlib.sha256()
        with open(os.path.join(prep_dir(), "set.pkl"), "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
    except (OSError, ValueError) as e:
        return False, f"prep files: {e}"
    if doc.get("set_sha256") != h.hexdigest():
        return False, "set.pkl does not match vocab.json's set_sha256"
    return True, ""


STALE_PICK = ("PREP FAILED: master.json was chosen before TKX completed -- "
              "follow the --no-wait retraction procedure (move the DS run "
              "folder and the prep folder aside)")


def stale_pick():
    """master.json was chosen while TKX had no completion marker (a
    --no-wait: TK-90 by rule), and TKX has completed since -- the
    pre-registered rule must then see TKX's rows (a half-done retraction;
    audit r3 LOW-2)."""
    path = os.path.join(prep_dir(), "master.json")
    pick = read_pick(path) if os.path.exists(path) else None
    return (pick is not None and pick["tkx_mean_240_270"] is None
            and not marker_missing(TKX_MARKER))


def run_prep(env, redo_ok=True):
    """The prep, once (vocab.json is its completion signal): choose and log
    the master, then experiments/distill.py up to PREP_ATTEMPTS times.
    Returns True when the prep is done. A done prep that does not check out
    intact is retried for a few minutes (a transient lock), then -- only
    when `redo_ok` (this run was never LAUNCHED: nothing is bound to the
    set yet) -- moved aside and redone; once launched it is NEVER redone
    (a launched DS's checkpoint is bound to the set's sha256 and FS to its
    vocabulary; the attempt loop's integrity gate ALERTs and waits; audit
    r1 B-M2). Before a launch, a master.json chosen while TKX had no marker
    stops the prep once TKX has completed -- also over a finished prep, and
    again after every guard wait (audit r3 LOW-2, r4 LOW-1 / NIT-1)."""
    if redo_ok and stale_pick():
        print(STALE_PICK, flush=True)
        return False
    if not redo_ok and not prep_done():
        # (audit r2 B-L2: nothing missing is ever re-recorded once DS ran)
        print("ALERT: the prep's vocab.json is missing after this run was "
              "launched -- never redone now; the launch waits for it",
              flush=True)
        return True
    if prep_done():
        pick_bad = False
        for k in range(3):
            ok, why = set_intact()
            # (audit r5 NIT: a half-retraction that moved only master.json
            # aside -- the set's master is unknown; read in the same retry
            # loop, so a transient lock is not fatal: r6 LOW)
            pick_bad = ok and redo_ok and read_pick(os.path.join(
                prep_dir(), "master.json")) is None
            if ok and not pick_bad:
                return True
            if k < 2:
                time.sleep(60)
        if pick_bad:
            print("PREP FAILED: a finished prep without a readable "
                  "master.json -- follow the retraction procedure (move the "
                  "prep folder aside)", flush=True)
            return False
        if not redo_ok:
            print(f"ALERT: PREP not intact ({why}) after this run started -- "
                  f"never redone now; the launch waits for it", flush=True)
            return True
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for name in ("vocab.json", "set.pkl"):
            src = os.path.join(prep_dir(), name)
            if os.path.exists(src):
                os.replace(src, f"{src}.damaged-{stamp}")
        print(f"PREP damaged ({why}) -- moved aside (.damaged-{stamp}), "
              f"redoing it", flush=True)
    os.makedirs(prep_dir(), exist_ok=True)
    pick_path = os.path.join(prep_dir(), "master.json")
    # A restarted supervisor keeps the first choice (written atomically;
    # an unreadable one is chosen again -- nothing ran on it yet).
    pick = read_pick(pick_path) if os.path.exists(pick_path) else None
    if pick is None:
        try:
            pick = choose_master()
        except SystemExit as e:
            print(f"PREP FAILED: {e}", flush=True)
            return False
        write_pick(pick_path, pick)
    print(f"MASTER {pick['which']}: {pick['snapshot']} (TKX 240-270 "
          f"{pick['tkx_mean_240_270']}, TK-90 60-90 "
          f"{pick['tk90_mean_60_90']})", flush=True)
    for attempt in range(1, PREP_ATTEMPTS + 1):
        wait_no_trainer()
        if stale_pick():                 # TKX completed during the wait
            print(STALE_PICK, flush=True)
            return False
        # An orphaned prep may have finished while the guard waited (audit
        # r2 B-L1): it counts. With no prep alive, its tmp files are stale.
        if prep_done() and set_intact()[0]:
            print(f"PREP done (by an earlier prep): {PREP_REL}", flush=True)
            return True
        for name in os.listdir(prep_dir()):
            if name.endswith(".tmp"):
                try:
                    os.remove(os.path.join(prep_dir(), name))
                except OSError:
                    pass
        ok, why = manifest_check()
        if not ok:
            print(f"PREP FAILED: {why}", flush=True)
            return False
        plog = os.path.join(prep_dir(), "prep.log")
        with open(plog, "ab") as log:
            p = subprocess.Popen(
                [PY, "experiments/distill.py", "--master", pick["snapshot"],
                 "--master-iterations", str(pick["iterations"]),
                 "--out", PREP_REL, "--seed", "0"],
                cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
                creationflags=0x08000000)
            born, told = time.time(), 0.0
            while p.poll() is None:
                try:
                    quiet = time.time() - max(os.path.getmtime(plog), born)
                except OSError:
                    quiet = 0.0
                if (quiet > PREP_SILENT_ALERT_S
                        and time.time() - told > 3600):
                    print(f"ALERT: the prep (pid {p.pid}) has been silent for "
                          f"{quiet / 60:.0f} min -- hung? (audit r1 B-L5)",
                          flush=True)
                    told = time.time()
                time.sleep(30)
        r = p
        if r.returncode == 0 and prep_done():
            ok, why = set_intact()
            if ok:
                print(f"PREP done (attempt {attempt}): {PREP_REL}", flush=True)
                return True
            print(f"PREP attempt {attempt}: {why}", flush=True)
        else:
            print(f"PREP attempt {attempt} failed (rc={r.returncode}; see "
                  f"{PREP_REL}/prep.log)", flush=True)
        if attempt < PREP_ATTEMPTS:
            time.sleep(PREP_RETRY_S)          # (audit r2 B-L5)
    print(f"PREP FAILED after {PREP_ATTEMPTS} attempts -- see "
          f"{PREP_REL}/prep.log", flush=True)
    return False


VOCAB_CHECK = (
    "import json, sys, torch; "
    "want = json.load(open(sys.argv[1], encoding='utf-8'))['vocab']; "
    "m = torch.load(sys.argv[2], map_location='cpu', weights_only=False); "
    "have = m.get('token_vocab') if isinstance(m, dict) else None; "
    "ok = have is not None and "
    "[list(t) for t in have] == [list(t) for t in want]; "
    "print('VOCAB OK' if ok else 'VOCAB MISMATCH')")


def vocab_is_ds(seed):
    """(ok, reason): vocab.json's vocabulary is the one DS trained with (the
    token_vocab its completion marker's recipe holds; audit r1 B-L1) --
    checked in a child process, so the long-lived fs supervisor never holds
    torch's ~1.3 GB of commit (audit r2 B-L4)."""
    try:
        r = subprocess.run(
            [PY, "-c", VOCAB_CHECK, os.path.join(prep_dir(), "vocab.json"),
             paths("ds", seed)[3]], cwd=REPO, capture_output=True, text=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}, timeout=900)
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, f"vocab check: {type(e).__name__}: {e}"
    out = r.stdout.strip().splitlines()
    if r.returncode or not out:
        return False, ("vocab check: rc=%s %s" % (r.returncode, (
            r.stderr.strip().splitlines() or ["?"])[-1]))
    if out[-1] != "VOCAB OK":
        return False, ("vocab.json is not the vocabulary DS trained with "
                       "(DS's completion marker)")
    return True, ""


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
        clean = [re.sub(r" at 0x[0-9a-fA-F]+", "", re.sub(
            r" in [0-9.]+s( \(\d+:\d\d:\d\d\))?", "", l)) for l in lines[-2:]]
        return False, f"tests failed (rc={r.returncode}): " + " | ".join(
            [re.sub(r" at 0x[0-9a-fA-F]+", "", l) for l in bad] or clean)
    return True, ""


def _procs():
    import psutil
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name = (p.info.get("name") or "").lower()
            cmd = list(p.info["cmdline"] or [])
        except (psutil.Error, KeyError, TypeError):
            continue
        if name.startswith("python"):
            yield p.info["pid"], cmd


def _names(cmd):
    return [os.path.basename(c.replace(chr(92), "/")) for c in cmd]


def live_trainers():
    """PIDs of ANY python running experiments/ablation.py (any tree, arm or
    seed) or experiments/distill.py (the prep: an orphan of a killed
    supervisor keeps the GPU and writes the prep's files; audit r1
    B-M1)."""
    return [pid for pid, cmd in _procs()
            if any(c.replace(chr(92), "/").endswith(
                ("experiments/ablation.py", "experiments/distill.py"))
                for c in cmd)]


def live_scripts(script, variant=None):
    """PIDs of python processes running `script` (exact basename), with
    `variant` as its first argument when given."""
    out = []
    for pid, cmd in _procs():
        names = _names(cmd)
        if script not in names:
            continue
        if variant is not None:
            i = names.index(script)
            if cmd[i + 1:i + 2] != [variant]:
                continue
        out.append(pid)
    return out


def _mine():
    import psutil
    out = {os.getpid()}
    try:
        out.update(a.pid for a in psutil.Process().parents())
    except psutil.Error:
        pass
    return out


def other_supervisors(variant):
    """PIDs of OTHER python processes running supervise_distill.py
    `variant` (never this process or its ancestors)."""
    mine = _mine()
    return [p for p in live_scripts("supervise_distill.py", variant)
            if p not in mine]


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


class Rows:
    """Stream new jsonl rows as EVAL lines (the run's in-run eval, the
    reward fields and the token diagnostics)."""

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
            for alert in row_alerts(self.variant, r, per):
                print(f"ALERT: {self.variant} s{self.seed} row "
                      f"{r.get('iteration')}: {alert}", flush=True)
            free = commit_free_gb()
            if free is not None and free < COMMIT_ALERT_GB:
                print(f"ALERT: {self.variant} s{self.seed} row "
                      f"{r.get('iteration')}: host commit free {free:.1f} GB "
                      f"< {COMMIT_ALERT_GB:g} GB -- memory creep?", flush=True)


def row_alerts(variant, r, per):
    """A ceiling that is not the variant's schedule, a pace past
    PACE_ALERT_MIN min/iteration (Kai's call), a stored record reward past
    REWARD_MAX_ALERT units."""
    out = []
    it, c = r.get("iteration"), r.get("reward_ceiling")
    if isinstance(it, int) and it >= 1:
        want = expected_ceiling(variant, it)
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
    """Block while ANY python runs experiments/ablation.py or
    experiments/distill.py (live_trainers); `rows` keep streaming
    meanwhile."""
    polls, told = 0, None
    while True:
        if rows is not None:
            rows.emit()
        pids = live_trainers()
        if not pids:
            return
        msg = (f"ALERT: an experiments/ablation.py or distill.py is already "
               f"running (pids {pids}) -- one run on the GPU at a time, and "
               f"never a second writer")
        if msg != told or polls % 60 == 0:
            print(f"{msg} [{polls} min]", flush=True)
            told = msg
        time.sleep(60)
        polls += 1


def abandoned(base):
    """Kai's call: an ABANDONED file in the run folder stops every launch."""
    return os.path.exists(os.path.join(base, "ABANDONED"))


def launched(base, ckpt):
    """This run's trainer was launched before: its checkpoint exists, or its
    own supervisor.log logged a LAUNCHED line."""
    if os.path.exists(ckpt):
        return True
    return any(l.startswith("LAUNCHED")
               for l in lines_of(os.path.join(base, "supervisor.log")))


def tkx_unsettled():
    """TKX has neither its completion marker nor its ABANDONED file."""
    return bool(marker_missing(TKX_MARKER)) and not os.path.exists(
        TKX_ABANDONED)


def no_wait_refusal():
    """Why ds's --no-wait may not (any longer) skip the wait, or None: a
    live TKX supervisor or hand-off, or a TKX with neither its completion
    marker nor an ABANDONED file (audit r1 B-L11)."""
    for script in ("supervise_rgrowx.py", "post_rgrowx.py"):
        alive = live_scripts(script)
        if alive:
            return f"{script} is alive (pids {alive}) -- kill it first"
    alive = live_trainers()
    if alive:
        return (f"a trainer or prep is alive (pids {alive}) -- stop it first "
                f"(audit r4 NIT-1)")
    if tkx_unsettled():
        return ("TKX has neither its completion marker nor an ABANDONED "
                "file -- abandon it first (its supervisor's procedure)")
    return None


def sanctioned(base, ckpt):
    """This run already started once: its checkpoint exists, or its own
    supervisor.log logged a launch or the --no-wait override."""
    if os.path.exists(ckpt):
        return True
    return any(l.startswith(("LAUNCHED", "NO-WAIT override"))
               for l in lines_of(os.path.join(base, "supervisor.log")))


def stray_snapshots(base, ckpt, own_ok=False):
    """Snapshots in the run folder -- its own or its offline curve's -- with
    no checkpoint beside them (a fresh start there would mix two
    timelines). `own_ok` (ds, sanctioned: its own death before its first
    checkpoint) excuses exactly what that leaves -- weights-iter000 and the
    offline curve, which the restart rewrites (audit r1 B-L4)."""
    import glob
    if os.path.exists(ckpt):
        return []
    found = sorted(glob.glob(os.path.join(base, "snapshots*", "*.pth"))
                   + glob.glob(os.path.join(base, "*-offline", "snapshots*",
                                            "*.pth")))
    if own_ok:
        found = [f for f in found
                 if os.path.basename(f) != "weights-iter000.pth"
                 and not os.path.basename(os.path.dirname(os.path.dirname(
                     f))).endswith("-offline")]
    return found


def stray_probe_rows(variant, seed, ckpt):
    """This run's probe jsonls (and its offline curve's) holding records
    while no checkpoint exists."""
    if os.path.exists(ckpt):
        return []
    run = run_name(variant, seed)
    out = []
    for name in (run, run + "-offline"):
        for suffix in ("-acting", ""):
            path = os.path.join(REPO, "benchmarks", "probes",
                                f"probe_exhaustive_eval-{name}{suffix}.jsonl")
            if any(l.strip() for l in lines_of(path)):
                out.append(os.path.basename(path))
    return out


def main(argv):
    try:
        sys.stdout.reconfigure(errors="backslashreplace")
    except (AttributeError, ValueError):
        pass
    if "--write-manifest" in argv:
        return write_manifest()
    no_wait = argv[2:] == ["--no-wait"]
    if (len(argv) != (3 if no_wait else 2) or argv[0] not in VARIANTS
            or not argv[1].isdigit() or int(argv[1]) not in SEEDS
            or (no_wait and argv[0] != "ds")):
        print(f"usage: supervise_distill.py <variant: {sorted(VARIANTS)}> "
              f"<seed: {sorted(SEEDS)}> [--no-wait: ds only] | "
              f"--write-manifest", flush=True)
        return 2
    variant, seed = argv[0], int(argv[1])
    arm = VARIANTS[variant]
    rel, base, jsonl, ckpt, deaths_log = paths(variant, seed)
    others = other_supervisors(variant)
    if others:
        print(f"refusing: another supervise_distill.py {variant} is alive "
              f"(pids {others}) -- one supervisor per run", flush=True)
        return 2
    if run_done(variant, seed):
        print(f"RUN COMPLETE {variant} s{seed} (checkpoint is a marker)",
              flush=True)
        return 0
    if abandoned(base):
        print(f"ABANDONED {variant} s{seed}: the run folder holds an "
              f"ABANDONED file -- not launching", flush=True)
        return 0
    own_ok = variant == "ds" and sanctioned(base, ckpt)
    stray = stray_snapshots(base, ckpt, own_ok)
    if stray:
        # (A death of DS before its first checkpoint leaves its row 0 and
        # offline curve behind: its own log's LAUNCHED line sanctions the
        # restart, which rewrites exactly those.)
        print(f"refusing: {len(stray)} snapshots but no checkpoint in "
              f"{rel} -- a fresh start would mix two timelines; move the run "
              f"folder and its probe jsonls aside", flush=True)
        return 2
    stray = stray_probe_rows(variant, seed, ckpt)
    if stray:
        print(f"refusing: probe rows ({', '.join(stray)}) but no checkpoint "
              f"-- a fresh start would keep the old timeline's scores; move "
              f"them (and the run folder) aside", flush=True)
        return 2
    if launched(base, ckpt):
        # (before --no-wait: a launched run resumes without any wait, so a
        # -NoWait relaunch must not be refused over TKX's processes; audit r3
        # NIT-2)
        print(f"RESUMING {variant} s{seed}: it started before (checkpoint or "
              f"launch logged) -- no wait", flush=True)
    elif no_wait:
        why = no_wait_refusal()
        if why:
            print(f"refusing --no-wait: {why}", flush=True)
            return 2
        print(f"NO-WAIT override: {variant} s{seed} starts without TKX's "
              f"completion marker and evals", flush=True)
    elif sanctioned(base, ckpt) and not tkx_unsettled():
        # A logged NO-WAIT whose PERSISTENT precondition still holds (TKX's
        # marker or ABANDONED file; a briefly live TKX supervisor after a
        # reboot does not matter -- an ABANDONED TKX never relaunches; audit
        # r3 LOW-1).
        print(f"RESUMING {variant} s{seed}: its logged NO-WAIT override still "
              f"holds -- no wait", flush=True)
    else:
        if sanctioned(base, ckpt):
            # A NO-WAIT line while TKX has neither its marker nor ABANDONED
            # (a half-done retraction; audit r2 B-L3): wait like a fresh start.
            print(f"ALERT: a logged NO-WAIT override no longer holds (TKX "
                  f"has neither its completion marker nor an ABANDONED "
                  f"file) -- waiting for TKX", flush=True)
        if not wait_external(variant, seed):
            print(f"ABANDONED {variant} s{seed}: the run folder holds an "
                  f"ABANDONED file -- not launching", flush=True)
            return 0

    os.makedirs(base, exist_ok=True)
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
        rows.emit()
    print(f"preflight ok (manifest + quick probe + {len(SUITES)} suites); "
          f"starting {variant} s{seed} ({arm}, {ITERATIONS} iterations)",
          flush=True)

    env = {**{k: v for k, v in os.environ.items()
              if k != "CUDA_VISIBLE_DEVICES"},
           "ABLATION_RESULTS": rel, "PYTHONFAULTHANDLER": "1"}
    if variant == "ds":
        if abandoned(base):
            print(f"ABANDONED {variant} s{seed}: the run folder holds an "
                  f"ABANDONED file -- no prep, not launching", flush=True)
            return 0
        if not run_prep({k: v for k, v in env.items()
                         if k != "ABLATION_RESULTS"},
                        redo_ok=not launched(base, ckpt)):
            return 1
    stall_s = 2 * 3600
    deaths, quick_deaths, stalls = [], 0, 0

    def ckpt_stamp():
        try:
            return os.path.getmtime(ckpt)
        except OSError:
            return None

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
        polls, told = 0, None
        while True:
            wait_no_trainer(rows)
            if (variant == "ds" and not launched(base, ckpt)
                    and stale_pick()):
                print(STALE_PICK, flush=True)       # (audit r4 NIT-1)
                return 1
            ok, why = manifest_check()
            if ok and variant == "ds":
                ok, why = set_intact()
            if ok and variant == "fs":
                ok, why = vocab_is_ds(seed)
            if ok:
                break
            if why != told or polls % 60 == 0:
                print(f"ALERT: {why} -- not relaunching until the tree is the "
                      f"audited revision and the prep intact [{polls} min]",
                      flush=True)
                told = why
            time.sleep(60)
            polls += 1
        if completed():
            print(f"RUN COMPLETE {variant} s{seed} (checkpoint is a marker)",
                  flush=True)
            return 0
        if abandoned(base):
            print(f"ABANDONED {variant} s{seed}: the run folder holds an "
                  f"ABANDONED file -- not relaunching", flush=True)
            return 0
        stray = (stray_snapshots(base, ckpt, variant == "ds"
                                 and sanctioned(base, ckpt))
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
        born, stamp0 = time.time(), ckpt_stamp()
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
        # Long-lived deaths without a new checkpoint (quick ones are CRASH
        # LOOP's): DS's each cost the ~1 h offline phase (audit r2 B-L6).
        if ckpt_stamp() != stamp0:
            stalls = 0
        elif lifetime >= 600:
            stalls += 1
        if stalls >= NO_PROGRESS_DEATHS:
            print(f"NO PROGRESS: {stalls} consecutive attempts died without "
                  f"writing a new checkpoint -- stopping for diagnosis "
                  f"(audit r2 B-L6)", flush=True)
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
