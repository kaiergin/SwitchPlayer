"""The sharper prior-acting temperature arm (Kai, 2026-10-04):
effzero-repro-faithful-prioract-noredo-temp05, one seed, --ray --cpu-actor
(pinned by the arm). Design, trigger and pre-registered reads: CURRICULUM.md
"DIRECTION + next experiment (pre-registered) -- sharper prior-acting
temperature". Waits for prioract2 seed 1 to complete AND for its exact
evals (rows 45-90), then ENFORCES the pre-registration: exit 3 if seed 1
reads WORSE (the DIRECTION proviso: revisit with Kai), exit 2 if its first
success is at iteration <= 65 (trigger not fired: SVE next), exit 4 on a
malformed record; launches only if seed 1 is not WORSE and its first
success is at iteration >= 70 or never:

    python benchmarks/supervisors/supervise_tempscale.py ray 0

Supervisor v2 pattern; a relaunch auto-resumes from the last checkpoint."""
import json
import math
import os
import re
import subprocess
import sys
import time

VARIANTS = {
    "ray": ("effzero-repro-faithful-prioract-noredo-temp05", "pong-tempscale",
            ["--ray", "--cpu-actor"]),
}
WRITE = "--write-manifest" in sys.argv
VARIANT = "ray" if WRITE else sys.argv[1]
SEED = 0 if WRITE else int(sys.argv[2])
REPO = r"Y:\SwitchPlayer"
ARM, _stem, _mode = VARIANTS[VARIANT]
REL = f"benchmarks/results/{_stem}-s{SEED}-2026-10-05"
WAIT_CKPTS = [os.path.join(
    REPO, "benchmarks", "results", "pong-prioract2-s1-2026-10-02",
    "resume-pong-skip4-effzero-repro-faithful-prioract-noredo-seed1.ckpt")]
# The pre-registered trigger (CURRICULUM, 2026-10-04): seed 1's exact evals.
PROBES = os.path.join(REPO, "benchmarks", "probes")
TRIGGER_RUN = "pong-prioract2-s1-2026-10-02"
TRIGGER_ROWS = tuple(range(45, 91, 5))
# Written by probe_exhaustive_eval.py run from the MAIN tree's probes dir.
TRIGGER_FILE = os.path.join(PROBES, f"probe_exhaustive_eval-{TRIGGER_RUN}.jsonl")
# The DIRECTION proviso (same CURRICULUM section, round-1 audit M1): if seed 1
# reads WORSE -- mean exact over rows 60-90 below the faithful control's
# WORSE edge -- revisit with Kai before building further on this stack.
MEAN_ROWS = tuple(range(60, 91, 5))
WORSE_BELOW = -2.32
BASE = os.path.join(REPO, *REL.split("/"))
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
JSONL = os.path.join(BASE, "ablation-pong-skip4.jsonl")
CKPT = os.path.join(BASE, f"resume-pong-skip4-{ARM}-seed{SEED}.ckpt")
DEATHS = os.path.join(BASE, "deaths.log")
COMMON = [PY, "experiments/ablation.py", "--arms", ARM, "--seed", str(SEED),
          "--iterations", "90", "--train-steps", "2000", "--eval-every", "5",
          "--env", "Pong-v4", "--frameskip", "4", "--episode-cap", "3000",
          "--episodes-per-iter", "1", "--checkpoint-every", "2",
          "--snapshot-weights"] + _mode
MAX_ATTEMPTS = 40

seen = set()


def rows():
    try:
        with open(JSONL, "rb") as f:
            return [l for l in f.read().decode("utf-8", "replace").splitlines()
                    if l.strip()]
    except OSError:
        return []


def emit_new_rows():
    for line in rows():
        key = line.strip()
        if key in seen:
            continue
        seen.add(key)
        try:
            r = json.loads(line)
            print("EVAL s%d iter %s  train %s  H(pi) %.3f  maxp %.3f  freshH %s  "
                  "erank %s  dead %s  return %.1f" % (
                      SEED, r.get("iteration"), r.get("train_steps"),
                      r.get("policy_entropy", float("nan")),
                      r.get("policy_maxp", float("nan")),
                      ("%.3f" % r["fresh_target_entropy"]
                       if isinstance(r.get("fresh_target_entropy"), float) else "n/a"),
                      r.get("proj_erank", "n/a"), r.get("proj_dead_frac", "n/a"),
                      r.get("return_mean", float("nan"))), flush=True)
            el = r.get("elapsed_s")
            per = ("%.1f" % ((el - emit_new_rows.last[0]) / 60.0 /
                             max(1, r.get("iteration", 0) - emit_new_rows.last[1]))
                   if isinstance(el, (int, float)) and emit_new_rows.last
                   and r.get("iteration", 0) > emit_new_rows.last[1] else "n/a")
            if isinstance(el, (int, float)):
                emit_new_rows.last = (el, r.get("iteration", 0))
            print("      env %s  len %s  capped %s | redo window %s %s (total %s)  "
                  "gn pre/post %s/%s  step gn %s loss %s | elapsed %s s, "
                  "%s min/iter since last row" % (
                      r.get("env_steps"), r.get("eval_len_mean"),
                      r.get("eval_capped"), r.get("redo_recycled_window", "n/a"),
                      r.get("redo_by_net_window", ""),
                      r.get("redo_recycled_total", "n/a"),
                      r.get("redo_gn_pre20", "n/a"), r.get("redo_gn_post20", "n/a"),
                      r.get("step_grad_norm_mean", "n/a"),
                      r.get("step_loss_mean", "n/a"), el, per), flush=True)
            print("      behaviour T %s  H_act %s  maxp %s  act_agree %s  "
                  "prior_agree %s  KL(tgt||prior) %s  rare %s | birth window "
                  "%s pos | eager [it, step] %s" % (
                      r.get("beh_T"), r.get("beh_H_act"), r.get("beh_prior_maxp"),
                      r.get("beh_act_agree"), r.get("beh_prior_agree"),
                      r.get("beh_kl_target_prior"), r.get("beh_rare_rate"),
                      r.get("birth_positions_window", "n/a"),
                      r.get("eager_collect_steps", "n/a")), flush=True)
        except Exception:
            pass


emit_new_rows.last = None     # (elapsed_s, iteration) of the previous row


def completed():
    try:
        return os.path.getsize(CKPT) < 1_000_000
    except OSError:
        return False


def snapshot(attempt, rc, lifetime):
    """Append the death-time environment to deaths.log; one-line summary
    to stdout for the event stream."""
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"=== death attempt={attempt} rc={rc} lifetime={lifetime:.0f}s "
             f"at {stamp}"]
    for cmd in (["nvidia-smi", "--query-gpu=memory.used,memory.total,"
                 "utilization.gpu,temperature.gpu", "--format=csv,noheader"],
                ["nvidia-smi", "--query-compute-apps=pid,process_name,"
                 "used_memory", "--format=csv,noheader"],
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
    with open(DEATHS, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    gpu = lines[1].split(",")[0].strip() if len(lines) > 1 else "?"
    print(f"DEATH s{SEED} rc={rc} after {lifetime:.0f}s; snapshot -> deaths.log "
          f"(gpu used {gpu})", flush=True)


def _glued_record(line):
    """The probe appends json + "\n" with no leading newline, so a re-run
    after a killed write glues its complete record onto the torn fragment
    (round-2 L1): return the first decodable dict that starts after
    position 0 and runs to the end of the line, else None. A '{"' inside
    the fragment (a nested dict) fails to decode to the end of the line, so
    the scan moves on to the glued record's own start."""
    pos = line.find('{"', 1)
    while pos != -1:
        try:
            r = json.loads(line[pos:])
        except ValueError:
            pos = line.find('{"', pos + 1)
            continue
        return r if isinstance(r, dict) else None
    return None


def read_records(path, run=TRIGGER_RUN):
    """({iteration: record}, n_bad) for `run` from an exact-eval jsonl. The
    last record for a row wins (a re-run probe supersedes). An undecodable
    line (a torn write from a killed probe) is counted; a complete record
    a re-run glued onto it is recovered (_glued_record), so re-running the
    probe heals the row (round-1 L2, round-2 L1)."""
    recs, bad = {}, 0
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    bad += 1
                    r = _glued_record(line.rstrip("\n"))
                    if r is None:
                        continue
                if isinstance(r, dict) and r.get("run") == run:
                    recs[r.get("iteration")] = r
    except OSError:
        pass
    return recs, bad


def first_success(recs, rows=TRIGGER_ROWS):
    """The first row with exact >= +10 or >= 3/4 phases won; None if no
    row qualifies. `recs` must hold every row (trigger_decision checks)."""
    for i in rows:
        if recs[i]["exact_mean"] >= 10 or recs[i]["phases_won"] >= 3:
            return i
    return None


def triggered(first):
    """RUN the arm iff seed 1 also needed T = 0.25: first success at
    iteration >= 70, or none."""
    return first is None or first >= 70


def _num(x):
    return (isinstance(x, (int, float)) and not isinstance(x, bool)
            and math.isfinite(x))


def trigger_decision(path, run=TRIGGER_RUN):
    """(code, message) from `run`'s exact-eval records:
      None -- keep waiting (a row 45-90 is missing);
      4    -- a needed record is malformed (no numeric exact_mean or
              integer phases_won): NOT launching, back to Kai;
      3    -- seed 1 reads WORSE (mean 60-90 < -2.32): the DIRECTION
              proviso -- revisit with Kai, NOT launching;
      2    -- the trigger does not fire (first success at <= 65): NOT
              launching, SVE on this stack is next;
      0    -- RUN."""
    recs, bad = read_records(path, run)
    missing = [i for i in TRIGGER_ROWS if i not in recs]
    if missing:
        return None, (f"WAITING for {run}'s exact evals: rows {missing} missing"
                      + (f" ({bad} torn/undecodable line(s); a complete record "
                         f"glued after one is recovered)" if bad else ""))
    for i in TRIGGER_ROWS:
        r = recs[i]
        if not (_num(r.get("exact_mean")) and isinstance(r.get("phases_won"), int)
                and not isinstance(r.get("phases_won"), bool)):
            return 4, (f"ALERT: malformed exact-eval record for {run} @{i} "
                       f"(needs exact_mean and phases_won) -- NOT launching; "
                       f"back to Kai")
    first = first_success(recs)
    mean = sum(recs[i]["exact_mean"] for i in MEAN_ROWS) / len(MEAN_ROWS)
    head = (f"trigger: {run} first success at "
            f"{first if first is not None else 'none'}, mean 60-90 {mean:+.4f}")
    if mean < WORSE_BELOW:
        return 3, (head + " -> seed 1 reads WORSE: the DIRECTION proviso says "
                   "revisit before building further on this stack -- NOT "
                   "launching; back to Kai")
    if not triggered(first):
        return 2, (head + " -> SKIP: seed 1 broke out at <= 65, the trigger "
                   "does not fire -- NOT launching; next is SVE on this stack")
    return 0, head + " -> RUN"


def _seed_state(ck):
    """'done' (completion marker -- a shrunk checkpoint -- or the seed's own
    supervisor logging RUN COMPLETE, if the checkpoint was moved), 'failed'
    (its supervisor gave up: CRASH LOOP / MAX ATTEMPTS), else 'running'."""
    try:
        if os.path.getsize(ck) < 1_000_000:
            return "done"
    except OSError:
        pass
    try:
        with open(os.path.join(os.path.dirname(ck), "supervisor.log"), "rb") as f:
            log = f.read()
    except OSError:
        return "running"
    if b"RUN COMPLETE" in log:
        return "done"
    if b"CRASH LOOP" in log or b"MAX ATTEMPTS" in log:
        return "failed"
    return "running"


# Files whose AUDITED content must be in the main tree (sha256 of the text
# with CRLF normalized to LF, so git's autocrlf round trip does not matter).
# Written by `supervise_tempscale.py --write-manifest` in the worktree when the
# audit loop closes; any later edit to one of them blocks the launch.
MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "tempscale_manifest.json")
# Every repo module the run executes (SVE audit r1 L4), plus the tests
# and this supervisor.
AUDITED = ["muzero.py", "redo.py", "muzero_vtree.py", "networks.py",
           "scalar_categorical.py", "memory.py", "EpisodicLifeEnv.py",
           "atari_wrappers.py", "experiments/ablation.py",
           "experiments/ray_actors.py", "tests/test_prioract.py",
           "tests/test_sve.py", "tests/test_searchact.py",
           "tests/test_factor.py", "tests/test_prioract2.py",
           "tests/test_tempscale.py", "tests/test_harness.py",
           "tests/test_ray_harness.py", "tests/test_checkpoint_resume.py",
           "tests/test_paper_mode.py", "tests/test_reanalyze_refresh.py",
           "tests/test_reanalysis.py", "tests/test_vtree.py", "pytest.ini",
           "benchmarks/probes/probe_exhaustive_eval.py",
           "benchmarks/supervisors/supervise_tempscale.py"]


def file_hashes(root):
    import hashlib
    out = {}
    for rel in AUDITED:
        with open(os.path.join(root, *rel.split("/")), "rb") as f:
            out[rel] = hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()
    return out


if "--write-manifest" in sys.argv:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(file_hashes(root), f, indent=1, sort_keys=True)
    print(f"wrote {MANIFEST} from {root}")
    sys.exit(0)


# The main tree must hold the AUDITED revision before anything launches (the
# change is merged there by hand). Otherwise argparse / import would kill
# every attempt and the supervisor would burn its crash-loop budget with the
# GPU idle -- or, worse, a stale revision would run.
QUICK = ("import sys; sys.path[:0] = ['.', 'experiments']; "
         "import muzero, ablation, redo; "
         f"assert {ARM!r} in ablation.ARMS and hasattr(muzero, 'BIRTH_SEARCH'); "
         "assert hasattr(redo, '_solve_bias') and hasattr(ablation, 'redo_due'); "
         "assert hasattr(muzero, 'EVAL_PRIOR_ONLY'); "
         "assert 'effzero-repro-faithful-prioract-noredo' in ablation.ARMS; "
         "assert hasattr(ablation, 'train_block') and "
         "hasattr(muzero, 'RAY_EAGER_COLLECT'); "
         "assert hasattr(muzero, 'PRIOR_ACT_TEMP_SCALE') and "
         "hasattr(ablation, 'prior_temp_gate')")


def manifest_check():
    """(ok, reason): the main tree's AUDITED files hash to the manifest.
    Run before EVERY launch attempt, not just the first (SVE audit r1 L4):
    a relaunch must never run an unaudited edit."""
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


def preflight():
    """(ok, reason). The audited-revision manifest, the quick import probe,
    then the prioract + SVE + searchact + factor + prioract2 + tempscale
    suites run IN the main tree."""
    ok, why = manifest_check()
    if not ok:
        return False, why
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}
    r = subprocess.run([PY, "-c", QUICK], cwd=REPO, capture_output=True,
                       text=True, env=env)
    if r.returncode:
        return False, "quick probe: " + (r.stderr.strip().splitlines() or ["?"])[-1]
    try:
        r = subprocess.run([PY, "-m", "pytest", "-x", "-q", "-p",
                            "no:cacheprovider", "tests/test_prioract.py",
                            "tests/test_sve.py", "tests/test_searchact.py",
                            "tests/test_factor.py", "tests/test_prioract2.py",
                            "tests/test_tempscale.py"],
                           cwd=REPO, capture_output=True, text=True, env=env,
                           timeout=1800)
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


waited, told = 0, None
while True:
    states = [_seed_state(c) for c in WAIT_CKPTS]
    if all(s == "done" for s in states):
        break
    msg = ("ALERT: prioract2 seed 1's supervisor gave up (%s) -- the trigger "
           "is undecidable; the choice goes back to Kai. Still waiting"
           % states if "failed" in states else
           "WAITING for prioract2 seed 1 to complete (%s)" % states)
    if msg != told or waited % 60 == 0:
        print(f"{msg} [{waited // 60} h]", flush=True)
        told = msg
    time.sleep(60)
    waited += 1
print("prioract2 seed 1 complete", flush=True)
# The pre-registration, enforced: wait for seed 1's exact evals (rows 45-90,
# probe_exhaustive_eval.py), decide, and refuse to launch unless it says RUN.
waited, told = 0, None
while True:
    code, msg = trigger_decision(TRIGGER_FILE)
    if code is not None:
        break
    if msg != told or waited % 60 == 0:
        print(f"{msg} [{waited // 60} h]", flush=True)
        told = msg
    time.sleep(60)
    waited += 1
print(msg, flush=True)
if code:
    sys.exit(code)
polls, told = 0, None
while True:
    ok, why = preflight()
    if ok:
        break
    if why != told or polls % 6 == 0:
        print(f"PREFLIGHT not ready ({polls * 10} min): {why}", flush=True)
        told = why
    time.sleep(600)
    polls += 1
print(f"preflight ok (manifest + quick probe + 6 suites); starting {VARIANT} "
      f"({ARM})", flush=True)

os.makedirs(BASE, exist_ok=True)
for line in rows():
    seen.add(line.strip())

# CUDA_VISIBLE_DEVICES is dropped: inherited from a shell that set -1 it
# would silently train on the CPU.
env = {**{k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"},
       "ABLATION_RESULTS": REL, "PYTHONFAULTHANDLER": "1"}
STALL_S = 2 * 3600    # the child logs at least every checkpoint (2 iters)
deaths = []
quick = 0
for attempt in range(1, MAX_ATTEMPTS + 1):
    if completed():
        print(f"RUN COMPLETE s{SEED} (checkpoint is a marker)", flush=True)
        sys.exit(0)
    polls, told = 0, None
    while True:
        ok, why = manifest_check()
        if ok:
            break
        if why != told or polls % 60 == 0:
            print(f"ALERT: {why} -- not relaunching until the main tree is "
                  f"the audited revision [{polls} min]", flush=True)
            told = why
        time.sleep(60)
        polls += 1
    args = COMMON + (["--resume"] if os.path.exists(CKPT) else [])
    log = open(os.path.join(BASE, f"supv2_{attempt}.log"), "ab")
    born = time.time()
    p = subprocess.Popen(args, cwd=REPO, env=env, stdout=log,
                         stderr=subprocess.STDOUT, creationflags=0x08000000)
    print(f"LAUNCHED s{SEED} attempt {attempt} "
          f"({'resume' if '--resume' in args else 'fresh'}), pid {p.pid}",
          flush=True)
    alerted = 0.0
    while p.poll() is None:
        emit_new_rows()
        try:
            idle = time.time() - max(os.path.getmtime(log.name), born)
        except OSError:
            idle = 0.0
        if idle > STALL_S and time.time() - alerted > 3600:
            print(f"ALERT: attempt {attempt} (pid {p.pid}) log silent for "
                  f"{idle / 3600:.1f} h -- possible hang", flush=True)
            alerted = time.time()
        time.sleep(30)
    log.close()
    emit_new_rows()
    rc = p.returncode
    lifetime = time.time() - born
    snapshot(attempt, rc, lifetime)
    if completed():
        print(f"RUN COMPLETE s{SEED}", flush=True)
        sys.exit(0)
    deaths.append(time.time())
    if len([t for t in deaths if time.time() - t < 3600]) >= 5:
        print("CRASH LOOP: 5 deaths within 60 min -- stopping for diagnosis",
              flush=True)
        sys.exit(1)
    if lifetime < 600:
        quick += 1
        wait = min(900, 60 * (2 ** quick))
        print(f"quick death #{quick}; backing off {wait}s", flush=True)
        time.sleep(wait)
    else:
        quick = 0
        time.sleep(15)
print("MAX ATTEMPTS REACHED", flush=True)
sys.exit(1)
