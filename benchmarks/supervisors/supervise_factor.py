"""The factorization after the prioract bisect (Kai, 2026-09-30): two
CONCURRENT one-seed runs with IDENTICAL flags that differ only in launch
mode, splitting the parallel-stack suspects together with searchact and
the faithful control (CURRICULUM.md "Next experiment -- the factorization"):

    python benchmarks/supervisors/supervise_factor.py noredo 0    # --ray --cpu-actor
    python benchmarks/supervisors/supervise_factor.py seqvtree 0  # sequential

Supervisor v2 pattern; a relaunch auto-resumes from the last checkpoint."""
import json
import os
import re
import subprocess
import sys
import time

VARIANTS = {
    "noredo": ("effzero-repro-faithful-searchact-noredo", "pong-noredo",
               ["--ray", "--cpu-actor"]),
    "seqvtree": ("effzero-repro-faithful-seqvtree", "pong-seqvtree", []),
}
WRITE = "--write-manifest" in sys.argv
VARIANT = "noredo" if WRITE else sys.argv[1]
SEED = 0 if WRITE else int(sys.argv[2])
REPO = r"Y:\SwitchPlayer"
ARM, _stem, _mode = VARIANTS[VARIANT]
REL = f"benchmarks/results/{_stem}-s{SEED}-2026-09-30"
WAIT_CKPTS = [os.path.join(
    REPO, "benchmarks", "results", "pong-searchact-s0-2026-09-29",
    "resume-pong-skip4-effzero-repro-faithful-searchact-seed0.ckpt")]
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
# Written by `supervise_factor.py --write-manifest` in the worktree when the
# audit loop closes; any later edit to one of them blocks the launch.
MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "factor_manifest.json")
# Every repo module the run executes (SVE audit r1 L4), plus the tests
# and this supervisor.
AUDITED = ["muzero.py", "redo.py", "muzero_vtree.py", "networks.py",
           "scalar_categorical.py", "memory.py", "EpisodicLifeEnv.py",
           "atari_wrappers.py", "experiments/ablation.py",
           "experiments/ray_actors.py", "tests/test_prioract.py",
           "tests/test_sve.py", "tests/test_searchact.py",
           "tests/test_factor.py", "tests/test_harness.py",
           "tests/test_ray_harness.py", "tests/test_checkpoint_resume.py",
           "tests/test_paper_mode.py", "tests/test_reanalyze_refresh.py",
           "tests/test_reanalysis.py", "tests/test_vtree.py", "pytest.ini",
           "benchmarks/supervisors/supervise_factor.py"]


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
         "assert all(a in ablation.ARMS for a in ('effzero-repro-faithful-"
         "searchact-noredo', 'effzero-repro-faithful-seqvtree'))")


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
    then the prioract + SVE + searchact + factor suites run IN the main tree."""
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
                            "tests/test_factor.py"],
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
    msg = ("ALERT: the searchact run's supervisor gave up (%s) -- diagnose; "
           "still waiting" % states if "failed" in states else
           "WAITING for the searchact run to complete (%s)" % states)
    if msg != told or waited % 60 == 0:
        print(f"{msg} [{waited // 60} h]", flush=True)
        told = msg
    time.sleep(60)
    waited += 1
print("searchact run complete", flush=True)
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
print(f"preflight ok (manifest + quick probe + 4 suites); starting {VARIANT} "
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
