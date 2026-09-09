"""Queue runner: when the phase-5 baseline (pong-async-s1, unthrottled)
finishes its 40 units or dies for good, clean up and launch TEST 6:
arm effzero-repro-async-throttle (admission floor 2.0 -- ONE variable
vs async-fresh: legacy's data diet made explicit; actor never blocked,
excess episodes dropped before eviction can churn stored data), seed 1,
40 units, recipe otherwise identical. Committed 2bdf7b9c, audit closed
NO FINDINGS. Reads: restores the satstale corridor (-18/-16/-14.5/-10
at u 10-25) => reuse was the lever and async keeps its wall-clock
properties at full speed; still slow => newcomer-refresh timing
(ASYNC_REFRESH_ON_COLLECT=False) is the isolated test 7. Supervisor
v2 pattern."""
import json
import os
import subprocess
import sys
import time

REPO = r"Y:\SwitchPlayer"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
RAY_EXE = os.path.join(REPO, ".venv", "Scripts", "ray.exe")

OLD_BASE = os.path.join(REPO, "benchmarks", "results",
                        "pong-async-s1-2026-09-06")
OLD_CKPT = os.path.join(
    OLD_BASE, "resume-pong-skip4-effzero-repro-async-fresh-seed1.ckpt")

BASE = os.path.join(REPO, "benchmarks", "results", "pong-throttle-s1-2026-09-07")
JSONL = os.path.join(BASE, "ablation-pong-skip4.jsonl")
CKPT = os.path.join(
    BASE, "resume-pong-skip4-effzero-repro-async-throttle-seed1.ckpt")
DEATHS = os.path.join(BASE, "deaths.log")
COMMON = [PY, "experiments/ablation.py", "--arms",
          "effzero-repro-async-throttle", "--seed", "1", "--iterations",
          "40", "--train-steps", "2000", "--eval-every", "5", "--env",
          "Pong-v4", "--frameskip", "4", "--episode-cap", "3000",
          "--episodes-per-iter", "1", "--checkpoint-every", "2",
          "--ray", "--cpu-actor"]
MAX_ATTEMPTS = 20


def pids_matching(pattern):
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
          "Where-Object { $_.CommandLine -match '%s' } | "
          "ForEach-Object { $_.ProcessId }" % pattern)
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, timeout=60)
        return [int(x) for x in out.stdout.split()]
    except Exception:
        return []


def old_done():
    try:
        return os.path.getsize(OLD_CKPT) < 1_000_000  # completion marker
    except OSError:
        return False


# -------- WAIT: replication completes (or dies for good) --------
print("QUEUE: waiting for async-s1 completion marker (40 iters)",
      flush=True)
gone_polls = 0
while True:
    if old_done():
        print("QUEUE: async-s1 COMPLETE -- proceeding", flush=True)
        break
    alive = pids_matching("supervise_async_s1|async-fresh")
    if not alive:
        gone_polls += 1
        if gone_polls >= 3:
            print("QUEUE: async-s1 procs gone without marker -- "
                  "proceeding anyway", flush=True)
            break
    else:
        gone_polls = 0
    time.sleep(60)

# The completed run's own supervisor exits by itself on the marker; only
# straggler cleanup is needed before the next cluster.
time.sleep(30)
for pid in pids_matching("supervise_async_s1|async-fresh"):
    subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)],
                   capture_output=True)
    print("QUEUE: killed straggler pid %d" % pid, flush=True)
try:
    subprocess.run([RAY_EXE, "stop", "--force"], capture_output=True,
                   timeout=120)
    print("QUEUE: ray stop --force done", flush=True)
except Exception as e:
    print("QUEUE: ray stop skipped (%s)" % type(e).__name__, flush=True)
time.sleep(60)

# -------- SUPERVISE: sat-guard seed 1 (test 4) --------
seen = set()


def rows_lines():
    try:
        with open(JSONL, "rb") as f:
            return [l for l in f.read().decode("utf-8", "replace").splitlines()
                    if l.strip()]
    except OSError:
        return []


def emit_new_rows():
    for line in rows_lines():
        key = line.strip()
        if key in seen:
            continue
        seen.add(key)
        try:
            r = json.loads(line)
            print("EVAL u %s  train %s  env %s  adm_ratio %s  "
                  "dropped %s  H(pi) %.3f  freshH %s  return %.1f  "
                  "elapsed %ss" % (
                      r.get("iteration"), r.get("train_steps"),
                      r.get("env_steps"),
                      r.get("train_per_admitted_env"),
                      r.get("episodes_dropped"),
                      r.get("policy_entropy", float("nan")),
                      ("%.3f" % r["fresh_target_entropy"]
                       if isinstance(r.get("fresh_target_entropy"), float)
                       else "n/a"),
                      r.get("return_mean", float("nan")),
                      round(r.get("elapsed_s", 0))), flush=True)
        except Exception:
            pass


def completed():
    try:
        return os.path.getsize(CKPT) < 1_000_000
    except OSError:
        return False


def snapshot(attempt, rc, lifetime):
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = ["=== death attempt=%d rc=%s lifetime=%.0fs at %s"
             % (attempt, rc, lifetime, stamp)]
    for cmd in (["nvidia-smi", "--query-gpu=memory.used,memory.total,"
                 "utilization.gpu,temperature.gpu", "--format=csv,noheader"],
                ["powershell", "-NoProfile", "-Command",
                 "$o=Get-CimInstance Win32_OperatingSystem; "
                 "'ram_free_gb: ' + [math]::Round($o.FreePhysicalMemory/1MB,1)"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=30).stdout.strip()
        except Exception as e:
            out = "<%s>" % type(e).__name__
        lines.append(out)
    with open(DEATHS, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("DEATH rc=%s after %.0fs; snapshot -> deaths.log" % (rc, lifetime),
          flush=True)


os.makedirs(BASE, exist_ok=True)
for line in rows_lines():
    seen.add(line.strip())

env = {**os.environ,
       "ABLATION_RESULTS": "benchmarks/results/pong-throttle-s1-2026-09-07",
       "PYTHONFAULTHANDLER": "1"}
deaths = []
quick = 0
for attempt in range(1, MAX_ATTEMPTS + 1):
    if completed():
        print("RUN COMPLETE (checkpoint is a marker)", flush=True)
        sys.exit(0)
    args = COMMON + (["--resume"] if os.path.exists(CKPT) else [])
    log = open(os.path.join(BASE, "sup%d.log" % attempt), "ab")
    born = time.time()
    p = subprocess.Popen(args, cwd=REPO, env=env, stdout=log,
                         stderr=subprocess.STDOUT, creationflags=0x08000000)
    print("LAUNCHED attempt %d (%s), pid %d"
          % (attempt, "resume" if "--resume" in args else "fresh", p.pid),
          flush=True)
    while p.poll() is None:
        emit_new_rows()
        time.sleep(30)
    log.close()
    emit_new_rows()
    rc = p.returncode
    lifetime = time.time() - born
    snapshot(attempt, rc, lifetime)
    if completed():
        print("RUN COMPLETE", flush=True)
        sys.exit(0)
    deaths.append(time.time())
    if len([t for t in deaths if time.time() - t < 3600]) >= 5:
        print("CRASH LOOP: 5 deaths within 60 min -- stopping for diagnosis",
              flush=True)
        sys.exit(1)
    if lifetime < 600:
        quick += 1
        wait = min(900, 60 * (2 ** quick))
        print("quick death #%d; backing off %ds" % (quick, wait), flush=True)
        time.sleep(wait)
    else:
        quick = 0
        time.sleep(15)
print("MAX ATTEMPTS REACHED", flush=True)
sys.exit(1)
