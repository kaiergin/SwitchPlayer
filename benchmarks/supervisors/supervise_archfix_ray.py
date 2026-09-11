"""CHECK D (2026-09-11): archfix's EXACT arm and recipe
(effzero-repro-refresh5x-communal, seed 0, 90 iters, communal quota
refresh on the sequential engine inline) on the PARALLEL stack
(--ray --cpu-actor). Only deltas vs pong-archfix-2026-08-30 (+17.1):
the CPU environment actor and one-iteration pipelining. Checks A and B
exonerated the evaluator and the GPU tree engine; B measured a large
fresh-root-vs-communal refresh-topology difference (42% best-action
disagreement, fresh-root sharper). Reproduces ~+17 => the stack is
clean and topology is the culprit; fails => acting/pipelining bug.
Supervisor v2 pattern."""
import json
import os
import subprocess
import sys
import time

REPO = r"Y:\SwitchPlayer"
BASE = os.path.join(REPO, "benchmarks", "results", "pong-archfix-ray-2026-09-11")
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
JSONL = os.path.join(BASE, "ablation-pong-skip4.jsonl")
CKPT = os.path.join(
    BASE, "resume-pong-skip4-effzero-repro-refresh5x-communal-seed0.ckpt")
DEATHS = os.path.join(BASE, "deaths.log")
COMMON = [PY, "experiments/ablation.py", "--arms",
          "effzero-repro-refresh5x-communal", "--seed", "0", "--iterations",
          "90", "--train-steps", "2000", "--eval-every", "5", "--env",
          "Pong-v4", "--frameskip", "4", "--episode-cap", "3000",
          "--episodes-per-iter", "1", "--checkpoint-every", "2",
          "--ray", "--cpu-actor"]
MAX_ATTEMPTS = 20

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
            print("EVAL u %s  train %s  env %s  ratio %s  H(pi) %.3f  "
                  "freshH %s  return %.1f  elapsed %ss" % (
                      r.get("iteration"), r.get("train_steps"),
                      r.get("env_steps"), r.get("train_per_env_step"),
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
for line in rows():
    seen.add(line.strip())

env = {**os.environ,
       "ABLATION_RESULTS": "benchmarks/results/pong-archfix-ray-2026-09-11",
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
