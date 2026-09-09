"""Pool-mode validation run (phase 3b): does the GPU inference server
work end-to-end and is it faster? Same arm and recipe as archfix, ONLY
the harness mode differs: --ray --ray-workers 6 (six CPU env/tree
workers, one GPU server batching across their trees; RECIPE CHANGE by
design -- 6 episodes/iteration). 10 iterations is the speed + soundness
measurement, not a science run: the verdicts are episodes/hour vs
archfix's sequential ~1030-1400 s/iter-of-1-episode, sane eval rows,
and no crashes. Supervisor v2 pattern: death snapshots, backoff,
crash-loop guard."""
import json
import os
import subprocess
import sys
import time

REPO = r"Y:\SwitchPlayer"
BASE = os.path.join(REPO, "benchmarks", "results", "poolval-2026-09-01")
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
JSONL = os.path.join(BASE, "ablation-pong-skip4.jsonl")
CKPT = os.path.join(
    BASE, "resume-pong-skip4-effzero-repro-refresh5x-communal-seed0.ckpt")
DEATHS = os.path.join(BASE, "deaths.log")
COMMON = [PY, "experiments/ablation.py", "--arms",
          "effzero-repro-refresh5x-communal", "--seed", "0", "--iterations",
          "10", "--train-steps", "2000", "--eval-every", "2", "--env",
          "Pong-v4", "--frameskip", "4", "--episode-cap", "3000",
          "--episodes-per-iter", "1", "--checkpoint-every", "2",
          "--ray", "--ray-workers", "6"]
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
            print("EVAL iter %s  train %s  H(pi) %.3f  freshH %s  "
                  "return %.1f  elapsed %ss" % (
                      r.get("iteration"), r.get("train_steps"),
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
    lines = [f"=== death attempt={attempt} rc={rc} lifetime={lifetime:.0f}s "
             f"at {stamp}"]
    for cmd in (["nvidia-smi", "--query-gpu=memory.used,memory.total,"
                 "utilization.gpu,temperature.gpu", "--format=csv,noheader"],
                ["powershell", "-NoProfile", "-Command",
                 "$o=Get-CimInstance Win32_OperatingSystem; "
                 "'ram_free_gb: ' + [math]::Round($o.FreePhysicalMemory/1MB,1)"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=30).stdout.strip()
        except Exception as e:
            out = f"<{type(e).__name__}>"
        lines.append(out)
    with open(DEATHS, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"DEATH rc={rc} after {lifetime:.0f}s; snapshot -> deaths.log",
          flush=True)


os.makedirs(BASE, exist_ok=True)
for line in rows():
    seen.add(line.strip())

env = {**os.environ,
       "ABLATION_RESULTS": "benchmarks/results/poolval-2026-09-01",
       "PYTHONFAULTHANDLER": "1"}
deaths = []
quick = 0
for attempt in range(1, MAX_ATTEMPTS + 1):
    if completed():
        print("RUN COMPLETE (checkpoint is a marker)", flush=True)
        sys.exit(0)
    args = COMMON + (["--resume"] if os.path.exists(CKPT) else [])
    log = open(os.path.join(BASE, f"sup{attempt}.log"), "ab")
    born = time.time()
    p = subprocess.Popen(args, cwd=REPO, env=env, stdout=log,
                         stderr=subprocess.STDOUT, creationflags=0x08000000)
    print(f"LAUNCHED attempt {attempt} ({'resume' if '--resume' in args else 'fresh'}), pid {p.pid}",
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
        print(f"quick death #{quick}; backing off {wait}s", flush=True)
        time.sleep(wait)
    else:
        quick = 0
        time.sleep(15)
print("MAX ATTEMPTS REACHED", flush=True)
sys.exit(1)
