"""EfficientZero-fidelity runs (2026-09-24, Kai): the edge arm with every
verified deviation from EfficientZero's released code closed at once (arm
effzero-repro-refresh5x-communal-edge-faithful; see its comment in
experiments/ablation.py). Seeds 0 and 1, one supervisor each, and each WAITS
for the same-seed edge run to complete first, so only two runs share the GPU:

    python benchmarks/supervisors/supervise_faithful.py 0
    python benchmarks/supervisors/supervise_faithful.py 1

Early readouts: loss divergence/NaN in the first iterations (SGD lr 0.2 with
no latent renormalisation); then batch-stat projector rank, shift/state
variance split and raw action discrimination on the per-eval snapshots at
iters 15-30 (probe_shift_collapse, probe_counterfactual_edge) against the
edge pair (rank 34-55 at iter 10 -> 2.5-3.8 at iter 15). Supervisor v2
pattern; a relaunch auto-resumes from the last checkpoint."""
import json
import os
import subprocess
import sys
import time

SEED = int(sys.argv[1])
REPO = r"Y:\SwitchPlayer"
ARM = "effzero-repro-refresh5x-communal-edge-faithful"
REL = f"benchmarks/results/pong-faithful-s{SEED}-2026-09-24"
EDGE_CKPT = os.path.join(
    REPO, "benchmarks", "results", f"pong-edge-s{SEED}-2026-09-21",
    f"resume-pong-skip4-effzero-repro-refresh5x-communal-edge-seed{SEED}.ckpt")
BASE = os.path.join(REPO, *REL.split("/"))
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
JSONL = os.path.join(BASE, "ablation-pong-skip4.jsonl")
CKPT = os.path.join(BASE, f"resume-pong-skip4-{ARM}-seed{SEED}.ckpt")
DEATHS = os.path.join(BASE, "deaths.log")
COMMON = [PY, "experiments/ablation.py", "--arms", ARM, "--seed", str(SEED),
          "--iterations", "90", "--train-steps", "2000", "--eval-every", "5",
          "--env", "Pong-v4", "--frameskip", "4", "--episode-cap", "3000",
          "--episodes-per-iter", "1", "--checkpoint-every", "2",
          "--snapshot-weights"]
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
        except Exception:
            pass


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


def edge_done():
    try:
        return os.path.getsize(EDGE_CKPT) < 1_000_000
    except OSError:
        return False


if not edge_done():
    print(f"WAITING for edge s{SEED} to complete", flush=True)
    while not edge_done():
        time.sleep(60)
    print(f"edge s{SEED} complete; starting", flush=True)

os.makedirs(BASE, exist_ok=True)
for line in rows():
    seen.add(line.strip())

env = {**os.environ, "ABLATION_RESULTS": REL, "PYTHONFAULTHANDLER": "1"}
deaths = []
quick = 0
for attempt in range(1, MAX_ATTEMPTS + 1):
    if completed():
        print(f"RUN COMPLETE s{SEED} (checkpoint is a marker)", flush=True)
        sys.exit(0)
    args = COMMON + (["--resume"] if os.path.exists(CKPT) else [])
    log = open(os.path.join(BASE, f"supv2_{attempt}.log"), "ab")
    born = time.time()
    p = subprocess.Popen(args, cwd=REPO, env=env, stdout=log,
                         stderr=subprocess.STDOUT, creationflags=0x08000000)
    print(f"LAUNCHED s{SEED} attempt {attempt} "
          f"({'resume' if '--resume' in args else 'fresh'}), pid {p.pid}",
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
