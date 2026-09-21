"""Early-collapse hypothesis, behaviour side: did self-play stop TRYING
actions? Reads replay buffers left in killed runs' checkpoints and reports,
per trajectory in birth order, the effective-action marginal (Pong's 6
actions are 3 behaviours: NOOP={0,1} UP={2,4} DOWN={3,5}) and the mean
per-step effective entropy of the stored policy targets."""
import sys, pathlib, glob
ROOT = pathlib.Path("Y:/SwitchPlayer"); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import numpy as np, ablation

GROUPS = ((0, 1), (2, 4), (3, 5))
def eff(p):
    p = np.asarray(p, float); return np.array([p[list(g)].sum() for g in GROUPS])
def H(p):
    p = np.asarray(p, float); p = p[p > 0]; return float(-(p * np.log(p)).sum())

RUNS = sys.argv[1:] or ["pong-vsat-2026-09-03", "pong-vtree-2026-09-02", "pong-throttle-s1-2026-09-07",
                        "pong-satguard-s1-2026-09-05", "pong-arch1-2026-08-29", "pong-satstale-2026-09-02",
                        "pong-vsat2-2026-09-03", "pong-keepnew90-2026-09-08"]
for name in RUNS:
    ck = glob.glob(str(ROOT / "benchmarks/results" / name / "resume-*.ckpt"))[0]
    st = ablation._read_checkpoint(ck)
    rp = st.get("replay") or []
    print(f"=== {name}  iter {st.get('iteration')}  env_steps {st.get('env_steps')}  trajs {len(rp)}", flush=True)
    rp = sorted(rp, key=lambda t: t[0].get("born_step", 0))
    allacts, allH = [], []
    for t in rp:
        acts = np.array([r["action"] for r in t[:-1]])          # last record = terminal/tail placeholder
        m = np.array([np.isin(acts, g).mean() for g in GROUPS])
        hs = [H(eff(r["target_policy"])) for r in t[:-1] if r.get("has_policy_target", True)]
        ret = sum(float(r["reward"]) for r in t)
        allacts.append(acts); allH += hs
        print(f"  born {t[0].get('born_step',0):>7}  len {len(t):>5}  ret {ret:+5.0f}  "
              f"act NOOP/UP/DOWN {m[0]:.2f}/{m[1]:.2f}/{m[2]:.2f}  H3(marg) {H(m):.3f}  "
              f"mean H3(target) {np.mean(hs):.3f}", flush=True)
    a = np.concatenate(allacts); m = np.array([np.isin(a, g).mean() for g in GROUPS])
    # run-lengths of the effective action: how long the agent holds one behaviour
    e = np.select([np.isin(a, g) for g in GROUPS], [0, 1, 2]); runs = np.diff(np.flatnonzero(np.diff(np.r_[-1, e, -1]) != 0))
    print(f"  BUFFER: NOOP/UP/DOWN {m[0]:.3f}/{m[1]:.3f}/{m[2]:.3f}  H3 {H(m):.3f} (max {np.log(3):.3f})  "
          f"mean per-step H3(target) {np.mean(allH):.3f}  mean hold {runs.mean():.2f} steps", flush=True)
    del st, rp
print("DONE")
