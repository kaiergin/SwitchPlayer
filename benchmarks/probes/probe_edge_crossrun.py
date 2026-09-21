"""Does the every-Nth-record anti-behaviour cliff track the communal WINDOW
BOUNDARY across runs? Stored targets' argmax==action by record index
modulo 16 and 8 for every checkpoint that carries a replay buffer:
communal runs at window 16 (archfix family, r12) should dip at 16k+15,
the pre-2026-08-30 communal runs (window 8) at 8k+7, and fresh-root
(non-communal) refresh runs (async-fresh / async-keepnew) nowhere."""
import sys, glob, pathlib
ROOT = pathlib.Path(r"Y:\SwitchPlayer"); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import numpy as np, ablation
cks = sorted(glob.glob(r"Y:\SwitchPlayer\benchmarks\results\*\resume-*.ckpt"))
for ck in cks:
    name = pathlib.Path(ck).parent.name
    try:
        st = ablation._read_checkpoint(ck)
        replay = st.get("replay")
    except Exception as e:
        print(f"{name}: unreadable ({type(e).__name__}: {e})"); continue
    if not replay:
        print(f"{name}: no replay"); continue
    m16 = {k: [] for k in range(16)}; m8 = {k: [] for k in range(8)}; n = 0
    for t in replay:
        for i, r in enumerate(t):
            if not r.get("has_policy_target", True) or "target_policy" not in r:
                continue
            p = np.asarray(r["target_policy"], float); a = int(r["action"]); hit = int(np.argmax(p) == a)
            m16[i % 16].append(hit); m8[i % 8].append(hit); n += 1
    s16 = [np.mean(m16[k]) if m16[k] else float("nan") for k in range(16)]
    s8 = [np.mean(m8[k]) if m8[k] else float("nan") for k in range(8)]
    others16 = np.nanmean(s16[:15]); others8 = np.nanmean(s8[:7])
    arm = st.get("arm", "?")
    print(f"{name} [{arm}] records {n}, trajs {len(replay)}, steps {st.get('training_steps', st.get('env_steps', '?'))}")
    print(f"   argmax==action by i%16: " + " ".join(f"{x:.2f}" for x in s16) + f"   slot15/others {s16[15]/others16:.2f}")
    print(f"   argmax==action by i%8 : " + " ".join(f"{x:.2f}" for x in s8) + f"   slot7/others {s8[7]/others8:.2f}", flush=True)
print("DONE", flush=True)
