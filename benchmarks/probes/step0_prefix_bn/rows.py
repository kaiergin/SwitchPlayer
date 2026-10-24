import json, math, sys
RUNS = {
 "TK": (r"Y:\SwitchPlayer-rgrow-run\benchmarks\results\mspacman-tokens-rgrow-s0-2026-10-19", 36, [30,55,75,90]),
 "RG": (r"Y:\SwitchPlayer-rgrow-run\benchmarks\results\mspacman-rgrow-s0-2026-10-19", 9, [30,55,75,90]),
 "TKX": (r"Y:\SwitchPlayer-rgrowx-run\benchmarks\results\mspacman-tokens-rgrow-x270-s0-2026-10-20", 36, [55,100,140,200]),
 "DS": (r"Y:\SwitchPlayer-distill-run\benchmarks\results\mspacman-student-distill-s0-2026-10-21", 14, [0,30,60,90]),
 "FS": (r"Y:\SwitchPlayer-distill-run\benchmarks\results\mspacman-student-fresh-s0-2026-10-21", 14, [45,60,75]),
 "TWIN": (r"Y:\SwitchPlayer\benchmarks\results\mspacman-s0-2026-10-16", 9, [30,55,75,90]),
 "G0": (r"Y:\SwitchPlayer-gen0-run\benchmarks\results\mspacman-tokens-gen0-s0-2026-10-24", 36, [5,10,15]),
}
out = {}
for name, (d, A, its) in RUNS.items():
    rows = {}
    with open(d + r"\ablation-mspacmannoframeskip-skip4.jsonl") as f:
        for l in f:
            try: r = json.loads(l)
            except Exception: continue
            rows[r.get("iteration")] = r
    print(f"== {name}: rows {sorted(k for k in rows if k is not None)[:3]}..{max(k for k in rows if k is not None)}")
    for it in its:
        r = rows.get(it)
        if r is None: print("  ", it, "no row"); continue
        fH = r.get("fresh_target_entropy")
        rec = dict(ceil=r.get("reward_ceiling"), maxp=r.get("policy_maxp"),
                   polH=(r.get("policy_entropy") or 0)/math.log(A),
                   fH=(fH/math.log(A) if fH is not None else None),
                   ret=r.get("return_mean"), prefix_err=r.get("prefix"), prefix_ctrl=r.get("control"),
                   train_steps=r.get("train_steps"))
        out[f"{name}-{it}"] = rec
        print("  ", it, {k: (round(v,3) if isinstance(v,float) else v) for k,v in rec.items()})
json.dump(out, open("logged_rows.json","w"), indent=1)
