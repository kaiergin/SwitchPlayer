import json, pathlib
HERE = pathlib.Path(__file__).resolve().parent
logged = json.load(open(HERE / "logged_rows.json"))
recs = {}
for l in open(HERE / "xrun_results.jsonl"):
    r = json.loads(l)
    recs[(r["label"], r["iteration"])] = r      # last wins
order = ["TK", "TKX", "FS", "G0", "RG", "DS", "TWIN"]
hdr = ("run", "it", "A", "ceil", "log maxp", "log tgtH", "prior H/maxp(probe)",
       "|p1| eval", "|p1| trainBN", "|p1| allA eval", "BN off s1/s2",
       "srch H eval", "srch H recal")
print(" | ".join(hdr))
for lab in order:
    for (l, it), r in sorted(recs.items(), key=lambda kv: kv[0][1]):
        if l != lab:
            continue
        g = logged.get(f"{lab}-{it}", {})
        ceil = g.get("ceil") if g.get("ceil") is not None else r.get("ceiling_at_steps")
        f = lambda v, n=3: "-" if v is None else (f"{v:.{n}f}" if isinstance(v, float) else str(v))
        print(" | ".join([lab, str(it), str(r["A"]), f(ceil, 1), f(g.get("maxp")), f(g.get("fH")),
                          f"{r['prior_H/lnA']:.2f}/{r['prior_maxp']:.2f}",
                          f(r["p1_abs_eval"], 2), f(r["p1_abs_trainBN"], 2),
                          f(r["p1_allactions_abs_eval"], 2),
                          f"{r['normout_offset_sd_step1']:.1f}/{r['normout_offset_sd_step2']:.1f}",
                          f(r.get("search_H/lnA_eval")), f(r.get("search_H/lnA_step1recal"))]))
        if r.get("missing_spec_keys"):
            print("   (spec keys unknown to this tree, dropped:", r["missing_spec_keys"], ")")
