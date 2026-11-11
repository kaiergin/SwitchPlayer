"""Test H, STAGE D2: the reads and the verdict (CURRICULUM "Next experiment
(pre-registered) -- Test H": READS (1)-(8), VERDICT, NEXT, and the BUILD NOTES
of the AUDIT r10 paragraph). numpy only, CPU; reads stage D1's logits, stage
B's truth, stage A's record and stage C's plan. Decides nothing it was not
registered to decide.

usage: python benchmarks/probes/score_move_guesser.py [--data DIR] [--truth DIR]
           [--train DIR] [--logits DIR] [--boot 2000]"""
import argparse
import json
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
MOVES = 9
SET_OWN, SET_CRN, SET_NOOP = 0, 1, 2
END_LIFE = 1
T4 = 2.7764451051977987                    # t(0.975, 4 df)
BOOT = 2000
SEED = 20261111
FLOOR_DECISIONS = 5000
FLOOR_GAMES = 16.0
FLOOR_J3, FLOOR_J4 = 300, 100
COMMIT_MIN = 100
BAR_R, BAR_RF = 0.5, 0.5
BAR_J = 0.25
BAR_K = 0.5
STAGES = (("10-80", 10, 80), ("90-120", 90, 120), ("130-160", 130, 160))
ARMS = ("S", "F1", "F2", "F4", "F8", "F16", "SIGMA")
SEEDS = (0, 1, 2, 3, 4)


# --------------------------------------------------------------------------
# the table: every item of every held-out game of one player
# --------------------------------------------------------------------------
class Table:
    """Items of one player's held-out games, concatenated, with the game of
    each item, and the logits of every model."""

    def __init__(self, games):
        """games: list of dicts, each a game's D1 file as arrays plus
        'name', 'iteration' and (optional) 'truth' arrays."""
        keys = [k for k in games[0] if not k.startswith("logits/") and k not in
                ("name", "iteration", "truth", "extra")]
        self.names = [g["name"] for g in games]
        self.iteration = np.array([g["iteration"] for g in games])
        self.game = np.concatenate([np.full(len(g["set"]), i) for i, g in enumerate(games)])
        for k in keys:
            setattr(self, k, np.concatenate([g[k] for g in games]))
        self.logits = {}
        for k in games[0]:
            if k.startswith("logits/"):
                self.logits[k[7:]] = np.concatenate([g[k] for g in games],
                                                    axis=1 if k.endswith("/cut") else 0)
        self.n_games = len(games)
        # decision key: (game, decision)
        self.dkey = self.game.astype(np.int64) * 10 ** 6 + self.decision

    def top(self, key):
        return self.logits[key].argmax(-1)

    def correct(self, key, cut=None):
        """Per item: the arm's top move lies in the item's class."""
        lg = self.logits[key + "/cut"][cut] if cut is not None else self.logits[key]
        t = lg.argmax(-1)
        return self.members[np.arange(len(t)), t]


# --------------------------------------------------------------------------
# per-game sums and the interval
# --------------------------------------------------------------------------
def game_sums(table, mask, values, weights):
    """(G,) sums of weights * values and of weights over the masked items."""
    g = table.game[mask]
    num = np.bincount(g, weights=(weights * values)[mask], minlength=table.n_games)
    den = np.bincount(g, weights=weights[mask], minlength=table.n_games)
    return num, den


def draws(n_games, boot=BOOT, seed=SEED):
    """(boot, G) resample counts of whole games."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_games, (boot, n_games))
    out = np.zeros((boot, n_games))
    for b in range(boot):
        out[b] = np.bincount(idx[b], minlength=n_games)
    return out


def interval(point, boot_values, seed_values):
    """The registered interval: the bootstrap's percentiles of the seed-mean
    and t(4) x the between-seed SD / sqrt(5), in quadrature, each side."""
    bv = np.asarray(boot_values, float)
    bv = bv[np.isfinite(bv)]               # a subset read's draws without its games
    if not len(bv) or not np.isfinite(point):
        return float("nan"), float("nan")
    q_lo, q_hi = np.percentile(bv, 2.5), np.percentile(bv, 97.5)
    sv = np.asarray(seed_values, float)
    seed = T4 * sv.std(ddof=1) / math.sqrt(len(sv)) if len(sv) > 1 else 0.0
    lo = point - math.sqrt(max(point - q_lo, 0.0) ** 2 + seed ** 2)
    hi = point + math.sqrt(max(q_hi - point, 0.0) ** 2 + seed ** 2)
    return lo, hi


def ratio_stat(num_a, den_a, num_b, den_b, C):
    """Per seed R = 1 - err_a / err_b and its draws: num/den (S, G) arrays
    per seed; returns (per-seed R, mean, draws of the mean). The seeds share
    their items, so a draw without the subset's games is undefined for all of
    them (NaN; interval() leaves it out); an empty subset gives NaN."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ea = num_a.sum(1) / den_a.sum(1)
        eb = num_b.sum(1) / den_b.sum(1)
        r = 1 - ea / eb
        ea_b = (C @ num_a.T) / (C @ den_a.T)        # (B, S)
        eb_b = (C @ num_b.T) / (C @ den_b.T)
        rb = 1 - ea_b / eb_b
    return r, float(r.mean()), rb.mean(1)


def keys(arm, plan_models):
    return [f"{arm}-s{s}" for s in SEEDS if f"{arm}-s{s}" in plan_models]


def R(table, mask, a_keys, b_keys, C, weights=None):
    """R of arm a against arm b on the masked items: per seed, the mean, the
    interval. Seeds pair by index."""
    w = table.weight if weights is None else weights
    na, da, nb, db = [], [], [], []
    for ka, kb in zip(a_keys, b_keys):
        n, d = game_sums(table, mask, 1.0 - table.correct(ka), w)
        na.append(n), da.append(d)
        n, d = game_sums(table, mask, 1.0 - table.correct(kb), w)
        nb.append(n), db.append(d)
    na, da, nb, db = map(np.array, (na, da, nb, db))
    per, mean, boot = ratio_stat(na, da, nb, db, C)
    lo, hi = interval(mean, boot, per)
    with np.errstate(divide="ignore", invalid="ignore"):
        err_a = float((na.sum(1) / da.sum(1)).mean())
        err_b = float((nb.sum(1) / db.sum(1)).mean())
    return {"per_seed": per.tolist(), "R": mean, "lo": lo, "hi": hi,
            "err_a": err_a, "err_b": err_b, "items": int(mask.sum()),
            "undefined_draws": int((~np.isfinite(boot)).sum())}


# --------------------------------------------------------------------------
# junctions: J and K
# --------------------------------------------------------------------------
def junction_counts(table, set_code, key, min_classes, decision_mask):
    """Per game: junctions, identified, committed, for one model, over the
    decisions of `set_code` with at least `min_classes` classes."""
    m = (table.set == set_code) & decision_mask
    tops = table.top(key)
    out = {"junctions": np.zeros(table.n_games), "identified": np.zeros(table.n_games),
           "committed": np.zeros(table.n_games)}
    idx = np.flatnonzero(m)
    order = np.argsort(table.dkey[idx], kind="stable")
    idx = idx[order]
    bounds = np.flatnonzero(np.diff(table.dkey[idx])) + 1
    for grp in np.split(idx, bounds):
        if len(grp) < min_classes:
            continue
        g = table.game[grp[0]]
        out["junctions"][g] += 1
        ok = table.members[grp, tops[grp]]
        if ok.all():
            out["identified"][g] += 1
        # the class each answer falls in: the item whose members hold it
        cls_of = []
        for t in tops[grp]:
            hit = np.flatnonzero(table.members[grp, t])
            cls_of.append(int(hit[0]) if len(hit) else -1)
        if len(set(cls_of)) == len(grp) and -1 not in cls_of:
            out["committed"][g] += 1
    return out


def effective_games(counts):
    p = counts / counts.sum() if counts.sum() else counts
    return float(1.0 / (p ** 2).sum()) if counts.sum() else 0.0


def JK(table, set_code, sigma_keys, C, decision_mask):
    """J3, J4, K3, K4 for SIGMA on one item set: per seed, mean, interval,
    floors (from the truth alone) and the commitment test (full sample)."""
    res = {}
    for tier, mc, floor in (("3", 3, FLOOR_J3), ("4", 4, FLOOR_J4)):
        per = [junction_counts(table, set_code, k, mc, decision_mask) for k in sigma_keys]
        junc = per[0]["junctions"]
        n_j = int(junc.sum())
        eff = effective_games(junc)
        unread = n_j < floor or eff < FLOOR_GAMES
        J_seed = np.array([p["identified"].sum() / max(n_j, 1) for p in per])
        Jb = np.array([(C @ p["identified"]) / np.maximum(C @ junc, 1) for p in per]).mean(0)
        commits = np.array([p["committed"].sum() for p in per])
        need = min(COMMIT_MIN, n_j / 2.0)
        commit_ok = bool(np.all(commits >= need))
        K_seed = np.array([p["identified"].sum() / p["committed"].sum()
                           if p["committed"].sum() else 0.0 for p in per])
        with np.errstate(divide="ignore", invalid="ignore"):
            kb = np.array([np.where(C @ p["committed"] > 0,
                                    (C @ p["identified"]) / (C @ p["committed"]), 0.0)
                           for p in per])
        Kb = kb.mean(0)
        J_lo, J_hi = interval(float(J_seed.mean()), Jb, J_seed)
        K_lo, K_hi = interval(float(K_seed.mean()), Kb, K_seed)
        res[tier] = {"junctions": n_j, "effective_games": eff, "unread": bool(unread),
                     "J": float(J_seed.mean()), "J_lo": J_lo, "J_hi": J_hi,
                     "J_seed": J_seed.tolist(),
                     "K": float(K_seed.mean()), "K_lo": K_lo, "K_hi": K_hi,
                     "K_seed": K_seed.tolist(), "commits": commits.tolist(),
                     "commit_need": need, "commit_ok": commit_ok}
    return res


# --------------------------------------------------------------------------
# the four conditions and the ladder
# --------------------------------------------------------------------------
def conditions(table, plan_models, C, dmask):
    S, G = keys("S", plan_models), keys("SIGMA", plan_models)
    crn = (table.set == SET_CRN) & dmask
    noop = (table.set == SET_NOOP) & dmask
    out = {"i": R(table, crn, G, S, C),
           "ii": R(table, crn & ~table.played, G, S, C),
           "iii_crn": JK(table, SET_CRN, G, C, dmask),
           "iii_noop": JK(table, SET_NOOP, G, C, dmask),
           "iv": R(table, noop, G, S, C)}
    c = {"i": out["i"]["lo"] >= BAR_R, "ii": out["ii"]["lo"] > 0,
         "iv": out["iv"]["lo"] >= BAR_RF}
    iii, unread = True, []
    for s in ("iii_crn", "iii_noop"):
        for tier in ("3", "4"):
            r = out[s][tier]
            if r["unread"]:
                unread.append(f"{s} J{tier}/K{tier}")
                iii = False
                continue
            ok = (r["J_lo"] >= BAR_J and r["K_lo"] > BAR_K and r["commit_ok"])
            iii = iii and ok
    c["iii"] = iii
    out["holds"], out["unread"] = c, unread
    return out


def ladder(cond, plan, truth_failures, contested_o0, eff_games):
    """The registered verdict, the first state that applies."""
    if truth_failures:
        return "TRUTH FAILURE", []
    why = []
    if contested_o0 < FLOOR_DECISIONS:
        why.append(f"{contested_o0} contested held-out O0 decisions < {FLOOR_DECISIONS}")
    if eff_games < FLOOR_GAMES:
        why.append(f"effective games {eff_games:.1f} < {FLOOR_GAMES}")
    why += [v for v in plan["void"] if v.startswith("S-") or v.startswith("SIGMA-")]
    if why:
        return "VOID", why
    s_capped = [f for f in plan["flags"] if f.startswith("S-")]
    g_capped = [f for f in plan["flags"] if f.startswith("SIGMA-")]
    if s_capped:
        return "CAPPED", s_capped
    r = cond["i"]
    h = cond["holds"]
    if all(h[k] for k in ("i", "ii", "iii", "iv")):
        return ("GUESSES THE GAME" + (" (SIGMA CAPPED, flagged)" if g_capped else ""),
                g_capped)
    if g_capped:
        return "CAPPED", g_capped
    if h["i"]:
        failed = [k for k in ("ii", "iii", "iv") if not h[k]] + cond["unread"]
        return "PARTLY", failed
    if r["lo"] > 0:
        return "PARTLY", ["R's lower bound below 0.5"]
    if r["hi"] < 0:
        return "WORSE THAN HABITS", []
    return "NO HELP", []


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_player(logits_dir, truth_dir, data_dir, names):
    games = []
    for name in names:
        with np.load(logits_dir / f"{name}-logits.npz") as z:
            g = {k: z[k] for k in z.files}
        meta = json.loads((data_dir / f"{name}.json").read_text())
        g["name"], g["iteration"] = name, meta["iteration"]
        with np.load(truth_dir / f"{name}-truth.npz") as z:
            g["truth"] = {k: z[k] for k in z.files}
        with np.load(data_dir / f"{name}.npz") as z:
            g["extra"] = {k: z[k] for k in ("kind", "move", "decision", "prior", "temp", "end")}
        games.append(g)
    t = Table(games)
    t.truth = [g["truth"] for g in games]
    t.extra = [g["extra"] for g in games]
    return t


def decision_mask(table, undo=False):
    """Items of decisions kept by the READS exclusion (obs_t not in FIT nor in
    another held-out game)."""
    if undo:
        return np.ones(len(table.set), bool)
    bad = table.h0_in_fit | table.h0_in_other
    # every item of a decision shares obs_t; take the flags from the own item
    own = table.set == SET_OWN
    bad_keys = set(table.dkey[own & bad].tolist())
    return ~np.isin(table.dkey, list(bad_keys))


# --------------------------------------------------------------------------
# the descriptive reads
# --------------------------------------------------------------------------
def raw_items(table, mask, key):
    """Items re-split by raw-frame classes: each pooled item's members divided
    by the decision's raw class ids; same input, so the same top move.
    Returns (game, correct, weight) arrays."""
    games, correct, weight = [], [], []
    tops = table.top(key)
    for i in np.flatnonzero(mask):
        g, a = table.game[i], table.a[i]
        raw = table.truth[g]["class_crn_raw"][a]
        mem = table.members[i]
        for r in sorted(set(int(x) for x in raw[mem])):
            sub = mem & (raw == r)
            games.append(g)
            correct.append(bool(sub[tops[i]]))
            weight.append(sub.sum() / MOVES)
    return np.array(games), np.array(correct), np.array(weight)


def R_raw(table, mask, a_keys, b_keys, C):
    na, da, nb, db = [], [], [], []
    for ka, kb in zip(a_keys, b_keys):
        for k, (nn, dd) in ((ka, (na, da)), (kb, (nb, db))):
            g, c, w = raw_items(table, mask, k)
            nn.append(np.bincount(g, weights=w * (1 - c), minlength=table.n_games))
            dd.append(np.bincount(g, weights=w, minlength=table.n_games))
    per, mean, boot = ratio_stat(*map(np.array, (na, da, nb, db)), C)
    lo, hi = interval(mean, boot, per)
    return {"R": mean, "lo": lo, "hi": hi, "per_seed": per.tolist()}


def oracle_error(table, mask):
    """The future-blind oracle's error, 1 - the largest class's size / 9 per
    decision, averaged over the masked decisions."""
    errs = []
    for k in np.unique(table.dkey[mask]):
        i = np.flatnonzero(table.dkey == k)
        i = i[table.set[i] == SET_CRN]
        if len(i):
            errs.append(1 - table.weight[i].max())
    return float(np.mean(errs)) if errs else float("nan")


def log_prob(table, key, mask):
    lg = table.logits[key][mask].astype(np.float64)
    lg = lg - lg.max(1, keepdims=True)
    p = np.exp(lg)
    p /= p.sum(1, keepdims=True)
    pc = (p * table.members[mask]).sum(1)
    w = table.weight[mask]
    return float((w * np.log(np.maximum(pc, 1e-300))).sum() / w.sum())


def own_play(table, plan_models, dmask, C):
    """Read (2): the played moves only, each decision weighted 1."""
    S, G = keys("S", plan_models), keys("SIGMA", plan_models)
    own = (table.set == SET_OWN) & dmask & np.isin(
        table.dkey, table.dkey[table.set == SET_CRN])          # contested
    w1 = np.ones(len(table.set))
    out = {}

    def acc(k, m):
        return float(table.correct(k)[m].mean()) if m.any() else float("nan")

    for label, lo_it, hi_it in (("all", 0, 10 ** 6),) + STAGES:
        it = table.iteration[table.game]
        m = own & (it >= lo_it) & (it <= hi_it)
        a_s = np.mean([acc(k, m) for k in S])
        a_g = np.mean([acc(k, m) for k in G])
        rec = {"decisions": int(m.sum()), "acc_S": a_s, "acc_SIGMA": a_g,
               "diff": a_g - a_s}
        if 1 - a_s >= 0.05 and m.any():
            rec["R"] = R(table, m, G, S, C, weights=w1)
        out[label] = rec
    return out


def references(table, plan, dmask):
    """Read (6) on O0's contested held-out decisions (own items)."""
    own = (table.set == SET_OWN) & dmask & np.isin(
        table.dkey, table.dkey[table.set == SET_CRN])
    idx = np.flatnonzero(own)
    maj = plan["majority"]
    out = {"majority": float(table.members[idx, maj].mean()),
           "random": float(table.weight[idx].mean())}
    rep, act, lp = [], [], []
    for i in idx:
        g, d = table.game[i], int(table.decision[i])
        ex = table.extra[g]
        e = int(np.flatnonzero((ex["kind"] == 2) & (ex["decision"] == d))[0])
        if e > 0 and int(ex["kind"][e - 1]) != 0:
            rep.append(bool(table.members[i, int(ex["move"][e - 1])]))
        p = ex["prior"][e].astype(np.float64)
        t = float(ex["temp"][e])
        q = p ** (1.0 / t)
        q /= q.sum()
        act.append(bool(table.members[i, int(np.argmax(q))]))
        lp.append(math.log(max(float(q[table.members[i]].sum()), 1e-300)))
    out["repeat_previous"] = float(np.mean(rep)) if rep else float("nan")
    out["acting_argmax"] = float(np.mean(act)) if act else float("nan")
    out["acting_log_prob"] = float(np.mean(lp)) if lp else float("nan")
    return out


def ceilings(table, dmask):
    """Read (4), from the truth arrays."""
    out = {}
    contested, total, pooled, raw, noop_raw, states = 0, 0, [], [], [], []
    per_k = {k: [0, 0] for k in range(1, 17)}
    per_k_items = {k: [0.0, 0.0] for k in range(1, 17)}
    differ = 0
    lost_crn, lost_noop, n_crn, n_noop = 0, 0, 0, 0
    for g, tru in enumerate(table.truth):
        keep = set(table.decision[(table.game == g) & dmask & (table.set == SET_OWN)].tolist())
        for a, d in enumerate(tru["decision"]):
            if int(d) not in keep:
                continue
            total += 1
            cls = tru["class_crn"][a]
            if not tru["contested"][a]:
                continue
            contested += 1
            pooled.append(len(set(cls.tolist())))
            raw.append(len(set(tru["class_crn_raw"][a].tolist())))
            noop_raw.append(len(set(tru["class_noop_raw"][a].tolist())))
            states.append(int(tru["n_states"][a]))
            final = _partition([cls[m] for m in range(MOVES)])
            differ += final != _partition([tru["class_noop"][a][m] for m in range(MOVES)])
            for k in range(1, 17):
                part = _partition([tru["crn_steps"][a, m, :k].tobytes() for m in range(MOVES)])
                per_k[k][0] += part == final
                per_k[k][1] += 1
                # item-weighted: a class already separated from every other by t+k
                part = np.array(part)
                for c in set(final):
                    mem = np.array(final) == c
                    sep = set(part[mem].tolist())
                    ok = len(sep) == 1 and (part == part[mem][0]).sum() == mem.sum()
                    per_k_items[k][0] += mem.sum() / MOVES * ok
                per_k_items[k][1] += 1.0
            pm = int(tru["played"][a])
            for c in set(cls.tolist()):
                m = int(np.flatnonzero(cls == c)[0])
                if c != cls[pm]:
                    n_crn += 1
                    lost_crn += int(tru["crn_end"][a, m]) == END_LIFE
            ncls = tru["class_noop"][a]
            for c in set(ncls.tolist()):
                m = int(np.flatnonzero(ncls == c)[0])
                n_noop += 1
                lost_noop += int(tru["noop_end"][a, m]) == END_LIFE
    out["decisions"] = total
    out["contested_share"] = contested / total if total else float("nan")
    out["classes_pooled"] = float(np.mean(pooled)) if pooled else float("nan")
    out["classes_raw"] = float(np.mean(raw)) if raw else float("nan")
    out["noop_classes_raw"] = float(np.mean(noop_raw)) if noop_raw else float("nan")
    out["states_per_decision"] = float(np.mean(states)) if states else float("nan")
    out["partition_final_by_k"] = {k: v[0] / v[1] for k, v in per_k.items() if v[1]}
    out["partition_final_by_k_items"] = {k: v[0] / v[1] for k, v in per_k_items.items() if v[1]}
    out["crn_noop_partitions_differ"] = differ / contested if contested else float("nan")
    out["branch_items_lost_life"] = lost_crn / n_crn if n_crn else float("nan")
    out["noop_items_lost_life"] = lost_noop / n_noop if n_noop else float("nan")
    out["E1_live_classes"], out["E0_all_classes"] = 2.33, 2.09
    return out


def _partition(keys_):
    ids = {}
    return [ids.setdefault(k, len(ids)) for k in keys_]


def perfect_fk(table, dmask):
    """Read (4)'s perfect F_k ceiling on the CRN items: a guesser seeing obs_t
    and obs_{t+k} only names, per group of items with the same obs_{t+k}, the
    heaviest class; its weighted accuracy."""
    out = {}
    for k in (1, 2, 4, 8, 16):
        acc_num = acc_den = 0.0
        for g, tru in enumerate(table.truth):
            keep = set(table.decision[(table.game == g) & dmask & (table.set == SET_OWN)].tolist())
            for a, d in enumerate(tru["decision"]):
                if int(d) not in keep or not tru["contested"][a]:
                    continue
                cls = tru["class_crn"][a]
                groups = {}
                for c in set(cls.tolist()):
                    m = int(np.flatnonzero(cls == c)[0])
                    key = tru["crn_steps"][a, m, max(0, k - 4):k].tobytes()
                    groups.setdefault(key, []).append((cls == c).sum() / MOVES)
                acc_num += sum(max(v) for v in groups.values())
                acc_den += 1.0
        out[k] = acc_num / acc_den if acc_den else float("nan")
    return out


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------
def score(data, truth, train, logits, boot=BOOT):
    plan = json.loads((train / "plan.json").read_text())
    summ = json.loads((logits / "logits.json").read_text())
    pm = plan["models"]
    truth_failures = []
    for f in sorted(truth.glob("*-truth.json")):
        info = json.loads(f.read_text())
        truth_failures += info["failures"]
    names = {p: [n for n in summ["games"] if n.startswith(p + "-")] for p in ("O0", "OF")}
    rep = {"plan": {k: plan[k] for k in ("cap", "rates_chosen", "flags", "void", "batch",
                                         "majority", "majority_acc")}}
    tables, conds = {}, {}
    for p in ("O0", "OF"):
        if not names[p]:
            continue
        t = load_player(logits, truth, data, names[p])
        tables[p] = t
        C = draws(t.n_games, boot)
        dm = decision_mask(t)
        crn = (t.set == SET_CRN) & dm
        eff = effective_games(np.bincount(t.game[crn], weights=t.weight[crn],
                                          minlength=t.n_games))
        contested = len(np.unique(t.dkey[crn]))
        cond = conditions(t, pm, C, dm)
        S, G, F1 = keys("S", pm), keys("SIGMA", pm), keys("F1", pm)
        extra = {
            "contested_decisions": contested, "effective_games": eff,
            "excluded_share": float(1 - dm[t.set == SET_OWN].mean()),
            "whole_item_in_fit_share": float(t.whole_in_fit[t.set == SET_OWN].mean()),
            "F1": R(t, crn, F1, S, C),
            "F1_noop": R(t, (t.set == SET_NOOP) & dm, F1, S, C),
            "junction_items": R(t, crn & _junction(t, crn), G, S, C),
            "junction_turn_items": R(t, crn & _junction(t, crn) & ~t.members[:, 0], G, S, C),
            "corridor_items": R(t, crn & ~_junction(t, crn), G, S, C),
            "raw_classes": R_raw(t, crn, G, S, C),
            "lost_life_out": R(t, crn & (t.end != END_LIFE), G, S, C),
            "exclusions_undone": R(t, t.set == SET_CRN, G, S, C),
            "oracle_error": oracle_error(t, crn),
            "near_dup_share": float((t.near_dup[t.set == SET_OWN] < 50).mean()),
        }
        nd_bad = set(t.dkey[(t.set == SET_OWN) & (t.near_dup < 50)].tolist())
        extra["near_dup_out"] = R(t, crn & ~np.isin(t.dkey, list(nd_bad)), G, S, C)
        extra["R_vs_oracle"] = 1 - cond["i"]["err_a"] / extra["oracle_error"]
        if "SIGMA-s0@final" in t.logits and "S-s0@final" in t.logits:
            # descriptive: selection by own-play move NLL also rewards habits
            extra["final_step_seed0"] = R(t, crn, ["SIGMA-s0@final"], ["S-s0@final"], C)
        extra["diff_SIGMA_F1"] = diff_R(t, crn, G, F1, S, C)
        extra["F_k_minus_S_seed0"] = {
            a: float(t.correct(f"{a}-s0")[crn].astype(float) @ t.weight[crn] / t.weight[crn].sum()
                     - t.correct("S-s0")[crn].astype(float) @ t.weight[crn] / t.weight[crn].sum())
            for a in ARMS if f"{a}-s0" in pm and a != "S"}
        extra["cut_profile"] = cut_profile(t, crn, (t.set == SET_NOOP) & dm, G, S)
        extra["sigma_weights"] = summ["sigma_weights"]
        extra["log_prob"] = {k: {"crn": log_prob(t, k, crn),
                                 "own": log_prob(t, k, (t.set == SET_OWN) & dm)}
                             for k in pm}
        extra["own_play"] = own_play(t, pm, dm, C)
        extra["ceilings"] = ceilings(t, dm)
        extra["perfect_fk"] = perfect_fk(t, dm)
        extra["by_stage"] = {lab: R(t, crn & np.isin(t.game, np.flatnonzero(
            (t.iteration >= lo) & (t.iteration <= hi))), G, S, C)
            for lab, lo, hi in STAGES}
        played_move = played_moves(t)
        extra["by_played_move"] = {
            int(m): R(t, crn & (played_move == m), G, S, C)
            for m in range(MOVES) if (crn & (played_move == m)).sum() >= 50}
        junc = _junction(t, crn)
        extra["junction_vs_corridor_acc"] = {
            a: {"junction": _wacc(t, f"{a}-s0", crn & junc),
                "corridor": _wacc(t, f"{a}-s0", crn & ~junc)}
            for a in ARMS if f"{a}-s0" in pm}
        extra["noop_R_every_arm_seed0"] = {
            a: R(t, (t.set == SET_NOOP) & dm, [f"{a}-s0"], ["S-s0"], C)["R"]
            for a in ARMS if f"{a}-s0" in pm and a != "S"}
        if p == "O0":
            extra["references"] = references(t, plan, dm)
        else:
            own = (t.set == SET_OWN) & dm & (t.o0_argmax >= 0) if hasattr(t, "o0_argmax") else None
            if own is not None:
                of_arg = []
                for i in np.flatnonzero(own):
                    g, d = t.game[i], int(t.decision[i])
                    ex = t.extra[g]
                    e = int(np.flatnonzero((ex["kind"] == 2) & (ex["decision"] == d))[0])
                    of_arg.append(int(np.argmax(ex["prior"][e])) == int(t.o0_argmax[i]))
                extra["o0_of_argmax_agreement"] = float(np.mean(of_arg)) if of_arg else float("nan")
        conds[p] = {"cond": cond, "extra": extra}
    o0 = conds["O0"]
    verdict, why = ladder(o0["cond"], plan, truth_failures,
                          o0["extra"]["contested_decisions"], o0["extra"]["effective_games"])
    rep["verdict"], rep["why"] = verdict, why
    if "OF" in conds:
        of = conds["OF"]
        floors_ok = (of["extra"]["contested_decisions"] >= FLOOR_DECISIONS
                     and of["extra"]["effective_games"] >= FLOOR_GAMES
                     and not of["cond"]["unread"])
        if not floors_ok:
            rep["five"] = "VOID"
        else:
            rep["five"] = "PASSES" if all(of["cond"]["holds"].values()) else "FAILS"
        rep["paired_drop"] = paired_drop(tables["O0"], tables["OF"], pm)
    rep["next"] = next_step(verdict, rep.get("five"))
    rep["players"] = {p: {"holds": c["cond"]["holds"], "unread": c["cond"]["unread"],
                          "i": c["cond"]["i"], "ii": c["cond"]["ii"],
                          "iii_crn": c["cond"]["iii_crn"], "iii_noop": c["cond"]["iii_noop"],
                          "iv": c["cond"]["iv"], **c["extra"]} for p, c in conds.items()}
    return rep


def _junction(t, crn):
    """Items of decisions with 3 or more CRN classes."""
    uk, inv = np.unique(t.dkey[crn], return_inverse=True)
    n = np.bincount(inv)
    return np.isin(t.dkey, uk[n >= 3])


def _wacc(t, key, m):
    w = t.weight[m]
    return float((t.correct(key)[m] * w).sum() / w.sum()) if m.any() else float("nan")


def played_moves(t):
    """Per item, its decision's played move (from the own item)."""
    own = t.set == SET_OWN
    pm = {}
    for i in np.flatnonzero(own):
        g = t.game[i]
        pm[int(t.dkey[i])] = int(t.truth[g]["played"][t.a[i]])
    return np.array([pm.get(int(k), -1) for k in t.dkey])


def diff_R(t, mask, a_keys, b_keys, base_keys, C):
    """R_a - R_b, both against base, per seed (seed i against seed i), with
    the interval whose seed term uses the per-seed differences."""
    ra = R(t, mask, a_keys, base_keys, C)
    rb = R(t, mask, b_keys, base_keys, C)
    per = np.array(ra["per_seed"]) - np.array(rb["per_seed"])
    # draws of the difference of seed-means
    bdiff = _boot_R(t, mask, a_keys, base_keys, C) - _boot_R(t, mask, b_keys, base_keys, C)
    point = float(per.mean())
    lo, hi = interval(point, bdiff, per)
    return {"diff": point, "lo": lo, "hi": hi, "per_seed": per.tolist()}


def _boot_R(t, mask, a_keys, b_keys, C):
    na, da, nb, db = [], [], [], []
    for ka, kb in zip(a_keys, b_keys):
        n, d = game_sums(t, mask, 1.0 - t.correct(ka), t.weight)
        na.append(n), da.append(d)
        n, d = game_sums(t, mask, 1.0 - t.correct(kb), t.weight)
        nb.append(n), db.append(d)
    return ratio_stat(*map(np.array, (na, da, nb, db)), C)[2]


def paired_drop(t0, t1, pm):
    """R(O0) - R(OF) per seed, O0's and OF's games resampled independently."""
    S, G = keys("S", pm), keys("SIGMA", pm)
    m0 = (t0.set == SET_CRN) & decision_mask(t0)
    m1 = (t1.set == SET_CRN) & decision_mask(t1)
    r0 = R(t0, m0, G, S, draws(t0.n_games))
    r1 = R(t1, m1, G, S, draws(t1.n_games))
    per = np.array(r0["per_seed"]) - np.array(r1["per_seed"])
    b = _boot_R(t0, m0, G, S, draws(t0.n_games, seed=SEED)) - \
        _boot_R(t1, m1, G, S, draws(t1.n_games, seed=SEED + 1))
    point = float(per.mean())
    lo, hi = interval(point, b, per)
    return {"drop": point, "lo": lo, "hi": hi, "per_seed": per.tolist()}


def cut_profile(t, crn, noop, G, S):
    """SIGMA cut to offsets j <= k (seed-mean R against S), CRN and NOOP."""
    out = {}
    for label, m in (("crn", crn), ("noop", noop)):
        prof = {}
        for k in range(16):
            rs = []
            for kg, ks in zip(G, S):
                ea = 1 - (t.correct(kg, cut=k)[m] * t.weight[m]).sum() / t.weight[m].sum()
                eb = 1 - (t.correct(ks)[m] * t.weight[m]).sum() / t.weight[m].sum()
                rs.append(1 - ea / eb)
            prof[k + 1] = float(np.mean(rs))
        out[label] = prof
    return out


def next_step(verdict, five):
    if verdict.startswith("GUESSES THE GAME"):
        if five in ("FAILS", "VOID"):
            return ("Kai's call: (5) " + five + " while (1) reads GUESSES THE GAME "
                    "(the last rule overrides the first)")
        return ("step 3, registered separately: the guesser (SIGMA, F_1 alongside) on "
                "Atari-HEAD's human Ms. Pac-Man, then a training run with guessed moves")
    if verdict in ("TRUTH FAILURE",):
        return "fix the data or truth stage and rerun it, before any training read"
    return "Kai's call, with reads (3) and (4)"


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--truth")
    ap.add_argument("--train")
    ap.add_argument("--logits")
    ap.add_argument("--boot", type=int, default=BOOT)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    base = HERE / "testh"
    rep = score(pathlib.Path(a.data or base / "data"), pathlib.Path(a.truth or base / "truth"),
                pathlib.Path(a.train or base / "train"), pathlib.Path(a.logits or base / "logits"),
                a.boot)
    out = pathlib.Path(a.out or base / "score.json")
    out.write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({k: rep[k] for k in ("verdict", "why", "next") if k in rep}, indent=1))
    print("five:", rep.get("five"))
    for p, r in rep["players"].items():
        print(f"{p}: R {r['i']['R']:.3f} [{r['i']['lo']:.3f}, {r['i']['hi']:.3f}] holds {r['holds']}"
              f" unread {r['unread']}")
    print("DONE")


if __name__ == "__main__":
    main(sys.argv[1:])
