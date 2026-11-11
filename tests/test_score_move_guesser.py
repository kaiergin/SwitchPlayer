"""Test H, stage D (CURRICULUM "Next experiment (pre-registered) -- Test H":
READS, VERDICT, BUILD NOTES). D2 on synthetic tables with hand-computed
answers: the interval (bootstrap percentiles and the t(4) seed term in
quadrature, each side), R per seed and its pairing, junction identification and
commitment (all right; a coin between turns about 0.5 on K; two answers in one
class not committed; a three-valued guesser at J4 = 0), the floors from the
truth alone, the commitment rule near the floor, every rung of the ladder in
order, the raw-class split, the READS exclusion. D1 on the CPU: block windows
(pushes, LIFE ENDS), a game's items against its record and truth, the exact
pixel search against brute force (and its own-game exclusion), the packing,
SIGMA's cut at 16 equal to its forward, and model_logits equal to the nets."""
import math
import pathlib
import sys

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROBES = ROOT / "benchmarks" / "probes"
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(PROBES))
import score_move_guesser as sc  # noqa: E402
import logits_move_guesser as lgm  # noqa: E402
import train_move_guesser as tg  # noqa: E402
import probe_move_guesser as mg  # noqa: E402


# --------------------------------------------------------------------------
# D2
# --------------------------------------------------------------------------
def test_interval_quadrature_each_side():
    boot = np.linspace(0.4, 0.8, 1001)                 # q2.5 = 0.41, q97.5 = 0.79
    seeds = np.array([0.5, 0.6, 0.7, 0.6, 0.6])
    lo, hi = sc.interval(0.6, boot, seeds)
    q_lo, q_hi = np.percentile(boot, 2.5), np.percentile(boot, 97.5)
    st = sc.T4 * seeds.std(ddof=1) / math.sqrt(5)
    assert lo == pytest.approx(0.6 - math.hypot(0.6 - q_lo, st))
    assert hi == pytest.approx(0.6 + math.hypot(q_hi - 0.6, st))


def _decision_items(set_code, classes, played_class, dec, w_override=None):
    """Items of one decision: classes is a list of member-move lists."""
    rows = []
    for c, mem in enumerate(classes):
        m = np.zeros(9, bool)
        m[mem] = True
        rows.append({"set": set_code, "decision": dec, "cls": c, "members": m,
                     "weight": len(mem) / 9, "played": c == played_class,
                     "end": 0, "a": dec})
    return rows


def _game(rows, logit_fns, name="g"):
    g = {"name": name, "iteration": 160}
    for k in ("set", "decision", "cls", "members", "weight", "played", "end", "a"):
        g[k] = np.array([r[k] for r in rows])
    n = len(rows)
    g["h0_in_fit"] = np.zeros(n, bool)
    g["h0_in_other"] = np.zeros(n, bool)
    g["whole_in_fit"] = np.zeros(n, bool)
    g["near_dup"] = np.full(n, 1000)
    for key, fn in logit_fns.items():
        g["logits/" + key] = np.stack([fn(r) for r in rows]).astype(np.float32)
    return g


def onehot(m):
    v = np.full(9, -5.0)
    v[m] = 5.0
    return v


T_CLASSES = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]          # 3 classes of 3


def test_R_per_seed_and_pairing():
    rows = []
    for d in range(40):
        rows += _decision_items(sc.SET_CRN, T_CLASSES, 0, d)
    # S always answers move 0: right on class 0 only, err 2/3
    # SIGMA seed i is right on every item of decisions d % 5 != i-ish
    fns = {f"S-s{s}": (lambda r: onehot(0)) for s in range(5)}
    for s in range(5):
        fns[f"SIGMA-s{s}"] = (lambda s: lambda r: onehot(r["members"].argmax())
                              if r["decision"] % 4 else onehot(0))(s)
    games = [_game(rows[i * 30:(i + 1) * 30], fns, f"g{i}") for i in range(4)]
    t = sc.Table(games)
    C = sc.draws(t.n_games, boot=200)
    m = t.set == sc.SET_CRN
    r = sc.R(t, m, sc.keys("SIGMA", fns), sc.keys("S", fns), C)
    # SIGMA errs only at d % 4 == 0 decisions (2/3 of those), S everywhere 2/3
    assert r["err_b"] == pytest.approx(2 / 3)
    assert r["err_a"] == pytest.approx(2 / 3 * 10 / 40)
    assert r["R"] == pytest.approx(0.75)
    assert all(x == pytest.approx(0.75) for x in r["per_seed"])
    assert r["lo"] <= 0.75 <= r["hi"]


def test_R_pairs_seeds_and_takes_ratios_of_sums():
    """Seeds and games that differ: per-seed R is 1 - (sum of SIGMA i's weighted
    errors / its weights) / (S i's), seed i against seed i, then the mean."""
    rows = []
    for d in range(60):
        rows += _decision_items(sc.SET_CRN, T_CLASSES, 0, d)
    # SIGMA seed s errs at decisions d % 6 < s (class-0 answer there); S seed s
    # errs everywhere except d % 5 < s (right answer there)
    fns = {}
    for s in range(5):
        fns[f"SIGMA-s{s}"] = (lambda s: lambda r: onehot(0) if r["decision"] % 6 < s
                              else onehot(r["members"].argmax()))(s)
        fns[f"S-s{s}"] = (lambda s: lambda r: onehot(r["members"].argmax())
                          if r["decision"] % 5 < s else onehot(0))(s)
    # games of unequal size
    cuts = [0, 30, 90, 180]
    games = [_game(rows[a:b], fns, f"g{i}") for i, (a, b) in enumerate(zip(cuts, cuts[1:]))]
    t = sc.Table(games)
    m = t.set == sc.SET_CRN
    r = sc.R(t, m, sc.keys("SIGMA", fns), sc.keys("S", fns), sc.draws(t.n_games, boot=50))
    want = []
    for s in range(5):
        ea = sum(2 / 3 for d in range(60) if d % 6 < s) / 60
        eb = sum(2 / 3 for d in range(60) if not d % 5 < s) / 60
        want.append(1 - ea / eb)
    assert r["per_seed"] == pytest.approx(want)       # distinct per seed: pairing pinned
    assert len(set(round(x, 9) for x in want)) == 5
    assert r["R"] == pytest.approx(np.mean(want))


def _jk(J_lo=0.3, K_lo=0.6, commit_ok=True, unread=False):
    return {"J_lo": J_lo, "K_lo": K_lo, "commit_ok": commit_ok, "unread": unread,
            "commits": [100] * 5, "commit_need": 100}


def _out(i=0.5, ii=0.01, iv=0.5, jk=None):
    jk = jk or {}
    return {"i": {"lo": i}, "ii": {"lo": ii}, "iv": {"lo": iv},
            "iii_crn": {"3": jk.get("c3", _jk()), "4": jk.get("c4", _jk())},
            "iii_noop": {"3": jk.get("n3", _jk()), "4": jk.get("n4", _jk())}}


def test_holds_at_the_bars():
    h, unread, failed = sc.holds(_out())
    assert all(h.values()) and not unread and not failed
    assert sc.holds(_out(i=0.4999))[0]["i"] is False
    assert sc.holds(_out(ii=0.0))[0]["ii"] is False              # strictly above 0
    assert sc.holds(_out(iv=0.4999))[0]["iv"] is False
    assert sc.holds(_out(jk={"c4": _jk(J_lo=0.25)}))[0]["iii"] is True   # at least 0.25
    assert sc.holds(_out(jk={"n3": _jk(J_lo=0.2499)}))[0]["iii"] is False
    h, _, failed = sc.holds(_out(jk={"c3": _jk(K_lo=0.5)}))      # strictly above 0.5
    assert h["iii"] is False and any("K3" in f for f in failed)
    h, _, failed = sc.holds(_out(jk={"n4": _jk(commit_ok=False)}))
    assert h["iii"] is False and any("commitments" in f for f in failed)
    h, unread, _ = sc.holds(_out(jk={"c4": _jk(unread=True)}))
    assert h["iii"] is False and unread == ["iii_crn J4/K4"]


def test_junction_identification_and_commitment():
    rows = []
    for d in range(4):
        rows += _decision_items(sc.SET_CRN, T_CLASSES, 0, d)
    # d0: all right; d1: classes 1 and 2 swapped (committed, not identified);
    # d2: two answers in one class (not committed); d3: all right
    def fn(r):
        d, c = r["decision"], r["cls"]
        if d == 1 and c in (1, 2):
            return onehot(6 if c == 1 else 3)
        if d == 2 and c == 2:
            return onehot(4)
        return onehot(r["members"].argmax())
    t = sc.Table([_game(rows, {"SIGMA-s0": fn})])
    out = sc.junction_counts(t, sc.SET_CRN, "SIGMA-s0", 3, np.ones(len(t.set), bool))
    assert out["junctions"].sum() == 4
    assert out["identified"].sum() == 2
    assert out["committed"].sum() == 3
    out4 = sc.junction_counts(t, sc.SET_CRN, "SIGMA-s0", 4, np.ones(len(t.set), bool))
    assert out4["junctions"].sum() == 0


def test_coin_between_turns_and_three_valued_guesser():
    rng = np.random.default_rng(0)
    four = [[0], [1, 2], [3, 4, 5], [6, 7, 8]]           # keep, reverse, two turns
    rows = []
    for d in range(2000):
        rows += _decision_items(sc.SET_CRN, four, 0, d)
    flips = rng.random(2000) < 0.5

    def coin(r):                                        # turns 2 and 3 swapped half the time
        c = r["cls"]
        if c in (2, 3) and flips[r["decision"]]:
            c = 5 - c
        return onehot([0, 1, 3, 6][c])

    def three(r):                                       # keep / reverse / "turned" (always turn 2)
        return onehot([0, 1, 3, 3][r["cls"]])
    t = sc.Table([_game(rows, {"SIGMA-s0": coin, "X-s0": three})])
    allm = np.ones(len(t.set), bool)
    c = sc.junction_counts(t, sc.SET_CRN, "SIGMA-s0", 4, allm)
    K = c["identified"].sum() / c["committed"].sum()
    assert c["committed"].sum() == 2000 and abs(K - 0.5) < 0.05
    x = sc.junction_counts(t, sc.SET_CRN, "X-s0", 4, allm)
    assert x["identified"].sum() == 0 and x["committed"].sum() == 0


def test_floors_and_commitment_rule():
    rows = []
    for d in range(110):                                # 110 junctions, J4 floor 100
        rows += _decision_items(sc.SET_CRN, [[0], [1, 2], [3, 4, 5], [6, 7, 8]], 0, d)
    # commits at 60 decisions (>= 55 = half the tier), right at all of them
    def fn(r):
        if r["decision"] < 60:
            return onehot(r["members"].argmax())
        return onehot(0)
    games = [_game(rows[i * 44:(i + 1) * 44], {f"SIGMA-s{s}": fn for s in range(5)}, f"g{i}")
             for i in range(10)]
    t = sc.Table(games)
    C = sc.draws(t.n_games, boot=100)
    res = sc.JK(t, sc.SET_CRN, sc.keys("SIGMA", {f"SIGMA-s{s}": 1 for s in range(5)}),
                C, np.ones(len(t.set), bool))
    r4 = res["4"]
    assert r4["junctions"] == 110 and r4["commit_need"] == 55.0
    assert r4["commit_ok"] and r4["K"] == pytest.approx(1.0)
    assert r4["unread"] is True                       # 10 games: effective games < 16
    assert res["3"]["unread"] is True                 # 110 < 300
    # one commitment fewer than half the tier fails the test
    def fn54(r):
        if r["decision"] < 54:
            return onehot(r["members"].argmax())
        return onehot(0)
    games = [_game(rows[i * 44:(i + 1) * 44], {f"SIGMA-s{s}": fn54 for s in range(5)},
                   f"g{i}") for i in range(10)]
    t = sc.Table(games)
    res = sc.JK(t, sc.SET_CRN, [f"SIGMA-s{s}" for s in range(5)], sc.draws(t.n_games, boot=50),
                np.ones(len(t.set), bool))
    assert res["4"]["commits"] == [54] * 5 and res["4"]["commit_ok"] is False


def _cond(lo, hi, holds=None, unread=()):
    h = {"i": lo >= 0.5, "ii": True, "iii": True, "iv": True}
    h.update(holds or {})
    failed = [k for k, v in h.items() if not v]
    return {"i": {"lo": lo, "hi": hi}, "holds": h, "unread": list(unread),
            "failed": failed}


PLAN = {"void": [], "flags": []}


def test_ladder_every_rung():
    assert sc.ladder(_cond(0.6, 0.9), PLAN, ["x"], 9000, 30)[0] == "TRUTH FAILURE"
    assert sc.ladder(_cond(0.6, 0.9), PLAN, [], 4000, 30)[0] == "VOID"
    assert sc.ladder(_cond(0.6, 0.9), PLAN, [], 9000, 12)[0] == "VOID"
    assert sc.ladder(_cond(0.6, 0.9), {"void": ["SIGMA-s2: training failed"], "flags": []},
                     [], 9000, 30)[0] == "VOID"
    assert sc.ladder(_cond(0.6, 0.9), {"void": ["F1-s2: training failed"], "flags": []},
                     [], 9000, 30)[0] == "GUESSES THE GAME"
    v, why = sc.ladder(_cond(0.6, 0.9), {"void": [], "flags": ["S-s1"]}, [], 9000, 30)
    assert v == "CAPPED" and "interval state beside it: GUESSES THE GAME" in why
    v = sc.ladder(_cond(0.6, 0.9), {"void": [], "flags": ["SIGMA-s1"]}, [], 9000, 30)[0]
    assert v.startswith("GUESSES THE GAME") and "flagged" in v
    v, why = sc.ladder(_cond(0.3, 0.9), {"void": [], "flags": ["SIGMA-s1"]}, [], 9000, 30)
    assert v == "CAPPED" and "interval state beside it: PARTLY" in why
    assert sc.ladder(_cond(0.6, 0.9), PLAN, [], 9000, 30)[0] == "GUESSES THE GAME"
    v, why = sc.ladder(_cond(0.6, 0.9, {"iii": False}, ["iii_crn J4/K4"]), PLAN, [], 9000, 30)
    assert v == "PARTLY" and "iii" in why and "UNREAD iii_crn J4/K4" in why
    assert sc.ladder(_cond(0.2, 0.9), PLAN, [], 9000, 30)[0] == "PARTLY"
    assert sc.ladder(_cond(0.0, 0.9), PLAN, [], 9000, 30)[0] == "NO HELP"
    assert sc.ladder(_cond(-0.2, 0.1), PLAN, [], 9000, 30)[0] == "NO HELP"
    assert sc.ladder(_cond(-0.4, -0.1), PLAN, [], 9000, 30)[0] == "WORSE THAN HABITS"


def test_next_step_precedence():
    assert sc.next_step("GUESSES THE GAME", "PASSES").startswith("step 3")
    assert "F_1 is offered" in sc.next_step("GUESSES THE GAME", "PASSES", f1_meets=True)
    assert "F_1 is offered" not in sc.next_step("GUESSES THE GAME", "PASSES")
    assert sc.next_step("GUESSES THE GAME", "FAILS").startswith("Kai's call")
    assert sc.next_step("GUESSES THE GAME", "VOID").startswith("Kai's call")
    assert sc.next_step("GUESSES THE GAME", None).startswith("Kai's call")
    assert sc.next_step("PARTLY", "PASSES").startswith("Kai's call")


def test_raw_split_and_exclusion():
    rows = _decision_items(sc.SET_CRN, [[0, 1, 2, 3], [4, 5, 6, 7, 8]], 0, 0)
    rows += _decision_items(sc.SET_OWN, [[0, 1, 2, 3]], 0, 0)
    g = _game(rows, {"S-s0": lambda r: onehot(1)})
    t = sc.Table([g])
    t.truth = [{"class_crn_raw": np.array([[0, 0, 1, 1, 2, 2, 2, 2, 2]])}]
    games, correct, weight = sc.raw_items(t, t.set == sc.SET_CRN, "S-s0")
    # pooled class 0 splits into raw {0,1} and {2,3}; class 1 stays whole
    assert sorted(weight.tolist()) == sorted([2 / 9, 2 / 9, 5 / 9])
    assert correct.tolist().count(True) == 1           # move 1 is in raw {0, 1} only
    # the exclusion: an own item flagged drops every item of its decision
    t.h0_in_other = np.array([False, False, True])
    assert not sc.decision_mask(t).any()
    assert sc.decision_mask(t, undo=True).all()


# --------------------------------------------------------------------------
# D1
# --------------------------------------------------------------------------
def test_block_window_pushes_and_repeats():
    rows = lgm.block_window([1, 2, 3, 4], 100, 3)
    assert rows[0] == [1, 2, 3, 4]
    assert rows[1] == [2, 3, 4, 100] and rows[3] == [4, 100, 101, 102]
    assert all(r == rows[3] for r in rows[3:])
    assert len(rows) == tg.K_MAX + 1


def _arr(n_moves=20):
    kinds = [0] + [1] * 2 + [2] * n_moves
    ends = [0] * len(kinds)
    ends[-1] = 1
    n = len(kinds)
    dec = np.full(n, -1, np.int32)
    dec[3:] = np.arange(n_moves)
    return {"kind": np.array(kinds, np.int8), "end": np.array(ends, np.int8),
            "move": np.arange(n, dtype=np.int8) % 9, "decision": dec,
            "frames": np.stack([np.full((3, 84, 84), i, np.uint8) for i in range(n)])}


def test_game_items_against_record_and_truth():
    arr = _arr()
    # truth for decisions 0..2: d0 contested (crn 3 classes, noop 2), d1 not, d2 contested
    crn = np.array([[0, 0, 0, 1, 1, 1, 2, 2, 2], [0] * 9, [0, 1, 1, 1, 1, 1, 1, 1, 1]], np.int8)
    noop = np.array([[0, 0, 0, 0, 1, 1, 1, 1, 1], [0] * 9, [0, 0, 1, 1, 1, 1, 1, 1, 1]], np.int8)
    played = np.array([4, 0, 0], np.int8)
    blocks = []                                        # (a, kind, cls, len)
    for a in (0, 2):
        for c in sorted(set(crn[a].tolist())):
            if c != crn[a][played[a]]:
                blocks.append((a, 0, c, 16))
        for c in sorted(set(noop[a].tolist())):
            blocks.append((a, 1, c, 5))
    nb = len(blocks)
    truth = {"decision": np.array([0, 1, 2]), "played": played,
             "contested": np.array([True, False, True]),
             "class_crn": crn, "class_noop": noop,
             "crn_len": np.full((3, 9), 16, np.int8), "crn_end": np.zeros((3, 9), np.int8),
             "blk_dec": np.array([b[0] for b in blocks]), "blk_kind": np.array([b[1] for b in blocks]),
             "blk_class": np.array([b[2] for b in blocks]), "blk_len": np.array([b[3] for b in blocks]),
             "blk_end": np.array([1 if b[1] == 1 else 0 for b in blocks]),
             "blk_frames": np.stack([np.full((16, 3, 84, 84), 200 + i, np.uint8)
                                     for i in range(nb)])}
    frames, items = lgm.game_items(arr, truth)
    assert len(frames) == len(arr["kind"]) + nb * 16
    own = items["set"] == lgm.SET_OWN
    assert own.sum() == 3
    for s in (lgm.SET_CRN, lgm.SET_NOOP):
        for d in (0, 2):
            m = (items["set"] == s) & (items["decision"] == d)
            assert items["weight"][m].sum() == pytest.approx(1.0)
            assert (items["members"][m].sum(0) == 1).all()       # a partition of the 9
        assert not ((items["set"] == s) & (items["decision"] == 1)).any()
    _, _, _, win = tg.decision_windows(arr)
    # the played CRN class's item is the recorded window
    m = (items["set"] == lgm.SET_CRN) & items["played"] & (items["decision"] == 0)
    assert np.array_equal(items["window"][m][0], win[0])
    # a NOOP item of length 5: rows past 5 repeat row 5, row 1 holds the block's first frame
    m = (items["set"] == lgm.SET_NOOP) & (items["decision"] == 0)
    w = items["window"][m][0]
    assert int(frames[w[1][-1]][0, 0, 0]) >= 200
    assert all((w[k] == w[5]).all() for k in range(5, 17))
    assert (w[0] == win[0][0]).all()
    # every block item reads exactly its own block (fill value 200 + b)
    for b, (a, kind, c, _) in enumerate(blocks):
        s = lgm.SET_CRN if kind == 0 else lgm.SET_NOOP
        m = (items["set"] == s) & (items["decision"] == a) & (items["cls"] == c)
        assert m.sum() == 1
        w = items["window"][m][0]
        assert int(frames[w[1][-1]][0, 0, 0]) == 200 + b


def test_pixel_search_exact():
    rng = np.random.default_rng(1)
    q = rng.integers(0, 3, (7, 3, 84, 84)).astype(np.uint8)
    r = rng.integers(0, 3, (23, 3, 84, 84)).astype(np.uint8)
    r[5] = q[2]
    r[9] = q[4]
    r[9, 1, 10, 10] ^= 1                                # one pixel, one channel apart
    got = lgm.min_pixel_diff(q, r, torch.device("cpu"), q_chunk=3, r_chunk=8)
    pq, pr = lgm.packed(q), lgm.packed(r)
    want = np.array([(pq[i][None] != pr).sum(1).min() for i in range(len(q))])
    assert np.array_equal(got, want) and got[2] == 0 and got[4] == 1
    ex = lgm.min_pixel_diff(q, r, torch.device("cpu"), q_chunk=3, r_chunk=8,
                            exclude=[(5, 6)] * len(q))
    assert ex[2] > 0
    a = np.zeros((2, 3, 84, 84), np.uint8)
    a[1, 2, 0, 0] = 1
    assert (lgm.packed(a)[0] != lgm.packed(a)[1]).sum() == 1


def test_cut_and_model_logits_on_the_cpu():
    torch.manual_seed(0)
    frames = torch.randint(0, 255, (40, 3, 84, 84), dtype=torch.uint8)
    win = np.random.default_rng(0).integers(0, 40, (6, 17, 4))
    nets = {"S-s0": tg.Guesser(tg.ARMS["S"]), "SIGMA-s0": tg.Guesser(tg.ARMS["SIGMA"])}
    with torch.no_grad():
        nets["SIGMA-s0"].w.copy_(torch.randn(16))
    sets = np.array([0, 1, 1, 2, 2, 0])
    out = lgm.model_logits(nets, frames, win, sets, torch.device("cpu"))
    wt = torch.from_numpy(win)
    with torch.no_grad():
        for k, net in nets.items():
            ref = net(tg.gather(frames, wt, net.offsets)).numpy()
            assert np.allclose(out[k], ref, atol=1e-4)
        z = lgm.sigma_features(nets["SIGMA-s0"], tg.gather(frames, wt, tg.ARMS["SIGMA"]))
        cuts = lgm.cut_logits(nets["SIGMA-s0"], z).numpy()
    assert np.allclose(cuts[-1], out["SIGMA-s0"], atol=1e-4)
    assert np.allclose(out["SIGMA-s0/cut"][:, sets != 0], cuts[:, sets != 0], atol=1e-4)
    assert not out["SIGMA-s0/cut"][:, sets == 0].any()
    # the cut at 1 is the first pair alone
    with torch.no_grad():
        one = nets["SIGMA-s0"](tg.gather(frames, wt, tg.ARMS["SIGMA"]), keep={1}).numpy()
    assert np.allclose(cuts[0], one, atol=1e-4)
