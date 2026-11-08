"""Test G, read (7): is there anything for exploration or search to find?
(CURRICULUM "Next experiment (pre-registered) -- Test G", read (7).) Pins:
the registered numbers and the script's environment (stage 2a's); the fifths
drawn inside strata by rho_w's ranks (ties averaged, bounds inclusive), from
the fixed seed, 120 a fifth; each class played by its highest acting-prior
slot, so the deployed slot stands for its class; ROOM's choose / score split
and its tie rules (the deployed class, then the lowest played slot), mean
class minus deployed, the search gains, all on the primary head; the
bootstrap's unit (the drawn states' clusters, whole) and an undefined draw;
the refusal without stage 2b's summary or its gates. On the real game with
stand-in policies: the deployed class at delay 0 is stage 2a's branch, the
slot comes before the delay and the delay before the policy, the delay's
reward joins the next base step, a life lost in the slot or the delay, the
cap, the census's classes (its window, T0's vocabulary in slot order) and
one branch per class and delay. On small real nets: the search runs on the
trainer's eval-mode BN, deep-copied before the acting overlay; the raw prior
is one forward pass on the same BN, the acting prior after the overlay;
noise and ties seeded per chunk, the same noise at both budgets; 50 and 200
simulations. Whole reads in-process on stage 2b's real outputs over stage
2b's own stub inputs (stand-in play and search): the route, the path checks
(VOID), resume, stamps, the summary never overwritten, read-only inputs.
Slow: T0's real snapshot on two of the pilot's states (the deployed class's
delay-0 branch against the pilot's own) and a real search of 8 roots at 50
and 200 simulations, timed, its peak memory measured."""
import contextlib
import copy
import hashlib
import json
import os
import pathlib
import shutil
import stat
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROBES = ROOT / "benchmarks" / "probes"
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(PROBES))
import ale_py  # noqa: E402
import muzero  # noqa: E402
import muzero_vtree  # noqa: E402
import ablation  # noqa: E402
import probe_action_census as pc  # noqa: E402
import probe_e3a_sequences as e3  # noqa: E402
import probe_gap_check as gc  # noqa: E402
import probe_gap_nets as gn  # noqa: E402
import probe_gap_stage2a as s2  # noqa: E402
import probe_gap_stage2b as s2b  # noqa: E402
import probe_gap_read7 as r7  # noqa: E402
import test_gap_stage2b as t2b  # noqa: E402
from test_vtree import dezero_heads  # noqa: E402

VOCAB = ((0, 4), (1, 8), (2, 16), (3, 32), (4, 4), (2, 4), (3, 8))
SMALL_CAP = 192
DRAWS = 300


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def listing(d):
    out = {}
    for p in sorted(pathlib.Path(d).rglob("*")):
        if p.is_file():
            st = p.stat()
            out[str(p.relative_to(d))] = (st.st_size, st.st_mtime_ns,
                                          hashlib.blake2b(p.read_bytes(),
                                                          digest_size=16)
                                          .digest())
    return out


def read_only(d, on=True):
    for p in pathlib.Path(d).iterdir():
        os.chmod(p, stat.S_IREAD if on else stat.S_IREAD | stat.S_IWRITE)


def _forbidden(*a, **k):
    raise AssertionError("must not run")


# --------------------------------------------------------------------------
# the module and its numbers
# --------------------------------------------------------------------------
def test_the_registered_numbers():
    assert r7.PER_FIFTH == 120
    assert r7.FIFTH_TOP == (4, 5) and r7.FIFTH_BOTTOM == (1, 5)
    assert r7.CHOOSE == (0, 1, 2, 3) and r7.SCORE == (8, 9, 10, 11)
    assert r7.DELAYS == (0, 1, 2, 3, 8, 9, 10, 11)
    assert not set(r7.CHOOSE) & set(r7.SCORE)
    assert r7.SIMS == (50, 200) and r7.ROOTS == 8
    assert r7.FRAME_CAP == gc.FRAME_CAP == 1536
    assert r7.PRIMARY == (0, 0)
    assert gc.SCALES[r7.PRIMARY[0]] == "clip"
    assert gc.DISCOUNTS[r7.PRIMARY[1]] == 0.997
    assert r7.DRAWS == 10_000 and r7.LIMIT == 0 and r7.CHUNK == 40
    assert r7.CHUNK % r7.ROOTS == 0 and not r7.SMOKE
    assert s2.INTERVAL == (2.5, 97.5)
    assert r7.SEARCH_NETS == ("representation", "dynamics", "prediction",
                              "value_prefix")
    # its own streams: not the pilot's (1-3), stage 2a's (4) or 2b's (5)
    streams = {r7.DRAW_SEED[1], r7.BOOT_SEED[1], r7.NOISE_STREAM,
               r7.TIE_STREAM}
    assert len(streams) == 4 and not streams & {1, 2, 3, 4, 5}
    assert r7.DRAW_SEED[0] == r7.BOOT_SEED[0] == gc.SEED
    assert s2.BOOT_SEED == (gc.SEED, 4) and s2b.BOOT_SEED == (gc.SEED, 5)
    assert r7.CROSS_TOL == s2b.CROSS_TOL == 1e-6
    assert r7.PILOT == pathlib.Path(
        r"Y:\SwitchPlayer-testg-run\benchmarks\probes\probe_gap_check-pilot")
    assert r7.GAP == pathlib.Path(
        r"Y:\SwitchPlayer-testg2a-run\benchmarks\probes\probe_gap_stage2a")
    assert r7.NETS == pathlib.Path(
        r"Y:\SwitchPlayer-testg-nets-run\benchmarks\probes\probe_gap_nets-out")
    assert r7.STAGE2B == pathlib.Path(
        r"Y:\SwitchPlayer-testg2b-run\benchmarks\probes\probe_gap_stage2b")


def test_the_birth_search_takes_the_same_four_nets():
    src = (ROOT / "experiments" / "ablation.py").read_text()
    body = src.split("def _birth_search(", 1)[1].split("modes = ", 1)[0]
    for n in r7.SEARCH_NETS:
        assert f'"{n}": self.trainer.{n}' in body
    assert "projector" not in body


def test_the_script_runs_in_stage_2as_environment():
    assert r7.ENV == s2.ENV == s2b.ENV


def test_the_source_hash_covers_this_read_and_the_engine():
    assert {p.name for p in r7.SOURCES} == {"probe_gap_read7.py",
                                           "muzero_vtree.py"}
    assert r7.SOURCE == gc.source_hash(r7.SOURCES)


def test_importing_the_read_changes_nothing_process_wide():
    code = ("import os, sys; sys.path[:0] = [{p!r}, {e!r}, {r!r}]; "
            "import torch; threads = torch.get_num_threads(); "
            "import ablation; before = (ablation.ENV_NAME, ablation.ACTIONS, "
            "str(ablation.DEVICE)); "
            "import probe_gap_read7; "
            "assert (ablation.ENV_NAME, ablation.ACTIONS, "
            "str(ablation.DEVICE)) == before; "
            "assert torch.get_num_threads() == threads; "
            "assert 'probe_exhaustive_eval' not in sys.modules; "
            "assert 'probe_prefix_bn' not in sys.modules; "
            "assert os.environ.get('PROBE_ACTING_STATES') is None; "
            "assert os.environ.get('PROBE_ENV') is None; "
            "assert os.environ.get('PROBE_THREADS') is None; "
            "print('ok')").format(p=str(PROBES), e=str(ROOT / "experiments"),
                                  r=str(ROOT))
    env = {k: v for k, v in os.environ.items()
           if k not in ("PROBE_ACTING_STATES", "PROBE_ENV", "PROBE_THREADS")}
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=env,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-2000:]


# --------------------------------------------------------------------------
# the fifths and the draw
# --------------------------------------------------------------------------
def masks(g, sid):
    t, b = r7.fifth_masks(g, sid)
    return t.tolist(), b.tolist()


def test_the_fifths_by_hand():
    assert masks([4.0], [0]) == ([False], [False])
    assert masks([1.0, 2.0], [0, 0]) == ([False] * 2, [False] * 2)
    assert masks([3.0, 1.0, 2.0], [0] * 3) == ([True, False, False],
                                               [False, True, False])
    assert masks([5., 1., 3., 2., 4.], [0] * 5) == (
        [True, False, False, False, False], [False, True, False, False, False])
    t, b = masks(list(range(10)), [0] * 10)
    assert t == [False] * 8 + [True] * 2 and b == [True] * 2 + [False] * 8


def test_ties_are_averaged_and_the_bounds_are_inclusive():
    # two tied at the top of five: average rank 4.5, (4.5 - 0.5) / 5 = 4/5
    t, b = masks([0., 1., 2., 3., 3.], [0] * 5)
    assert t == [False, False, False, True, True]
    assert b == [True, False, False, False, False]
    # two tied at the bottom: average rank 1.5, (1.5 - 0.5) / 5 = 1/5
    t, b = masks([0., 0., 1., 2., 3.], [0] * 5)
    assert b == [True, True, False, False, False]
    assert t == [False, False, False, False, True]
    assert masks([1.] * 5, [0] * 5) == ([False] * 5, [False] * 5)


def test_the_fifths_are_rho_ws_centred_ranks():
    rng = np.random.default_rng(0)
    g, sid = rng.normal(size=400), rng.integers(0, 30, 400)
    t, b = r7.fifth_masks(g, sid)
    c = np.zeros(len(g))
    for k in set(sid.tolist()):
        m = sid == k
        c[m] = gc.centred_ranks(g[m])
    assert (t == (c >= 0.3 - 1e-12)).all() and (b == (c <= -0.3 + 1e-12)).all()
    assert t.any() and b.any() and not (t & b).any()


def test_the_fifths_are_ranked_inside_strata():
    rng = np.random.default_rng(1)
    sid = np.repeat(np.arange(6), 10)
    g = rng.normal(size=60)
    t, b = r7.fifth_masks(g, sid)
    shifted = g + 100.0 * sid                 # a stratum's level moves nothing
    t2, b2 = r7.fifth_masks(shifted, sid)
    assert (t == t2).all() and (b == b2).all()
    for k in range(6):
        assert t[sid == k].sum() == 2 and b[sid == k].sum() == 2
    # ranked across strata, the top fifth would be the last stratum's alone
    t3, _ = r7.fifth_masks(shifted, np.zeros(60, int))
    assert t3[sid == 5].sum() > 2 and not t3[sid < 4].any()


def test_g_must_be_finite():
    with pytest.raises(SystemExit, match="not finite"):
        r7.fifth_masks([0.0, np.nan, 1.0], [0, 0, 0])


def test_the_draw_is_seeded_and_inside_the_fifths():
    rng = np.random.default_rng(2)
    n = 2000
    sid, g = rng.integers(0, 60, n), rng.normal(size=n)
    t, b = r7.fifth_masks(g, sid)
    d, again = r7.draw_fifths(g, sid), r7.draw_fifths(g, sid)
    for f, mask in (("top", t), ("bottom", b)):
        idx, size = d[f]
        assert size == mask.sum() > 120
        assert len(idx) == len(set(idx.tolist())) == 120
        assert mask[idx].all() and (np.diff(idx) > 0).all()
        assert (idx == again[f][0]).all()
    assert not set(d["top"][0].tolist()) & set(d["bottom"][0].tolist())
    by_hand = np.random.default_rng(r7.DRAW_SEED)
    want_top = np.sort(by_hand.choice(np.flatnonzero(t), 120, replace=False))
    want_bot = np.sort(by_hand.choice(np.flatnonzero(b), 120, replace=False))
    assert (d["top"][0] == want_top).all()
    assert (d["bottom"][0] == want_bot).all()
    other = r7.draw_fifths(g, sid, seed=(gc.SEED, 99))
    assert (other["top"][0] != d["top"][0]).any()


def test_a_small_fifth_is_taken_whole():
    g, sid = np.arange(30.0), np.repeat(np.arange(3), 10)
    d = r7.draw_fifths(g, sid)
    assert d["top"][1] == 6 and len(d["top"][0]) == 6
    assert d["bottom"][1] == 6 and len(d["bottom"][0]) == 6


# --------------------------------------------------------------------------
# classes and the per-state reads
# --------------------------------------------------------------------------
def test_each_class_is_played_by_its_highest_prior_slot():
    part = [0, 1, 0, 2, 1, 0]
    prior = [0.1, 0.05, 0.3, 0.2, 0.05, 0.3]
    # class 0: slots 2 and 5 tie at 0.3, the lowest; class 1: 1 and 4 tie
    assert r7.representatives(part, prior) == [2, 1, 3]
    rng = np.random.default_rng(3)
    for _ in range(300):
        m = int(rng.integers(1, 6))
        part = pc.partition(rng.integers(0, m, 9).tolist())
        prior = rng.dirichlet(np.ones(9))
        if rng.random() < 0.3:
            prior = np.round(prior, 1)            # ties
        reps = r7.representatives(part, prior)
        dep = int(np.argmax(prior))
        assert reps[part[dep]] == dep             # the deployed slot stands
        for c, s in enumerate(reps):
            members = [i for i, x in enumerate(part) if x == c]
            assert part[s] == c and prior[s] == max(prior[i] for i in members)


def record(choose, score, deployed, part, reps, other=0.0):
    """A play record: class c returns choose[c] on each of delays 0-3 and
    score[c] on each of 8-11 at the primary head; `other` on every other
    head."""
    m = len(reps)
    ret = np.full((m, len(r7.DELAYS), 2, 4), float(other))
    for c in range(m):
        ret[c, :4, 0, 0], ret[c, 4:, 0, 0] = choose[c], score[c]
    return {"partition": part, "deployed": deployed, "reps": reps,
            "delays": list(r7.DELAYS), "returns": ret}


TOP_DEP = {"raw": 4, "50": 4, "200": 4}


def test_room_is_zero_when_the_deployed_class_chooses_itself():
    part, reps = [0, 1, 0, 1, 2], [2, 1, 4]
    r = r7.state_reads(record([1, 0, 5], [0, 9, 3], 4, part, reps), TOP_DEP)
    assert r["room"] == 0.0 and r["moved"] == 0.0 and r["chosen"] == 2
    # the scoring delays' best (class 1) does not choose
    assert r["mean_minus_deployed"] == pytest.approx(4.0 - 3.0)


def test_room_scores_the_chosen_class_on_the_scoring_delays():
    part, reps = [0, 1, 0, 1, 2], [2, 1, 4]
    r = r7.state_reads(record([1, 7, 5], [0, -2, 3], 4, part, reps), TOP_DEP)
    assert r["chosen"] == 1 and r["moved"] == 1.0
    assert r["room"] == pytest.approx(-2.0 - 3.0)     # it can be negative


def test_room_ties_go_to_the_deployed_class_then_the_lowest_slot():
    part, reps = [0, 1, 0, 1, 2], [2, 1, 4]
    # a tie with the deployed class: the deployed class
    r = r7.state_reads(record([1, 5, 5], [0, 9, 3], 4, part, reps), TOP_DEP)
    assert r["chosen"] == 2 and r["room"] == 0.0
    # two others tie: the class whose played slot is lowest (class 1's slot
    # 1, not class 0, the lower class id, whose slot is 2)
    r = r7.state_reads(record([5, 5, 1], [10, 20, 0], 4, part, reps), TOP_DEP)
    assert r["chosen"] == 1 and r["room"] == pytest.approx(20.0)


def test_no_choice_reads_a_scoring_delay_nor_a_score_a_choosing_one():
    part, reps = [0, 1, 0, 1, 2], [2, 1, 4]
    rec = record([1, 2, 0], [0, 4, 3], 4, part, reps)
    base = r7.state_reads(rec, TOP_DEP)
    assert base["chosen"] == 1 and base["room"] == pytest.approx(1.0)
    up = copy.deepcopy(rec)
    up["returns"][0, 4:, 0, 0] = 100.0          # class 0 best on scoring only
    assert r7.state_reads(up, TOP_DEP)["chosen"] == 1
    down = copy.deepcopy(rec)
    down["returns"][1, :4, 0, 0] = 50.0         # class 1's choosing delays
    assert r7.state_reads(down, TOP_DEP)["room"] == pytest.approx(1.0)
    # the delays are found by value, not position (choosing and scoring
    # disagree here: class 1 chooses, class 0 scores best)
    rec = record([1, 2, 0], [5, 4, 3], 4, part, reps)
    base = r7.state_reads(rec, TOP_DEP)
    assert base["chosen"] == 1 and base["room"] == pytest.approx(1.0)
    perm = copy.deepcopy(rec)
    order = [7, 6, 5, 4, 3, 2, 1, 0]
    perm["delays"] = [r7.DELAYS[i] for i in order]
    perm["returns"] = rec["returns"][:, order]
    assert r7.state_reads(perm, TOP_DEP) == base


def test_the_reads_are_on_the_primary_head_only():
    part, reps = [0, 1, 0, 1, 2], [2, 1, 4]
    a = r7.state_reads(record([1, 2, 0], [0, 4, 3], 4, part, reps), TOP_DEP)
    b = r7.state_reads(record([1, 2, 0], [0, 4, 3], 4, part, reps,
                              other=1e6), TOP_DEP)
    assert a == b


def test_the_search_gains_score_the_top_slots_class():
    part, reps = [0, 1, 0, 1, 2], [2, 1, 4]
    rec = record([1, 0, 5], [6, 9, 3], 4, part, reps)
    # raw: slot 0 is class 0 (played as slot 2); 50: slot 3 is class 1;
    # 200: the deployed slot
    r = r7.state_reads(rec, {"raw": 0, "50": 3, "200": 4})
    assert r["gain_raw"] == pytest.approx(6 - 3) and r["moved_raw"] == 1.0
    assert r["gain_50"] == pytest.approx(9 - 3) and r["moved_50"] == 1.0
    assert r["gain_200"] == 0.0 and r["moved_200"] == 0.0
    assert r["mean_minus_deployed"] == pytest.approx(6.0 - 3.0)
    assert r["classes"] == 3.0
    # against the deployed class, not the class ROOM chose (class 1 here)
    rec = record([1, 7, 5], [6, 9, 3], 4, part, reps)
    r = r7.state_reads(rec, {"raw": 0, "50": 3, "200": 4})
    assert r["chosen"] == 1 and r["room"] == pytest.approx(9 - 3)
    assert r["gain_raw"] == pytest.approx(6 - 3) and r["gain_200"] == 0.0


# --------------------------------------------------------------------------
# top against bottom: the bootstrap
# --------------------------------------------------------------------------
def fifth_world(n=60, seed=4):
    rng = np.random.default_rng(seed)
    fifth = np.array(["top"] * (n // 2) + ["bottom"] * (n - n // 2))
    fifth = fifth[rng.permutation(n)]
    clusters = rng.integers(0, n // 3, n)
    v = rng.normal(size=n) + (fifth == "top")
    return fifth, clusters, v


def test_the_bootstrap_resamples_whole_clusters_by_hand():
    fifth, clusters, v = fifth_world()
    res = r7.boot({"room": v}, fifth, clusters, draws=DRAWS)["room"]
    top = fifth == "top"
    assert res["point"]["top"] == pytest.approx(v[top].mean())
    assert res["point"]["bottom"] == pytest.approx(v[~top].mean())
    assert res["point"]["top_minus_bottom"] == pytest.approx(
        v[top].mean() - v[~top].mean())
    assert res["point"]["both"] == pytest.approx(v.mean())
    by_hand = {q: [] for q in r7.PARTS}
    for idx in s2.cluster_draws(clusters, DRAWS, seed=r7.BOOT_SEED):
        t, b = idx[top[idx]], idx[~top[idx]]
        by_hand["top"].append(v[t].mean())
        by_hand["bottom"].append(v[b].mean())
        by_hand["top_minus_bottom"].append(v[t].mean() - v[b].mean())
        by_hand["both"].append(v[idx].mean())
    for q in r7.PARTS:
        assert res["intervals"][q] == pytest.approx(
            list(np.percentile(by_hand[q], (2.5, 97.5))))
    assert res["undefined"] == {q: 0 for q in r7.PARTS}


def test_one_cluster_leaves_no_spread():
    fifth, _, v = fifth_world()
    res = r7.boot({"room": v}, fifth, [7] * len(v), draws=50)["room"]
    for q in r7.PARTS:
        assert res["intervals"][q] == pytest.approx([res["point"][q]] * 2)


def test_each_state_is_not_its_own_unit():
    fifth, _, v = fifth_world()
    whole = r7.boot({"room": v}, fifth, [0] * 30 + [1] * 30, draws=200)
    alone = r7.boot({"room": v}, fifth, np.arange(60), draws=200)
    assert whole["room"]["intervals"] != alone["room"]["intervals"]


def test_a_draw_without_a_fifth_is_undefined():
    fifth, _, v = fifth_world()
    clusters = (fifth == "top").astype(int)        # each fifth one cluster
    res = r7.boot({"room": v}, fifth, clusters, draws=200)["room"]
    assert res["intervals"]["top"] is None
    assert res["intervals"]["top_minus_bottom"] is None
    assert res["undefined"]["top_minus_bottom"] > 0
    assert res["intervals"]["both"] is not None


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------
def rec2b(**over):
    rec = {"verdict": "NOT SHOWN",
           "gates": {"verdict": "PASS-SO-FAR", "G3": {"ok": True},
                     "G5": {"ok": True}},
           "G2_first": {"clause": "first", "verdict": gn.G2_FIRST_PASS},
           "reads": {"primary": {"intervals": {"g2_master": [0.185, 0.333],
                                               "g2_twin": [0.282, 0.419]}}}}
    for k, v in over.items():
        if k == "intervals":
            rec["reads"]["primary"]["intervals"].update(v)
        elif k in ("G3", "G5"):
            rec["gates"][k] = v
        elif k == "gates_verdict":
            rec["gates"]["verdict"] = v
        elif k == "g2":
            rec["G2_first"].update(v)
        else:
            rec[k] = v
    return rec


def test_stage_2bs_gates_pass_on_holds_or_not_shown():
    assert r7.stage2b_reasons(rec2b()) == []
    assert r7.stage2b_reasons(rec2b(verdict="HOLDS")) == []


@pytest.mark.parametrize("over, says", [
    ({"verdict": "VOID"}, "verdict is 'VOID'"),
    ({"verdict": None}, "verdict is None"),
    ({"gates_verdict": "VOID"}, "stage 2a"),
    ({"G3": {"ok": False}}, "G3"),
    ({"G5": {"ok": False}}, "G5"),
    ({"g2": {"verdict": gn.G2_FIRST_VOID}}, "first clause"),
    ({"g2": {"clause": "inside"}}, "first clause"),
    ({"intervals": {"g2_master": [0.0, 0.3]}}, "g2_master"),
    ({"intervals": {"g2_twin": [-0.1, 0.3]}}, "g2_twin"),
    ({"intervals": {"g2_twin": None}}, "g2_twin"),
    ({"reads": {}}, "g2_master"),
])
def test_stage_2bs_gates_refuse(over, says):
    reasons = r7.stage2b_reasons(rec2b(**over))
    assert reasons and any(says in r for r in reasons), reasons


# --------------------------------------------------------------------------
# the real game, stand-in policies, no snapshot
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def env():
    e = gc.raw_env()
    yield e
    e.close()


@pytest.fixture
def vocab(monkeypatch):
    monkeypatch.setattr(muzero, "TOKEN_VOCAB", VOCAB)
    monkeypatch.setattr(muzero, "USE_SEARCH_ACTING", False)


def prior_of(salt=b""):
    def prior(stack):
        h = hashlib.blake2b(salt + stack.tobytes(), digest_size=8).digest()
        rng = np.random.default_rng(int.from_bytes(h, "little"))
        return softmax(2.0 * rng.normal(size=len(muzero.TOKEN_VOCAB)))
    return prior


def decide_of(prior):
    return lambda stack: int(np.argmax(prior(stack)))


def live_state(env, frames=200, seed=0):
    stack, *_ = gc.start_game(env, gc.game_noops(seed),
                              gc.opening_moves(seed, 80))
    ale = env.unwrapped.ale
    stack, _, lost = gc.raw_frames(ale, env.unwrapped._action_set[2], frames,
                                   stack)
    assert not lost
    return {"state": ale.cloneState().serialize(), "stack": stack}


def before_death(env, k):
    """Standing still from the start: the state from which the k-th frame
    loses the first life."""
    u = env.unwrapped
    ale = u.ale
    u.reset(seed=gc.EVAL_SEED)
    u.reset()
    held = []
    while ale.lives() == 3:
        held = (held + [ale.cloneState()])[-k:]
        ale.act(u._action_set[0])
    stack = ablation.initial_frame_stack(np.zeros((3, 84, 84), dtype=np.uint8))
    return {"state": held[0].serialize(), "stack": stack}


def replay_end(env, s, b):
    """A played branch's inputs on raw ALE in the registered order -- the
    slot, the delay's NOOPs, then the policy's tokens -- up to its frames:
    the end state's digest."""
    actions = list(env.unwrapped._action_set)
    inputs = e3.input_frames(muzero.token_inputs(VOCAB[b["slot"]]), actions)
    inputs += [actions[0]] * b["delay"]
    for t in b["tokens"][1:]:
        inputs += e3.input_frames(muzero.token_inputs(VOCAB[t]), actions)
    inputs = inputs[:b["frames"]]
    _, end = pc.play_window(env.unwrapped.ale, ale_py.ALEState(s["state"]),
                            inputs, len(inputs), actions[0])
    return end


def end_of(env):
    return pc._digest(env.unwrapped.ale.cloneState().serialize())


@pytest.mark.parametrize("cap", [SMALL_CAP, gc.FRAME_CAP])
def test_the_deployed_class_at_delay_0_is_stage_2as_branch(env, vocab, cap):
    s = live_state(env)
    prior = prior_of()
    decide = decide_of(prior)
    saved = ale_py.ALEState(s["state"])
    dep = decide(s["stack"])
    b = r7.class_branch(env, decide, saved, s["stack"], dep, 0, cap=cap)
    end = end_of(env)
    want = gc.branch(env, decide, saved, s["stack"], 0, cap=cap)
    assert b["rewards"] == want["rewards"] and b["tokens"] == want["tokens"]
    assert b["frames"] == want["frames"] and b["ended"] == want["ended"]
    assert end == want["end"]
    assert (gc.head_returns(b["rewards"]) == gc.head_returns(
        want["rewards"])).all()


@pytest.mark.parametrize("slot, delay", [(3, 3), (1, 9), (4, 11), (6, 0)])
def test_the_slot_then_the_delay_then_the_policy(env, vocab, slot, delay):
    s = live_state(env, seed=2)
    decide = decide_of(prior_of())
    saved = ale_py.ALEState(s["state"])
    b = r7.class_branch(env, decide, saved, s["stack"], slot, delay,
                        cap=SMALL_CAP)
    end = end_of(env)
    assert b["tokens"][0] == slot and b["slot"] == slot
    assert b["delay"] == delay and len(b["tokens"]) > 1
    assert replay_end(env, s, b) == end
    # the delay before the slot is another branch
    if delay:
        other = gc.branch(env, decide, saved, s["stack"], delay, cap=SMALL_CAP)
        assert other["end"] != end or other["tokens"][:1] != [slot]


def test_the_delays_reward_joins_the_next_base_step(env, vocab, monkeypatch):
    s = live_state(env, seed=4)
    decide = decide_of(prior_of())
    saved = ale_py.ALEState(s["state"])
    plain = r7.class_branch(env, decide, saved, s["stack"], 1, 5,
                            cap=SMALL_CAP)
    real = gc.raw_frames

    def rewarded(ale, action, frames, stack, stop=True):
        st, r, lost = real(ale, action, frames, stack, stop)
        return st, r + 1000.0, lost

    monkeypatch.setattr(gc, "raw_frames", rewarded)
    b = r7.class_branch(env, decide, saved, s["stack"], 1, 5, cap=SMALL_CAP)
    n = VOCAB[1][1] // gc.BASE                     # the slot's base steps
    diff = np.array(b["rewards"]) - np.array(plain["rewards"])
    assert diff[n] == 1000.0 and not np.delete(diff, n).any()


def test_a_life_lost_in_the_slot_or_the_delay_ends_the_branch(env, vocab):
    decide = lambda stack: 0                      # noqa: E731  NOOP
    one = before_death(env, 1)
    b = r7.class_branch(env, decide, ale_py.ALEState(one["state"]),
                        one["stack"], 0, 3)
    assert b["ended"] == "life" and len(b["rewards"]) == 1
    assert b["tokens"] == [0]
    six = before_death(env, 6)                    # 4 in the slot, then 2
    b = r7.class_branch(env, decide, ale_py.ALEState(six["state"]),
                        six["stack"], 0, 3)
    assert b["ended"] == "delay" and len(b["rewards"]) == 2
    assert b["frames"] == 6 and b["tokens"] == [0]
    b = r7.class_branch(env, decide, ale_py.ALEState(six["state"]),
                        six["stack"], 0, 1)
    assert b["ended"] == "life" and b["tokens"] == [0, 0]


def test_the_cap_holds_the_slot_and_the_delay(env, vocab):
    s = live_state(env, seed=0)
    decide = decide_of(prior_of())
    saved = ale_py.ALEState(s["state"])
    ended = set()
    for slot in (0, 3, 6):
        for delay in r7.DELAYS:
            b = r7.class_branch(env, decide, saved, s["stack"], slot, delay,
                                cap=64)
            budget = (64 - delay) // gc.BASE
            assert len(b["rewards"]) <= budget
            if b["ended"] == "cap":
                assert len(b["rewards"]) == budget
            ended.add(b["ended"])
    assert "cap" in ended
    # a cap under the slot: its first base steps only
    b = r7.class_branch(env, decide, saved, s["stack"], 3, 0, cap=8)
    assert b["ended"] == "cap" and len(b["rewards"]) == 2


def test_the_default_cap_is_the_reads_frame_cap(env, vocab, monkeypatch):
    s = live_state(env, seed=0)
    decide = decide_of(prior_of())
    saved = ale_py.ALEState(s["state"])
    monkeypatch.setattr(r7, "FRAME_CAP", 64)
    b = r7.class_branch(env, decide, saved, s["stack"], 0, 2)
    assert b["ended"] == "cap" and len(b["rewards"]) == (64 - 2) // gc.BASE


def test_play_state_plays_each_census_class_once_per_delay(env, vocab,
                                                           monkeypatch):
    s = live_state(env, frames=120, seed=1)
    prior = prior_of(b"x")
    decide = decide_of(prior)
    actions = list(env.unwrapped._action_set)
    want_part, want_strict, _ = pc.token_census(
        env.unwrapped.ale, actions, ale_py.ALEState(s["state"]), grid=VOCAB,
        window=pc.WINDOW_TOKENS)
    seen, calls = [], []
    real_census, real_branch = pc.token_census, r7.class_branch

    def census(ale, acts, start, grid=pc.GRID, window=pc.WINDOW_TOKENS):
        seen.append((tuple(grid), window))
        return real_census(ale, acts, start, grid=grid, window=window)

    def branch(env_, decide_, state, stack, slot, delay, cap=None):
        calls.append((slot, delay))
        return real_branch(env_, decide_, state, stack, slot, delay, cap)

    monkeypatch.setattr(pc, "token_census", census)
    monkeypatch.setattr(r7, "class_branch", branch)
    rec = r7.play_state(env, decide, prior, s, VOCAB, cap=SMALL_CAP)
    assert seen == [(VOCAB, 92)]
    assert rec["partition"] == want_part and rec["strict"] == want_strict
    assert len(rec["reps"]) >= 2, "a state with one class tests nothing"
    p = prior(s["stack"])
    assert np.allclose(rec["prior"], p) and rec["deployed"] == np.argmax(p)
    assert rec["reps"] == r7.representatives(want_part, p)
    assert rec["reps"][want_part[rec["deployed"]]] == rec["deployed"]
    assert calls == [(r, d) for r in rec["reps"] for d in r7.DELAYS]
    assert rec["delays"] == list(r7.DELAYS)
    assert rec["returns"].shape == (len(rec["reps"]), 8, 2, 4)
    for c, row in enumerate(rec["branches"]):
        for j, b in enumerate(row):
            assert b["slot"] == rec["reps"][c] and b["delay"] == r7.DELAYS[j]
            assert (rec["returns"][c, j] == gc.head_returns(b["rewards"])).all()


def test_play_state_refuses_a_policy_that_is_not_the_priors_argmax(env,
                                                                    vocab):
    s = live_state(env, frames=120, seed=1)
    prior = prior_of()
    with pytest.raises(SystemExit, match="argmax"):
        r7.play_state(env, lambda stack: (decide_of(prior)(stack) + 1) % 7,
                      prior, s, VOCAB, cap=64)


def test_a_later_lifes_branch_after_an_earlier_lifes(env, vocab):
    """The stale-lives hazard: after a branch that kept 3 lives, a state of
    the second life must not read its first step as a death."""
    one = before_death(env, 1)
    ale = env.unwrapped.ale
    ale.restoreState(ale_py.ALEState(one["state"]))
    ale.act(env.unwrapped._action_set[0])
    assert ale.lives() == 2
    after = {"state": ale.cloneState().serialize(), "stack": one["stack"]}
    early = live_state(env, seed=0)
    decide = decide_of(prior_of())
    alone = r7.class_branch(env, decide, ale_py.ALEState(after["state"]),
                            after["stack"], 0, 0, SMALL_CAP)
    first = r7.class_branch(env, decide, ale_py.ALEState(early["state"]),
                            early["stack"], 0, 0, 64)
    assert first["ended"] == "cap" and env.lives == 3
    again = r7.class_branch(env, decide, ale_py.ALEState(after["state"]),
                            after["stack"], 0, 0, SMALL_CAP)
    assert again == alone and len(again["rewards"]) > 1


def test_the_classes_are_the_censuss_frame_classes(env, vocab, monkeypatch):
    """token_census gives (frame partition, strict partition, digests): the
    classes are the first."""
    s = live_state(env, frames=120, seed=1)
    real = pc.token_census

    def census(ale, acts, start, grid=pc.GRID, window=pc.WINDOW_TOKENS):
        _, _, d = real(ale, acts, start, grid=grid, window=window)
        return [0] * len(grid), list(range(len(grid))), d

    monkeypatch.setattr(pc, "token_census", census)
    prior = prior_of()
    rec = r7.play_state(env, decide_of(prior), prior, s, VOCAB, cap=64)
    assert rec["partition"] == [0] * 7 and rec["strict"] == list(range(7))
    assert rec["reps"] == [rec["deployed"]] and len(rec["branches"]) == 1


def test_a_branch_does_not_depend_on_the_states_before(env, vocab):
    one, six = before_death(env, 1), before_death(env, 400)
    decide = decide_of(prior_of())
    args = (ale_py.ALEState(six["state"]), six["stack"], 2, 3, SMALL_CAP)
    alone = r7.class_branch(env, decide, *args)
    r7.class_branch(env, decide, ale_py.ALEState(one["state"]), one["stack"],
                    0, 0, SMALL_CAP)
    assert r7.class_branch(env, decide, *args) == alone


# --------------------------------------------------------------------------
# the search, on small real nets
# --------------------------------------------------------------------------
A_S = 7


@pytest.fixture
def make_world(monkeypatch):
    """A builder of fresh stand-in runs: real factory nets (de-zeroed heads,
    BN statistics moved off their init) as the trainer's and the network
    manager's, a blob of their weights and acting states that differ in BN
    statistics and one policy bias, and stacks."""
    for key, val in (("USE_GUMBEL_SEARCH", False),
                     ("USE_VALUE_PREFIX_IN_TREE", True), ("MCTS_BATCHES", 1),
                     ("DIRICHLET_ALPHA", 0.3), ("MAX_DEPTH", 12),
                     ("PREFIX_LSTM_HORIZON", 3), ("MINMAX_RANGE_FLOOR", 0.01),
                     ("TOKEN_VOCAB", None), ("C_PUCT", 1.25),
                     ("DISCOUNT", 0.99), ("PREFIX_BN_PER_STEP", False)):
        monkeypatch.setattr(muzero, key, val)
    monkeypatch.setattr(ablation, "ACTIONS", A_S)
    stacks = list(np.random.default_rng(5).integers(
        0, 256, (16, 12, *muzero.IMAGE_RESHAPE), dtype=np.uint8))

    def make():
        torch.manual_seed(4)
        nm = muzero.NetworkManager(None, action_space=A_S,
                                   device=torch.device("cpu"))
        nets = {n: getattr(nm, n) for n in r7.SEARCH_NETS}
        dezero_heads(nets, seed=21)
        gen = torch.Generator().manual_seed(6)
        with torch.no_grad():
            for net in nets.values():
                for m in net.modules():
                    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                        m.running_mean.normal_(0, 0.3, generator=gen)
                        m.running_var.uniform_(0.5, 2.0, generator=gen)
        blob = {n: copy.deepcopy(net.state_dict()) for n, net in nets.items()}
        acting = {}
        for n in ("representation", "prediction"):
            sd = copy.deepcopy(blob[n])
            for k in sd:
                if k.endswith("running_mean"):
                    sd[k] = sd[k] + 0.7
            acting[n] = sd
        # and a policy bias, so the acting argmax differs from the raw one
        acting["prediction"]["policy_fc.bias"] = \
            acting["prediction"]["policy_fc.bias"] + 5.0 * torch.eye(A_S)[-1]
        blob["acting_states"] = acting

        def set_train_mode(training):
            for net in nets.values():
                net.train(training)

        run = SimpleNamespace(
            trainer=SimpleNamespace(**nets),
            searcher=SimpleNamespace(network_manager=nm),
            set_train_mode=set_train_mode)
        return SimpleNamespace(run=run, blob=blob, nets=nets, nm=nm,
                               stacks=stacks)

    return make


def nets_from(world, acting=False, train=False):
    """The blob's weights in fresh copies: raw, or with the acting states."""
    out = {}
    for n, net in world.nets.items():
        c = copy.deepcopy(net)
        c.load_state_dict(world.blob[n])
        if acting and n in world.blob["acting_states"]:
            c.load_state_dict(world.blob["acting_states"][n])
        c.train(train)
        out[n] = c
    return out


def reference(nets, stacks, first_chunk, sims):
    """The registered search by hand: chunks of 8, torch's global RNG and the
    tie generator seeded per chunk before each budget."""
    engine = muzero_vtree.VectorizedPUCT(nets, A_S, "cpu")
    out = {k: [] for k in sims}
    for j in range(0, len(stacks), 8):
        chunk = first_chunk + j // 8
        for k in sims:
            torch.manual_seed(r7.seed_of(r7.NOISE_STREAM, chunk))
            gen = torch.Generator().manual_seed(r7.seed_of(r7.TIE_STREAM,
                                                           chunk))
            _, _, c = engine.search(np.stack(stacks[j:j + 8]),
                                    num_simulations=k, c_puct=1.25,
                                    discount=0.99, root_noise=True,
                                    generator=gen, tie_break="random")
            out[k] += list(c)
    return {k: np.array(v) for k, v in out.items()}


def raw_priors(nets, stacks):
    with torch.no_grad():
        x = torch.tensor(np.stack(stacks), dtype=torch.float32) / 255.0
        out = []
        for i in range(len(stacks)):     # one row at a time, as acting runs
            logits, _, _ = nets["prediction"](nets["representation"](x[i:i + 1]))
            out.append(torch.softmax(logits, dim=1).double().numpy()[0])
    return np.array(out)


SIMS_S = (3, 9)


def test_the_search_runs_on_the_trainers_eval_bn_copied_before_the_overlay(
        make_world):
    w = make_world()
    w.run.set_train_mode(True)                    # whatever mode it was in
    res = r7.search_states(w.run, w.blob, w.stacks[:8], 0, sims=SIMS_S)
    want = reference(nets_from(w), w.stacks[:8], 0, SIMS_S)
    for k in SIMS_S:
        assert (res["search"][str(k)]["counts"] == want[k]).all()
    # the test has power: the acting states or train-mode BN search otherwise
    acting = reference(nets_from(w, acting=True), w.stacks[:8], 0, SIMS_S)
    train = reference(nets_from(w, train=True), w.stacks[:8], 0, SIMS_S)
    assert any((acting[k] != want[k]).any() for k in SIMS_S)
    assert any((train[k] != want[k]).any() for k in SIMS_S)


def test_the_raw_prior_is_one_forward_pass_before_the_overlay(make_world):
    w = make_world()
    w.run.set_train_mode(True)
    stacks = w.stacks[:8]
    res = r7.search_states(w.run, w.blob, stacks, 0, sims=(2,))
    raw, acting = raw_priors(nets_from(w), stacks), raw_priors(
        nets_from(w, acting=True), stacks)
    assert np.allclose(res["raw"], raw, rtol=0, atol=1e-6)
    assert np.allclose(res["acting"], acting, rtol=0, atol=1e-6)
    assert np.abs(raw - acting).max() > 1e-3      # the BN states differ
    assert res["top"]["raw"] == [int(np.argmax(p)) for p in raw]
    assert res["top"]["raw"] != [int(np.argmax(p)) for p in acting]
    # the run now holds the acting states, the copies the snapshot's own
    assert r7.same_weights(w.run.trainer.representation,
                           w.blob["acting_states"]["representation"])


def test_the_search_is_seeded_per_chunk_with_one_noise_for_both_budgets(
        make_world, monkeypatch):
    noise, gens, real = [], [], torch.distributions.Dirichlet.sample

    def sample(self, *a, **k):
        x = real(self, *a, **k)
        noise.append(x.clone())
        return x

    real_search = muzero_vtree.VectorizedPUCT.search

    def search(self, obs, **kw):
        gens.append((kw["generator"].get_state().clone(),
                     kw["num_simulations"], kw["root_noise"],
                     kw["tie_break"]))
        return real_search(self, obs, **kw)

    monkeypatch.setattr(torch.distributions.Dirichlet, "sample", sample)
    monkeypatch.setattr(muzero_vtree.VectorizedPUCT, "search", search)
    w = make_world()
    res = r7.search_states(w.run, w.blob, w.stacks, 0, sims=SIMS_S)
    assert len(noise) == 4                        # 2 chunks x 2 budgets
    assert torch.equal(noise[0], noise[1]) and torch.equal(noise[2], noise[3])
    assert not torch.equal(noise[0], noise[2])
    for (state, sims, root_noise, tie), (chunk, k) in zip(
            gens, [(0, 3), (0, 9), (1, 3), (1, 9)]):
        want = torch.Generator().manual_seed(r7.seed_of(r7.TIE_STREAM, chunk))
        assert torch.equal(state, want.get_state())
        assert sims == k and root_noise is True and tie == "random"
    assert res["search"]["3"]["chunks"] == [0] * 8 + [1] * 8
    # the same again, and the chunks alone: a state's search is its chunk's
    again = r7.search_states(make_world().run, w.blob, w.stacks, 0,
                             sims=SIMS_S)
    second = r7.search_states(make_world().run, w.blob, w.stacks[8:], 1,
                              sims=SIMS_S)
    for k in map(str, SIMS_S):
        assert (again["search"][k]["counts"] == res["search"][k]["counts"]
                ).all()
        assert (second["search"][k]["counts"] ==
                res["search"][k]["counts"][8:]).all()
        assert second["search"][k]["chunks"] == [1] * 8


def test_50_and_200_simulations(make_world, monkeypatch):
    budgets, real = [], muzero_vtree.VectorizedPUCT.search

    def search(self, obs, **kw):
        budgets.append(kw["num_simulations"])
        return real(self, obs, **{**kw, "num_simulations": 2})

    monkeypatch.setattr(muzero_vtree.VectorizedPUCT, "search", search)
    w = make_world()
    res = r7.search_states(w.run, w.blob, w.stacks[:8], 0)
    assert budgets == [50, 200]
    assert set(res["search"]) == {"50", "200"} and set(res["top"]) == {
        "raw", "50", "200"}
    for k in ("50", "200"):
        c = res["search"][k]["counts"]
        assert c.shape == (8, A_S)
        assert res["top"][k] == [int(np.argmax(x)) for x in c]


def test_the_search_spends_its_simulations(make_world):
    w = make_world()
    res = r7.search_states(w.run, w.blob, w.stacks[:3], 0, sims=(5, 11))
    assert (res["search"]["5"]["counts"].sum(axis=1) == 5).all()
    assert (res["search"]["11"]["counts"].sum(axis=1) == 11).all()
    assert len(res["raw"]) == len(res["acting"]) == 3


def test_the_search_refuses_nets_that_are_not_the_snapshots(make_world):
    w = make_world()
    r7.overlay(w.run, w.blob)                     # already overlaid
    with pytest.raises(SystemExit, match="not the snapshot's own"):
        r7.search_states(w.run, w.blob, w.stacks[:2], 0, sims=(2,))
    w = make_world()
    with pytest.raises(SystemExit, match="no acting_states"):
        r7.search_states(w.run, {**w.blob, "acting_states": {}},
                         w.stacks[:2], 0, sims=(2,))


# --------------------------------------------------------------------------
# whole reads in-process: stage 2b's real outputs on its own stub inputs
# --------------------------------------------------------------------------
VOCAB7 = tuple((a, 4) for a in range(t2b.A))


def patch_2b(mp):
    """What stage 2b's own route test runs under (its stamps need them)."""
    mp.setattr(s2b, "DRAWS", 200)
    mp.setattr(gc, "player", t2b.player_stand_in)
    mp.setattr(s2b, "trainer_run", t2b.trainer_stand_in())
    mp.setattr(s2b, "load_blob", lambda path: "BLOB")
    mp.setattr(gc, "snapshot", lambda name: pathlib.Path("T0.pth"))
    mp.setattr(s2b, "log_returns",
               lambda player: (t2b.fake_lives(player), {"lives": 6}))
    mp.setattr(gn, "check_totals", lambda player, lives, totals: None)
    mp.setattr(s2b, "run_pass", t2b.Passes())


def build_2b(tmp, **kw):
    """Stage 2b run for real on its stub inputs: the pilot's, stage 2a's, the
    nets' and stage 2b's directories, read-only."""
    tmp = pathlib.Path(tmp)
    with pytest.MonkeyPatch.context() as mp:
        patch_2b(mp)
        ins = t2b.stub_inputs(tmp / "in", **kw)
        out2b = tmp / "in" / "stage2b"
        out2b.mkdir()
        rec = s2b.stage2b(out2b, ins.gap_dir, ins.nets, ins.pilot)
    dirs = SimpleNamespace(pilot=ins.pilot, gap=ins.gap_dir, nets=ins.nets,
                           stage2b=out2b)
    for d in vars(dirs).values():
        read_only(d)
    return SimpleNamespace(dirs=dirs, states=ins.states, gap=ins.gap,
                           rec2b=rec, tmp=tmp)


def unlock(dirs):
    for d in vars(dirs).values():
        read_only(d, on=False)


@pytest.fixture(scope="module")
def holds(tmp_path_factory):
    b = build_2b(tmp_path_factory.mktemp("r7in"))
    assert b.rec2b["verdict"] == "HOLDS"
    yield b
    unlock(b.dirs)


def digest(stack):
    return hashlib.blake2b(stack.tobytes(), digest_size=16).digest()


class Stubs:
    """Stand-ins for the emulator and the snapshot: play_state from each
    state's stack (its deployed class's delay-0 returns stage 2a's, unless
    told otherwise), search_states from the stacks, the passes in-process."""

    def __init__(self, built):
        gap = built.gap
        t0 = np.asarray(gap["returns"]["T0"], np.float64)
        self.delay0 = {digest(s["stack"]): t0[k, 0]
                       for k, s in enumerate(built.states)}
        self.calls, self.die_at, self.bad_delay0 = [], None, set()
        self.acting_shift, self.chunk_shift, self.visits_scale = 0.0, 0, 1.0

    def play_state(self, env, decide, prior_of, s, vocab, delays=r7.DELAYS,
                   cap=None):
        assert env is None and vocab == VOCAB7
        prior = np.asarray(prior_of(s["stack"]), np.float64).reshape(-1)
        dep = int(np.argmax(prior))
        assert decide(s["stack"]) == dep
        h = hashlib.blake2b(s["stack"].tobytes(), digest_size=8).digest()
        rng = np.random.default_rng(int.from_bytes(h, "little"))
        part = pc.partition(rng.integers(0, int(rng.integers(1, 4)),
                                         len(vocab)).tolist())
        reps = r7.representatives(part, prior)
        ret = rng.normal(size=(len(reps), len(delays), 2, 4))
        ret[part[dep], list(delays).index(0)] = self.delay0[digest(s["stack"])]
        if digest(s["stack"]) in self.bad_delay0:
            ret[part[dep], list(delays).index(0), 1, 3] += 1.0
        rows = [[{"slot": r, "delay": d, "ended": "life", "rewards": [0.0]}
                 for d in delays] for r in reps]
        return {"partition": part, "strict": part, "prior": prior,
                "deployed": dep, "reps": reps, "delays": list(delays),
                "branches": rows, "returns": ret}

    def search_states(self, run, blob, stacks, first_chunk, sims=r7.SIMS,
                      roots=r7.ROOTS):
        assert blob == "BLOB" and run == "TRAINER"
        act = t2b.stand_in_run(b"acting", b"v").searcher.network_manager
        raw = np.array([t2b.softmax(t2b.logits(x, b"trainer"))
                        for x in stacks])
        acting = np.array([act.initial_inference(x)[0] for x in stacks],
                          np.float64) + self.acting_shift
        out = {}
        for k in sims:
            c = np.zeros((len(stacks), t2b.A))
            for j, x in enumerate(stacks):
                c[j, int(np.argmax(t2b.logits(x, b"s%d" % k)))] = k
            if k == 200:
                c = c * self.visits_scale
            out[str(k)] = {"counts": c, "values": [0.0] * len(stacks),
                           "chunks": [first_chunk + self.chunk_shift
                                      + j // roots
                                      for j in range(len(stacks))]}
        top = {"raw": [int(np.argmax(p)) for p in raw]}
        top.update({k: [int(np.argmax(c)) for c in v["counts"]]
                    for k, v in out.items()})
        return {"raw": raw, "acting": acting, "search": out, "top": top}

    def run_pass(self, ctx, stage, k=None):
        # as a fresh process: the inputs read and checked again
        c = r7.Context(ctx.out, ctx.pilot, ctx.gap, ctx.nets, ctx.stage2b)
        self.calls.append((stage, k))
        if (stage, k) == self.die_at:
            raise SystemExit(f"pass {stage} {k}: rc 1")
        if stage == "draw":
            r7.stage_draw(c)
        else:
            {"play": r7.stage_play, "search": r7.stage_search}[stage](c, k)


@contextlib.contextmanager
def player7(name):
    assert name == "T0"
    run = t2b.stand_in_run(b"acting", b"acting-value")
    run.env = None
    yield run


@contextlib.contextmanager
def trainer7(blob):
    yield "TRAINER", ["SOME_KEY"]


@pytest.fixture
def route(holds, tmp_path, monkeypatch):
    patch_2b(monkeypatch)
    stubs = Stubs(holds)
    monkeypatch.setattr(gc, "player", player7)
    monkeypatch.setattr(muzero, "TOKEN_VOCAB", VOCAB7)
    monkeypatch.setattr(r7, "play_state", stubs.play_state)
    monkeypatch.setattr(r7, "search_states", stubs.search_states)
    monkeypatch.setattr(r7, "trainer_run", trainer7)
    monkeypatch.setattr(r7, "load_blob", lambda path: "BLOB")
    monkeypatch.setattr(r7, "run_pass", stubs.run_pass)
    monkeypatch.setattr(r7, "DRAWS", DRAWS)
    monkeypatch.setattr(r7, "CHUNK", 8)
    monkeypatch.setattr(r7, "PER_FIFTH", 10)
    out = tmp_path / "out"
    out.mkdir()
    d = holds.dirs
    return SimpleNamespace(
        stubs=stubs, out=out, dirs=d, built=holds,
        run=lambda dirs=None: r7.read7(out, *(
            (dirs or d).pilot, (dirs or d).gap, (dirs or d).nets,
            (dirs or d).stage2b)),
        ctx=lambda: r7.Context(out, d.pilot, d.gap, d.nets, d.stage2b))


def test_the_route_reads_top_against_bottom(route, capsys):
    before = {n: listing(p) for n, p in vars(route.dirs).items()}
    rec = route.run()
    assert route.stubs.calls == [("draw", None), ("play", 0), ("play", 1),
                                 ("play", 2), ("search", 0), ("search", 1),
                                 ("search", 2)]
    assert rec["status"] == "READ" and rec["reasons"] == []
    assert rec["states"] == 20 and rec["fifth_counts"] == {"top": 10,
                                                           "bottom": 10}
    assert rec["draw"]["sample"] == t2b.N
    assert rec["draw"]["pools"] == {"top": 16, "bottom": 16}   # 2 + 2 of 10
    assert set(rec["reads"]) == set(r7.READS + r7.BESIDE)
    assert {n: listing(p) for n, p in vars(route.dirs).items()} == before
    assert sorted(p.name for p in route.out.iterdir()) == sorted(
        ["draw.pkl", r7.SUMMARY] + [f"play-{k:03d}.pkl" for k in range(3)]
        + [f"search-{k:03d}.pkl" for k in range(3)])
    ctx = route.ctx()
    assert r7.read(route.out / r7.SUMMARY, ctx.st)["status"] == "READ"
    out = capsys.readouterr().out
    assert "READ (7): descriptive" in out and "SMOKE" not in out
    # the reads from the pass files by hand
    draw = r7.read(r7.draw_file(route.out), ctx.st)
    plays, tops = [], []
    for k in range(3):
        plays += r7.read(r7.play_file(route.out, k), ctx.st)["states"]
        s = r7.read(r7.search_file(route.out, k), ctx.st)
        tops += [{t: s["top"][t][j] for t in s["top"]}
                 for j in range(len(s["kept"]))]
    per = [r7.state_reads(p, t) for p, t in zip(plays, tops)]
    fifth = np.array(draw["fifth"])
    for name in r7.READS:
        v = np.array([r[name] for r in per])
        point = rec["reads"][name]["point"]
        assert point["top"] == pytest.approx(v[fifth == "top"].mean())
        assert point["top_minus_bottom"] == pytest.approx(
            v[fifth == "top"].mean() - v[fifth == "bottom"].mean())
    assert [p["kept"] for p in rec["per_state"]] == draw["drawn"]


def test_the_draw_is_the_fifths_of_stage_2bs_sample(route):
    route.run()
    ctx = route.ctx()
    draw = r7.read(r7.draw_file(route.out), ctx.st)
    gap = r7.read(route.dirs.gap / "gap.pkl", s2.stamp(ctx.states))
    recs = {n: r7.read(s2b.pass_file(route.dirs.stage2b, n), ctx.st2b)
            for n in s2b.PASSES}
    data, _ = s2b.assemble(ctx.states, ctx.keys, gap, recs["nets"],
                           recs["t0"], recs["table"])
    top, bottom = r7.fifth_masks(data["g"], data["sid"])
    for k, f, g in zip(draw["drawn"], draw["fifth"], draw["g"]):
        assert (top if f == "top" else bottom)[k] and g == data["g"][k]
    assert draw["clusters"] == [gap["clusters"][k] for k in draw["drawn"]]
    assert (draw["t0_delay0"] == np.asarray(gap["returns"]["T0"])[
        draw["drawn"], 0]).all()
    assert draw["t0_argmax"] == [int(recs["t0"]["prior_argmax"][k])
                                 for k in draw["drawn"]]


def test_training_overlap_states_are_never_drawn(tmp_path, monkeypatch):
    b = build_2b(tmp_path / "ov", train={"master": (0, 1), "twin": (2, 13)})
    try:
        patch_2b(monkeypatch)
        monkeypatch.setattr(r7, "PER_FIFTH", 1000)
        out = tmp_path / "out"
        out.mkdir()
        ctx = r7.Context(out, b.dirs.pilot, b.dirs.gap, b.dirs.nets,
                         b.dirs.stage2b)
        r7.stage_draw(ctx)
        draw = r7.read(r7.draw_file(out), ctx.st)
        assert draw["sample"] == t2b.N - 4
        assert not {0, 1, 2, 13} & set(draw["drawn"])
        assert len(draw["drawn"]) == draw["pools"]["top"] + \
            draw["pools"]["bottom"]
    finally:
        unlock(b.dirs)


def copied(route, tmp_path):
    """A writable copy of the stage 2b route's inputs."""
    d = SimpleNamespace()
    for n, p in vars(route.dirs).items():
        q = tmp_path / "copy" / n
        shutil.copytree(p, q)
        read_only(q, on=False)
        setattr(d, n, q)
    return d


def test_no_stage_2b_summary_no_read(route, tmp_path):
    d = copied(route, tmp_path)
    (d.stage2b / s2b.SUMMARY).unlink()
    with pytest.raises(SystemExit, match="stage 2b has not finished"):
        route.run(d)
    assert route.stubs.calls == [] and not list(route.out.iterdir())
    threads = torch.get_num_threads()
    try:
        for stage in ("draw", "play", "search"):
            args = [stage, "--out", str(route.out), "--pilot", str(d.pilot),
                    "--gap", str(d.gap), "--nets", str(d.nets), "--stage2b",
                    str(d.stage2b)] + ([] if stage == "draw"
                                       else ["--chunk", "0"])
            with pytest.raises(SystemExit, match="stage 2b has not finished"):
                r7.main(args)
    finally:
        torch.set_num_threads(threads)
    assert not list(route.out.iterdir())


@pytest.mark.parametrize("over", [{"verdict": "VOID"},
                                  {"g2_inside": [-0.05, 0.2]}])
def test_a_stage_2b_past_no_gate_is_refused(route, tmp_path, over):
    d = copied(route, tmp_path)
    ctx = route.ctx()
    path = d.stage2b / s2b.SUMMARY
    rec = r7.read(path, ctx.st2b)
    rec = {k: v for k, v in rec.items() if k != "stamp"}
    if "verdict" in over:
        rec["verdict"] = over["verdict"]
    else:
        rec["reads"]["primary"]["intervals"]["g2_twin"] = over["g2_inside"]
    s2.write(path, rec, ctx.st2b)
    with pytest.raises(SystemExit, match="did not pass its gates"):
        route.run(d)
    assert route.stubs.calls == []


def test_stage_2as_gates_are_read_again(route, monkeypatch):
    monkeypatch.setattr(s2b, "read_gates",
                        lambda gap, nets, states: (None, None, [], ["G4 T0"]))
    with pytest.raises(SystemExit, match="G4 T0"):
        route.run()


def test_a_failed_path_check_is_void(route, capsys):
    ctx = route.ctx()
    route.run()                                    # the clean route first
    draw = r7.read(r7.draw_file(route.out), ctx.st)
    draw_states = [ctx.states[k] for k in draw["drawn"]]
    for p in list(route.out.iterdir()):
        if p.name != "draw.pkl":
            p.unlink()
    route.stubs.bad_delay0 = {digest(draw_states[3]["stack"])}
    rec = route.run()
    assert rec["status"] == "VOID" and len(rec["reasons"]) == 1
    assert "drawn state 3" in rec["reasons"][0]
    assert "not stage 2a's T0 branch" in rec["reasons"][0]
    assert "READ (7): VOID" in capsys.readouterr().out
    assert r7.read(route.out / r7.SUMMARY, ctx.st)["status"] == "VOID"


def test_the_acting_cross_check_is_void(route):
    route.stubs.acting_shift = 1e-3
    rec = route.run()
    assert rec["status"] == "VOID"
    assert all("acting prior differs" in r for r in rec["reasons"])
    assert len(rec["reasons"]) == rec["states"]


@pytest.mark.parametrize("knob, value", [("chunk_shift", 1),
                                         ("visits_scale", 0.25)])
def test_a_search_off_its_chunk_or_budget_is_void(route, knob, value):
    setattr(route.stubs, knob, value)
    rec = route.run()
    assert rec["status"] == "VOID"
    assert all("simulation search spent" in r for r in rec["reasons"])


def test_another_deployed_slot_than_t0_pkls_is_void(route, monkeypatch):
    @contextlib.contextmanager
    def other(name):
        run = t2b.stand_in_run(b"other", b"acting-value")
        run.env = None
        yield run

    monkeypatch.setattr(gc, "player", other)
    rec = route.run()
    assert rec["status"] == "VOID"
    assert any("t0.pkl's argmax" in r for r in rec["reasons"])


def test_a_stopped_read_resumes_where_it_stopped(route):
    route.stubs.die_at = ("search", 1)
    with pytest.raises(SystemExit, match="pass search 1"):
        route.run()
    route.stubs.die_at = None
    route.stubs.calls.clear()
    rec = route.run()
    assert route.stubs.calls == [("search", 1), ("search", 2)]
    assert rec["status"] == "READ"


def test_the_summary_is_never_overwritten(route, monkeypatch):
    route.run()
    before = listing(route.out)
    monkeypatch.setattr(s2, "load_states", _forbidden)
    monkeypatch.setattr(r7, "boot", _forbidden)
    with pytest.raises(SystemExit, match="exists: the summary is never "
                                         "overwritten"):
        route.run()
    assert listing(route.out) == before


def test_files_from_other_knobs_or_inputs_are_refused(route, tmp_path,
                                                      monkeypatch):
    route.run()
    (route.out / r7.SUMMARY).unlink()
    monkeypatch.setattr(r7, "DRAWS", DRAWS + 1)
    with pytest.raises(SystemExit, match="move it aside"):
        route.run()
    monkeypatch.setattr(r7, "DRAWS", DRAWS)
    monkeypatch.setattr(r7, "SOURCE", "another source")
    with pytest.raises(SystemExit, match="move it aside"):
        route.run()
    monkeypatch.setattr(r7, "SOURCE", gc.source_hash(r7.SOURCES))
    # one of stage 2b's files made again: everything after it is refused
    d = copied(route, tmp_path)
    ctx = route.ctx()
    path = s2b.pass_file(d.stage2b, "t0")
    obj = r7.read(path, ctx.st2b)
    obj["secs"] = obj["secs"] + 1.0
    s2.write(path, {k: v for k, v in obj.items() if k != "stamp"}, ctx.st2b)
    with pytest.raises(SystemExit, match="move it aside"):
        route.run(d)


@pytest.mark.parametrize("name", ["play", "search"])
def test_a_pass_file_for_other_drawn_states_is_refused(route, name):
    route.run()
    (route.out / r7.SUMMARY).unlink()
    f = {"play": r7.play_file, "search": r7.search_file}[name]
    shutil.copy(f(route.out, 1), f(route.out, 0))   # the same stamp
    with pytest.raises(SystemExit, match="not drawn states 0-8"):
        route.run()


def test_no_output_inside_the_inputs(route, tmp_path):
    for p in (*vars(route.dirs).values(), route.dirs.stage2b / "sub"):
        with pytest.raises(SystemExit, match="never writes"):
            r7.read7(p, route.dirs.pilot, route.dirs.gap, route.dirs.nets,
                     route.dirs.stage2b)
    with pytest.raises(SystemExit, match="never writes"):
        r7.guard(gc.RESULTS / "x")
    r7.guard(tmp_path / "elsewhere", *vars(route.dirs).values())


def test_a_smoke_read_says_so_and_reads_its_first_states(route, monkeypatch,
                                                         capsys):
    monkeypatch.setattr(r7, "LIMIT", 12)
    monkeypatch.setattr(r7, "SMOKE", True)
    rec = route.run()
    assert rec["smoke"] and rec["states"] == 12
    assert route.stubs.calls == [("draw", None), ("play", 0), ("play", 1),
                                 ("search", 0), ("search", 1)]
    assert "SMOKE READ (7)" in capsys.readouterr().out


def test_chunks_are_whole_search_chunks(monkeypatch):
    monkeypatch.setattr(r7, "CHUNK", 12)
    with pytest.raises(SystemExit, match="multiple of 8"):
        r7.chunks(30)
    monkeypatch.setattr(r7, "CHUNK", 16)
    assert r7.chunks(40) == [(0, 0, 16), (1, 16, 32), (2, 32, 40)]


# --------------------------------------------------------------------------
# slow: T0's real snapshot on the pilot's states (never g)
# --------------------------------------------------------------------------
_REAL = r"""
import json, sys, time
sys.path[:0] = [{probes!r}, {exp!r}, {root!r}]
import os
import probe_gap_read7 as r7
os.environ.update(r7.ENV)
import numpy as np
import torch
import muzero
import probe_gap_check as gc
import probe_gap_stage2a as s2
torch.set_num_threads(2)
sel = gc.read(r7.PILOT / "select.pkl")
pilot_rows = gc.read(r7.PILOT / "branches-T0.pkl")["table"]
out = {{"play": []}}
eight = [sel["kept"][sel["chosen"][j]] for j in range(0, 60, 7)][:8]
with gc.player("T0") as run:
    decide, m = gc.deployed(run), run.searcher.network_manager
    vocab = tuple(tuple(int(x) for x in t) for t in muzero.TOKEN_VOCAB)
    for j in (0, 31):
        s = sel["kept"][sel["chosen"][j]]
        t0 = time.time()
        rec = r7.play_state(run.env, decide, lambda x: m.initial_inference(x)[0],
                            s, vocab)
        secs = time.time() - t0
        c0 = rec["partition"][rec["deployed"]]
        b = rec["branches"][c0][r7.DELAYS.index(0)]
        p = pilot_rows[j][0]
        out["play"].append({{"secs": secs, "classes": len(rec["reps"]),
            "branches": sum(len(r) for r in rec["branches"]),
            "same": b["rewards"] == p["rewards"] and b["tokens"] == p["tokens"],
            "returns": bool(np.isfinite(rec["returns"]).all()),
            "ended": [x["ended"] for row in rec["branches"] for x in row]}})
    deployed = np.array([m.initial_inference(s["stack"])[0] for s in eight])
out["play_memory"] = s2.memory()
stacks = [s["stack"] for s in eight]
for sims in ((50,), (200,)):
    blob = r7.load_blob(gc.snapshot("T0"))
    with r7.trainer_run(blob) as (run, missing):
        t0 = time.time()
        res = r7.search_states(run, blob, stacks, 0, sims=sims)
        out[str(sims[0])] = {{"secs": time.time() - t0,
            "visits": res["search"][str(sims[0])]["counts"].sum(axis=1).tolist(),
            "cross": float(np.abs(res["acting"] - deployed).max()),
            "raw_vs_acting": float(np.abs(res["raw"] - deployed).max()),
            "top": res["top"][str(sims[0])], "raw_top": res["top"]["raw"],
            "memory": s2.memory()}}
    del blob
print("RESULT " + json.dumps(out))
"""


@pytest.mark.slow
def test_t0s_real_snapshot_on_the_pilots_states():
    try:
        gc.snapshot("T0")
        gc.read(r7.PILOT / "pilot.json")
    except (SystemExit, FileNotFoundError):
        pytest.skip("T0's snapshot or the pilot's files are not here")
    before = listing(r7.PILOT)
    code = _REAL.format(probes=str(PROBES), exp=str(ROOT / "experiments"),
                        root=str(ROOT))
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                       env={**os.environ, **r7.ENV}, capture_output=True,
                       text=True, timeout=3000)
    assert r.returncode == 0, (r.stdout + r.stderr)[-3000:]
    line = [x for x in r.stdout.splitlines() if x.startswith("RESULT ")][-1]
    out = json.loads(line[len("RESULT "):])
    print(json.dumps({k: v for k, v in out.items()}, indent=1))
    for p in out["play"]:
        assert p["same"], "the deployed class's delay-0 branch left the pilot's"
        assert p["returns"] and p["branches"] == 8 * p["classes"]
    for k in ("50", "200"):
        assert out[k]["visits"] == [float(k)] * 8
        assert out[k]["cross"] <= r7.CROSS_TOL
        assert out[k]["raw_vs_acting"] > r7.CROSS_TOL   # other BN states
    assert listing(r7.PILOT) == before
