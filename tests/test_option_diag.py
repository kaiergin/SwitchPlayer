"""Test O stage 4a (CURRICULUM.md "Test O": READS (3), (4), (6), DEPARTURES 3
and 7, DESIGN "Prefix BN"): the option arms' per-row diagnostics and read
(3)'s on/off root probe.

Expected numbers are worked BY HAND from the definitions (ablation.py's
option_state_counts, option_probe_stats, option_row_fields,
option_prefix_panel docstrings and the registration), never read off the
implementation: engagement counts across lives and rows, DEPARTURE 7's hit
rate, read (4)'s accuracy by step and its two baselines on constructed lives,
the probe's Jensen-Shannon divergence, flip share and signed shift on
constructed visit targets, the probe's root rule, the train-step window, the
prefix panel by row kind, checkpoint / resume of a partial row, and the probe
leaving the training RNG streams and every net's mode alone.

FLAG OFF / NO DRIFT: an O0 life, row and checkpoint head, and an option-arm
train step (loss, gradients, BN buffers, optimizer, PMAX writes, RNG), are
pinned bit for bit against the build tree before this stage (git show
b4e4d807: muzero.py, experiments/ablation.py), loaded from git as
test_option_train.py loads e11fd5f8 -- the pins FAIL, never skip, when git
cannot read those files. CPU only."""
import asyncio
import contextlib
import copy
import importlib.util
import json
import math
import pathlib
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import ablation  # noqa: E402
import muzero  # noqa: E402
import muzero_option_targets as mot  # noqa: E402
import muzero_vtree_options  # noqa: E402
from tests.test_prioract2 import _fake_pipe, restore_rngs  # noqa: E402,F401
from tests.test_testc import make_run  # noqa: E402
from tests.test_option_train import (  # noqa: E402
    A9, CPU, build_batch, dezero_policy, life3, real_lives, rng_states,
    same_rng, seed_all, suggest)

PRE_S4 = "b4e4d807"          # the build tree's head before stage 4a
O0 = ablation.PLAIN_ARM + "-o0"
OF = ablation.PLAIN_ARM + "-of"
OH = ablation.PLAIN_ARM + "-oh"
L3 = 3


# ==========================================================================
# fixtures and helpers
# ==========================================================================
@contextlib.contextmanager
def _arm(arm):
    saved = {k: getattr(ablation, k)
             for k in ("ACTIONS", "ENV_NAME", "FRAMESKIP", "DEVICE")}
    ablation.ACTIONS, ablation.ENV_NAME = A9, "MsPacmanNoFrameskip-v4"
    ablation.FRAMESKIP, ablation.DEVICE = 4, CPU
    try:
        with ablation.arm_config(arm) as spec, ablation.token_actions():
            yield spec
    finally:
        for k, v in saved.items():
            setattr(ablation, k, v)


@pytest.fixture
def of(restore_rngs):  # noqa: F811
    """Inside OF's config at Ms. Pac-Man's 9 moves on the CPU; yields the
    spec. Request it BEFORE monkeypatch (test_option_train's o0 order)."""
    with _arm(OF) as spec:
        yield spec


@pytest.fixture
def o0(restore_rngs):  # noqa: F811
    with _arm(O0) as spec:
        yield spec


@pytest.fixture(autouse=True)
def no_leak():
    yield
    assert muzero.OPTION_LENGTH == 1 and muzero.OPTION_EDGE is True
    assert muzero.OPTION_PRECHECK is False


def real_run(spec, arm=OF, seed=0):
    torch.manual_seed(seed)
    return make_run(spec, arm)


# ---- a stub option engine and stub nets over coded frames ----------------
class _Rep(torch.nn.Module):
    def forward(self, x):
        return x


class _Pred(torch.nn.Module):
    """Logits one-hot on the code's PMAX (the frame holds its code)."""
    option_length = L3

    def __init__(self, table):
        super().__init__()
        self.table = table

    def forward(self, x):
        codes = torch.round(x.reshape(len(x), -1)[:, 0] * 255).long().tolist()
        logits = torch.zeros(len(x), A9)
        for i, c in enumerate(codes):
            logits[i, self.table[c]["pmax"]] = 5.0
        return logits, None, None


def stub_nets(table):
    return {"representation": _Rep(), "dynamics": _Rep(),
            "prediction": _Pred(table), "value_prefix": _Rep()}


def coded_life(codes, played, closing=0):
    """3-field actor records whose frames hold `codes` (the last code is
    the closing record's)."""
    return [(np.full((1,), c, dtype=np.uint8),
             played[i] if i < len(played) else closing, (0.0,))
            for i, c in enumerate(codes)]


def stub_engine(monkeypatch, table, calls=None):
    """VectorizedOptionPUCT replaced by a table keyed on the frame's code:
    each root's policy, option, PMAX and search counters."""
    class Stub:
        def __init__(self, nets, actions, device):
            pass

        def search(self, obs, num_simulations=None, c_puct=None,
                   discount=None, root_noise=True, option_edge=None,
                   return_info=False, **kw):
            codes = [int(o.reshape(-1)[0]) for o in obs]
            rows = [table[c] for c in codes]
            if calls is not None:
                calls.append((codes, root_noise, option_edge, kw))
            opt = np.array([list(r["option"]) + [-1] * (L3 - len(r["option"]))
                            for r in rows])
            info = {
                "root_option": opt,
                "root_option_len": np.array([len(r["option"]) for r in rows]),
                "root_pmax": np.array([r["pmax"] for r in rows]),
                "expanded_nodes": np.array([r["nodes"] for r in rows]),
                "expanded_with_option": np.array(
                    [sum(r["len_hist"][2:]) for r in rows]),
                "option_len_hist": np.array([r["len_hist"] for r in rows]),
                "simulations": np.full(len(rows), num_simulations),
                "sims_crossing_option": np.array([r["cross"] for r in rows]),
                "crossed_len_hist": np.array([r["cross_hist"] for r in rows]),
                "leaf_depth_sum": np.array([r["leaf"] for r in rows]),
                "leaf_depth_max": np.array([r["leaf_max"] for r in rows]),
                "option_edge": bool(option_edge), "option_length": L3}
            pols = np.stack([r["pi"] for r in rows])
            return pols, np.zeros(len(rows)), None, info

    monkeypatch.setattr(muzero_vtree_options, "VectorizedOptionPUCT", Stub)
    return Stub


def pi_with(m, p, other=None):
    """A visit target with mass p on move m, the rest on (m + 1) % 9 (or
    `other`)."""
    v = np.zeros(A9)
    v[m] = p
    v[(m + 1) % A9 if other is None else other] += 1.0 - p
    return v


def entry(pmax, p, option, nodes, len_hist, cross, cross_hist, leaf,
          leaf_max):
    return {"pmax": pmax, "pi": pi_with(pmax, p), "option": tuple(option),
            "nodes": nodes, "len_hist": len_hist, "cross": cross,
            "cross_hist": cross_hist, "leaf": leaf, "leaf_max": leaf_max}


# Life A (codes 0-3 acted, 4 closing; terminated): played 2 2 5 2, PMAX 2 at
# every root, pi[PMAX] 0.6 0.7 0.1 0.9, options (2,2,5) (2,5,2) none (2,2).
# Life B (codes 10, 11 acted, 12 the cut tail's): played 7 7, PMAX 7 and 1,
# pi[PMAX] 0.5 0.25, options (7,7,7) and none.
TABLE = {
    0: entry(2, 0.6, (2, 2, 5), 10, [7, 0, 1, 2], 5, [0, 0, 3, 2], 100.0, 4),
    1: entry(2, 0.7, (2, 5, 2), 11, [9, 0, 2, 0], 0, [0, 0, 0, 0], 80.0, 3),
    2: entry(2, 0.1, (), 12, [12, 0, 0, 0], 0, [0, 0, 0, 0], 60.0, 2),
    3: entry(2, 0.9, (2, 2), 13, [12, 0, 1, 0], 2, [0, 0, 2, 0], 120.0, 6),
    10: entry(7, 0.5, (7, 7, 7), 5, [4, 0, 0, 1], 1, [0, 0, 0, 1], 10.0, 1),
    11: entry(1, 0.25, (), 6, [6, 0, 0, 0], 0, [0, 0, 0, 0], 20.0, 2),
    12: entry(4, 0.9, (4, 4), 7, [5, 0, 2, 0], 1, [0, 0, 1, 0], 30.0, 9),
}
LIFE_A = (list(range(5)), [2, 2, 5, 2], True)
LIFE_B = ([10, 11, 12], [7, 7], False)


def stub_run(spec):
    run = object.__new__(ablation.Run)
    run.arm, run.seed, run.spec = OF, 0, spec
    run.trainer = type("T", (), {"training_steps": 100, "device": CPU})()
    run.replay, run.env_steps = [], 0
    return run


def feed(run, life):
    codes, played, terminated = life
    out = run.paper_targets(coded_life(codes, played), terminated,
                            born_step=0, birth_nets=stub_nets(TABLE))
    run.replay.append(out)
    return out


# ==========================================================================
# read (4): state dependence, by hand
# ==========================================================================
#                t:  0       1       2       3       4       5      6
S4_PLAYED = [0, 1, 1, 2, 2, 2, 3]
S4_OPTIONS = [(0, 1, 1), (1, 1, 2), (), (2, 1, 2), (2, 2), (2, 3), (0, 3, 3)]


def test_state_dependence_by_hand():
    got = ablation.option_state_counts(S4_PLAYED, S4_OPTIONS, 3, 4)
    # t = 0 is in no step (no move before it for the repeat baseline).
    # t = 1 (1,1,2), prev 0: played 1 1 2 -> option 3/3, repeat 0/3
    # t = 3 (2,1,2), prev 1: played 2 2 2 -> option hit, MISS, hit (the
    #        comparison goes on past a miss); repeat 0/3
    # t = 4 (2,2),   prev 2: played 2 2 -> option 2/2, repeat 2/2
    # t = 5 (2,3),   prev 2: played 2 3 -> option 2/2, repeat 1/2
    # t = 6 (0,3,3), prev 2: played 3, then the life ends -> step 1 only:
    #        option miss, repeat miss
    assert got["s4_n"] == [5, 4, 2]
    assert got["s4_opt"] == [4, 3, 2]
    assert got["s4_rep"] == [2, 1, 0]
    # the played moves compared at each step: step 1 at t = 1 3 4 5 6,
    # step 2 at t+1 = 2 4 5 6, step 3 at t+2 = 3 5
    assert got["s4_hist"] == [[0, 1, 3, 1], [0, 1, 2, 1], [0, 0, 2, 0]]
    # options at t = 0 1 3 4 5 6; one move repeated: only (2,2)
    assert got["with_option"] == 6 and got["nonrepeat"] == 5
    assert got["root_len_hist"] == [1, 0, 2, 4]


def test_state_dependence_row_fields_and_the_marginal_baseline():
    w = ablation.option_window_empty(3, 4, 5)
    got = ablation.option_state_counts(S4_PLAYED, S4_OPTIONS, 3, 4)
    for k in ("s4_n", "s4_opt", "s4_rep", "s4_hist", "root_len_hist"):
        w[k] = got[k]
    w["with_option"], w["nonrepeat"] = 6, 5
    w["acted"], w["play"] = 7, [1, 2, 3, 1]     # the row's played moves
    row = ablation.option_row_fields(w, None, 4)
    # the row's most frequent played move is 2; it is played at 3 of step
    # 1's 5 positions, 2 of step 2's 4, 2 of step 3's 2
    assert row["opt_s4_mode"] == 2
    assert row["opt_s4_marginal_hits"] == [3, 2, 2]
    assert row["opt_s4_accuracy"] == [0.8, 0.75, 1.0]
    assert row["opt_s4_repeat_accuracy"] == [0.4, 0.25, 0.0]
    assert row["opt_s4_marginal_accuracy"] == [0.6, 0.5, 1.0]
    assert row["opt_root_options"] == 6
    assert row["opt_root_option_share"] == round(6 / 7, 4)
    assert row["opt_nonrepeat_share"] == round(5 / 6, 4)


def test_state_dependence_edge_cases():
    # no options at all: every step empty, None accuracies
    got = ablation.option_state_counts([1, 2, 3], [(), (), ()], 3, 4)
    assert got["s4_n"] == [0, 0, 0] and got["with_option"] == 0
    assert got["root_len_hist"] == [3, 0, 0, 0]
    # a one-position life: only t = 0, in no step
    got = ablation.option_state_counts([1], [(1, 1)], 3, 4)
    assert got["s4_n"] == [0, 0, 0] and got["with_option"] == 1
    # the marginal mode ties to the lowest move
    w = ablation.option_window_empty(3, 4, 5)
    w["play"] = [0, 2, 2, 0]
    assert ablation.option_row_fields(w, None, 4)["opt_s4_mode"] == 1


# ==========================================================================
# read (3)'s on/off probe statistic, by hand
# ==========================================================================
def _js(p, q):
    """JS by hand, natural log, 0 log 0 = 0."""
    m = [(a + b) / 2 for a, b in zip(p, q)]
    kl = (lambda x: sum(a * math.log(a / c) for a, c in zip(x, m) if a > 0))
    return 0.5 * kl(p) + 0.5 * kl(q)


PON = [[0.5, 0.3, 0.2], [0.6, 0.4, 0.0], [0.4, 0.4, 0.2], [0.45, 0.45, 0.1]]
POFF = [[0.5, 0.3, 0.2], [0.2, 0.8, 0.0], [0.2, 0.4, 0.4], [0.1, 0.2, 0.7]]
ROPT = [[1, 1, -1], [0, 1, -1], [-1, -1, -1], [2, 2, 2]]
RLEN = [2, 2, 0, 3]


def test_probe_statistic_by_hand():
    got = ablation.option_probe_stats(PON, POFF, ROPT, RLEN)
    # root 0: identical -> JS 0, no flip, shift 0 (an option on move 1)
    # root 1: tops 0 / 1 -> a flip; option (0, 1): shift 0.6 - 0.2
    # root 2: tops {0, 1} / {1, 2} share move 1 -> NO flip (a tie never
    #         flips by itself); no option -> out of the shift
    # root 3: tops {0, 1} / {2} -> a flip; option (2,2,2): 0.1 - 0.7
    assert got["roots"] == 4 and got["flips"] == 2
    assert got["shift_roots"] == 3
    assert got["shift_sum"] == pytest.approx(0.0 + 0.4 - 0.6)
    want = (0.0 + (0.5 * (0.6 * math.log(0.6 / 0.4) + 0.4 * math.log(0.4 / 0.6))
                   + 0.5 * (0.2 * math.log(0.2 / 0.4)
                            + 0.8 * math.log(0.8 / 0.6)))
            + (0.5 * (0.4 * math.log(0.4 / 0.3) + 0.2 * math.log(0.2 / 0.3))
               + 0.5 * (0.2 * math.log(0.2 / 0.3) + 0.4 * math.log(0.4 / 0.3)))
            + _js(PON[3], POFF[3]))
    assert got["js_sum"] == pytest.approx(want, abs=1e-12)
    row = ablation.option_row_fields(ablation.option_window_empty(3, 3, 5),
                                     {**got, "seconds": 1.234}, 4)
    assert row["opt_probe_roots"] == 4
    assert row["opt_probe_js"] == round(want / 4, 6)
    assert row["opt_probe_flips"] == 2 and row["opt_probe_flip_share"] == 0.5
    assert row["opt_probe_shift_roots"] == 3
    assert row["opt_probe_shift"] == round(-0.2 / 3, 6)
    assert row["opt_probe_s"] == 1.23


def test_probe_statistic_bounds_and_empty():
    got = ablation.option_probe_stats([[1.0, 0.0]], [[0.0, 1.0]],
                                      [[-1, -1]], [0])
    assert got["js_sum"] == pytest.approx(math.log(2))
    assert got["flips"] == 1 and got["shift_roots"] == 0
    row = ablation.option_row_fields(ablation.option_window_empty(3, 2, 5),
                                     None, 4)
    assert row["opt_probe_roots"] == 0
    for k in ("opt_probe_js", "opt_probe_flip_share", "opt_probe_shift",
              "opt_probe_s"):
        assert row[k] is None, k


# ==========================================================================
# the row window through paper_targets: lives and rows, by hand
# ==========================================================================
def test_engagement_hit_rate_and_state_dependence_over_two_lives(
        of, monkeypatch):
    calls = []
    stub_engine(monkeypatch, TABLE, calls)
    run = stub_run(of)
    feed(run, LIFE_A)
    # one search for the terminated life's 4 acted roots
    assert [(c, n) for c, n, _, _ in calls] == [([0, 1, 2, 3], True)]
    w = run._opt_window
    assert (w["roots"], w["expanded_nodes"], w["expanded_with_option"]) == \
        (4, 10 + 11 + 12 + 13, 3 + 2 + 0 + 1)
    feed(run, LIFE_B)
    # the cut life: its 2 acted roots, then its greedy tail (code 12)
    assert [(c, n) for c, n, _, _ in calls[1:]] == [([10, 11], True),
                                                   ([12], False)]
    w = run._opt_window
    sims = muzero.ACT_SEARCH_SIMS
    assert w["roots"] == 7
    assert w["expanded_nodes"] == 46 + 5 + 6 + 7
    assert w["expanded_with_option"] == 6 + 1 + 0 + 2
    # A [7,0,1,2]+[9,0,2,0]+[12,0,0,0]+[12,0,1,0]; B [4,0,0,1]+[6,0,0,0]
    # +[5,0,2,0]
    assert w["option_len_hist"] == [40 + 15, 0, 4 + 2, 2 + 1]
    assert w["simulations"] == 7 * sims
    assert w["sims_crossing_option"] == 7 + 2
    assert w["crossed_len_hist"] == [0, 0, 5 + 1, 2 + 1]
    assert w["leaf_depth_sum"] == 360.0 + 60.0
    assert w["leaf_depth_max"] == 9
    # DEPARTURE 7 over the 6 acted positions: hits at A's 0 1 3 and B's 0
    assert (w["acted"], w["hit"]) == (6, 4)
    assert w["pi_m"] == pytest.approx(0.6 + 0.7 + 0.1 + 0.9 + 0.5 + 0.25)
    # read (4): A at t = 1 (2,5,2) prev 2 vs played 2 5 2 -> 3/3, repeat
    # hits at steps 1 and 3; A at t = 3 (2,2) prev 5 vs played 2 then the
    # life's end -> step 1 only, a hit, repeat miss. B's t = 1 has none.
    assert w["s4_n"] == [2, 1, 1]
    assert w["s4_opt"] == [2, 1, 1]
    assert w["s4_rep"] == [1, 0, 1]
    assert w["s4_hist"][0][2] == 2 and sum(w["s4_hist"][0]) == 2
    assert w["s4_hist"][1][5] == 1 and w["s4_hist"][2][2] == 1
    assert (w["with_option"], w["nonrepeat"]) == (4, 2)
    assert w["root_len_hist"] == [2, 0, 1, 3]
    assert w["play"][2] == 3 and w["play"][5] == 1 and w["play"][7] == 2
    assert w["frames"] == 6 * 4
    row = ablation.option_row_fields(w, None, 4)
    assert row["opt_node_option_rate"] == round(9 / 64, 6)
    assert row["opt_crossing_rate"] == round(9 / (7 * sims), 6)
    assert row["opt_leaf_depth_mean"] == round(420 / (7 * sims), 4)
    assert row["opt_hit_rate"] == round(4 / 6, 4)
    assert row["opt_pi_m"] == round(3.05 / 6, 4)
    assert row["opt_s4_mode"] == 2 and row["opt_s4_marginal_hits"] == [2, 0, 1]
    assert row["opt_decisions_per_s"] == 15.0


def test_the_records_hold_the_birth_pmax_the_hit_rate_reads(of, monkeypatch):
    stub_engine(monkeypatch, TABLE)
    run = stub_run(of)
    out = feed(run, LIFE_B)
    assert [r["pmax"] for r in out[:-1]] == [7, 1]
    assert [r["root_option"] for r in out[:-1]] == [(7, 7, 7), (-1, -1, -1)]


def test_rows_restart_and_carry_counts_beside_rates(of, monkeypatch):
    stub_engine(monkeypatch, TABLE)
    run = stub_run(of)
    seen = []
    monkeypatch.setattr(ablation.Run, "_option_probe",
                        lambda self, obs, row: seen.append(
                            (len(obs), row)) or None)
    feed(run, LIFE_A)
    row1 = {}
    run._merge_option_stats(row1)
    assert seen == [(4, 0)]
    assert row1["opt_roots"] == 4 and row1["opt_acted"] == 4
    feed(run, LIFE_B)
    row2 = {}
    run._merge_option_stats(row2)
    # the second row holds life B alone; the probe's row index moved on
    assert seen[-1] == (2, 1)
    assert row2["opt_roots"] == 3 and row2["opt_acted"] == 2
    assert row2["opt_nodes"] == 18 and row2["opt_sims_crossing"] == 2
    # pooled over the two rows from their counts: the two-life window's
    assert row1["opt_nodes"] + row2["opt_nodes"] == 64
    assert (row1["opt_nodes_with_option"] + row2["opt_nodes_with_option"]
            == 9)
    row3 = {}
    run._merge_option_stats(row3)            # an empty row
    assert row3["opt_roots"] == 0 and row3["opt_node_option_rate"] is None
    assert seen[-1] == (0, 2)
    for r in (row1, row2, row3):
        assert json.loads(json.dumps(r)) == r
        for k, v in r.items():
            assert v is None or isinstance(v, (int, float, list)), k


def test_every_row_field_is_a_plain_json_number_or_list(of, monkeypatch):
    stub_engine(monkeypatch, TABLE)
    run = stub_run(of)
    monkeypatch.setattr(ablation.Run, "_option_probe",
                        lambda self, obs, row: {
                            **ablation.option_probe_stats(
                                PON, POFF, ROPT, RLEN), "seconds": 0.5})
    feed(run, LIFE_A)
    row = {}
    run._merge_option_stats(row)

    def plain(v):
        if isinstance(v, list):
            return all(plain(x) for x in v)
        return v is None or (type(v) in (int, float) and math.isfinite(v))
    assert row and all(plain(v) for v in row.values())
    assert all(k.startswith("opt_") for k in row)


# ==========================================================================
# the probe's roots: the rule
# ==========================================================================
def test_the_probe_takes_the_smallest_keys_of_a_pinned_stream(of,
                                                               monkeypatch):
    stub_engine(monkeypatch, TABLE)
    monkeypatch.setattr(ablation, "OPTION_PROBE_ROOTS", 3)
    run = stub_run(of)
    a = feed(run, LIFE_A)
    b = feed(run, LIFE_B)
    keys = np.random.default_rng([0, 0, 0x0B5E]).random(6)
    acted = [r["obs"] for r in a[:-1]] + [r["obs"] for r in b[:-1]]
    want = [acted[i] for i in np.argsort(keys, kind="stable")[:3]]
    pool = run._opt_probe
    assert len(pool["obs"]) == 3
    assert all(x is y for x, y in zip(pool["obs"], want))
    assert np.array_equal(pool["keys"], np.sort(keys)[:3])


def test_the_probe_roots_do_not_depend_on_how_lives_are_chunked(
        of, monkeypatch):
    # the same 6 acted positions as one life or as two lives: one sample
    stub_engine(monkeypatch, TABLE)
    monkeypatch.setattr(ablation, "OPTION_PROBE_ROOTS", 4)
    one, two = stub_run(of), stub_run(of)
    feed(one, ([0, 1, 2, 3, 10, 11, 12], [2, 2, 5, 2, 7, 7], False))
    feed(two, LIFE_A)
    feed(two, LIFE_B)
    codes = (lambda run: [int(o.reshape(-1)[0]) for o in run._opt_probe["obs"]])
    assert codes(one) == codes(two)
    assert np.array_equal(one._opt_probe["keys"], two._opt_probe["keys"])


def test_the_probe_roots_draw_no_training_stream(of, monkeypatch):
    stub_engine(monkeypatch, TABLE)
    run = stub_run(of)
    seed_all(3)
    before = rng_states()
    run._note_options([{"obs": np.zeros(1), "action": 1,
                        "target_policy": np.ones(A9) / A9, "pmax": 1,
                        "root_option": (-1, -1, -1)}])
    assert same_rng(before, rng_states())


# ==========================================================================
# the on/off probe on real option nets
# ==========================================================================
def _spy_search(monkeypatch, nets_ref):
    seen = []
    real = muzero_vtree_options.VectorizedOptionPUCT.search

    def watch(self, obs, **kw):
        seen.append({**kw, "T": len(obs),
                     "nets": self.nets,
                     "modes": {k: n.training for k, n in self.nets.items()},
                     "noise": np.array(kw["noise_override"])})
        out = real(self, obs, **kw)
        seen[-1]["out"] = out
        return out

    monkeypatch.setattr(muzero_vtree_options.VectorizedOptionPUCT, "search",
                        watch)
    return seen


def test_the_probe_searches_one_pinned_net_twice_with_one_noise_draw(
        of, monkeypatch):
    monkeypatch.setattr(muzero, "ACT_SEARCH_SIMS", 6)
    torch.set_num_threads(2)
    run = real_run(of)
    tr = run.trainer
    suggest(tr)                                  # options everywhere
    obs = [f for f, _, _ in life3(5, [0] * 5, seed=2)][:5]
    seen = _spy_search(monkeypatch, None)
    # leave a mix of modes behind: they must come back exactly
    tr.representation.train()
    tr.prediction.eval()
    tr.dynamics.train()
    tr.value_prefix.eval()
    modes = {n: getattr(tr, n).training for n in ablation.NET_NAMES}
    seed_all(11)
    before = rng_states()
    got = run._option_probe(obs, 7)
    assert same_rng(before, rng_states()), "a training stream was drawn"
    assert {n: getattr(tr, n).training for n in ablation.NET_NAMES} == modes
    assert [s["option_edge"] for s in seen] == [True, False]
    want_noise = np.random.default_rng([0, 7, 0x015E]).dirichlet(
        np.full(A9, muzero.DIRICHLET_ALPHA), size=5)
    for s in seen:
        assert s["T"] == 5 and s["tie_break"] == "first"
        assert s["root_noise"] is True and s["return_info"] is True
        assert s["num_simulations"] == 6
        assert s["c_puct"] == muzero.C_PUCT
        assert s["discount"] == muzero.DISCOUNT
        assert np.array_equal(s["noise"], want_noise)
        assert all(not m for m in s["modes"].values()), "eval mode"
        for k in ("representation", "dynamics", "prediction",
                  "value_prefix"):
            assert s["nets"][k] is getattr(tr, k), "the learner's nets"
    (pon, _, _, ion), (poff, _, _, ioff) = seen[0]["out"], seen[1]["out"]
    assert np.array_equal(ion["root_option"], ioff["root_option"])
    want = ablation.option_probe_stats(pon, poff, ion["root_option"],
                                       ion["root_option_len"])
    assert {k: v for k, v in got.items() if k != "seconds"} == want
    assert want["shift_roots"] == 5            # every root holds (0, 4, 4)
    assert ion["sims_crossing_option"].sum() > 0
    assert ioff["sims_crossing_option"].sum() == 0
    assert got["seconds"] > 0
    # pinned: the same roots at the same row give the same numbers
    again = run._option_probe(obs, 7)
    assert {k: v for k, v in again.items() if k != "seconds"} == want


def test_the_probe_batches_as_the_birth_search(of, monkeypatch):
    monkeypatch.setattr(muzero, "ACT_SEARCH_SIMS", 3)
    monkeypatch.setattr(muzero, "VTREE_REFRESH_BATCH", 2)
    torch.set_num_threads(2)
    run = real_run(of)
    seen = _spy_search(monkeypatch, None)
    obs = [f for f, _, _ in life3(5, [0] * 5, seed=1)][:5]
    got = run._option_probe(obs, 0)
    assert [s["T"] for s in seen] == [2, 2, 1, 2, 2, 1]
    assert got["roots"] == 5
    full = np.random.default_rng([0, 0, 0x015E]).dirichlet(
        np.full(A9, muzero.DIRICHLET_ALPHA), size=5)
    assert np.array_equal(np.concatenate([s["noise"] for s in seen[:3]]), full)
    assert np.array_equal(np.concatenate([s["noise"] for s in seen[3:]]), full)


def test_the_probe_without_roots_runs_nothing(of, monkeypatch):
    run = real_run(of)
    seen = _spy_search(monkeypatch, None)
    assert run._option_probe([], 0) is None and seen == []


def test_the_probe_runs_for_oh_too(monkeypatch, restore_rngs):  # noqa: F811
    # OH's nets carry the same heads and jump dynamics: the probe still
    # compares the edge on and off (the comparison is the subclass's)
    with _arm(OH) as spec:
        monkeypatch.setattr(muzero, "ACT_SEARCH_SIMS", 3)
        torch.set_num_threads(2)
        run = real_run(spec, OH)
        seen = _spy_search(monkeypatch, None)
        got = run._option_probe([f for f, _, _ in life3(2, [0, 0])][:2], 0)
        assert [s["option_edge"] for s in seen] == [True, False]
        assert got["roots"] == 2


# ==========================================================================
# the train-step window
# ==========================================================================
def _last(**over):
    base = {"option_loss": torch.tensor(0.5), "option_mask_share": 0.25,
            "option_jump_share": [0.5, 0.25, 0.0, 0.0, 0.0],
            "option_span_mean": 7.0, "cons_jump_sum": torch.tensor(-3.0),
            "cons_jump_n": 4, "cons_single_sum": torch.tensor(-9.0),
            "cons_single_n": 10, "birth_pos": 30, "birth_pos_n": 40,
            "birth_root": 6, "birth_root_n": 8}
    return {**base, **over}


def test_the_step_window_by_hand(of):
    run = stub_run(of)
    run.trainer._option_last = _last()
    ablation.note_option_step(run)
    assert run.trainer._option_last is None           # consumed
    ablation.note_option_step(run)                    # nothing to add
    run.trainer._option_last = _last(
        option_loss=torch.tensor(1.5), option_mask_share=0.75,
        option_jump_share=[0.0, 0.25, 0.5, 0.0, 0.0], option_span_mean=5.0,
        cons_jump_sum=torch.tensor(-1.0), cons_jump_n=2,
        cons_single_sum=torch.tensor(-3.0), cons_single_n=6, birth_pos=10,
        birth_pos_n=40, birth_root=2, birth_root_n=8)
    ablation.note_option_step(run)
    # a non-finite step: its batch shares count, its loss and consistency
    # do not
    run.trainer._option_last = _last(option_loss=torch.tensor(float("nan")),
                                     option_mask_share=0.5,
                                     option_span_mean=6.0)
    ablation.note_option_step(run)
    row = ablation.option_row_fields(run._opt_window, None, 4)
    assert row["opt_steps"] == 3 and row["opt_loss_steps"] == 2
    assert row["opt_steps_nonfinite"] == 1
    assert row["opt_loss"] == 1.0                       # (0.5 + 1.5) / 2
    assert row["opt_mask_share"] == 0.5                 # (.25 + .75 + .5)/3
    assert row["opt_jump_share"] == [round(1.0 / 3, 4), 0.25,
                                     round(0.5 / 3, 4), 0.0, 0.0]
    assert row["opt_span_mean"] == 6.0                  # (7 + 5 + 6) / 3
    assert row["opt_span_frames"] == 24.0               # x 4 frames
    assert row["opt_cons_jump"] == round(-4.0 / 6, 6)   # sums / edges
    assert row["opt_cons_jump_edges"] == 6
    assert row["opt_cons_single"] == round(-12.0 / 16, 6)
    assert row["opt_cons_single_edges"] == 16
    # DEPARTURE 3: (30 + 10 + 30) of 120 positions, (6 + 2 + 6) of 24 roots
    assert row["opt_birth_target_share"] == round(70 / 120, 4)
    assert row["opt_birth_target_positions"] == 120
    assert row["opt_birth_target_share_roots"] == round(14 / 24, 4)
    assert row["opt_birth_target_roots"] == 24


def test_after_train_step_feeds_it_only_on_option_arms(of, monkeypatch):
    run = stub_run(of)
    run.trainer._option_last = _last()
    ablation.after_train_step(run)
    assert run._opt_window["steps"] == 1
    monkeypatch.setattr(muzero, "OPTION_LENGTH", 1)
    run2 = stub_run(of)
    run2.trainer._option_last = _last()
    ablation.after_train_step(run2)
    assert not hasattr(run2, "_opt_window")
    assert run2.trainer._option_last is not None


def test_the_train_step_reports_jump_and_one_step_consistency(of,
                                                              monkeypatch):
    run = real_run(of)
    tr = run.trainer
    replay = real_lives(0, n_lives=((14, True), (14, True)))
    for r in replay[1]:                       # the second life was refreshed
        r["refreshed_step"] = 39000
    seed_all(1)
    b = build_batch(tr, replay, 12, A9)
    spans, real = b["option_span"], b["real_mask"]
    assert (spans > 1).any() and (~real).any(), "jumps and padding needed"
    projs = []
    orig = tr.projector.forward

    def fwd(latent, with_prediction):
        out = orig(latent, with_prediction)
        projs.append((with_prediction, out.detach().clone()))
        return out
    tr.projector.forward = fwd
    asyncio.run(tr.train_step(b))
    last = tr._option_last
    # the unrolled edges' (target, online) pairs, in order
    assert [w for w, _ in projs] == [False, True] * 5
    per = torch.stack([-torch.nn.functional.cosine_similarity(
        projs[2 * i + 1][1], projs[2 * i][1], dim=1) for i in range(5)],
        dim=1).numpy()
    jump = spans > 1
    single = (spans == 1) & real[:, 1:]
    assert last["cons_jump_n"] == int(jump.sum())
    assert last["cons_single_n"] == int(single.sum())
    assert float(last["cons_jump_sum"]) == pytest.approx(
        float(per[jump].sum()), rel=1e-5, abs=1e-6)
    assert float(last["cons_single_sum"]) == pytest.approx(
        float(per[single].sum()), rel=1e-5, abs=1e-6)
    # DEPARTURE 3 by hand: the windows' acted records and the roots, the
    # refreshed life's not birth targets
    pos = [r for row in b["pmax_refs"] for r in row if r is not None]
    life1 = {id(r) for r in replay[1]}
    assert last["birth_pos_n"] == len(pos)
    assert last["birth_pos"] == sum(1 for r in pos if id(r) not in life1)
    assert last["birth_root_n"] == 12
    assert last["birth_root"] == sum(1 for r in b["sample_refs"]
                                     if id(r) not in life1)
    assert 0 < last["birth_pos"] < last["birth_pos_n"]


# ==========================================================================
# the prefix panel on jump rows
# ==========================================================================
def test_the_prefix_panel_by_row_kind_by_hand():
    spans = np.array([[3, 1, 1], [1, 1, 1], [1, 2, 1], [2, 1, 1]])
    acc = {}
    ablation.option_prefix_panel(
        acc, spans,
        err_k=np.array([0.1, 0.2, 0.3, 0.4]), ctl_k=np.array([1., 2., 3., 4.]),
        err_1=np.array([0.01, 0.02, 0.03, 0.04]),
        ctl_1=np.array([.1, .2, .3, .4]),
        abs1_eval=np.array([1.0, 2.0, 3.0, 5.0]),
        abs1_train=np.array([0.5, 2.0, 3.0, 1.5]))
    f = ablation.option_prefix_fields(acc)
    # jump rows: 0, 2, 3 (any edge jumps); single: 1
    assert f["opt_prefix_jump_rows"] == 3 and f["opt_prefix_single_rows"] == 1
    assert f["opt_prefix_jump"] == round((0.1 + 0.3 + 0.4) / 3, 6)
    assert f["opt_prefix_jump_control"] == round(8.0 / 3, 6)
    assert f["opt_prefix_single"] == 0.2
    assert f["opt_prefix_single_control"] == 2.0
    # first edge jumps: rows 0 and 3
    assert f["opt_prefix_step1_jump_rows"] == 2
    assert f["opt_prefix_step1_jump"] == round(0.05 / 2, 6)
    assert f["opt_prefix_step1_jump_control"] == 0.25
    assert f["opt_prefix_step1_jump_bn_ratio"] == 3.0      # 6 / 2
    assert f["opt_prefix_step1_single_bn_ratio"] == 1.0    # 5 / 5
    assert ablation.option_prefix_fields({})["opt_prefix_jump"] is None


def test_the_row_runs_the_panel_and_the_probe_on_option_nets(of,
                                                             monkeypatch):
    monkeypatch.setattr(muzero, "ACT_SEARCH_SIMS", 3)
    torch.set_num_threads(2)
    run = real_run(of)
    run.replay = real_lives(3)
    run.seed = 0
    # two acted positions into the probe's roots
    run._note_options([{"obs": run.replay[0][t]["obs"], "action": 1,
                        "target_policy": np.ones(A9) / A9, "pmax": 1,
                        "root_option": (-1, -1, -1)} for t in range(2)])
    seen = _spy_search(monkeypatch, None)
    vp = run.trainer.value_prefix
    bn = {k: v.clone() for k, v in vp.state_dict().items()}
    seed_all(10)
    errs = run.evaluate_diagnostics(batches=2)
    # the train-mode BN copy left the head's statistics alone
    assert all(torch.equal(bn[k], v) for k, v in vp.state_dict().items())
    assert [s["T"] for s in seen] == [2, 2]
    assert errs["opt_probe_roots"] == 2 and errs["opt_acted"] == 2
    assert (errs["opt_prefix_jump_rows"] + errs["opt_prefix_single_rows"]
            == 64)
    assert errs["opt_prefix_jump_rows"] > 0
    for k in ("opt_prefix_jump", "opt_prefix_step1_jump_bn_ratio",
              "opt_probe_js"):
        assert errs[k] is not None and math.isfinite(errs[k]), k
    tb = muzero.TRAINING_BATCHES
    assert errs["opt_grad_uses_per_record"] == round(40000 * tb * 6 / 13000,
                                                     3)
    assert errs["opt_replay_obs_bytes"] == sum(r["obs"].nbytes
                                               for t in run.replay for r in t)
    assert json.loads(json.dumps(errs)) is not None
    # the window and the roots restarted, the next row's index pinned
    assert run._opt_window is None and run._opt_probe["row"] == 1
    assert run._opt_probe["obs"] == []


# ==========================================================================
# checkpoint / resume of a partial row
# ==========================================================================
def test_a_resume_keeps_the_partial_rows_counts_and_roots(of, monkeypatch,
                                                         tmp_path):
    stub_engine(monkeypatch, TABLE)
    monkeypatch.setattr(ablation, "OPTION_PROBE_ROOTS", 4)
    monkeypatch.setattr(ablation.Run, "_option_probe",
                        lambda self, obs, row: {
                            "roots": len(obs), "js_sum": float(sum(
                                int(o.reshape(-1)[0]) for o in obs)),
                            "flips": row, "shift_roots": 0,
                            "shift_sum": 0.0})
    torch.manual_seed(0)
    a = make_run(of, OF)
    a.seed = 0
    feed(a, LIFE_A)
    a.trainer._option_last = _last()
    ablation.note_option_step(a)
    first = {}
    a._merge_option_stats(first)                 # row 1 emitted
    feed(a, LIFE_B)                              # row 2 open: life B
    a.trainer._option_last = _last(option_span_mean=4.0)
    ablation.note_option_step(a)
    path = tmp_path / "r.ckpt"
    ablation.save_run_checkpoint(a, path, {
        "iteration": 7, "elapsed_s": 1.0,
        "training_steps": a.trainer.training_steps})
    head = ablation._read_checkpoint(path)
    assert set(head["option_diag"]) == {"window", "probe"}
    assert head["option_diag"]["probe"]["row"] == 1
    torch.manual_seed(1)
    b = make_run(of, OF)
    b.seed = 0
    ablation.load_run_checkpoint(b, path, {})
    assert b._opt_window == a._opt_window
    assert b._opt_window is not a._opt_window
    for run in (a, b):                           # the same rest of the row
        feed(run, LIFE_A)
        run.trainer._option_last = _last(option_mask_share=0.1)
        ablation.note_option_step(run)
    ra, rb = {}, {}
    a._merge_option_stats(ra)
    b._merge_option_stats(rb)
    assert ra == rb
    assert ra["opt_acted"] == 6 and ra["opt_steps"] == 2
    assert ra["opt_probe_roots"] == 4 and ra["opt_probe_flips"] == 1
    # the pinned stream continued, not restarted: row 2's draws over its 6
    # positions, the 4 smallest kept
    keys = np.random.default_rng([0, 1, 0x0B5E]).random(6)
    codes = [10, 11, 0, 1, 2, 3]
    want = sum(codes[i] for i in np.argsort(keys, kind="stable")[:4])
    assert ra["opt_probe_js"] == round(want / 4, 6)


def test_an_o0_checkpoint_head_carries_no_option_field(o0, tmp_path):
    torch.manual_seed(0)
    run = make_run(o0, O0)
    path = tmp_path / "o0.ckpt"
    ablation.save_run_checkpoint(run, path, {"iteration": 1,
                                             "elapsed_s": 1.0})
    assert "option_diag" not in ablation._read_checkpoint(path)


# ==========================================================================
# no drift: the pre-stage-4a code (git show b4e4d807)
# ==========================================================================
def _git_show(path):
    try:
        return subprocess.run(["git", "show", f"{PRE_S4}:{path}"], cwd=ROOT,
                              capture_output=True, text=True, check=True,
                              encoding="utf-8").stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.fail(f"git show {PRE_S4}:{path} failed, so the no-drift pin "
                    f"cannot run: {exc}")


def _load(name, src, d):
    path = d / f"{name}.py"
    path.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def pre_s4(tmp_path_factory):
    """muzero.py and experiments/ablation.py as they were before stage 4a
    (networks.py is unchanged since): the old ablation imports today's
    muzero (its own code is what it pins)."""
    srcs = {p: _git_show(p) for p in ("muzero.py", "experiments/ablation.py")}
    assert "cons_jump_sum" not in srcs["muzero.py"]
    assert "_merge_option_stats" not in srcs["experiments/ablation.py"]
    d = tmp_path_factory.mktemp("pre_s4")
    path = list(sys.path)
    try:
        old_mz = _load("muzero_pre_s4", srcs["muzero.py"], d)
        old_ab = _load("ablation_pre_s4", srcs["experiments/ablation.py"], d)
    finally:
        sys.path[:] = path
    assert old_ab.muzero is muzero
    return type("Pre", (), {"muzero": old_mz, "ablation": old_ab})


def _copy_knobs(src, dst):
    import re
    knob = re.compile(r"^[A-Z][A-Z0-9_]*$")
    for k in dir(src):
        if knob.match(k) and hasattr(dst, k):
            setattr(dst, k, getattr(src, k))


def _nets(mz):
    return dict(
        representation=mz.build_representation(),
        dynamics=mz.build_dynamics(A9, support_size=mz.REWARD_SUPPORT,
                                   num_bins=mz.REWARD_BINS, transform=True),
        prediction=mz.build_prediction(A9, support_size=mz.VALUE_SUPPORT,
                                       num_bins=mz.VALUE_BINS, transform=True),
        value_prefix=mz.build_value_prefix(
            support_size=mz.VALUE_PREFIX_SUPPORT,
            num_bins=mz.VALUE_PREFIX_BINS, transform=True),
        projector=mz.build_projector())


def _trainer(mz, nets):
    tr = object.__new__(mz.TrainingHelper)
    tr.device, tr.action_space, tr.training_steps = CPU, A9, 40000
    for n, net in nets.items():
        setattr(tr, n, net)
    for n, o in zip(ablation.NET_NAMES, ablation.OPT_NAMES):
        setattr(tr, o, mz.make_optimizer(getattr(tr, n).parameters()))
    return tr


def _where(replay, rec):
    for i, life in enumerate(replay):
        for j, r in enumerate(life):
            if r is rec:
                return i, j
    raise AssertionError("a ref outside the replay")


def _same_state(x, y):
    sx, sy = x.state_dict(), y.state_dict()
    assert list(sx) == list(sy)
    for k in sx:
        assert torch.equal(sx[k], sy[k]), k


def test_the_option_train_step_is_stage_3s_bit_for_bit(of, pre_s4):
    """The stage-4a lines in train_step / _option_inputs are diagnostics
    only: two option batches and steps (batch arrays, loss, every gradient,
    parameters, BN buffers, optimizer state, target nets, priorities, PMAX
    writes, RNG streams) equal the pre-4a code's, and its _option_last keys
    keep their values."""
    old = pre_s4.muzero
    _copy_knobs(muzero, old)
    assert old.OPTION_LENGTH == 3
    torch.manual_seed(0)
    new_nets = _nets(muzero)
    torch.manual_seed(0)
    old_nets = _nets(old)
    t_new, t_old = _trainer(muzero, new_nets), _trainer(old, old_nets)
    dezero_policy(t_new)
    dezero_policy(t_old)
    r_new, r_old = real_lives(4), real_lives(4)
    for r in r_new[1] + r_old[1]:
        r["refreshed_step"] = 39000
    for step in range(2):
        seed_all(100 + step)
        b_old = t_old.finish_batch_paper(
            t_old.assemble_batch_paper(r_old, 12, 5, A9))
        s_old = rng_states()
        seed_all(100 + step)
        b_new = t_new.finish_batch_paper(
            t_new.assemble_batch_paper(r_new, 12, 5, A9))
        assert same_rng(s_old, rng_states())
        assert list(b_new) == list(b_old)
        for k in b_new:
            if k in ("sample_refs",):
                assert [_where(r_new, x) for x in b_new[k]] == \
                    [_where(r_old, x) for x in b_old[k]]
            elif k == "pmax_refs":
                assert [[None if x is None else _where(r_new, x) for x in row]
                        for row in b_new[k]] == \
                    [[None if x is None else _where(r_old, x) for x in row]
                     for row in b_old[k]]
            elif isinstance(b_new[k], np.ndarray):
                assert np.array_equal(b_new[k], b_old[k]), k
            else:
                assert b_new[k] == b_old[k], k
        assert (b_new["option_span"] > 1).any()
        seed_all(200 + step)
        l_old = asyncio.run(t_old.train_step(b_old))
        s_old = rng_states()
        seed_all(200 + step)
        l_new = asyncio.run(t_new.train_step(b_new))
        assert same_rng(s_old, rng_states())
        assert l_new == l_old
        for n in ablation.NET_NAMES:
            pn = dict(getattr(t_new, n).named_parameters())
            po = dict(getattr(t_old, n).named_parameters())
            for k in pn:
                gn, go = pn[k].grad, po[k].grad
                assert (gn is None) == (go is None), (n, k)
                assert gn is None or torch.equal(gn, go), (n, k)
            _same_state(getattr(t_new, n), getattr(t_old, n))
        for o in ablation.OPT_NAMES:
            sx = getattr(t_new, o).state_dict()
            sy = getattr(t_old, o).state_dict()
            assert sx["param_groups"] == sy["param_groups"]
            for k in sx["state"]:
                for f in sx["state"][k]:
                    u, v = sx["state"][k][f], sy["state"][k][f]
                    assert (torch.equal(u, v) if torch.is_tensor(u)
                            else u == v), (o, k, f)
        for n in ("target_representation", "target_prediction",
                  "target_dynamics", "target_value_prefix"):
            _same_state(getattr(t_new, n), getattr(t_old, n))
        assert [[(r["priority"], r.get("pmax")) for r in life]
                for life in r_new] == \
            [[(r["priority"], r.get("pmax")) for r in life] for life in r_old]
        on, oo = t_new._option_last, t_old._option_last
        assert set(oo) < set(on)
        for k in oo:
            u, v = on[k], oo[k]
            assert (torch.equal(u, v) if torch.is_tensor(u) else u == v), k
        t_new.training_steps += 1
        t_old.training_steps += 1


def test_an_o0_life_row_and_step_are_the_pre_stage_4a_code(o0, pre_s4,
                                                           monkeypatch,
                                                           tmp_path):
    """FLAG OFF: an O0 life through paper_targets (records, windows, RNG),
    its row (evaluate_diagnostics: the prediction-error panel, the pre-check
    shares, the behaviour means, the dormant read), the step hook and the
    checkpoint head equal the pre-4a code's, and no option field or window
    appears."""
    assert muzero.OPTION_LENGTH == 1
    monkeypatch.setattr(muzero, "ACT_SEARCH_SIMS", 6)
    torch.set_num_threads(2)
    old_ab = pre_s4.ablation
    for k in ("ACTIONS", "ENV_NAME", "FRAMESKIP", "DEVICE"):
        monkeypatch.setattr(old_ab, k, getattr(ablation, k))
    torch.manual_seed(0)
    run = make_run(o0, O0)
    dezero_policy(run.trainer)
    run.trainer._ensure_target_model()
    old_run = object.__new__(old_ab.Run)
    old_run.__dict__.update(copy.copy(run.__dict__))
    run.replay, old_run.replay = [], []
    for terminated, seed in ((True, 1), (False, 2), (True, 3)):
        traj = life3(9, [1, 2, 3, 4, 5, 6, 7, 8, 0], seed=seed)
        seed_all(5 + seed)
        out_old = old_ab.Run.paper_targets(old_run, traj, terminated,
                                           born_step=39990)
        s_old = rng_states()
        seed_all(5 + seed)
        out_new = run.paper_targets(traj, terminated, born_step=39990)
        assert same_rng(s_old, rng_states())
        for x, y in zip(out_new, out_old):
            assert list(x) == list(y)
            for k in x:
                u, v = x[k], y[k]
                assert (np.array_equal(u, v) if isinstance(u, np.ndarray)
                        else u == v), k
        run.replay.append(out_new)
        old_run.replay.append(out_old)
    assert not hasattr(run, "_opt_window") and not hasattr(run, "_opt_probe")
    seed_all(9)
    e_old = old_run.evaluate_diagnostics(batches=2)
    s_old = rng_states()
    seed_all(9)
    e_new = run.evaluate_diagnostics(batches=2)
    assert same_rng(s_old, rng_states())
    drop = {"birth_search_s_last"}
    assert {k: v for k, v in e_new.items() if k not in drop} == \
        {k: v for k, v in e_old.items() if k not in drop}
    assert "pc_positions" in e_new and "beh_act_prob" in e_new
    assert not any(k.startswith("opt_") for k in e_new)
    # the step hook
    for r in (run, old_run):
        r.trainer._last_step = (1.25, torch.tensor(0.5))
        r._step_window = None
    ablation.after_train_step(run)
    w_new = run._step_window
    old_ab.after_train_step(old_run)
    assert w_new == old_run._step_window
    assert not hasattr(run, "_opt_window")
    # the checkpoint head
    p_new, p_old = tmp_path / "n.ckpt", tmp_path / "o.ckpt"
    seed_all(1)
    ablation.save_run_checkpoint(run, p_new, {"iteration": 3,
                                              "elapsed_s": 2.0})
    seed_all(1)
    old_ab.save_run_checkpoint(old_run, p_old, {"iteration": 3,
                                                "elapsed_s": 2.0})
    h_new = ablation._read_checkpoint(p_new)
    h_old = ablation._read_checkpoint(p_old)
    assert list(h_new) == list(h_old)

    def untimed(d):
        return {k: {kk: vv for kk, vv in v.items()
                    if kk != "birth_search_s_last"} for k, v in d.items()}
    assert untimed(h_new["diag_stats"]) == untimed(h_old["diag_stats"])
    assert "option_diag" not in h_new


# ==========================================================================
# end to end: real Ms. Pac-Man, in-process pipeline, CPU (OF)
# ==========================================================================
@pytest.mark.slow
def test_run_one_writes_the_option_fields_into_each_of_row(
        monkeypatch, tmp_path, restore_rngs):  # noqa: F811
    torch.set_num_threads(2)
    fake = _fake_pipe([], 0)
    fake.snapshot_acting_states = lambda self: {}
    monkeypatch.setattr(ablation, "RayPipeline", fake)
    monkeypatch.setattr(ablation, "ENV_NAME", "MsPacmanNoFrameskip-v4")
    monkeypatch.setattr(ablation, "FRAMESKIP", 4)
    monkeypatch.setattr(ablation, "ACTIONS", A9)
    monkeypatch.setattr(ablation, "EPISODE_CAP", 8)
    monkeypatch.setattr(ablation, "RESULTS", tmp_path)
    monkeypatch.setattr(ablation, "TAG", "")
    monkeypatch.setattr(ablation, "DEVICE", torch.device("cpu"))
    monkeypatch.setitem(ablation.ARMS, "e2e-of", {
        **ablation.ARMS[OF], "TRAINING_BATCHES": 16, "ACT_SEARCH_SIMS": 6})
    out = tmp_path / "a.jsonl"
    ablation.run_one("e2e-of", 0, 300, 1, 4, 1, out, checkpoint_every=1,
                     use_ray=True, cpu_actor=True, snapshot_weights=True,
                     stop_env_steps=16)
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert [r["iteration"] for r in rows] == [1, 2]    # 8 env steps a life
    for r in rows:
        # the pre-check beside the option fields, O0's fields kept
        assert r["pc_positions"] == 8 and r["opt_acted"] == 8
        assert r["opt_roots"] == 8 + 1                 # a cut life's tail
        assert r["opt_sims"] == 9 * 6
        assert r["opt_probe_roots"] == 8
        assert r["opt_steps"] == 4 and r["opt_loss_steps"] == 4
        assert len(r["opt_jump_share"]) == 5
        assert len(r["opt_len_hist"]) == len(r["opt_crossed_len_hist"]) == 4
        assert sum(r["opt_len_hist"]) == r["opt_nodes"]
        assert r["opt_nodes_with_option"] == sum(r["opt_len_hist"][2:])
        assert r["opt_decisions_per_s"] == 15.0
        assert r["opt_probe_s"] is not None
        assert r["replay_records"] > 0 and "dormant_025" in r
        assert not any(k.startswith("tok_") for k in r)
