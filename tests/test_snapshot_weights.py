"""2026-09-21 instrumentation for the edge-visit run: the archfix-exact edge
arm, --snapshot-weights (nets at every eval row), and the projector
dimensional-collapse panel (proj_erank / proj_dead_frac) in eval rows."""
import sys

import numpy as np
import pytest
import torch

import muzero
from tests.test_checkpoint_resume import make_loop_stub, make_run
from tests.test_reanalyze_refresh import obs_traj

sys.path.insert(0, str(__import__("pathlib").Path(
    __file__).resolve().parent.parent / "experiments"))
import ablation

WEIGHT_KEYS = {"arm", "seed", "spec", "training_steps", *ablation.NET_NAMES}


def test_edge_arm_is_archfix_plus_exactly_two_flags():
    base = ablation.ARMS["effzero-repro-refresh5x-communal"]
    edge = ablation.ARMS["effzero-repro-refresh5x-communal-edge"]
    delta = {k for k in set(base) | set(edge)
             if k != "_doc" and base.get(k, "<absent>") != edge.get(k, "<absent>")}
    assert delta == {"COMMUNAL_EDGE_VISITS", "BOOTSTRAP_CACHE"}
    assert edge["COMMUNAL_EDGE_VISITS"] is True
    # Both are REAL deltas against the module defaults the base arm runs
    # under, not restatements: archfix predates the cache (default on) and
    # the legacy readout is the default.
    assert edge["BOOTSTRAP_CACHE"] is False and muzero.BOOTSTRAP_CACHE is True
    assert muzero.COMMUNAL_EDGE_VISITS is False
    # archfix's sequential communal engine, not the vectorized one.
    assert not edge.get("REFRESH_VTREE") and edge["USE_COMMUNAL_REANALYSIS"]


def test_augfix_arm_is_edge_plus_exactly_the_augmentation_flag():
    edge = ablation.ARMS["effzero-repro-refresh5x-communal-edge"]
    fix = ablation.ARMS["effzero-repro-refresh5x-communal-edge-augfix"]
    delta = {k for k in set(edge) | set(fix)
             if k != "_doc" and edge.get(k, "<absent>") != fix.get(k, "<absent>")}
    assert delta == {"AUG_INDEPENDENT_TARGETS"}
    # A real delta against the module default the edge pair ran under.
    assert fix["AUG_INDEPENDENT_TARGETS"] is True
    assert muzero.AUG_INDEPENDENT_TARGETS is False
    assert fix["USE_AUGMENTATION"] is True
    with ablation.arm_config("effzero-repro-refresh5x-communal-edge-augfix"):
        assert muzero.AUG_INDEPENDENT_TARGETS is True
    assert muzero.AUG_INDEPENDENT_TARGETS is False


def test_projector_collapse_stats_measure_spread_and_dead_gammas():
    torch.manual_seed(0)
    proj_net = muzero.build_projector()
    d = 64
    iso = torch.randn(256, d)
    line = torch.randn(256, 1) * torch.randn(1, d)          # rank 1 after centering
    plane = (torch.randn(256, 1) * torch.randn(1, d)
             + torch.randn(256, 1) * torch.randn(1, d))      # rank 2
    same = torch.randn(1, d).repeat(256, 1)                  # no spread at all
    er = lambda x: ablation.projector_collapse_stats(proj_net, x)["proj_erank"]
    assert er(iso) > 0.75 * d
    assert er(line) == pytest.approx(1.0, abs=1e-6)
    assert 1.0 < er(plane) <= 2.0 + 1e-6
    assert er(same) == 0.0
    # The dead-gamma fraction reads the head's FINAL BatchNorm only.
    bns = [m for m in proj_net.projection if isinstance(m, torch.nn.BatchNorm1d)]
    assert ablation.projector_collapse_stats(proj_net, iso)["proj_dead_frac"] == 0.0
    with torch.no_grad():
        n = bns[-1].weight.numel()
        bns[-1].weight[: n // 2] = 0.005
        bns[0].weight.zero_()                               # an earlier BN: ignored
    assert ablation.projector_collapse_stats(proj_net, iso)["proj_dead_frac"] == 0.5


def _stub_env(monkeypatch, tmp_path):
    monkeypatch.setattr(ablation, "RESULTS", tmp_path)
    monkeypatch.setattr(ablation, "TAG", "")
    monkeypatch.setattr(ablation, "ENV_NAME", "Pong-v4")
    monkeypatch.setattr(ablation, "EPISODE_CAP", 600)
    monkeypatch.setattr(ablation, "Run", make_loop_stub({"train": 0, "play": 0}))


def test_snapshot_weights_saves_every_eval_row(tmp_path, monkeypatch):
    _stub_env(monkeypatch, tmp_path)
    ablation.run_one("per-step", 0, 5, 1, 2, 2, tmp_path / "a.jsonl",
                     checkpoint_every=0, resume=False, snapshot_weights=True)
    snap = tmp_path / "snapshots-per-step-seed0"
    # eval_every 2 over 5 iterations: rows 2, 4, and the final row 5.
    assert sorted(p.name for p in snap.iterdir()) == [
        "weights-iter002.pth", "weights-iter004.pth", "weights-iter005.pth"]
    for it in (2, 4, 5):
        d = torch.load(snap / f"weights-iter{it:03d}.pth", weights_only=False)
        assert set(d) == WEIGHT_KEYS | {"iteration"}
        assert d["iteration"] == it and d["arm"] == "per-step" and d["seed"] == 0
    # The final weights file keeps its exact legacy key set (weights_blob
    # refactor): every existing probe and --init-weights loader reads it.
    final = torch.load(tmp_path / "weights-per-step-seed0.pth", weights_only=False)
    assert set(final) == WEIGHT_KEYS


def test_no_snapshots_without_the_flag(tmp_path, monkeypatch):
    _stub_env(monkeypatch, tmp_path)
    ablation.run_one("per-step", 0, 2, 1, 1, 1, tmp_path / "a.jsonl",
                     checkpoint_every=0, resume=False)
    assert not any(p.name.startswith("snapshots") for p in tmp_path.iterdir())
    assert (tmp_path / "weights-per-step-seed0.pth").exists()


def _np_state():
    s = np.random.get_state()   # key array AND position: small draws only move pos
    return s[1].copy(), s[2], s[3], s[4]


def _np_equal(a, b):
    return np.array_equal(a[0], b[0]) and a[1:] == b[1:]


def test_diagnostics_panel_draws_no_rng_and_mutates_no_net(monkeypatch):
    """A real diagnostics pass (real nets, real prepare_batch). The whole
    pass -- projector forward included -- must draw exactly what its
    prepare_batch calls draw and nothing more, and leave every net's state
    (BN running stats and num_batches_tracked included) bitwise unchanged.
    The reference is prepare_batch ALONE, so a projector call that sampled
    or ran train-mode BN is caught (review finding: comparing two passes
    that both call the projector could not see it)."""
    import asyncio
    monkeypatch.setattr(ablation, "DEVICE", torch.device("cpu"))
    run = make_run()
    np.random.seed(4)
    run.replay = [obs_traj([0.0] * 12, [0.5] * 12)]
    run.set_train_mode(False)
    before = {n: {k: v.clone() for k, v in getattr(run.trainer, n).state_dict().items()}
              for n in ablation.NET_NAMES}

    np.random.seed(11); torch.manual_seed(11)
    for _ in range(2):
        asyncio.run(run.trainer.prepare_batch(run.replay, 32, muzero.UNROLL_STEPS,
                                              ablation.ACTIONS))
    ref_np, ref_torch = _np_state(), torch.get_rng_state()

    np.random.seed(11); torch.manual_seed(11)
    errs = run.evaluate_diagnostics(batches=2)
    assert _np_equal(_np_state(), ref_np)
    assert torch.equal(torch.get_rng_state(), ref_torch)
    for n in ablation.NET_NAMES:
        after = getattr(run.trainer, n).state_dict()
        assert after.keys() == before[n].keys()
        assert all(torch.equal(after[k], before[n][k]) for k in after), n

    assert 0.0 < errs["proj_erank"] <= 64
    assert 0.0 <= errs["proj_dead_frac"] <= 1.0


def test_projector_panel_logs_nan_instead_of_raising():
    proj_net = muzero.build_projector()
    x = torch.randn(32, 16)
    x[3, 5] = float("nan")
    out = ablation.projector_collapse_stats(proj_net, x)
    assert np.isnan(out["proj_erank"]) and out["proj_dead_frac"] == 0.0
    x[3, 5] = float("inf")
    assert np.isnan(ablation.projector_collapse_stats(proj_net, x)["proj_erank"])
