"""The factorization arms after the prioract bisect read WORSE (2026-09-30):
effzero-repro-faithful-searchact-noredo (--ray --cpu-actor) and
effzero-repro-faithful-seqvtree (sequential). Their flags must be
identical, so the ONLY difference between the two runs is the launch
mode (pipelined CPU actor vs sequential acting)."""
import sys

sys.path.insert(0, str(__import__("pathlib").Path(
    __file__).resolve().parents[1] / "experiments"))
import ablation
import muzero

NOREDO = "effzero-repro-faithful-searchact-noredo"
SEQ = "effzero-repro-faithful-seqvtree"
SEARCHACT = "effzero-repro-faithful-searchact"
FAITHFUL = "effzero-repro-refresh5x-communal-edge-faithful"


def _delta(a, b):
    a, b = ablation.ARMS[a], ablation.ARMS[b]
    return {k for k in set(a) | set(b) if a.get(k) != b.get(k)}


def test_noredo_is_searchact_with_redo_off():
    assert _delta(SEARCHACT, NOREDO) == {"REDO_EVERY", "STEP_DIAGNOSTICS",
                                         "_launch", "_doc"}
    assert ablation.ARMS[NOREDO]["REDO_EVERY"] == 0
    assert ablation.ARMS[NOREDO]["_launch"] == {"ray": True, "cpu_actor": True}


def test_seqvtree_is_the_faithful_control_plus_the_vectorized_research():
    assert _delta(FAITHFUL, SEQ) == {"REFRESH_VTREE", "VTREE_COMMUNAL_TRAJS",
                                     "EVAL_PRIOR_ONLY", "STEP_DIAGNOSTICS",
                                     "_launch", "_doc"}
    assert ablation.ARMS[SEQ]["_launch"] == {"ray": False}
    spec = ablation.ARMS[SEQ]
    assert spec["REFRESH_VTREE"] is True and spec["VTREE_COMMUNAL_TRAJS"] == 8
    assert spec["EVAL_PRIOR_ONLY"] is True
    assert not spec.get("REDO_EVERY") and not spec.get("BIRTH_SEARCH")
    assert spec["USE_SEARCH_ACTING"] is True


def _effective_delta(a, b):
    """Keys whose EFFECTIVE value differs: an arm's own setting, else the
    muzero module default (what arm_config leaves in place)."""
    a, b = ablation.ARMS[a], ablation.ARMS[b]

    def eff(spec, k):
        return spec[k] if k in spec else getattr(muzero, k, None)
    return {k for k in set(a) | set(b) if eff(a, k) != eff(b, k)}


def test_the_two_factor_arms_differ_only_in_inert_redo_knobs():
    """Effective flags identical except REDO_NETS (read only when
    REDO_EVERY > 0 -- 0 in both); the launch mode is the only real delta."""
    assert _effective_delta(NOREDO, SEQ) <= {"REDO_TAU", "REDO_NETS",
                                             "_launch", "_doc"}
    assert not ablation.ARMS[NOREDO]["REDO_EVERY"]
    assert not ablation.ARMS[SEQ].get("REDO_EVERY")



import pytest


@pytest.mark.parametrize("arm,use_ray,cpu_actor", [
    (NOREDO, False, False), (NOREDO, True, False), (SEQ, True, True)])
def test_launch_gate_refuses_the_wrong_mode(monkeypatch, arm, use_ray,
                                            cpu_actor):
    """Factor audit r1 L3: a relaunch in the wrong mode fails at startup
    (out=None: nothing can be touched)."""
    with pytest.raises(SystemExit, match="pins its launch mode"):
        ablation.run_one(arm, 0, 1, 1, 1, 1, None, use_ray=use_ray,
                         cpu_actor=cpu_actor)


def test_launch_gate_passes_the_right_mode_and_unpinned_arms():
    ablation.launch_gate(ablation.ARMS[NOREDO], True, True)
    ablation.launch_gate(ablation.ARMS[SEQ], False, False)
    ablation.launch_gate(ablation.ARMS[SEARCHACT], False, False)   # unpinned


def test_supervisor_variants_match_the_pinned_launch_modes():
    """The shared supervisor's VARIANTS flags agree with each arm's pin."""
    import ast
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1] / "benchmarks" /
           "supervisors" / "supervise_factor.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and getattr(n.targets[0], "id", "") == "VARIANTS")
    variants = ast.literal_eval(node.value)
    for arm, _stem, flags in variants.values():
        want = ablation.ARMS[arm]["_launch"]
        assert ("--ray" in flags) == want["ray"], arm
        assert ("--cpu-actor" in flags) == want.get("cpu_actor", False), arm


def test_step_diagnostics_without_redo(monkeypatch):
    """STEP_DIAGNOSTICS records the row step window with ReDo off and never
    runs a ReDo pass; with both off after_train_step touches nothing."""
    import torch
    from tests.test_checkpoint_resume import make_run
    from tests.test_reanalyze_refresh import obs_traj
    run = make_run()
    run.replay = [obs_traj([0.0] * 3, [0.0] * 3)]
    monkeypatch.setattr(muzero, "REDO_EVERY", 0)
    monkeypatch.setattr(run, "redo_step", lambda: pytest.fail("ReDo ran"))
    monkeypatch.setattr(muzero, "STEP_DIAGNOSTICS", False)
    run.trainer.training_steps = 1500
    run.trainer._last_step = (0.5, torch.tensor(2.0))
    ablation.after_train_step(run)
    assert getattr(run, "_step_window", None) is None
    monkeypatch.setattr(muzero, "STEP_DIAGNOSTICS", True)
    for s, gn in ((1500, 2.0), (1501, 4.0)):
        run.trainer.training_steps = s
        run.trainer._last_step = (0.5, torch.tensor(gn))
        ablation.after_train_step(run)
    row = {}
    run._merge_row_stats(row)
    assert row["step_grad_norm_mean"] == pytest.approx(3.0)
    assert row["step_loss_mean"] == pytest.approx(0.5)
    assert "redo_gn_pre20" not in row
