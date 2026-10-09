"""Policy-only acting on the faithful control's refresh engine (Kai,
2026-10-05): effzero-repro-faithful-prioract-noredo-seqref. Pins that its
only change from prioract2's arm is the refresh engine, that its effective
delta from the faithful control is only the acting/pipeline side, and
that the --ray loop really refreshes through the sequential engine."""
import ast
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] /
                       "experiments"))
import ablation
import muzero
from tests.test_prioract2 import _fake_pipe, restore_rngs  # noqa: F401

ARM = "effzero-repro-faithful-prioract-noredo-seqref"
TWIN = "effzero-repro-faithful-prioract-noredo"
CONTROL = "effzero-repro-refresh5x-communal-edge-faithful"
SUP = (pathlib.Path(__file__).resolve().parents[1] / "benchmarks" /
       "supervisors" / "supervise_seqref.py")


def _eff(spec, k):
    return spec[k] if k in spec else getattr(muzero, k, None)


def test_only_the_refresh_engine_differs_from_prioract2():
    a, b = ablation.ARMS[TWIN], ablation.ARMS[ARM]
    assert {k for k in set(a) | set(b) if a.get(k) != b.get(k)} == {
        "REFRESH_VTREE", "VTREE_COMMUNAL_TRAJS", "_doc"}
    assert b["REFRESH_VTREE"] is False and b["VTREE_COMMUNAL_TRAJS"] == 1
    assert b["_launch"] == {"ray": True, "cpu_actor": True}


def test_effective_delta_from_the_control_is_only_the_acting_side():
    """Exactly the pre-registered confounds (CURRICULUM 2026-10-05)."""
    c, n = ablation.ARMS[CONTROL], ablation.ARMS[ARM]
    delta = {k for k in set(c) | set(n)
             if k != "_doc" and _eff(c, k) != _eff(n, k)}
    assert delta == {"USE_SEARCH_ACTING", "BIRTH_SEARCH",
                     "PRIOR_ACT_STAGED_TEMP", "RAY_EAGER_COLLECT",
                     "REDO_NETS", "STEP_DIAGNOSTICS", "_launch"}
    # the refresh knobs equal the control's
    for k in ("REFRESH_VTREE", "VTREE_COMMUNAL_TRAJS", "REFRESH_OFFLOAD",
              "REFRESH_SATURATE", "POLICY_REFRESH_EVERY",
              "POLICY_REFRESH_COUNT"):
        assert _eff(c, k) == _eff(n, k), k
    assert not _eff(n, "REDO_EVERY")               # REDO_NETS is inert


def test_every_startup_gate_passes():
    with ablation.arm_config(ARM) as spec:
        ablation.ray_scope_gate(spec)
        ablation.launch_gate(spec, True, True)
        ablation.eager_collect_gate(True)
        ablation.prior_temp_gate()
        ablation.birth_search_gate()
        ablation.sve_gate()
    with pytest.raises(SystemExit, match="pins its launch mode"):
        ablation.launch_gate(ablation.ARMS[ARM], False, False)


@pytest.mark.parametrize("ready_after", [0, 10_000])
def test_ray_loop_refreshes_through_the_sequential_engine(
        monkeypatch, tmp_path, restore_rngs, ready_after):  # noqa: F811
    """run_one --ray (in-process fake pipeline) for 2 iterations: the
    iteration-2 refresh goes through run.refresh_targets (the control's
    sequential engine), never refresh_targets_vtree or an offload -- and
    (seqref audit r1 L1) it SEES both games (game 2 collected mid-block or
    at the iteration end), stamps every position, and the row counts them."""
    import torch
    torch.set_num_threads(2)
    calls = []
    seen = {}
    real = ablation.Run.refresh_targets

    def spy_seq(self, count):
        calls.append(("seq", count, self.trainer.training_steps))
        seen["run"], seen["n_trajs"] = self, len(self.replay)
        out = real(self, count)
        seen["stamps"] = [{r.get("refreshed_step") for r in t}
                          for t in self.replay]
        seen["positions"] = sum(len(t) for t in self.replay)
        return out

    def no_vtree(self, count):
        raise AssertionError("refresh_targets_vtree called")

    monkeypatch.setattr(ablation.Run, "refresh_targets", spy_seq)
    monkeypatch.setattr(ablation.Run, "refresh_targets_vtree", no_vtree)
    events = []
    fake = _fake_pipe(events, ready_after)
    fake.request_refresh = lambda self: (_ for _ in ()).throw(
        AssertionError("offloaded refresh"))
    monkeypatch.setattr(ablation, "RayPipeline", fake)
    monkeypatch.setattr(ablation, "ENV_NAME", "Pong-v4")
    monkeypatch.setattr(ablation, "FRAMESKIP", 4)
    monkeypatch.setattr(ablation, "EPISODE_CAP", 10)
    monkeypatch.setattr(ablation, "RESULTS", tmp_path)
    monkeypatch.setattr(ablation, "TAG", "")
    monkeypatch.setitem(ablation.ARMS, "e2e-seqref", {
        **ablation.ARMS[ARM], "TRAINING_BATCHES": 16})
    out = tmp_path / "ablation.jsonl"
    ablation.run_one("e2e-seqref", 0, 2, 1, 10, 1, out, checkpoint_every=1,
                     use_ray=True, cpu_actor=True)
    assert calls == [("seq", _eff(ablation.ARMS[ARM], "POLICY_REFRESH_COUNT"), 20)]
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert [r["iteration"] for r in rows] == [1, 2]
    assert seen["n_trajs"] == 2                    # both games, both modes
    assert seen["stamps"] == [{20}, {20}]          # every position stamped
    assert rows[1]["refresh_steps_last_iter"] == seen["positions"] > 0


def test_supervisor_variant_and_seed_order():
    tree = ast.parse(SUP.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and getattr(n.targets[0], "id", "") == "VARIANTS")
    variants = ast.literal_eval(node.value)
    assert set(variants) == {"ray"}
    arm, stem, flags = variants["ray"]
    assert arm == ARM and stem == "pong-seqref"
    assert flags == ["--ray", "--cpu-actor"]
    # seed 1 waits for seed 0's completion marker; seed 0 waits for nothing
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == "wait_ckpts")
    ns = {"os": __import__("os")}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "w", "exec"), ns)
    assert ns["wait_ckpts"](0, "R", "A") == []
    assert ns["wait_ckpts"](1, "R", "A") == [__import__("os").path.join(
        "R", "benchmarks", "results", "pong-seqref-s0-2026-10-05",
        "resume-pong-skip4-A-seed0.ckpt")]
