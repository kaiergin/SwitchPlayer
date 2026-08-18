"""Checkpoint/resume for the ablation harness: a killed run must come back
with its full state -- nets, optimizers, the target-model double buffer,
replay buffer, counters, and RNG streams."""
import random
import sys

import numpy as np
import pytest
import torch

import muzero
from tests.test_paper_mode import ACTIONS, make_trainer
from tests.test_reanalyze_refresh import obs_traj

sys.path.insert(0, str(__import__("pathlib").Path(
    __file__).resolve().parent.parent / "experiments"))
import ablation

CPU = torch.device("cpu")

META = {"arm": "effzero-repro", "seed": 0, "env": "Pong-v4", "iterations": 60,
        "train_steps_per_iter": 2000, "episodes_per_iter": 1,
        "iteration": 20, "training_steps": 40000, "elapsed_s": 1234.5}
EXPECT = {k: META[k] for k in ("arm", "seed", "env", "iterations",
                               "train_steps_per_iter", "episodes_per_iter")}


def make_run():
    run = object.__new__(ablation.Run)
    run.trainer = make_trainer()
    run.replay = []
    run.env_steps = 0
    return run


def state_dicts_equal(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    return (sa.keys() == sb.keys()
            and all(torch.equal(sa[k], sb[k]) for k in sa))


def test_checkpoint_round_trips_every_piece_of_state(tmp_path):
    np.random.seed(3)
    a = make_run()
    a.trainer.training_steps = META["training_steps"]
    a.env_steps = 12345
    a.replay = [obs_traj([1.0] * 4, [2.0] * 4)]
    # Non-trivial optimizer moments on every net.
    for net_name, opt_name in zip(ablation.NET_NAMES, ablation.OPT_NAMES):
        net, opt = getattr(a.trainer, net_name), getattr(a.trainer, opt_name)
        loss = sum((p ** 2).sum() for p in net.parameters())
        opt.zero_grad(); loss.backward(); opt.step()
    # A target model whose state genuinely differs from the online nets.
    a.trainer._ensure_target_model()
    with torch.no_grad():
        torch.manual_seed(5)
        for p in a.trainer.prediction.parameters():
            p.add_(torch.randn_like(p))
    path = tmp_path / "resume.ckpt"
    ablation.save_run_checkpoint(a, path, META)
    # RNG draws AFTER the save: the restored streams must reproduce these.
    want_np = np.random.rand(3)
    want_py = random.random()
    want_torch = torch.rand(2)

    b = make_run()
    loaded = ablation.load_run_checkpoint(b, path, EXPECT)
    assert loaded == {"iteration": 20, "elapsed_s": 1234.5}
    assert b.trainer.training_steps == META["training_steps"]
    assert b.env_steps == 12345
    for net_name in ablation.NET_NAMES:
        assert state_dicts_equal(getattr(a.trainer, net_name),
                                 getattr(b.trainer, net_name)), net_name
    for net_name in ablation.NET_NAMES[:4]:
        assert state_dicts_equal(getattr(a.trainer, f"target_{net_name}"),
                                 getattr(b.trainer, f"target_{net_name}"))
    # The double buffer itself: snapshot tensors and the sync clock. The
    # online prediction was mutated after the sync, so target != online here
    # -- a load that silently re-synced would fail the target comparison.
    assert b.trainer._target_synced_at == a.trainer._target_synced_at
    for da, db in zip(a.trainer._target_snapshot, b.trainer._target_snapshot):
        assert da.keys() == db.keys()
        assert all(torch.equal(da[k].cpu(), db[k].cpu()) for k in da)
    assert not state_dicts_equal(b.trainer.prediction,
                                 b.trainer.target_prediction)
    # Optimizer moments.
    for opt_name in ablation.OPT_NAMES:
        oa = getattr(a.trainer, opt_name).state_dict()
        ob = getattr(b.trainer, opt_name).state_dict()
        assert len(oa["state"]) == len(ob["state"]) > 0, opt_name
        for k in oa["state"]:
            assert torch.equal(oa["state"][k]["exp_avg"].cpu(),
                               ob["state"][k]["exp_avg"].cpu())
    # Replay byte-exact.
    assert len(b.replay) == 1 and len(b.replay[0]) == len(a.replay[0])
    for ta, tb in zip(a.replay[0], b.replay[0]):
        assert np.array_equal(ta["obs"], tb["obs"])
        assert ta["reward"] == tb["reward"]
    # RNG streams continue exactly where the save left them.
    assert np.allclose(np.random.rand(3), want_np)
    assert random.random() == want_py
    assert torch.equal(torch.rand(2), want_torch)


def test_resume_refuses_a_mismatched_recipe(tmp_path):
    np.random.seed(4)
    a = make_run()
    path = tmp_path / "resume.ckpt"
    ablation.save_run_checkpoint(a, path, META)
    b = make_run()
    with pytest.raises(ValueError, match="iterations"):
        ablation.load_run_checkpoint(b, path, {**EXPECT, "iterations": 90})


def test_completed_marker_skips_the_run(tmp_path):
    path = tmp_path / "resume.ckpt"
    ablation.mark_run_completed(path, EXPECT)
    assert path.stat().st_size < 100_000   # marker, not a buffer dump
    b = make_run()
    assert ablation.load_run_checkpoint(b, path, EXPECT) is None


def test_completed_marker_with_wrong_recipe_raises(tmp_path):
    """Audit finding: the completed check ran BEFORE recipe validation, so a
    finished 60-iter marker silently skipped a NEW 90-iter --resume
    invocation instead of erroring."""
    path = tmp_path / "resume.ckpt"
    ablation.mark_run_completed(path, EXPECT)
    b = make_run()
    with pytest.raises(ValueError, match="iterations"):
        ablation.load_run_checkpoint(b, path, {**EXPECT, "iterations": 90})


def test_resume_prunes_stale_timeline_rows(tmp_path):
    """Audit finding (HIGH): rows written after the last checkpoint belong
    to the abandoned timeline; resume must drop them or plot() averages
    both timelines as if they were extra seeds."""
    out = tmp_path / "ablation.jsonl"
    rows = [{"arm": "effzero-repro", "seed": 0, "iteration": it, "return_mean": it}
            for it in (10, 20, 30)]
    rows.append({"arm": "effzero-repro", "seed": 1, "iteration": 30,
                 "return_mean": 99})   # other seed: must survive
    with open(out, "w") as fh:
        for r in rows:
            fh.write(__import__("json").dumps(r) + "\n")
    pruned = ablation.prune_stale_rows(out, "effzero-repro", 0, start_it=20)
    assert pruned == 1
    kept = [__import__("json").loads(l) for l in open(out)]
    assert [(r["seed"], r["iteration"]) for r in kept] == [(0, 10), (0, 20), (1, 30)]
    # Idempotent when nothing is stale.
    assert ablation.prune_stale_rows(out, "effzero-repro", 0, start_it=20) == 0


def test_fresh_start_refuses_an_existing_checkpoint(tmp_path):
    """Audit finding: a fresh run at the same path clobbered a killed run's
    resumable state at its first save; its completion marker then satisfied
    the original run's later --resume silently."""
    ck = tmp_path / "resume.ckpt"
    ablation.refuse_unresumed_checkpoint(ck, resume=False)   # missing: fine
    ck.write_bytes(b"x")
    ablation.refuse_unresumed_checkpoint(ck, resume=True)    # resuming: fine
    with pytest.raises(SystemExit, match="--resume"):
        ablation.refuse_unresumed_checkpoint(ck, resume=False)


def test_failed_save_preserves_the_previous_checkpoint(tmp_path, monkeypatch):
    np.random.seed(6)
    a = make_run()
    path = tmp_path / "resume.ckpt"
    ablation.save_run_checkpoint(a, path, META)

    def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(ablation.torch, "save", boom)
    with pytest.raises(RuntimeError):
        ablation.save_run_checkpoint(a, path, {**META, "iteration": 30})
    monkeypatch.undo()
    b = make_run()
    loaded = ablation.load_run_checkpoint(b, path, EXPECT)
    assert loaded["iteration"] == 20, "failed save clobbered the checkpoint"
