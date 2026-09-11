"""BatchPrefetcher: CPU batch assembly overlapped with GPU training.

The contract that matters for science is BIT-IDENTITY: with PER off, the
sequence of batches a training block consumes must be exactly what the
serial loop would have built for the same seed (the worker is the only
numpy-RNG consumer during a block, and batch i is assembled with
training_steps = base + i). These tests pin that, the split of the
paper builder into a CPU half and a main-thread bootstrap half, and the
lifecycle guarantees (exception surfacing, early close joins).
"""
import asyncio
import copy
import sys

import numpy as np
import pytest
import torch

import muzero
from tests.test_paper_mode import ACTIONS, make_trainer
from tests.test_reanalyze_refresh import obs_traj

sys.path.insert(0, str(__import__("pathlib").Path(
    __file__).resolve().parent.parent / "experiments"))
import ablation  # noqa: E402


def _replay(n=4, seed=3):
    np.random.seed(seed)
    out = []
    for k in range(n):
        traj = obs_traj([1.0] * 6, [3.0] * 6, truncated=(k % 2 == 1))
        for i, t in enumerate(traj):
            t["born_step"] = 10 * k
            t.setdefault("priority", 1.0)
        out.append(traj)
    return out


@pytest.fixture
def world(monkeypatch):
    for key, val in (("PAPER_TARGETS", True), ("PER_ENABLED", False),
                     ("USE_TARGET_MODEL", True), ("UNROLL_STEPS", 3),
                     ("TRAINING_BATCHES", 8), ("OFFPOLICY_DECAY_STEPS", None)):
        monkeypatch.setattr(muzero, key, val)
    trainer = make_trainer()
    trainer.training_steps = 500
    return trainer


def _serial(trainer, replay, n):
    out = []
    for i in range(n):
        out.append(asyncio.run(trainer.prepare_batch(
            replay, muzero.TRAINING_BATCHES, muzero.UNROLL_STEPS, ACTIONS)))
        trainer.training_steps += 1
    return out


def _prefetched(trainer, replay, n):
    out = []
    with muzero.BatchPrefetcher(trainer, replay, muzero.TRAINING_BATCHES,
                                muzero.UNROLL_STEPS, ACTIONS, n) as pf:
        for i in range(n):
            out.append(pf.next())
            trainer.training_steps += 1
    return out


def _assert_same(a, b):
    assert a.keys() == b.keys()
    for k in a:
        if k == "sample_refs":
            assert all(x is y for x, y in zip(a[k], b[k]))
        else:
            np.testing.assert_array_equal(np.asarray(a[k]), np.asarray(b[k]), k)


def test_prefetched_batches_are_bit_identical_to_serial(world):
    replay = _replay()
    trainer = world
    np.random.seed(11); torch.manual_seed(11)
    base = trainer.training_steps
    serial = _serial(trainer, replay, 6)
    trainer.training_steps = base
    np.random.seed(11); torch.manual_seed(11)
    pre = _prefetched(trainer, replay, 6)
    assert len(serial) == len(pre) == 6
    for s, p in zip(serial, pre):
        _assert_same(s, p)


def test_assemble_uses_explicit_training_steps_like_serial(world, monkeypatch):
    """A prefetched batch i must be built as if training_steps were
    base + i (age-shrunk horizon and per_beta both key on it). With the
    horizon decay ON the two counters yield different value targets, so
    this test can actually fail (audit F5)."""
    monkeypatch.setattr(muzero, "OFFPOLICY_DECAY_STEPS", 100)
    replay = _replay()
    trainer = world
    np.random.seed(5)
    trainer.training_steps = 500
    a = trainer.prepare_batch_paper(replay, 8, 3, ACTIONS)
    np.random.seed(5)
    trainer.training_steps = 999
    b = trainer.prepare_batch_paper(replay, 8, 3, ACTIONS)
    assert not np.array_equal(a["target_value"], b["target_value"]), \
        "fixture must make the counter matter"
    np.random.seed(5)
    trainer.training_steps = 500
    ref = trainer.prepare_batch_paper(replay, 8, 3, ACTIONS)
    np.random.seed(5)
    trainer.training_steps = 999  # wrong live counter ...
    got = trainer.finish_batch_paper(trainer.assemble_batch_paper(
        replay, 8, 3, ACTIONS, training_steps=500))  # ... explicit wins
    _assert_same(ref, got)


def test_worker_never_needs_the_target_model_created(world):
    """The CPU half must not create/touch target nets; the main-thread
    half creates them at the serial point."""
    replay = _replay()
    trainer = world
    for n in ("target_representation", "target_prediction"):
        if hasattr(trainer, n):
            delattr(trainer, n)
    trainer._target_synced_at = None
    a = trainer.assemble_batch_paper(replay, 8, 3, ACTIONS)
    assert getattr(trainer, "target_representation", None) is None, \
        "assemble must not build target nets (worker thread)"
    batch = trainer.finish_batch_paper(a)
    assert getattr(trainer, "target_representation", None) is not None
    assert batch["target_value"].shape == (8, 4)


def test_worker_exception_surfaces_on_next(world, monkeypatch):
    trainer = world
    replay = _replay()

    def boom(*a, **k):
        raise ValueError("synthetic worker failure")
    monkeypatch.setattr(trainer, "assemble_batch", boom)
    pf = muzero.BatchPrefetcher(trainer, replay, 8, 3, ACTIONS, 3)
    with pytest.raises(RuntimeError, match="prefetch worker failed"):
        pf.next()
    pf.close()


def test_early_close_joins_worker(world):
    trainer = world
    replay = _replay()
    pf = muzero.BatchPrefetcher(trainer, replay, 8, 3, ACTIONS, 50, depth=2)
    pf.next()  # consume one of fifty, then abandon the block
    pf.close()
    assert not pf.thread.is_alive()
    assert pf.consumed == 1


def test_exhausted_prefetcher_refuses_extra_batches(world):
    trainer = world
    replay = _replay()
    with muzero.BatchPrefetcher(trainer, replay, 8, 3, ACTIONS, 2) as pf:
        pf.next(); pf.next()
        with pytest.raises(RuntimeError, match="exhausted"):
            pf.next()


def test_train_batches_helper_falls_back_to_serial(world, monkeypatch):
    """TRAIN_PREFETCH off -> the helper yields the serial builder and
    produces the same batches."""
    trainer = world
    replay = _replay()
    run = object.__new__(ablation.Run)
    run.trainer, run.replay = trainer, replay
    monkeypatch.setattr(ablation, "ACTIONS", ACTIONS)
    np.random.seed(2); base = trainer.training_steps
    ref = _serial(trainer, replay, 3)
    for flag in (False, True):
        monkeypatch.setattr(muzero, "TRAIN_PREFETCH", flag)
        trainer.training_steps = base
        np.random.seed(2)
        got = []
        with ablation.train_batches(run, 3) as nb:
            for _ in range(3):
                got.append(nb()); trainer.training_steps += 1
        for r, g in zip(ref, got):
            _assert_same(r, g)


def test_plain_mode_prefetch_matches_serial(world, monkeypatch):
    monkeypatch.setattr(muzero, "PAPER_TARGETS", False)
    trainer = world
    replay = _replay()
    for traj in replay:
        for t in traj:
            t["target_value"] = 0.5
    np.random.seed(7); base = trainer.training_steps
    ref = _serial(trainer, replay, 3)
    trainer.training_steps = base
    np.random.seed(7)
    got = _prefetched(trainer, replay, 3)
    for r, g in zip(ref, got):
        _assert_same(r, g)


def test_bootstrap_cache_matches_uncached_and_invalidates(world):
    """The target-model bootstrap cache must reproduce the uncached values
    (to batch-shape numerics), hit on repeats within a sync window, and
    drop on a sync-stamp change and on invalidate_bootstrap_cache()."""
    trainer = world
    replay = _replay(n=3)
    trainer.training_steps = 500
    trainer._ensure_target_model()
    np.random.seed(21)
    a = trainer.assemble_batch_paper(replay, 8, 3, ACTIONS)
    assert a["pending"], "fixture must exercise the bootstrap path"
    trainer.invalidate_bootstrap_cache()
    first = trainer.finish_batch_paper(copy.deepcopy(a))
    assert trainer.boot_cache_misses > 0 and trainer.boot_cache_hits == 0
    again = trainer.finish_batch_paper(copy.deepcopy(a))
    assert trainer.boot_cache_hits == len(a["pending"]), "second pass must hit"
    np.testing.assert_allclose(first["target_value"], again["target_value"],
                               rtol=0, atol=1e-5)
    # A sync-stamp change drops the cache.
    trainer._target_synced_at = (trainer._target_synced_at or 0) + 1
    m0 = trainer.boot_cache_misses
    trainer.finish_batch_paper(copy.deepcopy(a))
    assert trainer.boot_cache_misses == m0 + len(a["pending"])
    # Explicit invalidation drops it too.
    trainer.invalidate_bootstrap_cache()
    m1 = trainer.boot_cache_misses
    trainer.finish_batch_paper(copy.deepcopy(a))
    assert trainer.boot_cache_misses == m1 + len(a["pending"])


def test_train_batches_invalidates_cache_at_block_start(world, monkeypatch):
    trainer = world
    replay = _replay()
    run = object.__new__(ablation.Run)
    run.trainer, run.replay = trainer, replay
    monkeypatch.setattr(ablation, "ACTIONS", ACTIONS)
    trainer._boot_cache = {("stale", 0): 1.0}
    trainer._boot_cache_stamp = 123
    with ablation.train_batches(run, 1) as nb:
        assert trainer._boot_cache is None, "block start must drop the cache"
        nb()


def test_worker_death_with_full_queue_raises_instead_of_hanging(world, monkeypatch):
    """Audit F1: if the worker dies while the queue is full (its sentinel
    put cannot land), the consumer must still get an error on next()."""
    import time
    trainer = world
    replay = _replay()
    calls = {"n": 0}
    real = trainer.assemble_batch

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] > 2:          # queue depth 2 is full at this point
            raise MemoryError("synthetic")
        return real(*a, **k)
    monkeypatch.setattr(trainer, "assemble_batch", flaky)
    pf = muzero.BatchPrefetcher(trainer, replay, 8, 3, ACTIONS, 10, depth=2)
    time.sleep(0.5)                 # let the worker fill the queue and die
    pf.next(); pf.next()            # the two queued batches
    t0 = time.time()
    with pytest.raises(RuntimeError, match="prefetch worker failed"):
        pf.next()                   # must not block forever
    assert time.time() - t0 < 10
    pf.close()
