"""Replay/gameplay buffer versioning, exercised without a Ray cluster.

StorageHelper's methods are plain coroutines; the only thing Ray adds is the
`.remote()` call shape. LocalHandle supplies that so StorageManager can be
driven directly under asyncio.run.
"""
import asyncio
import types

import pytest

from muzero import StorageHelper, StorageManager


class LocalHandle:
    """Mimics a Ray actor handle over an in-process instance."""

    def __init__(self, target):
        self._target = target

    def __getattr__(self, name):
        method = getattr(self._target, name)
        return types.SimpleNamespace(remote=lambda *a, **k: method(*a, **k))


@pytest.fixture
def storage():
    helper = StorageHelper(max_replay=20, max_gameplay=20)
    return helper, LocalHandle(helper)


def sync(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------
# Replay versioning
# --------------------------------------------------------------------------
def test_manager_mirrors_the_helper_without_duplicates(storage):
    """Regression: update_if_needed called get_replay_buffer_updates() with no
    argument, so the helper looked up version=None, hit its ValueError branch
    and returned the *entire* buffer. extend() then appended a full copy on top
    of what was already there, duplicating every trajectory, and shipped the
    whole buffer over IPC on every training iteration.
    """
    helper, handle = storage

    async def scenario():
        manager = StorageManager(handle, update_replay=True)
        await manager.setup()
        for i in range(5):
            await helper.append_replay_buffer(f"traj{i}")
            await manager.update_if_needed()
        return list(manager.replay_buffer)

    mirrored = sync(scenario())
    assert mirrored == [f"traj{i}" for i in range(5)]
    assert len(mirrored) == len(set(mirrored)), "buffer contains duplicates"


def test_update_is_a_noop_when_already_current(storage):
    helper, handle = storage

    async def scenario():
        manager = StorageManager(handle, update_replay=True)
        await manager.setup()
        await helper.append_replay_buffer("a")
        await manager.update_if_needed()
        first = list(manager.replay_buffer)
        for _ in range(3):
            await manager.update_if_needed()
        return first, list(manager.replay_buffer)

    first, after = sync(scenario())
    assert first == after == ["a"]


def test_only_new_entries_are_transferred(storage):
    helper, handle = storage

    async def scenario():
        manager = StorageManager(handle, update_replay=True)
        await manager.setup()
        await helper.append_replay_buffer("a")
        await manager.update_if_needed()
        await helper.append_replay_buffer("b")
        version, updates = await helper.get_replay_buffer_updates(manager.replay_version)
        return updates

    assert sync(scenario()) == ["b"], "should send the delta, not the whole buffer"


def test_unknown_version_falls_back_to_the_whole_buffer(storage):
    """If a consumer lags further behind than the deque retains, its version has
    been evicted and a full resend is the only correct answer."""
    helper, handle = storage

    async def scenario():
        await helper.append_replay_buffer("a")
        await helper.append_replay_buffer("b")
        return await helper.get_replay_buffer_updates("not-a-real-version")

    _, updates = sync(scenario())
    assert updates == ["a", "b"]


def test_version_changes_on_append(storage):
    helper, _ = storage

    async def scenario():
        before = helper.replay_buffer_version
        await helper.append_replay_buffer("a")
        return before, helper.replay_buffer_version, await helper.is_replay_up_to_date(before)

    before, after, still_current = sync(scenario())
    assert before != after
    assert not still_current


def test_replay_buffer_is_bounded(storage):
    helper, _ = storage

    async def scenario():
        for i in range(30):
            await helper.append_replay_buffer(i)
        return list(helper.replay_buffer)

    kept = sync(scenario())
    assert len(kept) == 20
    assert kept[-1] == 29, "keeps the most recent"


def test_manager_with_updates_disabled_stays_empty(storage):
    """MCTS and the agent do not read replay data; mirroring it would copy
    gigabytes of frame stacks for nothing."""
    helper, handle = storage

    async def scenario():
        manager = StorageManager(handle, update_replay=False)
        await manager.setup()
        await helper.append_replay_buffer("a")
        await manager.update_if_needed()
        return list(manager.replay_buffer)

    assert sync(scenario()) == []


# --------------------------------------------------------------------------
# Gameplay buffer
# --------------------------------------------------------------------------
def test_taking_a_trajectory_removes_it(storage):
    """This is what makes running several MCTS actors safe -- two of them must
    never reanalyse the same trajectory."""
    helper, _ = storage

    async def scenario():
        for i in range(3):
            await helper.append_gameplay_buffer(f"game{i}")
        taken = [await helper.take_random_gameplay() for _ in range(3)]
        return taken, list(helper.gameplay_buffer)

    taken, left = sync(scenario())
    assert sorted(taken) == ["game0", "game1", "game2"]
    assert len(set(taken)) == 3, "same trajectory handed out twice"
    assert left == []


def test_empty_gameplay_buffer_returns_none(storage):
    helper, _ = storage
    assert sync(helper.take_random_gameplay()) is None
