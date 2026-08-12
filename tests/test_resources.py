"""The Ray resource budget and its preflight check."""
import pytest

import muzero
from muzero import (CPU_AGENT, CPU_MCTS, CPU_NETWORK, CPU_STORAGE, CPU_TRAINING,
                    GPU_AGENT, GPU_NETWORK, GPU_TRAINING, check_cluster_resources,
                    mcts_gpu_share, resource_plan)


@pytest.fixture
def cluster(monkeypatch):
    """Stub ray.cluster_resources so no cluster has to exist."""
    def _set(cpu, gpu=None):
        resources = {"CPU": cpu}
        if gpu is not None:
            resources["GPU"] = gpu
        monkeypatch.setattr(muzero.ray, "cluster_resources", lambda: resources)
    return _set


# --------------------------------------------------------------------------
# The budget cannot oversubscribe the GPU by construction
# --------------------------------------------------------------------------
@pytest.mark.parametrize("actors", range(1, 9))
def test_gpu_total_never_exceeds_one_device(actors):
    """Regression: the fractions were fixed at 0.05/0.5/0.4/0.05, summing to
    exactly 1.0, so a second MCTS actor needed 1.5 GPUs and Ray queued it
    forever without raising. Deriving the MCTS share from the actor count makes
    that impossible."""
    _, gpu = resource_plan(actors, use_cuda=True)
    assert gpu <= 1.0 + 1e-9, f"{actors} actors request {gpu} GPU"


@pytest.mark.parametrize("actors", range(1, 9))
def test_mcts_actors_split_the_same_remainder(actors):
    remainder = 1.0 - GPU_NETWORK - GPU_TRAINING - GPU_AGENT
    assert mcts_gpu_share(actors) * actors == pytest.approx(remainder, abs=1e-3)


def test_more_actors_means_a_smaller_share_each():
    shares = [mcts_gpu_share(n) for n in range(1, 6)]
    assert all(a > b for a, b in zip(shares, shares[1:]))


def test_cpu_scales_linearly_with_actor_count():
    base = CPU_STORAGE + CPU_NETWORK + CPU_TRAINING + CPU_AGENT
    for n in range(1, 5):
        assert resource_plan(n, use_cuda=True)[0] == base + CPU_MCTS * n


# --------------------------------------------------------------------------
# Preflight check
# --------------------------------------------------------------------------
def test_passes_on_the_reference_machine(cluster):
    cluster(cpu=16, gpu=1)
    check_cluster_resources(2, use_cuda=True)  # must not raise


@pytest.mark.parametrize("actors", [1, 2, 3])
def test_up_to_three_actors_fit_sixteen_cores(cluster, actors):
    cluster(cpu=16, gpu=1)
    check_cluster_resources(actors, use_cuda=True)


def test_cpu_oversubscription_raises_rather_than_hanging(cluster):
    """Ray queues unschedulable actors silently; the whole point of this check
    is to convert that into an error at startup."""
    cluster(cpu=4, gpu=1)
    with pytest.raises(RuntimeError, match="CPU: requesting"):
        check_cluster_resources(2, use_cuda=True)


def test_too_many_actors_for_the_cores_raises(cluster):
    cluster(cpu=16, gpu=1)
    with pytest.raises(RuntimeError, match="CPU"):
        check_cluster_resources(6, use_cuda=True)


def test_missing_gpu_raises(cluster):
    cluster(cpu=16, gpu=0)
    with pytest.raises(RuntimeError, match="GPU: requesting"):
        check_cluster_resources(2, use_cuda=True)


def test_error_names_the_constant_to_change(cluster):
    cluster(cpu=2, gpu=1)
    with pytest.raises(RuntimeError, match="NUM_MCTS_ACTORS"):
        check_cluster_resources(2, use_cuda=True)


# --------------------------------------------------------------------------
# CPU-only machines
# --------------------------------------------------------------------------
def test_cpu_only_requests_no_gpu(cluster):
    """Asking for a GPU that does not exist queues the actor forever, exactly
    like oversubscribing one."""
    assert resource_plan(2, use_cuda=False)[1] == 0
    cluster(cpu=16)
    check_cluster_resources(2, use_cuda=False)  # must not raise


def test_cpu_only_still_enforces_the_cpu_budget(cluster):
    cluster(cpu=2)
    with pytest.raises(RuntimeError, match="CPU"):
        check_cluster_resources(2, use_cuda=False)
