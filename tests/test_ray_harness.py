"""Concurrent (--ray) harness mode: actor override transport, the episode
shell's attribute contract, and process isolation. Runs a real (tiny) Ray
cluster: actors live in genuine worker processes, so the override test
exercises the actual cross-process transport the design exists for."""
import ast
import os
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "experiments"))

import muzero

ray = pytest.importorskip("ray")


@pytest.fixture(scope="module")
def local_ray():
    ray.init(include_dashboard=False, ignore_reinit_error=True,
             logging_level="ERROR", num_cpus=4,  # pool test: server + 2 workers
             runtime_env={"env_vars": {"PYTHONPATH": os.pathsep.join(
                 (str(REPO), str(REPO / "experiments")))}})
    try:
        yield ray
    finally:
        ray.shutdown()


def test_actor_applies_overrides_in_its_own_process(local_ray):
    """The whole reason the actor takes an overrides dict: a Ray worker is
    a fresh interpreter that never saw the driver's arm_config setattrs."""
    import ray_actors
    actor_cls = local_ray.remote(ray_actors.SelfPlayActor)
    actor = actor_cls.remote(
        {"DISCOUNT": 0.5, "MCTS_BATCHES": 3, "USE_ATARI_WRAPPERS": False,
         "USE_SEARCH_ACTING": False},
        6, seed=0,
        schedule={"TEMPERATURE_TOTAL_STEPS": 1000,
                  "OFFPOLICY_DECAY_STEPS": 300},
        env_name="Pong-v4", frameskip=30, episode_cap=8)
    assert local_ray.get(actor.get_constant.remote("DISCOUNT")) == 0.5
    assert local_ray.get(actor.get_constant.remote("MCTS_BATCHES")) == 3
    assert local_ray.get(actor.get_constant.remote("TEMPERATURE_TOTAL_STEPS")) == 1000


def test_actor_plays_a_real_episode(local_ray):
    """End to end through Run.play_episode on the actor's shell: raw-policy
    acting (no search) at high skip for speed."""
    import ray_actors
    actor_cls = local_ray.remote(ray_actors.SelfPlayActor)
    actor = actor_cls.remote(
        {"USE_ATARI_WRAPPERS": False, "USE_SEARCH_ACTING": False,
         "USE_GUMBEL_SEARCH": False},
        6, seed=1,
        schedule={"TEMPERATURE_TOTAL_STEPS": 1000,
                  "OFFPOLICY_DECAY_STEPS": 300},
        env_name="Pong-v4", frameskip=30, episode_cap=12)
    traj, terminated, steps = local_ray.get(actor.play.remote(1234))
    # The cap bounds stepping; a truncated episode's trajectory carries the
    # harness's tail record, so len(traj) may exceed steps by one.
    assert 1 <= steps <= 13
    assert len(traj) in (steps, steps + 1)
    assert isinstance(terminated, bool)


def test_play_episode_attribute_surface_matches_the_shell():
    """The actor's Run shell provides exactly env/searcher/env_steps; if
    play_episode grows a new self.* dependency, this fails BEFORE the
    actor crashes mysteriously in a worker process."""
    src = (pathlib.Path(__file__).resolve().parent.parent
           / "experiments" / "ablation.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    run_cls = next(n for n in tree.body
                   if isinstance(n, ast.ClassDef) and n.name == "Run")
    play = next(n for n in run_cls.body
                if isinstance(n, ast.FunctionDef) and n.name == "play_episode")
    attrs = {node.attr for node in ast.walk(play)
             if isinstance(node, ast.Attribute)
             and isinstance(node.value, ast.Name) and node.value.id == "self"}
    assert attrs <= {"env", "searcher", "env_steps"}, (
        f"play_episode now touches {sorted(attrs)}; update the actor shell "
        f"in experiments/ray_actors.py to provide the new attributes")


def test_refresh_actor_through_a_real_cluster(local_ray):
    """Phase 3a transport: RefreshActor in a genuine worker process --
    module import, ObjectRef argument deref (the driver ships records
    via ray.put, once per trajectory), set_weights, and a windowed
    communal re-search returning well-formed results."""
    import numpy as np
    import torch
    import ray_actors
    overrides = {"USE_GUMBEL_SEARCH": False, "USE_COMMUNAL_REANALYSIS": True,
                 "COMMUNAL_WINDOW_ROOTS": 2, "MCTS_BATCHES": 1,
                 "ACT_SEARCH_SIMS": 4, "MAX_DEPTH": 8,
                 "USE_VALUE_PREFIX_IN_TREE": True}
    cls = local_ray.remote(ray_actors.RefreshActor)
    actor = cls.remote(overrides, 6, seed=2,
                       schedule={"TEMPERATURE_TOTAL_STEPS": 1000,
                                 "OFFPOLICY_DECAY_STEPS": None})
    donor = muzero.MCTSHelper(action_space=6, device=torch.device("cpu"),
                              network_ref=None, storage_ref=None)
    states = {n: getattr(donor.network_manager, n).state_dict()
              for n in ("representation", "dynamics", "prediction",
                        "value_prefix")}
    assert local_ray.get(actor.set_weights.remote(states, 5)) == 5
    rng = np.random.default_rng(4)
    records = [{"obs": rng.integers(0, 255, (12, *muzero.IMAGE_RESHAPE),
                                    dtype=np.uint8),
                "action": int(rng.integers(6)), "reward": 0.0,
                "truncated": False, "has_policy_target": True}
               for _ in range(4)]
    records[-1]["truncated"] = True  # truncated tail stays searchable
    ref = local_ray.put(records)     # the production shipping shape
    results = local_ray.get(actor.refresh.remote(ref, [0, 1, 2]))
    assert set(results) == {0, 1, 2}
    for pol, val in results.values():
        pol = np.asarray(pol)
        assert pol.shape == (6,) and pol.sum() == pytest.approx(1.0)
        assert np.isfinite(val)


def test_refresh_worker_via_server_through_a_real_cluster(local_ray):
    """3b.5 transport: a CPU refresh WORKER whose searches route to the
    shared server's TARGET weight set -- the unified-inference path the
    entropyrefresh arm will run under --ray --ray-workers."""
    import numpy as np
    import torch
    import ray_actors
    overrides = {"USE_GUMBEL_SEARCH": False, "USE_COMMUNAL_REANALYSIS": True,
                 "COMMUNAL_WINDOW_ROOTS": 2, "MCTS_BATCHES": 1,
                 "ACT_SEARCH_SIMS": 4, "MAX_DEPTH": 8,
                 "USE_VALUE_PREFIX_IN_TREE": True}
    schedule = {"TEMPERATURE_TOTAL_STEPS": 1000, "OFFPOLICY_DECAY_STEPS": None}
    srv_cls = local_ray.remote(max_concurrency=64)(ray_actors.InferenceServer)
    server = srv_cls.remote(overrides, 6, schedule, device="cpu")
    donor = muzero.MCTSHelper(action_space=6, device=torch.device("cpu"),
                              network_ref=None, storage_ref=None)
    states = {n: getattr(donor.network_manager, n).state_dict()
              for n in ("representation", "dynamics", "prediction",
                        "value_prefix")}
    local_ray.get(server.set_weights.remote(states, 5, which="target"))
    worker_cls = local_ray.remote(ray_actors.RefreshActor)
    worker = worker_cls.remote(overrides, 6, seed=3, schedule=schedule,
                               server=server)
    assert local_ray.get(worker.set_weights.remote({}, 5)) == 5
    rng = np.random.default_rng(6)
    records = [{"obs": rng.integers(0, 255, (12, *muzero.IMAGE_RESHAPE),
                                    dtype=np.uint8),
                "action": int(rng.integers(6)), "reward": 0.0,
                "truncated": False, "has_policy_target": True}
               for _ in range(4)]
    records[-1]["truncated"] = True
    results = local_ray.get(worker.refresh.remote(
        local_ray.put(records), [0, 1, 2]))
    assert set(results) == {0, 1, 2}
    for pol, val in results.values():
        pol = np.asarray(pol)
        assert pol.shape == (6,) and pol.sum() == pytest.approx(1.0)
        assert np.isfinite(val)


def test_worker_pool_through_a_real_cluster(local_ray):
    """Phase 3b end to end on genuine Ray transport: an ASYNC
    InferenceServer actor (its batching needs Ray's concurrent method
    scheduling, which no in-process test exercises) serving two CPU
    env/tree workers that play real search-acting episodes concurrently.
    Tiny sims/caps keep it CPU-cheap; correctness of the search itself is
    pinned by test_inference_server's exact-parity test."""
    import ray_actors
    overrides = {"USE_ATARI_WRAPPERS": False, "USE_SEARCH_ACTING": True,
                 "USE_GUMBEL_SEARCH": False, "PAPER_TARGETS": True,
                 "USE_VALUE_PREFIX_IN_TREE": True, "MCTS_BATCHES": 1,
                 "ACT_SEARCH_SIMS": 4, "MAX_DEPTH": 8}
    schedule = {"TEMPERATURE_TOTAL_STEPS": 1000,
                "OFFPOLICY_DECAY_STEPS": 300}
    srv_cls = local_ray.remote(max_concurrency=64)(
        ray_actors.InferenceServer)
    server = srv_cls.remote(overrides, 6, schedule, device="cpu")
    worker_cls = local_ray.remote(ray_actors.SelfPlayActor)
    workers = [worker_cls.remote(overrides, 6, seed=10 + w,
                                 schedule=schedule, env_name="Pong-v4",
                                 frameskip=30, episode_cap=6, server=server)
               for w in range(2)]
    # Weight sync: server once, workers mirror the counter only.
    donor = muzero.MCTSHelper(action_space=6,
                              device=__import__("torch").device("cpu"),
                              network_ref=None, storage_ref=None)
    states = {n: getattr(donor.network_manager, n).state_dict()
              for n in ("representation", "dynamics", "prediction",
                        "value_prefix")}
    local_ray.get(server.set_weights.remote(states, 7))
    assert local_ray.get([w.set_weights.remote({}, 7) for w in workers]) \
        == [7, 7]
    # Two episodes CONCURRENTLY through the one server.
    results = local_ray.get([w.play.remote(999 + i)
                             for i, w in enumerate(workers)])
    for traj, terminated, steps in results:
        assert 1 <= steps <= 7
        assert len(traj) in (steps, steps + 1)
        # Search-acting records are 5-field (paper mode).
        assert len(traj[0]) == 5
        assert isinstance(terminated, bool)
