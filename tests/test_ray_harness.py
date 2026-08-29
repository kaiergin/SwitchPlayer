"""Concurrent (--ray) harness mode: actor override transport, the episode
shell's attribute contract, and process isolation. Runs a real (tiny) Ray
cluster: actors live in genuine worker processes, so the override test
exercises the actual cross-process transport the design exists for."""
import ast
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
             logging_level="ERROR", num_cpus=2,
             runtime_env={"env_vars": {
                 "PYTHONPATH": f"{REPO};{REPO / 'experiments'}"}})
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
        6, seed=0, run_length_steps=1000,
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
        6, seed=1, run_length_steps=1000,
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
