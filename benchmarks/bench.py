"""Measure the parts of the pipeline before tuning any of them.

    python benchmarks/bench.py components      # per-stage cost, no Ray
    python benchmarks/bench.py actors          # MCTS actor scaling, with Ray
    python benchmarks/bench.py startup         # ray.init + fleet setup
    python benchmarks/bench.py all

Everything reports wall-clock on the machine it runs on. The point is to find
out whether this hardware is actually limited by the things that look slow on
inspection, rather than guessing.
"""
import argparse
import asyncio
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

import muzero
from muzero import (C_PUCT, IMAGE_RESHAPE, LATENT_CHANNELS, MCTS_BATCHES,
                    MCTSHelper, NetworkManager, TOTAL_MCTS_SIMS, TRAINING_BATCHES,
                    initial_frame_stack, preprocess_frame, push_frame)
from networks import DynamicsNetwork, PredictionNetwork, RepresentationNetwork

ACTIONS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def timeit(fn, repeat=20, warmup=3):
    """Median and p90 seconds per call, with CUDA synchronised."""
    for _ in range(warmup):
        fn()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return statistics.median(samples), samples[int(0.9 * (len(samples) - 1))]


def row(label, median, p90, extra=""):
    print(f"  {label:<38} {median * 1000:>9.2f} ms {p90 * 1000:>9.2f} ms  {extra}")


def gpu_memory_mb():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        return int(out.splitlines()[0])
    except Exception:
        return -1


def fake_obs(n=1):
    return np.random.randint(0, 255, (n, 12, *IMAGE_RESHAPE), dtype=np.uint8)


def fake_trajectory(length):
    """Raw gameplay tuples, the shape MCTS_process receives them in."""
    return [(fake_obs(1)[0], i % ACTIONS, float(i % 40) * 10) for i in range(length)]


# ---------------------------------------------------------------------------
# component benchmarks
# ---------------------------------------------------------------------------
def bench_components(traj_len=200):
    print(f"\n=== components ({DEVICE}, latent {LATENT_CHANNELS}ch) ===")
    print(f"  {'stage':<38} {'median':>12} {'p90':>12}")

    # -- observation pipeline -------------------------------------------------
    raw = np.random.randint(0, 255, (210, 160, 3), dtype=np.uint8)
    m, p = timeit(lambda: preprocess_frame(raw, IMAGE_RESHAPE), repeat=200)
    row("preprocess_frame", m, p, f"{1/m:,.0f} frames/s")

    frame = preprocess_frame(raw, IMAGE_RESHAPE)
    stack = initial_frame_stack(frame)
    m, p = timeit(lambda: push_frame(stack, frame), repeat=200)
    row("push_frame", m, p)

    # -- env ------------------------------------------------------------------
    import gymnasium as gym
    from EpisodicLifeEnv import EpisodicLifeEnv
    env = EpisodicLifeEnv(gym.make("MsPacman-v4"))
    env.reset(seed=0)

    def step_env():
        _, _, term, trunc, _ = env.step(env.action_space.sample())
        if term or trunc:
            env.reset()
    m, p = timeit(step_env, repeat=200)
    row("env.step (ALE)", m, p, f"{1/m:,.0f} steps/s")
    env.close()

    # -- inference ------------------------------------------------------------
    manager = NetworkManager(None, action_space=ACTIONS, device=DEVICE)
    single = fake_obs(1)[0]
    m_single, p = timeit(lambda: manager.initial_inference(single), repeat=50)
    row("initial_inference (1 frame)", m_single, p, f"{1/m_single:,.0f} acts/s")

    for batch in (16, 32, 64):
        obs = fake_obs(batch)
        m, p = timeit(lambda: manager.batch_inference(obs), repeat=20)
        row(f"batch_inference (batch {batch})", m, p, f"{batch/m:,.0f} frames/s")

    # -- expansion ------------------------------------------------------------
    helper = MCTSHelper(action_space=ACTIONS, device=DEVICE,
                        network_ref=None, storage_ref=None)
    h, w = IMAGE_RESHAPE[0] // 4, IMAGE_RESHAPE[1] // 4

    def expand(n):
        nodes = [muzero.MCTSNode(latent=torch.randn(1, LATENT_CHANNELS, h, w, device=DEVICE),
                                 prior=1.0, policy=np.full(ACTIONS, 1 / ACTIONS))
                 for _ in range(n)]
        helper.batch_expand_nodes(nodes)
    for n in (8, 40):
        m, p = timeit(lambda n=n: expand(n), repeat=20)
        row(f"batch_expand_nodes ({n} nodes = {n*ACTIONS} evals)", m, p)

    # -- one full search ------------------------------------------------------
    # Each index is searched at most once per tree, as MCTS_process does: a
    # repeat search reuses an already-built subtree and reads far too cheap.
    CHAIN = 24
    state = {"idx": CHAIN}

    def one_search():
        if state["idx"] >= CHAIN:
            helper.create_tree_from_game(helper.eval_obs(fake_trajectory(CHAIN)))
            state["idx"] = 0
        i = state["idx"]
        state["idx"] += 1
        asyncio.run(helper.run_full_mcts(i, num_simulations=TOTAL_MCTS_SIMS,
                                         sim_batches=MCTS_BATCHES, c_puct=C_PUCT))
    m_search, p_search = timeit(one_search, repeat=16, warmup=2)
    row(f"run_full_mcts ({TOTAL_MCTS_SIMS} sims)", m_search, p_search,
        f"{1/m_search:,.1f} searches/s")

    # -- eval_obs over a whole trajectory -------------------------------------
    raw_traj = fake_trajectory(traj_len)
    helper2 = MCTSHelper(action_space=ACTIONS, device=DEVICE,
                         network_ref=None, storage_ref=None)
    m_eval, p = timeit(lambda: helper2.eval_obs(raw_traj), repeat=5, warmup=1)
    row(f"eval_obs ({traj_len}-step trajectory)", m_eval, p)

    # -- training -------------------------------------------------------------
    trainer = _local_trainer()
    batch = _fake_batch(TRAINING_BATCHES, unroll=5)
    m_train, p = timeit(lambda: asyncio.run(trainer.train_step(batch)), repeat=10, warmup=2)
    row(f"train_step (batch {TRAINING_BATCHES}, unroll 5)", m_train, p,
        f"{1/m_train:,.1f} steps/s")

    # -- what it all means ----------------------------------------------------
    print(f"\n  --- projected for a {traj_len}-step trajectory ---")
    reanalysis = m_eval + traj_len * m_search
    print(f"  eval_obs                     {m_eval:>8.2f} s")
    print(f"  {traj_len} searches @ {m_search*1000:.1f} ms      {traj_len * m_search:>8.2f} s")
    print(f"  total reanalysis             {reanalysis:>8.2f} s")
    selfplay = traj_len * (m_single + 1 / max(1 / m_single, 1))
    env_bound = traj_len * m_single
    print(f"  self-play (inference-bound)  {env_bound:>8.2f} s")
    print(f"\n  reanalysis is {reanalysis / max(env_bound, 1e-9):.1f}x slower than generating the data")
    print(f"  -> steady state needs ~{reanalysis / max(env_bound, 1e-9):.1f} MCTS actors per self-play actor")
    return {"search": m_search, "eval": m_eval, "selfplay_step": m_single, "train": m_train}


def _local_trainer():
    """A TrainingHelper without Ray or a storage handle."""
    from muzero import TrainingHelper
    trainer = object.__new__(TrainingHelper)
    trainer.device = DEVICE
    trainer.action_space = ACTIONS
    trainer.training_steps = 0
    trainer.representation = RepresentationNetwork(input_channels=12,
                                                   latent_channels=LATENT_CHANNELS).to(DEVICE)
    trainer.dynamics = DynamicsNetwork(latent_channels=LATENT_CHANNELS, action_space=ACTIONS,
                                       support_size=400, num_bins=51).to(DEVICE)
    trainer.prediction = PredictionNetwork(latent_channels=LATENT_CHANNELS, action_space=ACTIONS,
                                           support_size=2000, num_bins=51).to(DEVICE)
    import torch.optim as optim
    trainer.rep_optimizer = optim.SGD(trainer.representation.parameters(), lr=1e-4)
    trainer.dyn_optimizer = optim.SGD(trainer.dynamics.parameters(), lr=1e-4)
    trainer.pred_optimizer = optim.SGD(trainer.prediction.parameters(), lr=1e-4)
    return trainer


def _fake_batch(size, unroll):
    return {
        "obs": np.random.randint(0, 255, (size, unroll + 1, 12, *IMAGE_RESHAPE), dtype=np.uint8),
        "actions": np.random.randint(0, ACTIONS, (size, unroll)),
        "rewards": np.random.rand(size, unroll + 1) * 100,
        "target_policy": np.full((size, unroll + 1, ACTIONS), 1 / ACTIONS),
        "target_value": np.random.rand(size, unroll + 1) * 500,
    }


# ---------------------------------------------------------------------------
# actor scaling
# ---------------------------------------------------------------------------
def bench_actors(counts=(1, 2, 3), trajectories=6, traj_len=60):
    import ray

    print(f"\n=== MCTS actor scaling ({trajectories} trajectories x {traj_len} steps) ===")
    print(f"  {'actors':>6} {'wall':>9} {'traj/min':>10} {'speedup':>9} {'GPU MB':>9} {'GPU/actor':>10}")

    baseline = None
    results = []
    for n in counts:
        ray.init(include_dashboard=False, ignore_reinit_error=True,
                 logging_level="ERROR", log_to_driver=False)
        try:
            share = muzero.mcts_gpu_share(n)
            actors = [
                _BenchMCTS.options(num_cpus=muzero.CPU_MCTS,
                                   num_gpus=share if DEVICE.type == "cuda" else 0)
                .remote(action_space=ACTIONS, device=DEVICE,
                        network_ref=None, storage_ref=None, name=f"bench{i}")
                for i in range(n)
            ]
            ray.get([a.warm.remote() for a in actors])
            before = gpu_memory_mb()

            work = [fake_trajectory(traj_len) for _ in range(trajectories)]
            t0 = time.perf_counter()
            futures = [actors[i % n].reanalyse.remote(w) for i, w in enumerate(work)]
            ray.get(futures)
            wall = time.perf_counter() - t0
            peak = gpu_memory_mb()
        finally:
            ray.shutdown()
            time.sleep(2)

        per_min = trajectories / wall * 60
        baseline = baseline or per_min
        results.append((n, wall, per_min, per_min / baseline, peak, share))
        print(f"  {n:>6} {wall:>8.1f}s {per_min:>10.1f} {per_min/baseline:>8.2f}x "
              f"{peak:>9} {share:>10.2f}")

    print()
    best = max(results, key=lambda r: r[2])
    if best[0] == 1:
        print("  -> more actors did not help; the GPU is already saturated by one")
    else:
        eff = best[3] / best[0]
        print(f"  -> best at {best[0]} actors ({best[3]:.2f}x, {eff:.0%} scaling efficiency)")
        if eff < 0.6:
            print("     poor efficiency: actors are contending for the same GPU")
    return results


try:
    import ray as _ray

    @_ray.remote
    class _BenchMCTS(MCTSHelper):
        """Reuses the real reanalysis path without MCTS_process's storage loop
        or its STEPS_UNTIL_MCTS gate. Possible because the actor classes are
        plain Python with @ray.remote applied separately."""

        def warm(self):
            self.eval_obs(fake_trajectory(4))
            return True

        async def reanalyse(self, trajectory):
            t0 = time.perf_counter()
            evaluated = self.eval_obs(trajectory)
            self.create_tree_from_game(evaluated)
            for idx in range(len(evaluated)):
                await self.run_full_mcts(idx, num_simulations=TOTAL_MCTS_SIMS,
                                         sim_batches=MCTS_BATCHES, c_puct=C_PUCT)
            return time.perf_counter() - t0
except ImportError:
    _BenchMCTS = None


# ---------------------------------------------------------------------------
# startup
# ---------------------------------------------------------------------------
def bench_startup():
    import ray
    print("\n=== startup ===")
    t0 = time.perf_counter()
    ray.init(include_dashboard=False, ignore_reinit_error=True,
             logging_level="ERROR", log_to_driver=False)
    t_init = time.perf_counter() - t0
    try:
        t0 = time.perf_counter()
        network = muzero.NetworkActor.options(
            num_cpus=muzero.CPU_NETWORK,
            num_gpus=muzero.GPU_NETWORK if DEVICE.type == "cuda" else 0
        ).remote(action_space=ACTIONS, device=DEVICE)
        storage = muzero.StorageActor.options(num_cpus=muzero.CPU_STORAGE).remote()
        ray.get(network.get_state.remote())
        t_actors = time.perf_counter() - t0

        t0 = time.perf_counter()
        mcts = muzero.MCTSActor.options(
            num_cpus=muzero.CPU_MCTS,
            num_gpus=muzero.GPU_MCTS if DEVICE.type == "cuda" else 0
        ).remote(action_space=ACTIONS, device=DEVICE,
                 network_ref=network, storage_ref=storage)
        ray.get(mcts.setup.remote())
        t_setup = time.perf_counter() - t0
    finally:
        ray.shutdown()

    print(f"  ray.init                     {t_init:>8.2f} s")
    print(f"  network + storage actors     {t_actors:>8.2f} s")
    print(f"  MCTS actor construct+setup   {t_setup:>8.2f} s")
    print(f"  total to first search        {t_init + t_actors + t_setup:>8.2f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("suite", choices=["components", "actors", "startup", "all"])
    ap.add_argument("--trajectories", type=int, default=6)
    ap.add_argument("--traj-len", type=int, default=60)
    ap.add_argument("--actors", type=int, nargs="+", default=[1, 2, 3])
    args = ap.parse_args()

    print(f"device: {DEVICE}"
          + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))
    if args.suite in ("components", "all"):
        bench_components()
    if args.suite in ("startup", "all"):
        bench_startup()
    if args.suite in ("actors", "all"):
        bench_actors(args.actors, args.trajectories, args.traj_len)


if __name__ == "__main__":
    main()
