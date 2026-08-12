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
from networks import (DynamicsNetwork, PredictionNetwork, RepresentationNetwork,
                      ValuePrefixNetwork)

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


def bench_reanalysis(traj_len=200, strides=(1, 2, 4, 8, 16)):
    """Cost of reanalysing one trajectory as a function of REANALYSE_STRIDE.

    Skipping timesteps means each search starts from a colder tree -- lazy
    expansion lets neighbouring searches share subtrees -- so the saving is not
    automatically linear in the stride. That is the thing to measure.
    """
    print(f"\n=== reanalysis stride ({traj_len}-step trajectory) ===")
    print(f"  {'stride':>6} {'searched':>9} {'wall':>8} {'per search':>11} "
          f"{'traj/min':>9} {'speedup':>8}")

    helper = MCTSHelper(action_space=ACTIONS, device=DEVICE,
                        network_ref=None, storage_ref=None)
    raw = fake_trajectory(traj_len)
    baseline = None
    rows = []

    async def one_pass(stride):
        evaluated = helper.eval_obs(raw)
        helper.create_tree_from_game(evaluated)
        chosen = helper.select_reanalysis_indices(len(evaluated), stride=stride)
        t0 = time.perf_counter()
        for idx in range(len(evaluated)):
            if idx in chosen:
                await helper.run_full_mcts(idx, num_simulations=TOTAL_MCTS_SIMS,
                                           sim_batches=MCTS_BATCHES, c_puct=C_PUCT)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - t0, len(chosen)

    asyncio.run(one_pass(8))  # warm up
    for stride in strides:
        samples = [asyncio.run(one_pass(stride)) for _ in range(3)]
        wall = statistics.median(s[0] for s in samples)
        searched = samples[0][1]
        per_search = wall / max(searched, 1)
        per_min = 60.0 / wall
        baseline = baseline or per_min
        rows.append((stride, searched, wall, per_search, per_min, per_min / baseline))
        print(f"  {stride:>6} {searched:>9} {wall:>7.2f}s {per_search*1000:>10.1f}ms "
              f"{per_min:>9.1f} {per_min/baseline:>7.2f}x")

    print()
    cold = rows[-1][3] / rows[0][3]
    print(f"  per-search cost at stride {rows[-1][0]} is {cold:.2f}x the stride-1 cost")
    print("  (>1 means skipping indices really does cost tree reuse)")
    return rows


def collect_trajectories(episodes=20, cap=400, repeat=1):
    """Real MsPacman transitions in replay-buffer shape.

     holds each sampled action for that many frames. Uniform per-frame
    random makes Pacman reverse constantly and barely leave its start tile, so
    it eats almost nothing -- which leaves the reward heads with next to no
    signal to fit. Holding an action covers ground and raises reward density by
    roughly an order of magnitude, without needing a trained policy.
    """
    import gymnasium as gym
    from EpisodicLifeEnv import EpisodicLifeEnv
    from muzero import initial_frame_stack, push_frame, REWARD_CLIP

    env = EpisodicLifeEnv(gym.make("MsPacman-v4"))
    trajectories = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        stack = initial_frame_stack(preprocess_frame(obs, IMAGE_RESHAPE))
        traj, prev_reward, done, steps = [], 0.0, False, 0
        held = 0
        action = int(np.random.randint(ACTIONS))
        while not done and steps < cap:
            stack = push_frame(stack, preprocess_frame(obs, IMAGE_RESHAPE))
            if held >= repeat:
                action, held = int(np.random.randint(ACTIONS)), 0
            held += 1
            obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            traj.append({"obs": stack, "action": action, "reward": prev_reward,
                         "target_policy": np.full(ACTIONS, 1.0 / ACTIONS),
                         "target_value": 0.0, "has_policy_target": True})
            prev_reward = min(float(reward), REWARD_CLIP)
            steps += 1
        if len(traj) > muzero.UNROLL_STEPS + 1:
            trajectories.append(traj)
    env.close()
    return trajectories


def bench_prefix(train_steps=400, eval_batches=24, batch_size=32, discount=None,
                 prefix_support=None, unroll=None, repeat=1, episodes=20):
    """Value-prefix prediction error as a function of unroll depth.

    The direct test of EfficientZero's claim. Both heads live on the same trunk
    and train on the same batches, so this compares the heads themselves rather
    than two separately-trained models:

      per-step : decode a reward at each unrolled state, then discount and sum
      prefix   : decode the whole discounted sum from the LSTM in one shot

    State aliasing predicts the per-step error grows faster with depth, because
    each step's prediction is made from a latent that has drifted further from
    the truth and the errors accumulate through the sum.
    """
    discount = muzero.DISCOUNT if discount is None else discount
    K = muzero.UNROLL_STEPS if unroll is None else unroll
    print(f"\n=== value prefix vs per-step reward (gamma={discount}) ===")
    print(f"  collecting trajectories...", flush=True)
    trajectories = collect_trajectories(episodes=episodes, repeat=repeat)
    steps_total = sum(len(t) for t in trajectories)
    reward_total = sum(tr["reward"] for t in trajectories for tr in t)
    print(f"  {len(trajectories)} trajectories, {steps_total} transitions, "
          f"{reward_total / max(steps_total, 1):.3f} reward/step (action repeat {repeat})")

    trainer = _local_trainer(prefix_support)
    split = max(len(trajectories) // 5, 1)
    holdout = trajectories[-split:]
    train_set = trajectories[:-split] or trajectories

    print(f"  training {train_steps} steps (both heads, shared trunk)...", flush=True)
    t0 = time.perf_counter()
    marks = max(train_steps // 5, 1)
    for step in range(train_steps):
        batch = asyncio.run(trainer.prepare_batch(train_set, batch_size,
                                                  K, ACTIONS))
        loss = asyncio.run(trainer.train_step(batch))
        if step % marks == 0 or step == train_steps - 1:
            # Decoded prefix at the deepest step, so we can see whether the head
            # has moved off the support midpoint (= an untrained uniform output).
            with torch.no_grad():
                trainer.value_prefix.eval()
                lat = trainer.representation(
                    torch.tensor(batch["obs"][:, 0], dtype=torch.float32, device=DEVICE) / 255.0)
                st = trainer.value_prefix.initial_state(lat.size(0), DEVICE)
                for i in range(K):
                    lat, _, _ = trainer.dynamics(
                        lat, torch.tensor(batch["actions"][:, i], dtype=torch.long, device=DEVICE))
                    pfx, _, st = trainer.value_prefix(lat, st)
                trainer.value_prefix.train()
            print(f"    step {step:>5}  loss {loss:>9.2f}  decoded prefix {pfx.mean().item():>9.2f}"
                  f"  (midpoint {trainer.value_prefix.support_size / 2:.0f} = untrained)")
    print(f"  trained in {time.perf_counter() - t0:.1f}s")

    # ---- evaluate on held-out trajectories --------------------------------
    per_step_err = np.zeros(K + 1)
    prefix_err = np.zeros(K + 1)
    scale = np.zeros(K + 1)
    n = 0

    for name, net in (("representation", trainer.representation), ("dynamics", trainer.dynamics),
                      ("prediction", trainer.prediction), ("value_prefix", trainer.value_prefix)):
        net.eval()

    with torch.no_grad():
        for _ in range(eval_batches):
            batch = asyncio.run(trainer.prepare_batch(holdout, batch_size, K, ACTIONS))
            obs = torch.tensor(batch["obs"], dtype=torch.float32, device=DEVICE) / 255.0
            actions = torch.tensor(batch["actions"], dtype=torch.long, device=DEVICE)
            rewards = torch.tensor(batch["rewards"], dtype=torch.float32, device=DEVICE)
            truth = muzero.discounted_prefix_targets(rewards, discount=discount)

            latent = trainer.representation(obs[:, 0])
            state = trainer.value_prefix.initial_state(obs.size(0), DEVICE)
            running = torch.zeros(obs.size(0), 1, device=DEVICE)

            for i in range(1, K + 1):
                latent, reward, _ = trainer.dynamics(latent, actions[:, i - 1])
                running = running + (discount ** (i - 1)) * reward      # per-step, summed
                prefix, _, state = trainer.value_prefix(latent, state)  # LSTM, one shot

                target = truth[:, i:i + 1]
                per_step_err[i] += (running - target).abs().mean().item()
                prefix_err[i] += (prefix - target).abs().mean().item()
                # MAE of the trivial always-predict-zero predictor. A head that
                # cannot beat this has learned nothing, however good its ratio
                # against the other head happens to look.
                scale[i] += target.abs().mean().item()
            n += 1

    per_step_err /= n
    prefix_err /= n
    scale /= n

    print(f"\n  {'':>5} {'--- MAE ---':>30}   {'--- vs predict-0 ---':>22}")
    print(f"  {'depth':>5} {'predict-0':>10} {'per-step':>9} {'LSTM':>9}   "
          f"{'per-step':>10} {'LSTM':>10}")
    for i in range(1, K + 1):
        print(f"  {i:>5} {scale[i]:>10.2f} {per_step_err[i]:>9.2f} {prefix_err[i]:>9.2f}   "
              f"{per_step_err[i] / max(scale[i], 1e-9):>10.2f} "
              f"{prefix_err[i] / max(scale[i], 1e-9):>10.2f}")

    useful = 0.95   # has to beat the trivial predictor by a real margin
    beats_ps = int((per_step_err[1:K + 1] < scale[1:K + 1] * useful).sum())
    beats_lstm = int((prefix_err[1:K + 1] < scale[1:K + 1] * useful).sum())
    print(f"\n  depths beating predict-zero: per-step {beats_ps}/{K}, LSTM {beats_lstm}/{K}")
    if beats_ps == 0 and beats_lstm == 0:
        print("  -> NEITHER head learned anything useful. Any ratio between the")
        print("     two is noise, and a head that collapsed to predicting zero")
        print("     will look like it 'won'.")

    growth_ps = per_step_err[K] / max(per_step_err[1], 1e-9)
    growth_lstm = prefix_err[K] / max(prefix_err[1], 1e-9)
    print(f"  error growth from depth 1 to {K}: per-step {growth_ps:.2f}x, LSTM {growth_lstm:.2f}x")
    print("  (state aliasing predicts the per-step figure grows faster)")
    return {"per_step": per_step_err, "prefix": prefix_err, "scale": scale}


def _local_trainer(prefix_support=None):
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
    trainer.value_prefix = ValuePrefixNetwork(latent_channels=LATENT_CHANNELS,
                                              hidden_size=muzero.VALUE_PREFIX_HIDDEN,
                                              support_size=(muzero.VALUE_PREFIX_SUPPORT
                                                            if prefix_support is None else prefix_support),
                                              num_bins=51).to(DEVICE)
    import torch.optim as optim
    trainer.prefix_optimizer = muzero.make_optimizer(trainer.value_prefix.parameters())
    trainer.rep_optimizer = muzero.make_optimizer(trainer.representation.parameters())
    trainer.dyn_optimizer = muzero.make_optimizer(trainer.dynamics.parameters())
    trainer.pred_optimizer = muzero.make_optimizer(trainer.prediction.parameters())
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
def bench_actors(counts=(1, 2, 3), trajectories=6, traj_len=60, stride=None):
    import ray

    label = "every step" if stride == 1 else f"stride {stride or muzero.REANALYSE_STRIDE}"
    print(f"\n=== MCTS actor scaling ({trajectories} trajectories x {traj_len} steps, {label}) ===")
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
            futures = [actors[i % n].reanalyse.remote(w, stride) for i, w in enumerate(work)]
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

        async def reanalyse(self, trajectory, stride=None):
            t0 = time.perf_counter()
            evaluated = self.eval_obs(trajectory)
            self.create_tree_from_game(evaluated)
            chosen = self.select_reanalysis_indices(len(evaluated), stride=stride)
            for idx in range(len(evaluated)):
                if idx in chosen:
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
    ap.add_argument("suite", choices=["components", "actors", "startup", "reanalysis", "prefix", "all"])
    ap.add_argument("--trajectories", type=int, default=6)
    ap.add_argument("--traj-len", type=int, default=60)
    ap.add_argument("--actors", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--train-steps", type=int, default=400,
                    help="training steps for the prefix suite")
    ap.add_argument("--prefix-support", type=float, default=None,
                    help="override the value-prefix support ceiling (default: VALUE_PREFIX_SUPPORT)")
    ap.add_argument("--episodes", type=int, default=20,
                    help="episodes to collect for the prefix suite")
    ap.add_argument("--repeat", type=int, default=1,
                    help="hold each sampled action this many frames when collecting")
    ap.add_argument("--unroll", type=int, default=None,
                    help="unroll depth for the prefix suite (default: UNROLL_STEPS)")
    ap.add_argument("--gamma", type=float, default=None,
                    help="discount for the prefix suite (default: muzero.DISCOUNT)")
    ap.add_argument("--stride", type=int, default=None,
                    help="reanalysis stride for the actor suite (default: REANALYSE_STRIDE)")
    args = ap.parse_args()

    print(f"device: {DEVICE}"
          + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))
    if args.suite in ("components", "all"):
        bench_components()
    if args.suite in ("reanalysis", "all"):
        bench_reanalysis()
    if args.suite in ("prefix", "all"):
        bench_prefix(train_steps=args.train_steps, discount=args.gamma,
                     prefix_support=args.prefix_support, unroll=args.unroll,
                     repeat=args.repeat, episodes=args.episodes)
    if args.suite in ("startup", "all"):
        bench_startup()
    if args.suite in ("actors", "all"):
        bench_actors(args.actors, args.trajectories, args.traj_len, args.stride)


if __name__ == "__main__":
    main()
