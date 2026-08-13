import torch
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import numpy as np
import cv2
import random
import ray
from collections import deque
import scalar_categorical
from memory import print_memory_diagnostics
from networks import (DynamicsNetwork, PredictionNetwork, RepresentationNetwork,
                      ValuePrefixNetwork)
import logging
import os
import uuid
import asyncio
import sys
import time
from EpisodicLifeEnv import EpisodicLifeEnv

# Atari envs ship in the ale-py plugin. gymnasium >= 1.0 requires explicit
# registration; earlier versions register them on import of ale_py.
try:
    import ale_py
    gym.register_envs(ale_py)
except (ImportError, AttributeError):
    pass

TRAINING_BATCHES  = 64
MCTS_BATCHES = 8
TOTAL_MCTS_SIMS = 32
MAX_DEPTH = 8
MAX_GAMEPLAY = 20
MAX_REPLAY = 20
IMAGE_RESHAPE = (84,84)
STEPS_UNTIL_MCTS = 500
# 128 channels at 6x6 = 4,608 values (18 KB/node), against 64 at 21x21 = 28,224
# (110 KB). Channels widen as resolution falls, so the residual blocks run at
# small spatial sizes: roughly 3x less convolution work for 6x less memory.
LATENT_CHANNELS = 128
# Spatial size the representation network emits: 84 -> 42 -> 21 -> 11 -> 6
# through four stride-2 convolutions. Kept here so consumers stop hardcoding a
# divisor of the observation size; test_networks asserts it matches reality.
LATENT_HW = 6
UNROLL_STEPS = 5
# gamma. Effective planning horizon is roughly 1/(1-DISCOUNT), so ~333 steps
# here. Lowering it shortens credit assignment and converges faster to a worse
# policy -- useful as a fast A/B arm, but note it also shrinks the horizon the
# value prefix exists to fix, so a low value will understate that change.
DISCOUNT = 0.997
REWARD_CLIP = 400
# Raw-unit ceilings. The value prefix is a discounted SUM over UNROLL_STEPS
# rewards, so it needs a ceiling that covers the sum rather than one reward.
VALUE_PREFIX_CEILING = REWARD_CLIP * UNROLL_STEPS
VALUE_CEILING = 5000
# Supports are half-widths in h-TRANSFORMED space, derived from those ceilings.
# h() compresses hard -- a raw 2000 becomes ~46 -- which is what lets a fixed
# bin count cover the worst case AND still resolve values near zero. With a
# linear support the bins were 40 wide while realistic prefixes were 0-5, so
# every target landed in bin 0 and the head could not train (measured in
# benchmarks/RESULTS.md).
REWARD_SUPPORT = scalar_categorical.support_for(REWARD_CLIP)
VALUE_PREFIX_SUPPORT = scalar_categorical.support_for(VALUE_PREFIX_CEILING)
VALUE_SUPPORT = scalar_categorical.support_for(VALUE_CEILING)

# Target bin width in transformed space. h() shrinks the range but not the bin
# COUNT, so a fixed 51 bins still left 1.8-wide bins over the prefix support --
# raw 0 to 7 in one bin, against realistic prefixes of 0-5. Bins have to scale
# with the transformed range. MuZero uses 601 bins over +/-300, i.e. width 1.0;
# 0.5 buys finer resolution near zero, which is where the rewards actually are.
H_BIN_WIDTH = 0.5

def bins_for(support):
    """Odd bin count giving ~H_BIN_WIDTH resolution across +/- support.

    Odd so the centre bin sits exactly on zero -- a value of zero should encode
    as one bin, not split across two.
    """
    n = int(2 * support / H_BIN_WIDTH) + 1
    return n if n % 2 == 1 else n + 1

REWARD_BINS = bins_for(REWARD_SUPPORT)
VALUE_PREFIX_BINS = bins_for(VALUE_PREFIX_SUPPORT)
VALUE_BINS = bins_for(VALUE_SUPPORT)
# h() on BOTH sides of the categorical codec: encode (train_step reads this at
# call time, so the ablation can override it per arm) and decode (the networks
# take it at construction via transform=...). The first "no-transform" ablation
# arm only changed support sizes -- h() was still applied, so it measured a
# mis-scaled h-encoding rather than the linear one; this flag exists so that
# arm can test what its label says. Change both sides together or the codec
# stops being an inverse pair.
USE_H_TRANSFORM = True
VALUE_PREFIX_HIDDEN = 512
# Whether the tree takes its edge rewards from the value-prefix LSTM or from
# the per-step reward head. Both heads are always trained; this selects which
# one the search actually consumes, so the two can be compared end to end.
#
# Default False on measured evidence. Over a 5-arm, 2-seed, 60-iteration
# ablation (~15k training steps each, benchmarks/RESULTS.md) the per-step head
# reached 0.24x the error of a predict-zero control while the prefix head sat at
# exactly 1.000x in every arm -- collapsed to predicting zero, i.e. feeding the
# search edge rewards of ~0. Feeding the tree a signal that demonstrably learns
# beats feeding it one that demonstrably does not.
USE_VALUE_PREFIX_IN_TREE = False

# ---------------------------------------------------------------------------
# Loss weighting and gradient scaling (MuZero appendix G)
#
# Every loss term used to be summed raw and every unroll step contributed at
# full weight, so a K-step unroll multiplied the gradient reaching the
# representation by roughly K, and the value head -- whose target has by far the
# widest range -- dominated the others.
# ---------------------------------------------------------------------------
VALUE_LOSS_WEIGHT = 0.25        # MuZero's Atari setting
POLICY_LOSS_WEIGHT = 1.0
REWARD_LOSS_WEIGHT = 1.0
PREFIX_LOSS_WEIGHT = 1.0
CONSISTENCY_LOSS_WEIGHT = 1.0   # the latent MSE; EfficientZero weights its SimSiam loss 2.0
# Gradient flowing back through each recurrent dynamics step is halved, so it
# does not compound across the unroll.
LATENT_GRADIENT_SCALE = 0.5

# ---------------------------------------------------------------------------
# Real-time pacing
#
# The console runs at 60 frames/s and on hardware the agent would decide once
# per frame. In ALE one env.step covers `frameskip` raw frames, so real time is
# GAME_FPS / frameskip decisions per second -- 17.1 for MsPacman-v4's stochastic
# (2,5) skip, NOT 60. Pacing at 60 would run the game 3.5x faster than real time
# and defeat the point of the limiter.
#
# Unpaced, ALE runs ~80x real time, so producer/consumer ratios measured against
# it answer a question the real system never asks (see benchmarks/RESULTS.md).
# ---------------------------------------------------------------------------
GAME_FPS = 60.0
SELF_PLAY_REAL_TIME = True      # False lets self-play run as fast as the hardware allows
# Acting samples the policy head at this temperature. With no search on the
# acting path, this IS the exploration mechanism, so it stays at 1.0 while the
# get_temperature schedule sharpens the TARGETS instead. Lower it deliberately
# and one knob at a time -- decaying acting entropy and target temperature
# together is how exploration collapses.
ACTING_TEMPERATURE = 1.0
# Reanalysis searches every REANALYSE_STRIDE-th timestep instead of all of
# them. Measured on an RTX 4070, searching every step makes reanalysis ~2.5x
# slower than self-play generates data even with two MCTS actors, so the
# gameplay buffer overflows and take_random_gameplay discards the excess at random
# (see benchmarks/RESULTS.md). Unsearched timesteps still contribute
# observation, action, reward and n-step value targets (nstep_value); only
# their policy loss is masked. Set to 1 to search every step.
REANALYSE_STRIDE = 4
# n-step horizon for the value targets of unsearched timesteps, as in
# MuZero-Reanalyze: TD_STEPS real rewards, then bootstrap the CURRENT
# network's value. MuZero's Atari setting (td_steps=10).
TD_STEPS = 10
# Q is normalised to [0,1] by MinMaxStats before the exploration term is added,
# so this is on the usual AlphaZero/MuZero scale. It is NOT comparable to the
# values used before normalisation existed (61.8, then 32).
C_PUCT = 1.25

# ---------------------------------------------------------------------------
# Ray resource budget
#
# Fractional num_gpus is bookkeeping, not enforcement: it decides how many
# actors may share a device and does not cap memory. The requests must sum to
# no more than the cluster total, because Ray does NOT raise when an actor
# cannot be scheduled -- it queues it indefinitely, so ray.get(setup.remote())
# just never returns. The previous fractions summed to exactly 1.0 GPU, which
# is why a second MCTS actor could only ever hang.
#
# Resources are applied with .options() at construction rather than baked into
# the @ray.remote decorators, so the MCTS share can be split across actors and
# so GPU requests can be dropped entirely on a machine without CUDA.
# ---------------------------------------------------------------------------
NUM_MCTS_ACTORS = 2

CPU_STORAGE  = 1   # async data shuffling only
CPU_NETWORK  = 1   # async; the work is on the GPU
CPU_TRAINING = 2   # prepare_batch does real numpy work
CPU_AGENT    = 2   # event loop + the thread asyncio.to_thread runs env.step on
CPU_MCTS     = 2   # selection is single-threaded asyncio; eval_obs is numpy

GPU_NETWORK  = 0.05
GPU_TRAINING = 0.30
GPU_AGENT    = 0.05

def mcts_gpu_share(num_mcts_actors=NUM_MCTS_ACTORS):
    """The MCTS actors do by far the most inference, so they split whatever the
    other three actors leave behind. Derived from the actor count rather than
    hardcoded, so changing NUM_MCTS_ACTORS cannot silently oversubscribe."""
    return round((1.0 - GPU_NETWORK - GPU_TRAINING - GPU_AGENT) / num_mcts_actors, 4)

GPU_MCTS = mcts_gpu_share()

if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

def print_tree(node, depth=0):
    if depth > 6:
        return
    indent = "    " * depth  # 4 spaces per depth level.
    print(f"{indent}Node(prior={node.prior}, reward={node.reward}, visits={node.visit_count}, value={node.expanded_value}, depth={depth})")
    for action, child in node.children.items():
        print(f"{indent}  Action {action}:")
        print_tree(child, depth + 1)

def add_dirichlet_noise(prior, epsilon=0.25, alpha=0.25):
    """
    Adds Dirichlet noise to a given prior policy.
    
    Args:
      prior (np.array): Array of shape (num_actions,) representing the original prior probabilities.
      epsilon (float): Fraction of noise to mix in.
      alpha (float): Concentration parameter for the Dirichlet distribution.
    
    Returns:
      np.array: The noisy prior policy.
    """
    noise = np.random.dirichlet([alpha] * len(prior))
    noisy_prior = (1 - epsilon) * prior + epsilon * noise
    return noisy_prior

class MinMaxStats:
    """Tracks the value range seen during one search so Q can be normalised.

    Without this the PUCT exploration term is inert. Q here is a raw discounted
    return, which for MsPacman runs into the hundreds or thousands, while
    c_puct * prior * sqrt(N) / (1 + n) is order 10 -- so selection is
    effectively greedy no matter how c_puct is tuned. Normalising Q into [0,1]
    puts the two terms on the same scale and makes c_puct mean what it does in
    the AlphaZero/MuZero papers.
    """

    def __init__(self):
        self.maximum = -float('inf')
        self.minimum = float('inf')

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value

# -----------------------------
# MCTS Node Definition
# -----------------------------
class MCTSNode:
    def __init__(self, latent, prior, reward=0.0, policy = None, expanded_value = None,
                 is_gameplay = False, prefix = 0.0, prefix_state = None):
        self.latent = latent            # torch.Tensor (latent state)
        self.prior = prior              # float value
        self.reward = reward            # immediate reward
        # Cumulative UNDISCOUNTED reward from the search root to this node, and
        # the LSTM state that produced it. Undiscounted so that an edge reward is
        # exactly the difference between a child's prefix and its parent's --
        # discounting stays in backup_simulation, where it always was, rather
        # than being baked into the model's own prediction and then divided back
        # out by depth. The paper writes the prefix discounted; this keeps the
        # arithmetic in one place.
        self.prefix = prefix
        self.prefix_state = prefix_state
        self.children = {}              # dict: action -> child node
        self.visit_count = 0
        self.value_sum = 0.0
        self.id = uuid.uuid4()
        self.done = False
        
        self.is_expanded = False
        self.policy = policy              # cached policy (numpy array)
        self.lock = asyncio.Lock()
        self.expanded_value = expanded_value
        self.virtual_loss = 0.0
        self.is_gameplay = is_gameplay

    def q_value(self):
        # Virtual loss counts as a visit contributing zero value -- the standard
        # "assume the in-flight simulation lost" trick that makes concurrent
        # simulations diversify. Applying it here rather than subtracting a raw
        # count from the selection score keeps the penalty on the same scale as
        # the normalised Q.
        visits = self.visit_count + self.virtual_loss
        return self.value_sum / visits if visits > 0 else 0.0

def apply_temperature(visit_counts, temperature):
    """
    Adjusts a vector of visit counts using a temperature parameter.
    
    Args:
        visit_counts (list or np.array): MCTS visit counts for each action.
        temperature (float): Temperature parameter. Higher values lead to softer distributions.
        
    Returns:
        np.array: Probability distribution over actions.
    """
    visit_counts = np.array(visit_counts, dtype=np.float32)
    
    # When temperature is close to 0, choose the action with the highest count deterministically.
    if temperature < 1e-3:
        policy = np.zeros_like(visit_counts)
        policy[np.argmax(visit_counts)] = 1.0
        return policy
    
    # Adjust counts by temperature.
    adjusted_counts = visit_counts ** (1.0 / temperature)
    policy = adjusted_counts / np.sum(adjusted_counts)
    return policy

def get_temperature(epoch, initial_temp=1.0, final_temp=0.1, decay_train_steps=100000):
    """
    Linearly decays temperature from initial_temp to final_temp over decay_train_steps.
    If train steps exceeds decay_train_steps, return final_temp.
    """
    if epoch >= decay_train_steps:
        return final_temp
    return initial_temp - (initial_temp - final_temp) * (epoch / decay_train_steps)

def scale_gradient(tensor, scale):
    """Identity on the forward pass; multiplies the gradient by `scale`.

    The standard MuZero trick: `x*s + x.detach()*(1-s)` is exactly x going
    forward, and has derivative s going back.
    """
    return tensor * scale + tensor.detach() * (1.0 - scale)


def real_time_step_period(env, game_fps=GAME_FPS):
    """Seconds per agent decision corresponding to real-time play.

    Reads the env's frameskip rather than assuming one: MsPacman-v4 samples it
    from (2, 5), so a decision covers 3.5 raw frames on average and real time is
    ~17 decisions/second, not 60.
    """
    skip = getattr(env.unwrapped, "_frameskip", None) or 1
    if isinstance(skip, (tuple, list)):
        skip = sum(skip) / len(skip)
    return float(skip) / game_fps


class RateLimiter:
    """Paces a loop to a target period without drift or catch-up bursts."""

    def __init__(self, period):
        self.period = period
        self.next_at = None
        self.behind = 0

    async def wait(self):
        if not self.period:
            return
        now = time.perf_counter()
        if self.next_at is None:
            self.next_at = now
        self.next_at += self.period
        delay = self.next_at - now
        if delay > 0:
            await asyncio.sleep(delay)
        else:
            # Running behind. Reset the clock rather than sprinting to catch up,
            # which would produce exactly the faster-than-real-time burst the
            # limiter exists to prevent.
            self.behind += 1
            self.next_at = now


def discounted_prefix_targets(rewards, discount=None):
    """Cumulative discounted reward sums along an unroll.

    The default resolves DISCOUNT at call time, not in the signature -- the
    same default-argument binding that silently pinned backup_simulation's
    Monte Carlo targets to 0.997 and would make any runtime DISCOUNT override
    (the ablation harness) measure nothing here.

    `rewards` is (B, unroll+1) where rewards[:, i] is the reward for *entering*
    step i. rewards[:, 0] therefore belongs to the state the unroll starts from
    and is part of no prefix. Column i of the result is

        prefix_i = sum_{j=1..i} discount^(j-1) * rewards[:, j]

    with column 0 zero: nothing is accumulated before the first step.
    """
    discount = DISCOUNT if discount is None else discount
    steps = rewards.shape[1] - 1
    weights = discount ** torch.arange(steps, device=rewards.device, dtype=rewards.dtype)
    prefix = torch.cumsum(rewards[:, 1:] * weights, dim=1)
    return torch.cat([torch.zeros_like(rewards[:, :1]), prefix], dim=1)


def masked_policy_loss(policy_logits, target_policy, mask):
    """Cross-entropy averaged over the positions that carry a real target.

    Reanalysis searches a strided subset of timesteps, so most positions hold a
    uniform placeholder. Averaging across all of them would train the policy
    toward uniform in proportion to the stride -- the sharper the search got,
    the harder the placeholders would pull against it.

    Returns 0 when nothing in the batch is masked in; clamping the denominator
    keeps that branch-free so it does not force a device sync.
    """
    per_position = -torch.sum(target_policy * F.log_softmax(policy_logits, dim=1), dim=1)
    return (per_position * mask).sum() / mask.sum().clamp(min=1.0)


def to_cpu(obj):
    """Recursively move every tensor in a nested structure to CPU.

    Optimiser state_dicts nest their tensors two levels down, under
    state[param_id][key], so a flat isinstance check over the top level finds
    none of them. That was harmless only because SGD without momentum keeps no
    state at all; with Adam it would ship exp_avg and exp_avg_sq across Ray
    still resident on the GPU.
    """
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_cpu(v) for v in obj)
    return obj


def get_state_dict(network):
    return to_cpu(network.state_dict())


LEARNING_RATE = 1e-3

def make_optimizer(params):
    """Adam. The repo moved Adam -> SGD without a recorded reason, and the one
    plausible cause -- get_state_dict shipping Adam's GPU-resident state through
    Ray -- is fixed above. SGD at 1e-4 was slow enough that the value prefix head
    could not converge inside a benchmark run (benchmarks/RESULTS.md).
    """
    return optim.Adam(params, lr=LEARNING_RATE)

LOG_DIR = "logs"

def configure_actor_logger(name, enable_logging=False, log_dir=LOG_DIR):
    """Build a logger for a Ray actor, writing to logs/<name>.log.

    Each actor runs in its own process with its own stdout, and the console
    handler here used to be opt-in, so with default settings a failure inside
    an actor left no trace anywhere at all. The file handler is unconditional
    so there is always somewhere to look afterwards.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    if enable_logging:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger

class StorageHelper:
    def __init__(self, max_replay=MAX_REPLAY, max_gameplay=MAX_GAMEPLAY):
        self.gameplay_buffer = deque(maxlen=max_gameplay)
        self.replay_buffer = deque(maxlen=max_replay)
        self.replay_buffer_version = uuid.uuid4()
        self.gameplay_lock = asyncio.Lock()
        self.replay_lock = asyncio.Lock()
        self.replay_versions = deque(maxlen=max_replay)
    
    async def append_replay_buffer(self, new_item):
        async with self.replay_lock:
            self.replay_buffer.append(new_item)
            self.replay_buffer_version = uuid.uuid4()
            self.replay_versions.append(self.replay_buffer_version)

    async def get_replay_buffer_updates(self, version=None):
        async with self.replay_lock:
            try:
                # Find the index of the passed-in version.
                index = self.replay_versions.index(version)
                # Return all replays that were added after the given version.
                missing_updates = list(self.replay_buffer)[index + 1:]
            except ValueError:
                # If the version is not found, return the whole replay buffer.
                missing_updates = list(self.replay_buffer)
            return self.replay_buffer_version, missing_updates

    async def is_replay_up_to_date(self, version):
        async with self.replay_lock:
            return self.replay_buffer_version == version

    async def append_gameplay_buffer(self, new_item):
        async with self.gameplay_lock:
            self.gameplay_buffer.append(new_item)
    
    async def take_random_gameplay(self):
        """Pop a random trajectory for reanalysis, or None if the buffer is empty.

        Was get_max_gameplay: priority ("max") selection was the intent but
        never implemented, so the name now matches the behaviour. Removal is
        what makes several MCTS actors safe -- no two may reanalyse the same
        trajectory.
        """
        async with self.gameplay_lock:
            if len(self.gameplay_buffer) == 0:
                return None
            idx, item = random.choice(list(enumerate(self.gameplay_buffer)))
            del self.gameplay_buffer[idx]
            return item
    

class StorageManager:
    def __init__(self, storage_helper, update_replay=True):
        self.storage_helper = storage_helper
        self.update_replay = update_replay

    async def setup(self):
        self.replay_buffer = deque(maxlen=MAX_REPLAY)
        self.replay_version, updates = await self.storage_helper.get_replay_buffer_updates.remote()
        self.replay_buffer.extend(updates)
    
    async def update_if_needed(self):
        if self.update_replay and not await self.storage_helper.is_replay_up_to_date.remote(self.replay_version):
            # The version has to be passed through. Without it the helper falls
            # into its ValueError branch and ships the entire replay buffer --
            # potentially gigabytes of frame stacks -- on every single call,
            # and duplicates entries locally until the deque fills.
            self.replay_version, updates = await self.storage_helper.get_replay_buffer_updates.remote(self.replay_version)
            self.replay_buffer.extend(updates)
    
    async def append_replay_buffer(self, new_item):
        self.replay_buffer.append(new_item)
        await self.storage_helper.append_replay_buffer.remote(new_item)

    async def append_gameplay_buffer(self, new_item):
        await self.storage_helper.append_gameplay_buffer.remote(new_item)

    async def take_random_gameplay(self):
        return await self.storage_helper.take_random_gameplay.remote()

#create versioning like storage
class NetworkHelper:
    def __init__(self, action_space, device, checkpoint_path = 'checkpoint.pth', enable_logging=False):
        self.device = device
        self.action_space = action_space
        self.representation = RepresentationNetwork(input_channels=12,
                                                    latent_channels=LATENT_CHANNELS).to(device)
        self.dynamics = DynamicsNetwork(latent_channels=LATENT_CHANNELS,
                                        action_space=action_space,
                                        support_size=REWARD_SUPPORT,
                                        num_bins=REWARD_BINS).to(device)
        self.prediction = PredictionNetwork(latent_channels=LATENT_CHANNELS,
                                            action_space=action_space,
                                            support_size=VALUE_SUPPORT,
                                            num_bins=VALUE_BINS).to(device)
        self.value_prefix = ValuePrefixNetwork(latent_channels=LATENT_CHANNELS,
                                               hidden_size=VALUE_PREFIX_HIDDEN,
                                               support_size=VALUE_PREFIX_SUPPORT,
                                               num_bins=VALUE_PREFIX_BINS).to(device)
        self.rep_optimizer = make_optimizer(self.representation.parameters())
        self.dyn_optimizer = make_optimizer(self.dynamics.parameters())
        self.pred_optimizer = make_optimizer(self.prediction.parameters())
        self.prefix_optimizer = make_optimizer(self.value_prefix.parameters())
        self.training_steps = 0
        self.logger = configure_actor_logger("NetworkHelper", enable_logging)
        self.load_checkpoint(checkpoint_path)
        # The helper stores and serves weights -- TrainingHelper trains its own
        # copies and pushes them back. Its only forward passes are inference
        # (the viewer path), so BatchNorm runs in eval mode here too.
        for net in (self.representation, self.dynamics, self.prediction, self.value_prefix):
            net.eval()
        self.version = uuid.uuid4()

    def save_checkpoint(self, training_steps, checkpoint_path = 'checkpoint.pth'):
        self.training_steps = training_steps
        checkpoint = {
            'representation_state_dict': self.representation.state_dict(),
            'dynamics_state_dict': self.dynamics.state_dict(),
            'prediction_state_dict': self.prediction.state_dict(),
            'rep_optimizer_state_dict': self.rep_optimizer.state_dict(),
            'dyn_optimizer_state_dict': self.dyn_optimizer.state_dict(),
            'pred_optimizer_state_dict': self.pred_optimizer.state_dict(),
            'value_prefix_state_dict': self.value_prefix.state_dict(),
            'prefix_optimizer_state_dict': self.prefix_optimizer.state_dict(),
            'training_steps': training_steps
        }
        torch.save(checkpoint, checkpoint_path)
        self.version = uuid.uuid4()
        self.logger.info("Checkpoint saved at %s", checkpoint_path)
        return self.version

    def load_checkpoint(self, checkpoint_path):
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.representation.load_state_dict(checkpoint['representation_state_dict'])
            self.dynamics.load_state_dict(checkpoint['dynamics_state_dict'])
            self.prediction.load_state_dict(checkpoint['prediction_state_dict'])
            self.rep_optimizer.load_state_dict(checkpoint['rep_optimizer_state_dict'])
            self.dyn_optimizer.load_state_dict(checkpoint['dyn_optimizer_state_dict'])
            self.pred_optimizer.load_state_dict(checkpoint['pred_optimizer_state_dict'])
            self.training_steps =  checkpoint['training_steps']
            self.logger.info("Checkpoint loaded successfully from %s", checkpoint_path)
        else:
            self.logger.info("No checkpoint found at %s", checkpoint_path)
    
    def initial_inference(self, obs):
        # no_grad must be applied here, not at the call site: Ray actors run in
        # their own process, so a `with torch.no_grad()` in the driver has no
        # effect on the graph built inside this method.
        with torch.no_grad():
            converted_obs = torch.tensor(obs, dtype=torch.float32, device=self.device) / 255.0
            converted_obs = converted_obs.unsqueeze(0)
            latent = self.representation(converted_obs)
            policy_logits, value, _ = self.prediction(latent)
            policy = F.softmax(policy_logits, dim=1).cpu().numpy()[0]
        return policy, value.item()

    def update_state(self, state, path='checkpoint.pth'):
        self.representation.load_state_dict(state["representation"])
        self.dynamics.load_state_dict(state["dynamics"])
        self.prediction.load_state_dict(state["prediction"])
        self.value_prefix.load_state_dict(state["value_prefix"])
        self.rep_optimizer.load_state_dict(state["rep_optimizer"])
        self.dyn_optimizer.load_state_dict(state["dyn_optimizer"])
        self.pred_optimizer.load_state_dict(state["pred_optimizer"])
        self.prefix_optimizer.load_state_dict(state["prefix_optimizer"])
        self.save_checkpoint(state["training_steps"], checkpoint_path=path)

    def get_state(self):
        """Weights and optimiser state as a dict.

        This was an 8-tuple unpacked positionally in three places. Adding a
        fourth network to that arrangement is how misalignment bugs are made.
        """
        return {
            "representation":   get_state_dict(self.representation),
            "dynamics":         get_state_dict(self.dynamics),
            "prediction":       get_state_dict(self.prediction),
            "value_prefix":     get_state_dict(self.value_prefix),
            "rep_optimizer":    get_state_dict(self.rep_optimizer),
            "dyn_optimizer":    get_state_dict(self.dyn_optimizer),
            "pred_optimizer":   get_state_dict(self.pred_optimizer),
            "prefix_optimizer": get_state_dict(self.prefix_optimizer),
            "training_steps":   self.training_steps,
            "version":          self.version,
        }

    def is_up_to_date(self, version_cmp):
        return self.version == version_cmp

class NetworkManager:
    def __init__(self, network_helper, action_space, device, load_dynamics = True):
        self.network_helper = network_helper
        self.load_dynamics = load_dynamics
        self.device = device
        self.representation = RepresentationNetwork(input_channels=12,
                                                    latent_channels=LATENT_CHANNELS).to(device)
        self.prediction = PredictionNetwork(latent_channels=LATENT_CHANNELS,
                                            action_space=action_space,
                                            support_size=VALUE_SUPPORT,
                                            num_bins=VALUE_BINS).to(device)
        if self.load_dynamics:
            self.dynamics = DynamicsNetwork(latent_channels=LATENT_CHANNELS,
                                            action_space=action_space,
                                            support_size=REWARD_SUPPORT,
                                            num_bins=REWARD_BINS).to(device)
            self.value_prefix = ValuePrefixNetwork(latent_channels=LATENT_CHANNELS,
                                                   hidden_size=VALUE_PREFIX_HIDDEN,
                                                   support_size=VALUE_PREFIX_SUPPORT,
                                                   num_bins=VALUE_PREFIX_BINS).to(device)
        # Inference-only consumer: BatchNorm must read the running statistics
        # training accumulated, not the statistics of whatever batch -- or
        # single acting frame -- happens to pass through (which also polluted
        # the running stats themselves). load_state_dict refreshes weights and
        # running stats but never touches this flag, so once here holds across
        # every weight pull.
        self.representation.eval()
        self.prediction.eval()
        if self.load_dynamics:
            self.dynamics.eval()
            self.value_prefix.eval()

    async def setup(self):
        await self._pull_state()

    async def _pull_state(self):
        state = await self.network_helper.get_state.remote()
        self.training_steps = state["training_steps"]
        self.version = state["version"]
        self.representation.load_state_dict(state["representation"])
        self.prediction.load_state_dict(state["prediction"])
        if self.load_dynamics:
            self.dynamics.load_state_dict(state["dynamics"])
            self.value_prefix.load_state_dict(state["value_prefix"])

    def initial_inference(self, obs):
        # no_grad must be applied here, not at the call site: Ray actors run in
        # their own process, so a `with torch.no_grad()` in the driver has no
        # effect on the graph built inside this method.
        with torch.no_grad():
            converted_obs = torch.tensor(obs, dtype=torch.float32, device=self.device) / 255.0
            converted_obs = converted_obs.unsqueeze(0)
            latent = self.representation(converted_obs)
            policy_logits, value, _ = self.prediction(latent)
            policy = F.softmax(policy_logits, dim=1).cpu().numpy()[0]
        return policy, value.item()
    
    def batch_inference(self, batch_obs):
        # Without no_grad the returned latents carry grad_fn, and because every
        # gameplay node stores one for the lifetime of the episode, the entire
        # forward graph (all ResNet activations for the batch) is pinned in GPU
        # memory. That was several GB per episode.
        with torch.no_grad():
            converted = torch.tensor(batch_obs, dtype=torch.float32, device=self.device) / 255.0
            batch_latents = self.representation(converted)
            batch_policy_logits, batch_value, _ = self.prediction(batch_latents)
            batch_policy = F.softmax(batch_policy_logits, dim=1).cpu().numpy()
            # (B, 1) -> (B,) so node.expanded_value is a scalar, matching the
            # floats used for nodes created in batch_expand_nodes.
            batch_value = batch_value.squeeze(-1).cpu().numpy()
        return batch_latents, batch_policy, batch_value
    
    async def update_if_needed(self):
        if not await self.network_helper.is_up_to_date.remote(self.version):
            await self._pull_state()

class MCTSHelper:
    def __init__(self, action_space, device, network_ref, storage_ref, enable_logging=False, name="MCTSHelper"):
        self.network_manager = NetworkManager(network_ref, action_space=action_space, device=device)
        self.storage_manager = StorageManager(storage_ref, update_replay=False)
        self.device = device
        self.action_space = action_space
        self.selection_results = asyncio.Queue()
        self.min_max_stats = MinMaxStats()  # replaced per search in run_full_mcts
        self.logger = configure_actor_logger(name, enable_logging)
        self.logger.info("Initialized MCTSHelper logger.")

    async def setup(self):
        await self.network_manager.setup()
        await self.storage_manager.setup()
    
    def eval_obs(self, game_trajectory, batch_size=32):
        new_traj = []
        for i in range(0, len(game_trajectory), batch_size):
            batch_transitions = game_trajectory[i:i+batch_size]
            # Pre-allocate a numpy array for observations.
            # We assume each observation has the same shape, so we use the first element.
            sample_obs = batch_transitions[0][0]
            batch_obs = np.empty((len(batch_transitions), *sample_obs.shape), dtype=sample_obs.dtype)
            for j, transition in enumerate(batch_transitions):
                batch_obs[j] = transition[0]
            # Now batch_obs is a single numpy array instead of a list of arrays.
            latents, policies, values = self.network_manager.batch_inference(batch_obs)
            for j, transition in enumerate(batch_transitions):
                new_traj.append([*transition, latents[j:j+1], policies[j], values[j]])
        return new_traj

    async def MCTS_process(self):
        try:
            while True:
                await self.storage_manager.update_if_needed()
                await self.network_manager.update_if_needed()
                game_trajectory_tuple = await self.storage_manager.take_random_gameplay()
                if game_trajectory_tuple == None:
                    self.logger.debug("MCTS_process sleeping")
                    await asyncio.sleep(1)
                    continue
                else:
                    game_trajectory, _ = game_trajectory_tuple
                if not game_trajectory:
                    self.logger.warning("Skipping empty game trajectory")
                    continue
                self.logger.debug("Running MCTS_process")
                trajectory = []
                game_trajectory = self.eval_obs(game_trajectory=game_trajectory)
                self.create_tree_from_game(game_trajectory=game_trajectory)
                if self.network_manager.training_steps < STEPS_UNTIL_MCTS:
                    self.logger.info("%s steps until MCTS", STEPS_UNTIL_MCTS - self.network_manager.training_steps)
                    for idx, (obs_sample, action_taken, reward, _, _, _) in enumerate(game_trajectory):
                        transition = {
                            "obs": obs_sample,
                            "action": action_taken,
                            "reward": reward,
                            "target_policy": np.ones(self.action_space) / self.action_space,
                            # Pure Monte Carlo during warmup: the network is
                            # untrained, so an n-step target would bootstrap
                            # noise.
                            "target_value": self.get_value(idx),
                            # The uniform policy is a placeholder, not a
                            # target. Training toward it at full weight spent
                            # the whole warmup confidently pulling the logits
                            # flat, which the first real search targets then
                            # had to fight back out of. An untrained head is
                            # already high-entropy, so masking loses nothing.
                            "has_policy_target": False,
                        }
                        trajectory.append(transition)
                else:
                    searched = self.select_reanalysis_indices(len(game_trajectory))
                    temperature = get_temperature(self.network_manager.training_steps)
                    self.logger.info("MCTS on %s of %s timesteps (stride %s)",
                                     len(searched), len(game_trajectory), REANALYSE_STRIDE)
                    for idx, (obs_sample, action_taken, reward, _, _, _) in enumerate(game_trajectory):
                        if idx in searched:
                            self.logger.debug("Running MCTS index %s out of %s", idx, len(game_trajectory))
                            improved_policy, value_estimate, has_policy_target = await self.run_full_mcts(idx,
                                                                                sim_batches=MCTS_BATCHES,
                                                                                num_simulations=TOTAL_MCTS_SIMS,
                                                                                c_puct=C_PUCT,
                                                                                discount=DISCOUNT,
                                                                                temperature=temperature)
                        else:
                            # Not searched. The value target is an n-step
                            # return bootstrapped by the CURRENT network
                            # (MuZero-Reanalyze), not the trajectory's Monte
                            # Carlo return, which is the behaviour policy's
                            # result and goes stale as the policy moves. There
                            # is no improved policy here, so the policy loss is
                            # masked out rather than trained toward a
                            # placeholder that would drag it uniform.
                            improved_policy = np.ones(self.action_space) / self.action_space
                            value_estimate = self.nstep_value(idx)
                            has_policy_target = False
                        trajectory.append({
                            "obs": obs_sample,
                            "action": action_taken,
                            "reward": reward,
                            "target_policy": improved_policy,
                            "target_value": value_estimate,
                            "has_policy_target": has_policy_target,
                        })
                
                await self.storage_manager.append_replay_buffer(trajectory)
        except Exception:
            # exception() rather than error(): this is the only record of why
            # the actor stopped, and "%s" on the exception drops the traceback.
            self.logger.exception("MCTS_process failed")
        self.logger.info("MCTS process exiting")
    
    # -----------------------------
    # Batch expansion helper function
    # -----------------------------
    def batch_expand_nodes(self, nodes, batch_size=40):
        """
        Given a list of MCTSNode objects that need expansion, expand them in batches.
        
        Args:
        nodes: list of MCTSNode objects to expand.
        batch_size: number of nodes to process in a single batch. (actual batch size is batch_size*self.action_space)
        """
        # Terminal nodes have no successors, and re-expanding an expanded node
        # would discard its accumulated statistics. Nodes whose latent has been
        # released cannot be expanded at all. Filter all three out before the
        # network call rather than after, so we don't pay for them.
        nodes = [n for n in nodes
                 if not n.done and not n.is_expanded and n.latent is not None]
        if not nodes:
            return

        A = self.action_space  # number of actions
        N = len(nodes)
        
        # Process nodes in chunks of batch_size.
        for i in range(0, N, batch_size):
            chunk = nodes[i:i + batch_size]
            num_chunk = len(chunk)
            
            # Gather latent states from each node in the chunk.
            latent_list = [node.latent for node in chunk]  # each shape: (1, C, H, W)
            # Stack into a tensor of shape (num_chunk, C, H, W)
            stacked_latents = torch.cat(latent_list, dim=0)
            # For each node, we want to evaluate dynamics for each action.
            # Repeat each latent A times: resulting shape (num_chunk * A, C, H, W)
            latent_batch = stacked_latents.repeat_interleave(A, dim=0)
            latent_batch = latent_batch.to(self.device)
            
            # Create a tensor of action indices. For each node in the chunk, we need actions 0,...,A-1.
            action_tensor = torch.arange(A, dtype=torch.long, device=self.device).unsqueeze(0).repeat(num_chunk, 1)
            action_tensor = action_tensor.view(-1)  # shape: (num_chunk * A,)
        
            with torch.no_grad():
                # next_latents: (num_chunk*A, C, H, W)
                # rewards: (num_chunk*A, 1) or (num_chunk*A,) depending on implementation
                next_latents, rewards, _ = self.network_manager.dynamics(latent_batch, action_tensor)
                batch_policy_logits, estimated_values, _ = self.network_manager.prediction(next_latents)
                next_policy = F.softmax(batch_policy_logits, dim=1).cpu().numpy()

                child_prefixes, child_states = self.expand_prefixes(chunk, next_latents, A)
            
            # Update each node in the chunk.
            for j, node in enumerate(chunk):
                start = j * A
                end = start + A
                # Assume node.policy is already a numpy array of shape (A,)
                policy = node.policy
                for a in range(A):
                    if a in node.children and node.children[a].is_gameplay:
                        continue
                    idx = start + a
                    # Create the child node with latent state
                    if child_prefixes is None:
                        edge_reward = (rewards[idx].item() if rewards[idx].ndim == 0
                                       else rewards[idx].squeeze().item())
                        child_prefix, child_state = 0.0, None
                    else:
                        # The prefix accumulates from the search root, so the
                        # reward on this edge alone is the increment over the
                        # parent. Undiscounted, so backup_simulation applies the
                        # discount exactly as it does for real gameplay rewards.
                        child_prefix = float(child_prefixes[idx])
                        edge_reward = child_prefix - node.prefix
                        child_state = (child_states[0][:, idx:idx + 1].clone(),
                                       child_states[1][:, idx:idx + 1].clone())
                    # Create the child node with latent state
                    child = MCTSNode(
                        latent=next_latents[idx].unsqueeze(0),
                        prior=policy[a],
                        reward=edge_reward,
                        policy=next_policy[idx],
                        expanded_value=estimated_values[idx].item(),
                        prefix=child_prefix,
                        prefix_state=child_state,
                    )
                    node.children[a] = child
                node.is_expanded = True
        
    def expand_prefixes(self, chunk, next_latents, action_space):
        """Step the value-prefix LSTM from each parent's state, once per child.

        Returns (prefixes, state) aligned with next_latents, or (None, None) when
        the tree is configured to use the per-step reward head instead.
        """
        if not USE_VALUE_PREFIX_IN_TREE or not hasattr(self.network_manager, "value_prefix"):
            return None, None

        prefix_net = self.network_manager.value_prefix
        parent_states = []
        for node in chunk:
            state = node.prefix_state
            if state is None:
                state = prefix_net.initial_state(1, self.device)
            parent_states.append(state)

        # (1, num_chunk, H) -> one copy per action -> (1, num_chunk*A, H)
        h = torch.cat([s[0] for s in parent_states], dim=1).repeat_interleave(action_space, dim=1)
        c = torch.cat([s[1] for s in parent_states], dim=1).repeat_interleave(action_space, dim=1)

        was_training = prefix_net.training
        prefix_net.eval()
        try:
            prefixes, _, state = prefix_net(next_latents, (h.contiguous(), c.contiguous()))
        finally:
            prefix_net.train(was_training)
        return prefixes.squeeze(-1).cpu().numpy(), state

    # -----------------------------
    # Backup Phase: Propagate the leaf value up the simulation path.
    # -----------------------------
    def backup_simulation(self, path, leaf_value, discount=None, min_max_stats=None):
        # Resolved at call time, not baked into the signature: create_tree_from_game
        # calls this without a discount, so a hardcoded default silently pinned the
        # Monte Carlo value targets to 0.997 no matter what DISCOUNT was set to.
        # That made any experiment varying gamma measure nothing.
        discount = DISCOUNT if discount is None else discount
        backup_value = leaf_value
        for n in reversed(path):
            n.visit_count += 1
            n.value_sum += backup_value
            n.virtual_loss = 0.0
            backup_value = n.reward + discount * backup_value
            if min_max_stats is not None:
                # Track the same composite selection normalises: r + discount*V,
                # not V alone, or the recorded range is on the wrong scale.
                min_max_stats.update(n.reward + discount * n.q_value())

    def create_tree_from_game(self, game_trajectory):
        prev_node = None
        prev_action_taken = None
        path = []
        for (obs_sample, action_taken, reward, latent, policy, expanded_value) in game_trajectory:
            new_node = MCTSNode(latent=latent, prior=1.0, reward=reward, is_gameplay=True)
            new_node.policy = policy
            new_node.expanded_value = expanded_value
            if prev_node != None:
                new_node.prior = prev_node.policy[prev_action_taken]
                prev_node.children[prev_action_taken] = new_node
            path.append(new_node)
            prev_node = new_node
            prev_action_taken = action_taken
        path[-1].done = True

        # Nodes are expanded lazily, the first time a simulation selects them.
        # Expanding the whole chain up front allocated action_space child
        # latents for every timestep in the episode (~5x the episode length)
        # before the search had even started.
        self.backup_simulation(path, 0)
        self.current_trajectory = path
    
    # alpha=0.25 is MuZero's Atari setting. The old 0.03 was AlphaGo's value,
    # tuned for 361 moves: over 5 actions it made the Dirichlet sample
    # effectively one-hot, ~92% of the noise mass on a single random action.
    def add_noise_to_node(self, idx, epsilon = 0.25, alpha = 0.25):
        node = self.current_trajectory[idx]
        node.policy = add_dirichlet_noise(node.policy, epsilon=epsilon, alpha=alpha)
        for action, child in node.children.items():
            child.prior = node.policy[action]

    async def selection_task_no_expand(self, index, c_puct, discount, max_depth=MAX_DEPTH):
        """
        Traverse the tree from the given root until a leaf is reached.
        """
        node = self.current_trajectory[index]
        path = []
        while len(path) < max_depth:
            if not node.is_expanded or node.done:
                break  # leaf: either not yet expanded, or terminal
            path.append(node)
            total_visits = sum(child.visit_count + child.virtual_loss for child in node.children.values())
            best_score = -float('inf')
            best_candidates = []  # List to store (action, child) pairs with highest score.
            for a, child in node.children.items():
                u = c_puct * child.prior * np.sqrt(total_visits) / (1 + child.visit_count + child.virtual_loss)
                # Score the edge as r + discount * V(child) -- the paper's
                # Q(s,a). q_value() alone is the value from the child ONWARD:
                # backup folds the edge reward into the parent's statistics, so
                # scoring on it left selection blind to immediate reward
                # differences between actions. Virtual loss is already folded
                # into q_value() via the visit count; subtracting it again here
                # would swamp a [0,1] Q. An unvisited child reads its network
                # value estimate rather than 0, which on this tree's raw-return
                # scale would rank every imagined child beneath the gameplay
                # child before a single simulation had judged them.
                if child.visit_count + child.virtual_loss > 0:
                    onward = child.q_value()
                elif child.done:
                    onward = 0.0
                else:
                    onward = child.expanded_value
                score = self.min_max_stats.normalize(child.reward + discount * onward) + u
                if score > best_score:
                    best_score = score
                    best_candidates = [child]
                elif score == best_score:
                    best_candidates.append(child)
            # Randomly select one of the best candidates in case of a tie.
            chosen_child = random.choice(best_candidates)
            async with chosen_child.lock:
                chosen_child.virtual_loss += 1
            node = chosen_child
        path.append(node)
        await self.selection_results.put((node, path))

    async def expand_selected(self, discount):
        simulations = []
        while not self.selection_results.empty():
            simulations.append(await self.selection_results.get())

        # Expand each distinct leaf once; batch_expand_nodes drops the terminal
        # and already-expanded ones itself.
        unique_leaves = []
        seen = set()
        for leaf, _ in simulations:
            if leaf.id not in seen:
                seen.add(leaf.id)
                unique_leaves.append(leaf)
        self.batch_expand_nodes(unique_leaves)

        # Back up EVERY simulation, not just the ones that caused an expansion.
        # Simulations dropped here previously (a duplicate leaf within the
        # batch, or a leaf that was already expanded) never reached
        # backup_simulation -- the only place virtual_loss is cleared. So
        # virtual loss accumulated monotonically on the most-visited nodes and
        # progressively steered the search away from precisely the actions it
        # had found to be good. Their visits were lost too, so the effective
        # simulation budget was well under num_simulations.
        for leaf, path in simulations:
            leaf_value = 0.0 if leaf.done else leaf.expanded_value
            self.backup_simulation(path, leaf_value, discount=discount,
                                   min_max_stats=self.min_max_stats)

    def seed_min_max_stats(self, root, discount):
        """Prime the per-search normalisation range from values already in the tree.

        MinMaxStats starts empty and normalize() is the identity until it has
        seen a range, so the first selection round used to score on RAW values:
        the gameplay child's backed-up Monte Carlo return (order hundreds)
        against exploration terms of order 0.3, which sent the entire first
        batch of simulations down the behaviour trajectory. Seeding from the
        root's children puts round one on the same [0,1] footing as every
        later round.
        """
        for child in root.children.values():
            if child.visit_count > 0:
                onward = child.q_value()
            elif child.done:
                onward = 0.0
            else:
                onward = child.expanded_value
            self.min_max_stats.update(child.reward + discount * onward)

    def drop_non_gameplay_children_and_compute_improvements(self, idx, temperature=1.0, baseline=None):
        baseline = baseline or {}
        children_to_drop = []
        node = self.current_trajectory[idx]
        for action, child in node.children.items():
            if not child.is_gameplay:
                children_to_drop.append(action)
        # Improved policy from the visits THIS search allocated. Raw counts also
        # include the chain backup's bookkeeping visit and simulations from
        # earlier searches, which reach this node only along the gameplay chain
        # -- both sit on the behaviour action, so counting them biases every
        # policy target toward the action the agent already took. The inherited
        # visits still warm the Q estimates; they just don't get read out as
        # this search's conclusion.
        visit_counts = np.array([
            node.children[a].visit_count - baseline.get(a, 0) if a in node.children else 0
            for a in range(self.action_space)
        ])
        has_policy = bool(visit_counts.sum() > 0)
        if has_policy:
            improved_policy = apply_temperature(visit_counts, temperature)
        else:
            # No visits to read -- a terminal root has no successors to search.
            # The uniform vector is a shape placeholder only: has_policy False
            # masks it out of the loss, rather than training the policy head
            # toward uniform on terminal observations.
            improved_policy = np.ones(self.action_space) / self.action_space
        root_value = node.q_value()
        for action in children_to_drop:
            del node.children[action]
        # Release the latent: the search is finished with it, and holding one
        # per timestep for a whole episode is the memory cost this avoids.
        # Assigning None rather than `del` keeps the attribute defined, so
        # searching the same index twice -- which is what re-analysing a stored
        # trajectory means -- raises nothing.
        node.latent = None
        return improved_policy, root_value, has_policy

    # -----------------------------
    # Full MCTS: Run multiple simulations, batch expand unexpanded leaves,
    # then backup their values.
    # -----------------------------
    async def run_full_mcts(self, index, num_simulations=128, sim_batches=8, c_puct=1.0, discount=None, max_depth=MAX_DEPTH, temperature=1.0):
        discount = DISCOUNT if discount is None else discount
        # Now that expansion is lazy the root has to be expanded explicitly:
        # selection stops at unexpanded nodes, and add_noise_to_node writes the
        # noisy prior through to the children, which must therefore exist.
        root = self.current_trajectory[index]
        # Value normalisation is per-search, as in the MuZero paper.
        self.min_max_stats = MinMaxStats()
        # The prefix accumulates from the root, so it restarts at every search.
        root.prefix = 0.0
        root.prefix_state = None
        if not root.is_expanded:
            self.batch_expand_nodes([root])
        self.add_noise_to_node(index)
        self.seed_min_max_stats(root, discount)
        # Visits the root's children carry INTO this search: the chain backup's
        # bookkeeping visit plus traffic from earlier searches through the
        # shared tree. Snapshotted so the improved policy can count only what
        # this search's own budget decides.
        baseline_visits = {a: child.visit_count for a, child in root.children.items()}
        for x in range(max(int(num_simulations // sim_batches), 1)):
            futures = [
                self.selection_task_no_expand(index, c_puct, discount, max_depth)
                for _ in range(sim_batches)
            ]
            await asyncio.gather(*futures)
            await self.expand_selected(discount)
        return self.drop_non_gameplay_children_and_compute_improvements(
            index, temperature=temperature, baseline=baseline_visits)

    def select_reanalysis_indices(self, length, stride=None):
        """Timesteps to run a search on, evenly spaced with a random offset.

        A stride rather than a random sample, so coverage along the episode is
        uniform; a random offset so successive passes over the buffer land on
        different positions and the whole episode gets covered over time.
        """
        stride = REANALYSE_STRIDE if stride is None else stride
        if length <= 0:
            return set()
        if stride <= 1:
            return set(range(length))
        # Cap the offset at the trajectory length: an episode shorter than the
        # stride would otherwise draw an offset past its end and be reanalysed
        # with no searched timesteps at all, contributing zero policy targets.
        offset = random.randrange(min(stride, length))
        return set(range(offset, length, stride))

    def get_value(self, idx):
        return self.current_trajectory[idx].q_value()

    def nstep_value(self, idx, n=None, discount=None):
        """MuZero-Reanalyze value target: n real rewards, then bootstrap the
        CURRENT network's value estimate.

        The Monte Carlo alternative -- q_value() from the chain backup -- is
        the return the BEHAVIOUR policy earned when the trajectory was played,
        which mis-values positions in proportion to how far the policy has
        moved since. expanded_value was computed by eval_obs with today's
        weights, so bootstrapping it re-values everything past the horizon on
        every reanalysis pass, at no extra cost.

        Falls back to the pure Monte Carlo return when the episode ends inside
        the horizon: nothing follows a terminal state, so there is nothing to
        bootstrap -- and a terminal tail node's onward value is exactly 0.
        """
        n = TD_STEPS if n is None else n
        discount = DISCOUNT if discount is None else discount
        traj = self.current_trajectory
        value, g = 0.0, 1.0
        for k in range(1, n + 1):
            if idx + k >= len(traj):
                return value
            value += g * traj[idx + k].reward
            g *= discount
        tail = traj[idx + n]
        return value + g * (0.0 if tail.done else tail.expanded_value)

class TrainingHelper:
    def __init__(self, action_space, device, network_ref, storage_ref, enable_logging=False):
        self.network_helper = network_ref
        self.storage_manager = StorageManager(storage_ref)  
        self.device = device
        self.action_space = action_space
        self.training_steps = 0
        self.representation = RepresentationNetwork(input_channels=12,
                                                    latent_channels=LATENT_CHANNELS).to(device)
        self.dynamics = DynamicsNetwork(latent_channels=LATENT_CHANNELS,
                                        action_space=action_space,
                                        support_size=REWARD_SUPPORT,
                                        num_bins=REWARD_BINS).to(device)
        self.prediction = PredictionNetwork(latent_channels=LATENT_CHANNELS,
                                            action_space=action_space,
                                            support_size=VALUE_SUPPORT,
                                            num_bins=VALUE_BINS).to(device)
        self.value_prefix = ValuePrefixNetwork(latent_channels=LATENT_CHANNELS,
                                               hidden_size=VALUE_PREFIX_HIDDEN,
                                               support_size=VALUE_PREFIX_SUPPORT,
                                               num_bins=VALUE_PREFIX_BINS).to(device)
        self.rep_optimizer = make_optimizer(self.representation.parameters())
        self.dyn_optimizer = make_optimizer(self.dynamics.parameters())
        self.pred_optimizer = make_optimizer(self.prediction.parameters())
        self.prefix_optimizer = make_optimizer(self.value_prefix.parameters())
        self.logger = configure_actor_logger("TrainingHelper", enable_logging)
        self.logger.info("Initialized TrainingHelper logger.")

    async def setup(self):
        await self.storage_manager.setup()
        await self.load_networks()
    
    async def load_networks(self):
        state = await self.network_helper.get_state.remote()
        self.training_steps = state["training_steps"]
        self.network_version = state["version"]
        self.representation.load_state_dict(state["representation"])
        self.dynamics.load_state_dict(state["dynamics"])
        self.prediction.load_state_dict(state["prediction"])
        self.value_prefix.load_state_dict(state["value_prefix"])
        self.rep_optimizer.load_state_dict(state["rep_optimizer"])
        self.dyn_optimizer.load_state_dict(state["dyn_optimizer"])
        self.pred_optimizer.load_state_dict(state["pred_optimizer"])
        self.prefix_optimizer.load_state_dict(state["prefix_optimizer"])

    async def save_networks(self):
        await self.network_helper.update_state.remote({
            "representation":   self.representation.state_dict(),
            "dynamics":         self.dynamics.state_dict(),
            "prediction":       self.prediction.state_dict(),
            "value_prefix":     self.value_prefix.state_dict(),
            "rep_optimizer":    self.rep_optimizer.state_dict(),
            "dyn_optimizer":    self.dyn_optimizer.state_dict(),
            "pred_optimizer":   self.pred_optimizer.state_dict(),
            "prefix_optimizer": self.prefix_optimizer.state_dict(),
            "training_steps":   self.training_steps,
        })
        
    
    # -----------------------------
    # Training Step: remote method to update networks
    # -----------------------------
    async def train_step(self, batch):
        """
        Multi-step unrolling training step.
        
        Expects a batch dictionary with keys:
        "obs": (B, unroll_steps+1, C, H, W) uint8 images,
        "actions": (B, unroll_steps) ints,
        "rewards": (B, unroll_steps+1) floats,
        "target_policy": (B, unroll_steps+1, action_space) probabilities,
        "target_value": (B, unroll_steps+1) floats.
        """
        # Determine the number of unroll steps from the batch.
        unroll_steps = batch["actions"].shape[1]
        
        # Convert inputs to tensors and move them to the appropriate device.
        obs = torch.tensor(batch["obs"], dtype=torch.float32, device=self.device) / 255.0
        actions = torch.tensor(batch["actions"], dtype=torch.long, device=self.device)
        rewards = torch.tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        target_policy = torch.tensor(batch["target_policy"], dtype=torch.float32, device=self.device)
        target_value = torch.tensor(batch["target_value"], dtype=torch.float32, device=self.device)
        # Positions whose target_policy is a real improved policy rather than a
        # placeholder. Reanalysis only searches every REANALYSE_STRIDE-th step,
        # and trajectory padding is never a real target.
        policy_mask = torch.tensor(np.asarray(batch["has_policy_target"]),
                                   dtype=torch.float32, device=self.device)
        
        # Zero out the gradients for all optimizers.
        self.rep_optimizer.zero_grad()
        self.pred_optimizer.zero_grad()
        self.dyn_optimizer.zero_grad()
        self.prefix_optimizer.zero_grad()
        
        total_loss = 0.0

        # ---------------------
        # Step 0: Initial Inference
        # ---------------------
        init_obs = obs[:, 0]
        latent = self.representation(init_obs)
        policy_logits, _, value_logits = self.prediction(latent)
        
        converted_target_rewards = scalar_categorical.scalar_to_categorical_batch(
            rewards, self.dynamics.support_size, self.dynamics.num_bins, self.device,
            transform=USE_H_TRANSFORM)

        # Value prefix targets: the cumulative discounted reward sum along the
        # unroll, which is what the LSTM head predicts in one shot instead of
        # per-step rewards that are summed afterwards.
        # UNDISCOUNTED (discount=1.0), matching what the tree consumes: an edge
        # reward there is the difference between a child's prefix and its
        # parent's, and that identity only holds without discounting baked in.
        # backup_simulation applies the discount, as it does for real rewards.
        target_prefix = discounted_prefix_targets(rewards, discount=1.0)
        converted_target_prefix = scalar_categorical.scalar_to_categorical_batch(
            target_prefix, self.value_prefix.support_size, self.value_prefix.num_bins, self.device,
            transform=USE_H_TRANSFORM)
        # The hidden state IS the accumulated reward, so it starts empty at the
        # beginning of every unroll. Same reason it must reset at a search root.
        prefix_state = self.value_prefix.initial_state(obs.size(0), self.device)

        # Compute losses for the initial step (no reward loss here).
        policy_loss_0 = masked_policy_loss(policy_logits, target_policy[:, 0, :], policy_mask[:, 0])
        converted_target_values = scalar_categorical.scalar_to_categorical_batch(
            target_value, self.prediction.support_size, self.prediction.num_bins, self.device,
            transform=USE_H_TRANSFORM)
        value_loss_0 = -torch.sum(converted_target_values[:, 0, :] * F.log_softmax(value_logits, dim=1), dim=1).mean()
        total_loss += (POLICY_LOSS_WEIGHT * policy_loss_0
                       + VALUE_LOSS_WEIGHT * value_loss_0)

        # ---------------------
        # Unroll for each subsequent step
        # ---------------------
        for i in range(1, unroll_steps + 1):
            # Halve the gradient entering each recurrent step so it does not
            # compound across the unroll.
            latent = scale_gradient(latent, LATENT_GRADIENT_SCALE)
            # Use the action from the previous step to roll forward the latent state.
            latent, _, reward_logits = self.dynamics(latent, actions[:, i - 1])
            with torch.no_grad():
                true_latent = self.representation(obs[:, i])
            latent_loss = F.mse_loss(latent, true_latent, reduction="mean")
            
            # Loss for reward prediction. (reward[0] not used since it is reward for entering obs)
            reward_loss = -torch.sum(converted_target_rewards[:, i, :] * F.log_softmax(reward_logits, dim=1), dim=1).mean()

            # Value prefix over the same unroll. Both heads are trained on the
            # same trunk for now so they can be compared directly on identical
            # representations; the per-step head goes away once the tree search
            # is migrated to consume prefixes.
            _, prefix_logits, prefix_state = self.value_prefix(latent, prefix_state)
            prefix_loss = -torch.sum(converted_target_prefix[:, i, :] * F.log_softmax(prefix_logits, dim=1), dim=1).mean()

            policy_logits, _, value_logits = self.prediction(latent)

            # Loss for policy prediction, over the positions that have one.
            # Only the POLICY is masked here, and deliberately: a padded position
            # carries a uniform placeholder policy, which is fabricated, but its
            # target_value of 0.0 and reward of 0.0 are TRUE -- padding repeats
            # the terminal observation, and a terminal state really is worth
            # nothing. Masking those would discard correct signal, not noise.
            policy_loss = masked_policy_loss(policy_logits, target_policy[:, i, :], policy_mask[:, i])
            value_loss = -torch.sum(converted_target_values[:, i, :] * F.log_softmax(value_logits, dim=1), dim=1).mean()

            step_loss = (REWARD_LOSS_WEIGHT * reward_loss
                         + PREFIX_LOSS_WEIGHT * prefix_loss
                         + POLICY_LOSS_WEIGHT * policy_loss
                         + VALUE_LOSS_WEIGHT * value_loss
                         + CONSISTENCY_LOSS_WEIGHT * latent_loss)

            # Each unroll step contributes 1/K of the gradient, so the total does
            # not scale with how far the unroll runs.
            total_loss += scale_gradient(step_loss, 1.0 / unroll_steps)

        # Backpropagate the total loss.
        total_loss.backward()
        
        # Step the optimizers.
        self.rep_optimizer.step()
        self.dyn_optimizer.step()
        self.prefix_optimizer.step()
        self.pred_optimizer.step()
        
        return total_loss.item()

    async def training_process(self):
        try:
            while True:
                await self.storage_manager.update_if_needed()
                if len(self.storage_manager.replay_buffer) < MAX_REPLAY // 2:
                    self.logger.debug("Training sleeping")
                    await asyncio.sleep(1)
                    continue
                batch = await self.prepare_batch(self.storage_manager.replay_buffer, TRAINING_BATCHES, 5, self.action_space)
                loss = await self.train_step(batch)
                self.training_steps += 1
                self.logger.info(f"Step: {self.training_steps} Training loss: {loss}")
                if self.training_steps % 100 == 0 and self.training_steps > 1:
                    self.logger.info(f"Saving network at {self.training_steps} training steps")
                    await self.save_networks()
        except Exception:
            self.logger.exception("training_process failed")
        self.logger.info("Training process exiting")
    
    async def prepare_batch(self, replay_buffer, batch_size, unroll_steps, action_space):
        """
        Prepares a training batch for multi-step unrolling.
        
        Args:
        replay_buffer: list of trajectories. Each trajectory is a list of transitions.
                        Each transition is a tuple: 
                        (obs, action, reward, next_obs, target_policy, target_value)
        batch_size: number of samples to include in the batch.
        unroll_steps: number of unroll steps to sample.
        action_space: number of possible actions.
        
        Returns:
        A dictionary with keys:
            "obs": (B, unroll_steps, C, H, W) initial observations,
            "actions": (B, unroll_steps) sequence of actions,
            "rewards": (B, unroll_steps+1) sequence of rewards,
            "target_policy": (B, unroll_steps+1, action_space) target policies,
            "target_value": (B, unroll_steps+1) target values,
            "has_policy_target": (B, unroll_steps+1) bool, whether that position
                carries a real improved policy. False for timesteps reanalysis
                skipped and for padding past the end of a trajectory; the
                policy loss is masked there.
        """
        batch_obs = []
        batch_actions = []
        batch_rewards = []
        batch_target_policy = []
        batch_target_value = []
        batch_has_policy = []
        
        # Sample uniformly over TRANSITIONS, not trajectories: weighting each
        # trajectory by its length. Picking a trajectory uniformly first
        # oversamples the transitions of short episodes in proportion to how
        # short they are -- a 100-step death was sampled 6x as often per
        # transition as a 600-step run.
        trajectories = list(replay_buffer)
        lengths = np.array([len(t) for t in trajectories], dtype=np.float64)
        traj_probs = lengths / lengths.sum()
        for _ in range(batch_size):
            traj = trajectories[np.random.choice(len(trajectories), p=traj_probs)]
            traj_length = len(traj)
            start_index = np.random.randint(0, traj_length)
            
            # Prepare lists for unroll data.
            obs = []
            actions = []
            rewards = []
            target_policy = []
            target_value = []
            has_policy = []
            
            # We need unroll_steps+1 target entries (for step 0 and each unroll step).
            for step in range(unroll_steps + 1):
                if start_index + step < traj_length:
                    obs.append(traj[start_index + step]["obs"])
                    node = traj[start_index + step]
                    # Append the target policy and value for each step.
                    target_policy.append(node["target_policy"])
                    target_value.append(node["target_value"])
                    # Trajectories written before reanalysis subsampling have no
                    # such key; every position in them was searched.
                    has_policy.append(node.get("has_policy_target", True))
                    # reward for entering node (reward[0] will not be used)
                    rewards.append(node["reward"])
                    if step < unroll_steps:
                        # For actions, only collect for unroll_steps.
                        actions.append(node["action"])
                else:
                    obs.append(traj[traj_length - 1]["obs"])
                    # If the trajectory doesn't have enough transitions, pad with zeros.
                    target_policy.append(np.ones(action_space) / action_space)
                    target_value.append(0.0)
                    rewards.append(0.0)
                    has_policy.append(False)   # padding is not a real target
                    if step < unroll_steps:
                        actions.append(0)         # or a default action
            
            batch_obs.append(obs)
            batch_actions.append(actions)
            batch_rewards.append(rewards)
            batch_target_policy.append(target_policy)
            batch_target_value.append(target_value)
            batch_has_policy.append(has_policy)
        
        # Convert lists to numpy arrays.
        batch_obs = np.array(batch_obs)  # (B, H, W, C)
        batch_actions = np.array(batch_actions)  # (B, unroll_steps)
        batch_rewards = np.array(batch_rewards)  # (B, unroll_steps)
        batch_target_policy = np.array(batch_target_policy)  # (B, unroll_steps+1, action_space)
        batch_target_value = np.array(batch_target_value)  # (B, unroll_steps+1)
        batch_has_policy = np.array(batch_has_policy)  # (B, unroll_steps+1)
        
        return {
            "obs": batch_obs,
            "actions": batch_actions,
            "rewards": batch_rewards,
            "target_policy": batch_target_policy,
            "target_value": batch_target_value,
            "has_policy_target": batch_has_policy
        }

#Agent will be single threaded for real time learning
class MuZeroAgent:
    def __init__(self, action_space, device, network_ref, storage_ref, enable_logging=False):
        self.network_manager = NetworkManager(network_ref, action_space=action_space, device=device, load_dynamics=False)
        # The agent only ever writes to the gameplay buffer, so it has no reason
        # to mirror the replay buffer.
        self.storage_manager = StorageManager(storage_ref, update_replay=False)
        self.device = device
        self.action_space = action_space
        self.training_steps = 0
        self.logger = configure_actor_logger("MuZeroAgent", enable_logging)
        self.logger.info("Initialized actor logger.")

    async def setup(self):
        await self.network_manager.setup()
        await self.storage_manager.setup()

    async def gameplay_process(self):
        try:
            if not hasattr(np, 'bool8'):
                np.bool8 = np.bool_
            env = gym.make("MsPacman-v4")
            env = EpisodicLifeEnv(env)
            period = real_time_step_period(env) if SELF_PLAY_REAL_TIME else 0.0
            self.logger.info("Self-play paced at %.1f decisions/s (%s)",
                             (1.0 / period) if period else float("inf"),
                             "real time" if period else "unlimited")
            while True:
                # Refresh weights between episodes. Without this the behaviour
                # policy stayed frozen at whatever setup() loaded, so self-play
                # kept generating data from the initial network forever and the
                # policy improvement loop never closed. (Refreshing per episode
                # rather than per step keeps acting real-time.)
                await self.network_manager.update_if_needed()
                self.logger.debug("Starting new game")
                limiter = RateLimiter(period)
                obs, info = await asyncio.to_thread(env.reset)
                preprocessed_obs = preprocess_frame(obs, resize_shape=IMAGE_RESHAPE)
                prev_reward = 0.0
                done = False
                episode_reward = 0
                game_trajectory = []  # store tuples of (obs, latent, action)
                frame_stack = initial_frame_stack(preprocessed_obs)
                
                # Play one episode.
                while not done:
                    preprocessed_obs = preprocess_frame(obs, resize_shape=IMAGE_RESHAPE)
                    frame_stack = push_frame(frame_stack, preprocessed_obs)
                    with torch.no_grad():
                        policy, value = self.network_manager.initial_inference(frame_stack)
                    if ACTING_TEMPERATURE != 1.0:
                        policy = policy ** (1.0 / ACTING_TEMPERATURE)
                        policy = policy / policy.sum()
                    action = np.random.choice(np.arange(self.action_space), p=policy)
                    obs, reward, terminated, truncated, _ = await asyncio.to_thread(env.step, action)
                    done = terminated or truncated
                    await limiter.wait()
                    # obs, chosen action, reward for entering obs
                    reward = min(reward, 400)
                    game_trajectory.append((frame_stack, action, prev_reward))
                    episode_reward += reward
                    prev_reward = reward

                # Record the terminal state. The loop above assigns the final
                # step's reward to prev_reward and then exits, so without this
                # the last reward of every episode -- often the informative one
                # -- was dropped, and create_tree_from_game ended up marking a
                # non-terminal node as done. The action is a placeholder; no
                # action is ever taken from a terminal state.
                preprocessed_obs = preprocess_frame(obs, resize_shape=IMAGE_RESHAPE)
                frame_stack = push_frame(frame_stack, preprocessed_obs)
                game_trajectory.append((frame_stack, 0, prev_reward))

                await self.storage_manager.append_gameplay_buffer((game_trajectory, episode_reward))
        except Exception:
            self.logger.exception("gameplay_process failed")
        self.logger.info("Gameplay process exiting")


def initial_frame_stack(frame, num_frames=4):
    """Build the starting (num_frames*C, H, W) stack by repeating one frame.

    np.tile, not np.repeat: repeat interleaves the channel planes into
    [R,R,R,R,G,G,G,G,B,B,B,B], whereas push_frame below treats the stack as
    consecutive whole frames. The mismatch corrupted the first observation of
    every episode and stayed wrong for num_frames steps as it shifted through.
    """
    return np.tile(frame, (num_frames, 1, 1))


def push_frame(frame_stack, frame):
    """Drop the oldest frame from the stack and append the newest."""
    channels = frame.shape[0]
    return np.concatenate([frame_stack[channels:, :, :], frame], axis=0)


def preprocess_frame(frame, resize_shape=(84, 84)):
    """
    Preprocess a single game frame.

    Args:
        frame (np.array): Input image, expected to be in RGB format.
        resize_shape (tuple): Target size as (width, height). Default is (84, 84).

    Returns:
        np.array: Processed image.
    """
    # Convert from RGB to grayscale.
    #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # Resize the image to the desired shape.
    frame = cv2.resize(frame, resize_shape)#, interpolation=cv2.INTER_AREA)
    frame = np.transpose(frame, (2, 0, 1))
    return frame


# ---------------------------------------------------------------------------
# Ray actor wrappers
#
# The classes above are plain Python so they can be constructed directly in
# tests -- @ray.remote returns an ActorClass, which cannot be instantiated
# normally. Decorating here keeps both paths available: tests import the class,
# main uses the actor.
#
# max_concurrency governs concurrent *Ray* calls into an actor. MCTSHelper's
# was tied to MCTS_BATCHES on the assumption that it limited the simulation
# coroutines, but those are created by asyncio.gather inside MCTS_process and
# are not Ray calls at all. Worse, MCTSHelper keeps per-search state on the
# instance (current_trajectory, min_max_stats, selection_results), so two
# concurrent Ray calls into one actor would silently corrupt each other's
# searches. The design is one long-running MCTS_process call per actor, after
# a setup call that completes first; max_concurrency=1 enforces it.
# ---------------------------------------------------------------------------
StorageActor  = ray.remote(max_concurrency=8)(StorageHelper)
NetworkActor  = ray.remote(max_concurrency=8)(NetworkHelper)
MCTSActor     = ray.remote(max_concurrency=1)(MCTSHelper)
TrainingActor = ray.remote(TrainingHelper)
AgentActor    = ray.remote(max_concurrency=1)(MuZeroAgent)


def resource_plan(num_mcts_actors=NUM_MCTS_ACTORS, use_cuda=True):
    """Total CPU and GPU the actor fleet will request."""
    cpu = (CPU_STORAGE + CPU_NETWORK + CPU_TRAINING + CPU_AGENT
           + CPU_MCTS * num_mcts_actors)
    gpu = 0.0
    if use_cuda:
        gpu = (GPU_NETWORK + GPU_TRAINING + GPU_AGENT
               + mcts_gpu_share(num_mcts_actors) * num_mcts_actors)
    return cpu, gpu


def check_cluster_resources(num_mcts_actors=NUM_MCTS_ACTORS, use_cuda=True):
    """Fail loudly when the fleet cannot possibly be scheduled.

    Ray does not raise when an actor's resource request exceeds what the
    cluster has -- it queues the actor indefinitely, so ray.get() on its
    setup() call simply never returns. Turning that silent hang into an error
    at startup is the difference between a one-line config fix and an evening
    spent wondering which actor deadlocked.
    """
    resources = ray.cluster_resources()
    have_cpu = resources.get("CPU", 0)
    have_gpu = resources.get("GPU", 0)
    want_cpu, want_gpu = resource_plan(num_mcts_actors, use_cuda)

    problems = []
    if want_cpu > have_cpu:
        problems.append(f"CPU: requesting {want_cpu:g}, cluster has {have_cpu:g}")
    if want_gpu > have_gpu + 1e-9:
        problems.append(f"GPU: requesting {want_gpu:g}, cluster has {have_gpu:g}")
    if problems:
        raise RuntimeError(
            "Actor resource requests exceed the Ray cluster. Ray would queue "
            "the surplus actors forever rather than raising:\n  "
            + "\n  ".join(problems)
            + f"\nLower NUM_MCTS_ACTORS (currently {num_mcts_actors}) or the "
              "per-actor budgets at the top of muzero.py."
        )
    print(f"Resource plan: {want_cpu:g}/{have_cpu:g} CPU, "
          f"{want_gpu:g}/{have_gpu:g} GPU, {num_mcts_actors} MCTS actor(s)")


# -----------------------------
# Main Loop Example with Full MCTS Sample Generation
# -----------------------------
if __name__ == '__main__':
    ray.init(include_dashboard=True, dashboard_port=8265)#log_to_driver=True)
    ACTION_SPACE = 5
    ENABLE_LOGGING = True
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    def gpu(fraction):
        # Requesting a GPU on a machine that has none queues the actor forever,
        # exactly like over-subscribing one. Ask for nothing instead.
        return fraction if use_cuda else 0

    check_cluster_resources(NUM_MCTS_ACTORS, use_cuda)

    network = NetworkActor.options(
                                num_cpus=CPU_NETWORK,
                                num_gpus=gpu(GPU_NETWORK)).remote(
                                action_space=ACTION_SPACE,
                                device=device,
                                enable_logging=ENABLE_LOGGING)
    storage = StorageActor.options(num_cpus=CPU_STORAGE).remote()
    agent = AgentActor.options(
                                num_cpus=CPU_AGENT,
                                num_gpus=gpu(GPU_AGENT)).remote(
                                action_space=ACTION_SPACE,
                                device=device,
                                network_ref=network,
                                storage_ref=storage,
                                enable_logging=ENABLE_LOGGING)
    mcts_instances = [
        MCTSActor.options(
                                num_cpus=CPU_MCTS,
                                num_gpus=gpu(GPU_MCTS)).remote(
                                action_space=ACTION_SPACE,
                                device=device,
                                network_ref=network,
                                storage_ref=storage,
                                enable_logging=ENABLE_LOGGING,
                                name=f'MCTSHelper{i}')
        for i in range(NUM_MCTS_ACTORS)
    ]
    training_instance = TrainingActor.options(
                                num_cpus=CPU_TRAINING,
                                num_gpus=gpu(GPU_TRAINING)).remote(
                                action_space=ACTION_SPACE,
                                device=device,
                                network_ref=network,
                                storage_ref=storage,
                                enable_logging=ENABLE_LOGGING)

    ray.get(agent.setup.remote())
    ray.get([m.setup.remote() for m in mcts_instances])
    ray.get(training_instance.setup.remote())

    gameplay_process = agent.gameplay_process.remote()
    mcts_processes = [m.MCTS_process.remote() for m in mcts_instances]
    training_process = training_instance.training_process.remote()

    viewenv = gym.make("MsPacman-v4", render_mode='human')
    viewenv = EpisodicLifeEnv(viewenv)
    try:
        while True:
            obs, _ = viewenv.reset()
            preprocessed_obs = preprocess_frame(obs, resize_shape=IMAGE_RESHAPE)
            done = False
            episode_reward = 0
            frame_stack = initial_frame_stack(preprocessed_obs)
            while not done:
                preprocessed_obs = preprocess_frame(obs, resize_shape=IMAGE_RESHAPE)
                frame_stack = push_frame(frame_stack, preprocessed_obs)
                with torch.no_grad():
                    policy, value = ray.get(network.initial_inference.remote(frame_stack))
                    action = np.random.choice(np.arange(ACTION_SPACE), p=policy)
                    obs, reward, terminated, truncated, _ = viewenv.step(action)
                    done = terminated or truncated
                    episode_reward += reward
            print(f"Reward: {episode_reward}")
            print_memory_diagnostics()
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Exiting gracefully...")
        ray.cancel(gameplay_process)
        for mcts_process in mcts_processes:
            ray.cancel(mcts_process)
        ray.cancel(training_process)