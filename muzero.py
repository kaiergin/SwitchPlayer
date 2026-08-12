import torch
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import numpy as np
import cv2
import random
import threading
import concurrent.futures
import ray
import copy
from pympler import asizeof
import pickle
import queue
from collections import deque
import scalar_categorical
from memory import print_memory_diagnostics
from networks import DynamicsNetwork, PredictionNetwork, RepresentationNetwork
import logging
import os
import uuid
import time
import asyncio
import sys
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
LATENT_CHANNELS = 64

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

def add_dirichlet_noise(prior, epsilon=0.25, alpha=0.03):
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

# -----------------------------
# MCTS Node Definition
# -----------------------------
class MCTSNode:
    def __init__(self, latent, prior, reward=0.0, policy = None, expanded_value = None, is_gameplay = False):
        self.latent = latent            # torch.Tensor (latent state)
        self.prior = prior              # float value
        self.reward = reward            # immediate reward
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
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

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

def get_state_dict(network):
    return {k: (v.cpu() if isinstance(v, torch.Tensor) else v)
        for k, v in network.state_dict().items()}

@ray.remote(max_concurrency=8, num_cpus=2)
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
    
    async def get_max_gameplay(self):
        async with self.gameplay_lock:
            if len(self.gameplay_buffer) == 0:
                return None
            idx, max_item = random.choice(list(enumerate(self.gameplay_buffer)))
            del self.gameplay_buffer[idx]
            return max_item
    

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

    async def get_max_gameplay(self):
        return await self.storage_helper.get_max_gameplay.remote()

#create versioning like storage
@ray.remote(max_concurrency=8, num_gpus=0.05, num_cpus=2)
class NetworkHelper:
    def __init__(self, action_space, device, checkpoint_path = 'checkpoint.pth', enable_logging=False):
        self.device = device
        self.action_space = action_space
        self.representation = RepresentationNetwork(input_channels=12,
                                                    latent_channels=LATENT_CHANNELS).to(device)
        self.dynamics = DynamicsNetwork(latent_channels=LATENT_CHANNELS,
                                        action_space=action_space,
                                        support_size=400,
                                        num_bins=51).to(device)
        self.prediction = PredictionNetwork(latent_channels=LATENT_CHANNELS,
                                            action_space=action_space,
                                            support_size=2000,
                                            num_bins=51).to(device)
        self.rep_optimizer = optim.SGD(self.representation.parameters(), lr=1e-4)
        self.dyn_optimizer = optim.SGD(self.dynamics.parameters(), lr=1e-4)
        self.pred_optimizer = optim.SGD(self.prediction.parameters(), lr=1e-4)
        self.training_steps = 0
        self.configure_logger(enable_logging)
        self.load_checkpoint(checkpoint_path)
        self.version = uuid.uuid4()

    def configure_logger(self, enable_logging):
        # Create and configure a logger for this actor
        self.logger = logging.getLogger("NetworkHelper")
        self.logger.setLevel(logging.DEBUG)
        if enable_logging:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG)
            self.logger.addHandler(console_handler)
    
    def save_checkpoint(self, training_steps, checkpoint_path = 'checkpoint.pth'):
        self.training_steps = training_steps
        checkpoint = {
            'representation_state_dict': self.representation.state_dict(),
            'dynamics_state_dict': self.dynamics.state_dict(),
            'prediction_state_dict': self.prediction.state_dict(),
            'rep_optimizer_state_dict': self.rep_optimizer.state_dict(),
            'dyn_optimizer_state_dict': self.dyn_optimizer.state_dict(),
            'pred_optimizer_state_dict': self.pred_optimizer.state_dict(),
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

    def update_state(self, rep_state_dict, dyn_state_dict, pred_state_dict, rep_opt, dyn_opt, pred_opt, steps, path = 'checkpoint.pth'):
        self.representation.load_state_dict(rep_state_dict)
        self.dynamics.load_state_dict(dyn_state_dict)
        self.prediction.load_state_dict(pred_state_dict)
        self.rep_optimizer.load_state_dict(rep_opt)
        self.dyn_optimizer.load_state_dict(dyn_opt)
        self.pred_optimizer.load_state_dict(pred_opt)
        self.save_checkpoint(steps, checkpoint_path=path)

    def get_state(self):
        return get_state_dict(self.representation), get_state_dict(self.dynamics), get_state_dict(self.prediction),\
                get_state_dict(self.rep_optimizer), get_state_dict(self.dyn_optimizer), get_state_dict(self.pred_optimizer), self.training_steps, self.version

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
                                            support_size=2000,
                                            num_bins=51).to(device)
        if self.load_dynamics:
            self.dynamics = DynamicsNetwork(latent_channels=LATENT_CHANNELS,
                                            action_space=action_space,
                                            support_size=400,
                                            num_bins=51).to(device)
            

    async def setup(self):
        rep_dict, dyn_dict, pred_dict, _, _, _,\
        self.training_steps, self.version = await self.network_helper.get_state.remote()
        self.representation.load_state_dict(rep_dict)
        self.prediction.load_state_dict(pred_dict)
        if self.load_dynamics:
            self.dynamics.load_state_dict(dyn_dict)

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
            rep_dict, dyn_dict, pred_dict, _, _, _,\
            self.training_steps, self.version = await self.network_helper.get_state.remote()
            self.representation.load_state_dict(rep_dict)
            self.prediction.load_state_dict(pred_dict)
            if self.load_dynamics:
                self.dynamics.load_state_dict(dyn_dict)

@ray.remote(max_concurrency=MCTS_BATCHES, num_gpus=0.5, num_cpus=MCTS_BATCHES)
class MCTSHelper:
    def __init__(self, action_space, device, network_ref, storage_ref, enable_logging=False, name="MCTSHelper"):
        self.network_manager = NetworkManager(network_ref, action_space=action_space, device=device)
        self.storage_manager = StorageManager(storage_ref, update_replay=False)
        self.device = device
        self.action_space = action_space
        self.selection_results = asyncio.Queue()
        self.configure_logger(name, enable_logging)

    def configure_logger(self, name, enable_logging):
        # Create and configure a logger for this actor
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        if enable_logging:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG)
            self.logger.addHandler(console_handler)
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
                game_trajectory_tuple = await self.storage_manager.get_max_gameplay()
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
                            "target_value": self.get_value(idx)
                        }
                        trajectory.append(transition)
                else:
                    self.logger.info("starting mcts tree search")
                    for idx, (obs_sample, action_taken, reward, _, _, _) in enumerate(game_trajectory):
                        self.logger.info("Running MCTS index %s out of %s", idx, len(game_trajectory))
                        temperature = get_temperature(self.network_manager.training_steps)
                        improved_policy, value_estimate = await self.run_full_mcts(idx,
                                                                            sim_batches=MCTS_BATCHES,
                                                                            num_simulations=TOTAL_MCTS_SIMS,
                                                                            c_puct=32,
                                                                            discount=0.997,
                                                                            temperature=temperature)
                        # Here we create a training sample using the improved target.
                        mcts_sample = {
                            "obs": obs_sample,
                            "action": action_taken,
                            "reward": reward,
                            "target_policy": improved_policy,
                            "target_value": value_estimate
                        }
                        trajectory.append(mcts_sample)
                
                await self.storage_manager.append_replay_buffer(trajectory)
        except Exception as e:
            self.logger.error("Exception thrown in MCTS_process %s", e)
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
        # would discard its accumulated statistics. Filter both out before the
        # network call rather than after, so we don't pay for them.
        nodes = [n for n in nodes if not n.done and not n.is_expanded]
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
                    child = MCTSNode(
                        latent=next_latents[idx].unsqueeze(0), 
                        prior=policy[a], 
                        reward=rewards[idx].item() if rewards[idx].ndim == 0 else rewards[idx].squeeze().item(),
                        policy=next_policy[idx],
                        expanded_value=estimated_values[idx].item()
                    )
                    node.children[a] = child
                node.is_expanded = True
        
    # -----------------------------
    # Backup Phase: Propagate the leaf value up the simulation path.
    # -----------------------------
    def backup_simulation(self, path, leaf_value, discount=0.997):
        backup_value = leaf_value
        for n in reversed(path):
            n.visit_count += 1
            n.value_sum += backup_value
            backup_value = n.reward + discount * backup_value
            n.virtual_loss = 0.0

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
    
    def add_noise_to_node(self, idx, epsilon = 0.25, alpha = 0.03):
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
            total_visits = sum(child.visit_count for child in node.children.values())
            best_score = -float('inf')
            best_candidates = []  # List to store (action, child) pairs with highest score.
            for a, child in node.children.items():
                u = c_puct * child.prior * np.sqrt(total_visits) / (1 + child.visit_count + child.virtual_loss)
                score = child.q_value() - child.virtual_loss + u
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
        unique_leaves = []
        ids = []
        while not self.selection_results.empty():
            leaf, path = await self.selection_results.get()
            if not leaf.is_expanded and leaf.id not in ids:
                ids.append(leaf.id)
                unique_leaves.append([leaf, path])
        self.batch_expand_nodes([t[0] for t in unique_leaves])

        # Backup: For each simulation, backup the leaf value along its path.
        for leaf, path in unique_leaves:
            if leaf.done:
                leaf_value = 0
            else:
                leaf_value = leaf.expanded_value
            self.backup_simulation(path, leaf_value, discount=discount)

    def drop_non_gameplay_children_and_compute_improvements(self, idx, temperature=1.0):
        children_to_drop = []
        node = self.current_trajectory[idx]
        for action, child in node.children.items():
            if not child.is_gameplay:
                children_to_drop.append(action)
        # Compute improved policy from the root's children.
        visit_counts = np.array([
            node.children[a].visit_count if a in node.children else 0
            for a in range(self.action_space)
        ])
        if visit_counts.sum() > 0:
            improved_policy = apply_temperature(visit_counts, temperature)
        else:
            improved_policy = np.ones(self.action_space) / self.action_space
        root_value = node.q_value()
        for action in children_to_drop:
            del node.children[action]
        del node.latent
        return improved_policy, root_value

    # -----------------------------
    # Full MCTS: Run multiple simulations, batch expand unexpanded leaves,
    # then backup their values.
    # -----------------------------
    async def run_full_mcts(self, index, num_simulations=128, sim_batches=8, c_puct=1.0, discount=0.997, max_depth=MAX_DEPTH, temperature=1.0):
        # Now that expansion is lazy the root has to be expanded explicitly:
        # selection stops at unexpanded nodes, and add_noise_to_node writes the
        # noisy prior through to the children, which must therefore exist.
        root = self.current_trajectory[index]
        if not root.is_expanded:
            self.batch_expand_nodes([root])
        self.add_noise_to_node(index)
        for x in range(max(int(num_simulations // sim_batches), 1)):
            futures = [
                self.selection_task_no_expand(index, c_puct, discount, max_depth)
                for _ in range(sim_batches)
            ]
            await asyncio.gather(*futures)
            await self.expand_selected(discount)
        return self.drop_non_gameplay_children_and_compute_improvements(index, temperature=temperature)

    def get_value(self, idx):
        return self.current_trajectory[idx].q_value()

@ray.remote(num_gpus=0.4, num_cpus=1)
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
                                        support_size=400,
                                        num_bins=51).to(device)
        self.prediction = PredictionNetwork(latent_channels=LATENT_CHANNELS,
                                            action_space=action_space,
                                            support_size=2000,
                                            num_bins=51).to(device)
        self.rep_optimizer = optim.SGD(self.representation.parameters(), lr=1e-4)
        self.dyn_optimizer = optim.SGD(self.dynamics.parameters(), lr=1e-4)
        self.pred_optimizer = optim.SGD(self.prediction.parameters(), lr=1e-4)
        self.configure_logger(enable_logging)
    
    async def setup(self):
        await self.storage_manager.setup()
        await self.load_networks()

    def configure_logger(self, enable_logging):
        # Create and configure a logger for this actor
        self.logger = logging.getLogger("TrainingHelper")
        self.logger.setLevel(logging.DEBUG)
        if enable_logging:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG)
            self.logger.addHandler(console_handler)
        self.logger.info("Initialized TrainingHelper logger.")
    
    async def load_networks(self):
        representation_state_dict, dynamics_state_dict, prediction_state_dict,\
                rep_optimizer_state_dict, dyn_optimizer_state_dict, pred_optimizer_state_dict, self.training_steps, self.network_version = await self.network_helper.get_state.remote()
        self.representation.load_state_dict(representation_state_dict)
        self.dynamics.load_state_dict(dynamics_state_dict)
        self.prediction.load_state_dict(prediction_state_dict)
        self.rep_optimizer.load_state_dict(rep_optimizer_state_dict)
        self.dyn_optimizer.load_state_dict(dyn_optimizer_state_dict)
        self.pred_optimizer.load_state_dict(pred_optimizer_state_dict)

    async def save_networks(self):
        await self.network_helper.update_state.remote(self.representation.state_dict(), self.dynamics.state_dict(), self.prediction.state_dict(),\
                                                self.rep_optimizer.state_dict(), self.dyn_optimizer.state_dict(), self.pred_optimizer.state_dict(), self.training_steps)
        
    
    # -----------------------------
    # Training Step: remote method to update networks
    # -----------------------------
    async def train_step(self, batch, only_dynamics = False):
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
        
        # Zero out the gradients for all optimizers.
        self.rep_optimizer.zero_grad()
        self.pred_optimizer.zero_grad()
        self.dyn_optimizer.zero_grad()
        
        total_loss = 0.0

        # ---------------------
        # Step 0: Initial Inference
        # ---------------------
        init_obs = obs[:, 0]
        latent = self.representation(init_obs)
        policy_logits, _, value_logits = self.prediction(latent)
        
        converted_target_rewards = scalar_categorical.scalar_to_categorical_batch(rewards, self.dynamics.support_size, self.dynamics.num_bins, self.device)

        # Compute losses for the initial step (no reward loss here).
        if not only_dynamics:
            policy_loss_0 = -torch.sum(target_policy[:, 0, :] * F.log_softmax(policy_logits, dim=1), dim=1).mean()
            converted_target_values = scalar_categorical.scalar_to_categorical_batch(target_value, self.prediction.support_size, self.prediction.num_bins, self.device)
            value_loss_0 = -torch.sum(converted_target_values[:, 0, :] * F.log_softmax(value_logits, dim=1), dim=1).mean()
            total_loss += policy_loss_0 + value_loss_0

        # ---------------------
        # Unroll for each subsequent step
        # ---------------------
        for i in range(1, unroll_steps + 1):
            # Use the action from the previous step to roll forward the latent state.
            latent, _, reward_logits = self.dynamics(latent, actions[:, i - 1])
            with torch.no_grad():
                true_latent = self.representation(obs[:, i])
            latent_loss = F.mse_loss(latent, true_latent, reduction="mean")
            
            # Loss for reward prediction. (reward[0] not used since it is reward for entering obs)
            reward_loss = -torch.sum(converted_target_rewards[:, i, :] * F.log_softmax(reward_logits, dim=1), dim=1).mean()

            if not only_dynamics:
                policy_logits, _, value_logits = self.prediction(latent)
                
                # Loss for policy prediction.
                policy_loss = -torch.sum(target_policy[:, i, :] * F.log_softmax(policy_logits, dim=1), dim=1).mean()
                # Loss for value prediction.
                value_loss = -torch.sum(converted_target_values[:, i, :] * F.log_softmax(value_logits, dim=1), dim=1).mean()
                
                total_loss += reward_loss + policy_loss + value_loss + latent_loss
            else:
                total_loss += reward_loss + latent_loss

        # Backpropagate the total loss.
        total_loss.backward()
        
        # Step the optimizers.
        self.rep_optimizer.step()
        self.dyn_optimizer.step()
        if not only_dynamics:
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
        except Exception as e:
            self.logger.error("Exception thrown in training_process %s", e)
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
            "target_value": (B, unroll_steps+1) target values.
        """
        batch_obs = []
        batch_actions = []
        batch_rewards = []
        batch_target_policy = []
        batch_target_value = []
        
        # For each sample, randomly choose a trajectory and a starting index.
        for _ in range(batch_size):
            # Sample a trajectory uniformly.
            traj = random.choice(replay_buffer)
            traj_length = len(traj)
            start_index = np.random.randint(0, traj_length)
            
            # Prepare lists for unroll data.
            obs = []
            actions = []
            rewards = []
            target_policy = []
            target_value = []
            
            # We need unroll_steps+1 target entries (for step 0 and each unroll step).
            for step in range(unroll_steps + 1):
                if start_index + step < traj_length:
                    obs.append(traj[start_index + step]["obs"])
                    node = traj[start_index + step]
                    # Append the target policy and value for each step.
                    target_policy.append(node["target_policy"])
                    target_value.append(node["target_value"])
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
                    if step < unroll_steps:
                        actions.append(0)         # or a default action
            
            batch_obs.append(obs)
            batch_actions.append(actions)
            batch_rewards.append(rewards)
            batch_target_policy.append(target_policy)
            batch_target_value.append(target_value)
        
        # Convert lists to numpy arrays.
        batch_obs = np.array(batch_obs)  # (B, H, W, C)
        batch_actions = np.array(batch_actions)  # (B, unroll_steps)
        batch_rewards = np.array(batch_rewards)  # (B, unroll_steps)
        batch_target_policy = np.array(batch_target_policy)  # (B, unroll_steps+1, action_space)
        batch_target_value = np.array(batch_target_value)  # (B, unroll_steps+1)
        
        return {
            "obs": batch_obs,
            "actions": batch_actions,
            "rewards": batch_rewards,
            "target_policy": batch_target_policy,
            "target_value": batch_target_value
        }

#Agent will be single threaded for real time learning
@ray.remote(max_concurrency=1, num_gpus=0.05, num_cpus=1)
class MuZeroAgent:
    def __init__(self, action_space, device, network_ref, storage_ref, enable_logging=False):
        self.network_manager = NetworkManager(network_ref, action_space=action_space, device=device, load_dynamics=False)
        # The agent only ever writes to the gameplay buffer, so it has no reason
        # to mirror the replay buffer.
        self.storage_manager = StorageManager(storage_ref, update_replay=False)
        self.device = device
        self.action_space = action_space
        self.revised_buffer = queue.Queue()
        self.training_steps = 0
        self.configure_logger(enable_logging)
    
    async def setup(self):
        await self.network_manager.setup()
        await self.storage_manager.setup()

    def configure_logger(self, enable_logging):
        # Create and configure a logger for this actor
        self.logger = logging.getLogger("MuZeroAgent")
        self.logger.setLevel(logging.DEBUG)
        if enable_logging:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG)
            self.logger.addHandler(console_handler)
        self.logger.info("Initialized actor logger.")

    async def gameplay_process(self):
        try:
            if not hasattr(np, 'bool8'):
                np.bool8 = np.bool_
            env = gym.make("MsPacman-v4")
            env = EpisodicLifeEnv(env)
            while True:
                # Refresh weights between episodes. Without this the behaviour
                # policy stayed frozen at whatever setup() loaded, so self-play
                # kept generating data from the initial network forever and the
                # policy improvement loop never closed. (Refreshing per episode
                # rather than per step keeps acting real-time.)
                await self.network_manager.update_if_needed()
                self.logger.debug("Starting new game")
                obs, info = await asyncio.to_thread(env.reset)
                preprocessed_obs = preprocess_frame(obs, resize_shape=IMAGE_RESHAPE)
                prev_reward = 0.0
                done = False
                episode_reward = 0
                game_trajectory = []  # store tuples of (obs, latent, action)
                frame_stack = np.repeat(preprocessed_obs, 4, axis=0)
                
                # Play one episode.
                while not done:
                    preprocessed_obs = preprocess_frame(obs, resize_shape=IMAGE_RESHAPE)
                    frame_stack = np.concatenate([frame_stack[3:, :, :], preprocessed_obs], axis=0)
                    with torch.no_grad():
                        policy, value = self.network_manager.initial_inference(frame_stack)
                    action = np.random.choice(np.arange(self.action_space), p=policy)
                    obs, reward, done, _, _ = await asyncio.to_thread(env.step, action)
                    # obs, chosen action, reward for entering obs
                    reward = min(reward, 400)
                    game_trajectory.append((frame_stack, action, prev_reward))
                    episode_reward += reward
                    prev_reward = reward
                await self.storage_manager.append_gameplay_buffer((game_trajectory, episode_reward))
        except Exception as e:
            self.logger.error("Exception thrown in gameplay_process %s", e)
        self.logger.info("Gameplay process exiting")


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


# -----------------------------
# Main Loop Example with Full MCTS Sample Generation
# -----------------------------
if __name__ == '__main__':
    ray.init(include_dashboard=True, dashboard_port=8265)#log_to_driver=True)
    ACTION_SPACE = 5
    ENABLE_LOGGING = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = NetworkHelper.remote(
                                action_space=ACTION_SPACE,
                                device=device,
                                enable_logging=ENABLE_LOGGING)
    storage = StorageHelper.remote()
    agent = MuZeroAgent.remote(
                                action_space=ACTION_SPACE,
                                device=device,
                                network_ref=network,
                                storage_ref=storage,
                                enable_logging=ENABLE_LOGGING)
    mcts_instance = MCTSHelper.remote(
                                action_space=ACTION_SPACE,
                                device=device,
                                network_ref=network,
                                storage_ref=storage,
                                enable_logging=ENABLE_LOGGING,
                                name='process1')
    # mcts_instance2 = MCTSHelper.remote(
    #                             action_space=ACTION_SPACE,
    #                             device=device,
    #                             network_ref=network,
    #                             storage_ref=storage,
    #                             enable_logging=ENABLE_LOGGING,
    #                             name='process2')
    training_instance = TrainingHelper.remote(
                                action_space=ACTION_SPACE,
                                device=device,
                                network_ref=network,
                                storage_ref=storage,
                                enable_logging=ENABLE_LOGGING)

    ray.get(agent.setup.remote())
    ray.get(mcts_instance.setup.remote())
    # ray.get(mcts_instance2.setup.remote())
    ray.get(training_instance.setup.remote())

    gameplay_process = agent.gameplay_process.remote()
    MCTS_process = mcts_instance.MCTS_process.remote()
    # MCTS_process2 = mcts_instance2.MCTS_process.remote()
    training_process = training_instance.training_process.remote()

    viewenv = gym.make("MsPacman-v4", render_mode='human')
    viewenv = EpisodicLifeEnv(viewenv)
    try:
        while True:
            obs, _ = viewenv.reset()
            preprocessed_obs = preprocess_frame(obs, resize_shape=IMAGE_RESHAPE)
            done = False
            episode_reward = 0
            frame_stack = np.repeat(preprocessed_obs, 4, axis=0)
            while not done:
                preprocessed_obs = preprocess_frame(obs, resize_shape=IMAGE_RESHAPE)
                frame_stack = np.concatenate([frame_stack[3:, :, :], preprocessed_obs], axis=0)
                with torch.no_grad():
                    policy, value = ray.get(network.initial_inference.remote(frame_stack))
                    action = np.random.choice(np.arange(ACTION_SPACE), p=policy)
                    obs, reward, done, _, _ = viewenv.step(action)
                    episode_reward += reward
            print(f"Reward: {episode_reward}")
            print_memory_diagnostics()
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Exiting gracefully...")
        ray.cancel(gameplay_process)
        ray.cancel(MCTS_process)
        # ray.cancel(MCTS_process2)
        ray.cancel(training_process)