import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import numpy as np
import cv2
import random
from collections import deque
import threading
import concurrent.futures
import ray
import hashlib
import copy
import psutil
from pympler import asizeof

if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

# Set this to number of threads your processor has
NUM_PROCESSES = 16

def print_memory_diagnostics():
    process = psutil.Process()
    mem_info = process.memory_info()
    # mem_info.rss: Resident Set Size (physical memory)
    # mem_info.vms: Virtual Memory Size
    vm = psutil.virtual_memory()
    print(f"Process Memory: RSS = {mem_info.rss / (1024 * 1024):.2f} MB, VMS = {mem_info.vms / (1024 * 1024):.2f} MB")
    print(f"System Memory: Total = {vm.total / (1024 * 1024):.2f} MB, Available = {vm.available / (1024 * 1024):.2f} MB, Usage = {vm.percent}%")


# -----------------------------
# Basic Residual Block
# -----------------------------
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

# -----------------------------
# Representation Network
# -----------------------------
class RepresentationNetwork(nn.Module):
    def __init__(self, input_channels=3, latent_channels=128):
        super(RepresentationNetwork, self).__init__()
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(input_channels, 64,
                               kernel_size=3, stride=2, padding=1)
        self.block1 = BasicBlock(64, 64)
        self.block2 = BasicBlock(64, 64)
        self.conv2 = nn.Conv2d(64, latent_channels,
                               kernel_size=3, stride=2, padding=1)
        self.block3 = BasicBlock(latent_channels, latent_channels)
        self.block4 = BasicBlock(latent_channels, latent_channels)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.relu(self.conv2(x))
        x = self.block3(x)
        x = self.block4(x)
        return x  # latent state of shape (batch, latent_channels, H', W')

# -----------------------------
# Dynamics Network
# -----------------------------
class DynamicsNetwork(nn.Module):
    def __init__(self, latent_channels=128, action_space=9):
        super(DynamicsNetwork, self).__init__()
        self.action_space = action_space
        self.action_embedding = nn.Embedding(action_space, latent_channels)
        self.conv1 = nn.Conv2d(latent_channels * 2, latent_channels,
                               kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=False)
        self.block1 = BasicBlock(latent_channels, latent_channels)
        self.block2 = BasicBlock(latent_channels, latent_channels)
        self.reward_conv = nn.Conv2d(latent_channels, 1,
                                     kernel_size=3, stride=1, padding=1)
        # Assuming latent spatial size is 8x8:
        self.reward_fc = nn.Linear(8 * 8, 1)
        
    def forward(self, latent, action):
        batch_size = latent.size(0)
        act_emb = self.action_embedding(action)  # shape: (batch, latent_channels)
        act_emb = act_emb.contiguous().view(batch_size, latent.size(1), 1, 1)
        act_emb = act_emb.expand(-1, -1, latent.size(2), latent.size(3))
        x = torch.cat([latent, act_emb], dim=1)
        x = self.relu(self.conv1(x))
        x = self.block1(x)
        x = self.block2(x)
        next_latent = x
        r = self.relu(self.reward_conv(x))
        # Force reward to a fixed spatial size (e.g. 8x8)
        r = F.adaptive_avg_pool2d(r, (8, 8))
        r = r.contiguous().view(batch_size, -1)
        reward = torch.tanh(self.reward_fc(r))
        return next_latent, reward

# -----------------------------
# Prediction Network
# -----------------------------
class PredictionNetwork(nn.Module):
    def __init__(self, latent_channels=128, action_space=9):
        super(PredictionNetwork, self).__init__()
        self.block = BasicBlock(latent_channels, latent_channels)
        self.relu = nn.ReLU(inplace=False)
        # Policy head:
        self.policy_conv = nn.Conv2d(latent_channels, 32, kernel_size=1)
        self.policy_fc = nn.Linear(32 * 8 * 8, action_space)
        # Value head:
        self.value_conv = nn.Conv2d(latent_channels, 32, kernel_size=1)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 1)
        
    def forward(self, latent):
        x = self.block(latent)
        x = self.relu(x)
        # Policy branch:
        p = self.relu(self.policy_conv(x))
        # Force spatial size to 8x8:
        p = F.adaptive_avg_pool2d(p, (8, 8))
        p = p.contiguous().view(p.size(0), -1)
        policy_logits = self.policy_fc(p)
        
        # Value branch:
        v = self.relu(self.value_conv(x))
        v = F.adaptive_avg_pool2d(v, (8, 8))
        v = v.contiguous().view(v.size(0), -1)
        v = self.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))
        return policy_logits, value

def prepare_batch(replay_buffer, batch_size, unroll_steps, action_space):
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
         "obs": (B, H, W, C) initial observations,
         "actions": (B, unroll_steps) sequence of actions,
         "rewards": (B, unroll_steps) sequence of rewards,
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
        
        # Get initial observation.
        init_obs = traj[start_index]["obs"]
        
        # Prepare lists for unroll data.
        actions = []
        rewards = []
        target_policy = []
        target_value = []
        
        # We need unroll_steps+1 target entries (for step 0 and each unroll step).
        for step in range(unroll_steps + 1):
            if start_index + step < traj_length:
                transition = traj[start_index + step]
                # Append the target policy and value for each step.
                target_policy.append(transition["target_policy"])
                target_value.append(transition["target_value"])
                if step < unroll_steps:
                    # For actions and rewards, only collect for unroll_steps.
                    actions.append(transition["action"])
                    rewards.append(transition["reward"])
            else:
                # If the trajectory doesn't have enough transitions, pad with zeros.
                if step < unroll_steps:
                    actions.append(0)         # or a default action
                    rewards.append(0.0)
                target_policy.append(np.zeros(action_space))
                target_value.append(0.0)
        
        batch_obs.append(init_obs)
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

def hash_latent(latent, precision=3):
    """
    Compute a hash string for a latent tensor.
    
    Parameters:
      latent (torch.Tensor): The latent state (expected to be on GPU or CPU).
      precision (int): Number of decimals to round to, to reduce sensitivity to small changes.
      
    Returns:
      A hexadecimal MD5 hash string.
    """
    # Ensure the tensor is on CPU and detached from the graph.
    arr = latent.cpu().detach().numpy()
    # Round the array to reduce numerical precision issues.
    arr_rounded = np.round(arr, decimals=precision)
    # Convert the array to bytes.
    arr_bytes = arr_rounded.tobytes()
    # Compute and return an MD5 hash.
    return hashlib.md5(arr_bytes).hexdigest()

# -----------------------------
# MCTS Node Definition
# -----------------------------
class MCTSNode:
    def __init__(self, latent, prior, reward=0.0, policy = None, expanded_value = None):
        self.latent = latent            # torch.Tensor (latent state)
        self.prior = prior              # float value
        self.reward = reward            # immediate reward
        self.children = {}              # dict: action -> child node
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False
        self.policy = policy              # cached policy (numpy array)
        self.latent_hash = hash_latent(latent)
        self.lock = threading.Lock()
        self.expanded_value = expanded_value
        self.virtual_loss = 0.0

    def q_value(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

    def __hash__(self):
        # Use the hash of the latent hash string.
        return hash(self.latent_hash)

    def __eq__(self, other):
        # Two nodes are equal if their latent hash strings are equal.
        if isinstance(other, MCTSNode):
            return self.latent_hash == other.latent_hash
        return False
    
    def __getstate__(self):
        """
        Remove the lock from the state before pickling.
        """
        state = self.__dict__.copy()
        # Remove the lock because threading.Lock() is not picklable.
        if "lock" in state:
            del state["lock"]
        return state

    def __setstate__(self, state):
        """
        Reinitialize the lock when unpickling.
        """
        self.__dict__.update(state)
        self.lock = threading.Lock()


# -----------------------------
# Remote MCTS Selection Task
# -----------------------------
@ray.remote
class MCTSHelper:
    def __init__(self, path):
        self.path = path

    def selection_task_no_expand(self, start, c_puct, discount, max_depth=float('inf')):
        """
        Traverse the tree from the given root until a leaf is reached.
        This function is run remotely so it works on a serialized copy of the tree.
        It ensures that the latent tensors are on CPU before returning.
        """
        path = []
        node = self.path[start]
        cumulative_discount = 1.0
        while len(path) < max_depth:
            if not node.children:
                break  # reached a leaf
            total_visits = sum(child.visit_count for child in node.children.values())
            best_score = -float('inf')
            best_candidates = []  # List to store (action, child) pairs with highest score.
            for a, child in node.children.items():
                u = c_puct * child.prior * np.sqrt(total_visits) / (1 + max(child.visit_count, 0))
                score = child.q_value() - child.virtual_loss + u
                if score > best_score:
                    best_score = score
                    best_candidates = [(a, child)]
                elif score == best_score:
                    best_candidates.append((a, child))
            # Randomly select one of the best candidates in case of a tie.
            chosen_action, chosen_child = random.choice(best_candidates)
            with chosen_child.lock:
                chosen_child.virtual_loss += 1
            path.append(chosen_action)
            node = chosen_child
            cumulative_discount *= discount
        return (start, path, cumulative_discount)

class MuZeroAgent:
    def __init__(self, action_space, device):
        self.device = device
        self.action_space = action_space
        self.latent_channels = 128

        self.representation = RepresentationNetwork(input_channels=3,
                                                    latent_channels=self.latent_channels).to(device)
        self.dynamics = DynamicsNetwork(latent_channels=self.latent_channels,
                                        action_space=action_space).to(device)
        self.prediction = PredictionNetwork(latent_channels=self.latent_channels,
                                            action_space=action_space).to(device)
        self.representation_old = copy.deepcopy(self.representation)
        self.prediction_old = copy.deepcopy(self.prediction)
        self.rep_optimizer = optim.Adam(self.representation.parameters(), lr=1e-5)
        self.dyn_optimizer = optim.Adam(self.dynamics.parameters(), lr=1e-5)
        self.pred_optimizer = optim.Adam(self.prediction.parameters(), lr=1e-5)

        self.mcts_hash_table = {}

    def initial_inference(self, observation):
        obs = cv2.resize(observation, (95, 95))
        obs = np.transpose(obs, (2, 0, 1))
        converted_obs = torch.tensor(obs, dtype=torch.float32, device=self.device) / 255.0
        converted_obs = converted_obs.unsqueeze(0)
        latent = self.representation_old(converted_obs)
        policy_logits, value = self.prediction_old(latent)
        return latent, policy_logits, value, obs

    def recurrent_inference(self, latent, action):
        a_tensor = torch.tensor([action], dtype=torch.long, device=self.device)
        latent, reward = self.dynamics(latent, a_tensor)
        policy_logits, value = self.prediction_old(latent)
        return latent, reward, policy_logits, value

    def update_representation(self):
        self.mcts_hash_table = {}
        self.representation_old.load_state_dict(self.representation.state_dict())

    def update_prediction(self):
        self.prediction_old.load_state_dict(self.prediction.state_dict())

    def soft_update_prediction(self, tau):
        """
        Perform a soft update of the target network parameters.
        target_net: the network to be updated (e.g. prediction_old)
        source_net: the live network (e.g. prediction)
        tau: mixing coefficient (0 < tau <= 1)
        """
        for target_param, param in zip(self.prediction_old.parameters(), self.prediction.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    # -----------------------------
    # Training Step: remote method to update networks
    # -----------------------------
    def train_step(self, batch):
        """
        Multi-step unrolling training step.
        
        Expects a batch dictionary with keys:
        "obs": (B, H, W, C) uint8 images,
        "actions": (B, unroll_steps) ints,
        "rewards": (B, unroll_steps) floats,
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
        latent = self.representation(obs)
        policy_logits, value = self.prediction(latent)
        
        # Compute losses for the initial step (no reward loss here).
        policy_loss_0 = -torch.sum(target_policy[:, 0, :] * F.log_softmax(policy_logits, dim=1), dim=1).mean()
        value_loss_0 = F.mse_loss(value.squeeze(), target_value[:, 0])
        total_loss += policy_loss_0 + value_loss_0

        # ---------------------
        # Unroll for each subsequent step
        # ---------------------
        for i in range(1, unroll_steps + 1):
            # Use the action from the previous step to roll forward the latent state.
            latent, reward_prediction = self.dynamics(latent, actions[:, i - 1])
            policy_logits, value = self.prediction(latent)
            
            # Loss for reward prediction.
            reward_loss = F.mse_loss(reward_prediction.squeeze(), rewards[:, i - 1])
            # Loss for policy prediction.
            policy_loss = -torch.sum(target_policy[:, i, :] * F.log_softmax(policy_logits, dim=1), dim=1).mean()
            # Loss for value prediction.
            value_loss = F.mse_loss(value.squeeze(), target_value[:, i])
            
            total_loss += reward_loss + policy_loss + value_loss

        # Backpropagate the total loss.
        total_loss.backward()
        
        # Step the optimizers.
        self.rep_optimizer.step()
        self.pred_optimizer.step()
        self.dyn_optimizer.step()
        
        return total_loss.item()

    
    # -----------------------------
    # Batch expansion helper function
    # -----------------------------
    def batch_expand_nodes(self, nodes, batch_size=32):
        """
        Given a list of MCTSNode objects that need expansion, expand them in batches.
        
        Args:
        nodes: list of MCTSNode objects to expand.
        batch_size: number of nodes to process in a single batch. (actual batch size is batch_size*self.action_space)
        """
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
                next_latents, rewards = self.dynamics(latent_batch, action_tensor)
                batch_policy_logits, estimated_values = self.prediction(next_latents)
                next_policy = F.softmax(batch_policy_logits, dim=1).cpu().numpy()
            
            # Update each node in the chunk.
            for j, node in enumerate(chunk):
                start = j * A
                end = start + A
                # Assume node.policy is already a numpy array of shape (A,)
                policy = node.policy
                node.children = {}
                for a in range(A):
                    idx = start + a
                    # Create the child node with latent state moved to CPU.
                    child = MCTSNode(
                        latent=next_latents[idx].unsqueeze(0).cpu(), 
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
            backup_value = n.reward + discount * backup_value
            n.visit_count += 1
            n.value_sum += backup_value
            n.virtual_loss = 0.0

    # -----------------------------
    # Full MCTS: Run multiple simulations, batch expand unexpanded leaves,
    # then backup their values.
    # -----------------------------
    def run_full_mcts(self, node, helper, idx, num_simulations=128, num_processes=8, c_puct=1.0, discount=0.997, max_depth=float('inf')):
        selection_results = []
        for _ in range(max(int(num_simulations // num_processes), 1)):
            futures = [
                helper.selection_task_no_expand.remote(idx, c_puct, discount, max_depth)
                for _ in range(num_processes)
            ]
            selection_results += ray.get(futures)

        # Batch expansion: collect unique leaves that have not been expanded.
        unique_leaves = set()
        for start, path, cum_discount in selection_results:
            leaf = node
            new_path = [node]
            for a in path:
                leaf = leaf.children[a]
                new_path.append(leaf)
            if not leaf.is_expanded:
                unique_leaves.add((leaf, tuple(new_path)))
        unique_leaves = list(unique_leaves)
        self.batch_expand_nodes([t[0] for t in unique_leaves])

        # Backup: For each simulation, backup the leaf value along its path.
        for leaf, path in unique_leaves:
            leaf_value = leaf.expanded_value
            self.backup_simulation(path, leaf_value, discount=discount)

        # Compute improved policy from the root's children.
        visit_counts = np.array([node.children[a].visit_count for a in range(self.action_space)])
        if visit_counts.sum() > 0:
            improved_policy = visit_counts / visit_counts.sum()
        else:
            improved_policy = np.ones(self.action_space) / self.action_space
        root_value = node.q_value()
        return improved_policy, root_value
    
    def create_tree_from_game(self, game_trajectory):
        root_node = None
        prev_node = None
        prev_action_taken = None
        path = []
        for (obs_sample, latent, action_taken, policy, reward, expanded_value) in game_trajectory:
            new_node = MCTSNode(latent=latent.cpu(), prior=1.0, reward=reward)
            # If new node is not already in hash table, add it
            if new_node.latent_hash not in self.mcts_hash_table:
                self.mcts_hash_table[new_node.latent_hash] = new_node
            else: # Else use the previously found node
                new_node = self.mcts_hash_table[new_node.latent_hash]
            new_node.policy = policy
            new_node.expanded_value = expanded_value
            if prev_node != None:
                new_node.prior = prev_node.policy[prev_action_taken]
                prev_node.children[prev_action_taken] = new_node
            else:
                root_node = new_node
            path.append(new_node)
            prev_node = new_node
            prev_action_taken = action_taken
        
        self.batch_expand_nodes(path)

        self.backup_simulation(path, 0)
        return path

    '''
    def reanalyze_trajectory(self, trajectory, batch_size=128, mcts_simulations=128, c_puct=1.0, discount=0.997):
        """
        Reanalyzes a single trajectory using the current agent parameters.
        
        Args:
            trajectory: A list of transitions (each a dict with keys "obs", "action", "reward", "target_policy", "target_value").
            mcts_simulations: Number of simulations to run during MCTS.
            c_puct: The exploration constant for MCTS.
            discount: Discount factor for value estimation.
        
        Returns:
            A new trajectory with updated target_policy and target_value for each transition.
        """

        trajectory_obs = [transition["obs"] for transition in trajectory]
        num_transitions = len(trajectory)
        for start in range(0, num_transitions, batch_size):
            end = min(start + batch_size, num_transitions)
            batch_obs = torch.as_tensor(np.array(trajectory_obs[start:end]), dtype=torch.float32, device=self.device)
            with torch.no_grad()
                latent_batch = self.representation(batch_obs)
                policy_batch, value_batch = self.prediction(batch_obs)
            for i in range(start, end):
                trajectory[i]["latent"] = latent_batch[i - start].cpu()

        path = self.create_tree_from_game(trajectory)
        helper = MCTSHelper.remote(path)
        for transition, node in zip(trajectory, path):
            improved_policy, value_estimate = self.run_full_mcts(node,
                                                                    helper,
                                                                    num_simulations=mcts_simulations, 
                                                                    c_puct=c_puct, 
                                                                    discount=discount)
            transition["target_policy"] = improved_policy
            transition["target_value"] = value_estimate

    # If you want to reanalyze the entire replay buffer (which is assumed to be a deque or list of trajectories):
    def reanalyze_replay_buffer(self, replay_buffer, mcts_simulations=128, c_puct=1.0, discount=0.997):
        """
        Reanalyzes each trajectory in the replay buffer using the current agent.
        
        Args:
            replay_buffer: A list (or deque) of trajectories.
            mcts_simulations, c_puct, discount: Hyperparameters for MCTS.
        
        Returns:
            A new replay buffer with updated trajectories.
        """
        print("Reanalyzing replay buffer")
        for trajectory in replay_buffer:
            self.reanalyze_trajectory(trajectory, mcts_simulations, c_puct, discount)
    '''


# -----------------------------
# Main Loop Example with Full MCTS Sample Generation
# -----------------------------
if __name__ == '__main__':
    ray.init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("MsPacman-v4")
    action_space = env.action_space.n
    agent = MuZeroAgent(action_space=action_space, device=device)
    
    num_episodes = 6000
    num_episodes_until_mcts = 8
    replay_buffer = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        prev_reward = 0.0
        done = False
        episode_reward = 0
        game_trajectory = []  # store tuples of (obs, latent, action)
        
        # Play one episode.
        while not done:
            with torch.no_grad():
                latent, policy_logits, value, converted_obs = agent.initial_inference(obs)
            policy = F.softmax(policy_logits, dim=1).cpu().detach().numpy()[0]
            action = np.random.choice(np.arange(action_space), p=policy)
            game_trajectory.append((converted_obs, latent, action, policy, prev_reward, value.item()))
            
            obs, reward, done, _, _ = env.step(action)
            episode_reward += reward
            prev_reward = reward
            #env.render()
        
        print(f"Episode {episode} reward: {episode_reward}")
        print_memory_diagnostics()
        
        path = agent.create_tree_from_game(game_trajectory)

        trajectory = []
        # Store one-step transition (for a basic training sample)
        if episode < num_episodes_until_mcts:
            for (obs_sample, latent, action_taken, policy, reward, expanded_value), node in zip(game_trajectory, path):
                transition = {
                    "obs": obs_sample,
                    "action": action_taken,
                    "reward": reward,
                    "target_policy": np.ones(action_space) / action_space,
                    "target_value": node.q_value()
                }
                trajectory.append(transition)
            replay_buffer.append(trajectory)

        if episode >= num_episodes_until_mcts:
            print("starting mcts tree search")
            helper = MCTSHelper.remote(path)
            idx = 0
            for (obs_sample, latent, action_taken, policy, reward, expanded_value), node in zip(game_trajectory, path):
                improved_policy, value_estimate = agent.run_full_mcts(node, helper, idx, num_processes=NUM_PROCESSES, num_simulations=16, c_puct=1.5, discount=0.997)
                # Here we create an additional training sample using the improved target.
                mcts_sample = {
                    "obs": obs_sample,
                    "action": action_taken,
                    "reward": node.reward,  # For illustration; in practice use multi-step return.
                    "target_policy": improved_policy,
                    "target_value": value_estimate
                }
                trajectory.append(mcts_sample)
                idx += 1
            replay_buffer.append(trajectory)
            ray.kill(helper)
        
        print(f"Replay buffer size: {sum([len(traj) for traj in replay_buffer])}")
        print(f"Size of tree {asizeof.asizeof(agent.mcts_hash_table)}")
        
        for _ in range(max(2 * episode, 20)):
            batch = prepare_batch(replay_buffer, 128, 5, action_space)
            loss = agent.train_step(batch)
            print(f"Training loss: {loss}")
        
        agent.soft_update_prediction(0.05)
        if episode % 20 == 0 and episode > 1:
            agent.update_representation()
            #agent.reanalyze_replay_buffer(replay_buffer, mcts_simulations=16)
        if len(replay_buffer) > 20:
            replay_buffer.pop(0)
