import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import numpy as np
import cv2
import random
import threading
import concurrent.futures
import ray
import copy
import psutil
from pympler import asizeof
import pickle
import queue
from collections import deque
import scalar_categorical

if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

# Set this to number of threads your processor has
NUM_PROCESSES = 16
MAX_DEPTH = 5
IMAGE_RESHAPE = (84,84)

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

# Helper functions to create the support vector and convert logits to a scalar.
def create_support(support_size, num_bins, device):
    """
    Creates a support vector from -support_size to support_size with num_bins discrete values.
    """
    return torch.linspace(0, support_size, num_bins, device=device)

def scalar_from_logits(logits, support):
    """
    Converts categorical logits into a scalar by applying softmax, then computing the expected value.
    """
    probs = F.softmax(logits, dim=1)
    expected_value = torch.sum(probs * support.unsqueeze(0).expand_as(probs), dim=1, keepdim=True)
    return expected_value

# Assume BasicBlock is defined as in your original code.

# Revised PredictionNetwork with categorical value representation and policy clipping.
class PredictionNetwork(nn.Module):
    def __init__(self, latent_channels=128, action_space=9, support_size=10, num_bins=21):
        super(PredictionNetwork, self).__init__()
        self.block = BasicBlock(latent_channels, latent_channels)
        self.relu = nn.ReLU(inplace=False)
        
        # Policy head:
        self.policy_conv = nn.Conv2d(latent_channels, 32, kernel_size=1)
        self.policy_fc = nn.Linear(32 * 8 * 8, action_space)
        
        # Value head now outputs categorical logits instead of a single scalar.
        self.value_conv = nn.Conv2d(latent_channels, 32, kernel_size=1)
        self.value_fc = nn.Linear(32 * 8 * 8, num_bins)
        
        self.support_size = support_size
        self.num_bins = num_bins
        
    def forward(self, latent):
        x = self.block(latent)
        x = self.relu(x)
        
        # Policy branch:
        p = self.relu(self.policy_conv(x))
        p = F.adaptive_avg_pool2d(p, (8, 8))
        p = p.contiguous().view(p.size(0), -1)
        policy_logits = self.policy_fc(p)
        # Convert logits to probabilities and clip for numerical stability.
        policy_probs = F.softmax(policy_logits, dim=1)
        policy_probs = torch.clamp(policy_probs, min=1e-6, max=1.0)
        
        # Value branch:
        v = self.relu(self.value_conv(x))
        v = F.adaptive_avg_pool2d(v, (8, 8))
        v = v.contiguous().view(v.size(0), -1)
        value_logits = self.value_fc(v)  # shape: (batch_size, num_bins)
        support = create_support(self.support_size, self.num_bins, latent.device)
        value = scalar_from_logits(value_logits, support)
        
        return policy_probs, value, value_logits

# Revised DynamicsNetwork with categorical reward representation.
class DynamicsNetwork(nn.Module):
    def __init__(self, latent_channels=128, action_space=9, support_size=10, num_bins=21):
        super(DynamicsNetwork, self).__init__()
        self.action_space = action_space
        self.support_size = support_size
        self.num_bins = num_bins
        self.action_embedding = nn.Embedding(action_space, latent_channels)
        self.conv1 = nn.Conv2d(latent_channels * 2, latent_channels,
                               kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=False)
        self.block1 = BasicBlock(latent_channels, latent_channels)
        self.block2 = BasicBlock(latent_channels, latent_channels)
        
        # Reward head now outputs categorical logits.
        self.reward_conv = nn.Conv2d(latent_channels, 32, kernel_size=1)
        self.reward_fc = nn.Linear(32 * 8 * 8, num_bins)
        
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
        
        # Reward branch:
        r = self.relu(self.reward_conv(x))
        r = F.adaptive_avg_pool2d(r, (8, 8))
        r = r.contiguous().view(batch_size, -1)
        reward_logits = self.reward_fc(r)
        support = create_support(self.support_size, self.num_bins, latent.device)
        reward = scalar_from_logits(reward_logits, support)
        
        return next_latent, reward, reward_logits

def print_tree(node, depth=0):
    if depth > 6:
        return
    indent = "    " * depth  # 4 spaces per depth level.
    print(f"{indent}Node(prior={node.prior}, reward={node.reward}, visits={node.visit_count}, value={node.expanded_value}, depth={depth})")
    for action, child in node.children.items():
        print(f"{indent}  Action {action}:")
        print_tree(child, depth + 1)

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
                # If the trajectory doesn't have enough transitions, pad with zeros.
                target_policy.append(np.ones(action_space) / action_space)
                target_value.append(0.0)
                rewards.append(0.0)
                if step < unroll_steps:
                    actions.append(0)         # or a default action
        
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
        
        self.is_expanded = False
        self.policy = policy              # cached policy (numpy array)
        self.lock = threading.Lock()
        self.expanded_value = expanded_value
        self.virtual_loss = 0.0
        self.is_gameplay = is_gameplay

    def q_value(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

# -----------------------------
# Full MCTS: Run multiple simulations, batch expand unexpanded leaves,
# then backup their values.
# -----------------------------
def run_full_mcts(agent, index, num_simulations=128, num_processes=8, c_puct=1.0, discount=0.997, max_depth=MAX_DEPTH, temperature=1.0):
    ray.get(agent.add_noise_to_node.remote(index))
    for x in range(max(int(num_simulations // num_processes), 1)):
        futures = [
            agent.selection_task_no_expand.remote(index, c_puct, discount, max_depth)
            for _ in range(num_processes)
        ]
        ray.get(futures)
        ray.get(agent.expand_selected.remote(discount))
    return ray.get(agent.drop_non_gameplay_children_and_compute_improvements.remote(index, temperature=temperature))

def reanalyze_trajectory(agent, trajectory, mcts_simulations=128, c_puct=1.0, discount=0.997, temperature=1.0):
    """
    Reanalyzes a single trajectory using the current agent parameters.
    
    Args:
        trajectory: A list of transitions (each a dict with keys "obs", "action", "reward", "target_policy", "target_value").
        mcts_simulations: Number of simulations to run during MCTS.
        c_puct: The exploration constant for MCTS.
        discount: Discount factor for value estimation.
    """

    remade_traj = []
    for x in range(len(trajectory)):
        with torch.no_grad():
            latent, policy, value = ray.get(agent.initial_inference.remote(trajectory[x]["obs"]))
        remade_traj.append((trajectory[x]["obs"], latent, trajectory[x]["action"], policy, trajectory[x]["reward"], value))

    ray.get(agent.create_tree_from_game.remote(remade_traj))
    for transition, idx in zip(trajectory, list(range(len(remade_traj)))):
        print("Running MCTS index", idx, "out of", len(remade_traj))
        improved_policy, value_estimate = run_full_mcts(agent, idx,
                                                            num_simulations=mcts_simulations, 
                                                            c_puct=c_puct, 
                                                            discount=discount,
                                                            temperature=temperature)
        transition["target_policy"] = improved_policy
        transition["target_value"] = value_estimate

# If you want to reanalyze the entire replay buffer (which is assumed to be a deque or list of trajectories):
def reanalyze_replay_buffer(agent, replay_buffer, mcts_simulations=128, c_puct=1.0, discount=0.997, temperature=1.0):
    """
    Reanalyzes each trajectory in the replay buffer using the current agent.
    
    Args:
        replay_buffer: A list (or deque) of trajectories.
        mcts_simulations, c_puct, discount: Hyperparameters for MCTS.
    """
    print("Reanalyzing replay buffer")
    for trajectory in replay_buffer:
        print("starting next trajectory")
        reanalyze_trajectory(agent, trajectory, mcts_simulations, c_puct, discount, temperature)

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

def get_temperature(epoch, initial_temp=1.0, final_temp=0.1, decay_epochs=100):
    """
    Linearly decays temperature from initial_temp to final_temp over decay_epochs.
    If epoch exceeds decay_epochs, return final_temp.
    """
    if epoch >= decay_epochs:
        return final_temp
    return initial_temp - (initial_temp - final_temp) * (epoch / decay_epochs)

@ray.remote
class MuZeroAgent:
    def __init__(self, action_space, device):
        self.device = device
        self.action_space = action_space
        self.latent_channels = 128

        self.representation = RepresentationNetwork(input_channels=4,
                                                    latent_channels=self.latent_channels).to(device)
        self.dynamics = DynamicsNetwork(latent_channels=self.latent_channels,
                                        action_space=action_space,
                                        support_size=800,
                                        num_bins=51).to(device)
        self.prediction = PredictionNetwork(latent_channels=self.latent_channels,
                                            action_space=action_space,
                                            support_size=3500,
                                            num_bins=51).to(device)
        self.representation_old = copy.deepcopy(self.representation)
        self.prediction_old = copy.deepcopy(self.prediction)
        self.rep_optimizer = optim.Adam(self.representation.parameters(), lr=1e-4)
        self.dyn_optimizer = optim.Adam(self.dynamics.parameters(), lr=1e-4)
        self.pred_optimizer = optim.Adam(self.prediction.parameters(), lr=1e-4)
        self.selection_results = queue.Queue()

    def initial_inference(self, obs):
        converted_obs = torch.tensor(obs, dtype=torch.float32, device=self.device) / 255.0
        converted_obs = converted_obs.unsqueeze(0)
        latent = self.representation_old(converted_obs)
        policy_logits, value, _ = self.prediction_old(latent)
        policy_logits = F.softmax(policy_logits, dim=1).cpu().detach().numpy()[0]
        latent = latent.cpu()
        return latent, policy_logits, value.item()
    
    #not working?
    def batch_inference(self, batch_obs, old = False):
        converted = torch.tensor(batch_obs, dtype=torch.float32, device=self.device) / 255.0
        if old:
            batch_latents = self.representation_old(coverted)
            batch_policy, batch_value, _ = self.prediction_old(batch_latents)
        else:
            batch_latents = self.representation(converted)
            batch_policy, batch_value, _ = self.prediction(batch_latents)
        batch_policy = F.softmax(batch_policy, dim=1).cpu().detach().numpy()
        batch_latents = batch_latents.cpu()
        batch_value = batch_value.cpu()
        return batch_latents, batch_policy, batch_value

    def recurrent_inference(self, latent, action):
        a_tensor = torch.tensor([action], dtype=torch.long, device=self.device)
        latent, reward = self.dynamics(latent, a_tensor)
        policy_logits, value, _ = self.prediction_old(latent)
        return latent, reward, policy_logits, value

    def update_representation(self):
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

    def add_noise_to_node(self, idx, epsilon = 0.25, alpha = 0.03):
        node = self.current_trajectory[idx]
        node.policy = add_dirichlet_noise(node.policy, epsilon=epsilon, alpha=alpha)
        for action, child in node.children.items():
            child.prior = node.policy[action]

    # -----------------------------
    # Training Step: remote method to update networks
    # -----------------------------
    def train_step(self, batch, only_dynamics = False):
        """
        Multi-step unrolling training step.
        
        Expects a batch dictionary with keys:
        "obs": (B, H, W, C) uint8 images,
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
        latent = self.representation(obs)
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
            
            # Loss for reward prediction. (reward[0] not used since it is reward for entering obs)
            reward_loss = -torch.sum(converted_target_rewards[:, i, :] * F.log_softmax(reward_logits, dim=1), dim=1).mean()

            if not only_dynamics:
                policy_logits, _, value_logits = self.prediction(latent)
                
                # Loss for policy prediction.
                policy_loss = -torch.sum(target_policy[:, i, :] * F.log_softmax(policy_logits, dim=1), dim=1).mean()
                # Loss for value prediction.
                value_loss = -torch.sum(converted_target_values[:, i, :] * F.log_softmax(value_logits, dim=1), dim=1).mean()
                
                total_loss += reward_loss + policy_loss + value_loss
            else:
                total_loss += reward_loss

        # Backpropagate the total loss.
        total_loss.backward()
        
        # Step the optimizers.
        self.rep_optimizer.step()
        self.dyn_optimizer.step()
        if not only_dynamics:
            self.pred_optimizer.step()
        
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
                next_latents, rewards, _ = self.dynamics(latent_batch, action_tensor)
                batch_policy_logits, estimated_values, _ = self.prediction(next_latents)
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
            n.visit_count += 1
            n.value_sum += backup_value
            backup_value = n.reward + discount * backup_value
            n.virtual_loss = 0.0

    def create_tree_from_game(self, game_trajectory):
        root_node = None
        prev_node = None
        prev_action_taken = None
        path = []
        for (obs_sample, latent, action_taken, policy, reward, expanded_value) in game_trajectory:
            new_node = MCTSNode(latent=latent.to(self.device), prior=1.0, reward=reward, is_gameplay=True)
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
        self.current_trajectory = path

    def selection_task_no_expand(self, index, c_puct, discount, max_depth=MAX_DEPTH):
        """
        Traverse the tree from the given root until a leaf is reached.
        """
        node = self.current_trajectory[index]
        path = []
        cumulative_discount = 1.0
        while len(path) <= max_depth:
            if not node.children:
                break  # reached a leaf
            path.append(node)
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
            node = chosen_child
            cumulative_discount *= discount
        path.append(node)
        self.selection_results.put((node, path, cumulative_discount))

    def expand_selected(self, discount):
        unique_leaves = set()
        while not self.selection_results.empty():
            leaf, path, cumulative_discound = self.selection_results.get()
            if not leaf.is_expanded:
                unique_leaves.add((leaf, tuple(path)))
        unique_leaves = list(unique_leaves)
        self.batch_expand_nodes([t[0] for t in unique_leaves])

        # Backup: For each simulation, backup the leaf value along its path.
        for leaf, path in unique_leaves:
            leaf_value = leaf.expanded_value
            self.backup_simulation(path, leaf_value, discount=discount)

    def drop_non_gameplay_children_and_compute_improvements(self, idx, temperature=1.0):
        children_to_drop = []
        node = self.current_trajectory[idx]
        for action, child in node.children.items():
            if not child.is_gameplay:
                children_to_drop.append(action)
        # Compute improved policy from the root's children.
        visit_counts = np.array([node.children[a].visit_count for a in range(self.action_space)])
        if visit_counts.sum() > 0:
            improved_policy = apply_temperature(visit_counts, temperature)
        else:
            improved_policy = np.ones(self.action_space) / self.action_space
        root_value = node.q_value()
        for action in children_to_drop:
            del node.children[action]
        return improved_policy, root_value

    def get_value(self, idx):
        return self.current_trajectory[idx].q_value()


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
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # Resize the image to the desired shape.
    frame = cv2.resize(frame, resize_shape, interpolation=cv2.INTER_AREA)
    return frame


# -----------------------------
# Main Loop Example with Full MCTS Sample Generation
# -----------------------------
if __name__ == '__main__':
    ray.init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("MsPacman-v4")#, render_mode='human')
    action_space = env.action_space.n
    agent = MuZeroAgent.remote(action_space=action_space, device=device)
    
    num_episodes = 6000
    num_episodes_until_mcts = 10
    replay_buffer = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        preprocessed_obs = preprocess_frame(obs, resize_shape=IMAGE_RESHAPE)
        prev_reward = 0.0
        done = False
        episode_reward = 0
        game_trajectory = []  # store tuples of (obs, latent, action)
        frame_stack = np.repeat(preprocessed_obs[np.newaxis, :, :], 4, axis=0)
        
        # Play one episode.
        while not done:
            preprocessed_obs = preprocess_frame(obs, resize_shape=IMAGE_RESHAPE)
            frame_stack = np.concatenate([frame_stack[1:, :, :], preprocessed_obs[np.newaxis, :, :]], axis=0)
            with torch.no_grad():
                latent, policy, value = ray.get(agent.initial_inference.remote(frame_stack))
            action = np.random.choice(np.arange(action_space), p=policy)
            obs, reward, done, _, _ = env.step(action)
            # obs, chosen action, reward for entering obs
            game_trajectory.append((frame_stack, latent, action, policy, prev_reward, value))
            episode_reward += reward
            prev_reward = reward
            #env.render()
        
        print(f"Episode {episode} reward: {episode_reward}")
        print_memory_diagnostics()
        
        ray.get(agent.create_tree_from_game.remote(game_trajectory))

        trajectory = []
        # Store one-step transition (for a basic training sample)
        if episode < num_episodes_until_mcts:
            for (obs_sample, latent, action_taken, policy, reward, expanded_value), idx in zip(game_trajectory, list(range(len(game_trajectory)))):
                transition = {
                    "obs": obs_sample,
                    "action": action_taken,
                    "reward": reward,
                    "target_policy": np.ones(action_space) / action_space,
                    "target_value": ray.get(agent.get_value.remote(idx))
                }
                trajectory.append(transition)
            replay_buffer.append(trajectory)

        if episode >= num_episodes_until_mcts:
            print("starting mcts tree search")
            for (obs_sample, latent, action_taken, policy, reward, expanded_value), idx in zip(game_trajectory, list(range(len(game_trajectory)))):
                print("Running MCTS index", idx, "out of", len(game_trajectory))
                temperature = get_temperature(episode)
                improved_policy, value_estimate = run_full_mcts(agent, idx,
                                                                    num_processes=NUM_PROCESSES,
                                                                    num_simulations=128,
                                                                    c_puct=1.5,
                                                                    discount=0.997,
                                                                    temperature=temperature)
                # Here we create a training sample using the improved target.
                mcts_sample = {
                    "obs": obs_sample,
                    "action": action_taken,
                    "reward": reward,  # For illustration; in practice use multi-step return.
                    "target_policy": improved_policy,
                    "target_value": value_estimate
                }
                trajectory.append(mcts_sample)
            replay_buffer.append(trajectory)
        
        print(f"Replay buffer size: {sum([len(traj) for traj in replay_buffer])}")
        
        for _ in range(20):
            batch = prepare_batch(replay_buffer, 128, 5, action_space)
            loss = ray.get(agent.train_step.remote(batch))
            print(f"Training loss: {loss}")
        
        #agent.soft_update_prediction.remote(0.05)
        if episode % 5 == 0 and episode > 1 and episode > num_episodes_until_mcts:
            ray.get(agent.update_prediction.remote())
            ray.get(agent.update_representation.remote())
            temperature = get_temperature(episode)
            reanalyze_replay_buffer(agent, replay_buffer, temperature=temperature)
        if len(replay_buffer) > 30:
            replay_buffer.pop(0)
