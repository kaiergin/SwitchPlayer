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

if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

if torch.cuda.is_available():
    print("CUDA is available! Using GPU:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available. Using CPU.")

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

@ray.remote(num_gpus=1, max_concurrency=8)
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
        self.rep_optimizer = optim.Adam(self.representation.parameters(), lr=1e-4)
        self.dyn_optimizer = optim.Adam(self.dynamics.parameters(), lr=1e-4)
        self.pred_optimizer = optim.Adam(self.prediction.parameters(), lr=1e-4)

    def initial_inference(self, observation):
        obs = cv2.resize(observation, (95, 95))
        obs = np.transpose(obs, (2, 0, 1))
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device) / 255.0
        obs = obs.unsqueeze(0)
        latent = self.representation(obs)
        policy_logits, value = self.prediction(latent)
        return latent, policy_logits, value

    def recurrent_inference(self, latent, action):
        a_tensor = torch.tensor([action], dtype=torch.long, device=self.device)
        latent, reward = self.dynamics(latent, a_tensor)
        policy_logits, value = self.prediction(latent)
        return latent, reward, policy_logits, value

    # -----------------------------
    # Training Step: remote method to update networks
    # -----------------------------
    def train_step(self, batch):
        """
        Expects a batch dictionary with keys:
          "obs": (N, H, W, C) uint8 images,
          "action": (N,) int,
          "reward": (N,) float,
          "target_policy": (N, action_space) one-hot or probability distribution,
          "target_value": (N,) float.
        """
        obs = torch.tensor(batch["obs"], dtype=torch.float32, device=self.device) / 255.0
        obs = obs.permute(0, 3, 1, 2)  # (batch, C, H, W)
        actions = torch.tensor(batch["action"], dtype=torch.long, device=self.device)
        rewards = torch.tensor(batch["reward"], dtype=torch.float32, device=self.device)
        target_policy = torch.tensor(batch["target_policy"], dtype=torch.float32, device=self.device)
        target_value = torch.tensor(batch["target_value"], dtype=torch.float32, device=self.device)
        
        self.rep_optimizer.zero_grad()
        self.pred_optimizer.zero_grad()
        self.dyn_optimizer.zero_grad()
        
        latent = self.representation(obs)
        policy_logits, value = self.prediction(latent)
        
        policy_loss = -torch.sum(target_policy * F.log_softmax(policy_logits, dim=1), dim=1).mean()
        value_loss = F.mse_loss(value.squeeze(), target_value)
        pred_loss = policy_loss + value_loss
        
        latent_next, predicted_reward = self.dynamics(latent.detach(), actions)
        reward_loss = F.mse_loss(predicted_reward.squeeze(), rewards)
        
        total_loss = pred_loss + reward_loss
        total_loss.backward()
        
        self.rep_optimizer.step()
        self.pred_optimizer.step()
        self.dyn_optimizer.step()
        
        return total_loss.item()

    # -----------------------------
    # MCTS Node Class
    # -----------------------------
    class MCTSNode:
        def __init__(self, latent, prior, reward=0.0):
            self.latent = latent            # Tensor representing the latent state.
            self.prior = prior              # float (from network prediction)
            self.reward = reward            # immediate reward (float)
            self.children = {}              # dict mapping action -> child MCTSNode
            self.visit_count = 0
            self.value_sum = 0.0
            self.is_expanded = False
            self.policy = None              # cached policy (numpy array)

        def q_value(self):
            return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

    # -----------------------------
    # Phase 1: Selection (Without Expansion)
    # -----------------------------
    @ray.remote
    def selection_task_no_expand(self, root, c_puct, discount):
        """
        Traverse the tree from the root until a leaf is reached.
        Returns the simulation path (list of nodes), the leaf node, and the cumulative discount.
        """
        path = []
        node = root
        cumulative_discount = 1.0
        while True:
            path.append(node)
            if not node.children:
                break
            total_visits = sum(child.visit_count for child in node.children.values())
            best_score = -float('inf')
            best_child = None
            for a, child in node.children.items():
                u = c_puct * child.prior * np.sqrt(total_visits) / (1 + child.visit_count)
                score = child.q_value() + u
                if score > best_score:
                    best_score = score
                    best_child = child
            node = best_child
            cumulative_discount *= discount
        return (path, node, cumulative_discount)

    # -----------------------------
    # Phase 3: Backup
    # -----------------------------
    def backup_simulation(self, path, discount, leaf_value):
        backup_value = leaf_value
        for n in reversed(path):
            backup_value = n.reward + discount * backup_value
            n.visit_count += 1
            n.value_sum += backup_value
        return

    # -----------------------------
    # Full MCTS with Batched Expansion Across Leaf Nodes
    # -----------------------------
    def run_full_mcts(self, root_latent, num_simulations=50, c_puct=1.0, discount=0.997):
        # Build root node.
        root = self.MCTSNode(latent=root_latent, prior=1.0, reward=0.0)
        with torch.no_grad():
            policy_logits, value = self.prediction(root_latent)
            policy = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
        root.policy = policy
        root.is_expanded = True
        root.children = {}
        for a in range(self.action_space):
            a_tensor = torch.tensor([a], dtype=torch.long, device=self.device)
            next_latent, reward = self.dynamics(root_latent, a_tensor)
            child = self.MCTSNode(latent=next_latent, prior=policy[a], reward=reward.item())
            root.children[a] = child

        # Phase 1: Run selection tasks in parallel.
        selection_futures = [type(self).selection_task_no_expand.remote(self, root, c_puct, discount) for _ in range(num_simulations)]
        selection_results = ray.get(selection_futures)  # Each is (path, leaf, cumulative_discount)

        # Phase 2: Batch expansion across all leaves that are not yet expanded.
        unique_leaves = {}
        for path, leaf, cum_discount in selection_results:
            if not leaf.is_expanded:
                unique_leaves[id(leaf)] = leaf
        if unique_leaves:
            latent_list = [leaf.latent for leaf in unique_leaves.values()]
            # Each latent is of shape (1, channels, H, W); stack along the batch dimension.
            batch_latents = torch.cat(latent_list, dim=0)
            with torch.no_grad():
                batch_policy_logits, batch_values = self.prediction(batch_latents)
                batch_policies = F.softmax(batch_policy_logits, dim=1).cpu().numpy()
            idx = 0
            for leaf in unique_leaves.values():
                leaf_policy = batch_policies[idx]
                leaf_value = batch_values[idx].item()
                with self._lock:
                    leaf.policy = leaf_policy
                    leaf.is_expanded = True
                    leaf.children = {}
                    for a in range(self.action_space):
                        a_tensor = torch.tensor([a], dtype=torch.long, device=self.device)
                        next_latent, reward = self.dynamics(leaf.latent, a_tensor)
                        child = self.MCTSNode(latent=next_latent, prior=leaf_policy[a], reward=reward.item())
                        leaf.children[a] = child
                # Store the expanded value for use during backup.
                leaf.expanded_value = leaf_value
                idx += 1

        # Phase 3: For each simulation, back up the leaf value along the simulation path.
        backup_futures = []
        for path, leaf, cum_discount in selection_results:
            # Use the expanded value if available.
            if hasattr(leaf, 'expanded_value'):
                leaf_value = leaf.expanded_value
            else:
                with torch.no_grad():
                    _, value = self.prediction(leaf.latent)
                leaf_value = value.item()
            backup_futures.append(self.backup_simulation.remote(path, discount, leaf_value))
        ray.get(backup_futures)

        visit_counts = np.array([root.children[a].visit_count for a in range(self.action_space)])
        if visit_counts.sum() > 0:
            improved_policy = visit_counts / visit_counts.sum()
        else:
            improved_policy = np.ones(self.action_space) / self.action_space
        root_value = root.q_value()
        return improved_policy, root_value, root

    # (The train_step method remains as before.)


# -----------------------------
# Main Loop Example with Full MCTS Sample Generation
# -----------------------------
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("MsPacman-v4")
    action_space = env.action_space.n
    agent = MuZeroAgent.remote(action_space=action_space, device=device)
    
    num_episodes = 2000
    replay_buffer = deque(maxlen=10000)
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        game_trajectory = []  # store tuples of (obs, latent, action)
        
        # Play one episode.
        while not done:
            with torch.no_grad():
                latent, policy_logits, value = ray.get(agent.initial_inference.remote(obs))
            policy = F.softmax(policy_logits, dim=1).cpu().detach().numpy()[0]
            action = np.random.choice(np.arange(action_space), p=policy)
            game_trajectory.append((obs, latent, action))
            
            next_obs, reward, done, _, _ = env.step(action)
            episode_reward += reward
            
            # Store one-step transition (for a basic training sample).
            transition = {
                "obs": obs,
                "action": action,
                "reward": reward,
                "target_policy": np.eye(action_space)[action],
                "target_value": reward  # Using immediate reward as a simple target.
            }
            replay_buffer.append(transition)
            
            obs = next_obs
            #env.render()
        
        print(f"Episode {episode} reward: {episode_reward}")
        
        # After the game, for each state in the trajectory, run full MCTS to get improved targets.
        for (obs_sample, latent, action_taken) in game_trajectory:
            improved_policy, value_estimate, tree = ray.get(agent.run_full_mcts.remote(latent,
                                                                        num_simulations=10,
                                                                        c_puct=1.0,
                                                                        discount=0.997))
            # Here we create an additional training sample using the improved target.
            mcts_sample = {
                "obs": obs_sample,
                "action": action_taken,
                "reward": 0.0,  # For illustration; in practice use multi-step return.
                "target_policy": improved_policy,
                "target_value": value_estimate
            }
            replay_buffer.append(mcts_sample)
        
        print(f"Replay buffer size: {len(replay_buffer)}")
        
        # If enough samples have been collected, sample a batch and train.
        if len(replay_buffer) >= 128:
            for _ in range(100):
                batch_samples = random.sample(list(replay_buffer), 128)
                batch = {
                    "obs": np.array([sample["obs"] for sample in batch_samples]),
                    "action": np.array([sample["action"] for sample in batch_samples]),
                    "reward": np.array([sample["reward"] for sample in batch_samples]),
                    "target_policy": np.array([sample["target_policy"] for sample in batch_samples]),
                    "target_value": np.array([sample["target_value"] for sample in batch_samples])
                }
                loss = agent.train_step(batch)
                print(f"Training loss: {loss}")
