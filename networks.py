import torch
import torch.nn as nn
import torch.nn.functional as F

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