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

# PredictionNetwork with categorical value representation.
# NOTE: the policy head returns raw logits. Callers are responsible for
# applying softmax (for acting/priors) or log_softmax (for the training loss).
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

        # Value branch:
        v = self.relu(self.value_conv(x))
        v = F.adaptive_avg_pool2d(v, (8, 8))
        v = v.contiguous().view(v.size(0), -1)
        value_logits = self.value_fc(v)  # shape: (batch_size, num_bins)
        support = create_support(self.support_size, self.num_bins, latent.device)
        value = scalar_from_logits(value_logits, support)
        
        return policy_logits, value, value_logits

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


class ValuePrefixNetwork(nn.Module):
    """Predicts the discounted reward SUM over an unroll, end to end.

    MuZero predicts a reward at every unrolled state and adds them up. That
    forces the hardest possible question at every depth -- *exactly when* does
    the reward land -- and the errors compound down the tree. EfficientZero
    (Ye et al. 2021, section 4.2) predicts the sum directly from the sequence of
    unrolled latents instead:

        value_prefix = f(s_t, s^_t+1, ..., s^_t+k-1)     f = LSTM

    which lets the model stay vague about *when* while staying accurate about
    *whether*. The LSTM is supervised at every unroll step, so nothing is lost
    in supervision density.

    The hidden state carries the accumulated reward so far, which is why it must
    be reset wherever a new unroll begins -- the start of a training unroll, and
    the root of each search.
    """

    def __init__(self, latent_channels=64, hidden_size=512, support_size=2000,
                 num_bins=51, pool=(4, 4)):
        super(ValuePrefixNetwork, self).__init__()
        self.support_size = support_size
        self.num_bins = num_bins
        self.hidden_size = hidden_size
        self.pool = pool

        self.conv = nn.Conv2d(latent_channels, 32, kernel_size=1)
        self.relu = nn.ReLU(inplace=False)
        features = 32 * pool[0] * pool[1]
        self.lstm = nn.LSTM(input_size=features, hidden_size=hidden_size)
        # BatchNorm on the LSTM output; the accumulated prefix drifts in scale as
        # the unroll deepens and this keeps the head's input distribution stable.
        self.norm = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, num_bins)

    def initial_state(self, batch_size, device):
        """Zeroed hidden state: no reward accumulated yet."""
        shape = (self.lstm.num_layers, batch_size, self.hidden_size)
        return (torch.zeros(*shape, device=device),
                torch.zeros(*shape, device=device))

    def forward(self, latent, state):
        """One unroll step. Returns (prefix_scalar, prefix_logits, next_state)."""
        x = self.relu(self.conv(latent))
        x = F.adaptive_avg_pool2d(x, self.pool)
        x = x.contiguous().view(x.size(0), -1).unsqueeze(0)   # (seq=1, batch, feat)

        out, state = self.lstm(x, state)
        out = out.squeeze(0)
        # BatchNorm needs more than one sample; a single-row batch happens during
        # search, where eval-mode running stats apply anyway.
        if out.size(0) > 1 or not self.training:
            out = self.norm(out)
        prefix_logits = self.fc(out)

        support = create_support(self.support_size, self.num_bins, latent.device)
        prefix = scalar_from_logits(prefix_logits, support)
        return prefix, prefix_logits, state