import torch
import torch.nn as nn
import torch.nn.functional as F

from scalar_categorical import create_support, scalar_from_logits

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
    """Encodes an 84x84 observation stack down to 6x6, a 16x downsample.

    Previously 84x84 -> 21x21, only 4x. That produced a latent LARGER in bytes
    than the observation it encoded (110 KB against 83 KB, and 1.12x a raw
    uncompressed Atari frame) and carried 12x more spatial positions than either
    MuZero or EfficientZero use on Atari. No head ever consumed more than 8x8 of
    it, so the resolution was computed, propagated through every dynamics
    rollout, stored on every tree node, and then pooled away unused.

    Channels widen as resolution falls, so the expensive residual blocks sit at
    small spatial sizes rather than large ones.
    """

    def __init__(self, input_channels=3, latent_channels=128):
        super(RepresentationNetwork, self).__init__()
        self.relu = nn.ReLU(inplace=False)
        c1, c2 = max(latent_channels // 4, 8), max(latent_channels // 2, 16)

        self.conv1 = nn.Conv2d(input_channels, c1, kernel_size=3, stride=2, padding=1)  # 84 -> 42
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1)              # 42 -> 21
        self.block1 = BasicBlock(c2, c2)
        self.conv3 = nn.Conv2d(c2, latent_channels, kernel_size=3, stride=2, padding=1)  # 21 -> 11
        self.conv4 = nn.Conv2d(latent_channels, latent_channels,
                               kernel_size=3, stride=2, padding=1)                       # 11 -> 6
        self.block2 = BasicBlock(latent_channels, latent_channels)
        self.block3 = BasicBlock(latent_channels, latent_channels)

    def forward(self, x):
        # Downsample hard before spending residual blocks. At batch 1 -- the
        # acting path -- this network is kernel-launch bound rather than compute
        # bound, so layer COUNT costs more than layer width. Three blocks here
        # against the previous four, and the two cheap early convolutions run
        # back to back instead of sandwiching a block at 42x42.
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.block1(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.block2(x)
        x = self.block3(x)
        return x  # latent state of shape (batch, latent_channels, 6, 6)

# The support and the encode/decode pair live in scalar_categorical, so the
# transform applied to targets and the one undone on predictions cannot drift
# apart. support_size is a half-width in TRANSFORMED space; decoded scalars come
# back in raw units.

# PredictionNetwork with categorical value representation.
# NOTE: the policy head returns raw logits. Callers are responsible for
# applying softmax (for acting/priors) or log_softmax (for the training loss).
class PredictionNetwork(nn.Module):
    def __init__(self, latent_channels=128, action_space=9, support_size=10, num_bins=21,
                 pool=(6, 6), transform=True):
        super(PredictionNetwork, self).__init__()
        # Whether decoded scalars come back through h_inverse. Must match the
        # transform flag the training side passes to scalar_to_categorical_batch
        # (muzero.USE_H_TRANSFORM), or encode and decode stop being inverses.
        self.transform = transform
        # Pool to the latent's own spatial size. This was hardcoded to 8x8, which
        # against a 6x6 latent would INTERPOLATE UP -- inventing resolution that
        # is not there and paying for it in the fully connected layer.
        self.pool = pool
        self.block = BasicBlock(latent_channels, latent_channels)
        self.relu = nn.ReLU(inplace=False)
        
        # Policy head:
        self.policy_conv = nn.Conv2d(latent_channels, 32, kernel_size=1)
        self.policy_fc = nn.Linear(32 * pool[0] * pool[1], action_space)
        
        # Value head now outputs categorical logits instead of a single scalar.
        self.value_conv = nn.Conv2d(latent_channels, 32, kernel_size=1)
        self.value_fc = nn.Linear(32 * pool[0] * pool[1], num_bins)
        
        self.support_size = support_size
        self.num_bins = num_bins
        
    def forward(self, latent):
        x = self.block(latent)
        x = self.relu(x)
        
        # Policy branch:
        p = self.relu(self.policy_conv(x))
        p = F.adaptive_avg_pool2d(p, self.pool)
        p = p.contiguous().view(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        # Value branch:
        v = self.relu(self.value_conv(x))
        v = F.adaptive_avg_pool2d(v, self.pool)
        v = v.contiguous().view(v.size(0), -1)
        value_logits = self.value_fc(v)  # shape: (batch_size, num_bins)
        support = create_support(self.support_size, self.num_bins, latent.device)
        value = scalar_from_logits(value_logits, support, invert=self.transform)
        
        return policy_logits, value, value_logits

# Revised DynamicsNetwork with categorical reward representation.
class DynamicsNetwork(nn.Module):
    def __init__(self, latent_channels=128, action_space=9, support_size=10, num_bins=21,
                 pool=(6, 6), transform=True):
        super(DynamicsNetwork, self).__init__()
        # See PredictionNetwork: must match the encode side's transform flag.
        self.transform = transform
        self.pool = pool
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
        self.reward_fc = nn.Linear(32 * pool[0] * pool[1], num_bins)
        
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
        r = F.adaptive_avg_pool2d(r, self.pool)
        r = r.contiguous().view(batch_size, -1)
        reward_logits = self.reward_fc(r)
        support = create_support(self.support_size, self.num_bins, latent.device)
        reward = scalar_from_logits(reward_logits, support, invert=self.transform)

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

    Layer order follows LightZero's EfficientZero reference implementation:
    conv1x1 -> BatchNorm2d -> activation -> flatten -> LSTM -> BatchNorm1d ->
    activation -> fc. Both norms matter: the accumulated prefix drifts in scale
    as the unroll deepens, and without the post-LSTM activation the path from
    hidden state to logits is linear.
    """

    def __init__(self, latent_channels=64, hidden_size=512, support_size=2000,
                 num_bins=51, head_channels=32, pool=(6, 6), transform=True):
        super(ValuePrefixNetwork, self).__init__()
        # See PredictionNetwork: must match the encode side's transform flag.
        self.transform = transform
        self.support_size = support_size
        self.num_bins = num_bins
        self.hidden_size = hidden_size
        self.pool = pool

        self.conv = nn.Conv2d(latent_channels, head_channels, kernel_size=1, bias=False)
        self.norm_in = nn.BatchNorm2d(head_channels)
        self.relu = nn.ReLU(inplace=False)
        features = head_channels * pool[0] * pool[1]
        self.lstm = nn.LSTM(input_size=features, hidden_size=hidden_size)
        self.norm_out = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, num_bins)

    def initial_state(self, batch_size, device):
        """Zeroed hidden state: no reward accumulated yet."""
        shape = (self.lstm.num_layers, batch_size, self.hidden_size)
        return (torch.zeros(*shape, device=device),
                torch.zeros(*shape, device=device))

    def forward(self, latent, state):
        """One unroll step. Returns (prefix_scalar, prefix_logits, next_state)."""
        x = self.conv(latent)
        # BatchNorm needs more than one sample in train mode; a single-row batch
        # happens during search, where eval-mode running stats apply anyway.
        batched = latent.size(0) > 1 or not self.training
        if batched:
            x = self.norm_in(x)
        x = self.relu(x)
        x = F.adaptive_avg_pool2d(x, self.pool)
        x = x.contiguous().view(x.size(0), -1).unsqueeze(0)   # (seq=1, batch, feat)

        out, state = self.lstm(x, state)
        out = out.squeeze(0)
        if batched:
            out = self.norm_out(out)
        prefix_logits = self.fc(self.relu(out))

        support = create_support(self.support_size, self.num_bins, latent.device)
        prefix = scalar_from_logits(prefix_logits, support, invert=self.transform)
        return prefix, prefix_logits, state