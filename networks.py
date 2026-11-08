import torch
import torch.nn as nn
import torch.nn.functional as F

from scalar_categorical import create_support, scalar_from_logits


def renormalize(latent):
    """MuZero appendix G hidden-state scaling: min-max each sample to [0, 1].

    Applied to the output of BOTH latent-emitting networks. Without it the
    latent scale is unbounded and drifts with depth across the 5-step unroll
    and the search tree -- the instability the paper says this exists to
    prevent, and which the SimSiam loss was otherwise fighting alone.
    """
    flat = latent.flatten(start_dim=1)
    mn = flat.min(dim=1, keepdim=True).values
    mx = flat.max(dim=1, keepdim=True).values
    flat = (flat - mn) / (mx - mn + 1e-5)
    return flat.view_as(latent)

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

    def __init__(self, input_channels=3, latent_channels=128, trunk_blocks=2,
                 renormalize=True):
        super(RepresentationNetwork, self).__init__()
        # Appendix G min-max scaling of the emitted latent. A plain attribute,
        # not a buffer: weight files stay interchangeable. EfficientZero's
        # Atari config runs without it (muzero.LATENT_RENORMALIZE).
        self.renormalize_output = renormalize
        self.relu = nn.ReLU(inplace=False)
        c1, c2 = max(latent_channels // 4, 8), max(latent_channels // 2, 16)

        # Audit finding 6: EfficientZero's DownSample normalizes after EVERY
        # strided convolution. These four previously ran bare (bias=True, no
        # norm), so inter-layer scale drift was unconstrained and the final
        # renormalize could only bound the output, not condition the path.
        self.conv1 = nn.Conv2d(input_channels, c1, kernel_size=3, stride=2,
                               padding=1, bias=False)                # 84 -> 42
        self.bn1 = nn.BatchNorm2d(c1)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, stride=2,
                               padding=1, bias=False)                # 42 -> 21
        self.bn2 = nn.BatchNorm2d(c2)
        self.block1 = BasicBlock(c2, c2)
        self.conv3 = nn.Conv2d(c2, latent_channels, kernel_size=3, stride=2,
                               padding=1, bias=False)                # 21 -> 11
        self.bn3 = nn.BatchNorm2d(latent_channels)
        self.conv4 = nn.Conv2d(latent_channels, latent_channels, kernel_size=3,
                               stride=2, padding=1, bias=False)      # 11 -> 6
        self.bn4 = nn.BatchNorm2d(latent_channels)
        # Full-channel trunk depth is a size knob (finding 7): the repo
        # default is 2; the EffZero-dims arm runs 1.
        self.trunk = nn.ModuleList([BasicBlock(latent_channels, latent_channels)
                                    for _ in range(trunk_blocks)])

    def forward(self, x):
        # Downsample hard before spending residual blocks. At batch 1 -- the
        # acting path -- this network is kernel-launch bound rather than compute
        # bound, so layer COUNT costs more than layer width.
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.block1(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        for block in self.trunk:
            x = block(x)
        # latent state of shape (batch, latent_channels, 6, 6), scaled to [0,1]
        # unless renormalization is off
        return renormalize(x) if self.renormalize_output else x

# The support and the encode/decode pair live in scalar_categorical, so the
# transform applied to targets and the one undone on predictions cannot drift
# apart. support_size is a half-width in TRANSFORMED space; decoded scalars come
# back in raw units.

# PredictionNetwork with categorical value representation.
# NOTE: the policy head returns raw logits. Callers are responsible for
# applying softmax (for acting/priors) or log_softmax (for the training loss).
#
# option_length=L > 1 (muzero.OPTION_LENGTH; CURRICULUM "Test O", DESIGN
# "Heads"): OptionZero's option heads (network_unit.py 27-67). Their
# PolicyNetwork runs ONE 1x1 conv + BatchNorm with
# num_output_channels * option_seq_length channels and gives head i its own
# channel slice and its own fc (38); a 1x1 conv's output channel and a
# BatchNorm channel each read only their own filter / statistics, so after
# the shared trunk the heads share nothing. Mirrored at our head sizes: each
# extra head i = 1..L-1 is a copy of OUR policy branch (own 1x1 conv of
# head_channels, own [fc_hidden] layer when set, own final fc) over A moves
# plus a stop class (index A), reading the same trunk output x as the
# policy and value branches; default PyTorch init everywhere, the final
# bias zero except 10 on stop (39-44). Built after every existing layer of
# THIS net, so its policy/value layers init exactly as an L = 1 net's at the
# same seed; the extra heads do draw RNG, so every net built after the
# prediction net (in NetworkHelper's order the value prefix, the projector
# and anything later) inits differently from an L = 1 build at that seed
# (no paired init is required). L = 1 builds nothing (state_dict, outputs
# and RNG use unchanged). The
# policy head keeps EfficientZero's zero-init and stays head 0, with a fixed
# stop logit of -100 (61-66). forward(latent) is unchanged;
# forward(latent, with_options=True) also returns option logits [B, L, A+1].
class PredictionNetwork(nn.Module):
    def __init__(self, latent_channels=128, action_space=9, support_size=10, num_bins=21,
                 pool=(6, 6), transform=True, head_channels=32, fc_hidden=None,
                 option_length=1):
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
        # Head dims are size knobs (audit finding 7): repo default 32-channel
        # reductions with a direct feature->output FC; the EffZero-dims arm
        # runs 16 channels with a [32] hidden (their A.1 head shape, minus
        # the BN their FCs carry -- prediction runs batch-1 constantly on the
        # acting path, where single-row BatchNorm is the prefix head's known
        # hazard; documented deviation).
        features = head_channels * pool[0] * pool[1]

        # Policy head:
        self.policy_conv = nn.Conv2d(latent_channels, head_channels, kernel_size=1)
        if fc_hidden:
            self.policy_hidden = nn.Linear(features, fc_hidden)
            self.policy_fc = nn.Linear(fc_hidden, action_space)
        else:
            self.policy_hidden = None
            self.policy_fc = nn.Linear(features, action_space)

        # Value head now outputs categorical logits instead of a single scalar.
        self.value_conv = nn.Conv2d(latent_channels, head_channels, kernel_size=1)
        if fc_hidden:
            self.value_hidden = nn.Linear(features, fc_hidden)
            self.value_fc = nn.Linear(fc_hidden, num_bins)
        else:
            self.value_hidden = None
            self.value_fc = nn.Linear(features, num_bins)

        # EfficientZero A.1: zero the FINAL head layers, so at init the
        # categorical heads decode to exactly 0 (uniform logits over a
        # symmetric support) and the policy is exactly uniform -- warmup
        # searches and bootstraps start neutral instead of from init noise.
        for fc in (self.value_fc, self.policy_fc):
            nn.init.zeros_(fc.weight)
            nn.init.zeros_(fc.bias)

        self.support_size = support_size
        self.num_bins = num_bins
        self.head_channels = head_channels
        self.fc_hidden = fc_hidden

        # Test O's option heads (see the note above the class). Registered
        # last: L = 1 adds no module, no key and no RNG draw.
        self.action_space = action_space
        self.option_length = int(option_length)
        if self.option_length < 1:
            raise ValueError(f"option_length must be >= 1, got {option_length!r}")
        if self.option_length > 1:
            self.option_convs = nn.ModuleList()
            self.option_hiddens = nn.ModuleList() if fc_hidden else None
            self.option_fcs = nn.ModuleList()
            for _ in range(self.option_length - 1):
                self.option_convs.append(
                    nn.Conv2d(latent_channels, head_channels, kernel_size=1))
                if fc_hidden:
                    self.option_hiddens.append(nn.Linear(features, fc_hidden))
                fc = nn.Linear(fc_hidden if fc_hidden else features,
                               action_space + 1)
                with torch.no_grad():
                    fc.bias.zero_()
                    fc.bias[action_space] = 10.0   # the stop class
                self.option_fcs.append(fc)

    def forward(self, latent, with_options=False):
        x = self.block(latent)
        x = self.relu(x)

        # Policy branch:
        p = self.relu(self.policy_conv(x))
        p = F.adaptive_avg_pool2d(p, self.pool)
        p = p.contiguous().view(p.size(0), -1)
        if self.policy_hidden is not None:
            p = self.relu(self.policy_hidden(p))
        policy_logits = self.policy_fc(p)

        # Value branch:
        v = self.relu(self.value_conv(x))
        v = F.adaptive_avg_pool2d(v, self.pool)
        v = v.contiguous().view(v.size(0), -1)
        if self.value_hidden is not None:
            v = self.relu(self.value_hidden(v))
        value_logits = self.value_fc(v)  # shape: (batch_size, num_bins)
        support = create_support(self.support_size, self.num_bins, latent.device)
        value = scalar_from_logits(value_logits, support, invert=self.transform)

        if not with_options:
            return policy_logits, value, value_logits
        return policy_logits, value, value_logits, self._option_logits(
            x, policy_logits)

    def _option_logits(self, x, policy_logits):
        """[B, L, A+1]: head 0 = the policy logits plus a fixed stop logit
        of -100 (network_unit.py 61-66), heads 1..L-1 the option heads."""
        stop = torch.full((policy_logits.size(0), 1), -100.0,
                          dtype=policy_logits.dtype, device=policy_logits.device)
        heads = [torch.cat([policy_logits, stop], dim=1)]
        for i in range(self.option_length - 1):
            o = self.relu(self.option_convs[i](x))
            o = F.adaptive_avg_pool2d(o, self.pool)
            o = o.contiguous().view(o.size(0), -1)
            if self.option_hiddens is not None:
                o = self.relu(self.option_hiddens[i](o))
            heads.append(self.option_fcs[i](o))
        return torch.stack(heads, dim=1)

# Revised DynamicsNetwork with categorical reward representation.
class DynamicsNetwork(nn.Module):
    """Audit finding 5: EfficientZero's dynamics is an explicit RESIDUAL
    update -- conv(cat(state, action)) -> BN -> += state -> relu -> blocks --
    identity plus a learned delta. Without the skip, conv1 had to reproduce
    the whole state before it could modify it, which is the axis multi-step
    latent drift lives on (and the axis the SimSiam loss then fights alone).

    action_encoding selects how the action reaches conv1:
      "embed"  -- learned latent_channels-dim embedding broadcast spatially
                  (the repo's original, richest, 2x conv1 width);
      "plane"  -- EfficientZero's single constant plane of a/|A| (thinnest,
                  imposes an ordinal fiction the reference tolerates);
      "onehot" -- |A| binary planes (thin, no fake metric; the curriculum
                  branch's choice, and it extends to duration planes).

    option_length=L > 1 (muzero.OPTION_LENGTH; CURRICULUM "Test O", DESIGN
    "Dynamics"): one forward can be a JUMP of up to L moves. "embed" only
    (anything else raises): per-position embedding rows, position 1 =
    action_embedding (today's parameter, same name), positions 2..L the new
    option_embeddings[0..L-2], initialised to ZERO (no RNG drawn). A [B]
    long action is today's path exactly; a [B, k] long action (k <= L),
    padded with -1 (a first move in every row, the padding a suffix, else
    ValueError), is encoded as the SUM of each position's row, a -1
    contributing nothing -- OptionZero's per-position action planes
    (atari.cpp 141-150), summed into one embedding. L = 1 builds nothing:
    state_dict keys, shapes and outputs are today's.
    """

    def __init__(self, latent_channels=128, action_space=9, support_size=10, num_bins=21,
                 pool=(6, 6), transform=True, blocks=2, action_encoding="embed",
                 reward_channels=32, renormalize=True, option_length=1):
        super(DynamicsNetwork, self).__init__()
        self.renormalize_output = renormalize   # see RepresentationNetwork
        # See PredictionNetwork: must match the encode side's transform flag.
        self.transform = transform
        self.pool = pool
        self.action_space = action_space
        self.support_size = support_size
        self.num_bins = num_bins
        self.action_encoding = action_encoding
        if action_encoding == "embed":
            self.action_embedding = nn.Embedding(action_space, latent_channels)
            action_channels = latent_channels
        elif action_encoding == "plane":
            self.action_embedding = None
            action_channels = 1
        elif action_encoding == "onehot":
            self.action_embedding = None
            action_channels = action_space
        else:
            raise ValueError(f"unknown action_encoding {action_encoding!r}")
        self.conv1 = nn.Conv2d(latent_channels + action_channels, latent_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(latent_channels)
        self.relu = nn.ReLU(inplace=False)
        self.blocks = nn.ModuleList([BasicBlock(latent_channels, latent_channels)
                                     for _ in range(blocks)])

        # Reward head now outputs categorical logits. Channel width is a
        # size knob like every other head's (audit follow-up: it was the
        # one head left hardcoded when the others got knobs).
        self.reward_channels = reward_channels
        self.reward_conv = nn.Conv2d(latent_channels, reward_channels, kernel_size=1)
        self.reward_fc = nn.Linear(reward_channels * pool[0] * pool[1], num_bins)
        # EfficientZero A.1 zero-init: reward decodes exactly 0 at init.
        nn.init.zeros_(self.reward_fc.weight)
        nn.init.zeros_(self.reward_fc.bias)

        # Test O's jump rows (see the class docstring). Registered last and
        # built from explicit zeros (no reset_parameters, no RNG draw).
        self.option_length = int(option_length)
        if self.option_length < 1:
            raise ValueError(f"option_length must be >= 1, got {option_length!r}")
        if self.option_length > 1:
            if action_encoding != "embed":
                raise ValueError(
                    f"option_length {self.option_length} > 1 needs "
                    f"action_encoding 'embed' (per-position embedding rows); "
                    f"got {action_encoding!r}")
            self.option_embeddings = nn.ModuleList([
                nn.Embedding(action_space, latent_channels,
                             _weight=torch.zeros(action_space, latent_channels))
                for _ in range(self.option_length - 1)])

    def _encode_jump(self, action):
        """[B, k] long, -1 padded -> [B, C]: the sum of each position's row
        (position 1 = action_embedding), a -1 adding nothing. Every row
        must hold a first move, and its padding must be a suffix (no move
        after a -1): a padding bug fails loudly here instead of training on
        a no-action or a holed embedding (one host sync per jump call)."""
        if action.size(1) > self.option_length:
            raise ValueError(
                f"a jump of {action.size(1)} positions on a dynamics net "
                f"built for option_length {self.option_length}")
        valid = action >= 0
        if not bool(valid[:, 0].all()):
            raise ValueError("a jump row without a first move (-1 at "
                             "position 1)")
        if action.size(1) > 1 and bool((valid[:, 1:] & ~valid[:, :-1]).any()):
            raise ValueError("a jump row with a move after its -1 padding "
                             "(the padding must be a suffix)")
        e = None
        for j in range(action.size(1)):
            col = action[:, j]
            table = (self.action_embedding if j == 0
                     else self.option_embeddings[j - 1])
            row = table(col.clamp(min=0))
            row = torch.where((col >= 0).unsqueeze(1), row,
                              torch.zeros_like(row))
            e = row if e is None else e + row
        return e

    def _encode_action(self, action, batch, h, w, device):
        if self.action_encoding == "embed":
            # A [B, k] action is a jump (option_length > 1; k = 1 is one
            # move, the same row as the [B] path); a [B] action is today's.
            e = (self._encode_jump(action) if action.dim() == 2
                 else self.action_embedding(action))
            return e.contiguous().view(batch, -1, 1, 1).expand(-1, -1, h, w)
        if self.action_encoding == "plane":
            plane = action.to(torch.float32) / float(self.action_space)
            return plane.view(batch, 1, 1, 1).expand(batch, 1, h, w)
        planes = torch.zeros(batch, self.action_space, h, w, device=device)
        index = action.view(batch, 1, 1, 1).expand(batch, 1, h, w)
        return planes.scatter_(1, index, 1.0)

    def forward(self, latent, action):
        batch_size = latent.size(0)
        act = self._encode_action(action, batch_size,
                                  latent.size(2), latent.size(3), latent.device)
        x = self.conv1(torch.cat([latent, act], dim=1))
        x = self.bn1(x)
        # The residual update: the network learns a DELTA on the incoming
        # latent rather than a from-scratch reconstruction of it.
        x = x + latent
        x = self.relu(x)
        for block in self.blocks:
            x = block(x)
        # Appendix G scaling on the emitted state; the reward branch below
        # reads the same renormalized latent every other head will see.
        if self.renormalize_output:
            x = renormalize(x)
        next_latent = x

        # Reward branch:
        r = self.relu(self.reward_conv(x))
        r = F.adaptive_avg_pool2d(r, self.pool)
        r = r.contiguous().view(batch_size, -1)
        reward_logits = self.reward_fc(r)
        support = create_support(self.support_size, self.num_bins, latent.device)
        reward = scalar_from_logits(reward_logits, support, invert=self.transform)

        return next_latent, reward, reward_logits


class LatentProjector(nn.Module):
    """EfficientZero's self-supervised consistency heads (section 4.1; SimSiam).

    Plain latent MSE toward a moving target -- what this replaces -- has a
    known failure mode: the easiest way for the dynamics output to match the
    representation's is for BOTH to drift toward low-information latents that
    are easy to predict rather than useful. Two properties remove it:

    * the loss is cosine similarity, which is scale-free, so shrinking latent
      norms buys nothing;
    * the branches are asymmetric -- online goes through projection AND
      prediction, the target through projection only with gradients stopped --
      which is the SimSiam arrangement that makes the matching problem
      non-trivial.

    Layer shapes follow the EfficientZero reference implementation:
    projection fc-BN-ReLU / fc-BN-ReLU / fc-BN, prediction fc-BN-ReLU / fc.
    Training-only: inference consumers never project, so NetworkManager does
    not carry one.
    """

    def __init__(self, latent_dim, proj_hidden=1024, proj_out=1024, pred_hidden=512):
        super(LatentProjector, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(latent_dim, proj_hidden),
            nn.BatchNorm1d(proj_hidden),
            nn.ReLU(inplace=False),
            nn.Linear(proj_hidden, proj_hidden),
            nn.BatchNorm1d(proj_hidden),
            nn.ReLU(inplace=False),
            nn.Linear(proj_hidden, proj_out),
            nn.BatchNorm1d(proj_out),
        )
        self.prediction = nn.Sequential(
            nn.Linear(proj_out, pred_hidden),
            nn.BatchNorm1d(pred_hidden),
            nn.ReLU(inplace=False),
            nn.Linear(pred_hidden, proj_out),
        )

    def forward(self, latent, with_prediction):
        """Project a (B, C, H, W) latent. with_prediction=True is the online
        branch; False is the target branch (detach at the call site)."""
        x = self.projection(latent.flatten(start_dim=1))
        return self.prediction(x) if with_prediction else x


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

    per_step=K > 0 (muzero.PREFIX_BN_PER_STEP; CURRICULUM.md "PB"): every
    BatchNorm here also keeps K sets of running statistics, one per LSTM step
    since the last reset (1..K). Training calls the head once per unroll step
    in train mode, so each step is normalized with its OWN batch statistics;
    eval mode (the search) used the running statistics pooled over all K
    steps, and step 1 -- a fresh state -- has a very different distribution,
    so its eval-mode prefix came out inflated at large reward scales (Step 0
    (iii)). Train mode is unchanged (the same modules on batch statistics,
    the pooled running statistics still updated) and additionally folds step
    k's batch statistics into set k; eval mode normalizes each row with its
    own step's set. The buffers exist only when K > 0: an off net's
    state_dict, outputs and RNG use are the original class's.
    """

    def __init__(self, latent_channels=64, hidden_size=512, support_size=2000,
                 num_bins=51, head_channels=32, pool=(6, 6), transform=True,
                 fc_hidden=None, per_step=0):
        super(ValuePrefixNetwork, self).__init__()
        # See PredictionNetwork: must match the encode side's transform flag.
        self.transform = transform
        self.support_size = support_size
        self.num_bins = num_bins
        self.hidden_size = hidden_size
        self.head_channels = head_channels
        self.fc_hidden = fc_hidden
        self.pool = pool

        self.conv = nn.Conv2d(latent_channels, head_channels, kernel_size=1, bias=False)
        self.norm_in = nn.BatchNorm2d(head_channels)
        self.relu = nn.ReLU(inplace=False)
        features = head_channels * pool[0] * pool[1]
        self.lstm = nn.LSTM(input_size=features, hidden_size=hidden_size)
        self.norm_out = nn.BatchNorm1d(hidden_size)
        if fc_hidden:
            # EfficientZero A.1's exact head: a bottleneck FC (their 32 units,
            # BN + ReLU) between the LSTM and the categorical output. Split
            # into named modules -- NOT a Sequential -- so forward can honour
            # this class's single-row train-mode contract (the `batched`
            # guard) for the bottleneck's BatchNorm too.
            self.fc_hidden_layer = nn.Linear(hidden_size, fc_hidden)
            self.fc_hidden_norm = nn.BatchNorm1d(fc_hidden)
            self.fc = nn.Linear(fc_hidden, num_bins)
            final = self.fc
        else:
            self.fc_hidden_layer = None
            self.fc_hidden_norm = None
            self.fc = nn.Linear(hidden_size, num_bins)
            final = self.fc
        # EfficientZero A.1 zero-init: the prefix decodes exactly 0 at init.
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        # Per-step running statistics (see the class docstring). Registered
        # after every parameter, and only when on: no RNG is drawn, so an on
        # net's weights init exactly as an off net's.
        self.per_step = int(per_step or 0)
        if self.per_step:
            for name in self.norm_names():
                c = getattr(self, name).num_features
                self.register_buffer(f"{name}_step_mean",
                                     torch.zeros(self.per_step, c))
                self.register_buffer(f"{name}_step_var",
                                     torch.ones(self.per_step, c))
                self.register_buffer(f"{name}_step_count",
                                     torch.zeros(self.per_step,
                                                 dtype=torch.long))

    def norm_names(self):
        """The head's BatchNorm attributes, in forward order."""
        names = ["norm_in", "norm_out"]
        if self.fc_hidden_norm is not None:
            names.append("fc_hidden_norm")
        return names

    def initial_state(self, batch_size, device):
        """Zeroed hidden state: no reward accumulated yet."""
        shape = (self.lstm.num_layers, batch_size, self.hidden_size)
        return (torch.zeros(*shape, device=device),
                torch.zeros(*shape, device=device))

    def forward(self, latent, state, step=None):
        """One unroll step. Returns (prefix_scalar, prefix_logits, next_state).

        step: the LSTM step this call takes since the last reset -- 1 from a
        fresh or reset state -- as an int or a LongTensor of shape (batch,).
        Ignored unless per_step, where every call must pass it (a missed
        caller fails loudly): one uniform step in train mode, per row in
        eval mode (a batched expansion mixes rows at different depths)."""
        x = self.conv(latent)
        # BatchNorm needs more than one sample in train mode; a single-row batch
        # happens during search, where eval-mode running stats apply anyway.
        batched = latent.size(0) > 1 or not self.training
        k = self._step_index(step, latent) if self.per_step else None
        if batched:
            x = self._norm("norm_in", x, k)
        x = self.relu(x)
        x = F.adaptive_avg_pool2d(x, self.pool)
        x = x.contiguous().view(x.size(0), -1).unsqueeze(0)   # (seq=1, batch, feat)

        out, state = self.lstm(x, state)
        out = out.squeeze(0)
        if batched:
            out = self._norm("norm_out", out, k)
        h = self.relu(out)
        if self.fc_hidden_layer is not None:
            h = self.fc_hidden_layer(h)
            if batched:
                h = self._norm("fc_hidden_norm", h, k)
            h = self.relu(h)
        prefix_logits = self.fc(h)

        support = create_support(self.support_size, self.num_bins, latent.device)
        prefix = scalar_from_logits(prefix_logits, support, invert=self.transform)
        return prefix, prefix_logits, state

    def _step_index(self, step, latent):
        """per_step's validated step: an int in train mode (one step per
        call), a LongTensor (batch,) in eval mode; values in 1..per_step."""
        K = self.per_step
        if step is None:
            raise ValueError("PREFIX_BN_PER_STEP: a value-prefix call passed "
                             "no LSTM step index")
        if self.training:
            if torch.is_tensor(step):
                flat = step.reshape(-1)
                k = int(flat[0])
                if bool((flat != k).any()):
                    raise ValueError("PREFIX_BN_PER_STEP: a train-mode call "
                                     "must take one step, got several")
            else:
                k = int(step)
            if not 1 <= k <= K:
                raise ValueError(f"PREFIX_BN_PER_STEP: step {k} outside "
                                 f"1..{K}")
            return k
        B = latent.size(0)
        if torch.is_tensor(step):
            rows = step.to(device=latent.device, dtype=torch.long).reshape(-1)
            if rows.numel() == 1:
                rows = rows.expand(B)
        else:
            rows = torch.full((B,), int(step), dtype=torch.long,
                              device=latent.device)
        if rows.numel() != B:
            raise ValueError(f"PREFIX_BN_PER_STEP: {rows.numel()} step "
                             f"indices for a batch of {B}")
        if bool(((rows < 1) | (rows > K)).any()):
            raise ValueError(f"PREFIX_BN_PER_STEP: a step outside 1..{K}")
        return rows

    def _norm(self, name, x, k):
        """BatchNorm `name` on x. Off (k None): the module itself. Train
        mode: the module itself, plus step k's batch statistics folded into
        set k. Eval mode: each row with its own step's running statistics
        and the module's affine weight and bias."""
        bn = getattr(self, name)
        if k is None:
            return bn(x)
        if self.training:
            self._fold_step_stats(name, bn, x, k)
            return bn(x)
        shape = (x.size(0), x.size(1)) + (1,) * (x.dim() - 2)
        mean = getattr(self, f"{name}_step_mean")[k - 1].view(shape)
        inv = torch.rsqrt(getattr(self, f"{name}_step_var")[k - 1]
                          + bn.eps).view(shape)
        y = (x - mean) * inv
        if bn.weight is not None:
            aff = (1, -1) + (1,) * (x.dim() - 2)
            y = y * bn.weight.view(aff) + bn.bias.view(aff)
        return y

    @torch.no_grad()
    def _fold_step_stats(self, name, bn, x, k):
        """BatchNorm's own running update, into step k's set: this call's
        batch mean and UNBIASED variance (what running_var tracks), its
        momentum (a cumulative average when momentum is None)."""
        dims = [0] + list(range(2, x.dim()))
        mean = x.mean(dim=dims)
        var = x.var(dim=dims, correction=1)
        count = getattr(self, f"{name}_step_count")
        count[k - 1] += 1
        f = (bn.momentum if bn.momentum is not None
             else 1.0 / float(count[k - 1]))
        rm = getattr(self, f"{name}_step_mean")
        rv = getattr(self, f"{name}_step_var")
        rm[k - 1] = (1 - f) * rm[k - 1] + f * mean
        rv[k - 1] = (1 - f) * rv[k - 1] + f * var