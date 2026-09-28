"""ReDo: recycling dormant units (Sokar, Agarwal, Castro, Evci -- "The Dormant
Neuron Phenomenon in Deep Reinforcement Learning", ICML 2023,
arXiv 2302.12902), adapted to this repo's networks (2026-09-26).

Why: the faithful arm's SGD with coupled weight decay slowly kills units --
by iter 75-85 half of the representation's first-layer filters had decayed
to ~0 in one seed and ~8% of all BatchNorm channels had collapsed scales.
A unit that stops receiving gradient never comes back on its own.

The paper's algorithm, as verified against the text and the official code
(google/dopamine, dopamine/labs/redo):
  * score s_i = E|h_i| / mean_k E|h_k| over a layer, h the post-ReLU
    activation; a conv "neuron" is a channel, averaged over batch AND space;
  * tau-dormant if s_i <= tau (0.025 in the paper's default setting, 0.1
    elsewhere); scored on a FRESH uniform replay sample every 1000 updates;
  * recycle: incoming weights redrawn from the layer's initializer, the
    unit's bias set to 0, OUTGOING weights zeroed (so recycling does not
    move the network's output), optimizer moments zeroed for those entries;
    the output layer is never recycled. All incoming reinits happen before
    any outgoing zeroing (so a recycled consumer row cannot refill a
    recycled producer's zeroed column).

Adaptations, each deliberate:
  * Only units with ONE consumer are recycled: a unit feeding a residual
    stream also flows through the skip path, so zeroing its outgoing
    weights would not isolate it (the official ResNet code likewise
    recycles only the block-internal convs). See recyclable_units().
  * REDO_NETS scopes which nets are recycled. The arm excludes the SimSiam
    projector (Kai, 2026-09-26): ~94% of all dormant units live in its
    1024-wide hidden layers (they die abruptly at 10-20k steps with score
    exactly 0, and the faithful run reached +21 with them dead), so
    including it would re-randomize most of the projector every pass.
  * BatchNorm between producer and ReLU (the paper's nets have none): the
    recycled channel's running stats are reset (mean 0, var 1), and gamma /
    beta are set to the MEAN |gamma| (floored at GAMMA_FLOOR) and MEAN beta
    of the layer's surviving channels -- not the identity (1, 0). Under
    coupled weight decay the survivors sit at |gamma| ~0.03-0.1 with
    beta/|gamma| ~ -0.5..-1.2, so the identity made recycled units 7-75x
    louder than their peers and blew the loss up 5-6x for several steps
    after each pass (round-1/2 audits; the global grad clip at 5 was never
    reached on real batches).
  * Producers WITHOUT BN (the prediction heads' 1x1 policy/value convs):
    the row is redrawn from the default init, UNSCALED, and the unit's BIAS
    -- 0 in the paper -- is SOLVED (1-D bisection: the mean post-ReLU output
    is continuous and non-decreasing in it) so the unit's mean activation on
    the scoring batch equals the MEDIAN surviving unit's. The survivors
    carry their activity through learned positive biases and fire on 83-99%
    of positions; a bias-0 unit fires on ~50%. Two earlier matchings
    failed: rescaling the row to the survivors' mean row NORM left units
    dead on arrival (round-2 audit), and rescaling the row at bias 0 until
    it hit the target activity inflated it 4-24x (median; p99 158-416x)
    into a sparse, stiff unit (round-3 audit). Measured with the bias
    solve (faithful snapshots s1 @85/@40, s0 @20; real Pong windows; 120
    forced recycles per head): row norm p50 1.0-6.2x the survivors' median,
    p99 <= 13x; held-out activity 0.97-1.03x the target; active on p50
    92-99% of positions for the value head, 57-68% for the late policy
    head (p10 0.22; its survivors decayed to <1/3 of the init scale). The
    official code's weight_scaling (row rescaled to the survivors' mean
    norm) plus the bias solve was also measured -- policy p50 66-84%,
    value 100% (fully linear) -- a second-order difference; the paper's
    init scale was kept so the bias is the only deviation.
  * Scores are measured on throwaway TRAIN-mode copies (batch statistics,
    what training sees) over the training-shaped forward -- root
    representation + prediction, then UNROLL_STEPS dynamics steps with the
    stored actions, each followed by prediction, the value-prefix LSTM
    (from its zero state) and, if recycled, the online projector branch --
    so a unit used at several depths is scored over all of them. The live
    nets' BN running statistics are never touched by scoring.
  * Conv -> Linear/LSTM outgoing mapping uses PyTorch's NCHW flatten:
    channel c owns the CONTIGUOUS column block [c*HW, (c+1)*HW) (the
    official JAX code's NHWC repeat() mapping is a known bug; not ported).
    The value-prefix conv's consumer is lstm.weight_ih_l0, all four gates.
  * Optimizer state: SGD momentum buffers (and Adam's exp_avg/exp_avg_sq,
    if an arm uses AdamW) are zeroed for every recycled entry.
  * Reinit draws come from a dedicated torch.Generator, so recycling never
    perturbs the global RNG stream (augmentation, root noise).
"""
import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

ALL_NETS = ("representation", "dynamics", "prediction", "value_prefix",
            "projector")
# Floor for a recycled BN channel's gamma, so a layer whose survivors have
# all collapsed toward 0 does not hand the recycled unit a dead scale.
GAMMA_FLOOR = 0.05
BIAS_ITERS = 60  # bisection steps of the no-BN bias solve (float64)


@dataclass(frozen=True)
class Unit:
    """One recyclable layer: `producer` (Conv2d/Linear) makes the units,
    `norm` (BatchNorm, or None) normalizes them before the ReLU, and the
    parameter `consumer`.`consumer_attr` is their ONLY reader, its column
    block [c*hw, (c+1)*hw) belonging to unit c."""
    net: str
    producer: str
    norm: object
    consumer: str
    hw: int
    consumer_attr: str = "weight"


def _blocks(module):
    """Inner units (conv1 -> bn1 -> ReLU -> conv2) of every BasicBlock
    under `module`, by dotted module name."""
    return [(f"{name}.conv1", f"{name}.bn1", f"{name}.conv2", 1)
            for name, m in module.named_modules()
            if type(m).__name__ == "BasicBlock"]


def _consumer_param(nets, u):
    return getattr(dict(nets[u.net].named_modules())[u.consumer],
                   u.consumer_attr)


def recyclable_units(nets, include=ALL_NETS):
    """The recyclable layers of the repo's five trained nets, by module
    name, restricted to the nets in `include`. Everything not listed is
    deliberately excluded: residual-stream producers (representation conv2
    -> block1, conv4 -> trunk; dynamics conv1 -> residual update; every
    block's conv2/bn2), output layers, the value-prefix LSTM's hidden units
    (also read by the recurrence), and the dynamics reward head (weight 0
    in the paper arms)."""
    assert isinstance(include, (tuple, list, set, frozenset)) and \
        set(include) <= set(ALL_NETS), (
            f"REDO_NETS must be a collection of {ALL_NETS}, got {include!r}")
    units = []
    if "representation" in include:
        rep = nets["representation"]
        units += [Unit("representation", "conv1", "bn1", "conv2", 1),
                  Unit("representation", "conv3", "bn3", "conv4", 1)]
        units += [Unit("representation", *u) for u in _blocks(rep)]
    if "dynamics" in include:
        units += [Unit("dynamics", *u) for u in _blocks(nets["dynamics"])]
    if "prediction" in include:
        pred = nets["prediction"]
        units += [Unit("prediction", *u) for u in _blocks(pred)]
        hw = int(pred.pool[0] * pred.pool[1])
        for head in ("policy", "value"):
            hidden = getattr(pred, f"{head}_hidden")
            units.append(Unit("prediction", f"{head}_conv", None,
                              f"{head}_hidden" if hidden is not None
                              else f"{head}_fc", hw))
            if hidden is not None:
                units.append(Unit("prediction", f"{head}_hidden", None,
                                  f"{head}_fc", 1))
    if "value_prefix" in include:
        vp = nets["value_prefix"]
        units.append(Unit("value_prefix", "conv", "norm_in", "lstm",
                          int(vp.pool[0] * vp.pool[1]), "weight_ih_l0"))
        if vp.fc_hidden_layer is not None:
            units.append(Unit("value_prefix", "fc_hidden_layer",
                              "fc_hidden_norm", "fc", 1))
    if "projector" in include:
        units += [Unit("projector", "projection.0", "projection.1",
                       "projection.3", 1),
                  Unit("projector", "projection.3", "projection.4",
                       "projection.6", 1),
                  Unit("projector", "prediction.0", "prediction.1",
                       "prediction.3", 1)]
    for u in units:
        mods = dict(nets[u.net].named_modules())
        n_out = mods[u.producer].weight.shape[0]
        n_in = _consumer_param(nets, u).shape[1]
        assert n_in == n_out * u.hw, (
            f"{u.net}.{u.consumer}.{u.consumer_attr} reads {n_in} inputs, "
            f"but {u.producer} makes {n_out} units x hw {u.hw}: the layer "
            f"map is stale")
        if u.norm is not None:
            assert mods[u.norm].num_features == n_out, u
    return units


@torch.no_grad()
def dormancy_scores(nets, obs, actions, units):
    """Per-unit normalized activation scores {unit: tensor[n_units]}; see
    dormancy_stats."""
    return dormancy_stats(nets, obs, actions, units)[0]


@torch.no_grad()
def dormancy_stats(nets, obs, actions, units):
    """(scores, raw, inputs) from a training-shaped forward on train-mode
    COPIES (the live nets and their BN running statistics are untouched):
    scores = normalized activation scores {unit: tensor[n_units]}; raw =
    the unnormalized mean post-ReLU activations; inputs = for units WITHOUT
    BN, the producer's inputs over every call (concatenated on the batch
    axis), which recycle() uses for activity matching. obs: [B, C, H, W]
    float in [0, 1]; actions: [B, K] long. `nets` must hold every net the
    forward runs (representation, dynamics, prediction, value_prefix;
    projector only when a projector unit is scored)."""
    with_proj = any(u.net == "projector" for u in units)
    names = ["representation", "dynamics", "prediction", "value_prefix"]
    names += ["projector"] if with_proj else []
    copies = {k: copy.deepcopy(nets[k]).train() for k in names}
    sums = {u: None for u in units}
    counts = {u: 0 for u in units}
    captured = {}
    hooks = []
    for u in units:
        mods = dict(copies[u.net].named_modules())
        tap = mods[u.norm] if u.norm is not None else mods[u.producer]

        def hook(_m, _i, out, u=u):
            a = F.relu(out).abs()
            m = a.mean(dim=(0, 2, 3)) if a.dim() == 4 else a.mean(dim=0)
            sums[u] = m if sums[u] is None else sums[u] + m
            counts[u] += 1

        hooks.append(tap.register_forward_hook(hook))
        if u.norm is None:
            def pre(_m, inp, u=u):
                captured.setdefault(u, []).append(inp[0])
            hooks.append(mods[u.producer].register_forward_pre_hook(pre))
    try:
        latent = copies["representation"](obs)
        copies["prediction"](latent)
        state = copies["value_prefix"].initial_state(obs.size(0), obs.device)
        for k in range(actions.shape[1]):
            latent, _, _ = copies["dynamics"](latent, actions[:, k])
            copies["prediction"](latent)
            _, _, state = copies["value_prefix"](latent, state)
            if with_proj:
                copies["projector"](latent, with_prediction=True)
    finally:
        for h in hooks:
            h.remove()
    scores, raw = {}, {}
    for u in units:
        if counts[u] == 0:
            continue
        mean = sums[u] / counts[u]
        raw[u] = mean
        scores[u] = mean / (mean.mean() + 1e-9)
    inputs = {u: torch.cat(v, dim=0) for u, v in captured.items()}
    return scores, raw, inputs


def _unit_pre(prod, x, w):
    """Pre-activation of ONE unit of `prod` with weights `w`, no bias."""
    if isinstance(prod, torch.nn.Conv2d):
        return F.conv2d(x, w.unsqueeze(0), None, prod.stride, prod.padding,
                        prod.dilation, prod.groups)
    return F.linear(x, w.unsqueeze(0))


def _unit_activity(prod, x, w, b=0.0):
    """Mean post-ReLU output of ONE unit with weights `w` and bias `b`."""
    return float(F.relu(_unit_pre(prod, x, w) + b).mean())


def _solve_bias(z, target, iters=BIAS_ITERS):
    """The bias b with mean(relu(z + b)) == target > 0. The left side is
    continuous and non-decreasing in b, 0 at b = -max(z), and >= mean(z) + b
    everywhere (relu(v) >= v), so [-max(z), target - mean(z)] brackets the
    root. Bisected in float64 on the CPU."""
    z = z.detach().double().flatten().cpu()
    lo, hi = -float(z.max()), target - float(z.mean())
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if float(F.relu(z + mid).mean()) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _uniform_(t, fan_in, generator):
    """PyTorch's default Conv/Linear weight init, kaiming_uniform(a=sqrt 5)
    = U(-1/sqrt(fan_in), 1/sqrt(fan_in)), drawn from `generator`."""
    bound = 1.0 / np.sqrt(fan_in)
    draw = torch.empty(t.shape, dtype=t.dtype).uniform_(
        -bound, bound, generator=generator)
    t.copy_(draw.to(t.device))


def _zero_state(optimizers, param, index):
    """Zero every optimizer-state tensor shaped like `param` at `index`."""
    for opt in optimizers:
        st = opt.state.get(param)
        if not st:
            continue
        for key in ("momentum_buffer", "exp_avg", "exp_avg_sq"):
            buf = st.get(key)
            if isinstance(buf, torch.Tensor) and buf.shape == param.shape:
                buf[index] = 0


@torch.no_grad()
def recycle(nets, optimizers, scores, tau, generator, raw=None, inputs=None):
    """Recycle every unit whose score <= tau, in two passes (all incoming
    reinits, then all outgoing zeroing). `raw` / `inputs` (dormancy_stats)
    enable the bias solve for producers without BN; without them (or
    without a bias) those units keep the paper's bias 0. Returns
    {unit: [indices]}."""
    plan = {}
    for u, s in scores.items():
        idx = torch.nonzero(s <= tau).flatten().tolist()
        if idx:
            plan[u] = idx
    # Pass 1: incoming side -- producer rows, biases, BN channels.
    for u, idx in plan.items():
        mods = dict(nets[u.net].named_modules())
        prod = mods[u.producer]
        fan_in = prod.weight[0].numel()
        keep = [c for c in range(prod.weight.shape[0]) if c not in set(idx)]
        bn = mods[u.norm] if u.norm is not None else None
        if bn is not None:
            g = (max(float(bn.weight[keep].abs().mean()), GAMMA_FLOOR)
                 if keep else 1.0)
            b = float(bn.bias[keep].mean()) if keep else 0.0
        target = None
        if (bn is None and prod.bias is not None and raw is not None
                and inputs is not None and u in inputs and keep):
            target = float(raw[u][keep].median())
        for c in idx:
            _uniform_(prod.weight[c], fan_in, generator)
            _zero_state(optimizers, prod.weight, c)
            if prod.bias is not None:
                prod.bias[c] = 0.0
                if target is not None and target > 0:
                    prod.bias[c] = _solve_bias(
                        _unit_pre(prod, inputs[u], prod.weight[c]), target)
                _zero_state(optimizers, prod.bias, c)
            if bn is not None:
                bn.weight[c] = g
                bn.bias[c] = b
                bn.running_mean[c] = 0.0
                bn.running_var[c] = 1.0
                _zero_state(optimizers, bn.weight, c)
                _zero_state(optimizers, bn.bias, c)
    # Pass 2: outgoing side -- the consumer's column block of each unit.
    for u, idx in plan.items():
        cons = _consumer_param(nets, u)
        for c in idx:
            cols = (slice(None), slice(c * u.hw, (c + 1) * u.hw))
            cons[cols] = 0.0
            _zero_state(optimizers, cons, cols)
    return plan
