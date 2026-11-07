"""PB (Kai, 2026-10-24; CURRICULUM.md "Next experiment (pre-registered) --
PB: per-step prefix BatchNorm statistics on the token arm"): the value-
prefix head's BatchNorms keep one set of running statistics per LSTM step
since the last reset (muzero.PREFIX_BN_PER_STEP). Pins: off = the pre-PB
head bit for bit (keys, outputs, RNG); on = training unchanged, step k's
set the BatchNorm running update of step k's batches alone, eval mode per
row; a missed step fails loudly; every caller's step (the training unroll,
the sequential and both vectorized engines, the diagnostics, ReDo, the
inference server) against an independent depth oracle; engine parity with
the flag on; the PB arm's delta and gates; the buffers through actors,
checkpoints, snapshots and their loaders; the new diagnostic fields. CPU
only, no launches."""
import asyncio
import copy
import pathlib
import random
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import ablation  # noqa: E402
import muzero  # noqa: E402
from muzero_vtree import VectorizedCommunalPUCT, VectorizedPUCT  # noqa: E402
from networks import ValuePrefixNetwork  # noqa: E402
from scalar_categorical import create_support, scalar_from_logits  # noqa: E402
from tests.test_gumbelact8 import probe  # noqa: E402,F401
from tests.test_prioract2 import restore_rngs  # noqa: E402,F401

TK_ARM = ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-"
          "tokens-rgrow")
PB_ARM = TK_ARM + "-pbn"
# Test 2's generation-1 students, rebuilt with the fix (CURRICULUM Test 2
# "REBUILD WITH THE FIX", 2026-10-25)
GEN1_ARMS = tuple("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-"
                  f"mspacman-tokens-gen1-{k}" for k in ("distill", "fresh"))
# Generation 2's student (DS2's arm, 2026-10-27): the fix inherited
GEN2_ARM = ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-"
            "mspacman-tokens-gen2-distill")
# Test B's five generations (2026-10-27): the fix inherited
FAST_ARMS = tuple("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-"
                  f"mspacman-tokens-fast-f{k}" for k in range(5))
# Test B2's three generations (2026-10-29): the fix inherited
TWO_ARMS = tuple("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-"
                 f"mspacman-tokens-two-t{k}" for k in range(3))
# Test C's two runs (2026-10-29): the fix inherited
TESTC_ARMS = tuple("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-"
                   f"mspacman-tokens-{v}" for v in ("long-n0", "reset-r0"))
# Test C2's run (2026-11-02): C plus the ramps, the fix inherited
TESTC2_ARM = ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-"
              "mspacman-tokens-ramp-a0")
# Test B2W's three generations (2026-11-04): T0-T2 with BBF's AdamW, the fix
# inherited
TESTB2W_ARMS = tuple("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-"
                     f"mspacman-tokens-adam-w{k}" for k in range(3))
# E3's two generations (2026-11-05): T1 / T2 with mining, the fix inherited
E3_ARMS = tuple("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-"
                f"mspacman-tokens-mined-m{k}" for k in (1, 2))
# Test Q's 17 (2026-11-07): the same recipe on Q*bert
TESTQ_ARMS = (tuple(f"{ablation.QT_ARM}-q{k}" for k in range(9))
              + tuple(f"{ablation.QM_ARM}-p{k}" for k in range(1, 9)))
ENV_ID = "MsPacmanNoFrameskip-v4"
CPU = torch.device("cpu")
C, HW = muzero.LATENT_CHANNELS, muzero.LATENT_HW
STEP_BUFFERS = ("_step_mean", "_step_var", "_step_count")


# --------------------------------------------------------------------------
# the pre-PB head, verbatim (6420f2e5) but for its docstrings
# --------------------------------------------------------------------------
class PrePB(nn.Module):
    def __init__(self, latent_channels=64, hidden_size=512, support_size=2000,
                 num_bins=51, head_channels=32, pool=(6, 6), transform=True,
                 fc_hidden=None):
        super().__init__()
        self.transform = transform
        self.support_size = support_size
        self.num_bins = num_bins
        self.hidden_size = hidden_size
        self.head_channels = head_channels
        self.fc_hidden = fc_hidden
        self.pool = pool

        self.conv = nn.Conv2d(latent_channels, head_channels, kernel_size=1,
                              bias=False)
        self.norm_in = nn.BatchNorm2d(head_channels)
        self.relu = nn.ReLU(inplace=False)
        features = head_channels * pool[0] * pool[1]
        self.lstm = nn.LSTM(input_size=features, hidden_size=hidden_size)
        self.norm_out = nn.BatchNorm1d(hidden_size)
        if fc_hidden:
            self.fc_hidden_layer = nn.Linear(hidden_size, fc_hidden)
            self.fc_hidden_norm = nn.BatchNorm1d(fc_hidden)
            self.fc = nn.Linear(fc_hidden, num_bins)
            final = self.fc
        else:
            self.fc_hidden_layer = None
            self.fc_hidden_norm = None
            self.fc = nn.Linear(hidden_size, num_bins)
            final = self.fc
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def initial_state(self, batch_size, device):
        shape = (self.lstm.num_layers, batch_size, self.hidden_size)
        return (torch.zeros(*shape, device=device),
                torch.zeros(*shape, device=device))

    def forward(self, latent, state):
        x = self.conv(latent)
        batched = latent.size(0) > 1 or not self.training
        if batched:
            x = self.norm_in(x)
        x = self.relu(x)
        x = F.adaptive_avg_pool2d(x, self.pool)
        x = x.contiguous().view(x.size(0), -1).unsqueeze(0)

        out, state = self.lstm(x, state)
        out = out.squeeze(0)
        if batched:
            out = self.norm_out(out)
        h = self.relu(out)
        if self.fc_hidden_layer is not None:
            h = self.fc_hidden_layer(h)
            if batched:
                h = self.fc_hidden_norm(h)
            h = self.relu(h)
        prefix_logits = self.fc(h)

        support = create_support(self.support_size, self.num_bins,
                                 latent.device)
        prefix = scalar_from_logits(prefix_logits, support,
                                    invert=self.transform)
        return prefix, prefix_logits, state


def make(cls, seed=3, **kw):
    torch.manual_seed(seed)
    return cls(latent_channels=C, hidden_size=64,
               support_size=muzero.VALUE_PREFIX_SUPPORT,
               num_bins=muzero.VALUE_PREFIX_BINS, **kw)


def dezero(net, seed=8):
    """The final layer is zero-init (every prefix decodes 0): give it
    weights so the outputs carry the BatchNorms' arithmetic."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in net.fc.parameters():
            p.normal_(0, 0.3, generator=g)


def latents(n, seed, channels=C):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, channels, HW, HW, generator=g)


def randomize_step_sets(net, seed=5):
    """Distinct statistics per step (the init sets are all (0, 1))."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name in net.norm_names():
            getattr(net, f"{name}_step_mean").normal_(0, 0.5, generator=g)
            getattr(net, f"{name}_step_var").uniform_(0.3, 3.0, generator=g)


def equal_outputs(a, b):
    (pa, la, (ha, ca)), (pb, lb, (hb, cb)) = a, b
    return all(torch.equal(x, y) for x, y in
               ((pa, pb), (la, lb), (ha, hb), (ca, cb)))


def equal_states(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    return list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)


# --------------------------------------------------------------------------
# off: the pre-PB head, bit for bit
# --------------------------------------------------------------------------
@pytest.mark.parametrize("fc_hidden", [None, 32])
def test_off_head_is_the_pre_pb_head_bit_for_bit(fc_hidden):
    ref = make(PrePB, fc_hidden=fc_hidden)
    rng_ref = torch.get_rng_state()
    new = make(ValuePrefixNetwork, fc_hidden=fc_hidden)
    assert torch.equal(torch.get_rng_state(), rng_ref)     # same RNG use
    assert new.per_step == 0
    assert equal_states(ref, new)                          # keys and values
    assert not any(k.endswith(STEP_BUFFERS) for k in new.state_dict())
    dezero(ref)
    dezero(new)
    for training in (True, False):
        ref.train(training)
        new.train(training)
        for n in (8, 1):                       # batched and single-row
            sr, sn = ref.initial_state(n, CPU), new.initial_state(n, CPU)
            for i in range(1, 6):
                lat = latents(n, 10 * n + i)
                out_r = ref(lat, sr)
                out_n = new(lat, sn, step=i)
                assert equal_outputs(out_r, out_n), (training, n, i)
                sr, sn = out_r[2], out_n[2]
        assert equal_states(ref, new)          # running stats included


def test_off_head_ignores_its_step_argument():
    net = make(ValuePrefixNetwork)
    dezero(net)
    net.eval()
    lat, st = latents(4, 1), net.initial_state(4, CPU)
    want = net(lat, st)
    for step in (None, 1, 99, torch.tensor([0, 7, 3, 1])):
        assert equal_outputs(net(lat, st, step=step), want)


def test_off_arms_build_the_pre_pb_head():
    """Every arm but PB and the arms built on it after its result (Test 2's
    students, generation 2's, Test B's, Test B2's, Test C's, Test C2's,
    Test B2W's, E3's, Test Q's) leaves the flag off: the factory's head, the
    harness's nets and the target model carry no step buffers."""
    assert muzero.PREFIX_BN_PER_STEP is False
    on = {name for name, spec in ablation.ARMS.items()
          if spec.get("PREFIX_BN_PER_STEP")}
    assert on == {PB_ARM, *GEN1_ARMS, GEN2_ARM, *FAST_ARMS, *TWO_ARMS,
                  *TESTC_ARMS, TESTC2_ARM, *TESTB2W_ARMS, *E3_ARMS,
                  *TESTQ_ARMS}, on
    for name in on:
        assert ablation.ARMS[name]["PREFIX_BN_PER_STEP"] is True, name
    with ablation.arm_config(TK_ARM) as spec:
        nets = ablation.build_networks(spec)
        assert nets["value_prefix"].per_step == 0
        keys = set(nets["value_prefix"].state_dict())
        torch.manual_seed(0)
        ref = PrePB(latent_channels=muzero.LATENT_CHANNELS,
                    hidden_size=muzero.VALUE_PREFIX_HIDDEN,
                    support_size=muzero.VALUE_PREFIX_SUPPORT,
                    num_bins=muzero.VALUE_PREFIX_BINS,
                    head_channels=muzero.PREFIX_HEAD_CHANNELS,
                    fc_hidden=muzero.PREFIX_FC_HIDDEN)
        assert keys == set(ref.state_dict())


# --------------------------------------------------------------------------
# on: training unchanged, per-step sets, eval per row
# --------------------------------------------------------------------------
@pytest.mark.parametrize("fc_hidden", [None, 32])
def test_on_head_trains_exactly_as_the_pre_pb_head(fc_hidden):
    ref = make(PrePB, fc_hidden=fc_hidden)
    rng_ref = torch.get_rng_state()
    on = make(ValuePrefixNetwork, fc_hidden=fc_hidden, per_step=5)
    assert torch.equal(torch.get_rng_state(), rng_ref)
    names = on.norm_names()
    assert names == (["norm_in", "norm_out"]
                     + (["fc_hidden_norm"] if fc_hidden else []))
    extra = {f"{n}{s}" for n in names for s in STEP_BUFFERS}
    assert set(on.state_dict()) == set(ref.state_dict()) | extra
    for k, v in ref.state_dict().items():
        assert torch.equal(on.state_dict()[k], v), k
    dezero(ref)
    dezero(on)
    ref.train()
    on.train()
    for n in (8, 1):
        sr, so = ref.initial_state(n, CPU), on.initial_state(n, CPU)
        for i in range(1, 6):
            lat = latents(n, 20 * n + i)
            out_r, out_o = ref(lat, sr), on(lat, so, step=i)
            assert equal_outputs(out_r, out_o), (n, i)
            sr, so = out_r[2], out_o[2]
    for k, v in ref.state_dict().items():       # pooled stats still update
        assert torch.equal(on.state_dict()[k], v), k
    for name in names:     # five batched calls, one per step; none single
        assert getattr(on, f"{name}_step_count").tolist() == [1] * 5


def _bn_inputs(net):
    seen = {}
    hooks = [getattr(net, n).register_forward_pre_hook(
        lambda m, i, n=n: seen.setdefault(n, []).append(i[0].detach()))
        for n in net.norm_names()]
    return seen, hooks


def test_step_sets_are_the_bn_update_of_their_own_step_alone():
    net = make(ValuePrefixNetwork, fc_hidden=32, per_step=5)
    dezero(net)
    net.train()
    seen, hooks = _bn_inputs(net)
    calls = [2, 2, 4]
    st = net.initial_state(8, CPU)
    for j, k in enumerate(calls):
        net(latents(8, 40 + j), st, step=k)
    for h in hooks:
        h.remove()
    for name in net.norm_names():
        bn = getattr(net, name)
        mean = getattr(net, f"{name}_step_mean")
        var = getattr(net, f"{name}_step_var")
        want_m = torch.zeros_like(mean)
        want_v = torch.ones_like(var)
        for x, k in zip(seen[name], calls):
            dims = [0] + list(range(2, x.dim()))
            n = x.numel() // x.size(1)
            bm = x.mean(dim=dims)
            bv = ((x - bm.view(1, -1, *[1] * (x.dim() - 2))) ** 2).sum(
                dim=dims) / (n - 1)
            want_m[k - 1] = 0.9 * want_m[k - 1] + 0.1 * bm
            want_v[k - 1] = 0.9 * want_v[k - 1] + 0.1 * bv
        assert bn.momentum == 0.1
        assert torch.allclose(mean, want_m, rtol=1e-5, atol=1e-6), name
        assert torch.allclose(var, want_v, rtol=1e-5, atol=1e-6), name
        for k in (1, 3, 5):                 # untouched sets stay at init
            assert torch.equal(mean[k - 1], torch.zeros_like(mean[0]))
            assert torch.equal(var[k - 1], torch.ones_like(var[0]))
        assert getattr(net, f"{name}_step_count").tolist() == [0, 2, 0, 1, 0]


@pytest.mark.parametrize("momentum", [0.1, None])
def test_a_single_step_set_matches_bns_own_running_stats(momentum):
    """A head trained at one step only: that step's set IS BatchNorm's
    own running update (momentum None: the cumulative average)."""
    net = make(ValuePrefixNetwork, fc_hidden=32, per_step=5)
    net.train()
    for name in net.norm_names():
        getattr(net, name).momentum = momentum
    for j in range(3):
        net(latents(8, 60 + j), net.initial_state(8, CPU), step=3)
    for name in net.norm_names():
        bn = getattr(net, name)
        assert torch.allclose(getattr(net, f"{name}_step_mean")[2],
                              bn.running_mean, rtol=1e-5, atol=1e-6), name
        assert torch.allclose(getattr(net, f"{name}_step_var")[2],
                              bn.running_var, rtol=1e-5, atol=1e-6), name


def test_eval_mode_normalizes_each_row_with_its_own_step():
    net = make(ValuePrefixNetwork, fc_hidden=32, per_step=5)
    dezero(net)
    randomize_step_sets(net)
    net.eval()
    B = 10
    lat = latents(B, 70)
    g = torch.Generator().manual_seed(1)
    st = (torch.randn(1, B, 64, generator=g), torch.randn(1, B, 64,
                                                          generator=g))
    rows = torch.tensor([1, 3, 2, 5, 1, 4, 4, 2, 5, 3])
    with torch.no_grad():
        mixed = net(lat, st, step=rows)
        per_k = {k: net(lat, st, step=k) for k in range(1, 6)}
        for r in range(B):
            k = int(rows[r])
            # the same batch shape: bitwise
            assert torch.equal(mixed[0][r], per_k[k][0][r])
            assert torch.equal(mixed[1][r], per_k[k][1][r])
            # one row at a time, its own step
            one = net(lat[r:r + 1], (st[0][:, r:r + 1].contiguous(),
                                     st[1][:, r:r + 1].contiguous()),
                      step=k)
            assert torch.allclose(one[1][0], mixed[1][r], atol=1e-5)
        # a (1,) tensor and an int are one step for every row
        assert equal_outputs(net(lat, st, step=torch.tensor([4])), per_k[4])
        # the steps genuinely differ
        assert not torch.allclose(per_k[1][1], per_k[2][1])


def test_eval_set_k_is_bn_eval_on_set_ks_statistics():
    """Eval mode on step k's set equals the module's own eval mode with
    running statistics equal to that set."""
    on = make(ValuePrefixNetwork, fc_hidden=32, per_step=5)
    dezero(on)
    randomize_step_sets(on)
    ref = make(PrePB, fc_hidden=32)
    dezero(ref)
    with torch.no_grad():
        for name in on.norm_names():
            src, dst = getattr(on, name), getattr(ref, name)
            src.weight.normal_(1.0, 0.2)
            src.bias.normal_(0.0, 0.2)
            dst.weight.copy_(src.weight)
            dst.bias.copy_(src.bias)
            dst.running_mean.copy_(getattr(on, f"{name}_step_mean")[1])
            dst.running_var.copy_(getattr(on, f"{name}_step_var")[1])
    on.eval()
    ref.eval()
    lat = latents(6, 80)
    with torch.no_grad():
        a = on(lat, on.initial_state(6, CPU), step=2)
        b = ref(lat, ref.initial_state(6, CPU))
    assert torch.allclose(a[1], b[1], atol=1e-5)
    assert torch.allclose(a[2][0], b[2][0], atol=1e-6)


def test_a_missing_or_bad_step_fails_loudly():
    net = make(ValuePrefixNetwork, per_step=5)
    lat = latents(4, 90)
    st = net.initial_state(4, CPU)
    for training in (False, True):
        net.train(training)
        with pytest.raises(ValueError, match="no LSTM step"):
            net(lat, st)
        for bad in (0, 6, torch.tensor([1, 2, 6, 1]), torch.tensor([0])):
            with pytest.raises(ValueError):
                net(lat, st, step=bad)
    net.eval()
    with pytest.raises(ValueError):
        net(lat, st, step=torch.tensor([1, 2]))         # wrong length
    net.train()
    with pytest.raises(ValueError, match="one step"):
        net(lat, st, step=torch.tensor([1, 2, 1, 1]))   # mixed in training
    with pytest.raises(ValueError, match="no LSTM step"):
        net(latents(1, 91), net.initial_state(1, CPU))  # single-row too


# --------------------------------------------------------------------------
# the factory, the arm, the gates
# --------------------------------------------------------------------------
def test_the_factory_follows_the_flag(monkeypatch):
    assert muzero.build_value_prefix().per_step == 0
    monkeypatch.setattr(muzero, "PREFIX_BN_PER_STEP", True)
    assert muzero.PREFIX_LSTM_HORIZON == muzero.UNROLL_STEPS == 5
    net = muzero.build_value_prefix()
    assert net.per_step == 5
    assert net.norm_out_step_mean.shape == (5, muzero.VALUE_PREFIX_HIDDEN)
    # a horizon the unroll does not train (or the reverse) is refused
    for horizon, unroll in ((6, 5), (3, 5)):
        monkeypatch.setattr(muzero, "PREFIX_LSTM_HORIZON", horizon)
        monkeypatch.setattr(muzero, "UNROLL_STEPS", unroll)
        with pytest.raises(ValueError, match="PREFIX_LSTM_HORIZON"):
            muzero.build_value_prefix()


def test_pb_arm_delta_is_exactly_the_flag():
    tk, pb = ablation.ARMS[TK_ARM], ablation.ARMS[PB_ARM]
    delta = {k for k in set(tk) | set(pb) if tk.get(k) != pb.get(k)}
    assert delta == {"PREFIX_BN_PER_STEP", "_doc"}
    assert pb["PREFIX_BN_PER_STEP"] is True
    assert "PREFIX_BN_PER_STEP" not in tk
    assert pb["_launch"] == {"ray": True, "cpu_actor": True}
    with ablation.arm_config(PB_ARM) as spec:
        assert muzero.PREFIX_BN_PER_STEP is True
        nets = ablation.build_networks(spec)
        assert nets["value_prefix"].per_step == muzero.PREFIX_LSTM_HORIZON
    assert muzero.PREFIX_BN_PER_STEP is False             # restored


def test_every_startup_gate_passes_for_pb(monkeypatch):
    for k in ablation.ARMS[PB_ARM]:
        if hasattr(muzero, k):
            monkeypatch.setattr(muzero, k, getattr(muzero, k))
    for k in ("ENV_NAME", "FRAMESKIP", "ACTIONS"):
        monkeypatch.setattr(ablation, k, getattr(ablation, k))
    ablation.ENV_NAME, ablation.FRAMESKIP, ablation.ACTIONS = ENV_ID, 4, 9
    with ablation.arm_config(PB_ARM) as spec, \
            ablation._run_length_globals(90, 2000), ablation.token_actions():
        assert ablation.ACTIONS == 36
        ablation.ray_scope_gate(spec)
        ablation.launch_gate(spec, True, True)
        ablation.eager_collect_gate(True)
        ablation.prior_temp_gate()
        ablation.bn_recal_gate(True)
        ablation.search_act_birth_gate()
        ablation.birth_search_gate()
        ablation.sve_gate()
        ablation.token_gate(spec, True, True, 0)
        ablation.reward_growth_gate(True, True, 0)
        assert spec.get("_student") is None
        ablation.student_gate(spec, None, True)


# --------------------------------------------------------------------------
# every caller passes the step
# --------------------------------------------------------------------------
def record_steps(monkeypatch, net, log):
    """Wrap net.forward: log (training, per-row steps or None) per call.
    Wraps the CLASS forward, so a second call replaces the first."""
    real = type(net).forward.__get__(net, type(net))

    def forward(latent, state, step=None):
        rows = None
        if step is not None:
            rows = torch.as_tensor(step).reshape(-1)
            if rows.numel() == 1:
                rows = rows.expand(latent.size(0))
            rows = [int(s) for s in rows]
        log.append((net.training, rows))
        return real(latent, state, step=step)

    monkeypatch.setattr(net, "forward", forward)


def test_the_training_unroll_takes_steps_1_to_unroll_steps(monkeypatch):
    """train_step: call i is step i (fresh state per unroll, no resets),
    every call in train mode; and the step-set bookkeeping leaves the
    training itself bit for bit the off head's."""
    from tests.test_reanalysis import build_trainer, fake_batch
    K = muzero.UNROLL_STEPS
    torch.manual_seed(0)
    off = build_trainer()
    on = copy.deepcopy(off)
    on.value_prefix = ValuePrefixNetwork(
        latent_channels=C, hidden_size=64,
        support_size=muzero.VALUE_PREFIX_SUPPORT,
        num_bins=muzero.VALUE_PREFIX_BINS, per_step=K)
    on.value_prefix.load_state_dict(off.value_prefix.state_dict(),
                                    strict=False)
    for t in (off, on):
        dezero(t.value_prefix)
        t.prefix_optimizer = muzero.make_optimizer(
            t.value_prefix.parameters())
    log = []
    record_steps(monkeypatch, on.value_prefix, log)
    np.random.seed(2)
    batch = fake_batch(4, K, np.ones((4, K + 1), dtype=bool))
    losses = []
    for t in (off, on):
        torch.manual_seed(9)
        losses.append(asyncio.run(t.train_step(batch)))
    assert log == [(True, [i] * 4) for i in range(1, K + 1)]
    assert losses[0] == losses[1]
    for name in ("representation", "dynamics", "prediction", "projector"):
        assert equal_states(getattr(off, name), getattr(on, name)), name
    a, b = off.value_prefix.state_dict(), on.value_prefix.state_dict()
    for k in a:
        assert torch.equal(a[k], b[k]), k
    for name in on.value_prefix.norm_names():
        assert getattr(on.value_prefix,
                       f"{name}_step_count").tolist() == [1] * K


def _search_world(monkeypatch, horizon):
    """A de-zeroed 6-action world with the flag on at `horizon` (= the
    unroll) and distinct per-step sets; the sequential helper and both
    engines share its nets."""
    from tests.test_vtree import dezero_heads
    for key, val in (("USE_GUMBEL_SEARCH", False),
                     ("USE_VALUE_PREFIX_IN_TREE", True),
                     ("MCTS_BATCHES", 1), ("DIRICHLET_ALPHA", 0.3),
                     ("MAX_DEPTH", 12), ("MINMAX_RANGE_FLOOR", 0.01),
                     ("COMMUNAL_WINDOW_ROOTS", 16),
                     ("PREFIX_LSTM_HORIZON", horizon),
                     ("UNROLL_STEPS", horizon),
                     ("PREFIX_BN_PER_STEP", True)):
        monkeypatch.setattr(muzero, key, val)
    monkeypatch.setattr(muzero.random, "choice", lambda seq: seq[0])
    monkeypatch.setattr(muzero, "add_dirichlet_noise",
                        lambda prior, epsilon=0.25, alpha=0.25: prior)
    torch.manual_seed(4)
    helper = muzero.MCTSHelper(action_space=6, device=CPU, network_ref=None,
                               storage_ref=None)
    nm = helper.network_manager
    nets = {n: getattr(nm, n) for n in ("representation", "dynamics",
                                        "prediction", "value_prefix")}
    dezero_heads(nets, seed=21)
    assert nets["value_prefix"].per_step == horizon
    randomize_step_sets(nets["value_prefix"])
    return helper, nets


def _expansions(node, depth, out):
    """(depth since the last state-less node, A) per expanded node."""
    if node.is_expanded and not node.done:
        out.append(depth)
    for child in node.children.values():
        if not child.is_gameplay:
            _expansions(child, depth + 1, out)
    return out


def oracle(depths, horizon, A=6):
    """The step every child row of an expansion at `depth` must take:
    the LSTM restarts at the state-less node and every `horizon` plies."""
    return sorted(s for d in depths for s in [d % horizon + 1] * A)


def flat(log):
    return [s for _, rows in log for s in rows]


@pytest.mark.parametrize("horizon,max_depth", [(2, 12), (3, 12), (2, 3)])
def test_fresh_root_engines_pass_each_rows_step(monkeypatch, horizon,
                                                max_depth):
    """The sequential engine (run_full_mcts) and VectorizedPUCT on the
    same root: every prefix call in eval mode, the root expansion at step
    1, step 2 below it, back to 1 every `horizon` plies (the depth
    oracle); the two engines' per-call steps identical; their visit
    counts and values exact with the flag on."""
    helper, nets = _search_world(monkeypatch, horizon)
    rng = np.random.default_rng(11)
    obs = rng.integers(0, 255, (2, 12, *muzero.IMAGE_RESHAPE),
                       dtype=np.uint8)
    engine = VectorizedPUCT(nets, 6, CPU)
    # The search releases its root's subtree at the readout: read the
    # expansion depths off the tree just before.
    depths = []
    real_release = helper.release_root

    def release_root(node):
        depths[:] = _expansions(node, 0, [])
        return real_release(node)

    monkeypatch.setattr(helper, "release_root", release_root)
    for t in range(len(obs)):
        seq_log, vec_log = [], []
        record_steps(monkeypatch, nets["value_prefix"], seq_log)
        latent, policies, values = helper.network_manager.batch_inference(
            obs[t:t + 1])
        root = muzero.MCTSNode(latent=latent[0:1], prior=1.0)
        root.policy = policies[0]
        root.expanded_value = float(values[0])
        helper.current_trajectory = [root]
        pol, val, has = asyncio.run(helper.run_full_mcts(
            0, num_simulations=16, sim_batches=1, c_puct=1.25, discount=0.99,
            max_depth=max_depth, root_noise=False))
        helper.current_trajectory = None
        assert depths and depths[0] == 0
        assert all(not tr for tr, _ in seq_log)
        assert seq_log[0][1] == [1] * 6
        assert sorted(flat(seq_log)) == oracle(depths, horizon)
        if max_depth > horizon:
            assert max(depths) >= horizon      # a horizon reset happened
        record_steps(monkeypatch, nets["value_prefix"], vec_log)
        _, v, c = engine.search(obs[t:t + 1], num_simulations=16,
                                c_puct=1.25, discount=0.99,
                                max_depth=max_depth, root_noise=False,
                                tie_break="first")
        assert [rows for _, rows in vec_log] == \
            [rows for _, rows in seq_log]
        assert c[0].astype(int).tolist() == \
            np.round(pol * 16).astype(int).tolist()
        assert float(v[0]) == pytest.approx(val, rel=0, abs=1e-12)


def _communal_sequential(helper, traj, window, sims):
    """tests.test_vtree_communal.sequential_windows for one window, with
    the chain's expansion depths read off the tree before release."""
    np.random.seed(5)
    L = len(traj["actions"])
    evaluated = helper.eval_obs([(traj["obs"][i], traj["actions"][i],
                                  traj["rewards"][i]) for i in range(L)])
    helper.create_tree_from_game(evaluated, terminated=traj["terminated"],
                                 seed_values=True)
    res = asyncio.run(helper.run_communal_mcts(
        window, num_simulations=sims, c_puct=1.25, discount=0.99,
        chain_seeded=True))
    raw = {idx: [helper.current_trajectory[idx].children[a].visit_count
                 if a in helper.current_trajectory[idx].children else 0
                 for a in range(6)] for idx in window}
    depths = []
    for node in helper.current_trajectory:      # every chain node: depth 0
        _expansions(node, 0, depths)
    helper.release_span(0, L - 1)
    helper.current_trajectory = None
    return res, raw, depths


@pytest.mark.parametrize("horizon", [2, 3])
def test_communal_engines_pass_each_rows_step(monkeypatch, horizon):
    """run_communal_mcts and VectorizedCommunalPUCT on one shared chain
    tree: chain nodes (gameplay restarts) expand at step 1, imagined
    nodes by their depth below the chain (the oracle); both engines'
    step sequences identical; parity exact with the flag on."""
    from tests.test_vtree_communal import (assert_parity, make_traj,
                                           vectorized_windows)
    helper, nets = _search_world(monkeypatch, horizon)
    traj = make_traj(4, 10, True)
    window = [1, 2, 3, 4]
    seq_log, vec_log = [], []
    record_steps(monkeypatch, nets["value_prefix"], seq_log)
    seq, raw, depths = _communal_sequential(helper, traj, window, 12)
    assert all(not tr for tr, _ in seq_log)
    assert sorted(flat(seq_log)) == oracle(depths, horizon)
    assert depths.count(0) >= len(window)       # the chain roots
    assert max(depths) >= horizon
    record_steps(monkeypatch, nets["value_prefix"], vec_log)
    engine = VectorizedCommunalPUCT(nets, 6, CPU)
    vec, _ = vectorized_windows(engine, traj, [window], 12)
    assert flat(vec_log) == flat(seq_log)
    assert all(not tr for tr, _ in vec_log)
    assert_parity(seq, raw, vec)


def _clone(net):
    """A deep copy running the class forward (a record_steps wrapper on
    the original would be copied too, and call back into the original)."""
    out = copy.deepcopy(net)
    out.__dict__.pop("forward", None)
    return out


def _step1_panel(rep, dyn, vp, batches):
    """The diagnostics' step-1 fields recomputed from the batches they
    drew: the head in eval mode from a fresh state, step 1, against
    truth[:, 1] (= rewards[:, 1] at discount 1; truth[:, 0] is 0); and
    the two wrong readings -- the step-K prefix, truth[:, 0:1]."""
    K = muzero.UNROLL_STEPS
    vp.eval()
    got = {"want": 0.0, "control": 0.0, "step_k": 0.0, "truth0": 0.0}
    with torch.no_grad():
        for b in batches:
            obs = torch.tensor(b["obs"], dtype=torch.float32) / 255.0
            acts = torch.tensor(b["actions"], dtype=torch.long)
            t1 = torch.tensor(b["rewards"], dtype=torch.float32)[:, 1:2]
            latent = dyn(rep(obs[:, 0]), acts[:, 0])[0]
            p1, _, state = vp(latent, vp.initial_state(len(obs), CPU),
                              step=1)
            pk = p1
            for i in range(2, K + 1):
                latent = dyn(latent, acts[:, i - 1])[0]
                pk, _, state = vp(latent, state, step=i)
            got["want"] += (p1 - t1).abs().mean().item()
            got["control"] += t1.abs().mean().item()
            got["step_k"] += (pk - t1).abs().mean().item()
            got["truth0"] += p1.abs().mean().item()
    return {k: v / len(batches) for k, v in got.items()}


def test_the_diagnostics_pass_steps_in_eval_mode_and_log_step_1(
        monkeypatch):
    """evaluate_diagnostics: per batch, steps 1..UNROLL_STEPS from a fresh
    state, the head in EVAL mode (the search's function) even when called
    in train mode -- restored after -- and the new fields beside the
    step-K ones (tests/test_snapshot_weights.py pins the panel's RNG and
    net-state use). prefix_step1 / _control equal an independent
    recomputation on the panel's own batches (captured at prepare_batch)
    and differ from the step-K prefix's and truth[:, 0]'s readings."""
    from tests.test_checkpoint_resume import make_run
    from tests.test_reanalyze_refresh import obs_traj
    monkeypatch.setattr(ablation, "DEVICE", CPU)
    K = muzero.UNROLL_STEPS
    for per_step in (0, K):
        run = make_run()
        if per_step:
            run.trainer.value_prefix = ValuePrefixNetwork(
                latent_channels=C, hidden_size=64,
                support_size=muzero.VALUE_PREFIX_SUPPORT,
                num_bins=muzero.VALUE_PREFIX_BINS, per_step=K)
            randomize_step_sets(run.trainer.value_prefix)
        dezero(run.trainer.value_prefix)
        np.random.seed(4)
        run.replay = [obs_traj([0.0, 1.0] * 6, [0.5] * 12)]
        drawn = []
        real_prepare = run.trainer.prepare_batch

        async def prepare_batch(*args, **kwargs):
            b = await real_prepare(*args, **kwargs)
            drawn.append(copy.deepcopy(b))
            return b

        monkeypatch.setattr(run.trainer, "prepare_batch", prepare_batch)
        for training in (False, True):
            run.set_train_mode(training)
            # Pre-call copies in the call's modes: train-mode BN outputs
            # are batch statistics, so the panel's running-stat updates
            # do not change what the copies compute.
            nets = [_clone(getattr(run.trainer, n)) for n in
                    ("representation", "dynamics", "value_prefix")]
            drawn.clear()
            log = []
            record_steps(monkeypatch, run.trainer.value_prefix, log)
            errs = run.evaluate_diagnostics(batches=2)
            assert log == [(False, [i] * 32) for i in range(1, K + 1)] * 2
            assert run.trainer.value_prefix.training is training
            for k in ("prefix_step1", "prefix_step1_control", "prefix",
                      "control"):
                assert np.isfinite(errs[k]), k
            assert errs["prefix_step1_control"] > 0
            assert len(drawn) == 2
            want = _step1_panel(*nets, drawn)
            assert errs["prefix_step1"] == pytest.approx(
                want["want"], rel=1e-6, abs=1e-9), (per_step, training)
            assert errs["prefix_step1_control"] == pytest.approx(
                want["control"], rel=1e-6, abs=1e-9)
            # a step-K prefix or truth[:, 0:1] would read differently
            for wrong in ("step_k", "truth0"):
                assert want[wrong] != pytest.approx(want["want"],
                                                    rel=1e-3), wrong


def test_redo_passes_steps_and_resets_recycled_step_channels(monkeypatch):
    import redo
    monkeypatch.setattr(ablation, "DEVICE", CPU)
    with ablation.arm_config(PB_ARM) as spec:
        nets = ablation.build_networks(spec)
    vp = nets["value_prefix"]
    randomize_step_sets(vp)
    log = []
    real_deepcopy = copy.deepcopy

    def spy(obj, memo=None):
        out = real_deepcopy(obj, memo)
        if obj is vp:
            record_steps(monkeypatch, out, log)
        return out

    monkeypatch.setattr(redo.copy, "deepcopy", spy)
    units = redo.recyclable_units(nets, include=("value_prefix",))
    obs = torch.rand(4, 12, *muzero.IMAGE_RESHAPE)
    acts = torch.randint(0, ablation.ACTIONS, (4, muzero.UNROLL_STEPS))
    scores, raw, inputs = redo.dormancy_stats(nets, obs, acts, units)
    assert log == [(True, [i] * 4) for i in range(1, muzero.UNROLL_STEPS + 1)]
    u = next(u for u in units if u.norm == "norm_in")
    s = {u: torch.ones_like(scores[u])}
    s[u][3] = 0.0
    before_m = vp.norm_in_step_mean.clone()
    before_v = vp.norm_in_step_var.clone()
    opts = [muzero.make_optimizer(n.parameters()) for n in nets.values()]
    redo.recycle(nets, opts, s, 0.1, torch.Generator().manual_seed(0))
    assert torch.equal(vp.norm_in_step_mean[:, 3], torch.zeros(5))
    assert torch.equal(vp.norm_in_step_var[:, 3], torch.ones(5))
    keep = [c for c in range(vp.norm_in.num_features) if c != 3]
    assert torch.equal(vp.norm_in_step_mean[:, keep], before_m[:, keep])
    assert torch.equal(vp.norm_in_step_var[:, keep], before_v[:, keep])


# --------------------------------------------------------------------------
# the inference server path
# --------------------------------------------------------------------------
def test_the_inference_server_batches_each_payloads_steps(
        monkeypatch, restore_rngs):  # noqa: F811
    import ray_actors
    over = {k: v for k, v in ablation.ARMS[PB_ARM].items()
            if not k.startswith("_")}
    sched = {"TEMPERATURE_TOTAL_STEPS": 1000, "OFFPOLICY_DECAY_STEPS": None}
    for k in list(over) + list(sched):
        if hasattr(muzero, k):
            monkeypatch.setattr(muzero, k, getattr(muzero, k))
    srv = ray_actors.InferenceServer(over, 36, sched, device="cpu")
    for which in ("live", "target", "eval"):
        assert srv.nets[which]["value_prefix"].per_step == 5
    vp = srv.nets["live"]["value_prefix"]
    dezero(vp)
    randomize_step_sets(vp)
    lat = latents(6, 100, vp.conv.in_channels).numpy()
    h = np.zeros((1, 6, vp.hidden_size), dtype=np.float32)
    payloads = [(lat[:2], h[:, :2], h[:, :2], 1),
                (lat[2:], h[:, 2:], h[:, 2:], np.array([2, 3, 3, 5]))]
    outs = srv._run("value_prefix", payloads)
    rows = torch.tensor([1, 1, 2, 3, 3, 5])
    with torch.no_grad():
        want = vp(torch.tensor(lat), (torch.tensor(h), torch.tensor(h)),
                  step=rows)[0].numpy()
    got = np.concatenate([o[0] for o in outs])
    assert np.array_equal(got, want)
    with pytest.raises(ValueError, match="no LSTM step"):
        srv._run("value_prefix", [(lat[:2], h[:, :2], h[:, :2])])
    # the worker proxy forwards the step over the wire
    sent = {}

    class Owner:
        prefix_hidden = (1, vp.hidden_size)

        def _call(self, method, *args):
            sent[method] = args
            return outs[1]

    mod = ray_actors._RemoteModule(Owner(), "value_prefix")
    mod(torch.tensor(lat[2:]), (torch.tensor(h[:, 2:]),
                                torch.tensor(h[:, 2:])),
        step=torch.tensor([2, 3, 3, 5]))
    assert sent["prefix_step"][3].tolist() == [2, 3, 3, 5]


# --------------------------------------------------------------------------
# the flag reaches the actors; the buffers ride every ship and load
# --------------------------------------------------------------------------
SCHEDULE = {"TEMPERATURE_TOTAL_STEPS": 180000, "OFFPOLICY_DECAY_STEPS": 54000}


def _pb_overrides(monkeypatch):
    """The arm's overrides as RayPipeline ships them; the muzero / ablation
    globals the actors setattr pinned first, so they restore."""
    over = {k: v for k, v in ablation.ARMS[PB_ARM].items()
            if not k.startswith("_")}
    for k in list(over) + list(SCHEDULE):
        if hasattr(muzero, k):
            monkeypatch.setattr(muzero, k, getattr(muzero, k))
    for k in ("ACTIONS", "EPISODE_CAP", "ENV_NAME", "FRAMESKIP", "DEVICE"):
        monkeypatch.setattr(ablation, k, getattr(ablation, k))
    return over


def _pb_states():
    """A driver-side PB value-prefix state with non-trivial step sets."""
    with ablation.arm_config(PB_ARM):
        vp = muzero.build_value_prefix()
    randomize_step_sets(vp)
    return vp.state_dict()


def test_the_actors_build_and_load_the_per_step_head(
        monkeypatch, restore_rngs):  # noqa: F811
    """SelfPlayActor (the --cpu-actor lane) and RefreshActor apply the
    overrides BEFORE building their nets, so their heads carry the step
    buffers, and the driver's state loads strictly into them."""
    import ray_actors
    over = _pb_overrides(monkeypatch)
    torch.set_num_threads(2)
    state = _pb_states()
    actor = ray_actors.SelfPlayActor(over, 36, 0, SCHEDULE, ENV_ID, 4, 200,
                                     server=None, force_cpu=True)
    try:
        assert actor.get_constant("PREFIX_BN_PER_STEP") is True
        vp = actor.searcher.network_manager.value_prefix
        assert vp.per_step == 5 and not vp.training
        actor.set_weights({"value_prefix": state}, 10)
        assert torch.equal(vp.norm_out_step_var, state["norm_out_step_var"])
        assert not vp.training
    finally:
        actor.env.close()
    ref = ray_actors.RefreshActor(over, 36, 0, SCHEDULE)
    vp = ref.searcher.network_manager.value_prefix
    assert vp.per_step == 5
    ref.set_weights({"value_prefix": state}, 10)
    assert torch.equal(vp.norm_in_step_mean.cpu(), state["norm_in_step_mean"])


def _pb_run(make_run):
    """make_run's trainer with a PB head (and its optimizer)."""
    run = make_run()
    run.trainer.value_prefix = ValuePrefixNetwork(
        latent_channels=C, hidden_size=64,
        support_size=muzero.VALUE_PREFIX_SUPPORT,
        num_bins=muzero.VALUE_PREFIX_BINS, per_step=5)
    run.trainer.prefix_optimizer = muzero.make_optimizer(
        run.trainer.value_prefix.parameters())
    return run


def test_checkpoint_and_target_model_carry_the_step_sets(
        tmp_path, restore_rngs):  # noqa: F811
    from tests.test_checkpoint_resume import EXPECT, META, make_run
    from tests.test_reanalyze_refresh import obs_traj
    np.random.seed(3)
    a = _pb_run(make_run)
    randomize_step_sets(a.trainer.value_prefix)
    a.trainer.training_steps = META["training_steps"]
    a.replay = [obs_traj([1.0] * 4, [2.0] * 4)]
    a.trainer._ensure_target_model()
    tgt = a.trainer.target_value_prefix
    assert tgt.per_step == 5 and not tgt.training
    assert equal_states(tgt, a.trainer.value_prefix)   # the synced copy
    path = tmp_path / "resume.ckpt"
    ablation.save_run_checkpoint(a, path, META)
    b = _pb_run(make_run)
    ablation.load_run_checkpoint(b, path, EXPECT)
    for n in ablation.NET_NAMES:
        assert equal_states(getattr(a.trainer, n), getattr(b.trainer, n)), n
    assert equal_states(a.trainer.target_value_prefix,
                        b.trainer.target_value_prefix)
    # an off run cannot silently drop the sets: the strict load refuses
    with pytest.raises(RuntimeError, match="step_mean"):
        ablation.load_run_checkpoint(make_run(), path, EXPECT)


@pytest.fixture
def pb_runs(monkeypatch, tmp_path):
    """Real Runs (Ms. Pac-Man, 36 tokens) under a given arm; snapshots go
    to tmp_path."""
    _pb_overrides(monkeypatch)
    monkeypatch.setattr(ablation, "RESULTS", tmp_path)
    monkeypatch.setattr(ablation, "TAG", "")
    ablation.ENV_NAME, ablation.FRAMESKIP, ablation.ACTIONS = ENV_ID, 4, 9
    ablation.DEVICE = CPU
    torch.set_num_threads(2)
    made = []

    def build(arm):
        with ablation.arm_config(arm) as spec, \
                ablation._run_length_globals(90, 2000), \
                ablation.token_actions():
            run = ablation.Run(arm, 0, spec)
        made.append(run)
        return run

    yield build
    for run in made:
        run.env.close()


def test_snapshots_load_under_the_snapshots_arm(
        pb_runs, restore_rngs):  # noqa: F811
    """save_weight_snapshot / weights_blob carry the step sets; a PB Run,
    load_init_weights and distill.load_master load them strictly; a TK
    Run refuses them loudly."""
    import distill
    pb = pb_runs(PB_ARM)
    assert pb.trainer.value_prefix.per_step == 5
    assert pb.searcher.network_manager.value_prefix is pb.trainer.value_prefix
    randomize_step_sets(pb.trainer.value_prefix)
    spec = ablation.ARMS[PB_ARM]
    path = ablation.save_weight_snapshot(pb, PB_ARM, 0, spec, 30)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    assert blob["spec"]["PREFIX_BN_PER_STEP"] is True
    assert "norm_out_step_mean" in blob["value_prefix"]
    other = pb_runs(PB_ARM)
    for n in ablation.NET_NAMES:
        getattr(other.trainer, n).load_state_dict(blob[n])
    assert equal_states(other.trainer.value_prefix, pb.trainer.value_prefix)
    fresh = pb_runs(PB_ARM)
    ablation.load_init_weights(fresh.trainer, path, arm=PB_ARM)
    assert equal_states(fresh.trainer.value_prefix, pb.trainer.value_prefix)
    master = pb_runs(PB_ARM)
    with ablation.arm_config(PB_ARM):
        distill.load_master(master, blob)
    assert equal_states(master.trainer.value_prefix, pb.trainer.value_prefix)
    tk = pb_runs(TK_ARM)
    assert tk.trainer.value_prefix.per_step == 0
    with pytest.raises(RuntimeError, match="step_mean"):
        tk.trainer.value_prefix.load_state_dict(blob["value_prefix"])


def test_the_exact_eval_probe_loads_a_pb_snapshot(
        probe, monkeypatch, tmp_path, restore_rngs):  # noqa: F811
    """probe_exhaustive_eval builds its nets with the snapshot's prefix
    layout (its arm is fixed), so a PB snapshot loads; one call never
    mixes layouts."""
    pe = probe
    torch.set_num_threads(2)
    monkeypatch.setattr(muzero, "PREFIX_BN_PER_STEP", True)
    with ablation.arm_config(pe.ARM), ablation._run_length_globals(90, 2000):
        run = ablation.Run(pe.ARM, 0, ablation.ARMS[pe.ARM])
        assert run.trainer.value_prefix.per_step == 5
        snap = tmp_path / "run-p" / "snaps" / "weights-iter005.pth"
        snap.parent.mkdir(parents=True)
        torch.save({**{n: getattr(run.trainer, n).state_dict()
                       for n in ablation.NET_NAMES},
                    "training_steps": 0,
                    "spec": {"PREFIX_BN_PER_STEP": True}}, snap)
        run.env.close()
    monkeypatch.setattr(muzero, "PREFIX_BN_PER_STEP", False)
    real_play = ablation.Run.play_episode
    monkeypatch.setattr(ablation.Run, "play_episode",
                        lambda self, greedy=False, cap=None:
                        real_play(self, greedy=greedy, cap=40))
    pe.main([str(snap)])
    assert (tmp_path / "probe_exhaustive_eval-run-p-noops1.jsonl").exists()
    assert muzero.PREFIX_BN_PER_STEP is False                 # restored
    off = tmp_path / "off.pth"
    blob = torch.load(snap, weights_only=False)
    blob["spec"] = {}
    torch.save(blob, off)
    with pytest.raises(SystemExit, match="PREFIX_BN_PER_STEP"):
        pe.main([str(snap), str(off)])


@pytest.fixture
def prefix_probe(monkeypatch):
    """benchmarks/probes/probe_prefix_bn.py as a module; its import-time
    CUDA_VISIBLE_DEVICES and thread count undone after."""
    import os
    threads = torch.get_num_threads()
    cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    probes = ROOT / "benchmarks" / "probes"
    sys.path.insert(0, str(probes))
    try:
        import probe_prefix_bn as pp
    finally:
        sys.path.remove(str(probes))
        if cuda is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda
    for k in ("ACTIONS", "EPISODE_CAP", "ENV_NAME", "FRAMESKIP"):
        monkeypatch.setattr(ablation, k, getattr(ablation, k))
    yield pp
    torch.set_num_threads(threads)


def _offset_sd(x, mean, var):
    """The channel mean of |x's batch mean - mean| / sqrt(var), written
    out here rather than taken from the probe."""
    sd = var.sqrt().clamp(min=1e-5)
    return float(((x.mean(dim=0) - mean).abs() / sd).mean())


def test_the_prefix_probe_reads_a_snapshots_own_step_1_statistics(
        prefix_probe, pb_runs, tmp_path, capsys, restore_rngs):  # noqa: F811
    """PB read (1)'s probe on a PB snapshot whose step sets and pooled
    statistics all differ: p1_abs_eval is the head's eval output on step
    1's set (not step 2's, not the pooled statistics), p1_abs_trainBN
    its batch-statistics value, and normout_offset_sd_step{1,2} read sets
    [0] / [1] (not the pooled, not another index) -- each against an
    independent computation on copies of the saved nets, same states."""
    import json
    pb = pb_runs(PB_ARM)
    vp = pb.trainer.value_prefix
    dezero(vp)
    with torch.no_grad():
        for name in vp.norm_names():
            bn = getattr(vp, name)
            bn.running_mean.normal_(0, 0.2)
            bn.running_var.uniform_(0.5, 2.0)
    randomize_step_sets(vp)
    spec = ablation.ARMS[PB_ARM]
    path = ablation.save_weight_snapshot(pb, PB_ARM, 0, spec, 30)
    states = tmp_path / "states.npz"
    rng = np.random.default_rng(0)
    obs = rng.integers(0, 255, (8, 12, *muzero.IMAGE_RESHAPE),
                       dtype=np.uint8)
    np.savez_compressed(states, obs=obs)
    out = tmp_path / "pp.jsonl"
    prefix_probe.main(["--states", str(states), "--out", str(out),
                       str(path)])
    rec = json.loads(out.read_text().splitlines()[0])
    assert rec["per_step"] == 5 and rec["arm"] == PB_ARM
    assert rec["iteration"] == 30 and rec["n_states"] == 8
    for k in ("p1_abs_eval", "p1_abs_trainBN", "p1_ratio",
              "normout_offset_sd_step1", "normout_offset_sd_step1_pooled",
              "p1_allactions_abs_eval"):
        assert np.isfinite(rec[k]), k
    assert rec["fixed"] == (rec["p1_ratio"] <= 2.0)
    assert json.loads(capsys.readouterr().out.splitlines()[-1]) == rec

    # The same reads, here, on eval-mode copies of the snapshot's nets.
    rep, dyn, pred, head = (
        copy.deepcopy(getattr(pb.trainer, n)).eval() for n in
        ("representation", "dynamics", "prediction", "value_prefix"))
    pooled = copy.deepcopy(head)       # every step set := pooled stats
    with torch.no_grad():
        for name in pooled.norm_names():
            bn = getattr(pooled, name)
            getattr(pooled, f"{name}_step_mean")[:] = bn.running_mean
            getattr(pooled, f"{name}_step_var")[:] = bn.running_var
    seen = []
    hook = head.lstm.register_forward_hook(
        lambda m, i, o: seen.append(o[0].squeeze(0)))
    with torch.no_grad():
        l0 = rep(torch.tensor(obs, dtype=torch.float32) / 255.0)
        l1 = dyn(l0, pred(l0)[0].argmax(1))[0]
        l2 = dyn(l1, pred(l1)[0].argmax(1))[0]
        s0 = head.initial_state(len(obs), CPU)
        p1, _, s1 = head(l1, s0, step=1)
        head(l2, s1, step=2)
        hook.remove()
        x1, x2 = seen             # the post-LSTM BN's inputs, steps 1, 2
        p1_set2 = head(l1, s0, step=2)[0]
        p1_pooled = pooled(l1, s0, step=1)[0]
        p1_train = copy.deepcopy(head).train()(l1, s0, step=1)[0]
    p1_abs = float(p1.abs().mean())
    assert rec["p1_abs_eval"] == pytest.approx(p1_abs, rel=1e-5)
    for wrong in (p1_set2, p1_pooled):
        assert float(wrong.abs().mean()) != pytest.approx(p1_abs, rel=1e-3)
    assert rec["p1_abs_trainBN"] == pytest.approx(
        float(p1_train.abs().mean()), rel=1e-5)
    m, v = head.norm_out_step_mean, head.norm_out_step_var
    rm, rv = head.norm_out.running_mean, head.norm_out.running_var
    for s, xs in ((1, x1), (2, x2)):
        own = _offset_sd(xs, m[s - 1], v[s - 1])
        assert rec[f"normout_offset_sd_step{s}"] == pytest.approx(
            own, rel=1e-5), s
        assert rec[f"normout_offset_sd_step{s}_pooled"] == pytest.approx(
            _offset_sd(xs, rm, rv), rel=1e-5), s
        others = [_offset_sd(xs, m[j], v[j]) for j in range(len(m))
                  if j != s - 1] + [_offset_sd(xs, rm, rv)]
        for j, wrong in enumerate(others):
            assert wrong != pytest.approx(own, rel=1e-3), (s, j)


# --------------------------------------------------------------------------
# flag on, production wrappers, real size
# --------------------------------------------------------------------------
def test_a_pb_runs_birth_search_steps_every_prefix_call(
        pb_runs, monkeypatch, restore_rngs):  # noqa: F811
    """Smoke: a real PB Run's _birth_search -- the arm's target search,
    VectorizedPUCT over the live nets in eval mode, at the arm's sims and
    36 tokens -- on a short policy-only trajectory completes, and every
    value-prefix call it makes is eval mode with a per-row step in
    1..PREFIX_LSTM_HORIZON (a missing step would raise in the head)."""
    pb = pb_runs(PB_ARM)
    vp = pb.trainer.value_prefix
    dezero(vp)
    randomize_step_sets(vp)
    rng = np.random.default_rng(6)
    traj = [(rng.integers(0, 255, (12, *muzero.IMAGE_RESHAPE),
                          dtype=np.uint8), 0, 0.0) for _ in range(4)]
    log = []
    record_steps(monkeypatch, vp, log)
    mode = vp.training
    with ablation.arm_config(PB_ARM), ablation._run_length_globals(90, 2000), \
            ablation.token_actions():
        K, A = muzero.PREFIX_LSTM_HORIZON, ablation.ACTIONS
        assert muzero.PREFIX_BN_PER_STEP and vp.per_step == K and A == 36
        out = pb._birth_search(traj, False)
    assert [len(r) for r in out] == [5] * len(traj)
    for _, _, _, pol, val in out:
        assert pol.shape == (A,) and pol.sum() == pytest.approx(1.0)
        assert np.isfinite(val)
    assert pb._birth_stats["birth_positions_last"] == len(traj)
    assert log and all(not tr and rows for tr, rows in log)
    steps = flat(log)
    assert set(steps) <= set(range(1, K + 1))
    assert 1 in steps and 2 in steps          # root expansions and below
    assert vp.training is mode                # the head's mode restored
