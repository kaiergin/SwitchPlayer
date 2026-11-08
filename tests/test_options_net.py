"""Test O stage 1 (CURRICULUM.md "Test O", DESIGN "Heads" and "Dynamics"):
the option heads and the jump dynamics. Pins: OPTION_LENGTH 1 is the
pre-Test-O class bit for bit (state_dict keys, shapes, values, outputs and
RNG use, against networks.py at 7c60e1a6); L > 1 adds only its own keys,
built last so every existing layer inits as at L = 1; OptionZero's head
init (network_unit.py 38-44: default weights, bias 0 with 10 on stop) and
head 0 = policy + a -100 stop logit (61-66); a jump's encoding is the sum
of per-position rows, -1 adding nothing; the factories and the target
model follow the knob. CPU only."""
import importlib.util
import pathlib
import subprocess

import pytest
import torch

import muzero
from networks import DynamicsNetwork, PredictionNetwork

ROOT = pathlib.Path(__file__).resolve().parents[1]
PRE_O = "7c60e1a6"      # the build tree's head before Test O's option code
C, A = 16, 5
CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def pre_o(tmp_path_factory):
    """networks.py as it was before the option code (git show; imported
    from a temp dir, its own scalar_categorical import resolving to the
    tree's as ever)."""
    try:
        src = subprocess.run(["git", "show", f"{PRE_O}:networks.py"],
                             cwd=ROOT, capture_output=True, text=True,
                             check=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"git show {PRE_O}:networks.py unavailable: {exc}")
    assert "option_length" not in src
    path = tmp_path_factory.mktemp("pre_o") / "networks_pre_o.py"
    path.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("networks_pre_o", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def latents(n, seed=0, channels=C):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, channels, 6, 6, generator=g)


def same_state(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    return (list(sa) == list(sb)
            and all(sa[k].shape == sb[k].shape and torch.equal(sa[k], sb[k])
                    for k in sa))


def build_after_seed(cls, seed, **kw):
    torch.manual_seed(seed)
    net = cls(**kw)
    return net, torch.get_rng_state()


PRED_KW = [dict(latent_channels=C, action_space=A, support_size=10,
                num_bins=21, head_channels=8),
           dict(latent_channels=C, action_space=A, support_size=10,
                num_bins=21, head_channels=8, fc_hidden=12)]


def dezero_policy(net, seed=3):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for fc in (net.policy_fc, net.value_fc):
            for p in fc.parameters():
                p.normal_(0, 0.5, generator=g)


# --------------------------------------------------------------------------
# L = 1: the pre-Test-O classes, bit for bit
# --------------------------------------------------------------------------
@pytest.mark.parametrize("kw", PRED_KW)
def test_l1_prediction_is_the_pre_o_class_bit_for_bit(pre_o, kw):
    old, rng_old = build_after_seed(pre_o.PredictionNetwork, 7, **kw)
    new, rng_new = build_after_seed(PredictionNetwork, 7, **kw,
                                    option_length=1)
    assert torch.equal(rng_old, rng_new), "L = 1 drew RNG the old class did not"
    assert same_state(old, new)
    assert not hasattr(new, "option_convs") and not hasattr(new, "option_fcs")
    dezero_policy(old)
    dezero_policy(new)
    x = latents(4, 1)
    for net in (old, new):
        net.eval()
    with torch.no_grad():
        a, b = old(x), new(x)
    assert len(b) == 3
    assert all(torch.equal(u, v) for u, v in zip(a, b))


@pytest.mark.parametrize("encoding", ["embed", "plane", "onehot"])
def test_l1_dynamics_is_the_pre_o_class_bit_for_bit(pre_o, encoding):
    kw = dict(latent_channels=C, action_space=A, support_size=10,
              num_bins=21, action_encoding=encoding, reward_channels=8)
    old, rng_old = build_after_seed(pre_o.DynamicsNetwork, 9, **kw)
    new, rng_new = build_after_seed(DynamicsNetwork, 9, **kw,
                                    option_length=1)
    assert torch.equal(rng_old, rng_new)
    assert same_state(old, new)
    assert not hasattr(new, "option_embeddings")
    with torch.no_grad():
        for net in (old, new):
            net.reward_fc.weight.normal_(0, 0.5,
                                         generator=torch.Generator().manual_seed(2))
            net.eval()
        x = latents(5, 2)
        act = torch.tensor([0, 4, 2, 1, 3])
        a, b = old(x, act), new(x, act)
    assert all(torch.equal(u, v) for u, v in zip(a, b))


def test_the_default_factories_build_l1(monkeypatch):
    assert muzero.OPTION_LENGTH == 1 and muzero.OPTION_EDGE is True
    pred = muzero.build_prediction(A)
    dyn = muzero.build_dynamics(A)
    assert pred.option_length == 1 and dyn.option_length == 1
    assert not any(k.startswith("option_") for k in pred.state_dict())
    assert not any(k.startswith("option_") for k in dyn.state_dict())


# --------------------------------------------------------------------------
# L > 1: the option heads
# --------------------------------------------------------------------------
@pytest.mark.parametrize("kw", PRED_KW)
def test_l3_adds_only_its_heads_and_inits_every_old_layer_as_l1(kw):
    torch.manual_seed(11)
    one = PredictionNetwork(**kw, option_length=1)
    torch.manual_seed(11)
    three = PredictionNetwork(**kw, option_length=3)
    s1, s3 = one.state_dict(), three.state_dict()
    extra = [k for k in s3 if k not in s1]
    assert list(s3)[:len(s1)] == list(s1), "new keys must come after the old"
    assert all(torch.equal(s1[k], s3[k]) for k in s1)
    want = []
    for i in range(2):
        want += [f"option_convs.{i}.weight", f"option_convs.{i}.bias"]
        if kw.get("fc_hidden"):
            want += [f"option_hiddens.{i}.weight", f"option_hiddens.{i}.bias"]
        want += [f"option_fcs.{i}.weight", f"option_fcs.{i}.bias"]
    assert sorted(extra) == sorted(want)


@pytest.mark.parametrize("kw", PRED_KW)
def test_option_head_init_is_optionzeros(kw):
    """network_unit.py 38-44 at our head sizes: each extra head its own
    1x1 conv (head_channels), [hidden], fc over A + 1 classes; default
    weights, bias 0 with 10 on the stop class; the policy head keeps its
    EfficientZero zero-init."""
    torch.manual_seed(5)
    net = PredictionNetwork(**kw, option_length=3)
    assert len(net.option_convs) == len(net.option_fcs) == 2
    for i in range(2):
        conv, fc = net.option_convs[i], net.option_fcs[i]
        assert conv.out_channels == net.policy_conv.out_channels
        assert fc.out_features == A + 1
        assert fc.in_features == net.policy_fc.in_features
        want = torch.zeros(A + 1)
        want[A] = 10.0
        assert torch.equal(fc.bias.data, want)
        assert fc.weight.abs().sum() > 0, "option fc weights must be default-init"
        # default nn.Linear init: kaiming_uniform(a=sqrt(5)) -> |w| <= 1/sqrt(fan_in)
        assert fc.weight.abs().max() <= fc.in_features ** -0.5 + 1e-7
        if kw.get("fc_hidden"):
            assert net.option_hiddens[i].out_features == kw["fc_hidden"]
    assert net.policy_fc.weight.abs().sum() == 0
    assert net.policy_fc.bias.abs().sum() == 0


def test_with_options_returns_head0_as_policy_plus_stop_and_leaves_the_rest():
    torch.manual_seed(5)
    net = PredictionNetwork(**PRED_KW[0], option_length=3).eval()
    dezero_policy(net)
    x = latents(3, 4)
    with torch.no_grad():
        plain = net(x)
        full = net(x, with_options=True)
    assert len(plain) == 3 and len(full) == 4
    assert all(torch.equal(u, v) for u, v in zip(plain, full[:3]))
    opt = full[3]
    assert opt.shape == (3, 3, A + 1)
    assert torch.equal(opt[:, 0, :A], plain[0])
    assert torch.equal(opt[:, 0, A], torch.full((3,), -100.0))
    # at init the extra heads put almost all their mass on stop (bias 10)
    probs = torch.softmax(opt, dim=2)
    assert (probs[:, 1:, A] > 0.99).all()
    assert (probs[:, 0, A] < 1e-30).all()


def test_option_heads_read_the_trunk_and_train():
    torch.manual_seed(5)
    net = PredictionNetwork(**PRED_KW[1], option_length=3)
    x = latents(4, 6)
    _, _, _, opt = net(x, with_options=True)
    opt[:, 1:].sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0
               for p in net.option_fcs.parameters())
    assert net.block.conv1.weight.grad.abs().sum() > 0, \
        "the option loss must reach the shared trunk (not detached)"
    assert net.policy_fc.weight.grad is None or \
        net.policy_fc.weight.grad.abs().sum() == 0


def test_bad_option_length_raises():
    with pytest.raises(ValueError):
        PredictionNetwork(**PRED_KW[0], option_length=0)
    with pytest.raises(ValueError):
        DynamicsNetwork(latent_channels=C, action_space=A, option_length=0)


# --------------------------------------------------------------------------
# L > 1: the jump dynamics
# --------------------------------------------------------------------------
def dyn(L=3, seed=9, encoding="embed"):
    torch.manual_seed(seed)
    return DynamicsNetwork(latent_channels=C, action_space=A, support_size=10,
                           num_bins=21, reward_channels=8,
                           action_encoding=encoding, option_length=L)


def test_l3_dynamics_adds_zero_rows_without_rng_and_keeps_position_1():
    torch.manual_seed(9)
    one = dyn(L=1)
    rng1 = torch.get_rng_state()
    three = dyn(L=3)
    rng3 = torch.get_rng_state()
    assert torch.equal(rng1, rng3), "the zero rows must draw no RNG"
    s1, s3 = one.state_dict(), three.state_dict()
    assert list(s3)[:len(s1)] == list(s1)
    assert all(torch.equal(s1[k], s3[k]) for k in s1)
    assert sorted(k for k in s3 if k not in s1) == [
        "option_embeddings.0.weight", "option_embeddings.1.weight"]
    assert three.action_embedding is not None      # position 1, today's name
    for emb in three.option_embeddings:
        assert emb.weight.shape == (A, C)
        assert emb.weight.abs().sum() == 0
        assert emb.weight.requires_grad


def test_a_1d_action_is_todays_path_and_a_padded_move_matches_it():
    one, three = dyn(L=1), dyn(L=3)
    with torch.no_grad():
        for net in (one, three):
            net.reward_fc.weight.normal_(0, 0.5,
                                         generator=torch.Generator().manual_seed(2))
            for emb in getattr(net, "option_embeddings", []):
                emb.weight.normal_(0, 1.0,
                                   generator=torch.Generator().manual_seed(3))
            net.eval()
        x = latents(4, 8)
        act = torch.tensor([3, 0, 4, 1])
        a, b = one(x, act), three(x, act)
        assert all(torch.equal(u, v) for u, v in zip(a, b))
        padded = torch.tensor([[3, -1, -1], [0, -1, -1], [4, -1, -1],
                               [1, -1, -1]])
        c = three(x, padded)
        assert all(torch.equal(u, v) for u, v in zip(a, c))
        # a width-1 jump is the same move too
        d = three(x, padded[:, :1])
        assert all(torch.equal(u, v) for u, v in zip(a, d))


def test_a_jump_encodes_the_sum_of_its_positions_rows():
    net = dyn(L=3)
    g = torch.Generator().manual_seed(4)
    with torch.no_grad():
        net.action_embedding.weight.normal_(0, 1, generator=g)
        for emb in net.option_embeddings:
            emb.weight.normal_(0, 1, generator=g)
    jump = torch.tensor([[1, 2, 3], [4, 4, -1], [0, -1, -1], [2, 0, 1]])
    with torch.no_grad():
        enc = net._encode_action(jump, 4, 6, 6, CPU)
    E0 = net.action_embedding.weight
    E1, E2 = (e.weight for e in net.option_embeddings)
    want = torch.stack([E0[1] + E1[2] + E2[3], E0[4] + E1[4], E0[0],
                        E0[2] + E1[0] + E2[1]])
    assert enc.shape == (4, C, 6, 6)
    assert torch.allclose(enc[:, :, 0, 0], want, atol=1e-6)
    assert torch.equal(enc[:, :, 0, 0], enc[:, :, 5, 3])
    # -1 adds nothing, whichever row index 0 holds
    assert torch.equal(enc[2, :, 0, 0], E0[0])


def test_at_init_a_jump_is_its_first_move():
    """Rows 2..L start at zero: before training a jump is its first move
    (OF's jump rows start from today's dynamics)."""
    net = dyn(L=3).eval()
    x = latents(2, 3)
    with torch.no_grad():
        a = net(x, torch.tensor([2, 1]))
        b = net(x, torch.tensor([[2, 4, 4], [1, 0, -1]]))
    assert all(torch.equal(u, v) for u, v in zip(a, b))


def test_a_jump_trains_only_its_own_rows():
    net = dyn(L=3)
    x = latents(3, 5)
    nxt, _, _ = net(x, torch.tensor([[1, 2, -1], [3, 2, -1], [0, -1, -1]]))
    nxt.sum().backward()
    g1, g2 = (e.weight.grad for e in net.option_embeddings)
    assert g1[2].abs().sum() > 0
    assert torch.count_nonzero(g1.abs().sum(1)).item() == 1   # only move 2 at position 2
    assert g2 is None or g2.abs().sum() == 0                   # position 3 unused
    rows0 = net.action_embedding.weight.grad.abs().sum(1)
    assert set(torch.nonzero(rows0).flatten().tolist()) == {0, 1, 3}


def test_jump_errors_are_loud():
    with pytest.raises(ValueError, match="embed"):
        dyn(L=3, encoding="plane")
    with pytest.raises(ValueError, match="embed"):
        dyn(L=2, encoding="onehot")
    net = dyn(L=2)
    with pytest.raises(ValueError, match="option_length 2"):
        net(latents(1, 0), torch.tensor([[1, 2, 3]]))
    with pytest.raises(ValueError, match="option_length 1"):
        dyn(L=1)(latents(1, 0), torch.tensor([[1, 2]]))


# --------------------------------------------------------------------------
# wiring: factories and the lagged target model
# --------------------------------------------------------------------------
def test_the_factories_follow_the_knob(monkeypatch):
    monkeypatch.setattr(muzero, "OPTION_LENGTH", 3)
    pred = muzero.build_prediction(A)
    dyn3 = muzero.build_dynamics(A)
    assert pred.option_length == 3 and len(pred.option_fcs) == 2
    assert dyn3.option_length == 3 and len(dyn3.option_embeddings) == 2
    monkeypatch.setattr(muzero, "ACTION_ENCODING", "onehot")
    with pytest.raises(ValueError, match="embed"):
        muzero.build_dynamics(A)


@pytest.mark.parametrize("L", [1, 3])
def test_the_target_model_mirrors_the_live_option_length(monkeypatch, L):
    monkeypatch.setattr(muzero, "OPTION_LENGTH", L)
    t = object.__new__(muzero.TrainingHelper)
    t.device = CPU
    t.action_space = A
    t.training_steps = 0
    t.representation = muzero.build_representation()
    t.prediction = muzero.build_prediction(A)
    t.dynamics = muzero.build_dynamics(A)
    t.value_prefix = muzero.build_value_prefix()
    # the knob changed after construction: the target must follow the nets
    monkeypatch.setattr(muzero, "OPTION_LENGTH", 4 - L)
    muzero.TrainingHelper._ensure_target_model(t)    # strict load inside
    assert t.target_prediction.option_length == L
    assert t.target_dynamics.option_length == L
    assert same_state(t.prediction, t.target_prediction)
    assert same_state(t.dynamics, t.target_dynamics)
