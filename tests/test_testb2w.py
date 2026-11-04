"""Test B2W (Kai, 2026-11-04; CURRICULUM.md "Next experiment (pre-registered)
-- Test B2W: Test B2's lineage with BBF's AdamW in place of EfficientZero's
SGD"): W0-W2 are Test B2's T0-T2 with one change, the optimizer -- BBF's
AdamW (bbf/configs/BBF.gin, bbf/agents/spr_agent.py create_scaling_optimizer:
lr 0.0001, eps 0.00015, decoupled weight decay 0.1, rank-1 parameters
undecayed, no warmup). Pins muzero.make_optimizer's two new knobs (off: every
other arm bit for bit; refused under SGD), BBF's mask, eps and decoupled
decay, the W arms against the T arms, the five optimizers a W run builds,
the warmup's absence and a checkpoint's round trip of the two param groups.
CPU only."""
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] /
                       "experiments"))
import ablation  # noqa: E402
import muzero  # noqa: E402
from tests.test_prioract2 import restore_rngs  # noqa: E402,F401
from tests.test_testc import make_run  # noqa: E402

PREFIX = ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-"
          "tokens-")
T = [f"{PREFIX}two-t{k}" for k in range(3)]
W = [f"{PREFIX}adam-w{k}" for k in range(3)]
# the pre-registration's two optimizers, written out here
BBF = {"OPTIMIZER": "adamw", "LEARNING_RATE": 1e-4, "WEIGHT_DECAY": 0.1,
       "ADAM_EPS": 0.00015, "DECAY_RANK1": False, "LR_WARMUP_STEPS": 0}
SGD = {"OPTIMIZER": "sgd", "LEARNING_RATE": 0.2, "WEIGHT_DECAY": 1e-4,
       "SGD_MOMENTUM": 0.9, "ADAM_EPS": None, "DECAY_RANK1": True,
       "LR_WARMUP_STEPS": 1000}
KNOBS = ("OPTIMIZER", "LEARNING_RATE", "WEIGHT_DECAY", "SGD_MOMENTUM",
         "ADAM_EPS", "DECAY_RANK1", "LR_WARMUP_STEPS")


@pytest.fixture(autouse=True)
def no_leak():
    """Torn down last: every test leaves the optimizer knobs as it found
    them."""
    before = {k: getattr(muzero, k) for k in KNOBS}
    yield
    assert {k: getattr(muzero, k) for k in KNOBS} == before


@pytest.fixture
def bbf(monkeypatch):
    """BBF's AdamW on the module, outside any arm."""
    for k, v in BBF.items():
        monkeypatch.setattr(muzero, k, v)


def _net(seed=0):
    """Every parameter rank there is: a conv's 4-D weight, a Linear's 2-D
    one, biases and BN affines (rank 1)."""
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Conv2d(2, 3, 3),
                               torch.nn.BatchNorm2d(3), torch.nn.Flatten(),
                               torch.nn.Linear(12, 4))


def _steps(net, opt, n=3, seed=1):
    torch.manual_seed(seed)
    x = torch.randn(5, 2, 4, 4)
    for _ in range(n):
        opt.zero_grad()
        net(x).pow(2).mean().backward()
        opt.step()


def test_the_knobs_default_off():
    assert muzero.ADAM_EPS is None and muzero.DECAY_RANK1 is True
    assert muzero.OPTIMIZER == "adamw"
    assert ablation.BBF_ADAMW == BBF


def test_the_defaults_build_the_old_adamw_bit_for_bit():
    net, ref_net = _net(), _net()
    opt = muzero.make_optimizer(net.parameters())
    assert type(opt) is torch.optim.AdamW and len(opt.param_groups) == 1
    g = opt.param_groups[0]
    assert (g["lr"], g["weight_decay"], g["eps"]) == (
        muzero.LEARNING_RATE, muzero.WEIGHT_DECAY, 1e-8)
    ref = torch.optim.AdamW(ref_net.parameters(), lr=muzero.LEARNING_RATE,
                            weight_decay=muzero.WEIGHT_DECAY)
    _steps(net, opt)
    _steps(ref_net, ref)
    for p, q in zip(net.parameters(), ref_net.parameters()):
        assert torch.equal(p, q)


def test_sgd_is_unchanged_and_refuses_the_adamw_knobs(monkeypatch):
    for k, v in SGD.items():
        monkeypatch.setattr(muzero, k, v)
    net, ref_net = _net(), _net()
    opt = muzero.make_optimizer(net.parameters())
    assert type(opt) is torch.optim.SGD and len(opt.param_groups) == 1
    ref = torch.optim.SGD(ref_net.parameters(), lr=0.2, momentum=0.9,
                          weight_decay=1e-4)
    _steps(net, opt)
    _steps(ref_net, ref)
    for p, q in zip(net.parameters(), ref_net.parameters()):
        assert torch.equal(p, q)
    for knob, val in (("ADAM_EPS", 0.00015), ("DECAY_RANK1", False)):
        with monkeypatch.context() as m:
            m.setattr(muzero, knob, val)
            with pytest.raises(ValueError, match="AdamW knobs"):
                muzero.make_optimizer(_net().parameters())


def test_bbfs_mask_eps_and_settings(bbf):
    net = _net()
    params = list(net.parameters())
    opt = muzero.make_optimizer(net.parameters())
    assert type(opt) is torch.optim.AdamW and len(opt.param_groups) == 2
    decay, keep = opt.param_groups
    assert [id(p) for p in decay["params"]] == [id(p) for p in params
                                                if p.ndim != 1]
    assert [id(p) for p in keep["params"]] == [id(p) for p in params
                                               if p.ndim == 1]
    assert len(keep["params"]) == 4             # conv bias, BN x2, fc bias
    assert (decay["weight_decay"], keep["weight_decay"]) == (0.1, 0.0)
    for g in (decay, keep):
        assert (g["lr"], g["eps"], g["betas"]) == (1e-4, 0.00015,
                                                   (0.9, 0.999))


def test_a_zero_gradient_step_is_the_decoupled_decay_alone(bbf):
    """With zero gradients Adam's update is 0 (0 / eps), so one step is
    BBF's decay alone: p <- p x (1 - lr x wd) on the rank > 1 parameters,
    nothing on the rank-1 ones."""
    net = _net()
    opt = muzero.make_optimizer(net.parameters())
    before = [p.detach().clone() for p in net.parameters()]
    for p in net.parameters():
        p.grad = torch.zeros_like(p)
    opt.step()
    for p, b in zip(net.parameters(), before):
        want = b * (1 - 1e-4 * 0.1) if p.ndim != 1 else b
        assert torch.allclose(p, want, rtol=1e-7, atol=0), p.shape
        assert p.ndim == 1 or not torch.equal(p, b)


def test_bbfs_steps_are_the_two_group_adamw(bbf):
    net, ref_net = _net(), _net()
    opt = muzero.make_optimizer(net.parameters())
    ps = list(ref_net.parameters())
    ref = torch.optim.AdamW(
        [{"params": [p for p in ps if p.ndim != 1]},
         {"params": [p for p in ps if p.ndim == 1], "weight_decay": 0.0}],
        lr=1e-4, weight_decay=0.1, eps=0.00015)
    _steps(net, opt, n=5)
    _steps(ref_net, ref, n=5)
    for p, q in zip(net.parameters(), ref_net.parameters()):
        assert torch.equal(p, q)


def test_an_empty_group_is_left_out(bbf):
    bn = torch.nn.BatchNorm1d(3)                # rank-1 parameters only
    opt = muzero.make_optimizer(bn.parameters())
    assert [g["weight_decay"] for g in opt.param_groups] == [0.0]
    lin = torch.nn.Linear(3, 2, bias=False)     # no rank-1 parameter
    opt = muzero.make_optimizer(lin.parameters())
    assert [g["weight_decay"] for g in opt.param_groups] == [0.1]


def test_the_w_arms_are_t_with_bbfs_adamw():
    for k in range(3):
        w, t = ablation.ARMS[W[k]], ablation.ARMS[T[k]]
        assert {o: w[o] for o in BBF} == BBF
        assert {o: t.get(o, getattr(muzero, o)) for o in SGD} == SGD
        skip = set(BBF) | {"_master_arm", "_doc"}
        assert ({a: b for a, b in w.items() if a not in skip}
                == {a: b for a, b in t.items() if a not in skip})
        assert w.get("_master_arm") == (W[k - 1] if k else None)
        assert t.get("_master_arm") == (T[k - 1] if k else None)
        assert "SGD_MOMENTUM" not in w


def test_a_w_runs_five_optimizers_take_bbfs_adamw_and_no_warmup(
        restore_rngs):  # noqa: F811
    """Inside W0's arm config (its 36 tokens), a run's five optimizers --
    built as Run.__init__ builds them -- are BBF's, the BN affines and
    biases undecayed; the warmup leaves the lr at 1e-4 from step 0 (T0's
    sets it to 0 there)."""
    with ablation.arm_config(W[0]) as spec, ablation.token_actions():
        run = make_run(spec, arm=W[0], steps=0)
        opts = [getattr(run.trainer, o) for o in ablation.OPT_NAMES]
        for n, opt in zip(ablation.NET_NAMES, opts):
            assert type(opt) is torch.optim.AdamW, n
            net = getattr(run.trainer, n)
            undecayed = {id(p) for g in opt.param_groups
                         if g["weight_decay"] == 0.0 for p in g["params"]}
            assert undecayed == {id(p) for p in net.parameters()
                                 if p.ndim == 1}, n
            for g in opt.param_groups:
                assert (g["lr"], g["eps"]) == (1e-4, 0.00015), n
                assert g["weight_decay"] in (0.0, 0.1), n
            assert (sum(len(g["params"]) for g in opt.param_groups)
                    == len(list(net.parameters()))), n
        muzero.apply_lr_warmup(opts, 0)
        assert all(g["lr"] == 1e-4 for o in opts for g in o.param_groups)
    with ablation.arm_config(T[0]) as spec, ablation.token_actions():
        run = make_run(spec, arm=T[0], steps=0)
        opts = [getattr(run.trainer, o) for o in ablation.OPT_NAMES]
        assert all(type(o) is torch.optim.SGD for o in opts)
        muzero.apply_lr_warmup(opts, 0)
        assert all(g["lr"] == 0.0 for o in opts for g in o.param_groups)


def test_a_checkpoints_optimizer_state_round_trips(restore_rngs,  # noqa: F811
                                                    tmp_path):
    """The two groups' settings and Adam's moments survive the checkpoint's
    state_dict / load_state_dict, as a resume reloads them."""
    with ablation.arm_config(W[1]) as spec, ablation.token_actions():
        run = make_run(spec, arm=W[1], steps=0)
        tr = run.trainer
        for n, o in zip(ablation.NET_NAMES, ablation.OPT_NAMES):
            net, opt = getattr(tr, n), getattr(tr, o)
            opt.zero_grad()
            sum((p ** 2).sum() for p in net.parameters()).backward()
            opt.step()
        path = tmp_path / "optims.pt"
        torch.save({o: getattr(tr, o).state_dict()
                    for o in ablation.OPT_NAMES}, path)
        fresh = make_run(spec, arm=W[1], steps=0)
        saved = torch.load(path)
        for o in ablation.OPT_NAMES:
            opt = getattr(fresh.trainer, o)
            opt.load_state_dict(saved[o])
            a, b = getattr(tr, o).state_dict(), opt.state_dict()
            assert a["param_groups"] == b["param_groups"], o
            assert a["state"].keys() == b["state"].keys(), o
            for i in a["state"]:
                for key in ("exp_avg", "exp_avg_sq"):
                    assert torch.equal(a["state"][i][key],
                                       b["state"][i][key]), (o, i, key)
