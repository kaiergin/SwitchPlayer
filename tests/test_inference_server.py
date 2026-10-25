"""Phase 3b: the shared GPU inference server and the remote network
manager the env/tree workers search through.

The load-bearing claims: (1) concurrent requests are served from ONE
batched forward per kind (the cross-worker batching that pays for the
architecture), (2) a search running entirely through the remote path
produces exactly what the same nets produce locally (per-tree semantics
untouched), (3) the module-protocol surface expand_prefixes and
batch_expand_nodes rely on is honoured by the proxies.

The server is undecorated, so these tests drive its coroutines on a
local asyncio loop; the ray transport itself is covered by the real
tiny-cluster tests in test_ray_harness.py's pattern.
"""
import asyncio
import sys

import numpy as np
import pytest
import torch

import muzero

sys.path.insert(0, str(__import__("pathlib").Path(
    __file__).resolve().parent.parent / "experiments"))
import ray_actors

CPU = torch.device("cpu")
ACTIONS = 6

OVERRIDES = {"USE_GUMBEL_SEARCH": False, "USE_SEARCH_ACTING": True,
             "MCTS_BATCHES": 1, "USE_VALUE_PREFIX_IN_TREE": True}
SCHEDULE = {"TEMPERATURE_TOTAL_STEPS": 1000, "OFFPOLICY_DECAY_STEPS": None}


@pytest.fixture
def guard_constants(monkeypatch):
    """The server ctor setattrs its overrides into the shared muzero
    module; monkeypatch the same keys first so they restore."""
    for key, val in OVERRIDES.items():
        monkeypatch.setattr(muzero, key, val)
    yield


def make_server(**kw):
    # CPU-pinned: parity asserts compare against CPU-local search, and
    # the suite must not contend with any live GPU training run.
    return ray_actors.InferenceServer(OVERRIDES, ACTIONS, SCHEDULE,
                                      device="cpu", **kw)


def fake_obs(n):
    rng = np.random.default_rng(7)
    return rng.integers(0, 255, (n, 12, *muzero.IMAGE_RESHAPE),
                        dtype=np.uint8)


# --------------------------------------------------------------------------
# Batching
# --------------------------------------------------------------------------

def test_concurrent_initial_requests_share_one_forward(guard_constants,
                                                       monkeypatch):
    srv = make_server(max_wait_ms=5.0)
    calls = []
    real_run = srv._run

    def counting_run(kind, payloads, which="live"):
        calls.append((kind, len(payloads)))
        return real_run(kind, payloads, which)

    monkeypatch.setattr(srv, "_run", counting_run)

    async def go():
        return await asyncio.gather(srv.initial(fake_obs(2)),
                                    srv.initial(fake_obs(3)))

    r2, r3 = asyncio.run(go())
    assert calls == [("initial", 2)], \
        "two concurrent requests must land in ONE batched forward"
    assert r2[0].shape[0] == 2 and r3[0].shape[0] == 3, \
        "per-request splits must respect each payload's own batch size"
    # Parity with a direct forward on the server's own nets.
    with torch.no_grad():
        obs = torch.tensor(fake_obs(2), dtype=torch.float32) / 255.0
        lat = srv.nets["live"]["representation"](obs)
    assert np.allclose(r2[0], lat.numpy(), atol=1e-5)


def test_batch_failure_surfaces_not_hangs(guard_constants, monkeypatch):
    srv = make_server(max_wait_ms=1.0)
    monkeypatch.setattr(srv, "_run",
                        lambda *a: (_ for _ in ()).throw(ValueError("boom")))

    async def go():
        return await srv.initial(fake_obs(1))

    with pytest.raises(RuntimeError, match="initial batch failed"):
        asyncio.run(go())


def test_pool_size_full_house_skips_the_fill_window(guard_constants,
                                                    monkeypatch):
    """With pool_size workers all waiting, the flush must fire the
    instant the last request arrives -- a 500ms window that is never
    paid proves the count trigger works."""
    import time
    srv = make_server(max_wait_ms=500.0, pool_size=2)
    calls = []
    real_run = srv._run

    def counting_run(kind, payloads, which="live"):
        calls.append(len(payloads))
        return real_run(kind, payloads, which)

    monkeypatch.setattr(srv, "_run", counting_run)
    lat = np.zeros((1, muzero.LATENT_CHANNELS, muzero.LATENT_HW,
                    muzero.LATENT_HW), dtype=np.float32)

    async def go():
        t0 = time.perf_counter()
        await asyncio.gather(srv.predict(lat), srv.predict(lat.copy()))
        return time.perf_counter() - t0

    dt = asyncio.run(go())
    assert dt < 0.25, f"full house waited the window ({dt:.3f}s)"
    assert calls == [2]


def test_max_batch_splits_oversized_queues(guard_constants, monkeypatch):
    srv = make_server(max_wait_ms=5.0, max_batch=2)
    calls = []
    real_run = srv._run

    def counting_run(kind, payloads, which="live"):
        calls.append(len(payloads))
        return real_run(kind, payloads, which)

    monkeypatch.setattr(srv, "_run", counting_run)

    async def go():
        return await asyncio.gather(*[srv.predict(
            np.random.default_rng(i).standard_normal(
                (1, muzero.LATENT_CHANNELS, muzero.LATENT_HW,
                 muzero.LATENT_HW)).astype(np.float32))
            for i in range(5)])

    outs = asyncio.run(go())
    assert len(outs) == 5
    assert calls == [2, 2, 1]


# --------------------------------------------------------------------------
# Remote path == local path
# --------------------------------------------------------------------------

class _Box:
    def __init__(self, value):
        self.value = value


class _Method:
    def __init__(self, fn):
        self.fn = fn

    def remote(self, *args, **kwargs):
        return _Box(self.fn(*args, **kwargs))


class _SyncHandle:
    """Server handle whose .method.remote computes synchronously through
    the server's own _run (batching itself is covered above); pairs with
    _FakeRay so RemoteNetworkManager runs fully in-process. Honours the
    3b.5 `which` weight-set tag exactly as the real endpoints do."""

    def __init__(self, srv):
        self.initial = _Method(
            lambda obs, which="live": srv._run("initial", [obs], which)[0])
        self.dynamics_step = _Method(
            lambda lat, act, which="live":
            srv._run("dynamics", [(lat, act)], which)[0])
        self.predict = _Method(
            lambda lat, which="live":
            srv._run("prediction", [lat], which)[0])
        self.prefix_step = _Method(
            lambda lat, h, c, step=None, which="live":
            srv._run("value_prefix", [(lat, h, c, step)], which)[0])
        self.prefix_hidden_size = _Method(srv.prefix_hidden_size)


class _FakeRay:
    @staticmethod
    def get(box):
        return box.value


def remote_manager(srv):
    return ray_actors.RemoteNetworkManager(_SyncHandle(srv), _FakeRay,
                                           ACTIONS)


def test_remote_search_matches_local_exactly(guard_constants):
    """A full 20-sim act_search through the remote proxies must produce
    the same policy, value, and action as the same search on the same
    nets held locally -- the per-tree semantics are untouched by the
    server architecture."""
    srv = make_server()
    local = muzero.MCTSHelper(action_space=ACTIONS, device=CPU,
                              network_ref=None, storage_ref=None)
    # One weight source for both sides.
    states = {n: getattr(local.network_manager, n).state_dict()
              for n in ("representation", "dynamics", "prediction",
                        "value_prefix")}
    srv.set_weights(states, 0)

    remote = muzero.MCTSHelper(action_space=ACTIONS, device=CPU,
                               network_ref=None, storage_ref=None)
    remote.network_manager = remote_manager(srv)

    obs = fake_obs(1)[0]
    import random as pyrandom
    np.random.seed(31)
    pyrandom.seed(31)
    a_l, pi_l, v_l = asyncio.run(local.act_search(obs, num_simulations=20))
    np.random.seed(31)
    pyrandom.seed(31)
    a_r, pi_r, v_r = asyncio.run(remote.act_search(obs, num_simulations=20))
    assert np.allclose(pi_l, pi_r, atol=1e-6), (pi_l, pi_r)
    assert v_l == pytest.approx(v_r, abs=1e-5)
    assert a_l == a_r


def test_remote_module_protocol(guard_constants):
    """The proxy surface expand_prefixes/batch_expand_nodes rely on:
    callable modules returning 3-tuples, .training/.eval()/.train(),
    initial_state, and hasattr(nm, 'value_prefix')."""
    srv = make_server()
    nm = remote_manager(srv)
    assert hasattr(nm, "value_prefix")
    vp = nm.value_prefix
    assert vp.training is False
    assert vp.eval() is vp and vp.train() is vp and vp.training is False
    h, c = vp.initial_state(3, CPU)
    assert h.shape == c.shape and h.shape[1] == 3
    lat = torch.zeros(3, muzero.LATENT_CHANNELS, muzero.LATENT_HW,
                      muzero.LATENT_HW)
    prefixes, _, (nh, nc) = vp(lat, (h, c))
    assert prefixes.shape[0] == 3
    assert nh.shape == h.shape
    nxt, rew, _ = nm.dynamics(lat, torch.zeros(3, dtype=torch.long))
    assert nxt.shape == lat.shape
    assert rew.shape[0] == 3
    logits, values, _ = nm.prediction(lat)
    assert logits.shape == (3, ACTIONS)
    policy, value = nm.initial_inference(fake_obs(1)[0])
    assert policy.shape == (ACTIONS,)
    assert np.isfinite(value)


def test_weight_sets_route_independently(guard_constants):
    """3b.5: the server hosts {live, target, eval}; a request's `which` tag
    must select the set. Load DIFFERENT weights into the two sets and
    verify the same input produces the two sets' respective outputs --
    and that a target-tagged manager (the refresh workers') gets the
    target function, not the live one."""
    srv = make_server()
    donor_a = muzero.MCTSHelper(action_space=ACTIONS, device=CPU,
                                network_ref=None, storage_ref=None)
    torch.manual_seed(99)
    donor_b = muzero.MCTSHelper(action_space=ACTIONS, device=CPU,
                                network_ref=None, storage_ref=None)
    states = lambda d: {n: getattr(d.network_manager, n).state_dict()
                        for n in ("representation", "dynamics",
                                  "prediction", "value_prefix")}
    srv.set_weights(states(donor_a), 1)                  # live
    srv.set_weights(states(donor_b), 1, which="target")  # target
    obs = fake_obs(1)
    live_lat, _, live_val = srv._run("initial", [obs])[0]
    tgt_lat, _, tgt_val = srv._run("initial", [obs], "target")[0]
    assert not np.allclose(live_lat, tgt_lat),         "the two sets must be genuinely different functions here"
    # Direct forwards against each donor confirm which set answered.
    with torch.no_grad():
        t = torch.tensor(obs, dtype=torch.float32) / 255.0
        ref_live = donor_a.network_manager.representation(t).numpy()
        ref_tgt = donor_b.network_manager.representation(t).numpy()
    assert np.allclose(live_lat, ref_live, atol=1e-5)
    assert np.allclose(tgt_lat, ref_tgt, atol=1e-5)
    # A target-tagged RemoteNetworkManager routes every call to target.
    nm = ray_actors.RemoteNetworkManager(_SyncHandle(srv), _FakeRay,
                                         ACTIONS, which="target")
    lat, _, _ = nm.batch_inference(obs)
    assert np.allclose(lat.numpy(), ref_tgt, atol=1e-5)
    # The eval set is its own frozen copy: shipping donor_a there must
    # not read back the (different) live-set weights object.
    srv.set_weights(states(donor_a), 2, which="eval")
    eval_lat, _, _ = srv._run("initial", [obs], "eval")[0]
    assert np.allclose(eval_lat, ref_live, atol=1e-5)
    srv.set_weights(states(donor_b), 3)  # live overwritten...
    eval_lat2, _, _ = srv._run("initial", [obs], "eval")[0]
    assert np.allclose(eval_lat2, ref_live, atol=1e-5),         "a live-set ship must never mutate the frozen eval set"


def test_worker_set_weights_skips_when_remote(guard_constants):
    """A pool worker's set_weights must only mirror training_steps --
    shipping weights to N workers would defeat the single-server sync."""
    srv = make_server()
    shell = object.__new__(ray_actors.SelfPlayActor)
    helper = muzero.MCTSHelper(action_space=ACTIONS, device=CPU,
                               network_ref=None, storage_ref=None)
    helper.network_manager = remote_manager(srv)
    shell.searcher = helper
    assert ray_actors.SelfPlayActor.set_weights(shell, {}, 4242) == 4242
    assert helper.network_manager.training_steps == 4242
