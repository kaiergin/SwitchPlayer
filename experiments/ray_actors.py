"""Self-play actor for the concurrent (Ray) harness mode.

The actor owns its OWN network copies and environment, applies the arm's
overrides inside its own process (a Ray worker is a fresh interpreter --
the driver's arm_config setattrs never reach it), and plays episodes while
the driver trains. Weights arrive by explicit set_weights() sync at
iteration boundaries, so the acting policy's staleness is identical to the
sequential harness (episode k plays with end-of-iteration-(k-1) weights);
what changes is data incorporation, which lags one iteration (episode k
first trains in iteration k+1).

The class is deliberately UNDECORATED: the driver wraps it with
ray.remote(num_gpus=...) at creation, and tests run it in ray local mode.
"""
import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import torch


class SelfPlayActor:
    def __init__(self, overrides, action_space, seed, schedule,
                 env_name, frameskip, episode_cap, server=None):
        import muzero as mz
        self.mz = mz
        # The arm's world, rebuilt locally: constants first (network
        # factories and the search read them at call time), then the
        # run-length-keyed schedule VALUES the driver computed -- shipped,
        # never re-derived, so the 0.3 horizon-decay formula lives in one
        # place (_run_length_globals).
        for key, value in overrides.items():
            if hasattr(mz, key):
                setattr(mz, key, value)
        for key, value in schedule.items():
            setattr(mz, key, value)
        np.random.seed(seed * 7919 + 17)
        torch.manual_seed(seed * 7919 + 17)
        # Pool workers are CPU actors by CONSTRUCTION, not just by Ray's
        # CUDA masking (3b audit F5): the searcher's device places node
        # latents and search tensors, and a worker must never park them
        # on the GPU it is supposed to have no claim on.
        self.device = (torch.device("cpu") if server is not None else
                       torch.device(
                           "cuda" if torch.cuda.is_available() else "cpu"))
        self.action_space = action_space
        self.episode_cap = episode_cap

        self.searcher = mz.MCTSHelper(action_space=action_space,
                                      device=self.device, network_ref=None,
                                      storage_ref=None, enable_logging=False)
        if server is not None:
            # Phase 3b env/tree worker: no local nets -- every forward
            # goes to the shared InferenceServer, which batches across
            # workers. The locally-built nets are discarded with the
            # manager they hang off; the search runs on CPU tensors.
            import ray
            self.searcher.network_manager = RemoteNetworkManager(
                server, ray, action_space)
        else:
            nm = self.searcher.network_manager
            for net in (nm.representation, nm.prediction, nm.dynamics,
                        nm.value_prefix):
                net.eval()

        # ONE env implementation and ONE episode implementation, both
        # ablation's own (build_env and Run.play_episode run UNMODIFIED
        # here) -- no drifting copies. play_episode's dependency surface is
        # pinned by tests/test_ray_harness.py: the self.* attributes (env,
        # searcher, env_steps) AND the ablation module globals it reads
        # (ACTIONS, EPISODE_CAP), which a fresh worker interpreter would
        # otherwise leave at their MsPacman defaults -- the first test run
        # crashed exactly there, 5-action choice against a 6-action policy.
        import ablation
        ablation.ACTIONS = action_space
        ablation.EPISODE_CAP = episode_cap
        ablation.ENV_NAME = env_name
        ablation.FRAMESKIP = frameskip
        self._ablation = ablation
        self.env = ablation.build_env(env_name, frameskip)
        self.shell = object.__new__(ablation.Run)
        self.shell.env = self.env
        self.shell.searcher = self.searcher
        self.shell.env_steps = 0

    def set_weights(self, state_dicts, training_steps):
        nm = self.searcher.network_manager
        if isinstance(nm, RemoteNetworkManager):
            # Weights live on the InferenceServer (the driver syncs it
            # once for all workers); only the temperature mirror is ours.
            nm.training_steps = training_steps
            return training_steps
        for name, state in state_dicts.items():
            net = getattr(nm, name)
            net.load_state_dict(state)
            net.eval()
        # The staged acting temperature keys on training progress.
        nm.training_steps = training_steps
        return training_steps

    def play(self, episode_seed):
        """One self-play episode; returns (trajectory, terminated, steps)."""
        self.env.reset(seed=int(episode_seed))
        before = self.shell.env_steps
        traj, _, _, terminated = self._ablation.Run.play_episode(
            self.shell, greedy=False, cap=self.episode_cap)
        return traj, terminated, self.shell.env_steps - before

    def eval_episodes(self, seeds):
        """Greedy evaluation episodes; returns their returns. Env steps from
        eval are protocol-exempt (the driver never counts them) and the
        shell's counter is scratch here."""
        returns = []
        for s in seeds:
            self.env.reset(seed=int(s))
            returns.append(self._ablation.Run.play_episode(
                self.shell, greedy=True, cap=self.episode_cap)[1])
        return returns

    def get_constant(self, name):
        """Test probe: what this PROCESS's muzero constant resolves to --
        counterfactual-proof that overrides crossed the process boundary."""
        return getattr(self.mz, name)


class InferenceServer:
    """Phase 3b: ONE GPU-resident copy of the four nets serving every
    env/tree worker, batching requests ACROSS workers into single large
    forwards -- the acting-side twin of communal refresh's cross-root
    batching. Each worker runs strictly sequential per-tree search, so a
    worker has at most one outstanding request; the server's natural
    batch is "whatever is queued", converging on the worker count. A
    small fill window (max_wait_ms) trades one wire-latency quantum for
    batch width; requests never wait longer than that.

    An ASYNC actor: the driver must create it with ray.remote(...) on
    this class and Ray will run its coroutine methods on one event loop,
    interleaving arrivals -- that interleaving is what lets _flush see
    multiple workers' requests at once. Tests drive the class directly
    on a local asyncio loop.

    Determinism scope (3b audit F3): batched forwards are bitwise-equal
    to solo forwards at every width >= 2, but width-1-vs-N differs at
    ~1e-6 (oneDNN/cuDNN kernel selection, not cross-sample mixing) -- so
    a search's exact float path depends on which workers shared a flush
    window. Semantically sound; pool runs are not seed-reproducible.

    pool_size: when set, a flush fires the moment that many requests are
    queued across kinds (each worker has at most one outstanding, so a
    full house means nobody else is coming) -- the fill window then only
    pays when some workers are env-stepping. On Windows the window
    itself would otherwise cost ~15ms real time at 2ms nominal (timer
    quantization); __init__ requests 1ms resolution to fix that.
    """

    def __init__(self, overrides, action_space, schedule,
                 max_wait_ms=2.0, max_batch=64, device=None, pool_size=0):
        import muzero as mz
        self.mz = mz
        for key, value in overrides.items():
            if hasattr(mz, key):
                setattr(mz, key, value)
        for key, value in schedule.items():
            setattr(mz, key, value)
        torch.manual_seed(1)
        # device override is for tests (CPU parity against local search);
        # production leaves it None and takes the GPU.
        self.device = (torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"))
        # Two weight sets (3b.5): "live" serves acting/eval, "target"
        # serves refresh workers -- refresh must search the LAGGED target
        # model, and hosting both here is what lets every GPU consumer
        # route through one server (no private net copies, no fractional
        # GPU shares, one weight ship per set per boundary).
        def build():
            nets = {
                "representation": mz.build_representation().to(self.device),
                "dynamics": mz.build_dynamics(action_space).to(self.device),
                "prediction": mz.build_prediction(action_space).to(self.device),
                "value_prefix": mz.build_value_prefix().to(self.device),
            }
            for net in nets.values():
                net.eval()
            return nets

        self.nets = {"live": build(), "target": build()}
        self.max_wait = max_wait_ms / 1000.0
        self.max_batch = max_batch
        self.pool_size = pool_size
        if sys.platform == "win32":
            # asyncio.sleep(0.002) really sleeps ~15ms under the default
            # Windows timer (3b audit F2, measured); 1ms resolution
            # restores the design constant (re-measured 2.0-3.0ms).
            import ctypes
            ctypes.windll.winmm.timeBeginPeriod(1)
        # (kind, which) -> list of (payload, asyncio.Future); keys created
        # lazily. Batches never mix weight sets.
        self._queues = {}
        self._flusher = None

    def set_weights(self, state_dicts, training_steps, which="live"):
        for name, state in state_dicts.items():
            self.nets[which][name].load_state_dict(state)
            self.nets[which][name].eval()
        self.training_steps = training_steps
        return training_steps

    def prefix_hidden_size(self):
        """Workers build LSTM zero-states locally; they need (L, H) --
        hardcoding L=1 would mis-shape the state if the LSTM ever gains
        depth (3b audit F4). Geometry is set-independent."""
        h = self.nets["live"]["value_prefix"].initial_state(1, "cpu")[0]
        return (h.shape[0], h.shape[-1])

    async def _enqueue(self, kind, which, payload):
        fut = asyncio.get_event_loop().create_future()
        self._queues.setdefault((kind, which), []).append((payload, fut))
        if (self.pool_size
                and sum(len(q) for q in self._queues.values())
                >= self.pool_size):
            # Full house: every worker's request is here, nobody else is
            # coming -- serve now, skip the fill window entirely. The
            # sleeping flusher wakes later onto empty queues (harmless).
            self._flush()
        elif self._flusher is None or self._flusher.done():
            self._flusher = asyncio.ensure_future(self._flush_soon())
        return await fut

    async def _flush_soon(self):
        # The fill window: yield so concurrently-arriving requests (other
        # workers' calls interleaved on this actor's loop) can queue up,
        # then serve everything present.
        await asyncio.sleep(self.max_wait)
        self._flush()

    def _flush(self):
        for key in list(self._queues):
            kind, which = key
            pending, self._queues[key] = self._queues[key], []
            for i in range(0, len(pending), self.max_batch):
                chunk = pending[i:i + self.max_batch]
                try:
                    outs = self._run(kind, [p for p, _ in chunk], which)
                    for (_, fut), out in zip(chunk, outs):
                        if not fut.done():
                            fut.set_result(out)
                except Exception as e:  # surface, never hang a worker
                    for _, fut in chunk:
                        if not fut.done():
                            fut.set_exception(
                                RuntimeError(f"{kind} batch failed: {e}"))

    def _run(self, kind, payloads, which="live"):
        """One batched forward over concatenated payloads against ONE
        weight set; returns one result per payload, split by each
        payload's own batch size."""
        nets = self.nets[which]
        with torch.no_grad():
            if kind == "initial":
                sizes = [np.asarray(p).shape[0] for p in payloads]
                obs = torch.tensor(np.concatenate(payloads, axis=0),
                                   dtype=torch.float32,
                                   device=self.device) / 255.0
                latents = nets["representation"](obs)
                logits, values, _ = nets["prediction"](latents)
                policy = torch.softmax(logits, dim=1).cpu().numpy()
                lat = latents.cpu().numpy()
                val = values.squeeze(-1).cpu().numpy()
                out, at = [], 0
                for s in sizes:
                    out.append((lat[at:at + s], policy[at:at + s],
                                val[at:at + s]))
                    at += s
                return out
            if kind == "dynamics":
                sizes = [p[0].shape[0] for p in payloads]
                lat = torch.tensor(
                    np.concatenate([p[0] for p in payloads], axis=0),
                    dtype=torch.float32, device=self.device)
                act = torch.tensor(
                    np.concatenate([p[1] for p in payloads], axis=0),
                    dtype=torch.long, device=self.device)
                nxt, rew, _ = nets["dynamics"](lat, act)
                nxt = nxt.cpu().numpy()
                rew = rew.cpu().numpy()
                out, at = [], 0
                for s in sizes:
                    out.append((nxt[at:at + s], rew[at:at + s]))
                    at += s
                return out
            if kind == "prediction":
                sizes = [p.shape[0] for p in payloads]
                lat = torch.tensor(np.concatenate(payloads, axis=0),
                                   dtype=torch.float32, device=self.device)
                logits, values, _ = nets["prediction"](lat)
                logits = logits.cpu().numpy()
                values = values.cpu().numpy()
                out, at = [], 0
                for s in sizes:
                    out.append((logits[at:at + s], values[at:at + s]))
                    at += s
                return out
            # value_prefix: payload = (latents, h, c); batch along the
            # LSTM's batch dim (dim 1 for states, dim 0 for latents).
            sizes = [p[0].shape[0] for p in payloads]
            lat = torch.tensor(np.concatenate([p[0] for p in payloads], 0),
                               dtype=torch.float32, device=self.device)
            h = torch.tensor(np.concatenate([p[1] for p in payloads], 1),
                             dtype=torch.float32, device=self.device)
            c = torch.tensor(np.concatenate([p[2] for p in payloads], 1),
                             dtype=torch.float32, device=self.device)
            prefixes, _, (nh, nc) = nets["value_prefix"](
                lat, (h.contiguous(), c.contiguous()))
            prefixes = prefixes.cpu().numpy()
            nh, nc = nh.cpu().numpy(), nc.cpu().numpy()
            out, at = [], 0
            for s in sizes:
                out.append((prefixes[at:at + s], nh[:, at:at + s],
                            nc[:, at:at + s]))
                at += s
            return out

    # One coroutine per request kind; Ray serves them concurrently on the
    # actor's event loop, which is what fills the batch window.
    async def initial(self, obs, which="live"):
        return await self._enqueue("initial", which, obs)

    async def dynamics_step(self, latents, actions, which="live"):
        return await self._enqueue("dynamics", which, (latents, actions))

    async def predict(self, latents, which="live"):
        return await self._enqueue("prediction", which, latents)

    async def prefix_step(self, latents, h, c, which="live"):
        return await self._enqueue("value_prefix", which, (latents, h, c))


class _RemoteModule:
    """Callable stand-in for a torch module living on the InferenceServer.
    Converts torch<->numpy at the wire, honours the .training/.eval()
    protocol expand_prefixes uses, and blocks on the server's reply --
    the worker's search is strictly sequential, so blocking is correct
    and the server's cross-worker batching supplies the parallelism."""

    def __init__(self, owner, kind):
        self._owner = owner
        self._kind = kind
        self.training = False

    def eval(self):
        return self

    def train(self, mode=True):
        self.training = False  # workers never train
        return self

    def initial_state(self, batch, device):
        layers, hidden = self._owner.prefix_hidden
        z = torch.zeros(layers, batch, hidden)
        return (z, z.clone())

    def __call__(self, *args):
        owner = self._owner
        if self._kind == "dynamics":
            lat, act = args
            nxt, rew = owner._call("dynamics_step",
                                   lat.detach().cpu().numpy(),
                                   act.detach().cpu().numpy())
            return (torch.from_numpy(np.array(nxt)),
                    torch.from_numpy(np.array(rew)), None)
        if self._kind == "prediction":
            logits, values = owner._call(
                "predict", args[0].detach().cpu().numpy())
            return (torch.from_numpy(np.array(logits)),
                    torch.from_numpy(np.array(values)), None)
        # value_prefix
        lat, (h, c) = args
        prefixes, nh, nc = owner._call(
            "prefix_step", lat.detach().cpu().numpy(),
            h.detach().cpu().numpy(), c.detach().cpu().numpy())
        return (torch.from_numpy(np.array(prefixes)), None,
                (torch.from_numpy(np.array(nh)),
                 torch.from_numpy(np.array(nc))))


class RemoteNetworkManager:
    """NetworkManager facade whose forwards run on the InferenceServer.
    Exposes exactly the surface the search uses: batch_inference,
    initial_inference, callable dynamics/prediction/value_prefix, and
    training_steps for the temperature schedule. Latents live on the
    worker as CPU torch tensors."""

    def __init__(self, server, ray_module, action_space, which="live"):
        self.server = server
        self.ray = ray_module
        self.action_space = action_space
        self.which = which  # weight set every call runs against (3b.5)
        self.device = torch.device("cpu")
        self.training_steps = 0
        self.dynamics = _RemoteModule(self, "dynamics")
        self.prediction = _RemoteModule(self, "prediction")
        self.value_prefix = _RemoteModule(self, "value_prefix")
        self.prefix_hidden = tuple(self.ray.get(
            server.prefix_hidden_size.remote()))

    def _call(self, method, *args):
        return self.ray.get(
            getattr(self.server, method).remote(*args, which=self.which))

    def batch_inference(self, batch_obs):
        lat, policy, value = self._call("initial", np.asarray(batch_obs))
        return torch.from_numpy(np.array(lat)), policy, value

    def initial_inference(self, obs):
        lat, policy, value = self._call("initial", np.asarray(obs)[None])
        return policy[0], float(value[0])


class RefreshActor:
    """Phase 3a: re-searches stored trajectories so raised refresh
    coverage runs concurrently with the driver's training.

    Env-less -- refresh touches only stored observations. Two modes:
    OWN-NETS (server=None, the 3a original): its nets play the driver's
    lagged target model, shipped by set_weights each stamp change.
    SERVER mode (3b.5): a CPU refresh WORKER whose every forward routes
    to the shared InferenceServer's "target" weight set -- no private
    nets, no private GPU slice, and refresh parallelizes across
    trajectories by running several such workers. Either way the search
    itself is Run._search_traj run UNMODIFIED through a shell -- no
    drifting copy of the communal topology -- and results return as
    {index: (policy, root_value)} for the DRIVER's _apply_refresh.

    Undecorated like SelfPlayActor: the driver wraps with
    ray.remote(...); tests drive it directly.
    """

    def __init__(self, overrides, action_space, seed, schedule, server=None):
        import muzero as mz
        self.mz = mz
        for key, value in overrides.items():
            if hasattr(mz, key):
                setattr(mz, key, value)
        for key, value in schedule.items():
            setattr(mz, key, value)
        np.random.seed(seed * 7919 + 23)
        torch.manual_seed(seed * 7919 + 23)
        # CPU by construction in server mode, same reasoning as the pool
        # workers (3b audit F5).
        self.device = (torch.device("cpu") if server is not None else
                       torch.device(
                           "cuda" if torch.cuda.is_available() else "cpu"))
        self.searcher = mz.MCTSHelper(action_space=action_space,
                                      device=self.device, network_ref=None,
                                      storage_ref=None, enable_logging=False)
        if server is not None:
            import ray
            self.searcher.network_manager = RemoteNetworkManager(
                server, ray, action_space, which="target")
        else:
            nm = self.searcher.network_manager
            for net in (nm.representation, nm.prediction, nm.dynamics,
                        nm.value_prefix):
                net.eval()
        import ablation
        ablation.ACTIONS = action_space
        self._ablation = ablation
        self.shell = object.__new__(ablation.Run)
        self.shell.searcher = self.searcher

    def set_weights(self, state_dicts, training_steps):
        """Load the driver's LAGGED TARGET nets; these are the networks
        every refresh search runs with (EffZero's reanalyze searches the
        target model, never the live weights). In server mode the target
        set lives on the InferenceServer (the driver ships it there once
        per stamp change); only the counter mirror is ours."""
        nm = self.searcher.network_manager
        if isinstance(nm, RemoteNetworkManager):
            nm.training_steps = training_steps
            return training_steps
        for name, state in state_dicts.items():
            net = getattr(nm, name)
            net.load_state_dict(state)
            net.eval()
        nm.training_steps = training_steps
        return training_steps

    def refresh(self, records, indices):
        """Search one trajectory's eligible indices; returns
        {index: (policy, root_value)}. Records carry obs/action/reward
        plus the truncated/has_policy_target flags -- everything
        _search_traj reads, nothing it doesn't."""
        return self._ablation.Run._search_traj(self.shell, records,
                                               list(indices))
