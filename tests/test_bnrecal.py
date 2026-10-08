"""The BN recalibration stabilizer (Kai, 2026-10-08): BN_RECAL_ACTING and the
arm effzero-repro-faithful-prioract-noredo-temp05-bnrecal. Pins: the
calibration set (newest frames, deterministic, no RNG), the recalibration
(cumulative root-only statistics on COPIES; weights and the trainer
untouched), that only the actor / eval-actor ships carry it (birth search
keeps the raw states), the snapshot's acting states, the probe's
PROBE_ACTING_STATES path, the gate, and the supervisor."""
import ast
import copy
import pathlib
import random
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] /
                       "experiments"))
import ablation
import muzero
from tests.test_checkpoint_resume import make_run
from tests.test_prioract2 import restore_rngs  # noqa: F401

ARM = "effzero-repro-faithful-prioract-noredo-temp05-bnrecal"
TEMP = "effzero-repro-faithful-prioract-noredo-temp05"
ROOT = pathlib.Path(__file__).resolve().parents[1]
SUP = ROOT / "benchmarks" / "supervisors" / "supervise_bnrecal.py"
PROBE = ROOT / "benchmarks" / "probes" / "probe_exhaustive_eval.py"
BN = torch.nn.modules.batchnorm._BatchNorm


def _traj(n, base):
    """A replay trajectory of n dict records with distinct, identifiable
    observations (every pixel = (base + i) % 251)."""
    return [{"obs": np.full((12, *muzero.IMAGE_RESHAPE), (base + i) % 251,
                            dtype=np.uint8)} for i in range(n)]


def _rng_states():
    return (random.getstate(), np.random.get_state()[1].tobytes(),
            torch.get_rng_state().numpy().tobytes())


# --------------------------------------------------------------------------
# the arm, defaults, gate
# --------------------------------------------------------------------------
def test_arm_is_the_temperature_arm_plus_the_stabilizer():
    a, b = ablation.ARMS[TEMP], ablation.ARMS[ARM]
    assert {k for k in set(a) | set(b) if a.get(k) != b.get(k)} == {
        "BN_RECAL_ACTING", "_doc"}
    assert b["BN_RECAL_ACTING"] is True and b["PRIOR_ACT_TEMP_SCALE"] == 0.5
    assert b["_launch"] == {"ray": True, "cpu_actor": True}


def test_defaults_off():
    assert muzero.BN_RECAL_ACTING is False and muzero.BN_RECAL_FRAMES == 4096


def test_gate(monkeypatch):
    ablation.bn_recal_gate(False)                       # flag off: inert
    with ablation.arm_config(ARM):
        ablation.bn_recal_gate(True)
        with pytest.raises(SystemExit, match="needs --ray"):
            ablation.bn_recal_gate(False)
    monkeypatch.setattr(muzero, "BN_RECAL_ACTING", True)
    for flag in ("USE_SEARCH_ACTING", "ASYNC_LOOP"):
        monkeypatch.setattr(muzero, flag, True)
        with pytest.raises(SystemExit, match="policy-only acting"):
            ablation.bn_recal_gate(True)
        monkeypatch.setattr(muzero, flag, False)
    for bad in (255, True, 4096.0, None):
        monkeypatch.setattr(muzero, "BN_RECAL_FRAMES", bad)
        with pytest.raises(SystemExit, match="BN_RECAL_FRAMES"):
            ablation.bn_recal_gate(True)


def test_gate_is_wired_into_run_one():
    import inspect
    assert "bn_recal_gate(use_ray)" in inspect.getsource(ablation.run_one)


# --------------------------------------------------------------------------
# the calibration set
# --------------------------------------------------------------------------
def test_calibration_frames_newest_first_deterministic_no_rng():
    replay = [_traj(10, 0), _traj(10, 100), _traj(10, 200)]
    before = _rng_states()
    f = ablation.bn_calibration_frames(replay, 15)
    assert _rng_states() == before
    # pool = newest trajectory (200..209) then the previous (100..109): 20
    # frames thinned by stride 20/15 -> the first frame is the newest
    # trajectory's first.
    assert isinstance(f, list)                 # no full-pool stack (L6)
    vals = [int(x[0, 0, 0]) for x in f]
    assert len(vals) == 15 and vals[0] == 200
    assert set(vals) <= set(range(200, 210)) | set(range(100, 110))
    assert vals == [int(x[0, 0, 0]) for x in
                    ablation.bn_calibration_frames(replay, 15)]
    small = ablation.bn_calibration_frames(replay, 1000)   # all 30 frames
    assert len(small) == 30 and int(small[0][0, 0, 0]) == 200
    assert ablation.bn_calibration_frames([], 10) is None


def test_calibration_frames_accept_tuple_records():
    replay = [[(np.zeros((12, *muzero.IMAGE_RESHAPE), np.uint8), 0, 0.0)] * 3]
    assert len(ablation.bn_calibration_frames(replay, 8)) == 3


# --------------------------------------------------------------------------
# the recalibration
# --------------------------------------------------------------------------
def _frames(n, seed=0):
    g = np.random.default_rng(seed)
    return g.integers(0, 255, (n, 12, *muzero.IMAGE_RESHAPE), dtype=np.uint8)


def _age_bn(trainer):
    """Give every BN layer non-default running statistics and a large
    batch count, as a trained net has -- so a recalibration that forgot to
    reset them would be dominated by the old values."""
    for n in ("representation", "prediction"):
        for m in getattr(trainer, n).modules():
            if isinstance(m, BN):
                m.running_mean.fill_(5.0)
                m.running_var.fill_(3.0)
                m.num_batches_tracked.fill_(1000)


def test_recalibration_touches_only_bn_statistics_of_copies():
    run = make_run()
    tr = run.trainer
    _age_bn(tr)
    before = {n: copy.deepcopy(getattr(tr, n).state_dict())
              for n in ("representation", "prediction")}
    rng = _rng_states()
    out = ablation.bn_recalibrated_states(tr, _frames(16), batch=8)
    assert _rng_states() == rng                            # no RNG drawn
    bn_buffers = set()
    for n in ("representation", "prediction"):
        for mname, m in getattr(tr, n).named_modules():
            if isinstance(m, BN):
                bn_buffers |= {f"{mname}.{b}" for b in
                               ("running_mean", "running_var",
                                "num_batches_tracked")}
        now = getattr(tr, n).state_dict()
        assert all(torch.equal(now[k], before[n][k]) for k in now)  # trainer
        changed = {k for k in now
                   if not torch.equal(out[n][k].to(now[k].device), now[k])}
        assert changed and changed <= bn_buffers, n         # only BN stats
        assert all(v.device.type == "cpu" for v in out[n].values()
                   if torch.is_tensor(v))
    assert bn_buffers, "no BN layers found -- the test would be vacuous"


def test_recalibrated_statistics_are_the_cumulative_root_batch_average():
    """momentum=None over two equal INTERLEAVED batches (frames[0::2],
    frames[1::2]): the first BN layer's running mean is the average of the
    two batches' per-channel input means."""
    run = make_run()
    _age_bn(run.trainer)                  # a missing reset would show here
    rep = run.trainer.representation
    frames = _frames(16, seed=1)
    probe = copy.deepcopy(rep).train()
    first = next(m for m in probe.modules() if isinstance(m, BN))
    seen = []
    first.register_forward_hook(lambda m, i, o: seen.append(i[0].detach()))
    with torch.no_grad():
        for b in (0, 1):
            probe(torch.tensor(frames[b::2], dtype=torch.float32) / 255.0)
    dims = [0] + list(range(2, seen[0].dim()))
    want = (seen[0].mean(dims) + seen[1].mean(dims)) / 2
    out = ablation.bn_recalibrated_states(run.trainer, list(frames), batch=8)
    name = next(n for n, m in rep.named_modules() if isinstance(m, BN))
    assert torch.allclose(out["representation"][f"{name}.running_mean"],
                          want, atol=1e-5)
    # The VARIANCE pins the batching: the mean of batch means is partition-
    # free, the mean of within-batch (unbiased) variances is not -- so this
    # distinguishes interleaved from contiguous batches (bnrecal r1 L1).
    var = (seen[0].var(dims, unbiased=True) + seen[1].var(dims, unbiased=True)) / 2
    assert torch.allclose(out["representation"][f"{name}.running_var"],
                          var, rtol=1e-4, atol=1e-6)


# --------------------------------------------------------------------------
# the pipeline: only the actor / eval-actor ships carry it
# --------------------------------------------------------------------------
class _Remote:
    def __init__(self, log, name):
        self.log, self.name = log, name

    def remote(self, *a, **k):
        self.log.append((self.name, a, k))
        return (self.name, a)


class _Actor:
    def __init__(self, log):
        self.set_weights = _Remote(log, "set_weights")
        self.play = _Remote(log, "play")
        self.eval_episodes = _Remote(log, "eval_episodes")


def _pipeline(monkeypatch, flag):
    monkeypatch.setattr(muzero, "BN_RECAL_ACTING", flag)
    monkeypatch.setattr(muzero, "BN_RECAL_FRAMES", 256)
    monkeypatch.setattr(muzero, "BIRTH_SEARCH", True)
    run = make_run()
    run.replay = [[{"obs": f} for f in _frames(140, seed=s)] for s in (2, 3)]
    run.trainer.training_steps = 500
    p = object.__new__(ablation.RayPipeline)
    log = []
    p.run, p.server, p.seed, p.episodes_per_iter = run, None, 0, 1
    p._states_cache = (None, None)
    p._acting_cache = (None, None)
    p.ray = type("R", (), {"get": staticmethod(lambda x: x)})()
    p.actor, p.eval_actor = _Actor(log), _Actor(log)
    p.pending, p.pending_eval = None, None
    return p, log


def _shipped(log):
    return [a[0] for name, a, _ in log if name == "set_weights"]


def test_flag_off_ships_the_raw_states(monkeypatch):
    p, log = _pipeline(monkeypatch, False)
    p.sync()
    assert _shipped(log)[0] is p._states()
    assert p.snapshot_acting_states() is None


def test_flag_on_ships_recalibrated_states_to_actor_and_eval(monkeypatch):
    p, log = _pipeline(monkeypatch, True)
    p.sync()
    p.request_eval(5, 2, {}, None, 0.0)
    acting, ev = _shipped(log)
    raw = p._states()
    assert acting is ev                                   # one cached set
    for n in ("dynamics", "value_prefix"):
        assert acting[n] is raw[n]
    for n in ("representation", "prediction"):
        diff = [k for k in raw[n] if not torch.equal(acting[n][k], raw[n][k])]
        assert diff and all(("running_" in k) or ("num_batches" in k)
                            for k in diff), n
    snap = p.snapshot_acting_states()
    assert set(snap) == {"representation", "prediction"}
    assert snap["prediction"] is acting["prediction"]
    # a new training step recomputes; the same step reuses the cache
    assert p._acting_states() is acting
    p.run.trainer.training_steps += 1
    assert p._acting_states() is not acting


def test_birth_search_keeps_the_raw_states(monkeypatch):
    p, log = _pipeline(monkeypatch, True)
    p.sync()
    p.request(3)
    assert p._birth_states is p._states()


def test_empty_buffer_ships_raw_statistics(monkeypatch):
    p, log = _pipeline(monkeypatch, True)
    p.run.replay = []
    p.sync()
    raw = p._states()
    assert all(torch.equal(_shipped(log)[0][n][k], raw[n][k])
               for n in raw for k in raw[n])


# --------------------------------------------------------------------------
# snapshot + probe
# --------------------------------------------------------------------------
def test_snapshot_stores_the_acting_states(monkeypatch, tmp_path):
    run = make_run()
    monkeypatch.setattr(ablation, "RESULTS", tmp_path)
    monkeypatch.setattr(ablation, "TAG", "")
    acting = ablation.bn_recalibrated_states(run.trainer, _frames(8), batch=8)
    p1 = ablation.save_weight_snapshot(run, ARM, 1, {}, 5, acting_states=acting)
    d1 = torch.load(p1, map_location="cpu", weights_only=False)
    assert set(d1["acting_states"]) == {"representation", "prediction"}
    p2 = ablation.save_weight_snapshot(run, ARM, 1, {}, 10)
    assert "acting_states" not in torch.load(p2, map_location="cpu",
                                             weights_only=False)


def _probe_fn():
    tree = ast.parse(PROBE.read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == "apply_acting_states")
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "p", "exec"), ns)
    return ns["apply_acting_states"]


def test_probe_applies_the_acting_states_and_refuses_without():
    apply = _probe_fn()
    run = make_run()
    modes = []
    run.set_train_mode = modes.append
    acting = ablation.bn_recalibrated_states(run.trainer, _frames(8), batch=8)
    apply(run, {"acting_states": acting})
    now = run.trainer.prediction.state_dict()
    assert all(torch.equal(now[k], acting["prediction"][k].to(now[k].device))
               for k in now)
    assert modes == [False]                    # eval mode after loading
    with pytest.raises(SystemExit, match="no acting_states"):
        apply(run, {})


def test_probe_main_loop_applies_acting_states_under_the_flag():
    """The eval loop itself (not just the helper's def): `d = load(...)`,
    then `if ACTING: apply_acting_states(run, d)`, and the -acting file."""
    src = PROBE.read_text(encoding="utf-8")
    assert 'ACTING = os.environ.get("PROBE_ACTING_STATES") == "1"' in src
    loops = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.For)
             and "load(run, path)" in ast.unparse(n)]
    assert len(loops) == 1
    body = loops[0].body
    i = next(k for k, n in enumerate(body)
             if ast.unparse(n) == "d = load(run, path)")
    nxt = body[i + 1]
    assert isinstance(nxt, ast.If) and ast.unparse(nxt.test) == "ACTING"
    assert [ast.unparse(x) for x in nxt.body] == ["apply_acting_states(run, d)"]
    loop = ast.unparse(loops[0])
    assert "suffix = '-acting' if ACTING else ''" in loop
    opens = [n for n in ast.walk(loops[0]) if isinstance(n, ast.Call)
             and ast.unparse(n.func) == "open"]
    assert len(opens) == 1 and "{suffix}" in ast.unparse(opens[0].args[0])
    rec = next(n for n in ast.walk(loops[0]) if isinstance(n, ast.Assign)
               and ast.unparse(n.targets[0]) == "rec")
    keys = [ast.unparse(k) for k in rec.value.keys]
    assert "'acting_states'" in keys and "'variant'" in keys


# --------------------------------------------------------------------------
# supervisor
# --------------------------------------------------------------------------
def test_supervisor_variant():
    tree = ast.parse(SUP.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and getattr(n.targets[0], "id", "") == "VARIANTS")
    arm, stem, flags = ast.literal_eval(node.value)["ray"]
    assert arm == ARM and stem == "pong-bnrecal"
    assert flags == ["--ray", "--cpu-actor"]
    aud = next(n for n in tree.body if isinstance(n, ast.Assign)
               and getattr(n.targets[0], "id", "") == "AUDITED")
    audited = ast.literal_eval(aud.value)
    assert "tests/test_bnrecal.py" in audited
    assert "benchmarks/probes/probe_exhaustive_eval.py" in audited


def test_run_one_snapshots_carry_the_pipelines_acting_states(
        monkeypatch, tmp_path, restore_rngs):  # noqa: F811
    """run_one --ray --snapshot-weights (in-process fake pipeline): each
    eval-row snapshot stores pipeline.snapshot_acting_states()."""
    import torch as _t
    _t.set_num_threads(2)
    from tests.test_prioract2 import _fake_pipe
    marker = {"representation": {"marker": _t.tensor([7.0])},
              "prediction": {"marker": _t.tensor([8.0])}}
    fake = _fake_pipe([], 0)
    fake.snapshot_acting_states = lambda self: marker
    monkeypatch.setattr(ablation, "RayPipeline", fake)
    monkeypatch.setattr(ablation, "ENV_NAME", "Pong-v4")
    monkeypatch.setattr(ablation, "FRAMESKIP", 4)
    monkeypatch.setattr(ablation, "EPISODE_CAP", 10)
    monkeypatch.setattr(ablation, "RESULTS", tmp_path)
    monkeypatch.setattr(ablation, "TAG", "")
    monkeypatch.setitem(ablation.ARMS, "e2e-bnrecal", {
        **ablation.ARMS[ARM], "TRAINING_BATCHES": 16})
    ablation.run_one("e2e-bnrecal", 1, 1, 1, 10, 1, tmp_path / "a.jsonl",
                     checkpoint_every=1, use_ray=True, cpu_actor=True,
                     snapshot_weights=True)
    snaps = sorted(tmp_path.glob("snapshots*/weights-iter*.pth"))
    assert snaps
    d = _t.load(snaps[-1], map_location="cpu", weights_only=False)
    assert _t.equal(d["acting_states"]["prediction"]["marker"], _t.tensor([8.0]))



# --------------------------------------------------------------------------
# round-1 fixes
# --------------------------------------------------------------------------
def test_calibration_size_follows_BN_RECAL_FRAMES(monkeypatch):
    p, log = _pipeline(monkeypatch, True)
    seen = []
    real = ablation.bn_calibration_frames
    monkeypatch.setattr(ablation, "bn_calibration_frames",
                        lambda replay, n: seen.append(n) or real(replay, n))
    monkeypatch.setattr(muzero, "BN_RECAL_FRAMES", 300)
    p.sync()
    assert seen == [300]


def test_pool_branches_ship_the_acting_states(monkeypatch):
    p, log = _pipeline(monkeypatch, True)
    server_log = []
    p.server = type("Srv", (), {"set_weights": _Remote(server_log, "srv")})()
    p.workers = []
    p.sync()
    p.request_eval(5, 1, {}, None, 0.0)
    acting = p._acting_states()
    shipped = [a[0] for _, a, _ in server_log]
    assert len(shipped) == 2 and all(x is acting for x in shipped)
    assert server_log[1][2] == {"which": "eval"}


def test_behaviour_prior_comes_from_the_acting_nets(monkeypatch):
    """bnrecal r1 MEDIUM-1: birth search searches with the raw nets but
    logs the prior the actor SAMPLED -- the acting (recalibrated) nets."""
    import muzero_vtree
    from tests.test_vtree import dezero_heads
    run = make_run()
    _age_bn(run.trainer)
    tr = run.trainer
    # zero-init heads give a uniform prior whatever the BN statistics;
    # re-randomized heads give unsaturated priors the BN shift moves
    # (raw maxp ~0.57 vs acting ~0.83 on these frames)
    dezero_heads({n: getattr(tr, n) for n in ("representation", "dynamics",
                                              "prediction", "value_prefix")}, 3)

    class _Engine:
        def __init__(self, nets, actions, device):
            pass

        def search(self, obs, num_simulations, c_puct, discount, root_noise):
            n = len(obs)
            return np.full((n, ablation.ACTIONS), 1.0 / ablation.ACTIONS), np.zeros(n), None

    monkeypatch.setattr(muzero_vtree, "VectorizedPUCT", _Engine)
    traj = [(o, 0, 0.0) for o in _frames(6, seed=4)]
    raw = {n: getattr(tr, n) for n in ("representation", "dynamics",
                                       "prediction", "value_prefix")}
    for net in raw.values():
        net.eval()        # birth search runs eval mode (running statistics)
    acting_states = ablation.bn_recalibrated_states(tr, list(_frames(16)), batch=8)
    acting = {n: copy.deepcopy(getattr(tr, n)) for n in ("representation", "prediction")}
    for n, net in acting.items():
        net.load_state_dict(acting_states[n])
        net.eval()

    def prior_of(nets):
        with torch.no_grad():
            x = torch.tensor(np.stack([r[0] for r in traj[:-1]]),
                             dtype=torch.float32) / 255.0
            logits, _, _ = nets["prediction"](nets["representation"](x))
            return torch.softmax(logits.float(), dim=1).numpy()

    assert not np.allclose(prior_of(raw), 1.0 / prior_of(raw).shape[1])
    run._birth_search(traj, True, nets=raw, prior_nets=acting)
    assert np.allclose(run._birth_priors, prior_of(acting), atol=1e-6)
    assert not np.allclose(run._birth_priors, prior_of(raw), atol=1e-3)
    run._birth_search(traj, True, nets=raw)                  # default: raw
    assert np.allclose(run._birth_priors, prior_of(raw), atol=1e-6)


def test_collect_passes_the_actors_nets_for_the_behaviour_prior(monkeypatch):
    """The pipeline captures the acting set at request time and hands
    collect's reanalyse birth_nets = raw, birth_prior_nets = acting."""
    p, log = _pipeline(monkeypatch, True)
    p.sync()
    p.request(3)
    traj = [(o, 0, 0.0) for o in _frames(3, seed=5)]
    p.pending = (["fut"], 500)
    p.ray = type("R", (), {"get": staticmethod(
        lambda x: [(traj, True, 2)] if x == ["fut"] else x)})()
    p._birth_nets = None
    got = {}

    def fake_reanalyse(t, terminated, born_step, birth_nets, birth_prior_nets):
        got.update(birth=birth_nets, prior=birth_prior_nets)
        return t

    p.run.reanalyse = fake_reanalyse
    monkeypatch.setattr(ablation, "evict_replay", lambda r: None)
    acting, raw = p._acting_states(), p._states()
    p.collect()
    for n in ("representation", "prediction"):
        sd = got["prior"][n].state_dict()
        assert all(torch.equal(sd[k], acting[n][k].to(sd[k].device)) for k in sd)
        bd = got["birth"][n].state_dict()
        assert all(torch.equal(bd[k], raw[n][k].to(bd[k].device)) for k in bd)


def test_supervisor_refuses_any_seed_but_1():
    import subprocess
    r = subprocess.run([sys.executable, str(SUP), "ray", "0"], capture_output=True,
                       text=True, cwd=str(ROOT), timeout=60)
    assert r.returncode != 0 and "pre-registered as SEED 1" in r.stderr
