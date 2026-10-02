"""Policy-only acting without ReDo, always parallel, with each game added
to the buffer as soon as it finishes (RAY_EAGER_COLLECT; Kai, 2026-10-01):
effzero-repro-faithful-prioract-noredo."""
import contextlib
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(__import__("pathlib").Path(
    __file__).resolve().parents[1] / "experiments"))
import ablation
import muzero

ARM = "effzero-repro-faithful-prioract-noredo"
PRIOR = "effzero-repro-faithful-prioract"
NOREDO = "effzero-repro-faithful-searchact-noredo"
SEQVTREE = "effzero-repro-faithful-seqvtree"


def _eff_delta(a, b):
    a, b = ablation.ARMS[a], ablation.ARMS[b]

    def eff(spec, k):
        return spec[k] if k in spec else getattr(muzero, k, None)
    return {k for k in set(a) | set(b) if eff(a, k) != eff(b, k)}


def test_arm_is_prioract_with_redo_off_and_eager_collect():
    a, b = ablation.ARMS[PRIOR], ablation.ARMS[ARM]
    raw = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert raw == {"REDO_EVERY", "STEP_DIAGNOSTICS", "RAY_EAGER_COLLECT",
                   "_launch", "_doc"}
    assert b["REDO_EVERY"] == 0 and b["RAY_EAGER_COLLECT"] is True
    assert b["USE_SEARCH_ACTING"] is False and b["BIRTH_SEARCH"] is True
    assert b["_launch"] == {"ray": True, "cpu_actor": True}
    assert "effzero-repro-faithful-prioract-noredo-seq" not in ablation.ARMS


ACTING = {"USE_SEARCH_ACTING", "BIRTH_SEARCH", "PRIOR_ACT_STAGED_TEMP",
          "EVAL_PRIOR_ONLY"}


def test_deltas_vs_the_search_acting_runs():
    """vs noredo (same stack, lagged collect): who acts + eager collect;
    vs seqvtree (search acting, sequential -- no lag): who acts + launch
    mode (+ inert ReDo knobs)."""
    assert _eff_delta(ARM, NOREDO) <= ACTING | {"RAY_EAGER_COLLECT", "_launch",
                                               "_doc"}
    assert _eff_delta(ARM, NOREDO) >= {"USE_SEARCH_ACTING", "RAY_EAGER_COLLECT"}
    d = _eff_delta(ARM, SEQVTREE)
    assert d <= ACTING | {"RAY_EAGER_COLLECT", "REDO_NETS", "REDO_TAU",
                          "_launch", "_doc"}
    assert d >= {"USE_SEARCH_ACTING", "BIRTH_SEARCH", "RAY_EAGER_COLLECT"}


def test_defaults_off():
    assert muzero.RAY_EAGER_COLLECT is False
    assert muzero.EAGER_POLL_STEPS == 10


# --------------------------------------------------------------------------
# train_block (the iteration loop's gradient steps)
# --------------------------------------------------------------------------
def _harness(monkeypatch, ready_at=None, pending=True, eager=True):
    """A stub trainer + fake pipeline; records every train_batches block
    and every collect (with the step count at that moment)."""
    blocks, collects, polls = [], [], []
    trainer = SimpleNamespace(training_steps=0)

    async def train_step(batch):
        return 0.0

    trainer.train_step = train_step
    run = SimpleNamespace(trainer=trainer)

    @contextlib.contextmanager
    def fake_batches(r, n):
        blocks.append(n)
        yield lambda: None

    monkeypatch.setattr(ablation, "train_batches", fake_batches)
    monkeypatch.setattr(muzero, "RAY_EAGER_COLLECT", eager)
    monkeypatch.setattr(muzero, "REDO_EVERY", 0)
    monkeypatch.setattr(muzero, "STEP_DIAGNOSTICS", False)

    def acting_ready():
        polls.append(trainer.training_steps)
        return ready_at is not None and trainer.training_steps >= ready_at

    def collect():
        collects.append(trainer.training_steps)
        pipe.pending = None

    pipe = SimpleNamespace(pending=object() if pending else None,
                           acting_ready=acting_ready, collect=collect)
    return run, pipe, blocks, collects, polls


def test_default_is_one_unchanged_block(monkeypatch):
    run, pipe, blocks, collects, polls = _harness(monkeypatch, ready_at=0,
                                                 eager=False)
    ablation.train_block(run, 100, pipeline=pipe)
    assert blocks == [100] and collects == [] and polls == []
    assert run.trainer.training_steps == 100
    assert getattr(run, "_eager_window", None) is None


def test_eager_collect_mid_block_then_fresh_block(monkeypatch):
    """Ready from step 23 -> the poll at step 30 sees it: collect at 30,
    then the remaining 70 steps in a NEW block (caches reopened)."""
    run, pipe, blocks, collects, polls = _harness(monkeypatch, ready_at=23)
    ablation.train_block(run, 100, pipeline=pipe, it=7)
    assert collects == [30]
    assert blocks == [100, 70]
    assert polls == [0, 10, 20, 30]           # before step 1, then every 10
    assert run.trainer.training_steps == 100 and pipe.pending is None
    assert run._eager_window == [[7, 30]]


def test_a_game_finished_before_the_block_is_collected_first(monkeypatch):
    run, pipe, blocks, collects, polls = _harness(monkeypatch, ready_at=0)
    ablation.train_block(run, 100, pipeline=pipe, it=4)
    assert collects == [0] and polls == [0] and blocks == [100]
    assert run._eager_window == [[4, 0]]
    ablation.train_block(run, 50, pipeline=pipe, it=5)  # nothing in flight now
    assert run._eager_window == [[4, 0]] and polls == [0]


def test_row_field_and_window_reset(monkeypatch):
    run, pipe, blocks, collects, polls = _harness(monkeypatch, ready_at=None)
    ablation.train_block(run, 20, pipeline=pipe, it=2)
    pipe.pending = object()
    ablation.train_block(run, 20, pipeline=pipe, it=3)
    assert run._eager_window == [[2, None], [3, None]]
    row = {}
    ablation.Run._merge_row_stats(run, row)
    assert row["eager_collect_steps"] == [[2, None], [3, None]]
    assert run._eager_window is None
    row = {}
    ablation.Run._merge_row_stats(run, row)
    assert "eager_collect_steps" not in row


def test_never_ready_leaves_the_episode_for_the_iteration_end(monkeypatch):
    run, pipe, blocks, collects, polls = _harness(monkeypatch, ready_at=None)
    ablation.train_block(run, 35, pipeline=pipe, it=9)
    assert collects == [] and blocks == [35]
    assert polls == [0, 10, 20, 30]               # never on the last step
    assert pipe.pending is not None and run._eager_window == [[9, None]]


def test_ready_only_at_the_last_step_does_not_split(monkeypatch):
    run, pipe, blocks, collects, polls = _harness(monkeypatch, ready_at=40)
    ablation.train_block(run, 40, pipeline=pipe)
    assert collects == [] and blocks == [40]


def test_nothing_in_flight_means_no_polling(monkeypatch):
    """The primed first iteration (collected before training)."""
    run, pipe, blocks, collects, polls = _harness(monkeypatch, ready_at=0,
                                                 pending=False)
    ablation.train_block(run, 50, pipeline=pipe)
    assert polls == [] and collects == [] and blocks == [50]


def test_iteration_end_collect_skips_an_already_collected_episode():
    import inspect
    src = inspect.getsource(ablation.run_one)
    assert "if not primed_sequential and pipeline.pending is not None:" in src
    assert "eager_collect_gate(use_ray)" in src            # startup gate wired


@pytest.mark.parametrize("flag,value", [("ASYNC_LOOP", True),
                                        ("REFRESH_SATURATE", True),
                                        ("TRAIN_PREFETCH", True)])
def test_eager_gate_refuses_unsupported_loops(monkeypatch, flag, value):
    monkeypatch.setattr(muzero, "RAY_EAGER_COLLECT", True)
    monkeypatch.setattr(muzero, flag, value)
    with pytest.raises(SystemExit, match="pipelined iteration loop only"):
        ablation.eager_collect_gate(True)


@pytest.mark.parametrize("poll", [0, -10, 2.5, True, None])
def test_eager_gate_refuses_a_bad_poll_interval(monkeypatch, poll):
    monkeypatch.setattr(muzero, "RAY_EAGER_COLLECT", True)
    monkeypatch.setattr(muzero, "EAGER_POLL_STEPS", poll)
    with pytest.raises(SystemExit, match="EAGER_POLL_STEPS"):
        ablation.eager_collect_gate(True)


def test_eager_gate_needs_ray_and_passes_the_arm(monkeypatch):
    monkeypatch.setattr(muzero, "RAY_EAGER_COLLECT", True)
    with pytest.raises(SystemExit, match="needs --ray"):
        ablation.eager_collect_gate(False)
    ablation.eager_collect_gate(True)
    with ablation.arm_config(ARM):
        ablation.eager_collect_gate(True)        # the arm's own config passes


def test_launch_pin():
    ablation.launch_gate(ablation.ARMS[ARM], True, True)
    with pytest.raises(SystemExit, match="pins its launch mode"):
        ablation.launch_gate(ablation.ARMS[ARM], False, False)


def test_supervisor_variants_match_the_pin():
    import ast
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1] / "benchmarks" /
           "supervisors" / "supervise_prioract2.py").read_text(encoding="utf-8")
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.Assign)
                and getattr(n.targets[0], "id", "") == "VARIANTS")
    variants = ast.literal_eval(node.value)
    assert set(variants) == {"ray"}
    for arm, _stem, flags in variants.values():
        want = ablation.ARMS[arm]["_launch"]
        assert ("--ray" in flags) == want["ray"], arm
        assert ("--cpu-actor" in flags) == want.get("cpu_actor", False), arm



# --------------------------------------------------------------------------
# run_one end to end (audit r1 L3): RayPipeline replaced by an in-process
# fake -- the episode is played on the driver at request time and becomes
# "ready" a fixed number of training steps later.
# --------------------------------------------------------------------------
@pytest.fixture
def restore_rngs():
    import random

    import numpy as np
    import torch
    states = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
              torch.get_num_threads())
    yield
    random.setstate(states[0])
    np.random.set_state(states[1])
    torch.set_rng_state(states[2])
    torch.set_num_threads(states[3])      # round-2 audit L-A: no leak


def _fake_pipe(events, ready_after):
    class FakePipe:
        def __init__(self, run, spec, seed, iterations, train_steps,
                     episodes_per_iter, ray_workers=0, cpu_actor=False):
            self.run, self.pending, self.epi = run, None, episodes_per_iter
            self.pending_eval = None

        def sync(self):
            events.append(("sync", self.run.trainer.training_steps))

        def request(self, it):
            before = self.run.env_steps
            eps = [self.run.play_episode() for _ in range(self.epi)]
            self.run.env_steps = before          # counted at collect
            self.pending = (eps, self.run.trainer.training_steps)
            events.append(("request", it, self.run.trainer.training_steps))

        def acting_ready(self):
            return (self.pending is None or self.run.trainer.training_steps
                    >= self.pending[1] + ready_after)

        def collect(self, admit=None):
            eps, steps = self.pending
            self.pending = None
            for traj, _, n, term in eps:
                self.run.env_steps += n
                self.run.replay.append(self.run.reanalyse(
                    traj, terminated=term, born_step=steps))
                ablation.evict_replay(self.run.replay)
            events.append(("collect", self.run.trainer.training_steps, steps,
                           self.run.trainer.representation.training))

        def request_eval(self, it, episodes, diagnostics, write_row,
                         elapsed_at):
            self.pending_eval = (it, diagnostics, write_row,
                                 self.run.trainer.training_steps,
                                 self.run.env_steps, elapsed_at)

        def drain_eval(self):
            if self.pending_eval is not None:
                it, d, w, ts, es, el = self.pending_eval
                self.pending_eval = None
                w(it, {**ablation.eval_summary([(0.0, 1, True)]), **d}, ts,
                  es, el)

        def poll_refresh(self):
            pass

        def shutdown(self):
            self.drain_eval()
    return FakePipe


@pytest.mark.parametrize("ready_after,want", [(15, [20, 20]), (0, [0, 0]),
                                              (10_000, [None, None])])
def test_run_one_ray_loop_end_to_end(monkeypatch, tmp_path, restore_rngs,
                                     ready_after, want):
    """3 iterations x 30 steps: iteration 1 is primed (played and collected
    before training), then one request at each boundary and exactly ONE
    collect per iteration -- mid-block at the first poll after the actor is
    ready (or before step 1, or at the iteration end if it never is) --
    stamped with the request-time step, nets in train mode, and the row
    records where each game joined."""
    import json
    import torch
    torch.set_num_threads(2)
    events = []
    monkeypatch.setattr(ablation, "RayPipeline", _fake_pipe(events, ready_after))
    monkeypatch.setattr(ablation, "ENV_NAME", "Pong-v4")
    monkeypatch.setattr(ablation, "FRAMESKIP", 4)
    monkeypatch.setattr(ablation, "EPISODE_CAP", 10)
    monkeypatch.setattr(ablation, "RESULTS", tmp_path)
    monkeypatch.setattr(ablation, "TAG", "")
    monkeypatch.setitem(ablation.ARMS, "e2e-eager", {
        **ablation.ARMS[ARM], "TRAINING_BATCHES": 16})
    out = tmp_path / "ablation.jsonl"
    ablation.run_one("e2e-eager", 0, 3, 1, 30, 1, out,
                     checkpoint_every=1, use_ray=True, cpu_actor=True)
    req = [e for e in events if e[0] == "request"]
    col = [e for e in events if e[0] == "collect"]
    assert [(e[1], e[2]) for e in req] == [(1, 0), (2, 30), (3, 60)]
    first = {20: 50, 0: 30, None: 60}
    second = {20: 80, 0: 60, None: 90}
    assert [e[1] for e in col] == [0, first[want[0]], second[want[1]]]
    assert [e[2] for e in col] == [0, 30, 60]         # request-time stamps
    # mid-block collects run inside the training block (train mode); the
    # iteration-end one after set_train_mode(False)
    assert [e[3] for e in col[1:]] == [w is not None for w in want]
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    rows.sort(key=lambda r: r["iteration"])
    assert [r["iteration"] for r in rows] == [1, 2, 3]
    assert "eager_collect_steps" not in rows[0]       # primed iteration
    assert [r["eager_collect_steps"] for r in rows[1:]] == [
        [[2, want[0]]], [[3, want[1]]]]
    assert [r["env_steps"] for r in rows] == [10, 20, 30]
