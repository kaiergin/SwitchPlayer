#!/usr/bin/env python3
"""Sharded storage for recorded Switch play: frames, actions, rewards, timing.

What gets stored, and why:

* The REGISTERED 90x160 reference frame, not the raw 1920x1080 capture and not
  the 84x84 observation. Raw frames are 60x larger; the observation throws away
  everything outside PLAY_AREA. The reference frame is what every ROI in
  tetris_masks is defined against, so observations, board occupancy and the
  play-area crop can all be recomputed later. That matters because the reward
  function and the ROIs have both changed repeatedly.

* The RAW 64-bit controller command, not an index into tetris_enum. Human play
  produces chords (left + rotate on the same frame) that have no index, and an
  inverse-dynamics model that can only emit 8 mutually exclusive actions cannot
  represent what a human actually did. The index is stored alongside when the
  command happens to match one.

* Monotonic timestamps on every step, because the action logged at time t does
  not appear on screen until t + latency, and that offset has to be measured
  before any of this is usable as IDM training data.

* Native-resolution numeral strips ONLY on frames with a glyph candidate. The
  garbage counter is read at native resolution (clear_numeral_hires), so
  recomputing reward from the reference frame alone is impossible -- but the
  strips are 370 KB each and storing them every step would cost ~26 GB/hour.
  Candidate frames are rare, so this keeps recomputation possible for the
  frames where it matters.

Layout:
    <dir>/meta.json              provenance, shapes, constants
    <dir>/shard_00000.npz        t, command, action, frames, reward, event, state
    <dir>/strips/000123.png      native numeral crops, keyed by global step
"""

import io
import json
import os
import subprocess
import time

import numpy as np

SHARD_STEPS = 1500          # ~75 s at 20 Hz; ~65 MB uncompressed per shard


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
            text=True).strip()
    except Exception:
        return "unknown"


class SessionRecorder:
    """Append steps, flush in shards. Call close() or use as a context manager."""

    def __init__(self, outdir, source="human", note="", shard_steps=SHARD_STEPS,
                 frame_shape=(90, 160, 3), timing=None):
        self.dir = outdir
        self.strips = os.path.join(outdir, "strips")
        os.makedirs(self.strips, exist_ok=True)
        self.shard_steps = shard_steps
        self.frame_shape = tuple(frame_shape)
        self.n = 0                      # global step count
        self.shard_i = 0
        self._reset_buffers()
        self.events, self.states = ["none"], ["none"]
        self.meta = {
            "source": source, "note": note,
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "git": _git_commit(),
            "frame_shape": list(self.frame_shape),
            "shard_steps": shard_steps,
            # An IDM is only valid for the timing it was recorded under: change
            # the step period or the link latency and the action-to-observation
            # offset moves, silently invalidating every label.
            "timing": timing or {},
        }
        self._write_meta()

    def _reset_buffers(self):
        self.t, self.cmd, self.act = [], [], []
        self.frames, self.reward, self.event, self.state = [], [], [], []

    def _idx(self, table, value):
        value = value or "none"
        if value not in table:
            table.append(value)
        return table.index(value)

    def add(self, t, command, frame, reward=0.0, event=None, state=None, action=-1):
        """One step. `frame` must already be the registered reference frame."""
        if tuple(frame.shape) != self.frame_shape:
            raise ValueError("frame is %s, expected %s -- pass the output of "
                             "tetris_masks.prepare()" % (frame.shape, self.frame_shape))
        self.t.append(float(t))
        self.cmd.append(int(command))
        self.act.append(int(action))
        self.frames.append(np.ascontiguousarray(frame, dtype=np.uint8))
        self.reward.append(float(reward))
        self.event.append(self._idx(self.events, event))
        self.state.append(self._idx(self.states, state))
        self.n += 1
        if len(self.t) >= self.shard_steps:
            self.flush()
        return self.n - 1

    def add_strip(self, step_index, strip):
        """Native-resolution numeral crop for one step (rare)."""
        import cv2
        cv2.imwrite(os.path.join(self.strips, "%06d.png" % step_index),
                    np.ascontiguousarray(strip))

    def flush(self):
        if not self.t:
            return
        path = os.path.join(self.dir, "shard_%05d.npz" % self.shard_i)
        np.savez_compressed(
            path,
            t=np.asarray(self.t, np.float64),
            command=np.asarray(self.cmd, np.uint64),
            action=np.asarray(self.act, np.int16),
            frames=np.stack(self.frames),
            reward=np.asarray(self.reward, np.float32),
            event=np.asarray(self.event, np.int16),
            state=np.asarray(self.state, np.int16),
        )
        self.shard_i += 1
        self._reset_buffers()
        self._write_meta()

    def _write_meta(self):
        self.meta.update({"steps": self.n, "shards": self.shard_i,
                          "events": self.events, "states": self.states})
        with io.open(os.path.join(self.dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2)

    def close(self):
        self.flush()
        self.meta["ended"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self._write_meta()
        return self.n

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def load_session(outdir):
    """Concatenate every shard. Returns (dict of arrays, meta)."""
    with io.open(os.path.join(outdir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    keys = ("t", "command", "action", "frames", "reward", "event", "state")
    parts = {k: [] for k in keys}
    for i in range(meta["shards"]):
        z = np.load(os.path.join(outdir, "shard_%05d.npz" % i))
        for k in keys:
            parts[k].append(z[k])
    return ({k: (np.concatenate(v) if v else np.array([])) for k, v in parts.items()},
            meta)


def observations(frames, play_area=None, obs_size=None):
    """Reference frames -> the exact array the agent sees. Recomputed, not stored."""
    import cv2
    import tetris_masks as T
    y0, y1, x0, x1 = play_area or T.PLAY_AREA
    size = obs_size or T.OBS_SIZE
    out = np.empty((len(frames), size[1], size[0], 3), np.uint8)
    for i, f in enumerate(frames):
        out[i] = cv2.resize(f[y0:y1, x0:x1], size, interpolation=cv2.INTER_AREA)
    return out


def transitions(data, min_gap=None):
    """Indices where the held command changed -- the only steps an inverse
    dynamics model can learn from, and where latency must be measured."""
    cmd = data["command"]
    return np.flatnonzero(np.diff(cmd) != 0) + 1
