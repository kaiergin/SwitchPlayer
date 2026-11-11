"""Test H: the decision windows and the READS exclusion's stack digest, one copy
for the trainer (stage C), the truth's floors (stage B) and the scorer (D1).
Kept out of stage A's fingerprinted module, so a fix here invalidates what
reads it (trained models, floors.json, logits) and never stage A's data or
stage B's truth. Nothing here depends on the emulator or the nets."""
import hashlib
import pathlib

import numpy as np

KIND_START, KIND_OPEN, KIND_MOVE = 0, 1, 2  # stage A's event kinds
K_MAX = 16                                  # future offsets, t+1 .. t+16


def source_hash():
    return hashlib.blake2b(pathlib.Path(__file__).read_bytes(), digest_size=16).hexdigest()


SOURCE = source_hash()


def event_stacks(kind, frame_offset=0):
    """Per event, the frame indices of the stack before it and after it, by
    play_episode's rule (a life's start frame tiled, each step's frame pushed
    before the next choice; stage A's stacks_from_frames as indices). Start
    events have no 'before'."""
    before, after = [None] * len(kind), [None] * len(kind)
    stack = frame = None
    for i, k in enumerate(kind):
        if int(k) == KIND_START:
            frame = frame_offset + i
            stack = [frame] * 4
            after[i] = list(stack)
            continue
        stack = stack[1:] + [frame]
        before[i] = list(stack)
        frame = frame_offset + i
        after[i] = stack[1:] + [frame]
    return before, after


def decision_windows(arr, frame_offset=0):
    """Every decision's 17 stacks as frame indices: (decision ids, event ids,
    moves, windows), windows[n] (K_MAX + 1, 4), row 0 obs_t, row k obs_{t+k};
    past the life's last base step a row repeats the life's last stack (LIFE
    ENDS). A life that runs out without an end flag is a malformed record:
    ValueError."""
    kind, end = arr["kind"], arr["end"]
    before, after = event_stacks(kind, frame_offset)
    dec, ev, mv, win = [], [], [], []
    n = len(kind)
    for i in range(n):
        if int(kind[i]) != KIND_MOVE:
            continue
        rows = [before[i]]
        last = None
        for k in range(1, K_MAX + 1):
            j = i + k - 1
            if last is not None:
                rows.append(last)
                continue
            if j >= n or int(kind[j]) == KIND_START:
                raise ValueError(f"event {i}: the life ends without an end flag")
            rows.append(after[j])
            if int(end[j]) != 0:
                last = after[j]
        dec.append(int(arr["decision"][i]))
        ev.append(i)
        mv.append(int(arr["move"][i]))
        win.append(rows)
    return (np.array(dec, np.int32), np.array(ev, np.int32), np.array(mv, np.int64),
            np.array(win, np.int64).reshape(-1, K_MAX + 1, 4))


def stack_digest(frames, rows):
    """The READS exclusion's key for one stack: blake2b of its 4 frames' bytes."""
    return hashlib.blake2b(frames[list(rows)].tobytes(), digest_size=16).digest()


def window_hashes(arr):
    """(decision ids, per decision its 17 window stacks' digests)."""
    d, _, _, win = decision_windows(arr)
    cache, out = {}, []
    for w in win:
        row = []
        for r in w:
            t = tuple(int(x) for x in r)
            h = cache.get(t)
            if h is None:
                h = cache[t] = stack_digest(arr["frames"], t)
            row.append(h)
        out.append(row)
    return d, out
