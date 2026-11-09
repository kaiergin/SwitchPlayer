"""Live detector monitor: watch the capture card and show what the masks see.

Read-only with respect to the console -- it never imports `controller`, so the
serial port stays closed and no button is ever sent. Safe to run while a human
plays.

Shows the downscaled 90x160 frame the detectors actually consume, magnified,
with every ROI drawn on it, plus the scores and a rolling event log. The point
is that misalignment is visible rather than inferred: if the two RAIL lines do
not sit on the playfield's yellow-green edges, the capture is framed
differently from the training corpus and every ROI is shifted.

    python tetris_live.py                 # auto-pick the widest device
    python tetris_live.py --device 1
    python tetris_live.py --list          # enumerate capture devices

Keys: q quit, r reset the state machine, space pause, s save the current frame.
"""
import argparse
import os
import sys
import time
from collections import deque

import cv2
import numpy as np

import tetris_masks as T

SCALE = 7
PANEL_W = 540


# A Tetris banner and its garbage counter fire a few frames apart -- measured
# 0..1s, counter first -- so "no counter for this Tetris" is only true if none
# appeared anywhere in this window.
COUNTER_WINDOW_S = 3.0


def open_capture(index, width=1920, height=1080):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def list_devices(n=4):
    for i in range(n):
        cap = open_capture(i)
        if not cap.isOpened():
            cap.release()
            continue
        ok, fr = False, None
        for _ in range(10):
            ok, fr = cap.read()
            if ok and fr is not None:
                break
        cap.release()
        if ok and fr is not None:
            h, w = fr.shape[:2]
            print(f"  device {i}: {w}x{h}  mean pixel {fr.mean():.1f}")
        else:
            print(f"  device {i}: opened but delivered no frame")


def draw(prep, masks, sm_out, events, fps, paused, show_board=True,
         raw=None, sm=None):
    """Left: the 90x160 frame the detector sees, magnified, with ROIs.
    Right: scores and the event log."""
    vis = cv2.resize(prep, (T.FRAME_W * SCALE, T.FRAME_H * SCALE),
                     interpolation=cv2.INTER_NEAREST)

    # Alignment aid: where the detector believes the playfield rails are.
    for x in (T.RAIL_L, T.RAIL_R):
        cv2.line(vis, (x * SCALE, 0), (x * SCALE, T.FRAME_H * SCALE), (0, 0, 255), 1)

    for roi, col, name in ((T.ROI_TETRIS, (0, 255, 0), "tetris"),
                           (T.ROI_GO, (255, 255, 0), "go"),
                           (T.NUM_STRIP, (0, 165, 255), "num")):
        y0, y1, x0, x1 = roi
        cv2.rectangle(vis, (x0 * SCALE, y0 * SCALE), (x1 * SCALE, y1 * SCALE), col, 1)
        cv2.putText(vis, name, (x0 * SCALE + 2, y0 * SCALE - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

    # The play area the agent will actually be given.
    y0, y1, x0, x1 = T.PLAY_AREA
    cv2.rectangle(vis, (x0 * SCALE, y0 * SCALE), (x1 * SCALE, y1 * SCALE),
                  (255, 0, 255), 2)
    cv2.putText(vis, "agent view", (x0 * SCALE + 3, y1 * SCALE - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)

    # Decoded board: a dot on every cell the clear detector counts as filled.
    # This is the thing the clear event is computed from, so it has to be
    # visible -- otherwise a spurious clear is unattributable.
    occ = masks.board_occupancy(prep) if show_board else None
    by0, by1, bx0, bx1 = T.BOARD
    rh = (by1 - by0) / T.BOARD_ROWS
    cw = (bx1 - bx0) / T.BOARD_COLS
    for r in range(T.BOARD_ROWS if show_board else 0):
        for c in range(T.BOARD_COLS):
            px = int((bx0 + (c + 0.5) * cw) * SCALE)
            py = int((by0 + (r + 0.5) * rh) * SCALE)
            if occ[r, c]:
                cv2.circle(vis, (px, py), 3, (0, 255, 0), -1)
            else:
                cv2.circle(vis, (px, py), 1, (60, 60, 60), -1)

    # Detected on the raw frame (11.5x the linear resolution) when available.
    num = (masks.clear_numeral_hires(raw, sm.box, sm.swap_rb)
           if raw is not None and sm is not None else masks.clear_numeral(prep))
    if num:
        x, y, w, h = num[:4]
        cv2.rectangle(vis, (x * SCALE, y * SCALE),
                      ((x + w) * SCALE, (y + h) * SCALE), (0, 0, 255), 2)

    panel = np.zeros((vis.shape[0], PANEL_W, 3), np.uint8)
    ts, gs = masks.tetris_score(prep), masks.go_score(prep)
    state = sm_out["state"]
    lines = [
        ("STATE", state, (0, 255, 255) if state == T.PLAYING else (200, 200, 200)),
        ("edge", f"{sm_out.get('edge', 0):6.1f}  "
                 f"{'IN GAME' if sm_out.get('edge', 0) >= T.IG_EDGE_THRESHOLD else 'out'}"
                 f"   (thr {T.IG_EDGE_THRESHOLD:.0f})",
         (120, 255, 120) if sm_out.get('edge', 0) >= T.IG_EDGE_THRESHOLD
         else (120, 160, 255)),
        ("tetris", f"{ts:+.3f}  {'FIRE' if ts >= T.TETRIS_THRESHOLD else ''}",
         (120, 255, 120) if ts >= T.TETRIS_THRESHOLD else (170, 170, 170)),
        ("go", f"{gs:+.3f}  {'FIRE' if gs >= T.GO_THRESHOLD else ''}",
         (120, 255, 120) if gs >= T.GO_THRESHOLD else (170, 170, 170)),
        ("numeral", ((f"sent {num[4]} lines  ->  +{T.REWARD_PER_GARBAGE*num[4]:.2f}"
                      if len(num) > 4 and num[4] else
                      f"x{num[0]} y{num[1]} {num[2]}x{num[3]}") if num else "-"),
         (120, 255, 120) if num else (170, 170, 170)),
        ("cells", f"{sm_out.get('cells', 0):3d} / 200    drop "
                  f"{sm_out.get('drop', 0.0):+5.1f}  (fires at {T.CLEAR_DROP})",
         (120, 255, 120) if sm_out.get('drop', 0) >= T.CLEAR_DROP
         else (170, 170, 170)),
        ("reward", f"{sm_out['reward']:+.2f}", (255, 255, 255)),
        ("EPISODE", f"{sm_out.get('ep_total', 0.0):+.2f}"
                    f"   T{sm_out.get('ep_tetris', 0)}"
                    f" G{sm_out.get('ep_garbage', 0)}"
                    f" D{sm_out.get('ep_died', 0)}",
         (0, 255, 255)),
        ("session", f"{sm_out.get('run_total', 0.0):+.2f}"
                    f"   T{sm_out.get('run_tetris', 0)}"
                    f" G{sm_out.get('run_garbage', 0)}"
                    f" D{sm_out.get('run_died', 0)}",
         (170, 170, 170)),
        ("fps", f"{fps:.1f}" + ("   PAUSED" if paused else ""), (170, 170, 170)),
        ("swap RB", str(sm_out.get("swap_rb", "?")), (170, 170, 170)),
        ("calib", sm_out.get("calib", "?"),
         (120, 255, 120) if sm_out.get("calib") == "ok" else (0, 128, 255)),
    ]
    for i, (k, v, col) in enumerate(lines):
        cv2.putText(panel, f"{k:>8s}", (8, 34 + i * 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (140, 140, 140), 1, cv2.LINE_AA)
        cv2.putText(panel, str(v), (110, 34 + i * 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, col, 1, cv2.LINE_AA)

    y = 34 + len(lines) * 30 + 16
    cv2.putText(panel, "events", (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (140, 140, 140), 1, cv2.LINE_AA)
    for i, e in enumerate(list(events)[-9:]):
        cv2.putText(panel, e, (12, y + 22 + i * 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(panel, "red lines = matrix boundary the edge test reads",
                (8, panel.shape[0] - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (120, 120, 200), 1, cv2.LINE_AA)
    cv2.putText(panel, "q quit  r reset  b board dots  c recalibrate  space pause  s save",
                (8, panel.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (140, 140, 140), 1, cv2.LINE_AA)
    # Inset: the garbage-counter strip at NATIVE resolution -- the actual
    # evidence behind a garbage reward, at the size the detector now sees it.
    pw, ph = 120, 300
    if raw is not None and sm is not None:
        strip = T.crop_region(raw, T.NUM_STRIP, sm.box)
        if strip.size:
            if sm.swap_rb:
                strip = strip[:, :, ::-1]
            strip = cv2.resize(np.ascontiguousarray(strip), (pw, ph),
                               interpolation=cv2.INTER_AREA)
            py = panel.shape[0] - ph - 8
            panel[py:py + ph, PANEL_W - pw - 8:PANEL_W - 8] = strip
            cv2.rectangle(panel, (PANEL_W - pw - 8, py), (PANEL_W - 8, py + ph),
                          (0, 165, 255), 1)
            cv2.putText(panel, "garbage strip (native res)",
                        (PANEL_W - pw - 150, py - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 165, 255), 1)
    return np.hstack([vis, panel])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--record", metavar="DIR", default=None,
                    help="save every Nth raw frame here for offline analysis")
    ap.add_argument("--record-every", type=int, default=6)
    ap.add_argument("--swap-rb", choices=("auto", "on", "off"), default="auto",
                    help="override red/blue channel correction")
    a = ap.parse_args()

    if a.list:
        list_devices()
        return 0

    masks = T.TetrisMasks()
    idx = a.device
    if idx is None:
        # Width alone cannot tell a capture card from a webcam: both honour a
        # 1920x1080 request, so a tie handed the pick to the lowest index --
        # the webcam -- and the monitor showed its black lens forever. Rank by
        # "is anything actually on screen" first, then by width.
        found = []
        for i in range(4):
            cap = open_capture(i, a.width, a.height)
            frames = []
            if cap.isOpened():
                for _ in range(10):
                    ok, fr = cap.read()
                    if ok and fr is not None:
                        frames.append(fr)
            cap.release()
            if not frames:
                continue
            lit = float(np.stack(frames).mean())
            w = frames[-1].shape[1]
            found.append((lit >= 8.0, w, lit, i))
            print(f"  device {i}: {frames[-1].shape[1]}x{frames[-1].shape[0]}"
                  f"  brightness {lit:.1f}" + ("" if lit >= 8.0 else "  (black)"),
                  flush=True)
        if not found:
            sys.exit("no capture device produced a frame")
        found.sort(reverse=True)
        idx = found[0][3]
        best = found[0][1]
        if not found[0][0]:
            print("  WARNING: every device looks black -- is the console on?", flush=True)
        print(f"auto-selected device {idx} ({best}px wide, "
              f"brightness {found[0][2]:.1f}).  override with --device N", flush=True)

    cap = open_capture(idx, a.width, a.height)
    if not cap.isOpened():
        sys.exit(f"could not open capture device {idx}")

    # Calibrate the letterbox crop from a bright frame before anything else:
    # a Cam Link inset the picture by ~2%, which shifts every ROI.
    sm = T.TetrisState(masks)
    votes = []
    for _ in range(160):
        ok, fr = cap.read()
        if ok and fr is not None and fr.mean() > 5:
            box, sw = sm.calibrate(fr)
            if box:
                votes.append(sw)
                if len(votes) >= 20:
                    break
        time.sleep(0.02)
    if votes:
        # Majority vote over frames: one frame is enough in principle, but a
        # vote costs nothing and removes any dependence on what was on screen.
        sm.swap_rb = sum(votes) * 2 > len(votes)
    if a.swap_rb != "auto":
        sm.swap_rb = (a.swap_rb == "on")
        print(f"swap_rb forced {sm.swap_rb} by --swap-rb", flush=True)
    if sm.box:
        y0, y1, x0, x1 = sm.box
        print(f"calibrated: content y {y0}..{y1} x {x0}..{x1} "
              f"(borders L{x0} R{a.width - x1} T{y0} B{a.height - y1});  "
              f"swap_rb={sm.swap_rb}"
              f"{'  (capture is RGB, correcting)' if sm.swap_rb else ''}",
              flush=True)
    else:
        print("NOT calibrated (frame too dark) -- press c when the picture "
              "is up", flush=True)
    events = deque(maxlen=40)
    # The counter and the banner are a few frames apart, not simultaneous, so a
    # missed counter can only be judged over a window -- see COUNTER_WINDOW_S.
    last_garbage_t = -1e9
    win = "tetris live detector  (q quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    prev_state, last, fps, paused, n = None, time.time(), 0.0, False, 0
    # Off by default: the dots are a diagnostic for the board decode, which no
    # longer feeds reward at all, and they obscure the board while playing.
    show_board = False                # 'b' toggles the grid overlay
    rec_n = 0
    # Cumulative reward: per episode (reset when a match starts) and for the
    # whole session, so a run can be compared against the numbers the agent
    # would actually be trained on.
    ep = {"total": 0.0, "tetris": 0, "garbage": 0, "died": 0}
    run = {"total": 0.0, "tetris": 0, "garbage": 0, "died": 0}
    if a.record:
        os.makedirs(a.record, exist_ok=True)
        print(f"recording every {a.record_every}th frame to {a.record}", flush=True)
    frame = None
    print("monitoring -- nothing is sent to the console. 'q' in the window to "
          "stop.\n", flush=True)

    while True:
        if not paused:
            ok, fr = cap.read()
            if not ok or fr is None:
                time.sleep(0.02)
                continue
            frame = fr
            # Keep trying to calibrate until it takes. Giving up after a fixed
            # window meant that if the screen happened to be dark at startup
            # (console asleep, a menu fade), the whole session ran with no
            # letterbox crop and no channel correction -- every detection
            # silently wrong, with only a line in the log to say so.
            if sm.box is None and fr.mean() > 5:
                if sm.calibrate(fr)[0]:
                    print(f"calibrated late: box {sm.box}  swap_rb {sm.swap_rb}",
                          flush=True)
            # Same crop the state machine applies, so the overlay shows exactly
            # what the detector scored rather than a differently-framed image.
            prep = T.prepare(frame, sm.box, sm.swap_rb)
            # Record ONLY while a match is in progress. An 8-hour session
            # wrote 154,429 strips through the idle menus, and a directory
            # that size makes every further write slower -- which is what
            # froze the display, not the detectors (6 ms/frame, 170 fps).
            if a.record and sm.state == T.PLAYING:
                rec_n += 1
                if rec_n % a.record_every == 0:
                    # Also the NATIVE-resolution garbage strip (~80 KB): the
                    # hi-res detector cannot be validated against downscaled
                    # frames, and every recording so far was downscaled.
                    st = T.crop_region(frame, T.NUM_STRIP, sm.box)
                    if st.size:
                        if sm.swap_rb:
                            st = st[:, :, ::-1]
                        cv2.imwrite(os.path.join(a.record, f"S{rec_n:06d}.png"),
                                    np.ascontiguousarray(st))
                    # Save the PREPARED 90x160 frame, not the raw 1080p one:
                    # 2.7 MB/frame at 45 fps is 120 MB/s of PNG encoding, which
                    # throttles the capture loop and drops the very frames we
                    # are trying to inspect. Already cropped and colour-
                    # corrected, so analysis must NOT re-apply either.
                    cv2.imwrite(os.path.join(a.record, f"f{rec_n:06d}.png"), prep)
            out = sm.step(frame)
            n += 1
            now = time.time()
            if now - last >= 0.5:
                fps = n / (now - last)
                n, last = 0, now
            stamp = time.strftime("%H:%M:%S")
            if out["state"] == T.PLAYING and prev_state != T.PLAYING:
                ep = {"total": 0.0, "tetris": 0, "garbage": 0, "died": 0}
            if out["reward"]:
                ep["total"] += out["reward"]
                run["total"] += out["reward"]
            # A frame can now pay a Tetris AND its garbage counter, arriving
            # as "tetris+garbage" -- count each part, not the compound string.
            parts = (out["event"] or "").split("+")
            for part in parts:
                if part in ("tetris", "garbage", "died"):
                    ep[part] += 1
                    run[part] += 1
            out.update({f"ep_{k}": v for k, v in ep.items()})
            out.update({f"run_{k}": v for k, v in run.items()})
            # Save the strip for EVERY frame with a glyph candidate, not only
            # rewarded ones: the template set is the bottleneck now, and misses
            # leave no evidence otherwise.
            if a.record and sm.state == T.PLAYING and rec_n % 2 == 0:
                st2 = T.crop_region(frame, T.NUM_STRIP, sm.box)
                if st2.size and masks.clear_numeral_hires(frame, sm.box, sm.swap_rb) is None:
                    g = T.m_glyph_local(st2[:, :, ::-1] if sm.swap_rb else st2)
                    if g.sum() > 400:
                        cv2.imwrite(os.path.join(a.record, f"C{rec_n:06d}.png"),
                                    np.ascontiguousarray(st2[:, :, ::-1] if sm.swap_rb else st2))
            if "garbage" in parts and a.record:
                st = T.crop_region(frame, T.NUM_STRIP, sm.box)
                if st.size:
                    if sm.swap_rb:
                        st = st[:, :, ::-1]
                    cv2.imwrite(os.path.join(
                        a.record, f"EVENT_{rec_n:06d}.png"),
                        np.ascontiguousarray(st))
            if out["event"]:
                # Every Tetris sends garbage, so a Tetris with no counter
                # anywhere in COUNTER_WINDOW_S means the numeral was MISSED.
                # Same-frame coincidence is the wrong test: measured on real
                # play the counter lands 0..1s BEFORE the banner, every time.
                if "garbage" in parts:
                    last_garbage_t = now
                miss = ("   <-- counter missed?"
                        if parts == ["tetris"]
                        and now - last_garbage_t > COUNTER_WINDOW_S else "")
                line = (f"{stamp} {out['event']} {out['reward']:+.2f}"
                        f"  ep={ep['total']:+.2f}"
                        + (f"  sent={out.get('sent')} lines"
                           if "garbage" in parts else "") + miss)
                events.append(line)
                print("  " + line, flush=True)
            out["swap_rb"] = sm.swap_rb
            out["calib"] = "ok" if sm.box else "WAITING (screen dark)"
            if out["state"] != prev_state:
                line = f"{stamp} -> {out['state']}"
                events.append(line)
                print("  " + line, flush=True)
                prev_state = out["state"]
        else:
            prep = T.prepare(frame, sm.box, sm.swap_rb)
            e, d = masks.in_game_features(prep)
            out = {"state": sm.state, "edge": e, "dark": d,
                   "cells": masks.board_cells(prep), "drop": 0.0,
                   "reward": 0.0, "event": None, "swap_rb": sm.swap_rb,
                   "calib": "ok" if sm.box else "WAITING (screen dark)"}
            out.update({f"ep_{k}": v for k, v in ep.items()})
            out.update({f"run_{k}": v for k, v in run.items()})

        cv2.imshow(win, draw(prep, masks, out, events, fps, paused, show_board,
                             frame, sm))
        k = cv2.waitKeyEx(1) & 0xFF
        if k in (ord("q"), 27):
            break
        elif k == ord("r"):
            sm = T.TetrisState(masks, content_box=sm.box, swap_rb=sm.swap_rb)
            prev_state = None
            ep = {"total": 0.0, "tetris": 0, "garbage": 0, "died": 0}
            print("  state machine reset", flush=True)
        elif k == ord("c") and frame is not None:
            if sm.calibrate(frame)[0]:
                print(f"  recalibrated: box {sm.box}  swap_rb {sm.swap_rb}",
                      flush=True)
            else:
                print("  frame too dark to calibrate", flush=True)
        elif k == ord("b"):
            show_board = not show_board
            print("  board overlay %s" % ("on" if show_board else "off"), flush=True)
        elif k == ord(" "):
            paused = not paused
        elif k == ord("s") and frame is not None:
            fn = f"live_{int(time.time())}.png"
            cv2.imwrite(fn, frame)
            print("  saved " + fn, flush=True)

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
