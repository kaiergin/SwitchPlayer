#!/usr/bin/env python3
"""Record your own Switch play: frames, actions, rewards, timing.

    python record_play.py --out sessions/2026-08-31_a --port COM3 --video 1

Plays through the keyboard exactly like tetris_keyboard.py, but every step is
logged: the registered reference frame, the raw controller command, the reward
and state from the live detector, and a monotonic timestamp.

Steps are taken at STEP_PERIOD, the same rate the agent acts at, so a recording
is directly comparable to an agent trajectory.

Keys come from the shared mapping in switch_keymap. Esc or Q stops and closes
the session cleanly. The window must be focused for input to be sent.
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

import session_recorder as SR
import switch_control
import tetris_masks as T

WINDOW = "Switch Recorder"
STEP_PERIOD = 1.0 / 30      # matches tetris.py; 30 Hz


def open_capture(index, width, height):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def pick_device(width, height):
    """Rank capture devices by whether anything is on screen, then by width.

    Width alone cannot separate a capture card from a webcam: both honour a
    1920x1080 request, and a tie hands the pick to index 0, which is the
    webcam. That silently fed a black lens to the monitor for a whole session.
    """
    found = []
    for i in range(4):
        cap = open_capture(i, width, height)
        if cap is None:
            continue
        frames = []
        for _ in range(8):
            ok, f = cap.read()
            if ok and f is not None:
                frames.append(f)
        cap.release()
        if not frames:
            continue
        lit = float(np.stack(frames).mean())
        note = "" if lit >= 8.0 else "  (black)"
        print("  device %d: %dx%d  brightness %.1f%s"
              % (i, frames[-1].shape[1], frames[-1].shape[0], lit, note))
        found.append((lit >= 8.0, frames[-1].shape[1], lit, i))
    if not found:
        return None
    found.sort(reverse=True)
    if not found[0][0]:
        print("  WARNING: every device looks black -- is the console on?")
    return found[0][3]


def hud(view, out, ctl, kept, dropped, fps):
    held = " + ".join(ctl.held) if ctl.held else (
        "PAUSED (unfocused)" if ctl.blocked else "idle")
    rows = [
        ("state", str(out.get("state")), (200, 200, 200)),
        ("reward", "%+.2f" % out.get("reward", 0.0),
         (120, 255, 120) if out.get("reward") else (150, 150, 150)),
        ("event", str(out.get("event") or "-"), (120, 255, 255)),
        ("held", held, (0, 190, 255) if ctl.blocked else (255, 255, 255)),
        ("recorded", "%d steps  (%d skipped)" % (kept, dropped), (255, 255, 255)),
        ("link", "synced" if ctl.synced else "SYNC FAILED",
         (120, 255, 120) if ctl.synced else (60, 60, 255)),
        ("fps", "%.1f" % fps, (170, 170, 170)),
    ]
    panel = np.zeros((26 * len(rows) + 16, 340, 3), np.uint8)
    for i, (k, v, col) in enumerate(rows):
        y = 24 + i * 26
        cv2.putText(panel, k, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (140, 140, 150), 1)
        cv2.putText(panel, v, (115, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    h = min(panel.shape[0], view.shape[0])
    w = min(panel.shape[1], view.shape[1])
    view[:h, :w] = panel[:h, :w]
    return view


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="session directory")
    ap.add_argument("--port", default=None, help="serial port, e.g. COM3")
    ap.add_argument("--video", type=int, default=None)
    ap.add_argument("--note", default="", help="free text, stored in meta.json")
    ap.add_argument("--source", default="human")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--step-period", type=float, default=STEP_PERIOD)
    ap.add_argument("--all-states", action="store_true",
                    help="record menus and dead time too (default: only "
                         "pregame and playing, so idle hours cost nothing)")
    a = ap.parse_args()

    if os.path.exists(a.out) and os.listdir(a.out):
        sys.exit("refusing to write into a non-empty directory: %s" % a.out)

    idx = a.video
    if idx is None:
        print("probing capture devices:")
        idx = pick_device(a.width, a.height)
        if idx is None:
            sys.exit("no capture device produced a frame")
        print("using device %d" % idx)
    cap = open_capture(idx, a.width, a.height)
    ok, frame = (cap.read() if cap else (False, None))
    if not ok or frame is None:
        sys.exit("device %s gave no frames" % idx)

    masks = T.TetrisMasks()
    sm = T.TetrisState(masks)
    try:
        ctl = switch_control.SwitchControl(a.port, WINDOW)
    except Exception as e:
        sys.exit("could not open the controller link: %s\n"
                 "  --port COMn to choose the adapter" % e)
    print("[control] %s" % ("synced" if ctl.synced else "SYNC FAILED"))

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, a.width // 2, a.height // 2)

    rec = SR.SessionRecorder(a.out, source=a.source, note=a.note,
                             timing={"step_period": a.step_period,
                                     "capture_fps_request": 60,
                                     "control": "send-on-change + keepalive",
                                     "key_latch": True})
    kept = dropped = 0
    last, n_frames, fps = time.time(), 0, 0.0
    calib_warned = False
    try:
        while True:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            if sm.box is None:
                if sm.calibrate(frame)[0] is None:
                    if not calib_warned:
                        print("waiting for a bright frame to calibrate...",
                              flush=True)
                        calib_warned = True
                else:
                    print("calibrated: box %s  swap_rb %s"
                          % (sm.box, sm.swap_rb), flush=True)

            cmd = ctl.poll()
            out = sm.step(frame)
            n_frames += 1

            live = out["state"] in (T.PREGAME, T.PLAYING)
            if sm.box is not None and (a.all_states or live):
                ref = T.prepare(frame, sm.box, sm.swap_rb)
                i = rec.add(time.perf_counter(), cmd, ref,
                            reward=out["reward"], event=out["event"],
                            state=out["state"], action=ctl.action_index(cmd))
                kept += 1
                # Native-resolution numeral crop only when a glyph is present.
                # Reward is read at native resolution, but storing every strip
                # would cost roughly 26 GB/hour. Candidates are rare.
                strip = T.crop_region(frame, T.NUM_STRIP, sm.box)
                if strip.size:
                    s = strip[:, :, ::-1] if sm.swap_rb else strip
                    if T.m_glyph_local(s).sum() > 400:
                        rec.add_strip(i, s)
            else:
                dropped += 1

            now = time.time()
            if now - last >= 0.5:
                fps, n_frames, last = n_frames / (now - last), 0, now

            view = cv2.resize(frame, (a.width // 2, a.height // 2))
            cv2.imshow(WINDOW, hud(view, out, ctl, kept, dropped, fps))
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break

            spare = a.step_period - (time.perf_counter() - t0)
            if spare > 0:
                time.sleep(spare)
    except KeyboardInterrupt:
        pass
    finally:
        ctl.release()
        steps = rec.close()
        cap.release()
        cv2.destroyAllWindows()
        size = sum(os.path.getsize(os.path.join(a.out, f))
                   for f in os.listdir(a.out) if f.endswith(".npz"))
        strips = len(os.listdir(os.path.join(a.out, "strips")))
        print("\nrecorded %d steps (%d skipped), %d numeral strips"
              % (steps, dropped, strips))
        if steps:
            per = size / steps
            print("%.1f MB  ->  %.0f bytes/step,  %.2f GB/hour at %.0f Hz"
                  % (size / 1e6, per, per * 3600 / a.step_period / 1e9,
                     1.0 / a.step_period))
        print("session: %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
