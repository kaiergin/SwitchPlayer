#!/usr/bin/env python3
"""
elgato_view.py - Live preview + low-latency audio passthrough for an Elgato
(or any UVC/USB) capture card.

Requires:  pip install opencv-python sounddevice numpy

Usage:
    python elgato_view.py --list                 # show video indices + audio devices
    python elgato_view.py                        # auto-detect, 1080p60
    python elgato_view.py --video 1 --audio 3    # explicit devices
    python elgato_view.py --width 1280 --height 720 --fps 60

Keys:  q/ESC quit   f fullscreen   m mute   +/- volume
"""

import argparse
import ctypes
import os
import platform
import sys
import threading
import time

import cv2
import numpy as np
import sounddevice as sd

SYSTEM = platform.system()

class SwitchControl:
    """Drive the Switch over the Arduino serial link from the PC keyboard.

    cv2.waitKey cannot do this job: it returns one key at a time, only on the
    OS auto-repeat schedule, and never reports release. A held D-pad would
    stutter after the repeat delay and diagonals would be impossible.
    GetAsyncKeyState reads the true up/down state of every key each frame, so
    holds and chords behave like a real controller.

    Input is gated on this window being focused, so alt-tabbing away stops
    sending rather than typing into whatever you switched to.
    """

    KEEPALIVE_S = 0.5
    WINDOW_MARK = "Elgato Capture"

    def __init__(self, port):
        import switch_keymap
        if port:
            os.environ["SWITCH_PORT"] = port
        import controller
        self.c = controller
        self.km = switch_keymap
        self.map = switch_keymap.by_vk(controller)
        # Arrow keys drive the right stick; they are resolved together rather
        # than summed, because stick angle and intensity share adjacent bit
        # fields in the command word.
        self.rstick_vk = {nm: vk for nm, _const, _pg, vk in switch_keymap.RSTICK}
        self.user32 = ctypes.windll.user32
        self.user32.GetAsyncKeyState.restype = ctypes.c_short
        self.last_cmd, self.last_sent, self.sent_n = None, 0.0, 0
        self.held, self.blocked = [], False
        self.synced = self.c.sync()
        print("[control] %s on %s" % (
            "synced" if self.synced else "SYNC FAILED",
            os.environ.get("SWITCH_PORT", "COM5")))

    def _focused(self):
        buf = ctypes.create_unicode_buffer(512)
        self.user32.GetWindowTextW(self.user32.GetForegroundWindow(), buf, 512)
        return self.WINDOW_MARK in buf.value

    def poll(self):
        self.blocked = not self._focused()
        cmd, held = 0, []
        if not self.blocked:
            for vk, (bits, key, name) in self.map.items():
                if self.user32.GetAsyncKeyState(vk) & 0x8000:
                    cmd += bits
                    held.append(name)
            down = {nm: bool(self.user32.GetAsyncKeyState(vk) & 0x8000)
                    for nm, vk in self.rstick_vk.items()}
            stick = self.km.rstick_bits(self.c, down["Up"], down["Down"],
                                        down["Left"], down["Right"])
            if stick:
                cmd += stick
                held.append(self.km.rstick_label(down["Up"], down["Down"],
                                                 down["Left"], down["Right"]))
        now = time.time()
        # Send only on change: the Switch runs its own auto-repeat from a held
        # direction, so retransmitting every frame adds latency for nothing.
        if cmd != self.last_cmd or now - self.last_sent > self.KEEPALIVE_S:
            self.c.send_cmd(cmd)
            self.last_cmd, self.last_sent = cmd, now
            self.sent_n += 1
        self.held = held

    def release(self):
        try:
            self.c.send_cmd(self.c.NO_INPUT)
        except Exception:
            pass


def draw_control_overlay(frame, ctl):
    """Held buttons and link state on a dimmed strip along the bottom."""
    h, w = frame.shape[:2]
    pad, bar = 12, 34
    y0 = h - bar - pad
    region = frame[y0:y0 + bar, pad:w - pad]
    cv2.addWeighted(region, 0.35, np.zeros_like(region), 0.0, 0.0, region)
    if not ctl.synced:
        txt, col = "CONTROL: sync failed", (60, 60, 255)
    elif ctl.blocked:
        txt, col = "CONTROL: paused (window not focused)", (0, 190, 255)
    else:
        txt = "CONTROL: " + (" + ".join(ctl.held) if ctl.held else "idle")
        col = (120, 255, 120) if ctl.held else (200, 200, 200)
    cv2.putText(frame, txt, (pad + 10, y0 + 23), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, col, 2, cv2.LINE_AA)



def video_backend():
    # The backend matters: on Windows the default MSMF backend refuses to set
    # MJPG on most Elgato cards and you get 5-10 fps. DSHOW works.
    if SYSTEM == "Windows":
        return cv2.CAP_DSHOW
    if SYSTEM == "Darwin":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_V4L2


def list_devices(max_probe=8):
    print("=== Audio devices ===")
    print(sd.query_devices())
    print("\n=== Video indices (probing, this takes a few seconds) ===")
    be = video_backend()
    for i in range(max_probe):
        cap = cv2.VideoCapture(i, be)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok:
                h, w = frame.shape[:2]
                print(f"  index {i}: OK, default frame {w}x{h}")
            else:
                print(f"  index {i}: opened but no frame")
        cap.release()
    if SYSTEM == "Linux":
        print("\nTip: `v4l2-ctl --list-devices` gives you real names on Linux.")


def open_video(index, width, height, fps, fourcc):
    cap = cv2.VideoCapture(index, video_backend())
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video index {index}")

    # Order matters: set the pixel format BEFORE resolution/fps. Elgato cards
    # only expose high-res/high-fps modes over MJPG (or NV12); raw YUY2 is
    # bandwidth-limited and will silently clamp you to ~10 fps at 1080p.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # ignored by some backends, harmless

    ok, _ = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(
            f"Opened index {index} but got no frames. Wrong device, or the card "
            "is claimed by another app (OBS / 4K Capture Utility)."
        )
    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    af = cap.get(cv2.CAP_PROP_FPS)
    print(f"[video] index {index}: {aw}x{ah} @ {af:.0f} fps ({fourcc})")
    return cap


# Elgato exposes its audio under a few different names depending on model and
# host API: HD60 X / 4K X usually show up as "Digital Audio Interface (...)",
# Cam Link as "Cam Link 4K", older cards as "Game Capture HD60 S+". WASAPI
# tends to use the bare endpoint name, MME truncates to 31 characters.
ELGATO_HINTS = [
    "game capture", "cam link", "elgato", "digital audio interface",
    "hd60", "4k capture", "capture",
]


def find_device(hint, kind="input", hostapi=None):
    """Find an audio device index whose name contains `hint`, optionally
    restricted to one host API."""
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    for i, d in enumerate(sd.query_devices()):
        if d[key] > 0 and hint.lower() in d["name"].lower():
            if hostapi is None or d["hostapi"] == hostapi:
                return i
    return None


def auto_find_input(hint=None):
    """Try each known Elgato naming pattern; report what matched, and dump the
    available inputs if nothing does."""
    hints = [hint] if hint else ELGATO_HINTS
    for h in hints:
        idx = find_device(h, "input")
        if idx is not None:
            print(f"[audio] matched input on '{h}': {sd.query_devices(idx)['name']}")
            return idx

    print(f"[audio] nothing matched {hints}. Input devices on this machine:")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            api = sd.query_hostapis(d["hostapi"])["name"]
            print(f"   --audio {i:<3} {d['name']}  [{api}, {d['max_input_channels']} ch]")
    print("Pick the one that is your capture card and pass it with --audio N.")
    return None


def default_output(prefer="wasapi"):
    """The playback device Windows is currently set to use.

    PortAudio's own default (sd.default.device[1]) is the MME default, which is
    the right *device* but a legacy path with ~100ms latency and rigid format
    rules. Each host API also advertises its own default_output_device, and on
    Windows they all track the same 'Default Device' you set in the sound
    flyout -- so asking WASAPI for its default gets the same speakers on a
    better path. Falls back cleanly on non-Windows or if WASAPI is absent.
    """
    if SYSTEM == "Windows" and prefer:
        for i, api in enumerate(sd.query_hostapis()):
            if prefer.lower() in api["name"].lower():
                idx = api["default_output_device"]
                if idx is not None and idx >= 0:
                    return idx, i
    idx = sd.default.device[1]
    if idx is None or idx < 0:
        raise RuntimeError("Windows reports no default playback device.")
    return idx, sd.query_devices(idx)["hostapi"]


def wasapi_settings(dev):
    """auto_convert lets WASAPI shared mode resample/remix for us, so a card at
    48k stereo can feed speakers at 44.1k without us doing any DSP."""
    try:
        if "wasapi" in sd.query_hostapis(sd.query_devices(dev)["hostapi"])["name"].lower():
            return sd.WasapiSettings(auto_convert=True)
    except Exception:
        pass
    return None


def probe_audio(in_dev, out_dev, rate):
    """Print exactly which channel counts / rates each end will accept."""
    if in_dev is None:
        print("No input device given or auto-matched. Run --list first.")
        return
    if out_dev is None:
        out_dev = default_output()[0]
    for label, dev, checker, key in (
        ("INPUT ", in_dev, sd.check_input_settings, "max_input_channels"),
        ("OUTPUT", out_dev, sd.check_output_settings, "max_output_channels"),
    ):
        d = sd.query_devices(dev)
        api = sd.query_hostapis(d["hostapi"])["name"]
        print(f"\n{label} [{dev}] {d['name']}  ({api})")
        print(f"  reports max {d[key]} ch, default {d['default_samplerate']:.0f} Hz")
        for ch in (1, 2, 6, 8):
            for r in dict.fromkeys([rate, 44100, 48000]):
                try:
                    checker(device=dev, channels=ch, samplerate=r, dtype="float32")
                    print(f"  OK: {ch} ch @ {r} Hz")
                except Exception:
                    pass


class FrameReader(threading.Thread):
    """Drains the capture buffer in its own thread and keeps only the newest
    frame, so the display never falls behind the card."""

    def __init__(self, cap):
        super().__init__(daemon=True)
        self.cap = cap
        self.lock = threading.Lock()
        self.frame = None
        self.running = True
        self.count = 0

    def run(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self.lock:
                self.frame = frame
                self.count += 1

    def latest(self):
        with self.lock:
            return None if self.frame is None else self.frame

    def stop(self):
        self.running = False
        self.join(timeout=1.0)


def resolve_audio(in_dev, out_dev, want_rate, in_ch_req, out_ch_req, prefer_api="wasapi"):
    """Work out real device indices, channel counts and a sample rate that both
    ends actually accept. Passing out_dev=None means 'whatever Windows is
    currently using for playback'."""
    if out_dev is None:
        out_dev, out_api = default_output(prefer_api)
        d = sd.query_devices(out_dev)
        print(f"[audio] Windows default playback: {d['name']} "
              f"({sd.query_hostapis(out_api)['name']})")
    if in_dev is None:
        in_dev = sd.default.device[0]
    if in_dev is None or in_dev < 0:
        raise RuntimeError("No audio input device selected (pass --audio N).")
    if out_dev is None or out_dev < 0:
        raise RuntimeError("No default audio output device (pass --output N).")

    # If the card also exists on the output's host API, use that copy: same-API
    # pairs open as one duplex stream instead of two clocked-apart ones.
    out_api = sd.query_devices(out_dev)["hostapi"]
    if sd.query_devices(in_dev)["hostapi"] != out_api:
        twin = find_device(sd.query_devices(in_dev)["name"].split("(")[0].strip(),
                           "input", hostapi=out_api)
        if twin is not None:
            in_dev = twin

    ind, outd = sd.query_devices(in_dev), sd.query_devices(out_dev)
    in_max, out_max = ind["max_input_channels"], outd["max_output_channels"]
    if in_max < 1:
        raise RuntimeError(f"Device {in_dev} '{ind['name']}' has no input channels.")
    if out_max < 1:
        raise RuntimeError(f"Device {out_dev} '{outd['name']}' has no output channels.")

    in_ch = min(in_ch_req or 2, in_max)
    out_ch = min(out_ch_req or 2, out_max)
    in_extra, out_extra = wasapi_settings(in_dev), wasapi_settings(out_dev)

    # Pick a rate both ends accept: requested, then each device's default.
    candidates = [want_rate, int(ind["default_samplerate"]), int(outd["default_samplerate"])]
    rate = None
    for r in dict.fromkeys(candidates):
        try:
            sd.check_input_settings(device=in_dev, channels=in_ch, samplerate=r,
                                    dtype="float32", extra_settings=in_extra)
            sd.check_output_settings(device=out_dev, channels=out_ch, samplerate=r,
                                     dtype="float32", extra_settings=out_extra)
            rate = r
            break
        except Exception:
            continue
    if rate is None:
        raise RuntimeError(
            f"No common sample rate. Input wants {ind['default_samplerate']:.0f} Hz, "
            f"output wants {outd['default_samplerate']:.0f} Hz. Match them in "
            "Windows Sound > Advanced (set both to 48000 Hz, 2 channel)."
        )

    same_api = (in_dev == out_dev)
    print(f"[audio] in : [{in_dev}] {ind['name']} ({in_ch}/{in_max} ch)")
    print(f"[audio] out: [{out_dev}] {outd['name']} ({out_ch}/{out_max} ch)")
    print(f"[audio] {rate} Hz, {'duplex' if same_api else 'split (separate devices)'}")
    return in_dev, out_dev, in_ch, out_ch, rate, same_api, (in_extra, out_extra)


def _mix(indata, outdata, gain):
    """Map whatever channel layout came in onto whatever the output wants."""
    ic, oc = indata.shape[1], outdata.shape[1]
    if ic == oc:
        outdata[:] = indata * gain
    elif ic == 1:
        outdata[:] = indata[:, [0] * oc] * gain          # mono -> N
    elif oc == 1:
        outdata[:, 0] = indata.mean(axis=1) * gain       # N -> mono downmix
    else:
        n = min(ic, oc)
        outdata[:] = 0.0
        outdata[:, :n] = indata[:, :n] * gain


class Audio:
    """Duplex when both devices share a host API, otherwise two streams joined
    by a small ring buffer."""

    def __init__(self, in_dev, out_dev, in_ch, out_ch, rate, blocksize, same_api,
                 extra=(None, None), mirror_dev=None):
        self.gain = 1.0
        self.muted = False
        self.same_api = same_api
        self.mirror_buf = None
        in_extra, out_extra = extra
        common = dict(samplerate=rate, blocksize=blocksize, dtype="float32", latency="low")

        if same_api:
            self.stream = sd.Stream(
                device=(in_dev, out_dev), channels=(in_ch, out_ch),
                extra_settings=(in_extra, out_extra) if (in_extra or out_extra) else None,
                callback=self._duplex_cb, **common)
            self.streams = [self.stream]
        else:
            from collections import deque
            self.buf = deque(maxlen=8)  # ~8 blocks of slack
            self.out_ch = out_ch
            self.istream = sd.InputStream(
                device=in_dev, channels=in_ch, extra_settings=in_extra,
                callback=self._in_cb, **common)
            self.ostream = sd.OutputStream(
                device=out_dev, channels=out_ch, extra_settings=out_extra,
                callback=self._out_cb, **common)
            self.streams = [self.ostream, self.istream]

        # Optional second destination (e.g. a virtual cable feeding Discord).
        # Runs at unity gain and ignores mute, so muting your own headset does
        # not silence what everyone else hears.
        if mirror_dev is not None:
            from collections import deque
            m = sd.query_devices(mirror_dev)
            mch = min(2, m["max_output_channels"])
            self.mirror_buf = deque(maxlen=8)
            self.mstream = sd.OutputStream(
                device=mirror_dev, channels=mch, extra_settings=wasapi_settings(mirror_dev),
                callback=self._mirror_cb, **common)
            self.streams.append(self.mstream)
            print(f"[audio] mirroring to: [{mirror_dev}] {m['name']} ({mch} ch)")

    def _mirror_cb(self, outdata, frames, t, status):
        if self.mirror_buf:
            block = self.mirror_buf.popleft()
            if block.shape[0] == frames:
                _mix(block, outdata, 1.0)
                return
        outdata[:] = 0.0

    def _g(self):
        return 0.0 if self.muted else self.gain

    def _duplex_cb(self, indata, outdata, frames, t, status):
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        if self.mirror_buf is not None:
            self.mirror_buf.append(indata.copy())
        _mix(indata, outdata, self._g())

    def _in_cb(self, indata, frames, t, status):
        self.buf.append(indata.copy())
        if self.mirror_buf is not None:
            self.mirror_buf.append(indata.copy())

    def _out_cb(self, outdata, frames, t, status):
        if self.buf:
            block = self.buf.popleft()
            if block.shape[0] == frames:
                _mix(block, outdata, self._g())
                return
        outdata[:] = 0.0  # underrun -> silence rather than garbage

    def __enter__(self):
        for s in self.streams:
            s.start()
        return self

    def __exit__(self, *exc):
        for s in self.streams:
            s.stop()
            s.close()


def main():
    p = argparse.ArgumentParser(description="Elgato capture card preview + audio")
    p.add_argument("--list", action="store_true", help="list devices and exit")
    p.add_argument("--video", type=int, default=None, help="OpenCV video index")
    p.add_argument("--audio", type=int, default=None, help="sounddevice input index")
    p.add_argument("--output", type=int, default=None,
                   help="sounddevice output index (default: whatever Windows is using)")
    p.add_argument("--api", default="wasapi",
                   help="host API whose default output to use: wasapi, mme, "
                        "directsound, or '' for PortAudio's own default")
    p.add_argument("--hint", default=None,
                   help="name substring to match the capture card input "
                        "(default: try several known Elgato names)")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--fourcc", default="MJPG", choices=["MJPG", "NV12", "YUY2"])
    p.add_argument("--rate", type=int, default=48000, help="audio sample rate")
    p.add_argument("--blocksize", type=int, default=256, help="audio block (latency)")
    p.add_argument("--in-channels", type=int, default=None, help="force input channel count")
    p.add_argument("--out-channels", type=int, default=None, help="force output channel count")
    p.add_argument("--probe-audio", action="store_true", help="test channel counts and exit")
    p.add_argument("--mirror", type=int, default=None,
                   help="also send capture audio to this output index "
                        "(e.g. a VB-Cable input, for Discord to pick up as a mic)")
    p.add_argument("--no-audio", action="store_true")
    p.add_argument("--port", default=None,
                   help="serial port of the Arduino adapter, e.g. COM3")
    p.add_argument("--no-control", action="store_true",
                   help="preview only; do not drive the Switch")
    args = p.parse_args()

    if args.list:
        list_devices()
        return

    if args.probe_audio:
        probe_audio(args.audio if args.audio is not None else auto_find_input(args.hint),
                    args.output, args.rate)
        return

    # --- video ---
    if args.video is not None:
        cap = open_video(args.video, args.width, args.height, args.fps, args.fourcc)
    else:
        cap = None
        for i in range(6):
            try:
                cap = open_video(i, args.width, args.height, args.fps, args.fourcc)
                break
            except RuntimeError:
                continue
        if cap is None:
            sys.exit("No usable video device found. Run with --list.")

    # --- audio ---
    audio_ctx = None
    if not args.no_audio:
        in_dev = args.audio if args.audio is not None else auto_find_input(args.hint)
        if in_dev is None:
            print("[audio] continuing with video only.")
        else:
            try:
                i, o, ic, oc, rate, same, extra = resolve_audio(
                    in_dev, args.output, args.rate, args.in_channels,
                    args.out_channels, args.api)
                audio_ctx = Audio(i, o, ic, oc, rate, args.blocksize, same, extra,
                                  mirror_dev=args.mirror)
            except Exception as e:
                print(f"[audio] failed to open stream: {e}", file=sys.stderr)
                print("[audio] try: py elgato_view.py --probe-audio", file=sys.stderr)

    # --- control ---
    ctl = None
    if not args.no_control:
        if SYSTEM != "Windows":
            print("[control] keyboard control is Windows-only.")
        else:
            try:
                ctl = SwitchControl(args.port)
            except Exception as e:
                print("[control] disabled: %s" % e, file=sys.stderr)
                print("[control] --port COMn to pick the adapter, "
                      "--no-control to silence this.", file=sys.stderr)

    reader = FrameReader(cap)
    reader.start()

    # 'm' is the B button now, so mute moved to 'k'. Keep the string
    # "Elgato Capture" in the title: SwitchControl matches on it to tell
    # whether this window has focus.
    win = "Elgato Capture  (q quit | f fullscreen | k mute | +/- volume)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, args.width // 2, args.height // 2)
    fullscreen = False

    ctx = audio_ctx if audio_ctx else _Null()
    with ctx:
        last, last_count, fps_disp = time.time(), 0, 0.0
        try:
            while True:
                frame = reader.latest()
                if frame is not None:
                    if ctl is not None:
                        ctl.poll()
                        frame = frame.copy()   # do not annotate the shared buffer
                        draw_control_overlay(frame, ctl)
                    cv2.imshow(win, frame)

                now = time.time()
                if now - last >= 1.0:
                    fps_disp = (reader.count - last_count) / (now - last)
                    last, last_count = now, reader.count
                    vol = "mute" if (audio_ctx and audio_ctx.muted) else (
                        f"{audio_ctx.gain:.1f}x" if audio_ctx else "off")
                    ctl_s = ("off" if ctl is None else
                             ("sync-fail" if not ctl.synced
                              else "%d pkts" % ctl.sent_n))
                    print(f"\r[{fps_disp:5.1f} fps]  audio {vol}  control {ctl_s}   ",
                          end="", flush=True)

                k = cv2.waitKey(1) & 0xFF
                if k in (ord("q"), 27):
                    break
                elif k == ord("f"):
                    fullscreen = not fullscreen
                    cv2.setWindowProperty(
                        win, cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
                elif k == ord("k") and audio_ctx:
                    audio_ctx.muted = not audio_ctx.muted
                elif k in (ord("+"), ord("=")) and audio_ctx:
                    audio_ctx.gain = min(4.0, audio_ctx.gain + 0.1)
                elif k == ord("-") and audio_ctx:
                    audio_ctx.gain = max(0.0, audio_ctx.gain - 0.1)

                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    break
        except KeyboardInterrupt:
            pass

    print()
    if ctl is not None:
        ctl.release()          # never leave a button stuck down
        print("[control] released all inputs; %d packets sent" % ctl.sent_n)
    reader.stop()
    cap.release()
    cv2.destroyAllWindows()


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    main()
