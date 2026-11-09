#!/usr/bin/env python3
"""Keyboard -> Switch control, shared by the viewer and the recorder.

Key state comes from Win32 GetAsyncKeyState rather than cv2.waitKey, which
returns one key at a time on the OS auto-repeat schedule and never reports
release -- a held D-pad would stutter and diagonals would be impossible.

Input is gated on a chosen window being focused, so alt-tabbing stops sending
instead of typing into whatever you switched to.
"""

import ctypes
import os
import time

import switch_keymap


class SwitchControl:

    KEEPALIVE_S = 0.5

    def __init__(self, port=None, window_mark="Switch"):
        if port:
            os.environ["SWITCH_PORT"] = port
        import controller
        self.c = controller
        self.km = switch_keymap
        self.map = switch_keymap.by_vk(controller)
        # Arrows drive the right stick and are resolved together, never summed:
        # stick angle and intensity occupy adjacent bit fields, so adding two
        # direction constants carries into the angle (U+R -> 91 deg, not 45).
        self.rstick_vk = {nm: vk for nm, _c, _pg, vk in switch_keymap.RSTICK}
        self.window_mark = window_mark
        self.user32 = ctypes.windll.user32
        self.user32.GetAsyncKeyState.restype = ctypes.c_short
        self.last_cmd, self.last_sent, self.sent_n = None, 0.0, 0
        self.held, self.blocked, self.command = [], False, 0
        self.synced = self.c.sync()          # sync() blocks; it must
        # From here on the ACK is collected on the following send instead of
        # stalling this one: 8.00 -> 4.12 ms median per packet.
        self.c.ACK_NONBLOCKING = True

    def _focused(self):
        buf = ctypes.create_unicode_buffer(512)
        self.user32.GetWindowTextW(self.user32.GetForegroundWindow(), buf, 512)
        return self.window_mark in buf.value

    def _down(self, vk, latch=True):
        """True if the key is down now, or WAS pressed since the last poll.

        Bit 0x8000 is 'currently down'; bit 0x0001 is 'pressed since the
        previous call for this key' and is cleared on read. Without the latch a
        tap shorter than the poll interval falls between samples and is never
        seen -- at a 20 Hz recording rate that silently drops any press under
        50 ms, which is a large share of human Tetris input. The latch is what
        makes the sampler faster than its own step rate.
        """
        v = self.user32.GetAsyncKeyState(vk)
        return bool(v & 0x8000) or (latch and bool(v & 0x0001))

    def poll(self):
        """Read the keyboard, send if changed. Returns the command now held."""
        self.blocked = not self._focused()
        cmd, held = 0, []
        if not self.blocked:
            for vk, (bits, _key, name) in self.map.items():
                if self._down(vk):
                    cmd += bits
                    held.append(name)
            d = {nm: self._down(vk) for nm, vk in self.rstick_vk.items()}
            stick = self.km.rstick_bits(self.c, d["Up"], d["Down"], d["Left"], d["Right"])
            if stick:
                cmd += stick
                held.append(self.km.rstick_label(d["Up"], d["Down"], d["Left"], d["Right"]))
        now = time.time()
        # Send only on change: the Switch runs its own auto-repeat from a held
        # direction, so retransmitting every frame adds latency for nothing.
        if cmd != self.last_cmd or now - self.last_sent > self.KEEPALIVE_S:
            self.c.send_cmd(cmd)
            self.last_cmd, self.last_sent = cmd, now
            self.sent_n += 1
        self.held, self.command = held, cmd
        return cmd

    def action_index(self, command):
        """Index into controller.tetris_enum, or -1 for a chord the agent's
        8-action space cannot represent."""
        try:
            return self.c.tetris_enum.index(command)
        except ValueError:
            return -1

    def release(self):
        try:
            self.c.send_cmd(self.c.NO_INPUT)
        except Exception:
            pass
