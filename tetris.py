"""Gym-style environment wrapping a live Tetris 99 session on a Switch.

Video in over a capture card, controller commands out over serial
(`controller.py`). State and reward come from `tetris_masks`: fixed regions of
interest, colour predicates and template correlation, with a state machine on
top.

This replaces three Keras CNNs (`model_environment`, `model_reward`,
`model_pregame`) and their training loops. What that bought:

* **No TensorFlow.** The env now needs only numpy and OpenCV, so it runs on the
  same interpreter as the rest of the repo instead of a separate Python 3.7
  environment.
* **Reward pays once per clear.** The old path granted +1 on *every* frame the
  reward net fired inside a 7-step window opened by a hard drop, so one Tetris
  was worth anywhere from 1 to 7 depending on timing -- and a line cleared by
  gravity or a soft drop was worth nothing at all, because the window only
  opened on DPAD_U. Detection is now edge-triggered and runs every frame.
* **Two reward classes.** "Tetris" and small clears are distinguished, where
  the old net mapped both of its non-neutral classes to the same +1 (and its
  middle class was never trained -- the label set had three entries, the data
  directory two).
* **A flash no longer ends the episode.** Any single non-neutral frame used to
  set done=True. Leaving the in-game state now needs several consecutive
  confirming frames.
* **No disk round-trip per frame.** Frames went to `temp.png` and back through
  `cv2.imread` purely to swap channels, twice per step, inside a 20 Hz loop.

Accuracy of the replacement, held out (`python tetris_masks.py --eval`):

    in-game vs out-of-game    recall 0.9943   reject 1.0000
    out-of-game screen id     0.9989
    GO glyph                  10/10
    "Tetris" glyph            0.970 recall, 0.003 false positives

The observation is the PLAY AREA only -- hold box, matrix, next queue -- not
the whole screen. The 98 opponent boards are a large, busy region the agent
cannot act on; excluding them is the single biggest reduction available in the
input. See `tetris_masks.PLAY_AREA`.

See `tetris_masks.py` for the ROIs, thresholds and how they were derived.
"""
import cv2
import numpy as np
from gym import spaces

from core import Env
import controller
from controller import (sync, send_cmd, tetris_enum, p_wait,
                        BTN_A, BTN_B, BTN_R, BTN_L)
from tetris_masks import (TetrisMasks, TetrisState, PREGAME, PLAYING,
                          PLAY_AREA, OBS_SIZE, crop_region)

RENDER = True
ACTION_SPACE = len(tetris_enum)
# Seconds between decisions. 0.05 -> 20 Hz, matching what the capture card and
# the serial link sustain.
STEP_PERIOD = 1.0 / 30      # 33.3 ms; Tetris 99 runs at 60 fps and DAS
                            # is specified in frames, so 20 Hz could neither
                            # observe nor reproduce charge timing
# Menu navigation. A button press has to be held, then released, and the
# console needs time to animate the transition; these are the timings the
# previous version used and they are known to work.
MENU_HOLD = 3.0
MENU_RELEASE = 3.0


class Tetris(Env):
    # -1 so the agent cannot choose BTN_NONE.
    action_space = spaces.Discrete(ACTION_SPACE - 1)

    def __init__(self, capture_device=0, debug=False, render=RENDER,
                 reward_tetris=None, reward_garbage=None, reward_death=None,
                 obs_size=OBS_SIZE):
        self.DEBUG = debug
        self.render_enabled = render
        # None returns the native-resolution crop instead of resizing.
        self.obs_size = tuple(obs_size) if obs_size else None
        self.clock = 0
        self.cumulative_reward = 0.0
        self.current_frame = None

        self.masks = TetrisMasks()
        rw = {}
        if reward_tetris is not None:
            rw["reward_tetris"] = reward_tetris
        if reward_garbage is not None:
            rw["reward_garbage"] = reward_garbage
        if reward_death is not None:
            rw["reward_death"] = reward_death
        self._reward_kwargs = rw
        self.sm = TetrisState(self.masks, **rw)

        # Sync to the switch so that we can send controller commands
        if self.DEBUG:
            print('Attempting to sync to MCU')
        sync()
        if self.DEBUG:
            print('Successfully synced to MCU')
        # Connect controller
        send_cmd(BTN_L + BTN_R)
        p_wait(0.5)
        send_cmd()
        p_wait(0.1)
        send_cmd(BTN_A)
        p_wait(0.5)
        send_cmd()
        if self.DEBUG:
            print('Successfully paired controller')
        # Capture switch screen
        try:
            self.switch_screen = cv2.VideoCapture(capture_device)
        except Exception:
            print('Unable to open capture device. Is it plugged in?')
            raise SystemExit(1)

    # -- frame handling ----------------------------------------------------
    def _read(self):
        ok, frame = self.switch_screen.read()
        if not ok:
            raise RuntimeError(
                'Error reading from switch screen. Switch disconnected?')
        self.current_frame = frame
        if self.render_enabled:
            self.render()
        return frame

    def _observation(self):
        """Only the player's own board: hold box, matrix, next queue.

        The previous version returned the central half of the capture, which
        still contained most of the 98 opponent boards -- a large, busy,
        constantly-changing region the agent cannot act on and which dominates
        the pixel statistics. PLAY_AREA is measured in `tetris_masks` and the
        crop is mapped through the letterbox rectangle, so it stays correct on
        a capture whose picture is inset.
        """
        obs = crop_region(self.current_frame, PLAY_AREA, self.sm.box)
        if self.obs_size:
            obs = cv2.resize(obs, self.obs_size, interpolation=cv2.INTER_AREA)
        return obs

    # -- gym API -----------------------------------------------------------
    def step(self, action):
        # Deferred ACK: the reply to this packet is collected at the start of
        # the next one, having sat in the OS buffer for a whole step. Measured
        # 8.00 -> 4.12 ms median, i.e. 24% -> 12% of a 33 ms step.
        controller.ACK_NONBLOCKING = True
        send_cmd(tetris_enum[action])
        p_wait(STEP_PERIOD)
        frame = self._read()

        out = self.sm.step(frame)
        reward, done = out['reward'], out['done']

        if self.DEBUG:
            if out['event']:
                print(f"[{self.clock}] {out['event']}  reward {reward:+.2f}")
            elif self.clock % 40 == 0:
                print(f"[{self.clock}] state {out['state']} "
                      f"edge {out['edge']:.0f}")

        self.clock += 1
        self.cumulative_reward += reward
        return self._observation(), reward, done, {
            'state': out['state'], 'event': out['event'],
            'edge': out['edge']}

    def reset(self):
        """Drive the menus until a match is actually under way.

        Returns once the state machine reports PLAYING, i.e. the GO glyph was
        seen (or the matrix started changing, the fallback for a dropped
        frame). The previous version returned as soon as its pregame net said
        anything other than 'pregame', with no notion of a match having
        started.
        """
        if self.DEBUG and self.clock:
            print('Previous cumulative reward:', self.cumulative_reward)
        self.clock = 0
        self.cumulative_reward = 0.0
        # Carry the capture calibration across episodes -- it is a property of
        # the card, not the match, and re-measuring costs frames.
        self.sm = TetrisState(self.masks, content_box=self.sm.box,
                              swap_rb=self.sm.swap_rb, **self._reward_kwargs)

        # Calibrate once per session; skipped on later resets.
        # Retry until it takes rather than giving up after a fixed window: a
        # dark screen at startup would otherwise leave the whole episode
        # running with no letterbox crop and no channel correction.
        attempts = 0
        while self.sm.box is None and attempts < 600:
            frame = self._read()
            attempts += 1
            if frame.mean() > 5 and self.sm.calibrate(frame)[0]:
                if self.DEBUG:
                    print(f'calibrated: box {self.sm.box} swap_rb {self.sm.swap_rb}')
                break
            p_wait(STEP_PERIOD)
        if self.sm.box is None:
            raise RuntimeError(
                'Could not calibrate the capture: no bright frame seen. Is the '
                'console awake and the capture card showing a picture?')

        while True:
            frame = self._read()
            out = self.sm.step(frame)
            state = out['state']

            if state == PLAYING:
                if self.DEBUG:
                    print('Match started')
                return self._observation()
            if state == PREGAME:
                # In game, waiting for GO. Nothing to press.
                p_wait(STEP_PERIOD)
                continue

            # A menu. end_screen / error / title_a advance with A, title_b
            # with B -- the mapping the previous env used, preserved.
            if self.DEBUG:
                print('Menu:', state)
            send_cmd(BTN_B if state == 'title_b' else BTN_A)
            p_wait(MENU_HOLD)
            send_cmd()
            p_wait(MENU_RELEASE)

    def render(self, mode='human'):
        if self.current_frame is None:
            return
        cv2.imshow('Switch Screen', cv2.resize(self.current_frame, (1280, 720)))
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()
        self.switch_screen.release()
