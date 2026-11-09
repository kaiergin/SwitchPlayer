# SwitchPlayer: real-time reinforcement learning for a live Nintendo Switch

The goal is an agent that learns *while* playing a live console. Video comes in
over a capture card and controller commands go out over serial. There is no
pausing, no rewinding and no faster-than-real-time simulator.

The learning side starts from MuZero / EfficientZero, the strongest
sample-efficient model-based methods, and changes them where the real-time goal
demands it. Stock MuZero runs a tree search before every action, and that does
not fit a 16.7 ms frame at 60 fps. Development runs on Atari games (Pong,
Ms. Pac-Man, Q*bert) through the Arcade Learning Environment, which runs about
80x faster than real time. The Switch side (an emulated controller, capture-card
state detection, Tetris) is built and tested on its own, but it is not yet wired
to the learner.

## Current state (November 2026)

- **EfficientZero, reproduced.** A from-scratch PyTorch implementation with
  Ray-distributed self-play. It reaches EfficientZero-level Pong within the
  Atari-100k budget.
- **A real-time variant.** The actor plays straight from the policy network
  with no search at act time. Search runs off the critical path, when a
  position is stored, to make training targets.
- **A generational curriculum.** Each new network learns from its predecessor
  by distillation, and the reward clip loosens only at generation swaps. On
  Ms. Pac-Man the score compounded over two swaps, and each student beat its
  teacher.
- **In progress: learned options (Test O).** OptionZero-style option heads
  predict short move sequences, which the search can jump across. The control
  run is live and the option arm is built and audited.
- **The Switch / Tetris environment.** The game-state and line-clear detector
  was rewritten without neural networks (fixed regions plus template matching).
  The repo also has keyboard control of the console, play recording and live
  monitoring tools.

## Results

All scores are unclipped whole-game scores.
- **Ms. Pac-Man** scores come from an exact evaluation of saved snapshots (30
  no-op starts). "Deployed" means the network exactly as the actor plays it,
  including its batch-norm statistics.
- **Pong** rows are means of 8 evaluation games.

Multi-seed runs are noted; the rest are one seed.

| Experiment | Game | Data | Result | Reference point |
|---|---|---|---|---|
| EfficientZero reproduction (the paper-faithful loss setup) | Pong | 100k env steps | 2-seed mean +15.0 (seeds +18.25 and +11.75); both seeds about +18.5 by iteration 80 | EfficientZero paper: 20.1 |
| Real-time acting (policy only, temperature + batch-norm recalibration) | Pong | 90 iterations | Learned to win on 2 of 2 seeds; deployed mean over iterations 60–90 of +5.80 on seed 1 | — |
| Generational curriculum, two swaps | Ms. Pac-Man | about 270k env steps in all | Generation 0: 1,114, then 1,657, then 2,118 deployed; each student above its teacher | EfficientZero V2: 2,251 and BBF: 2,008, both at 100k |
| Fast lineage, two swaps in one budget (Test B2) | Ms. Pac-Man | 100k env steps | 1,629.5 deployed over the last 10k steps | Five swaps in the same budget (Test B): 1,015.9 |

The research log, `CURRICULUM.md`, also keeps the experiments that did not work:
- **EfficientZero V2's search-based value estimation:** worse on both seeds.
- **Gumbel search at act time:** 1 of 2 seeds.
- **BBF's AdamW:** the student collapsed to random play.
- **BBF-style resets:** no gain.
- **BBF-style resets plus annealing ramps:** worse.
- **Duration tokens on Pong:** no breakout.
- **Action sequences mined from strong play:** no score difference, though they
  needed 44% fewer decisions.
- **A test of whether the master–student value gap points at real weaknesses:**
  primary result not shown.

## How it works

- **Model.** A representation, dynamics and prediction network with
  EfficientZero's additions:
  - an LSTM value prefix;
  - a self-supervised temporal-consistency loss through a projector;
  - off-policy target correction;
  - reanalysis.
  
  The paper-faithful setup also prevents a projector collapse that showed up in
  every earlier run (its rank fell to about 2).
- **Acting.** The actor samples the policy head, so the latency per action is one
  network call. A vectorized MCTS (`muzero_vtree.py`) runs on each position when
  it is stored, and its visit counts become the policy target. The acting
  network's batch-norm statistics are recalibrated on recent play, so the
  network behaves the same way it trained.
- **Curriculum** (`experiments/distill.py`, `benchmarks/supervisors/`).
  1. A master plays.
  2. A fresh student is trained offline on the master's recent play.
  3. The student continues with its own play.
  
  The reward clip changes only at a swap.
- **Options** (`muzero_vtree_options.py`, `muzero_option_targets.py`). Each
  extra head guesses the next move of a short sequence. The search can take a
  whole option as one edge, and the dynamics network jumps across it in one
  call.

## Running it

The pipeline targets **Python 3.12** with PyTorch and Ray, on Windows. It cannot
run on Python 3.13, because Ray publishes no `win_amd64` wheel for cp313.

```
winget install --id Python.Python.3.12 --exact
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pytest
```

It is verified on an RTX 4070 with torch 2.13.0+cu130, ray 2.57.0 and
gymnasium 1.3.0. `requirements.lock.txt` is a full freeze of that environment.

**Train the current recipe on Ms. Pac-Man** (Test O's control arm, from scratch
to 100k env steps, about 35 h on one RTX 4070):

```
.venv\Scripts\python.exe experiments/ablation.py ^
  --arms effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-plain-o0 ^
  --seed 0 --iterations 300 --train-steps 2000 --eval-every 5 ^
  --env MsPacmanNoFrameskip-v4 --frameskip 4 --episode-cap 3000 ^
  --episodes-per-iter 3 --checkpoint-every 2 --snapshot-weights ^
  --ray --cpu-actor --stop-env-steps 100000
```

Every experiment's configuration is a named arm in `experiments/ablation.py`.
Add `--resume` to continue from the last checkpoint.
`benchmarks/probes/probe_exhaustive_eval.py` gives the exact evaluation of the
saved snapshots.

**Multi-generation lineages** run under `benchmarks/supervisors/`. For example,
Test B2's two-swap lineage is `supervise_testb.py t0 0`, then `t1 0` and `t2 0`.
Before each launch a supervisor:
- checks a file-hash manifest;
- runs about 2,000 tests;
- prepares the distillation set at each swap;
- resumes after a crash.

The supervisors assume the author's machine layout (frozen git worktrees per
run).

## How experiments are run

- **Pre-registration.** Each experiment is written into `CURRICULUM.md` before
  it is built: its arms, its reads, and a decision rule (BETTER / NO DIFFERENT
  / WORSE against a noise band).
- **Code audits.** Code is audited by independent reviews until a round finds
  nothing above low severity. Mutation checks confirm that the tests catch
  planted bugs.
- **Frozen runs.** Each run executes from a frozen worktree at its audited
  commit, so later edits cannot touch a live run.
- **Honest scoring.** Results are reported unclipped, from whole games, for the
  network as deployed.

## Repository map

| Path | What it is |
|---|---|
| `muzero.py`, `networks.py` | The learner: networks, targets, replay, the train step |
| `muzero_vtree.py`, `muzero_vtree_options.py` | Vectorized MCTS, plain and with option edges |
| `experiments/ablation.py` | Every experiment's arm, and the training entry point |
| `experiments/ray_actors.py`, `experiments/distill.py` | Self-play actors; the distillation step at generation swaps |
| `benchmarks/supervisors/`, `benchmarks/probes/` | Run orchestration, and the analysis probes with their outputs |
| `tests/` | The pytest suites |
| `CURRICULUM.md` | The research log, newest first: pre-registrations, results, decisions |
| `DESIGN.md`, `ROADMAP.md`, `benchmarks/RESULTS.md` | Earlier design notes and latency benchmarks |
| `controller.py`, `core.py`, `tetris.py`, `envTest.py` | The Switch environment: serial controller link, environment base, Tetris |
| `tetris_masks.py`, `tetris_live.py`, `tetris_roi_tool.py` | Tetris state detection, the live detector monitor, the ROI picker |
| `tetris_keyboard.py`, `switch_control.py`, `switch_keymap.py`, `view_switch.py` | Keyboard control of the console and the capture-card viewer |
| `record_play.py`, `session_recorder.py`, `controller_diag.py` | Recording human play, and diagnosing the controller link |

## The Switch / Tetris environment

The environment (`controller.py`, `core.py`, `tetris.py`, `envTest.py`) needs
`gym` and `pyserial`, which are not in the lock file; see the notes at the
bottom of `requirements.txt`. It no longer needs TensorFlow. `tetris_masks.py`
replaced the three CNNs with measured regions of interest and template
correlation, so state and reward detection run on numpy and OpenCV alone.
- `python tetris_masks.py --eval` reproduces the accuracy figures against the
  labelled corpus, without any hardware. The corpus of 2020 Tetris 99 captures
  is not included in the repo.
- `--demo` steps through frames with the ROIs drawn on.
- `tetris_live.py` shows what the detector sees on the live capture feed. It
  never opens the controller port.

Only Windows is supported.

### Hardware used
* [Arduino Uno R3](https://docs.arduino.cc/hardware/uno-rev3/), or a
  compatible board with the ATmega16U2 USB chip (it gets reflashed, see below).
* A USB-to-TTL serial adapter at 5 V logic, like the Uno. Boards based on FTDI,
  CP2102 or CH340 chips work, for example FTDI's
  [TTL-232R-5V](https://ftdichip.com/products/ttl-232r-5v/).
* [Elgato Cam Link 4K](https://www.elgato.com/us/en/p/cam-link-4k), or any UVC
  capture card that `view_switch.py` can open.

The controller emulation is based on
[SwitchInputEmulator](https://github.com/wchill/SwitchInputEmulator).

### Emulated controller: wiring and firmware

The Arduino sits between two USB connections. It is **not** driven over its own
USB cable. Three wires carry the commands:

```
   PC ──USB── [USB-to-TTL adapter] ──3 wires── [Arduino Uno] ──USB-B── Switch dock

        adapter GND ─────────────── Uno GND
        adapter TX  ─────────────── Uno pin 0 (RX)
        adapter RX  ─────────────── Uno pin 1 (TX)
        adapter VCC ──── leave disconnected (the dock powers the Uno)
```

TX goes to RX and RX goes to TX; this crossover is the one people get wrong.
Leave the adapter's VCC off, so the board is not powered from two sources at
once.

The Uno's own USB-B port plugs into the **dock**, not the PC. Its ATmega16U2 is
reflashed with LUFA firmware, so the board enumerates to the Switch as a wired
controller. The ATmega328P runs a sketch that reads this UART and forwards the
inputs. Both come from the upstream project linked above.

**Flash the 328P sketch BEFORE reflashing the 16U2.** Once the 16U2 runs HID
firmware, the normal IDE upload path is gone, and you have to restore the stock
16U2 firmware before you can upload a sketch again.
- To put the 16U2 into DFU mode, short the two-pin reset header beside the USB
  jack.
- Disconnect the TTL adapter while uploading a sketch, because it shares pins
  0 and 1 with the upload path.

#### What the PC side expects

`controller.py` is a library, not a script:

| | |
|---|---|
| Port | `PORT` at `controller.py:14`, **hardcoded: edit it** |
| Baud | 19200, 8N1, 1 s timeout |
| Handshake | 9x `0xFF` -> `0xFF`; `0x33` -> `0xCC`; `0xCC` -> `0x33` |
| Packet | 8 bytes `[btn_hi, btn_lo, dpad, LX, LY, RX, RY, 0x00]` + CRC8-CCITT |
| Per-packet reply | `0x90` (`RESP_USB_ACK`) |

`serial.Serial(...)` runs at **module import**, so `import controller`, and
therefore `import tetris`, throws immediately if the port is absent. There is no
lazy-connect path.

#### Checking it works

```bash
pip install pyserial                       # not in the lock file
python -c "import serial.tools.list_ports as l; print([p.device for p in l.comports()])"
python -c "import controller; print('sync:', controller.sync())"   # True == talking
python -c "import controller; controller.sync(); controller.testbench_btn()"
```

- If `sync()` returns `True`, the handshake completed and the microcontroller is
  acknowledging packets.
- If it returns `False`, the wiring or the baud rate is wrong.
- If the Switch shows no controller at all, the problem is the 16U2 firmware,
  not this link.
