## Reinforcement Learning on the Nintendo Switch

Games to learn:
* Tetris 99 (current)
* Mario Maker 2

## Purpose

**This is a real-time reinforcement learning research project.** The goal is an
agent that learns *while* playing a live console — video in over a capture card,
controller commands out over serial — with no pausing, no rewinding, and no
faster-than-real-time simulation.

MuZero is the **starting point, not the destination**. It was chosen because it
is the strongest model-based algorithm that reliably converges to a good policy,
but it is not a real-time algorithm: stock MuZero runs hundreds of tree-search
simulations to select each action, which cannot fit inside a 16.7 ms frame at
60 fps. The implementation here is expected to diverge from the paper, and
already does.

### What "real-time" changes

The binding constraint is **latency per action**, not sample efficiency and not
throughput. A technique that halves the sample requirement but doubles acting
latency is a regression here.

Deviations from stock MuZero that are deliberate, and should not be "corrected"
back toward the paper without discussion:

| Stock MuZero | Here | Why |
|---|---|---|
| MCTS selects every action | Agent samples the policy head directly (`MuZeroAgent.gameplay_process`) | Search does not fit the frame budget |
| Search produces actions | Search runs off the critical path as offline re-analysis over stored trajectories (`MCTSHelper`) | Keeps acting latency independent of search cost |
| Bootstrapped n-step value targets | Monte Carlo only during warmup; afterwards every position gets the paper's n-step target — TD_STEPS real rewards, then the nearest search root value at/past the horizon (fresh network value where no search lies ahead). A position is never trained on its own search value | Pure MC holds only while the network is untrained — see "Custom ideas" |

### Measuring against the goal

Benchmarks must report **acting latency against the frame budget**, not only
throughput. `benchmarks/RESULTS.md` has the current numbers; the one that
matters most is `initial_inference` at 2.24 ms/frame — about 13% of a 60 fps
budget, which is the real headroom figure.

Be careful reading testbed throughput results: ALE runs MsPacman at ~4,900
steps/s, roughly 80x real time, so producer/consumer ratios measured there do
not reflect the real system, where self-play is capped at the console's frame
rate. Rate-limiting self-play to the target frame rate would make the testbed
reproduce the actual constraint — not yet implemented.

`MsPacman-v4` is only a testbed. `muzero.py` trains against it today; wiring it
to the Switch environment is still to do.

### Not yet implemented: MuZero successor work

Several papers improve on MuZero by replacing specific components, particularly
**how future reward is predicted through the dynamics network**. None are
implemented here yet. Verify details against the papers before building.

* **EfficientZero** — replaces per-step reward prediction with an LSTM *value
  prefix* over the unrolled dynamics; adds a self-supervised latent consistency
  loss (cosine similarity through a projector) and off-policy correction. The
  plain MSE latent consistency loss in `train_step` is a weaker form of the
  same idea.
* **Gumbel MuZero** — preserves the policy improvement guarantee at very small
  simulation budgets. The most directly relevant to the real-time constraint,
  since it attacks the reason search is unaffordable rather than working around it.
* **EfficientZero V2** — the above extended to continuous control and sampled actions.
* **Stochastic MuZero** — chance nodes, for environments that are not deterministic.
* **Sampled MuZero** — large or continuous action spaces.
* **MuZero Unplugged / Reanalyse** — the offline re-analysis framing `MCTSHelper`
  already resembles.

### To do list

See **[ROADMAP.md](ROADMAP.md)** for the prioritised plan and the reasoning
behind it. Current priority is EfficientZero's end-to-end value prefix.

Unsequenced items:
* Fill replay buffer with random data
* Mult-step reward?
* Hyperparameter Tuning -- c_puct and the search budget both need re-tuning now
  that MCTS values are min-max normalised

### Custom ideas
* Heavy reward discounting and small trajectory with no bootstrapped value as initial agent (warmup only: once search starts, targets bootstrap the network — tree leaves during search, n-step tails on unsearched positions)
* Human gameplay as training for dynamics network

### Dependencies

The active pipeline targets **Python 3.12** with PyTorch and Ray. Not 3.13:
ray publishes no `win_amd64` wheel for cp313.

```
winget install --id Python.Python.3.12 --exact
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pytest
```

Verified on an RTX 4070 with torch 2.13.0+cu130, ray 2.57.0, gymnasium 1.3.0.
`requirements.lock.txt` is a full freeze of that environment.

The Switch/Tetris environment (`controller.py`, `core.py`, `tetris.py`,
`envTest.py`) needs `gym` and `pyserial`, which are not in the lock file above
-- see the notes at the bottom of `requirements.txt`. It no longer needs
TensorFlow: `tetris_masks.py` replaced the three CNNs with measured regions of
interest and template correlation, so state and reward detection run on numpy
and OpenCV alone. `python tetris_masks.py --eval` reproduces the accuracy
figures against the labelled corpus without any hardware attached, and
`--demo` steps through frames with the ROIs drawn on.

Currently only support for Windows machines  

### References

[Repo for sending commands to switch](https://github.com/wchill/SwitchInputEmulator)  

### Hardware Used
* [Arduino Uno R3](https://www.amazon.com/Sintron-UNO-R3-ATMEGA328P-Arduino/dp/B073DYD97C/ref=sxin_2_ac_d_pm?ac_md=1-0-VW5kZXIgJDEw-ac_d_pm&cv_ct_cx=arduino+uno+r3&keywords=arduino+uno+r3&pd_rd_i=B073DYD97C&pd_rd_r=9305d6a8-c762-46e8-b3f8-356745bc8a6d&pd_rd_w=z4ysu&pd_rd_wg=MwvP1&pf_rd_p=0e223c60-bcf8-4663-98f3-da892fbd4372&pf_rd_r=RAANRF34YJKT950W9MVQ&psc=1&qid=1584519878&s=electronics)
* [USB to serial](https://www.amazon.com/gp/product/B01CYBHM26/ref=ppx_yo_dt_b_asin_title_o00_s00?ie=UTF8&psc=1)
* [Camlink 4k](https://www.amazon.com/gp/product/B07K3FN5MR/ref=ppx_yo_dt_b_asin_title_o02_s00?ie=UTF8&psc=1)

### Emulated controller: wiring and firmware

The Arduino sits between two USB connections and is **not** driven over its own
USB cable. Three wires carry the commands:

```
   PC ──USB── [USB-to-TTL adapter] ──3 wires── [Arduino Uno] ──USB-B── Switch dock

        adapter GND ─────────────── Uno GND
        adapter TX  ─────────────── Uno pin 0 (RX)
        adapter RX  ─────────────── Uno pin 1 (TX)
        adapter VCC ──── leave disconnected (the dock powers the Uno)
```

TX goes to RX and RX goes to TX — the one crossover people get wrong. Leave the
adapter's VCC off so the board is not powered from two sources at once.

The Uno's own USB-B port plugs into the **dock**, not the PC. Its ATmega16U2 is
reflashed with LUFA firmware so the board enumerates to the Switch as a wired
controller; the ATmega328P runs a sketch that reads this UART and forwards
inputs. Both come from the upstream project linked under References.

**Flash the 328P sketch BEFORE reflashing the 16U2.** Once the 16U2 is HID
firmware the normal IDE upload path is gone, and you have to restore the stock
16U2 firmware to upload a sketch again. Put the 16U2 into DFU mode by shorting
the two-pin reset header beside the USB jack. Disconnect the TTL adapter while
uploading a sketch — it shares pins 0/1 with the upload path.

#### What the PC side expects

`controller.py` is a library, not a script. Verified against its own code:

| | |
|---|---|
| Port | `PORT` at `controller.py:14` — **hardcoded, edit it** |
| Baud | 19200, 8N1, 1 s timeout |
| Handshake | 9x `0xFF` -> `0xFF`; `0x33` -> `0xCC`; `0xCC` -> `0x33` |
| Packet | 8 bytes `[btn_hi, btn_lo, dpad, LX, LY, RX, RY, 0x00]` + CRC8-CCITT |
| Per-packet reply | `0x90` (`RESP_USB_ACK`) |

`serial.Serial(...)` runs at **module import**, so `import controller` — and
therefore `import tetris` — throws immediately if the port is absent. There is
no lazy-connect path.

#### Checking it works

```bash
pip install pyserial                       # not in the lock file
python -c "import serial.tools.list_ports as l; print([p.device for p in l.comports()])"
python -c "import controller; print('sync:', controller.sync())"   # True == talking
python -c "import controller; controller.sync(); controller.testbench_btn()"
```

`sync()` returning `True` means the handshake completed and the MCU is ACKing.
If it returns `False`, the wiring or the baud rate is wrong. If the Switch shows
no controller at all, that is the 16U2 firmware, not this link.
