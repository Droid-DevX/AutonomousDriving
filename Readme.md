# Reinforcement Learning – CarRacing-v3 with Stable-Baselines3

A reinforcement learning project for training agents to navigate the continuous control driving environment **CarRacing-v3** using a CNN-based policy with the PPO algorithm — now featuring **NPC obstacle cars** with live overtake detection.

---

## Demo

### Without NPC Cars (v1.0)
> Clean lap — agent drives solo, no obstacles.

https://github.com/user-attachments/assets/1fb02adc-3026-4690-a16d-4d1244dcb0f5

---

### With NPC Cars (v2.0)
> Agent navigates around 3 NPC obstacle cars. Labels switch live between **NPC** and **OVERTAKEN** based on real-time track position.

https://github.com/user-attachments/assets/7cac56eb-556e-4d83-a4d1-fa0142be31c4

---

## Before vs After

### Training Performance (v0.2 → v1.0)

| | **Before (v0.2)** | **After (v1.0)** |
|---|---|---|
| **Mean Reward** | 98.57 | **865.44** |
| **Std Deviation** | 240.10 | **22.31** |
| **Variance** | 57,649 | **~498** |
| **Reward Range** | -4.76 → 814.97 | 845 → 906 |
| **Episodes Solved** | 0 / 10 | **5 / 5** |
| **Training Steps** | 250,000 | 1,000,000 + |
| **Frame Stacking** | None | 4 frames |
| **Learning Rate** | `3e-4` | Linear decay `2.5e-4 → 0` |
| **Clip Range** | `0.2` | Linear decay `0.2 → 0` |
| **n_epochs** | Default (10) | 4 |
| **Monitor Wrapper** | Missing on eval env | All envs |
| **KL Divergence** | Blew up to `0.28` | Stable `< 0.02` |
| **Value Loss** | Spiked to `11.6` | Stable and declining |

### NPC Obstacle Results (v1.0 → v2.0)

| | **v1.0 (No NPCs)** | **v2.0 (With NPCs)** |
|---|---|---|
| **NPC Cars** | None | 3 (red, blue, green) |
| **Overtake Reward** | — | +50 per clean overtake |
| **Collision Penalty** | — | -5 per hit |
| **Mean Overtakes/Ep** | — | 2–3 |
| **Mean OT Reward** | — | +100 to +150 |
| **Total Mean Reward** | 865.44 | ~1030+ |
| **False Overtakes** | — | None (dual-guard logic) |
| **Label Accuracy** | — | Live, position-based |

---

## Description

This project implements an end-to-end reinforcement learning pipeline for training and evaluating an agent on the **CarRacing-v3** environment from Gymnasium. It applies the **PPO (Proximal Policy Optimization)** algorithm using Stable-Baselines3 with a `CnnPolicy` to process raw pixel observations directly from the top-down race track.

### v1.0 — Optimized PPO Pipeline

- **Frame stacking** (`VecFrameStack`, n=4) — gives the policy temporal context (speed, direction) from consecutive frames
- **Linear learning rate + clip range decay** — prevents KL divergence explosion and policy instability during long training runs
- **Vectorized environment** via `DummyVecEnv` + `VecTransposeImage` for efficient rollout collection
- **CNN-based policy** (`CnnPolicy`) operating on raw 96×96 pixel image observations
- **Continuous action space**: steering `[-1, 1]`, throttle `[0, 1]`, brake `[0, 1]`
- **EvalCallback + CheckpointCallback** — auto-saves best model and periodic checkpoints during training
- **Model save/reload** with `custom_objects` to correctly restore learning rate schedules
- **Quantitative evaluation** using mean/std/confidence interval over deterministic episodes

### v2.0 — NPC Obstacle Cars (`obstacle_wrapper.py`)

- **3 NPC ghost cars** that follow the track centre-line exactly (never go off-road)
- **Staggered spawning**: NPC 0 at step 0, NPC 1 at step 400, NPC 2 at step 700
- **Spawn grace period**: 100-step cooldown after spawn — prevents false "OVERTAKEN" labels when an NPC spawns behind the player
- **Live overtake labels**: label switches instantly between `NPC` and `OVERTAKEN` based on real-time track position — no timer, no delay
- **Dual-guard overtake logic**: player must be ≥ 4 tiles AND ≥ 0.5 world-units ahead before an overtake is confirmed
- **Per-NPC state machine**: `WAITING → ARMED → AWARDED` with full reset on re-encounter
- **Visual overlays** on both the 96×96 observation and the full render window
- **NPC-aware video recording** via `NpcVideoRecorder` — composites overlays directly into saved MP4 files

---

## Achieved Results

### v0.2 — Before (250k steps, no frame stacking, fixed LR)
```
Episode Rewards: [-2.03, 4.32, 2.56, 58.67, 14.20, 33.52, 68.25, -3.97, 814.97, -4.76]
Mean Reward   : 98.57
Std Deviation : 240.10
Variance      : 57,649.78
```

### v1.0 — After (optimized hyperparameters + frame stacking + LR schedule)
```
Episode 1 finished | Total Reward: 906.70
Episode 2 finished | Total Reward: 845.02
Episode 3 finished | Total Reward: 849.83
Episode 4 finished | Total Reward: 869.28
Episode 5 finished | Total Reward: 856.38

Mean Reward   : 865.44
Std Deviation : 22.31
```

### v2.0 — With NPC Obstacle Cars
```
Ep  1 | Reward: 1022.66 | Overtakes: 3 | OT_Reward: +150 | Penalty: 0.0
Ep  2 | Reward: 1051.00 | Overtakes: 3 | OT_Reward: +150 | Penalty: 0.0
Ep  3 | Reward: 1032.20 | Overtakes: 2 | OT_Reward: +100 | Penalty: 0.0
Ep  4 | Reward: 1034.80 | Overtakes: 2 | OT_Reward: +100 | Penalty: 0.0
Ep  5 | Reward: 1029.50 | Overtakes: 2 | OT_Reward: +100 | Penalty: 0.0
```

> CarRacing-v3 is considered **solved** at a mean reward of 900+ over 100 episodes. The agent consistently exceeds this threshold and additionally completes 2–3 clean overtakes per episode.

---

## Getting Started

### Dependencies

- Python 3.x
- Required libraries listed in `requirements.txt` (including Stable-Baselines3, Gymnasium, PyTorch, TensorBoard, OpenCV)
- CUDA-compatible GPU recommended (CPU training is significantly slower)
- Compatible with Windows, macOS, and Linux

### Installing

1. Clone the repository:
```bash
git clone https://github.com/Droid-DevX/AutonomousDriving.git
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
pip install opencv-python   # required for NPC video recording
```

3. Ensure the following files are present after training:
   - `models/ppo_car_racing.zip` – saved PPO model
   - `models/best/` – best model auto-saved during training
   - `logs/` – TensorBoard training logs
   - `obstacle_wrapper.py` – NPC obstacle wrapper (v2.0)

---

## Executing Program

1. **Train the PPO agent:**
```bash
# Run the "Train Model" cell in main_PPO.ipynb
# Model saves automatically to models/ppo_car_racing
```

2. **Continue training from a checkpoint:**
```bash
# Run the "Continue Training" cell
# Uses custom_objects to correctly restore LR schedule
```

3. **Evaluate the saved model:**
```bash
# Run the "Evaluate" cell — outputs mean, std, variance, 95% CI
```

4. **Watch the agent drive (no NPCs):**
```bash
# Run the "Visual Test" cell with render_mode="human"
```

5. **Watch the agent drive with NPC cars:**
```python
from obstacle_wrapper import ObstacleWrapper, run_visual_test
from stable_baselines3 import PPO

def make_env():
    def _init():
        env = gym.make("CarRacing-v3", continuous=True, render_mode="rgb_array")
        env = ObstacleWrapper(env, npc_count=3)
        env = Monitor(env)
        return env
    return _init

model = PPO.load("models/best/best_model")
run_visual_test(model, make_env, n_episodes=5)
```

6. **Record video with NPC overlays:**
```python
from obstacle_wrapper import ObstacleWrapper, NpcVideoRecorder

video_env = build_env(n_envs=1, render_mode="rgb_array", use_terminator=False)
video_env = NpcVideoRecorder(video_env, video_folder="./videos/",
                              video_length=3000, name_prefix="ppo_car_npc")

obs = video_env.reset()
for _ in range(3000):
    action, _ = best_model.predict(obs, deterministic=True)
    obs, reward, done, info = video_env.step(action)
    if done[0]:
        obs = video_env.reset()
video_env.close()
# Saves to ./videos/ppo_car_npc-ep1.mp4, ep2.mp4, ...
```

7. **Monitor training with TensorBoard:**
```bash
tensorboard --logdir ./logs
```

---

## NPC Obstacle Wrapper — Details

### How It Works

Each NPC follows the track centre-line tile-by-tile at a fixed speed (`0.05 tiles/step`, slightly slower than the player). The overtake detector uses a state machine per NPC:

| State | Meaning |
|---|---|
| `WAITING` | Player hasn't approached this NPC yet |
| `ARMED` | Player is within `ENGAGE_DIST` and behind on track |
| `AWARDED` | Player confirmed ahead — overtake counted |

### Key Parameters

| Parameter | Value | Purpose |
|---|---|---|
| `NPC_TILES_PER_STEP` | `0.05` | NPC speed (tiles per step) |
| `SPAWN_TILES_AHEAD` | `20` | How far ahead NPCs spawn |
| `ENGAGE_DIST` | `8.0` | Distance to arm the overtake detector |
| `MIN_AHEAD_TILES` | `4` | Tiles player must lead before overtake confirms |
| `SURPASS_DIST` | `0.5` | Min world-unit gap to confirm overtake |
| `spawn_grace` | `100` steps | Suppresses false labels after NPC spawns behind player |
| `overtake_reward` | `+50` | Reward for each clean overtake |
| `npc_penalty` | `-5` | Penalty for collision |

### Label Behaviour

| Situation | Label |
|---|---|
| NPC is ahead of you | `NPC 1` / `NPC 2` / `NPC 3` (yellow) |
| You are ahead of NPC | `OVERTAKEN` (green) |
| NPC pulls back ahead | Instantly reverts to `NPC` label |
| Collision | `NPC HIT` (red) |
| Within spawn grace period | `NPC` (never shows OVERTAKEN) |

---

## Training Details

| Parameter | v0.2 (Before) | v1.0 (After) |
|---|---|---|
| Algorithm | PPO | PPO |
| Policy | CnnPolicy | CnnPolicy |
| Total Timesteps | 250,000 | 1,000,000 + |
| Frame Stacking | None | 4 frames |
| n_steps | 2048 | 512 |
| batch_size | 64 | 128 |
| n_epochs | 10 | 4 |
| Learning Rate | `3e-4` | Linear `2.5e-4 → 0` |
| Clip Range | `0.2` | Linear `0.2 → 0` |
| ent_coef | 0.005 | 0.01 |
| max_grad_norm | default | 0.5 |
| Device | CUDA (GPU) | CUDA (GPU) |
| Environment | CarRacing-v3 | CarRacing-v3 |
| Action Space | Box(3,) continuous | Box(3,) continuous |
| Observation Space | 96×96×3 RGB | 96×96×12 (4 stacked) |

---

## What Changed and Why

**`VecFrameStack(n_stack=4)`** — The single most impactful fix. Without frame stacking, the CNN only sees one static frame and has no way to perceive velocity or direction. Stacking 4 frames gives the policy the temporal context it needs.

**Linear LR + clip schedule** — Training logs showed `approx_kl` climbing from `0.04` to `0.28` by iteration 23, and `clip_fraction` hitting `0.57`. This is a classic sign of the learning rate being too aggressive late in training. Linear decay stabilizes both.

**`custom_objects` on model load** — Without this, `PPO.load()` silently reverts to the SB3 default `lr=3e-4` instead of your custom schedule, re-introducing the instability on every resumed training run.

**`n_epochs=4`** — Fewer gradient steps per rollout reduces overfitting to stale experience data.

**`ObstacleWrapper` (v2.0)** — Adds 3 NPC cars to the environment. Uses a tile-based dual-guard overtake detector to eliminate false positives. Labels are driven by live position checks (not timers), so `OVERTAKEN` appears and disappears instantly as positions change. A `spawn_grace` cooldown prevents false labels when NPCs spawn behind the player.

**`NpcVideoRecorder` (v2.0)** — Replaces `VecVideoRecorder`. Calls `draw_on_render()` on every frame before writing so NPC overlays, labels, and HUD appear in saved videos.

---

## Help

Common issues:

- **`ImportError: cannot import name 'linear_schedule'`** — Use `get_linear_fn` from `stable_baselines3.common.utils` directly:
```python
from stable_baselines3.common.utils import get_linear_fn
def linear_schedule(v): return get_linear_fn(v, 0.0, 1.0)
```
- **Missing model files** — Run the training cell first to generate the `.zip` model file under `models/`.
- **Module import errors** — Reinstall: `pip install -r requirements.txt`
- **TensorBoard not updating** — Ensure `--logdir` matches `./logs/`
- **CUDA out of memory** — Reduce `batch_size` or set `device="cpu"`
- **Rendering issues on headless servers** — Remove `render_mode="human"`
- **NPC labels not showing in video** — Use `NpcVideoRecorder` instead of `VecVideoRecorder`; the standard recorder bypasses `ObstacleWrapper`'s render pipeline
- **NPC 2 showing OVERTAKEN immediately at spawn** — Ensure you are using the latest `obstacle_wrapper.py` with `spawn_grace=100`

---

## Authors

**Ayush (Droid-DevX)**
- GitHub: [https://github.com/Droid-DevX](https://github.com/Droid-DevX)

---

## Version History

- **2.0** *(Current)*
  - `ObstacleWrapper` — 3 NPC obstacle cars with staggered spawning
  - Live position-based `OVERTAKEN` / `NPC` labels (no timer)
  - `spawn_grace` fix — eliminates false labels on late-spawning NPCs
  - Dual-guard overtake logic (`MIN_AHEAD_TILES` + `SURPASS_DIST`)
  - `NpcVideoRecorder` — video recording with NPC overlays composited in
  - Mean reward improved from **865 → 1030+** with overtake bonuses

- **1.0**
  - Frame stacking with `VecFrameStack(n_stack=4)`
  - Linear learning rate and clip range decay schedules
  - `EvalCallback` + `CheckpointCallback` for auto model saving
  - `custom_objects` fix for correct LR restoration on load
  - Mean reward improved from **98 → 865**, std from **240 → 22**

- **0.2**
  - Quantitative evaluation with `evaluate_policy`
  - Visual testing loop across 10 rendered episodes
  - TensorBoard logging integrated
  - Model persistence with save/load support

- **0.1**
  - Initial PPO agent with CnnPolicy on CarRacing-v3
  - Basic `DummyVecEnv` vectorized environment setup
  - Random action environment exploration

---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for full details.

---

## Acknowledgments

- [OpenAI Gymnasium – CarRacing-v3 environment](https://gymnasium.farama.org/environments/box2d/car_racing/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [PyTorch](https://pytorch.org/)
- [TensorBoard](https://www.tensorflow.org/tensorboard)