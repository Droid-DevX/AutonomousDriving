# Reinforcement Learning – CarRacing-v3 with Stable-Baselines3

A reinforcement learning project for training agents to navigate the continuous control driving environment **CarRacing-v3** using a CNN-based policy with the PPO algorithm.

---

##  Before vs After

| | **Before (v0.2)** | **After (v1.0)** |
|---|---|---|
| **Mean Reward** | 98.57 | **865.44** |
| **Std Deviation** | 240.10 | **22.31** |
| **Variance** | 57,649 | **~498** |
| **Reward Range** | -4.76 → 814.97 | 845 → 906 |
| **Episodes Solved** | 0 / 10 | **5 / 5** |
| **Training Steps** | 250,000 | 1,000,000 + |
| **Frame Stacking** |  None |  4 frames |
| **Learning Rate** |  `3e-4` | Linear decay `2.5e-4 → 0` |
| **Clip Range** | `0.2` | Linear decay `0.2 → 0` |
| **n_epochs** | Default (10) | 4 |
| **Monitor Wrapper** |  Missing on eval env |  All envs |
| **KL Divergence** | Blew up to `0.28` | Stable `< 0.02` |
| **Value Loss** | Spiked to `11.6` | Stable and declining |

---



## Description

This project implements an end-to-end reinforcement learning pipeline for training and evaluating an agent on the **CarRacing-v3** environment from Gymnasium. It applies the **PPO (Proximal Policy Optimization)** algorithm using Stable-Baselines3 with a `CnnPolicy` to process raw pixel observations directly from the top-down race track.

The optimized pipeline includes:

- **Frame stacking** (`VecFrameStack`, n=4) — gives the policy temporal context (speed, direction) from consecutive frames
- **Linear learning rate + clip range decay** — prevents KL divergence explosion and policy instability during long training runs
- **Vectorized environment** via `DummyVecEnv` + `VecTransposeImage` for efficient rollout collection
- **CNN-based policy** (`CnnPolicy`) operating on raw 96×96 pixel image observations
- **Continuous action space**: steering `[-1, 1]`, throttle `[0, 1]`, brake `[0, 1]`
- **EvalCallback + CheckpointCallback** — auto-saves best model and periodic checkpoints during training
- **Model save/reload** with `custom_objects` to correctly restore learning rate schedules
- **Quantitative evaluation** using mean/std/confidence interval over deterministic episodes
- **Visual rendering** of test episodes using `render_mode="human"`

---

## Achieved Results

### Before (v0.2 — 250k steps, no frame stacking, fixed LR)
```
Episode Rewards: [-2.03, 4.32, 2.56, 58.67, 14.20, 33.52, 68.25, -3.97, 814.97, -4.76]
Mean Reward   : 98.57
Std Deviation : 240.10
Variance      : 57,649.78
```

### After (v1.0 — optimized hyperparameters + frame stacking + LR schedule)
```
Episode 1 finished | Total Reward: 906.70
Episode 2 finished | Total Reward: 845.02
Episode 3 finished | Total Reward: 849.83
Episode 4 finished | Total Reward: 869.28
Episode 5 finished | Total Reward: 856.38

Mean Reward   : 865.44
Std Deviation : 22.31
```

> CarRacing-v3 is considered **solved** at a mean reward of 900+ over 100 episodes. The agent is approaching this threshold consistently.

---

## Getting Started

### Dependencies

- Python 3.x
- Required libraries listed in `requirements.txt` (including Stable-Baselines3, Gymnasium, PyTorch, TensorBoard)
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
```

3. Ensure the following files are present after training:
   - `models/ppo_car_racing.zip` – saved PPO model
   - `models/best/` – best model auto-saved during training
   - `logs/` – TensorBoard training logs

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

4. **Watch the agent drive:**
```bash
# Run the "Visual Test" cell with render_mode="human"
```

5. **Monitor training with TensorBoard:**
```bash
tensorboard --logdir ./logs
```

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
| Learning Rate |  `3e-4` | Linear `2.5e-4 → 0` |
| Clip Range |  `0.2` | Linear `0.2 → 0` |
| ent_coef | 0.005 | 0.01 |
| max_grad_norm | default | 0.5 |
| Device | CUDA (GPU) | CUDA (GPU) |
| Environment | CarRacing-v3 | CarRacing-v3 |
| Action Space | Box(3,) continuous | Box(3,) continuous |
| Observation Space | 96×96×3 RGB | 96×96×12 (4 stacked) |

---

## What Changed and Why

**`VecFrameStack(n_stack=4)`** — The single most impactful fix. Without frame stacking, the CNN only sees one static frame and has no way to perceive velocity or direction of movement. Stacking 4 frames gives the policy the temporal context it needs.

**Linear LR + clip schedule** — Training logs showed `approx_kl` climbing from `0.04` to `0.28` by iteration 23, and `clip_fraction` hitting `0.57`. This is a classic sign of the learning rate being too aggressive late in training. Linear decay stabilizes both.

**`custom_objects` on model load** — Without this, `PPO.load()` silently reverts to the SB3 default `lr=3e-4` instead of your custom schedule, re-introducing the instability on every resumed training run.

**`n_epochs=4`** — Fewer gradient steps per rollout reduces overfitting to stale experience data.

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

---

## Authors

**Ayush (Droid-DevX)**
- GitHub: [https://github.com/Droid-DevX](https://github.com/Droid-DevX)

---

## Version History

- **1.0** *(Current)*
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