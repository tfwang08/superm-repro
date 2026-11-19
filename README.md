# superm-repro

An **unofficial, incremental reproduction** of [SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model](https://arxiv.org/abs/2610.28480) (Liao et al., 2026).

The goal is to reproduce the training **mechanism first**, then move from a small articulated system to Unitree G1 motion tracking. This repository is not the authors' implementation and does not yet reproduce their humanoid results.

## Current milestone: an inverted pendulum in MuJoCo

We start with a **classic cart-pole** in the real MuJoCo simulator: a cart can move horizontally, a single inverted pendulum pivots freely on it, and the **only action is horizontal force on the cart**. The first objective is upright balance from small initial perturbations. An optional second task tracks a slowly moving cart-position target while keeping the pole upright. This provides an underactuated, unstable system without needing G1 assets or mocap data.

The implemented training cycle is:

1. **Collect:** Run the current deterministic policy in MuJoCo. Add Gaussian noise to the executed actions and store simulator transitions in a replay buffer.
2. **Fit the world model:** Learn `f(s, a, xi) -> s_next` from **8-step autoregressive replay windows**, using the true simulator state only at the beginning of each window.
3. **Improve policy:** Freeze world-model weights, imagine **32 steps** with the current policy, and differentiate the discounted tracking return through all 32 learned state transitions. A separate critic predicts the remaining return beyond the horizon.
4. **Fit critic:** Compute GAE-style targets backward across imagined transitions, bootstrapping with the critic's final value; update the critic and repeat.

For the cart-pole experiment, observation, tracking reward, and soft termination are *analytic, differentiable functions* of the state. The neural world model learns dynamics only. This is allowed by the paper's Section 3.2, where each head may be learned or analytic. Actions are **normalized cart forces**, not the G1 position-setpoint actions in the paper; the environment and network widths are deliberately simpler than those in the humanoid experiments.

## Install and run

Requires Python 3.10+ and the MuJoCo Python package. No GPU is needed for the first milestone.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[sim,dev]"
python -m superm_repro.train_mujoco --iterations 30
pytest -q
```

Run the optional, more difficult cart-position tracking task:

```bash
python -m superm_repro.train_mujoco --motion track --iterations 30
```

Run a short wiring smoke test:

```bash
python -m superm_repro.train_mujoco \
  --iterations 2 --initial-episodes 4 --episode-steps 64 \
  --pretrain-model-updates 5 --model-updates 2 --batch-size 8 \
  --eval-episodes 2 --eval-steps 64 --device cpu
```

The trainer prints simulator reward, world-model loss, imagined return, and critic loss. It also evaluates the deterministic policy from fixed randomized starts and reports survival duration, normalized return, failure rate, and 8-step open-loop state MSE on held-out MuJoCo transitions. The run saves a checkpoint to `checkpoints/mujoco_cartpole.pt` by default. These are **diagnostic metrics**, not evidence of reproducing the paper's reported tracking scores.

**First measured result:** Across three CPU runs (40 iterations each), mean deterministic balance duration improved from **38.6 to 171.5 out of 250 steps**. All seeds improved, but two still had failures. The commands, per-seed results, and limitations are in [docs/CARTPOLE_RESULTS.md](docs/CARTPOLE_RESULTS.md).

## Roadmap

| Stage | Target | Exit criterion |
| --- | --- | --- |
| **1 — MuJoCo cart-pole (in progress)** | Learn upright balance with one controlled cart and one passive pole; optionally follow a small cart trajectory. | Tests pass, the seeded training loop runs, and real-simulator balance duration and model prediction error are measured. |
| **2 — Triple inverted pendulum** | Increase the number of unstable links after the cart-pole pipeline works. | Balance the chosen three-link setup and measure recovery after perturbations. |
| **3 — G1 simple motions** | Switch to Unitree G1 standing, shallow squats, and a small reference set; match the paper's observations, rewards, model widths, and domain randomization. | Stable motion tracking in MuJoCo with reportable returns and termination rates. |
| **4 — Scaling and evaluation** | Import filtered BONES-SEED motions and run multi-motion, perturbation, and throughput evaluations. | Report deterministic benchmark failures, normalized return, model error, and GPU-hours alongside paper baselines. |

See [docs/STAGES.md](docs/STAGES.md) for implementation details and testable acceptance criteria.

## Where the code lives

- `src/superm_repro/mujoco_cartpole.py`: underactuated cart-pole MuJoCo model, force action, balance reward, and termination.
- `src/superm_repro/collect.py`: on-policy simulator rollout and action-noised transition logging.
- `src/superm_repro/replay.py`: bounded episode replay, 8-step windows, and reference-aligned imagination starts.
- `src/superm_repro/models.py`: dynamics MLP, reference encoder + policy decoder, and critic.
- `src/superm_repro/training.py`: world-model fitting, differentiable imagined return, GAE targets, and optimizer updates.
- `src/superm_repro/train_mujoco.py`: executable three-phase training loop.
- `src/superm_repro/evaluation.py`: deterministic balancing and held-out world-model accuracy.

## What is not reproduced yet

The first experiment uses **one passive pendulum, one cart motor, and five state features**, not a 29-joint G1. It does not implement G1 floating-base dynamics, contact-dependent reward/observation, BONES-SEED replay, motion-specific reset sampling, domain randomization beyond joint damping, or multi-GPU training. The paper uses MuJoCo Warp for GPU-batched simulation; Stage 1 uses the standard MuJoCo Python simulator on CPU. No performance claim is made for G1.

## Paper references

- [SuperMimic paper](https://arxiv.org/pdf/2610.28480): training loop (Section 3.1), learned world model (Section 3.2, Eq. 1), policy and critic (Section 3.3, Eqs. 2–3), observation/action/reward (Appendix A), architectures (Appendix C), training algorithm and hyperparameters (Appendix D).
- [MuJoCo Python documentation](https://mujoco.readthedocs.io/en/latest/python.html): official simulator Python API.
