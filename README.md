# superm-repro

An **unofficial, incremental reproduction** of [SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model](https://arxiv.org/abs/2610.28480) (Liao et al., 2026).

The goal is to reproduce the training **mechanism first**, then move from a small articulated system to Unitree G1 motion tracking. This repository is not the authors' implementation and does not yet reproduce their humanoid results.

## Current milestone: simple motion tracking in MuJoCo

We start with a real MuJoCo simulation of a two-joint arm tracking a smooth reference sequence. This tests the complete algorithm without needing G1 assets or a motion-capture dataset.

The implemented training cycle is:

1. **Collect:** Run the current deterministic policy in MuJoCo. Add Gaussian noise to the executed actions and store simulator transitions in a replay buffer.
2. **Fit the world model:** Learn `f(s, a, xi) -> s_next` from **8-step autoregressive replay windows**, using the true simulator state only at the beginning of each window.
3. **Improve policy:** Freeze world-model weights, imagine **32 steps** with the current policy, and differentiate the discounted tracking return through all 32 learned state transitions. A separate critic predicts the remaining return beyond the horizon.
4. **Fit critic:** Compute GAE-style targets backward across imagined transitions, bootstrapping with the critic's final value; update the critic and repeat.

For the small arm experiment, observation, tracking reward, and soft termination are *analytic, differentiable functions* of the state. The neural world model learns dynamics only. This is allowed by the paper's Section 3.2, where each head may be learned or analytic. The arm task and network widths are deliberately simpler than those in the G1 experiments.

## Install and run

Requires Python 3.10+ and the MuJoCo Python package. No GPU is needed for the first milestone.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[sim,dev]"
python -m superm_repro.train_mujoco --iterations 30
pytest -q
```

Run a short wiring smoke test:

```bash
python -m superm_repro.train_mujoco \
  --iterations 2 --initial-episodes 4 --episode-steps 64 \
  --model-updates 2 --batch-size 8 --device cpu
```

The trainer prints simulator reward, world-model loss, imagined return, and critic loss. It saves a checkpoint to `checkpoints/mujoco_joint.pt` by default. These are **diagnostic metrics**, not evidence of reproducing the paper's reported tracking scores.

## Roadmap

| Stage | Target | Exit criterion |
| --- | --- | --- |
| **1 — MuJoCo arm (in progress)** | Verify the full three-phase loop on a simple, smooth two-joint tracking motion. | Automated tests pass; a seeded MuJoCo training run completes; inspect 8-step prediction error and simulator reward. |
| **2 — G1 standing / shallow squat** | Replace the arm with Unitree G1, PD joint setpoints, realistic observations, and a short hand-authored reference. | Stable reference tracking in MuJoCo; correct reward and termination; no G1 hardware required. |
| **3 — Paper-aligned training** | Add full G1 state, sensor/history layout, friction/calibration/CoM context, BeyondMimic reward, return-based resets, and model widths. | Reproduce Section 3 and Appendix A–D design and report comparable metrics on a small motion set. |
| **4 — Scaling and evaluation** | Import filtered BONES-SEED motions and run multi-motion, perturbation, and throughput evaluations. | Report deterministic benchmark failures, normalized return, model error, and GPU-hours alongside paper baselines. |

See [docs/STAGES.md](docs/STAGES.md) for implementation details and testable acceptance criteria.

## Where the code lives

- `src/superm_repro/mujoco_joint.py`: two-link MuJoCo model, PD setpoint actions, tracking reward and termination.
- `src/superm_repro/collect.py`: on-policy simulator rollout and action-noised transition logging.
- `src/superm_repro/replay.py`: bounded episode replay, 8-step windows, and reference-aligned imagination starts.
- `src/superm_repro/models.py`: dynamics MLP, reference encoder + policy decoder, and critic.
- `src/superm_repro/training.py`: world-model fitting, differentiable imagined return, GAE targets, and optimizer updates.
- `src/superm_repro/train_mujoco.py`: executable three-phase training loop.

## What is not reproduced yet

The first experiment uses **two arm joints**, not 29 G1 joints. It does not implement G1 floating-base dynamics, contact-dependent reward/observation, BONES-SEED replay, motion-specific reset sampling, domain randomization beyond joint damping, or multi-GPU training. The paper uses MuJoCo Warp for GPU-batched simulation; Stage 1 uses the standard MuJoCo Python simulator on CPU. No performance claim is made for G1.

## Paper references

- [SuperMimic paper](https://arxiv.org/pdf/2610.28480): training loop (Section 3.1), learned world model (Section 3.2, Eq. 1), policy and critic (Section 3.3, Eqs. 2–3), observation/action/reward (Appendix A), architectures (Appendix C), training algorithm and hyperparameters (Appendix D).
- [MuJoCo Python documentation](https://mujoco.readthedocs.io/en/latest/python.html): official simulator Python API.
