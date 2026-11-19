# superm-repro

An **unofficial, incremental reproduction** of [SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model](https://arxiv.org/abs/2610.28480) (Liao et al., 2026).

The goal is to reproduce the training **mechanism first**, then move from a small articulated system to Unitree G1 motion tracking. This repository is not the authors' implementation and does not yet reproduce their humanoid results.

## Current milestone: an inverted pendulum in MuJoCo

We use a **MuJoCo cart-pole with one, two, or three passive inverted links**, driven by **one horizontal cart-force actuator**. The current experiment defaults to two links, and three links can be selected with `--links 3`. The task is intentionally underactuated; neither pole has its own motor.

The implemented training cycle is:

1. **Collect:** Run the current deterministic policy in MuJoCo. Add Gaussian noise to the executed actions and store simulator transitions in a replay buffer.
2. **Fit the world model:** Learn `f(s, a, xi) -> s_next` from **8-step autoregressive replay windows**, using the true simulator state only at the beginning of each window.
3. **Improve policy:** Freeze world-model weights, imagine **32 steps** with the current policy, and differentiate the discounted tracking return through all 32 learned state transitions. A separate critic predicts the remaining return beyond the horizon.
4. **Fit critic:** Compute GAE-style targets backward across imagined transitions, bootstrapping with the critic's final value; update the critic and repeat.

For the cart-pole experiment, observation, tracking reward, and soft termination are *analytic, differentiable functions* of the state. The neural world model learns dynamics only. This is allowed by the paper's Section 3.2, where each head may be learned or analytic. Actions are **normalized cart forces**, not the G1 position-setpoint actions in the paper; the environment and network widths are deliberately simpler than those in the humanoid experiments.

### Experimental policy and goal curriculum

The `experiment` profile adds three changes relative to the `baseline` profile:

1. **Reference-relative positions:** subtract the current target cart position from the ten observed cart positions and all ten future target positions before encoding them. Identical relative situations now have identical policy inputs.
2. **Exact left-right antisymmetry:** evaluate one shared network on the relative input and its mirror. Return half the difference of the two actions, so `policy(-x) = -policy(x)` for mirror-symmetric inputs.
3. **Step-target training:** once between control steps 15 and 50, switch the cart target by a random signed displacement of at most 1.15 m and hold it for the rest of the episode. The baseline instead trains on a 0.2 m slow sinusoidal reference.

Both profiles reuse the same world-model fitting, 32-step imagination, critic, and optimizer pipeline. The profiles can be compared with identical seeds, simulator budgets, checkpoint-selection rules, and long tests. See [docs/MULTILINK_EXPERIMENT.md](docs/MULTILINK_EXPERIMENT.md).

## Install and run

Requires Python 3.10+ and the MuJoCo Python package. No GPU is needed for the first milestone.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[sim,dev]"
python -m superm_repro.train_mujoco --profile experiment --links 2 --iterations 40
pytest -q
```

Try the three-link variant:

```bash
python -m superm_repro.train_mujoco --profile experiment --links 3 --iterations 40
```

For the full matched four-seed, 320-iteration comparison on the **same link count**:

```bash
python -m superm_repro.compare --links 1 --iterations 320 --episode-steps 256
```

The comparison runs eight independent trainings (four seeds per profile). Every 40 iterations it saves and validates a checkpoint; it selects the best using the same independent validation protocol and tests only that frozen checkpoint on a separate 96-episode repeated-reversal benchmark. The previously reported `0/96` versus `96/96` result is **not assumed to apply to this implementation** until independently reproduced.

Run a short wiring smoke test:

```bash
python -m superm_repro.train_mujoco \
  --profile experiment --links 2 \
  --iterations 2 --initial-episodes 4 --episode-steps 64 \
  --pretrain-model-updates 5 --model-updates 2 --batch-size 8 \
  --eval-episodes 2 --eval-steps 64 --checkpoint-every 1 --device cpu
```

The trainer prints simulator reward, world-model loss, imagined return, and critic loss. It records deterministic, independent validation metrics at saved checkpoints. The best validated checkpoint is copied to `checkpoints/mujoco_cartpole.pt` by default, with a machine-readable JSON report. Use `--long-test` to evaluate the selected frozen checkpoint on repeated direction changes.

**Earlier milestone:** At commit `7523978`, three single-link balance runs improved from 38.6 to 171.5 mean steps (250-step cap). This is historical single-link data and **does not describe the new step-target or multi-link experiment**. See [docs/CARTPOLE_RESULTS.md](docs/CARTPOLE_RESULTS.md).

## Roadmap

| Stage | Target | Exit criterion |
| --- | --- | --- |
| **1 — Single-link cart-pole** | Verify the full replay → world model → differentiable actor-critic pipeline. | Complete for basic training-loop verification; reproducibility metrics recorded. |
| **2 — Two/three-link pendulum (in progress)** | Learn step-target control with exact translation and reflection symmetries. | Tests pass; compare multiple seeds with independent checkpoint selection and repeated reversals. |
| **3 — G1 simple motions** | Switch to Unitree G1 standing, shallow squats, and a small reference set; match the paper's observations, rewards, model widths, and domain randomization. | Stable motion tracking in MuJoCo with reportable returns and termination rates. |
| **4 — Scaling and evaluation** | Import filtered BONES-SEED motions and run multi-motion, perturbation, and throughput evaluations. | Report deterministic benchmark failures, normalized return, model error, and GPU-hours alongside paper baselines. |

See [docs/STAGES.md](docs/STAGES.md) for implementation details and testable acceptance criteria.

## Where the code lives

- `src/superm_repro/mujoco_cartpole.py`: 1/2/3-link underactuated MuJoCo models, force action, reference curricula, and termination.
- `src/superm_repro/collect.py`: on-policy simulator rollout and action-noised transition logging.
- `src/superm_repro/replay.py`: bounded episode replay, 8-step windows, and reference-aligned imagination starts.
- `src/superm_repro/models.py`: dynamics MLP, reference encoder + policy decoder, and critic.
- `src/superm_repro/training.py`: world-model fitting, differentiable imagined return, GAE targets, and optimizer updates.
- `src/superm_repro/train_mujoco.py`: executable three-phase training loop.
- `src/superm_repro/evaluation.py`: deterministic balancing and held-out world-model accuracy.
- `src/superm_repro/compare.py`: matched four-seed baseline/experiment runs with independent validation and frozen long tests.

## What is not reproduced yet

These experiments use **one cart motor and up to three passive pendulums**, not a 29-joint G1. They do not implement G1 floating-base dynamics, contact-dependent reward/observation, BONES-SEED replay, motion-specific reset sampling, domain randomization beyond joint damping, or multi-GPU training. The paper uses MuJoCo Warp for GPU-batched simulation; we use standard MuJoCo on CPU. No humanoid performance claim is made.

## Paper references

- [SuperMimic paper](https://arxiv.org/pdf/2610.28480): training loop (Section 3.1), learned world model (Section 3.2, Eq. 1), policy and critic (Section 3.3, Eqs. 2–3), observation/action/reward (Appendix A), architectures (Appendix C), training algorithm and hyperparameters (Appendix D).
- [MuJoCo Python documentation](https://mujoco.readthedocs.io/en/latest/python.html): official simulator Python API.
