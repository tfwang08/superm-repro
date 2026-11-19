# superm-repro

An **unofficial, incremental reproduction** of [SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model](https://arxiv.org/abs/2610.28480) (Liao et al., 2026).

The goal is to reproduce the training **mechanism first**, then move from a small articulated system to Unitree G1 motion tracking. This repository is not the authors' implementation and does not yet reproduce their humanoid results.

## Canonical G1 reproduction: authors' open-source implementation

The authors have now [released SuperMimic](https://github.com/HybridRobotics/supermimic) with its [mjlab_research](https://github.com/HybridRobotics/mjlab_research) simulator dependency. **From this point on, our G1 experiments run their code directly**, pinned to exact upstream commits. Our own earlier G1 standing code is retained only as a historical teaching prototype.

```bash
bash scripts/official_setup.sh
bash scripts/official_train.sh SuperMimic-GenCoord-G1 \
  --motion-file /absolute/path/to/g1_reference_motions.npz \
  --runner.actor_obs proprio --runner.history_length 10 \
  --runner.max_iterations 320000
```

See [docs/OFFICIAL_REPRO.md](docs/OFFICIAL_REPRO.md) for the pinned revisions, Python 3.13/CUDA setup, reference-motion access, evaluation, and exact provenance. This wrapper invokes the authors' actual training runner, not our local G1 trainer.

## Existing milestone: inverted pendulums in MuJoCo

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

## Historical prototype: Unitree G1 standing (Stage 3.1)

The repository also includes a **floating-base, 29-actuator Unitree G1** with real MuJoCo foot contacts, a static standing target, and the same learned-world-model training algorithm. It uses a physics-only derivative of Google DeepMind's MuJoCo Menagerie G1 MJX model; visual meshes are removed, while the mass properties, collision primitives, PD actuators, and contact definitions are preserved. The upstream BSD-3-Clause license is included alongside the assets.

The G1 task uses 100 simulator-state features (36 positions, 35 velocities, and 29 previous normalized actions) and a **93-dimensional proprioceptive policy observation**, following the five groups and sensor-noise intervals in Appendix A.1. Actions are 29 joint setpoint offsets, and the static standing goal is 30-dimensional. All Appendix A.3 domain-randomization groups (friction, joint calibration, and torso COM) are sampled into a 33-dimensional context. A separate differentiable PyTorch forward-kinematics implementation supplies all Appendix A.2 body/root tracking-reward terms, action smoothness, and joint-limit penalties. Termination now follows the paper's reference-relative height/tilt thresholds. Action noise, 8-step world-model learning, differentiable actor rollouts, critic bootstrapping, and independent checkpoint selection are connected end-to-end.

The current implementation is **an executable first experiment, not an already solved humanoid standing controller**. A new two-iteration wiring test with the Appendix A interfaces completed but survived **0/2** held-out 96-step tests. Historical results from the prior simplified G1 protocol are not directly comparable. Time-varying full-body motion goals, the original reference-reset sampler, original-scale parallel data collection, and humanoid benchmark results have not been reproduced. See [docs/G1_STANDING.md](docs/G1_STANDING.md) for the exact G1 task definition, paper-alignment checklist, and experiment protocol.

## Install and run

Requires Python 3.10+ and the MuJoCo Python package. No GPU is needed for the first milestone.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[sim,dev]"
python -m superm_repro.train_mujoco --profile experiment --links 2 --iterations 40
pytest -q
```

To run the first G1 standing experiment:

```bash
python -m superm_repro.train_g1 --network-size small \
  --horizon 32 --iterations 200 --long-test
```

For a fast two-iteration G1 wiring test, see [docs/G1_STANDING.md](docs/G1_STANDING.md). `--network-size paper` selects the original Appendix C **MLP hidden widths**, while the default `small` profile reduces compute. `--horizon 200` is an optional longer-BPTT experiment, not a validated optimum. G1 simulator data collection runs in standard MuJoCo on CPU; the PyTorch networks run on the selected device.

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
| **3 — G1 whole-body tracking (official code)** | Clone the pinned authors' SuperMimic and mjlab_research revisions, train on licensed reference motions, and establish an unchanged official baseline. | Confirm full official training and deterministic evaluation before proposing ablations. |
| **4 — Scaling and evaluation** | Import filtered BONES-SEED motions and run multi-motion, perturbation, and throughput evaluations. | Report deterministic benchmark failures, normalized return, model error, and GPU-hours alongside paper baselines. |

See [docs/STAGES.md](docs/STAGES.md) for implementation details and testable acceptance criteria.

## Where the code lives

- `src/superm_repro/mujoco_cartpole.py`: 1/2/3-link underactuated MuJoCo models, force action, reference curricula, and termination.
- `src/superm_repro/mujoco_g1.py`: floating-base G1 with contact-aware MuJoCo dynamics, normalized PD joint-setpoint actions, and differentiable standing reward and termination.
- `src/superm_repro/g1_kinematics.py`: differentiable G1 body-position, quaternion, and spatial-velocity FK verified against MuJoCo.
- `src/superm_repro/assets/unitree_g1/`: BSD-licensed physics-only G1 MJCF and standing scene.
- `src/superm_repro/collect.py`: on-policy simulator rollout and action-noised transition logging.
- `src/superm_repro/replay.py`: bounded episode replay, 8-step windows, and reference-aligned imagination starts.
- `src/superm_repro/models.py`: dynamics MLP, reference encoder + policy decoder, and critic.
- `src/superm_repro/training.py`: world-model fitting, differentiable imagined return, GAE targets, and optimizer updates.
- `src/superm_repro/train_mujoco.py`: executable three-phase training loop.
- `src/superm_repro/train_g1.py`: G1 standing trainer with selectable network sizes, imagined horizon, checkpoint selection, and held-out evaluation.
- `src/superm_repro/evaluation.py`: deterministic balancing and held-out world-model accuracy.
- `src/superm_repro/compare.py`: matched four-seed baseline/experiment runs with independent validation and frozen long tests.

## What is not reproduced yet

The limitations below apply to **our independent legacy implementation**, not to the authors' official source linked above.

The cart-pole experiments remain small underactuated examples, and the G1 benchmark covers only **static standing**. Its policy now uses Appendix A's proprioception, body-tracking reward, and physical-parameter randomization; nevertheless, the world model uses raw simulator coordinates and the static standing reference does not represent the paper's time-varying motion targets. Full G1 motion imitation, BONES-SEED replay, return-based reset sampling, original actuator calibration, and multi-GPU training remain unimplemented. The paper uses MuJoCo Warp for GPU-batched simulation; this repository uses standard MuJoCo for simulator data collection. No humanoid success claim is made.

## Paper references

- [SuperMimic paper](https://arxiv.org/pdf/2610.28480): training loop (Section 3.1), learned world model (Section 3.2, Eq. 1), policy and critic (Section 3.3, Eqs. 2–3), observation/action/reward (Appendix A), architectures (Appendix C), training algorithm and hyperparameters (Appendix D).
- [MuJoCo Python documentation](https://mujoco.readthedocs.io/en/latest/python.html): official simulator Python API.
