# Reproduction milestones and verification

This project separates **algorithm correctness**, **humanoid task fidelity**, and **benchmark reproducibility**. Success in an earlier stage does not establish the later stages.

## Stage 1: Real MuJoCo, simple joint-reference motion

**Scope:** A two-joint actuated arm tracks smooth reference joint positions at 50 Hz. MuJoCo generates the simulator labels; a PyTorch dynamics model replaces MuJoCo during policy improvement.

**Paper components to validate:**

- Deterministic policy with ten observed frames and ten future reference frames.
- Executed action noise recorded in replay, with one randomized physical-context variable (joint damping).
- Contiguous `k=8` autoregressive world-model fitting.
- `H=32` imagination with gradient flow through every learned state transition.
- Differentiable survival weights and a bootstrap value at step 32.
- Critic targets with `gamma=0.99` and `lambda=0.95`.
- Repeated collection, model fitting, actor update, and critic update in one process.

**Verification:**

1. `pytest -q` must pass, including MuJoCo state/transition checks.
2. The `--iterations 2` smoke test must finish with finite metrics and a checkpoint.
3. Compare 8-step open-loop state error on held-out MuJoCo transitions against the untrained model; the trained model should be better.
4. Compare deterministic MuJoCo tracking return against the initialization over several seeds. Record failure cases instead of inferring success from imagined return.

## Stage 2: Unitree G1, static and shallow-squat references

**Scope:** Use a licensed G1 MuJoCo asset and a minimal reference clip for balancing and a small squat. No external motion dataset is necessary yet.

**Required changes:**

- Integrate the G1 rigid-body model and position-setpoint PD actuator mapping.
- Replace the toy state with full joint/root states and previous actions.
- Implement paper observation history, including IMU angular velocity and projected gravity, with the Appendix A.1 sensor-noise ranges.
- Implement differentiated forward kinematics and the body/root tracking reward from Appendix A.2, excluding the paper's removed self-collision penalty.
- Implement height/tilt reference-relative termination and soft imagination survival.
- Log separate real-simulator reward and predicted-world-model reward.

**Verification:** The actor runs deterministically in MuJoCo without premature termination on held-out stand and squat references; report errors and short-rollout model accuracy.

## Stage 3: Match the paper on a small motion set

**Scope:** Full G1 tracking observations, domain randomization, and a modest set of simple motion clips.

**Required changes:**

- Match Appendix C model widths: dynamics `(2048, 2048, 2048)`, motion encoder `(2048, 1024, 512, 256)`, decoder `(2048, 1024, 512, 256)`, critic `(256, 128)`.
- Use the Appendix A simulator context: ground friction, joint calibration offsets, torso center-of-mass offsets.
- Implement clipped/normalized state features, body tracking and action regularizers, and randomized reference start points.
- Implement Appendix D return-based reset sampler (uniform, low-return softmax, replay-history mix).
- Profile GPU batching and migrate collection to MuJoCo Warp when needed.

**Verification:** Motion-specific return, robustness to disturbances, 8-step prediction errors, and model throughput on a fixed evaluation set.

## Stage 4: Large-scale replication

**Scope:** Filter the public BONES-SEED G1 retargeting dataset, match the benchmark protocol, and train across many motions.

**Verification:** Report the clip count, filtering rules, seeds, GPU model, GPU-hours, simulator steps, normalized return, and failure count. Do not directly compare hardware or results unless their evaluation protocols match Appendix F.

## Important limitations

The authors' reported G1 results are not a claim about this repository. Stage 1 uses a low-dimensional CPU MuJoCo arm on purpose; even a perfect Stage 1 model would not demonstrate contact-rich humanoid tracking or transfer to a real robot.
