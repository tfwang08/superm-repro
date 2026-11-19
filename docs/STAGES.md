# Reproduction milestones and verification

This project separates **algorithm correctness**, **controlled instability**, **humanoid task fidelity**, and **benchmark reproducibility**. Success in an earlier stage does not establish the later stages.

## Stage 1: MuJoCo cart-pole balance

**Scope:** One freely hinged upright pendulum on a horizontally moving cart. The only actuator applies horizontal force to the cart. The initial reference asks for cart position zero and an upright pole; an optional `--motion track` command moves the cart target slowly. MuJoCo generates simulator labels; PyTorch dynamics replaces MuJoCo during policy improvement.

**Paper components to validate:**

- Deterministic policy with ten observed frames and ten future reference frames.
- Executed action noise recorded in replay, with a randomized joint-damping multiplier as simulator context.
- Contiguous `k=8` autoregressive world-model fitting.
- `H=32` imagination with gradients flowing through learned state transitions.
- Differentiable survival weights and a critic bootstrap after step 32.
- Critic targets with `gamma=0.99` and `lambda=0.95`.
- Repeated simulator collection, world-model fitting, policy update, and critic update.

**Verification:**

1. `pytest -q` passes, including a test that the unforced inverted pendulum falls away from upright.
2. `python -m superm_repro.train_mujoco --iterations 2 --initial-episodes 4 --episode-steps 64 --pretrain-model-updates 5 --model-updates 2 --batch-size 8 --eval-episodes 2 --eval-steps 64 --device cpu` completes with finite losses and a checkpoint.
3. Eight-step open-loop state error on held-out MuJoCo transitions decreases relative to the untrained world model.
4. Deterministic simulator balance duration and return improve over the initialized policy across several seeds. Report failures rather than inferring success from imagined return.
5. After balancing works, repeat with `--motion track` and evaluate cart-position tracking and pole angle separately.

## Stage 2: Triple-link inverted pendulum

**Scope:** Keep the same training algorithm but raise the control difficulty with three linked inverted rods. Specify and document actuator placement and termination before training; an underactuated cart-based variant is the eventual stress test.

**Verification:** Measure time until failure, angular tracking error, recovery from small disturbances, and model prediction error over the same horizon.

## Stage 3: Unitree G1 basic references

**Scope:** Use a licensed G1 MuJoCo asset, first for standing and shallow squats, then for short simple motion clips.

**Required changes:**

- Implement the G1 rigid-body model and normalized joint-setpoint PD actuator mapping.
- Implement the paper's joint/root state and 29-joint observation history, including IMU angular velocity, projected gravity, and the Appendix A.1 sensor-noise ranges.
- Implement differentiable forward kinematics and the body/root tracking reward of Appendix A.2 without the removed self-collision penalty.
- Implement height/tilt reference-relative termination and the corresponding soft survival signal.
- Use Appendix A context: friction, joint calibration offsets, and torso center-of-mass offsets.
- Match Appendix C model widths and Appendix D return-based reference reset sampler.
- Profile GPU batching and migrate collection to MuJoCo Warp when necessary.

**Verification:** Deterministic tracking returns, failure rates, recovery tests, and short-rollout model error on a fixed set of simple G1 references.

## Stage 4: Large-scale replication

**Scope:** Filter the public BONES-SEED G1 retargeting dataset, match the benchmark protocol, and train across many motions.

**Verification:** Report clip count, filtering rules, random seeds, GPU model, GPU-hours, simulator steps, normalized return, and failure count. Compare with the paper only when the evaluation protocols match Appendix F.

## Important limitations

The authors' G1 results do not apply to this repository. Stage 1 uses a low-dimensional, CPU MuJoCo cart-pole and cart-force actions; it cannot establish contact-rich humanoid tracking or transfer to a real robot.
