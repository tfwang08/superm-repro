# Stage 3.1: Unitree G1 standing in MuJoCo

This is the first humanoid implementation of the existing SuperMimic-style training loop. The standing task now uses the paper's Appendix A observation layout, tracking-reward terms, reference-relative termination conditions, and domain-randomization ranges. It is still **a static-reference standing benchmark**, not a reproduction of multi-motion training, the published humanoid training budget, or the deployed controller.

## Physical model and license

The bundled scene is a physics-only derivative of Google DeepMind's [MuJoCo Menagerie Unitree G1 MJX model](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1). It preserves the original 29 actuated joints, floating base, rigid-body inertias, primitive collision geometry, foot/ground contact pairs, PD joint-position actuators, and `home` keyframe. Only visual meshes were removed, so the example runs without downloading external `.STL` files. The upstream BSD-3-Clause license is reproduced in `src/superm_repro/assets/unitree_g1/LICENSE`.

The bundled model contains 36 generalized position coordinates, 35 generalized velocities, and 29 position actuators. It runs at a 4 ms physics timestep and a 20 ms policy timestep. You can optionally provide a fully downloaded Menagerie scene with `--scene /path/to/unitree_g1/scene_mjx.xml` to restore the original visual meshes.

## Task definition

| Quantity | Dimension | Definition |
| --- | ---: | --- |
| Simulator state | 100 | 36 `qpos`, 35 `qvel`, 29 previous normalized actions |
| Policy observation | 93 | 29 relative joint positions, 29 joint velocities, 3 base gyro, 3 projected gravity, 29 previous actions |
| Action | 29 | Normalized delta to the nominal `home` joint setpoints |
| Goal | 30 | Nominal pelvis height and 29 nominal joint positions |
| Simulation context | 33 | Foot friction, 29 joint calibration offsets, and 3 torso center-of-mass offsets |
| History and future reference | 10 frames each | The same history/goal interface used for the cart-poles |

The simulator executes `ctrl = clip(q_home + calibration_offset + 0.35 * action, joint_position_limits)` through the G1's built-in joint-position PD actuators. It logs the *executed normalized policy action with collection noise* and the resulting MuJoCo transition. The scale `0.35` and the Menagerie actuator PD gains are engineering choices for this standing benchmark, not verified copies of the paper's actuator calibration.

At reset, the robot starts around the Menagerie `home` standing keyframe with small randomized root tilt, joint offsets, and velocities. The goal is fixed during each episode. The reference body positions and orientations are computed at the standing keyframe in MuJoCo.

The differentiable standing reward implements **all eight rows of Appendix A.2**: body position, orientation, linear velocity, angular velocity, root position, root orientation, action smoothness, and joint-limit penalties, with the paper's coefficients and the `0.02 s` control-period multiplier. Body pose and velocity come from a separate **pure PyTorch forward-kinematics implementation** (all 30 non-world G1 bodies), so actor gradients do not pass through MuJoCo. Its body positions and orientations agree with MuJoCo to approximately `1e-7` on a perturbed test pose, and its linear velocities agree with MuJoCo finite differences within test tolerance. The reference is still the fixed standing pose; general motion-reference ingestion is not implemented.

Termination follows Appendix A: `abs(root_height - reference_height) > 0.25 m` or a projected-gravity z discrepancy greater than `0.8`. There is no separate end-effector or horizontal-distance failure condition. Imagined trajectories use a differentiable sigmoid relaxation of these thresholds.

Appendix A.1 observation noise is uniform per component: joint positions `±0.01 rad`, joint velocities `±0.5 rad/s`, base gyro `±0.2 rad/s`, and projected gravity `±0.05`, with no noise on the previous action. The world model still sees the full MuJoCo state and recorded simulator context; the actor sees only the 93-dimensional proprioceptive observation. The context uses Appendix A.3's friction `U[0.3, 1.2]`, joint calibration offsets `U[-0.01, 0.01]` rad, and torso COM offsets `±(0.025, 0.05, 0.05)` m. Frozen deterministic evaluation uses nominal calibration/COM and median friction.

Foot contacts are computed by MuJoCo and learned indirectly through the state-transition model; the reward, observations, and termination are analytic functions of predicted state. The learned world model also differentiably projects the four root-quaternion values back to unit length after every predicted transition.
The previous-action components of the next state are copied exactly from the executed action in the world model, instead of being unnecessarily regressed by the dynamics MLP. Both analytic operations remain differentiable through the policy input.

## Training

Install the project with MuJoCo, then run a short wiring test:

```bash
python -m pip install -e ".[sim,dev]"
python -m superm_repro.train_g1 \
  --network-size tiny --iterations 2 \
  --initial-episodes 4 --episode-steps 96 \
  --pretrain-model-updates 5 --model-updates 2 --batch-size 4 \
  --eval-episodes 2 --eval-steps 96 --checkpoint-every 1 \
  --long-test --long-test-episodes 2 --long-test-steps 96
```

Use the default CPU-friendly experimental widths for an initial standing run:

```bash
python -m superm_repro.train_g1 --network-size small \
  --horizon 32 --iterations 200 --long-test
```

The `--network-size paper` option restores Appendix C's hidden widths: dynamics `(2048, 2048, 2048)`, motion-goal encoder and action decoder `(2048, 1024, 512, 256)`, and critic `(256, 128)`. The smaller `small` and `tiny` profiles preserve the same network *types* and training logic, but do **not** match the paper's capacity. `paper` is memory- and compute-intensive without a GPU.

The BPTT horizon is controlled independently by `--horizon` and defaults to **32**. For a controlled comparison, train a separate seed-matched run with `--horizon 200` while holding the network width, simulator budget, validation set, and optimizer updates fixed. Longer horizons are not assumed to be better or more accurate.

The trainer collects MuJoCo transitions into replay, fits one learned dynamics model over **8-step** windows, differentiates the actor objective through **H** imagined steps, and fits a separate GAE-bootstrap critic. It writes a candidate checkpoint at each validation interval, selects the best on independently seeded deterministic standing episodes, freezes it, and optionally evaluates it on a held-out seed. Reports are saved as JSON beside the checkpoint.

## Current status and limitations

- The bundled 29-DOF model loads and simulates floor contacts in real MuJoCo.
- The 8-step replay fit, differentiable actor rollout, critic update, checkpoint selection, and held-out evaluation execute end-to-end.
- With Appendix A's observation, reward, and termination changes, a two-iteration `tiny` G1 smoke test completed, including independent validation and frozen long testing, with **0/2** held-out 96-step episodes surviving. This verifies execution, **not** learned balance.
- An earlier 20-iteration `small` pilot and H=200 wiring run used the **previous, simplified standing reward and privileged observation**. Their measurements must not be compared numerically with the new Appendix A protocol.
- The first goal is to achieve consistent 10-second standing from randomized small perturbations. Training convergence and multiple-seed reliability remain unverified.

Unlike the paper's full motion tracker, the goal interface is a fixed standing keyframe rather than time-varying multi-body motion targets. The dynamics model represents raw 100-dimensional MuJoCo generalized coordinates, rather than matching any original state canonicalization, and uses explicit quaternion normalization. There is no return-weighted reference-reset distribution, BONES-SEED motion dataset, 4096-parallel-environment MuJoCo Warp collection, original total training duration, cosine learning-rate schedule, or hardware deployment. The observation layout and injected noise match Appendix A.1, but real sensor timing/filtering and actuator calibration on hardware are not reproduced. Training convergence and multiple-seed robustness remain unverified.
