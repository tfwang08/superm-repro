# Stage 3.1: Unitree G1 standing in MuJoCo

This is the first humanoid implementation of the existing SuperMimic-style training loop. It is a **standing benchmark**, not a reproduction of the paper's full-body motion-tracking reward, humanoid training budget, or deployment controller.

## Physical model and license

The bundled scene is a physics-only derivative of Google DeepMind's [MuJoCo Menagerie Unitree G1 MJX model](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1). It preserves the original 29 actuated joints, floating base, rigid-body inertias, primitive collision geometry, foot/ground contact pairs, PD joint-position actuators, and `home` keyframe. Only visual meshes were removed, so the example runs without downloading external `.STL` files. The upstream BSD-3-Clause license is reproduced in `src/superm_repro/assets/unitree_g1/LICENSE`.

The bundled model contains 36 generalized position coordinates, 35 generalized velocities, and 29 position actuators. It runs at a 4 ms physics timestep and a 20 ms policy timestep. You can optionally provide a fully downloaded Menagerie scene with `--scene /path/to/unitree_g1/scene_mjx.xml` to restore the original visual meshes.

## Task definition

| Quantity | Dimension | Definition |
| --- | ---: | --- |
| Simulator state | 100 | 36 `qpos`, 35 `qvel`, 29 previous normalized actions |
| Noisy policy observation | 100 | Privileged full state with small position and velocity noise |
| Action | 29 | Normalized delta to the nominal `home` joint setpoints |
| Goal | 30 | Nominal pelvis height and 29 nominal joint positions |
| Simulation context | 1 | Per-episode scaling of explicit contact-pair friction |
| History and future reference | 10 frames each | The same history/goal interface used for the cart-poles |

The simulator executes `ctrl = clip(q_home + 0.35 * action, joint_position_limits)` through the G1's built-in joint-position PD actuators. It logs the *executed normalized policy action with collection noise* and the resulting MuJoCo transition.

At reset, the robot starts around the Menagerie `home` standing keyframe with small randomized root tilt, joint offsets, and velocities. The goal is fixed during each episode. The differentiable reward encourages an upright pelvis, correct pelvis height, low horizontal drift, nominal joint posture, and low base velocity. A hard failure occurs if pelvis height drops below 0.55 m, tilt exceeds about 0.7 rad, or horizontal drift exceeds 0.75 m. Imagined trajectories use a smooth soft-termination version of the same criteria.

Foot contacts are computed by MuJoCo and learned indirectly through the state-transition model; the reward, observations, and termination are analytic functions of predicted state. The learned world model also differentiably projects the four root-quaternion values back to unit length after every predicted transition.

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
- A two-iteration `tiny` smoke test completed; **0/2** held-out 96-step episodes survived. This is evidence of correct execution, **not** learned balance.
- A 20-iteration `small` pilot (seed 0, 16 initial episodes, 60 world-model pretraining updates, five world-model updates per iteration, batch size 12) decreased the held-out eight-step model-state MSE from **0.135** initially to **0.016** at the selected checkpoint. Validation survival was only **50.6/500 control steps**, and the frozen checkpoint survived **0/8** independent 500-step tests (mean survival **58.4 steps**). This diagnostic run did not solve standing.
- A one-iteration `tiny` run with `--horizon 200` also completed the differentiable update, but the critic loss was extremely large on its immature world model. Long-horizon numerical stability and rollout bias require systematic study; this was a wiring test, not an H=200 performance result.
- The first goal is to achieve consistent 10-second standing from randomized small perturbations. Training convergence and multiple-seed reliability remain unverified.

The policy still observes privileged full generalized coordinates; it does **not** yet use the paper's deployable proprioceptive observations or realistic IMU/actuator calibration errors. The reward is a simplified standing objective, not Appendix A's full body-tracking reward. Domain randomization currently covers foot friction only, not torso center of mass or joint calibration offsets. There is no return-weighted reset distribution, BONES-SEED motion dataset, multi-GPU simulator, motion tracking, or hardware deployment. These are the next Stage 3 milestones.
