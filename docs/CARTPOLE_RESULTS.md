# Stage 1: MuJoCo cart-pole results

This log records actual local training runs, not projected performance. The results establish that the initial algorithm wiring can improve control of an underactuated inverted pendulum. They **do not** reproduce the paper's humanoid benchmark.

## Evaluation protocol

- Simulator: MuJoCo 3.3.7 (CPU), control period 20 ms; Python 3.12; PyTorch 2.15.0 (CPU execution).
- Initial condition: cart at target; small randomized pole angle. Evaluation uses eight deterministic episodes for each policy, with the same random seed and zero sensor/action noise.
- Evaluation context: damping multiplier 1.0, no external pushes; survival ends when pole tilt exceeds 0.55 rad or cart displacement exceeds 1.5 m.
- Score: total actual simulator reward divided by the full 250-step evaluation budget; steps after early termination contribute zero. A perfect score is at most 1.0.
- World-model metric: mean per-state-feature MSE of 8-step open-loop rollouts on evaluation transitions, with the true initial state and the logged actions. It is measured on each policy's own visited states.

## Training setup

| Parameter | Value |
| --- | ---: |
| Random seeds | 0, 1, 2 |
| Training iterations | 40 |
| Initial simulator episodes | 16 |
| Maximum simulator episode length | 128 steps |
| World-model warm-start updates | 100 |
| World-model updates per iteration | 10 |
| World-model window `k` | 8 |
| Imagination horizon `H` | 32 |
| Actor/model batch size | 32 |
| `gamma`, GAE `lambda` | 0.99, 0.95 |
| PyTorch intra-op threads | 1 |
| Evaluation episodes / horizon | 8 / 250 steps |

## Measured results

| Seed | Balance steps before | Balance steps after | Return before | Return after | Failure rate after | Final model MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 41.75 | 174.38 | 0.142 | 0.554 | 75.0% | 0.000855 |
| 1 | 45.13 | 250.00 | 0.156 | 0.918 | 0.0% | 0.000267 |
| 2 | 28.88 | 90.25 | 0.094 | 0.287 | 100.0% | 0.001241 |
| **Mean** | **38.58** | **171.54** | **0.131** | **0.586** | **58.3%** | — |

All three policies improved their deterministic real-simulator survival time and return after 40 iterations. Seed 1 balanced the full 250-step horizon in all eight evaluation episodes. Seeds 0 and 2 still failed frequently. This is a useful first milestone, not yet a robust cart-pole controller.

## Reproduction command

Install using the root `README.md`, then run three independent seeds:

```bash
python -m superm_repro.train_mujoco \
  --motion balance --iterations 40 --initial-episodes 16 \
  --episode-steps 128 --pretrain-model-updates 100 \
  --model-updates 10 --batch-size 32 \
  --eval-episodes 8 --eval-steps 250 --eval-every 20 \
  --seed 0 --torch-threads 1 --device cpu
```

Repeat with `--seed 1` and `--seed 2`. Checkpoints are ignored by Git and are not included in the repository.

## Next acceptance criterion

Improve robustness so that all three seeds balance the full 250-step evaluation horizon with low failure rates, while keeping the learned world model's short-rollout error low. Report both physical-simulator and imagined rewards to detect model exploitation.
