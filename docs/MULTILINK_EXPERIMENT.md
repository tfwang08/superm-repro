# Reference-relative and antisymmetric cart-pole experiments

The experimental variant is based on the training loop introduced at commit `7523978`. It keeps the 8-step autoregressive world-model fit, 32-step differentiable policy rollout, survival weighting, critic, and GAE. It changes the policy's coordinates and the training reference distribution.

## Three controlled changes

### 1. Translation-invariant policy inputs

Let `g_now` be the cart-position target in the first future reference frame. Every cart position in the observation history and future reference is shifted by `-g_now`. Joint angles, velocities, and previous actions are unchanged. The same policy can therefore reuse a learned maneuver at different absolute cart positions.

### 2. Exact reflection antisymmetry

For relative input `x` (history and goals) and the unmodified policy network `h`, the executed normalized cart force is:

```text
action(x) = (h(x) - h(-x)) / 2
```

The reflection negates cart positions, all relative pole angles, all generalized velocities, previous force, and target values. All of these variables are signed in the current cart-pole state. Therefore `action(-x) = -action(x)` algebraically, regardless of neural-network weights. The two passes share parameters.

### 3. Step-target curriculum

The experimental reference starts at cart target zero. One switch occurs uniformly between control steps 15 and 50. The new target is uniformly sampled in `[-1.15, 1.15]` meters and then held constant for the rest of the episode. Pole-angle targets always remain upright. This trains acceleration, travel, braking, and settling. The baseline trains on a slow 0.2-meter sinusoid.

## Increasing the system order

| Links | Generalized coordinates | State dimension | Action dimension | Goal dimension |
| ---: | --- | ---: | ---: | ---: |
| 1 | Cart position + 1 hinged pole | 5 | 1 | 2 |
| 2 | Cart position + 2 serial hinged poles | 7 | 1 | 3 |
| 3 | Cart position + 3 serial hinged poles | 9 | 1 | 4 |

Only the cart is actuated in all cases. The upright tracking reward uses the accumulated world-frame angle of each pole, so neighboring joint angles cannot cancel the loss of a pole's upright orientation. Evaluation terminates if any absolute pole angle deviates from upright by more than 0.55 rad, or if the cart exceeds 1.5 m of target-relative displacement.

## Fair checkpoint selection and testing

- Use the **same link count** for baseline and experiment.
- Use four training seeds: `0, 1, 2, 3`.
- Train each profile for **320 iterations**, with simulator episodes capped at **256 control steps**.
- Write a checkpoint every **40 iterations** (and at the final step if needed).
- Score each checkpoint on the **same independent validation scenarios**, separate from the training seeds. Select the checkpoint with the highest deterministic normalized return.
- Load only the selected checkpoint, freeze its weights, and run a **separate 96-episode repeated-reversal test**. The long test changes cart targets between `-0.6` and `+0.6` meters every 64 steps (starting at step 20), using a 512-step horizon.
- Record selected iteration, validation history, model error, reward, survival length, and success count in JSON. Do not choose or tune a checkpoint using long-test results.

Run the complete matched single-link comparison:

```bash
python -m superm_repro.compare --links 1 --iterations 320 --episode-steps 256
```

Scale the **same experiment** to two or three links with `--links 2` or `--links 3`. A small wiring test is:

```bash
python -m superm_repro.train_mujoco --profile experiment --links 2 \
  --iterations 2 --initial-episodes 4 --episode-steps 64 \
  --pretrain-model-updates 5 --model-updates 2 --batch-size 8 \
  --eval-episodes 2 --eval-steps 64 --checkpoint-every 1
```

The user-reported `0/96` versus `96/96` difference motivated the changes; it is **not yet a measured result for this implementation**, whose long-test details are now explicit. A subsequent ablation should switch relative inputs, reflection antisymmetry, and the reference curriculum independently to establish each contribution.

Individual ablations are supported through `--no-relative-goal`, `--no-antisymmetric`, and `--motion track`. These overrides preserve the remaining `experiment` settings, so each mechanism can be disabled without changing the neural architecture or the training budget.

## Initial two-link pilot (not a solved benchmark)

A limited smoke-and-learning run used seed `0`, two links, 120 training iterations, 256-step training episodes, 16 initial episodes, 100 initial model updates, 10 model updates per iteration, and batch size 32. Checkpoints were independently validated at iterations 40, 80, and 120 on eight 256-step episodes. The selected checkpoint was iteration 120. Its validation mean survival was **66.9/256 steps**, and its normalized return was **0.180**. On 16 held-out repeated-reversal episodes of up to 320 steps, **0/16 succeeded**, with mean survival **74.1 steps** and eight-step world-model state MSE approximately **0.00031**.

This confirms the extended pipeline executes and learns a dynamics surrogate; it does **not** establish stable two-link balancing or reversal control. The full four-seed, 320-iteration protocol remains to be run. Learned dynamics accuracy by itself is not evidence of a successful policy.
