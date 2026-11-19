# Official SuperMimic reproduction path

The authors published their source at [HybridRobotics/supermimic](https://github.com/HybridRobotics/supermimic), under Apache-2.0. **This is now the canonical G1 implementation for our experiments.** We no longer port the authors' humanoid algorithm into the local `src/superm_repro` package. That package remains available for the earlier independent cart-pole experiments and the historical G1 standing prototype.

## Locked upstream revisions

| Repository | Pinned commit | Purpose |
| --- | --- | --- |
| [HybridRobotics/supermimic](https://github.com/HybridRobotics/supermimic) | `698ea39bc901d8fe733b3c8621df3199d241f7e4` | World model, actor/critic, replay, reset sampling, runner, evaluation |
| [HybridRobotics/mjlab_research](https://github.com/HybridRobotics/mjlab_research) | `6ea3e574b08605b79a290a6d464a812a26acd6fd` | MuJoCo Warp environment, G1 MJCF, control gains, actions, tracking rewards |

The scripts put both sources under `.upstream/supermimic_ws/` as siblings, as required by the upstream `pyproject.toml` (`mjlab = ../mjlab_research`). The checkouts are ignored by git and are never copied into this repo. Running setup again will **not overwrite** an existing checkout at a different commit.

## 1. Prepare the upstream environment

Prerequisites: Linux with a CUDA-capable NVIDIA GPU for paper-scale training, git, [uv](https://docs.astral.sh/uv/), a working network connection, and storage for motion references. Upstream requires Python **3.13**; `uv` resolves the upstream project environment. Our legacy `superm-repro` package uses its own smaller dependency set.

```bash
bash scripts/official_setup.sh
cd .upstream/supermimic_ws/supermimic
uv sync
cd ../../..
```

The setup script checks out exact commits, without modifying existing user work. It does not download motion data or launch training.

## 2. Run the authors' G1 training code

The upstream script is `src/supermimic/scripts/train.py`. Its registered G1 tasks are `SuperMimic-GenCoord-G1` and `SuperMimic-MaxCoord-G1`. The wrapper forwards flags unmodified.

```bash
bash scripts/official_train.sh SuperMimic-GenCoord-G1 \
  --motion-file /absolute/path/to/g1_reference_motions.npz \
  --runner.actor_obs proprio \
  --runner.history_length 10 \
  --runner.max_iterations 320000 \
  --runner.run_name g1_paper_baseline
```

The key overrides above matter: in this pinned upstream source, runner defaults are `actor_obs=rep_state`, `history_length=1`, and `max_iterations=1000000`. Appendix A of the paper specifies the 10-frame noisy proprioceptive history, and Appendix D reports 320,000 training iterations. The released code supplies `k=8`, `H=32`, `gamma=0.99`, `lambda=0.95`, the 4096-environment default, learned local-frame dynamics integration, G1-specific PD scaling, the 14-body reward, and return-based reset sampling.

For another representation, replace the task ID with `SuperMimic-MaxCoord-G1`. Review the pinned upstream source and `uv run src/supermimic/scripts/train.py <TASK_ID> --help` before changing other hyperparameters; the research code may expose behaviors not specified in the paper.

If `--motion-file` is omitted, upstream defaults to a Weights & Biases motion registry artifact. That may require access to the registry. A local motion file avoids depending on that artifact.

## 3. Motion data and evaluation

The paper's full-scale benchmark uses G1-retargeted [BONES-SEED](https://huggingface.co/datasets/bones-studio/seed) motion clips, with its filtering protocol in Appendix E. The dataset is **gated and licensed**: obtain access through its publisher and follow the license. Do not commit or redistribute motion files from a gated source.

The authors also provide a player:

```bash
bash scripts/official_play.sh SuperMimic-GenCoord-G1 --wandb-path YOUR_RUN_PATH
```

For the authors' evaluation scripts, run their `src/supermimic/scripts/benchmark.py` from the pinned upstream checkout and follow its CLI help. The original publication's large-dataset return and failure counts should not be compared with our historic static-standing trials.

## Scope and provenance

- This repository **pins and invokes** the original implementation; it is not a fork and does not duplicate the authors' licensed code.
- Our cart-pole/triple-pendulum studies remain independent experiments and can be used to test algorithmic changes in a small environment.
- The earlier local G1 standing script is retained only as a prototype and is not the authoritative paper baseline.
- A full published result still requires compatible motion data, GPU resources, complete training, and evaluation. Installing the source alone does not establish replication.
- All future G1 ablations should branch from the two pinned upstream revisions, record their patches and seeds, and compare against an unchanged official baseline.
