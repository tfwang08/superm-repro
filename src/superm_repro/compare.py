"""Run matched four-seed baseline/experiment trials with checkpoint selection."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--links", choices=(1, 2, 3), type=int, default=1)
    parser.add_argument("--iterations", type=int, default=320)
    parser.add_argument("--episode-steps", type=int, default=256)
    parser.add_argument("--validation-episodes", type=int, default=24)
    parser.add_argument("--long-test-episodes", type=int, default=96)
    parser.add_argument("--long-test-steps", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/comparison"))
    args = parser.parse_args()
    if min(
        args.iterations,
        args.episode_steps,
        args.validation_episodes,
        args.long_test_episodes,
        args.long_test_steps,
    ) < 1:
        parser.error("All training and evaluation sizes must be positive")

    reports = []
    for profile in ("baseline", "experiment"):
        for seed in range(4):
            checkpoint = args.output_dir / profile / f"links{args.links}_seed{seed}.pt"
            command = [
                sys.executable, "-m", "superm_repro.train_mujoco",
                "--profile", profile,
                "--links", str(args.links),
                "--seed", str(seed),
                "--iterations", str(args.iterations),
                "--episode-steps", str(args.episode_steps),
                "--eval-episodes", str(args.validation_episodes),
                "--checkpoint-every", "40",
                "--long-test",
                "--long-test-episodes", str(args.long_test_episodes),
                "--long-test-steps", str(args.long_test_steps),
                "--checkpoint", str(checkpoint),
            ]
            print(f"Running {profile} seed={seed} with {args.links} links", flush=True)
            subprocess.run(command, check=True)
            reports.append(json.loads(checkpoint.with_suffix(".json").read_text()))

    output = args.output_dir / f"links{args.links}_comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    print(f"Saved matched-seed results: {output}")


if __name__ == "__main__":
    main()
