#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "$script_dir/.." && pwd)"
source "$script_dir/official_versions.sh"
workspace="$project_dir/.upstream/supermimic_ws"

for item in "supermimic:$SUPERMIMIC_SHA" "mjlab_research:$MJLAB_SHA"; do
  name="${item%%:*}"
  expected="${item#*:}"
  checkout="$workspace/$name"
  if [[ ! -d "$checkout/.git" || "$(git -C "$checkout" rev-parse HEAD)" != "$expected" ]]; then
    printf 'Missing or unpinned %s. Run scripts/official_setup.sh first.\n' "$name" >&2
    exit 1
  fi
done

if ! command -v uv >/dev/null 2>&1; then
  printf 'uv is required: https://docs.astral.sh/uv/\n' >&2
  exit 1
fi
if [[ "$#" -eq 0 || "$1" == -* ]]; then
  set -- SuperMimic-GenCoord-G1 "$@"
fi
cd "$workspace/supermimic"
exec uv run src/supermimic/scripts/play.py "$@"
