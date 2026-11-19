#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "$script_dir/.." && pwd)"
source "$script_dir/official_versions.sh"
workspace="$project_dir/.upstream/supermimic_ws"

check_checkout() {
  local checkout="$1"
  local expected="$2"
  if [[ ! -d "$checkout/.git" ]]; then
    printf 'Missing official checkout: %s. Run scripts/official_setup.sh first.\n' "$checkout" >&2
    exit 1
  fi
  local current
  current="$(git -C "$checkout" rev-parse HEAD)"
  if [[ "$current" != "$expected" ]]; then
    printf 'Official checkout changed: %s; expected %s, found %s.\n' "$checkout" "$expected" "$current" >&2
    exit 1
  fi
}

check_checkout "$workspace/supermimic" "$SUPERMIMIC_SHA"
check_checkout "$workspace/mjlab_research" "$MJLAB_SHA"
if ! command -v uv >/dev/null 2>&1; then
  printf 'uv is required: https://docs.astral.sh/uv/\n' >&2
  exit 1
fi

if [[ "$#" -eq 0 || "$1" == -* ]]; then
  set -- SuperMimic-GenCoord-G1 "$@"
fi
cd "$workspace/supermimic"
exec uv run src/supermimic/scripts/train.py "$@"
