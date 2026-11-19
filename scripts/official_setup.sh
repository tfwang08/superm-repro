#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "$script_dir/.." && pwd)"
source "$script_dir/official_versions.sh"
workspace="$project_dir/.upstream/supermimic_ws"
mkdir -p "$workspace"

clone_pinned() {
  local repo_url="$1"
  local commit_sha="$2"
  local name="$3"
  local target="$workspace/$name"

  if [[ -e "$target" ]]; then
    if [[ ! -d "$target/.git" ]]; then
      printf 'Refusing to overwrite non-git path: %s\n' "$target" >&2
      return 1
    fi
    local current_sha
    current_sha="$(git -C "$target" rev-parse HEAD)"
    if [[ "$current_sha" != "$commit_sha" ]]; then
      printf 'Existing checkout has a different HEAD: %s\n' "$target" >&2
      printf 'Expected %s; found %s. Preserve your changes and choose a fresh directory.\n' "$commit_sha" "$current_sha" >&2
      return 1
    fi
    printf 'Using existing pinned checkout: %s\n' "$target"
    return 0
  fi

  git clone --no-tags "$repo_url" "$target"
  git -C "$target" checkout --detach "$commit_sha"
  printf 'Pinned %s at %s\n' "$name" "$commit_sha"
}

# The official pyproject specifies mjlab = ../mjlab_research.
clone_pinned "$MJLAB_REPO" "$MJLAB_SHA" mjlab_research
clone_pinned "$SUPERMIMIC_REPO" "$SUPERMIMIC_SHA" supermimic

printf '\nOfficial sources are ready. Install uv and then run:\n'
printf '  cd "%s/supermimic" && uv sync\n' "$workspace"
printf 'Or run scripts/official_train.sh from this repository.\n'
