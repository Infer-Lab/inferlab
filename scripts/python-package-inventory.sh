#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_root="${root}/python"
scope="${1:-}"

shopt -s nullglob

case "${scope}" in
  all)
    pyprojects=("${python_root}"/*/pyproject.toml)
    ;;
  workspace-side)
    pyprojects=(
      "${python_root}/inferlab-adapter-sdk/pyproject.toml"
      "${python_root}"/inferlab-integration-*/pyproject.toml
    )
    ;;
  release-owned)
    pyprojects=(
      "${python_root}/inferlab-bench-runner/pyproject.toml"
      "${python_root}/inferlab-eval-runner/pyproject.toml"
    )
    ;;
  *)
    echo "usage: $0 {all|workspace-side|release-owned}" >&2
    exit 2
    ;;
esac

test "${#pyprojects[@]}" -gt 0 || {
  echo "python package inventory is empty for scope: ${scope}" >&2
  exit 1
}

for pyproject in "${pyprojects[@]}"; do
  test -f "${pyproject}" || {
    echo "python package inventory entry has no pyproject.toml: ${pyproject}" >&2
    exit 1
  }
  basename "$(dirname "${pyproject}")"
done | LC_ALL=C sort
