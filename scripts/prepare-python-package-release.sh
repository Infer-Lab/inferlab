#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
package="${1:-}"

fail() {
  echo "prepare-python-package-release: $1" >&2
  exit 1
}

test "$#" -eq 1 || fail "usage: $0 PACKAGE"

cd "${root}"
workspace_inventory="$(scripts/python-package-inventory.sh workspace-side)"
selected=false
while IFS= read -r candidate; do
  if [ "${candidate}" = "${package}" ]; then
    selected=true
    break
  fi
done <<< "${workspace_inventory}"
${selected} || fail "${package} is not a workspace-side package"

pyproject="python/${package}/pyproject.toml"
metadata="$(pixi run python scripts/python-package-release-metadata.py "${package}")"
mapfile -t metadata_fields <<< "${metadata}"
test "${#metadata_fields[@]}" -eq 2 \
  || fail "${pyproject}: metadata reader returned ${#metadata_fields[@]} fields, expected 2"
distribution="${metadata_fields[0]}"
version="${metadata_fields[1]}"
test "${distribution}" = "${package}" \
  || fail "${pyproject}: distribution ${distribution} does not match inventory name ${package}"

# [[RFC-0006:C-INTEGRATIONS]] Build once through the standing package gates,
# then select only the requested distribution for publication. No registry or
# GitHub mutation occurs here.
pixi run build-python

shopt -s nullglob
wheel_distribution="${distribution//-/_}"
matches=(dist/"${wheel_distribution}-${version}"-*.whl)
test "${#matches[@]}" -eq 1 \
  || fail "expected one wheel for ${distribution} ${version}, found ${#matches[@]}"
wheel="${matches[0]}"
wheel_basename="$(basename "${wheel}")"
(cd dist && sha256sum "${wheel_basename}" > "${wheel_basename}.sha256")

tag="${distribution}-v${version}"
title="${distribution} ${version}"
notes="Python package release for ${distribution} ${version}."

echo
echo "== publication commands (ADR-0008: operator-performed, not run here) =="
echo "# 1. package index (wheel only):"
printf 'twine upload %q\n' "${wheel}"
echo "# 2. after pushing the reviewed package tag, create its GitHub release:"
printf 'gh release create %q --repo Infer-Lab/inferlab --verify-tag --title %q --notes %q \\\n' \
  "${tag}" "${title}" "${notes}"
printf '  %q %q LICENSE\n' "${wheel}" "${wheel}.sha256"
echo
echo "== stopping before publication; both acts remain operator-performed (ADR-0008) =="
