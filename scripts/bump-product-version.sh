#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
version="${1:-}"

fail() {
  echo "bump-product-version: $1" >&2
  exit 1
}

test "$#" -eq 1 || fail "usage: $0 VERSION"
echo "${version}" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+$' \
  || fail "VERSION must be strict semver (X.Y.Z), got: ${version}"

cd "${root}"

IFS=. read -r major minor _patch <<< "${version}"
if [ "${major}" = 0 ]; then
  crate_requirement="${major}.${minor}"
else
  crate_requirement="${major}"
fi

# [[ADR-0021]] Product bumps own the Cargo workspace, embedded plugin, and
# internal measurement-runner package metadata. Workspace-side integrations
# retain their independently released package versions and SDK requirements.
sed -i "s/^version = \"[^\"]*\"/version = \"${version}\"/" Cargo.toml
sed -E -i "/^inferlab-protocol = / s/version = \"[^\"]+\"/version = \"${crate_requirement}\"/" \
  crates/inferlab/Cargo.toml
sed -E -i "/^inferlab-proxy = / s/version = \"[^\"]+\"/version = \"${crate_requirement}\"/" \
  crates/inferlab/Cargo.toml

release_owned_inventory="$(scripts/python-package-inventory.sh release-owned)"
while IFS= read -r package; do
  pyproject="python/${package}/pyproject.toml"
  sed -i "0,/^version = \"[^\"]*\"/s//version = \"${version}\"/" "${pyproject}"
done <<< "${release_owned_inventory}"

for manifest in \
  .claude-plugin/marketplace.json \
  plugins/inferlab/.claude-plugin/plugin.json \
  plugins/inferlab/.codex-plugin/plugin.json \
  crates/inferlab/resources/plugin/.claude-plugin/marketplace.json \
  crates/inferlab/resources/plugin/plugins/inferlab/.claude-plugin/plugin.json \
  crates/inferlab/resources/plugin/plugins/inferlab/.codex-plugin/plugin.json; do
  sed -i "s/\"version\": \"[^\"]*\"/\"version\": \"${version}\"/" "${manifest}"
done

cargo build --workspace
pixi run build-python

echo
echo "== product artifacts bumped to ${version}; review the diff, then =="
echo "just verify && govctl release ${version} && govctl render changelog"
