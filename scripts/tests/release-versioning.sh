#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
temporary="$(mktemp -d)"
trap 'rm -rf "${temporary}"' EXIT

fail() {
  echo "release-versioning test: $1" >&2
  exit 1
}

expected_release_owned=$'inferlab-bench-runner\ninferlab-eval-runner'
actual_release_owned="$("${root}/scripts/python-package-inventory.sh" release-owned)"
test "${actual_release_owned}" = "${expected_release_owned}" \
  || fail "release-owned inventory did not contain exactly the measurement runners"

fixture="${temporary}/repo"
mkdir -p "${fixture}/scripts" "${fixture}/bin" "${fixture}/dist"
cp "${root}/scripts/python-package-inventory.sh" "${fixture}/scripts/"
cp "${root}/scripts/bump-product-version.sh" "${fixture}/scripts/"
cp "${root}/scripts/check-product-release-version.sh" "${fixture}/scripts/"
cp "${root}/scripts/prepare-python-package-release.sh" "${fixture}/scripts/"
cp "${root}/scripts/python-package-release-metadata.py" "${fixture}/scripts/"
cp "${root}/Cargo.toml" "${fixture}/"
cp "${root}/LICENSE" "${fixture}/"

while IFS= read -r path; do
  mkdir -p "${fixture}/$(dirname "${path}")"
  cp "${root}/${path}" "${fixture}/${path}"
done <<'EOF'
crates/inferlab/Cargo.toml
python/inferlab-adapter-sdk/pyproject.toml
python/inferlab-bench-runner/pyproject.toml
python/inferlab-eval-runner/pyproject.toml
python/inferlab-integration-sglang/pyproject.toml
python/inferlab-integration-tensorrt-llm/pyproject.toml
python/inferlab-integration-tokenspeed/pyproject.toml
python/inferlab-integration-vllm/pyproject.toml
.claude-plugin/marketplace.json
plugins/inferlab/.claude-plugin/plugin.json
plugins/inferlab/.codex-plugin/plugin.json
crates/inferlab/resources/plugin/.claude-plugin/marketplace.json
crates/inferlab/resources/plugin/plugins/inferlab/.claude-plugin/plugin.json
crates/inferlab/resources/plugin/plugins/inferlab/.codex-plugin/plugin.json
protocol/fixtures/valid/plan-serve-response.json
protocol/fixtures/valid/render-serve-response.json
protocol/fixtures/valid/render-serve-response-launch-file.json
EOF

true_path="$(type -P true)"
ln -s "${true_path}" "${fixture}/bin/cargo"
cp "${root}/scripts/tests/fixtures/pixi-release.sh" "${fixture}/bin/pixi"
chmod +x "${fixture}/bin/pixi"

workspace_projects=(
  "${fixture}/python/inferlab-adapter-sdk/pyproject.toml"
  "${fixture}/python/inferlab-integration-sglang/pyproject.toml"
  "${fixture}/python/inferlab-integration-tensorrt-llm/pyproject.toml"
  "${fixture}/python/inferlab-integration-tokenspeed/pyproject.toml"
  "${fixture}/python/inferlab-integration-vllm/pyproject.toml"
)
protocol_fixtures=("${fixture}"/protocol/fixtures/valid/*-response*.json)
workspace_hashes="$(sha256sum "${workspace_projects[@]}")"
protocol_hashes="$(sha256sum "${protocol_fixtures[@]}")"
sdk_version="$(sed -n 's/^version = "\([^"]*\)"$/\1/p' \
  "${fixture}/python/inferlab-adapter-sdk/pyproject.toml")"
vllm_version="$(sed -n 's/^version = "\([^"]*\)"$/\1/p' \
  "${fixture}/python/inferlab-integration-vllm/pyproject.toml")"
target_product_version="9.8.7"

PATH="${fixture}/bin:${PATH}" \
  "${fixture}/scripts/bump-product-version.sh" "${target_product_version}" \
  > "${temporary}/bump.out"

grep -Eq "^version = \"${target_product_version}\"$" "${fixture}/Cargo.toml" \
  || fail "product version was not updated"
grep -Eq 'inferlab-protocol = .*version = "9"' "${fixture}/crates/inferlab/Cargo.toml" \
  || fail "the binary crate dependency requirement did not follow the product major version"
for package in inferlab-bench-runner inferlab-eval-runner; do
  grep -Eq "^version = \"${target_product_version}\"$" \
    "${fixture}/python/${package}/pyproject.toml" \
    || fail "${package} did not follow the product version"
  grep -Fq "inferlab-adapter-sdk==${sdk_version}" \
    "${fixture}/python/${package}/pyproject.toml" \
    || fail "${package} SDK dependency changed during a product bump"
done
printf '%s\n' "${workspace_hashes}" | sha256sum --check --quiet \
  || fail "a workspace-side package changed during a product bump"
printf '%s\n' "${protocol_hashes}" | sha256sum --check --quiet \
  || fail "adapter fixture identity changed during a product bump"
grep -Eq "\"version\": \"${target_product_version}\"" \
  "${fixture}/plugins/inferlab/.codex-plugin/plugin.json" \
  || fail "embedded plugin did not follow the product version"
"${fixture}/scripts/check-product-release-version.sh" "v${target_product_version}"

vllm_wheel="inferlab_integration_vllm-${vllm_version}-py3-none-any.whl"
sdk_wheel="inferlab_adapter_sdk-${sdk_version}-py3-none-any.whl"
touch "${fixture}/dist/${vllm_wheel}"
touch "${fixture}/dist/${sdk_wheel}"

vllm_pyproject="${fixture}/python/inferlab-integration-vllm/pyproject.toml"
cp "${vllm_pyproject}" "${temporary}/vllm-pyproject.toml"
sed -i \
  "s/inferlab-adapter-sdk==${sdk_version}/inferlab-adapter-sdk>=${sdk_version}/" \
  "${vllm_pyproject}"
printf '\n[project.optional-dependencies]\ntest = ["inferlab-adapter-sdk==%s"]\n' \
  "${sdk_version}" >> "${vllm_pyproject}"
if PATH="${fixture}/bin:${PATH}" \
  "${fixture}/scripts/prepare-python-package-release.sh" inferlab-integration-vllm \
  > "${temporary}/non-exact.out" 2>&1; then
  fail "publication preparation accepted a non-exact runtime SDK dependency"
fi
grep -q 'exact inferlab-adapter-sdk runtime dependency' \
  "${temporary}/non-exact.out" \
  || fail "non-exact SDK dependency failure did not identify the runtime requirement"
cp "${temporary}/vllm-pyproject.toml" "${vllm_pyproject}"

PATH="${fixture}/bin:${PATH}" \
  "${fixture}/scripts/prepare-python-package-release.sh" inferlab-integration-vllm \
  > "${temporary}/release.out"

grep -Fq "twine upload dist/${vllm_wheel}" "${temporary}/release.out" \
  || fail "publication output did not select the requested wheel"
grep -Fq "gh release create inferlab-integration-vllm-v${vllm_version}" \
  "${temporary}/release.out" \
  || fail "publication output did not derive the package-scoped release tag"
grep -Eq 'gh release create .* --verify-tag ' "${temporary}/release.out" \
  || fail "publication output did not require the reviewed tag"
! grep -Fq "${sdk_wheel}" "${temporary}/release.out" \
  || fail "publication output included an unrelated wheel"
test -f "${fixture}/dist/${vllm_wheel}.sha256" \
  || fail "publication preparation did not write the selected wheel checksum"

if PATH="${fixture}/bin:${PATH}" \
  "${fixture}/scripts/prepare-python-package-release.sh" inferlab-eval-runner \
  > "${temporary}/invalid.out" 2>&1; then
  fail "publication preparation accepted a release-owned internal runner"
fi
grep -q 'not a workspace-side package' "${temporary}/invalid.out" \
  || fail "invalid package failure did not name the ownership boundary"
