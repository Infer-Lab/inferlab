#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dist="${root}/dist"
temporary="$(mktemp -d)"
trap 'rm -rf "${temporary}"' EXIT

rm -rf "${dist}"
mkdir -p "${dist}"

for package in \
  inferlab-adapter-sdk \
  inferlab-bench-runner \
  inferlab-eval-runner \
  inferlab-integration-sglang \
  inferlab-integration-vllm
do
  python -m build --no-isolation --outdir "${dist}" "${root}/python/${package}"
done

uv venv --system-site-packages --python "$(command -v python)" "${temporary}/venv"
uv pip install \
  --python "${temporary}/venv/bin/python" \
  --no-deps \
  --reinstall \
  "${dist}"/*.whl
"${temporary}/venv/bin/python" -c \
  'import inferlab_adapter_sdk, inferlab_bench_runner, inferlab_eval_runner, inferlab_integration_sglang, inferlab_integration_vllm'

for adapter in inferlab-adapter-sglang inferlab-adapter-vllm
do
  for operation in plan-serve render-serve
  do
    "${temporary}/venv/bin/${adapter}" \
      < "${root}/protocol/fixtures/valid/${operation}-request.json" \
      > "${temporary}/${operation}-response.json"
    "${temporary}/venv/bin/python" -c \
      'import pathlib, sys; from inferlab_adapter_sdk import AdapterResponse; AdapterResponse.model_validate_json(pathlib.Path(sys.argv[1]).read_text())' \
      "${temporary}/${operation}-response.json"
  done
done

# License retention gate (RFC-0001:C-LICENSE-RETENTION): every package LICENSE
# matches the repository's, and
# every built wheel and sdist actually carries it — an SPDX expression alone
# does not satisfy MIT's notice-retention condition.
for package in \
  inferlab-adapter-sdk \
  inferlab-bench-runner \
  inferlab-eval-runner \
  inferlab-integration-sglang \
  inferlab-integration-vllm
do
  cmp -s "${root}/LICENSE" "${root}/python/${package}/LICENSE" \
    || { echo "python/${package}/LICENSE drifted from the repository LICENSE" >&2; exit 1; }
done
for artifact in "${dist}"/*.whl "${dist}"/*.tar.gz; do
  python - "${artifact}" <<'PY'
import sys, tarfile, zipfile
path = sys.argv[1]
if path.endswith(".whl"):
    names = zipfile.ZipFile(path).namelist()
else:
    names = tarfile.open(path).getnames()
assert any(n.endswith("LICENSE") for n in names), f"{path}: no LICENSE inside"
PY
done
