#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dist="${root}/dist"
temporary="$(mktemp -d)"
trap 'rm -rf "${temporary}"' EXIT

rm -rf "${dist}"
mkdir -p "${dist}"

all_inventory="$("${root}/scripts/python-package-inventory.sh" all)"
workspace_inventory="$("${root}/scripts/python-package-inventory.sh" workspace-side)"
mapfile -t packages <<< "${all_inventory}"
mapfile -t workspace_packages <<< "${workspace_inventory}"

for package in "${packages[@]}"; do
  python -m build --no-isolation --outdir "${dist}" "${root}/python/${package}"
done

uv venv --system-site-packages --python "$(command -v python)" "${temporary}/venv"
uv pip install \
  --python "${temporary}/venv/bin/python" \
  --no-deps \
  --reinstall \
  "${dist}"/*.whl
for package in "${packages[@]}"; do
  "${temporary}/venv/bin/python" -c \
    'import importlib, sys; importlib.import_module(sys.argv[1])' \
    "${package//-/_}"
done

for package in "${workspace_packages[@]}"; do
  [[ "${package}" == inferlab-integration-* ]] || continue
  adapter="inferlab-adapter-${package#inferlab-integration-}"
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
for package in "${packages[@]}"; do
  cmp -s "${root}/LICENSE" "${root}/python/${package}/LICENSE" \
    || { echo "python/${package}/LICENSE drifted from the repository LICENSE" >&2; exit 1; }
done
for artifact in "${dist}"/*.whl "${dist}"/*.tar.gz; do
  python - "${artifact}" "${root}/LICENSE" <<'PY'
import pathlib
import sys
import tarfile
import zipfile

path = pathlib.Path(sys.argv[1])
expected = pathlib.Path(sys.argv[2]).read_bytes()
if path.suffix == ".whl":
    with zipfile.ZipFile(path) as archive:
        licenses = [
            (name, archive.read(name))
            for name in archive.namelist()
            if pathlib.PurePosixPath(name).name == "LICENSE"
        ]
else:
    with tarfile.open(path) as archive:
        licenses = []
        for member in archive.getmembers():
            if not member.isfile() or pathlib.PurePosixPath(member.name).name != "LICENSE":
                continue
            source = archive.extractfile(member)
            if source is not None:
                licenses.append((member.name, source.read()))
if not licenses:
    raise SystemExit(f"{path}: no LICENSE inside")
for name, contents in licenses:
    if contents != expected:
        raise SystemExit(f"{path}: {name} differs from the repository LICENSE")
PY
done
