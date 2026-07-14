#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:-check}"
schema="${root}/protocol/schema/adapter-protocol-v5.schema.json"
models="${root}/python/inferlab-adapter-sdk/src/inferlab_adapter_sdk/_generated.py"
resource_models="${root}/crates/inferlab/resources/toolchain-python/inferlab_adapter_sdk/_generated.py"
temporary="$(mktemp -d)"
trap 'rm -rf "${temporary}"' EXIT

mkdir -p "${temporary}/schema"
cargo run --quiet --locked --manifest-path "${root}/Cargo.toml" \
  -p inferlab-protocol --example generate_schema -- \
  "${temporary}/schema/adapter-protocol-v5.schema.json"

datamodel-codegen \
  --input "${temporary}/schema/adapter-protocol-v5.schema.json" \
  --input-file-type jsonschema \
  --output "${temporary}/_generated.py" \
  --output-model-type pydantic_v2.BaseModel \
  --target-python-version 3.12 \
  --disable-future-imports \
  --enable-generated-header-marker \
  --use-standard-collections \
  --use-union-operator \
  --use-annotated \
  --strict-nullable \
  --enum-field-as-literal one \
  --infer-union-variant-names \
  --use-one-literal-as-default \
  --extra-fields forbid \
  --formatters black isort \
  --disable-timestamp

case "${mode}" in
  generate)
    mkdir -p "$(dirname "${schema}")" "$(dirname "${models}")" "$(dirname "${resource_models}")"
    cp "${temporary}/schema/adapter-protocol-v5.schema.json" "${schema}"
    cp "${temporary}/_generated.py" "${models}"
    cp "${temporary}/_generated.py" "${resource_models}"
    ;;
  check)
    cmp "${temporary}/schema/adapter-protocol-v5.schema.json" "${schema}"
    cmp "${temporary}/_generated.py" "${models}"
    cmp "${temporary}/_generated.py" "${resource_models}"
    ;;
  *)
    printf 'usage: %s [generate|check]\n' "$0" >&2
    exit 2
    ;;
esac
