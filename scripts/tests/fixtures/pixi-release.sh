#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" = run ] && [ "${2:-}" = python ]; then
  shift 2
  exec python "$@"
fi

if [ "${1:-}" = run ] && [ "${2:-}" = build-python ]; then
  exit 0
fi

echo "unexpected fake pixi invocation: $*" >&2
exit 1
