#!/bin/sh
if [ "$1" = list ] && [ "$2" = --json ]; then
  cat <<'JSON'
[
  {"name": "python", "kind": "conda", "url": "https://conda.example/linux-64/python-3.12.0.conda", "sha256": "1111111111111111111111111111111111111111111111111111111111111111"},
  {"name": "vllm", "kind": "pypi", "editable": true, "url": "./vendor/vllm"},
  {"name": "flashinfer", "kind": "pypi", "editable": true, "url": "./vendor/flashinfer"}
]
JSON
  exit 0
fi
if [ "$1" = run ] && [ "$2" = --locked ] && [ "$3" = --no-install ] && [ "$4" = --executable ] && [ "$5" = -e ] && [ "$6" = vllm ] && [ "$7" = -- ]; then
  shift 7
elif [ "$1" = run ] && [ "$2" = --clean-env ] && [ "$3" = --as-is ] && [ "$4" = --executable ] && [ "$5" = -e ] && [ "$6" = vllm ] && [ "$7" = -- ]; then
  shift 7
elif [ "$1" = run ] && [ "$2" = --as-is ] && [ "$3" = --executable ] && [ "$4" = -e ] && [ "$5" = vllm ] && [ "$6" = -- ]; then
  shift 6
else
  printf 'unexpected pixi fixture arguments\n' >&2
  exit 2
fi
if [ "$1" = /bin/sh ] && [ "$2" = -c ]; then
  shift 4
  while [ $# -gt 0 ] && printf '%s' "$1" | grep -q =; do shift; done
fi
if [ "$1" = python ] && [ "$3" = pip ] && [ "$4" = wheel ] && [ "$7" = --wheel-dir ]; then
  printf 'stray\n' > vendor/vllm/stray-build-artifact.txt
  printf 'simulated backend crash\n' >&2
  exit 1
fi
exec "$@"
