#!/bin/sh
if [ "$1" = install ] && [ "$2" = --manifest-path ] && [ "$4" = --all ] && [ "$5" = --locked ]; then
  prefix="$(dirname "$3")"
  mkdir -p "$prefix/.pixi/envs/eval/bin" "$prefix/.pixi/envs/bench/bin"
  cat > "$prefix/.pixi/envs/eval/bin/python" <<'PYTHON'
#!/bin/sh
if [ "$2" = --handshake ]; then
  printf '{"runner_version":"0.1.0","lm_eval_version":"0.4.12"}\n'
  exit 0
fi
shift
exec fixture-eval-client "$@"
PYTHON
  cat > "$prefix/.pixi/envs/bench/bin/python" <<'PYTHON'
#!/bin/sh
if [ "$2" = --handshake ]; then
  printf '{"runner_version":"0.1.0","aiperf_version":"0.10.0"}\n'
  exit 0
fi
shift
exec fixture-bench-client "$@"
PYTHON
  chmod +x "$prefix/.pixi/envs/eval/bin/python" "$prefix/.pixi/envs/bench/bin/python"
  exit 0
fi
if [ "$1" = list ] && [ "$2" = --json ]; then
  cat <<'JSON'
[
  {"name": "python", "kind": "conda", "url": "https://conda.example/linux-64/python-3.12.0.conda", "sha256": "1111111111111111111111111111111111111111111111111111111111111111"},
  {"name": "inferlab-integration-vllm", "kind": "pypi", "url": "https://pypi.example/inferlab_integration_vllm-0.1.0-py3-none-any.whl", "sha256": "2222222222222222222222222222222222222222222222222222222222222222"},
  {"name": "vllm", "kind": "pypi", "editable": true, "url": "./vendor/vllm"},
  {"name": "flashinfer", "kind": "pypi", "editable": true, "url": "./vendor/flashinfer"}
]
JSON
  exit 0
fi
if [ "$1" = run ] && [ "$2" = --locked ] && [ "$3" = --no-install ] && [ "$4" = --executable ] && [ "$5" = -e ] && { [ "$6" = vllm ] || [ "$6" = adapter ]; } && [ "$7" = -- ]; then
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
  name="$(basename "$9")"
  printf 'wheel bytes for %s\n' "$name" > "$8/${name}-1.0-py3-none-any.whl"
  exit 0
fi
exec "$@"
