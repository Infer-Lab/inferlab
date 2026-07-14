# Workspace authoring

An Inferlab workspace has two authorities:

- committed `.inferlab/workspace.toml` and `.inferlab/workspace.d/*.toml` files
  describe shareable models, stacks, servers, cases, measurements, and recipes;
- git-ignored `.inferlab/local.toml` binds those definitions to model weights,
  machines, devices, ports, and placement for one operator.

Run `inferlab workspace show` to validate and browse the committed authority.
It does not read local bindings or inspect a stack realization. Use
`inferlab workspace show --json` when another tool needs the canonical merged
definition.

## Minimal workspace

The root file owns the schema version. Definitions may live there or in
identifier-disjoint fragments under `workspace.d/`.

```toml
schema_version = 2

[models.example]
served_name = "example"

[stacks.vllm]
integration = "vllm"
pixi_environment = "vllm"
source_paths = []

[servers.example]
stack = "vllm"
model = "example"
topology = "single"
readiness_timeout_seconds = 900

[servers.example.settings]
max_model_len = 8192

[servers.example.cases.tp2.parallelism.outer]
tensor_parallel_size = 2

[evals.smoke]
kind = "openai-smoke"
prompt = "Hello"
max_tokens = 16
timeout_seconds = 60

[workload_suites.smoke]
evals = ["smoke"]
gate = "smoke"

[recipes.smoke]
server = "example"
workload_suite = "smoke"
```

The sole `tp2` case is selected automatically, so this server does not need a
`default_case`. A server with no cases uses its base behavior. A server with
multiple cases must declare `default_case`; the operator may always select a
different one with `--case`.

Framework settings belong under `settings`, either on the server or on a
canonical role. Integrations validate their typed fields. `extra_args` remains
the explicit backend escape hatch and is replaced as one complete array by a
case or invocation patch.

## Prefill/decode servers

A P/D server uses the canonical `prefill` and `decode` roles. The router is
derived from `routing_backend`; do not declare a `router` role in the public
workspace.

```toml
[servers.example-pd]
stack = "vllm"
model = "example"
topology = "prefill_decode"
readiness_timeout_seconds = 1800
default_case = "builtin-nixl"

[servers.example-pd.roles.prefill]
replicas = 2

[servers.example-pd.roles.prefill.parallelism.outer]
tensor_parallel_size = 4

[servers.example-pd.roles.decode]
replicas = 2

[servers.example-pd.roles.decode.parallelism.outer]
tensor_parallel_size = 2

[servers.example-pd.cases.builtin-nixl]
routing_backend = "builtin"
kv_transfer = "nixl"

[servers.example-pd.cases.native-mooncake]
readiness_timeout_seconds = 900
routing_backend = "vllm-router"
kv_transfer = "mooncake"

[servers.example-pd.cases.native-mooncake.roles.prefill.settings]
kv_transfer_protocol = "rdma"
```

Common settings apply to every model-serving role. Role settings apply after
the common layer. A selected case may patch common or role settings,
parallelism, replica counts, routing, transfer, profiling, and readiness.

## Local bindings

For a single-machine TP2 server, a minimal `.inferlab/local.toml` is:

```toml
default_placement = "local"

[model_weights.example]
locator = "/models/example"

[machines.local]
host = "127.0.0.1"
devices = [0, 1]
ports = [8000]

[placements.local]
machines = ["local"]
```

Published workspaces should provide this shape as
`.inferlab/local.example.toml`; operators copy it to the ignored local file and
replace the generic values.

Use explicit rank placement when replicas span machines, roles use different
device counts, or the same model has different locators on each machine. This
example places two TP4 prefill replicas across pairs of machines, two TP2
decode replicas on individual machines, and a zero-device router on the
controller:

```toml
default_placement = "cluster"

[model_weights.example.machine_locators]
controller = "/models/example"
prefill-a = "/models/example-a"
prefill-b = "/models/example-b"
prefill-c = "/models/example-c"
prefill-d = "/models/example-d"
decode-a = "/models/example-a"
decode-b = "/models/example-b"

[machines.controller]
host = "controller.example"
devices = []
ports = [7000]

[machines.prefill-a]
host = "prefill-a.example"
devices = [0, 1]
ports = [8000, 8001, 8100]
workspace = "/srv/inferlab/example"
launch = { kind = "ssh", target = "prefill-a" }

[machines.prefill-b]
host = "prefill-b.example"
devices = [0, 1]
ports = [8000, 8001, 8100]
workspace = "/srv/inferlab/example"
launch = { kind = "ssh", target = "prefill-b" }

[machines.prefill-c]
host = "prefill-c.example"
devices = [0, 1]
ports = [8000, 8001, 8100]
workspace = "/srv/inferlab/example"
launch = { kind = "ssh", target = "prefill-c" }

[machines.prefill-d]
host = "prefill-d.example"
devices = [0, 1]
ports = [8000, 8001, 8100]
workspace = "/srv/inferlab/example"
launch = { kind = "ssh", target = "prefill-d" }

[machines.decode-a]
host = "decode-a.example"
devices = [0, 1]
ports = [8000, 8001]
workspace = "/srv/inferlab/example"
launch = { kind = "ssh", target = "decode-a" }

[machines.decode-b]
host = "decode-b.example"
devices = [0, 1]
ports = [8000, 8001]
workspace = "/srv/inferlab/example"
launch = { kind = "ssh", target = "decode-b" }

[[placements.cluster.roles.prefill.replicas]]
ranks = [
  { machine = "prefill-a", devices = [0, 1] },
  { machine = "prefill-b", devices = [0, 1] },
]

[[placements.cluster.roles.prefill.replicas]]
ranks = [
  { machine = "prefill-c", devices = [0, 1] },
  { machine = "prefill-d", devices = [0, 1] },
]

[placements.cluster.roles.decode]
replicas = [
  { machine = "decode-a", devices = [0, 1] },
  { machine = "decode-b", devices = [0, 1] },
]

[placements.cluster.roles.router]
machine = "controller"
devices = []
```

The role-level `machine` and `devices` form is one replica at rank 0. Use a
`ranks` list only when that one replica spans two or more machines; use a
`replicas` list only when the role has two or more replicas. Replica and rank
numbers are derived from list order.

## Invocation patches

Use repeatable `--set` for temporary typed changes. Values use TOML syntax and
later assignments win.

```sh
inferlab serve start example \
  --set server.readiness_timeout_seconds=1800 \
  --set server.settings.max_model_len=32768 \
  --set server.roles.serve.parallelism.outer.tensor_parallel_size=4 \
  --dry-run

inferlab recipe run qualify \
  --set evals.gsm8k.limit=100 \
  --set evals.gsm8k.concurrency=8 \
  --set 'benches.random-8k1k.concurrency=[1, 8]' \
  --dry-run
```

Recipe measurement patches may name only Eval and Bench definitions selected
by that recipe's workload suite. They cannot change identities, kinds, suite
membership, the gate, or the selected server.

## Validate before launch

Use the commands in increasing order of machine dependence:

```sh
inferlab workspace show
inferlab stack status
inferlab recipe run smoke --dry-run
inferlab recipe run smoke
```

`workspace show` validates the public catalog. `stack status` checks the
selected Pixi realizations. Dry-run then resolves local placement, effective
settings, endpoints, device assignments, commands, environment, and override
provenance without launching or writing an execution record.
