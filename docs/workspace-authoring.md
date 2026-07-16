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
  --set evals.gsm8k.trials=5 \
  --set evals.gsm8k.concurrency=8 \
  --set 'benches.random-8k1k.concurrency=[1, 8]' \
  --dry-run
```

Recipe measurement patches may name only Eval and Bench definitions selected
by that recipe's workload suite. They cannot change identities, kinds, suite
membership, the gate, or the selected server.

An lm-eval definition may set `trials` for repeated evaluation of one resolved
single-sample `generate_until` task. Its default is `1`. The definition seed
is the repeated base seed; when it is absent, repeated evaluation uses `1234`.
Trial `i` uses `base_seed + i - 1`. The existing `concurrency` field controls
those requests, and `request_body.seed` is rejected because the definition owns
the seed schedule. Each trial repeats the complete resolved Eval; Inferlab does
not rewrite task-owned response multiplicity, filters, or scorer behavior.

## lm-eval tasks and inference requests

An lm-eval definition selects exactly one task. Use a pinned lm-eval task name,
a release-bundled Inferlab task, or a workspace-owned task YAML:

```toml
[evals.builtin]
kind = "lm-eval"
task = "gsm8k"
metric = "exact_match"
metric_filter = "strict-match"
threshold = 0.90
timeout_seconds = 900

[evals.bundled]
kind = "lm-eval"
task = { bundled = "estonia" }
metric = "estonia_pass"
metric_filter = "strict-terminal-answer"
threshold = 0.50
timeout_seconds = 3600

[evals.workspace-task]
kind = "lm-eval"
task = { yaml = "evals/long-context.yaml" }
metric = "exact_match"
threshold = 0.80
timeout_seconds = 3600
```

The task, not a second Inferlab dataset layer, owns `dataset_path`,
`dataset_name`, split selection, prompting, output type, filters, and scoring.
Workspace YAML paths must be workspace-relative tracked `.yaml` or `.yml`
files. Inferlab resolves their YAML include closure, records the effective task
configuration and dataset selection, and includes that closure in source
identity. Release-bundled tasks are addressed only by their catalog name and
carry a release-owned closure digest.

Inferlab uses the resolved model-weight locator as the Hugging Face tokenizer
locator. This follows the normal model-directory convention and avoids a
second tokenizer setting; the locator must contain a usable tokenizer.
`generate_until` tasks use chat completions. Tasks whose
resolved output type is `loglikelihood`, `loglikelihood_rolling`, or
`multiple_choice` use completions and first run a prompt-logprob/tokenizer
alignment probe. Dynamic Python tasks are probed conservatively as well. A
probe failure makes support inconclusive rather than silently removing the
task.

Use `request_body` for task-specific inference parameters such as sampling,
reasoning effort, logprobs, or chat-template arguments:

```toml
[evals.reasoning]
kind = "lm-eval"
task = { yaml = "evals/reasoning.yaml" }
metric = "exact_match"
threshold = 0.80
timeout_seconds = 1800

[evals.reasoning.request_body]
temperature = 1.0
reasoning_effort = "high"
logprobs = true

[evals.reasoning.request_body.chat_template_kwargs]
enable_thinking = true
```

The same nested values may be patched for one run, for example
`--set evals.reasoning.request_body.temperature=0.6`. `request_body` is a JSON
request fragment, not a replacement request: Inferlab retains ownership of the
model, prompt or messages, streaming mode, one-completion policy, output bound,
and stop conditions. Eval also owns the repeated-trial seed schedule. Conflicts
with those fields fail during validation and the complete effective fragment is
preserved in dry-run and record evidence.

## Serving Bench warmup and metrics

A concurrency Bench may run a native warmup phase before its profiled phase:

```toml
[benches.random-8k1k]
kind = "serving"
request_source = { kind = "random", input_tokens = 8192, output_tokens = 1024 }
concurrency = [1, 8]
prompts_per_concurrency = 4
warmup_prompts_per_concurrency = 2
request_body = { temperature = 1.0 }
timeout_seconds = 900
```

For concurrency `c`, the resolved warmup count is
`c * warmup_prompts_per_concurrency`. Warmup uses the same route, request
source, request body, and concurrency as profiling, but consumes a disjoint
prefix of the frozen request population. It is excluded from normalized
metrics and profiling request counts. A requested prefix-cache reset happens
once before warmup, and the case timeout covers reset, warmup, profiling, and
result handling; process cleanup retains its separate grace.

Every successful Bench reports `request_throughput`, `output_throughput`, and
`total_token_throughput`. For each of `request_latency_ms`, `ttft_ms`, and
`tpot_ms`, Inferlab reports `mean`, `min`, `max`, `stddev`, `p50`, `p90`,
`p95`, and `p99` using names such as `p95_tpot_ms`. TPOT is not applicable to
an `output_tokens = 1` prefill-dominant workload and its TPOT metrics are then
omitted. `prompt_cache_read_ratio` is present only when AIPerf reports valid
cache-usage evidence. `good_request_ratio` and `goodput` are derived only when
a request SLO is configured.

## Serving Bench SLOs

A static or adaptive serving Bench may constrain normalized aggregate metrics,
individual request latency, or both. Aggregate constraints are inclusive and
AND-composed. Request SLOs count a request as good only when every configured
latency bound passes, then gate the case with `minimum_good_request_ratio`.

```toml
[benches.saturation]
kind = "adaptive-serving"
request_source = { kind = "random", input_tokens = 8192, output_tokens = 1024 }
initial_request_rates = [1.0, 4.0]
aggregate_slos = [
  { metric = "request_throughput", at_least = 1.0 },
  { metric = "p99_ttft_ms", at_most = 800.0 },
]
request_slo = { request_latency_ms = 5000.0, ttft_ms = 800.0, tpot_ms = 30.0, minimum_good_request_ratio = 0.99 }
max_search_steps = 6
min_rate_resolution = 0.25
duration_seconds = 60
timeout_seconds = 900
```

Adaptive Bench uses `highest-feasible-rate-v1`: it probes every initial rate,
then uses bounded doubling and directional bisection to select only the highest
observed feasible rate. `max_search_steps` covers automatically added probes;
it does not truncate the declared initial list. Use command-line `--set` to
override recipe-specific SLO values without changing the stored definition.

## Serving Bench request sources

Every serving Bench selects one closed request source. A random source keeps
the existing exact synthetic token shape:

```toml
request_source = { kind = "random", input_tokens = 8192, output_tokens = 1024 }
```

The release catalog currently exposes ShareGPT as a bounded conversational
source. Inferlab pins the Apache-2.0
[ShareGPT Vicuna snapshot](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/tree/bcd32a724d8460ebe14e1d05b0195e30e9a46cb1):

```toml
request_source = { kind = "dataset", dataset = "sharegpt", max_input_tokens = 8192 }
```

Inferlab downloads the release-pinned snapshot on first execution, verifies
its digest, and reuses it from
`$XDG_CACHE_HOME/inferlab/datasets/sha256/<digest>` (normally
`~/.cache/inferlab/datasets/sha256/<digest>`). Dry-run reports the catalog and
cache state but does not download missing data.

Each selected conversation becomes one independent chat-completions request.
The final assistant message is held out to derive the output limit. If the
rendered input exceeds `max_input_tokens`, Inferlab rolls back complete trailing
user/assistant exchanges until an earlier target fits; it never truncates a
message or discards the leading history. Set `output_tokens` inside the table
to replace target-derived output lengths, including `output_tokens = 1` for a
prefill-dominant run. The Bench-level `seed` controls deterministic sampling
without replacement. Command-line overrides may change fields within the
selected source, but cannot change `request_source.kind`.

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
