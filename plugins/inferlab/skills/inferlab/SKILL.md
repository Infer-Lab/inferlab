---
name: inferlab
description: "Use when operating InferLab: run reproducible LLM inference experiments through a versioned workspace — serve lifecycles, closed-loop eval/bench recipes, standalone benches, Nsight Systems capture, runtime images, and the scratchpad journal — always reading results from file-first execution records."
---

# InferLab Operator Workflow

InferLab runs reproducible LLM inference experiments. A committed **workspace**
fixes the shareable baseline (stacks, named servers and cases, recipes, and
eval/bench definitions); a git-ignored **local bindings** file
(`.inferlab/local.toml`) supplies the machine-private facts (model weight
paths, machines, devices, ports, and placement); managed serving, measurement,
and image-build workflows write durable **records** under
`.inferlab/records/<ID>/`. You select declared things and run them — you do not
hand-compose framework commands, processes, or measurements.

## Operating Principles

- **Records are the interface.** Every non-dry-run managed workflow prints JSON
  with a record `id`; the record directory holds `record.json`, logs, and raw
  artifacts. Read outcomes from records, not from scraped terminal output.
- **Dry-run first when unsure.** `--dry-run` on `serve start`, `recipe run`,
  `bench`, and `image build` resolves and validates the applicable placement,
  devices, commands, and realization without launching. It writes no record and
  starts no server; server resolution still executes the integration adapter,
  and `bench --dry-run` needs its target server running.
- **Overrides are explicit.** `--set PATH=VALUE` overrides one field with a
  TOML value; the record keeps both the declared and effective values. Serve
  paths begin with `server.`. Recipe runs additionally accept selected
  `evals.<ID>.*` and `benches.<ID>.*` fields. Backend settings use
  `server.settings.<KEY>` or `server.roles.<ROLE>.settings.<KEY>`; core server
  fields keep their own typed paths.
- **One thing owns each fact.** Workspace facts are committed TOML under
  `.inferlab/workspace.toml` (+ `.inferlab/workspace.d/*.toml`); private facts
  live only in local bindings (`--local <FILE>` selects an alternate bindings
  file); execution facts live only in records.

## First Run In A Fresh Checkout

From the workspace root:

```
inferlab workspace show                 # validate and browse the committed catalog
cp .inferlab/local.example.toml .inferlab/local.toml  # when the workspace provides it
pixi install --locked --all              # realize every stack's selected Pixi environment
inferlab stack status                    # confirm each stack realization
inferlab toolchain install               # eval/bench measurement runtimes (only when you measure)
```

`pixi install --locked` with no `--environment`/`--all` only realizes Pixi's
implicit `default` environment, not every named environment selected by the
workspace's stacks — pass `--all`, or `--environment <PIXI_ENV>` for one.
`inferlab stack status` confirms the result without requiring local bindings,
so it's the right first InferLab command in a fresh checkout: it reports each stack as `confirmed`,
`never-installed`, or `not-usable`, and exits nonzero if any isn't confirmed.
`toolchain install` adds only the eval/bench measurement runtimes and is
needed only when you run an lm-eval or Bench measurement. Bind your local
facts in `.inferlab/local.toml` before the first resolving command. The
tracked example, when present, contains generic placeholders only; replace
them with this machine's model locators, machines, devices, ports, and
placements. A missing bindings file fails with guidance naming what belongs
there.

Declared stack checks run automatically before launches; a failing
check prints a `repair_hint` you can usually run verbatim (e.g.
`pixi run <task>`).

## Command Map

```
inferlab workspace show [--json]             # validate/browse the public catalog; no local bindings needed
inferlab workspace lock                    # re-lock the committed Pixi workspace
inferlab stack status [STACK]              # is a stack realization confirmed usable? (no local bindings needed)
inferlab toolchain install                 # install the measurement toolchains
inferlab serve start <SERVER> [--case C] [--placement P]  # start a long-running server
inferlab serve status|logs|stop <ID>       # inspect / log paths / finalize it
inferlab recipe run <RECIPE> [--case C] [--placement P]   # closed loop: serve + eval/bench suite + cleanup
inferlab bench <BENCH> --serve <ID>        # one named Bench against a running server
inferlab run [--stack S] -- <CMD>...       # one ad-hoc command inside a stack realization
inferlab image build <IMAGE>               # build + validate a runtime image
inferlab scratchpad note|show              # append-only operator journal
inferlab agent install|update|uninstall|doctor   # manage this skill's plugin
```

`serve start` / `recipe run` accept `--image <IMAGE_BUILD_RECORD>` to launch
from an assembled image, or `--external-image <ID>` for a declared digest-pinned image
the workspace did not build. `recipe run --capture <WORKLOAD_ID>` (repeatable)
and `bench --capture` attach Nsight Systems to selected workloads.

## Typical Flows

**Manual serve + bench.** Start, measure, stop; every step shares the record:

```
inferlab serve start <SERVER> --case <CASE> # -> {"id": "...-serve-<server>-<case>-<pid>", ...}
inferlab serve status <ID>                 # readiness + observed process state
inferlab bench <BENCH> --serve <ID> --set concurrency=[1,2]
inferlab serve stop <ID>                   # finalizes with cleanup evidence
```

For a temporary recipe qualification change, patch only measurements selected
by that recipe's suite, for example `--set evals.gsm8k.limit=100` or
`--set 'benches.random-8k1k.concurrency=[1,8]'`. These patches cannot change
measurement identity, kind, suite membership, gate, or server selection.

**Measurement controls.** An lm-eval definition selects one task as a pinned
lm-eval name (`task = "gsm8k"`), a release-bundled task
(`task = { bundled = "estonia" }`), or a tracked workspace YAML
(`task = { yaml = "evals/long-context.yaml" }`). The task owns its dataset,
splits, prompting, output type, filters, and scorer. InferLab uses the resolved
model-weight locator as the Hugging Face tokenizer locator. It sends
`generate_until` tasks to chat completions; prompt-logprob tasks use
completions after a tokenizer/logprob alignment probe.

Set inference parameters in the Eval or Bench definition's `request_body`, and
patch them for one run with nested `--set` paths:

```
inferlab recipe run qualify \
  --set evals.reasoning.trials=5 \
  --set evals.reasoning.request_body.temperature=1.0 \
  --set 'evals.reasoning.request_body.reasoning_effort="high"' \
  --set evals.reasoning.request_body.chat_template_kwargs.enable_thinking=true \
  --dry-run
```

InferLab retains the model, prompt/messages, streaming, one-completion, output
bound, stop, and repeated-trial seed fields. A conflicting `request_body`
member fails validation. For a concurrency Bench,
`warmup_prompts_per_concurrency = n` resolves `concurrency * n` native warmup
requests before profiling; warmup is excluded from normalized metrics. Use
`output_tokens = 1` for a prefill-dominant Bench; TPOT is then inapplicable and
omitted. See the
[0.5.0 workspace authoring guide](https://github.com/Infer-Lab/inferlab/blob/v0.5.0/docs/workspace-authoring.md)
for task-source, request-body, warmup, request-source, metric, and SLO examples.

**Closed loop.** `inferlab recipe run <RECIPE> --case <CASE>` starts the
server (multi-role prefill/decode topologies included), runs the declared
eval/bench suite, stops everything, and aggregates child records. Failure
runs are still evidence: the record carries failure phase, per-process
cleanup verification, and logs.

**Reproduction.** A clean-workspace record names the revision and a source
digest. In a fresh checkout of that revision: bootstrap as above, bind your
local facts, and re-run the recipe. Verify the reproduction by comparing
`revision` and `source_digest` between the old and new records — they must
match. `revision_reproducible: true` asserts only that a run executed from
a clean workspace (its revision fully identifies its sources); it never
compares against an earlier record.

**Ad-hoc probes.** Run any one-off command inside a stack realization
through `inferlab run` — version checks, Python probes, quick repros:

```
inferlab run -- python -c "import vllm; print(vllm.__version__)"
inferlab run --stack vllm -- pytest tests/ -k smoke         # several declared stacks
inferlab run --image <RECORD> --devices 0 -- nvidia-smi -L  # inside a built image
inferlab run --external-image <ID> --mount /data -- python3 /data/probe.py
```

**Never invoke interpreter or tool binaries from a materialized environment
prefix** (`.pixi/envs/<env>/bin/python`, `.pixi/envs/<env>/bin/*` or any
direct path into an environment prefix): a directly invoked binary silently
loses the manifest's activation variables and the packages' activation
scripts, so your probe observes a different environment than product launches
— the output looks fine and the evidence is wrong. `inferlab run` applies the
exact activation the product uses; there is no correct direct-path shortcut.
Container modes take explicit `--mount PATH[:rw]` (same-path, read-only by
default) and `--devices <IDX[,IDX...]>`; nothing is mounted or exposed
implicitly. `inferlab run` writes no record — use it for probes, not for
evidence.

**Journal.** Keep the narrative next to the evidence — no setup needed:

```
inferlab scratchpad note "tp1 OOMs at prefill readiness" --record last --topic flash
inferlab scratchpad show --topic flash     # recent tail; --all for everything
```

## Reading Results

- Eval: `cases/eval/result.json` plus `cases/eval/artifacts/`; `record.json`
  carries normalized metrics, gate evidence, repeated-trial counts and pass
  rate, task resolution, probe artifacts, and the native lm-eval command.
- Bench: `cases/<case>/result.json` plus `cases/<case>/artifacts/`; `record.json`
  carries normalized metrics, profiling counts, SLO conclusions, warmup and
  frozen-population slices, request-source acquisition, and raw AIPerf artifacts.
- Server records carry placement, per-machine device hardware identity, role and
  rank bindings, readiness, and cleanup outcomes.
- Compare runs on record fields (effective settings, digests, metrics), never
  on log text.

## Failure Etiquette

- Error codes are stable, governed by the error-code registry in RFC-0001
  (`docs/rfc/RFC-0001.md`): published codes never change meaning, so branch
  on the code (`E1004` config, `E2001` adapter, `E4002` lifecycle, ...),
  never on message text. The message names the failing fact precisely; fix
  the named fact rather than retrying.
- A failed launch still finalizes its record — inspect it before rerunning.
- If a server seems leaked, `inferlab serve status <ID>` reports observed
  process state; `serve stop <ID>` is idempotent and records cleanup
  verification.

## Privacy Discipline

Never put private model paths, hostnames, ports, device UUIDs, usernames,
credentials, or local scratch paths into tracked files, commit messages, or
anything published. Records are intentionally unredacted **local** evidence —
sharing them verbatim is an operator decision, not a default. Workspace TOML
is shareable by construction; local bindings never are.
