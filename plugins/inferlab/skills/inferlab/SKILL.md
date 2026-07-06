---
name: inferlab
description: "Use when operating Inferlab: run reproducible LLM inference experiments through a versioned workspace — serve lifecycles, closed-loop eval/bench recipes, standalone benches, Nsight Systems capture, runtime images, and the scratchpad journal — always reading results from file-first execution records."
---

# Inferlab Operator Workflow

Inferlab runs reproducible LLM inference experiments. A committed **workspace**
fixes the shareable baseline (serving sources, Pixi environment, recipes,
serve profiles, eval/bench definitions); a git-ignored **local bindings** file
(`.inferlab/local.toml`) supplies the machine-private facts (model weight
paths, machines, GPUs, ports); every execution writes a durable **record**
under `.inferlab/records/<ID>/`. You select declared things and run them — you
do not hand-compose servers, environments, or measurements.

## Operating Principles

- **Records are the interface.** Every start/run command prints JSON with a
  record `id`; the record directory holds `record.json`, logs, and raw
  artifacts. Read outcomes from records, not from scraped terminal output.
- **Dry-run first when unsure.** `--dry-run` on `serve start`, `recipe run`,
  and `bench` resolves and validates everything (placement, GPUs, commands,
  environment) without launching. It writes no record and starts no server;
  resolution still executes the integration adapter, and `bench --dry-run`
  needs its target server running.
- **Overrides are explicit.** `--set server.PATH=VALUE` (serve/recipe) and
  `--set PATH=VALUE` (bench) override one field with a TOML value; the record
  keeps both the declared and the effective values.
- **One thing owns each fact.** Workspace facts are committed TOML under
  `.inferlab/workspace.toml` (+ `.inferlab/workspace.d/*.toml`); private facts
  live only in local bindings (`--local <FILE>` selects an alternate bindings
  file); execution facts live only in records.

## First Run In A Fresh Checkout

From the workspace root:

```
pixi install --locked                    # realize the committed serving environment from the lock
inferlab toolchain install               # eval/bench measurement runtimes (only when you measure)
```

`pixi install --locked` alone realizes the serving environment: the committed
lock pins the workspace-side adapter wheels by hash, so nothing has to be
materialized first. `toolchain install` adds only the eval/bench measurement
runtimes and is needed only when you run an lm-eval or Bench measurement.
Bind your local facts in `.inferlab/local.toml` before the first resolving
command — a missing bindings file fails with guidance naming what belongs
there.

Declared environment checks run automatically before launches; a failing
check prints a `repair_hint` you can usually run verbatim (e.g.
`pixi run <task>`).

## Command Map

```
inferlab env lock                          # re-lock the committed Pixi environment
inferlab toolchain install                 # install the measurement toolchains
inferlab serve start <RECIPE> [--case C]   # start a long-running server
inferlab serve status|logs|stop <ID>       # inspect / log paths / finalize it
inferlab recipe run <RECIPE> [--case C]    # closed loop: serve + eval/bench suite + cleanup
inferlab bench <BENCH> --serve <ID>        # one named Bench against a running server
inferlab image build <IMAGE>               # build + validate a runtime image
inferlab scratchpad note|show              # append-only operator journal
inferlab agent install|update|uninstall|doctor   # manage this skill's plugin
```

`serve start` / `recipe run` accept `--image <IMAGE_BUILD_RECORD>` to launch
from an assembled image, or `--external-image <REF>` for a digest-pinned image
the workspace did not build. `recipe run --capture <WORKLOAD_ID>` (repeatable)
and `bench --capture` attach Nsight Systems to selected workloads.

## Typical Flows

**Manual serve + bench.** Start, measure, stop; every step shares the record:

```
inferlab serve start <RECIPE>              # -> {"id": "...-serve-N", ...}
inferlab serve status <ID>                 # readiness + observed process state
inferlab bench <BENCH> --serve <ID> --set concurrency=[1,2]
inferlab serve stop <ID>                   # finalizes with cleanup evidence
```

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

**Journal.** Keep the narrative next to the evidence — no setup needed:

```
inferlab scratchpad note "tp1 OOMs at prefill readiness" --record last --topic flash
inferlab scratchpad show --topic flash     # recent tail; --all for everything
```

## Reading Results

- Bench cases: `cases/<case>/result.json` (normalized metrics) and
  `cases/<case>/artifacts/` (raw AIPerf output) inside the bench record.
- Server records carry placement, per-machine GPU hardware identity, role and
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

Never put private model paths, hostnames, ports, GPU UUIDs, usernames,
credentials, or local scratch paths into tracked files, commit messages, or
anything published. Records are intentionally unredacted **local** evidence —
sharing them verbatim is an operator decision, not a default. Workspace TOML
is shareable by construction; local bindings never are.
