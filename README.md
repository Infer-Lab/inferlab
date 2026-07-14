# Inferlab

Reproducible LLM inference experiments. A committed **workspace** fixes the
shareable baseline — stacks, named servers and cases, recipes, and eval/bench
definitions. A git-ignored **local bindings** file supplies machine-private
facts (model weights, machines, devices, ports, and placement). Managed serving,
measurement, and image-build workflows write durable, file-first **records**
you can inspect, compare, and reproduce.

- **Serve lifecycles** — long-running framework servers, single-role or
  prefill/decode disaggregated across machines, with readiness, logs, and
  verified cleanup in the record.
- **Closed-loop recipes** — serve + eval (lm-eval) + bench (AIPerf) suites in
  one command, with per-case metrics and raw artifacts preserved.
- **Runtime images** — build and validate OCI images from the same workspace
  baseline, then launch recipes from them.
- **Profiling** — attach Nsight Systems captures to selected workloads.
- **Source identity** — records carry the workspace revision and a source
  digest; a clean-workspace run is reproducible from a fresh checkout.
- **Operator journal** — an append-only scratchpad on the same time axis as
  the records.

## Install

Download a release binary (x86_64 / aarch64 Linux):

```sh
curl -fsSL https://github.com/Infer-Lab/inferlab/releases/latest/download/install.sh | sh
```

Or install from the crates registry (Rust 1.89+):

```sh
cargo install inferlab
```

Or build from a checkout with `cargo install --path crates/inferlab`. The
published library crates (`inferlab-protocol`, `inferlab-proxy`) exist to
build the binary; their APIs are experimental and carry no stability promise
yet.

### Agent skill

Inferlab ships an operator skill for Claude Code and Codex, embedded in the
binary at the same version — no checkout or network access needed:

```sh
inferlab agent install --agent all
```

`--from-checkout <DIR>` overrides the source with a local checkout or
unpacked release plugin tarball, for testing an unreleased change.

## Quick start

In a workspace (see [`docs/rfc/`](docs/rfc/) for the full contract, starting at RFC-0001):

```sh
pixi install --locked --all                 # realize every stack's selected Pixi environment
inferlab workspace show                     # validate and browse public definitions
inferlab stack status                       # confirm each stack realization
inferlab toolchain install                  # measurement runtimes (only for lm-eval/Bench measurements)

inferlab recipe run my-recipe --dry-run     # validate placement, devices, commands, environment
inferlab recipe run my-recipe --case tp2    # closed loop: serve + eval/bench + cleanup
inferlab serve start my-server --case tp2   # or drive the pieces manually
inferlab bench random-8k1k --serve <ID>
inferlab serve stop <ID>
```

Non-dry-run `serve start`, `recipe run`, `bench`, and `image build` print JSON
naming a record under `.inferlab/records/<ID>/`; `--dry-run` on
those commands resolves and validates without launching or writing one.

## Documentation

- [Workspace authoring](docs/workspace-authoring.md): public definitions,
  local bindings, heterogeneous P/D placement, typed patches, and dry-run.
- [Backend support](docs/backend-support.md): maintained backend capabilities
  and integration package names.
- [RFC-0001 — Specification Overview And Authority Map](docs/rfc/RFC-0001.md):
  the entry point of the normative external contract; topic RFCs under
  [docs/rfc/](docs/rfc/) own workspaces/stacks, servers/execution,
  measurements/toolchains, evidence, the integration protocol, runtime
  images, and agent plugin distribution.
- [Architecture decisions](docs/adr/): accepted ADRs.
- [`plugins/inferlab/skills/inferlab/SKILL.md`](plugins/inferlab/skills/inferlab/SKILL.md):
  the operator workflow, as taught to agents.

## License

MIT — see [LICENSE](LICENSE); `inferlab license` prints the full text.
