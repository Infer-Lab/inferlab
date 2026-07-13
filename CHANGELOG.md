# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-07-13

### Added

- TokenSpeed is now a supported serving integration through the new `inferlab-integration-tokenspeed` package, covering aggregated `ts serve` launches and SMG-routed prefill/decode serving over Mooncake with explicit attention, dense, and MoE parallelism.
- SGLang prefill/decode serving can use either the Inferlab built-in proxy or SGLang Router, independently of Mooncake or NIXL KV transfer.
- TensorRT-LLM prefill/decode serving can use its native disaggregated frontend or the Inferlab built-in proxy over NIXL.
- Framework integrations can render content-addressed launch files; the control plane validates, records, and atomically materializes them for local, SSH, and container launches.
- TensorRT-LLM is now a supported serving framework: the new `inferlab-integration-tensorrt-llm` package plans and renders `trtllm-serve` launches (declare `integration = "tensorrt-llm"` in a serve profile). It maps the shared parallelism vocabulary onto TensorRT-LLM's semantics — attention data parallelism is all-or-nothing (`--enable_attention_dp`), expert parallelism divides the tensor-parallel world — and rejects shapes TensorRT-LLM cannot serve (context parallel, MoE data parallel, dense tensor parallel) at planning time. Framework knobs only reachable through TensorRT-LLM's extra-LLM-API-options YAML (MoE backends, attention-DP balancing, KV block-reuse control) pass through the `extra_llm_api_options` path setting. Note: TensorRT-LLM exposes no prefix-cache flush endpoint, so benches against it cannot request `reset_prefix_cache`; disable KV block reuse at launch instead when a case needs cache isolation. The adapter boundary was smoke-validated against the official release image; the maintained DeepSeek-V4 SM120 baseline is source-built.
- `inferlab run [--environment ID] [--image RECORD | --external-image ID] [--mount PATH[:rw]]... [--gpus SPEC] -- CMD...` runs one ad-hoc command inside a serving environment — a local Pixi install, a built image, or an external image — attached to your terminal and exiting with the command's own status. There are no default mounts; `--mount` binds an absolute host path read-only unless suffixed `:rw`, and `--gpus` exposes an explicit device selection to a container.
- `inferlab env status [--environment ID]` reports whether each declared serving environment is `confirmed`, `never-installed`, or `not-usable`, as JSON, without needing local machine bindings or installing anything — useful right after a fresh checkout or a `git pull` to check before you launch anything. Exits non-zero if any environment isn't confirmed.
- A successful environment check is now remembered against the exact Pixi manifest and lock content that produced it, so a launch that finds nothing changed skips re-probing Pixi entirely; any edit to the manifest or lock invalidates the memory and forces a fresh check. `inferlab run` deliberately does not participate in this — it neither trusts nor produces this evidence, so an ad-hoc command can never make a real launch skip a check it should have made, or vice versa.
- `inferlab agent install` no longer needs a repository checkout: the Claude Code / Codex plugin package now ships embedded in the binary itself, so `inferlab agent install --agent all` works immediately after installing the CLI, offline. `--from-checkout <DIR>` is still available for testing local edits to the plugin before a release.
- Interrupting a recipe now reliably cleans up the eval/bench measurement processes it started, including a background sweep that catches any survivor left behind by an unclean exit.
- A toolchain removal that fails because something still has the install path open now names the exact holding process(es) in the error.

### Changed

- Recipe record IDs now include the selected recipe and case, omit process IDs, and use collision suffixes when needed.
- `inferlab agent install` defaults to the binary-embedded plugin package described above; `--from-checkout` remains a fully-supported explicit override.
- `scripts/install.sh` no longer downloads or unpacks the plugin package separately — it's already inside the binary it just installed.
- The operator skill now routes ad-hoc environment commands through `inferlab run` and calls out that invoking an interpreter or tool binary directly from inside a materialized environment prefix is unsupported.

### Fixed

- Bench requests to `/v1/completions` now preserve each synthetic prompt as a scalar string, allowing OpenAI-compatible servers without batched-prompt support to run the shared AIPerf workload.
- vLLM Router readiness can no longer preempt Inferlab's own readiness endpoint during prefill/decode startup.
- A serving environment that was never installed at all used to silently fall through to whatever happened to be on the ambient `PATH` instead of failing; environment checks (local and over SSH) now correctly catch this before anything launches, and separately catch an installed environment that's gone stale relative to a since-regenerated lock.
- The `inferlab` binary crate now packages and compiles cleanly from its published crate form (`cargo package`/`cargo publish`), with a test pinning the in-crate toolchain payload copies byte-identical to their Python sources so this can't silently regress.
- Serving from an external image could not report which adapter version did the lowering: the in-container adapter invocation saw the packages' code but not their distribution metadata. Each package's metadata now travels with its code, so records carry the exact pinned adapter version even for external-image launches.
- Stopping a server container launched with auto-remove could race the Docker daemon's own removal and get recorded as unverified cleanup even though the container was gone moments later; the stop now confirms by watching the container actually disappear (bounded), and only reports unverified if it never does.

## [0.1.0] - 2026-07-05

Initial release.

### Added

**Workspace model** — Workspaces declare recipes, serve profiles, serving environments, models, benchmarks, and correctness cases as typed, validated configuration, composed from `.inferlab/workspace.d/*.toml` fragments. Workspace state is tied to the git revision and content digest of the source tree, including submodules; a dirty working tree is recorded honestly instead of silently ignored. Local, machine-specific facts (model weight paths, machine bindings) live separately in `.inferlab/local.toml`, kept out of the versioned workspace definition. Workspace source integrity is enforced structurally — no path in the workspace configuration or a declared source set may be, or resolve through, a symlink that escapes the workspace root.

**Pixi environment lifecycle** — `inferlab env lock` produces the authoritative full workspace Pixi lock from a clean local prefix, with no manual manifest edits. Every workflow that touches a serving environment activates it read-only and fails before doing anything else if it isn't installed or the lock doesn't match the manifest — Inferlab itself never installs packages or updates the lock.

**Serving and recipes** — `serve start`/`status`/`logs`/`stop` and `recipe run` share one server lifecycle covering single- and multi-node deployments, local and over SSH, with full dry-run validation before anything launches. `recipe run` executes the closed loop end to end — start the server, wait for readiness, run the recipe's Eval gate and any eligible Benches, then tear the server down — recording every case, server, and measurement outcome in one aggregate record. Multi-node network setup is automatic: Inferlab probes every machine for a routable interface common to the whole placement and wires it into NCCL. Each server process gets its own deterministic runtime cache directory, derived from the workspace, environment, machine, and process identity.

**Measurement (Eval and Bench)** — `inferlab toolchain install` installs Inferlab's own release-pinned Eval (lm-eval-based) and Bench (AIPerf-based) runtimes, kept separate from your serving environment, on both x86_64 and aarch64 Linux. Bench runs go through a typed runner that translates recipe-declared cases into AIPerf configuration and returns normalized results and cleanup evidence even on failure or interruption. A standalone OpenAI-compatible smoke check needs no measurement toolchain at all. Workload profiling captures a Nsight Systems trace of a serving run, keyed to the actual assigned GPUs, with configurable capture windows and escape hatches for advanced `nsys` options.

**Framework integrations** — vLLM, including single-role and disaggregated prefill/decode (Mooncake and NIXL) topologies with a built-in reverse proxy or an external vLLM Router; and SGLang, using the shared tensor/data/expert/pipeline parallelism vocabulary.

**Runtime images** — `inferlab image build` assembles an OCI runtime image from a workspace's serving closure, deduplicating identical closures across requested model-validation targets and validating each output by actually running the eval/bench suite against it. Serving from a pre-built image reuses the same recipe and measurement machinery as a live install, including on multi-node and SSH placements. Workspaces can also declare a digest-pinned *external* image that Inferlab didn't build — Inferlab verifies it's present with the right digest on every machine that needs it and never pulls it for you. Declared environment checks and image postprocessing hooks run at defined points (image build, inside the built image, and host preflight) so per-site fixups are explicit and recorded.

**Distribution** — `inferlab agent install|update|uninstall|doctor` installs Inferlab's operator skill into Claude Code and/or Codex. Tagged releases publish Linux binaries (x86_64/aarch64), a reproducible plugin tarball, Python wheels for the workspace-side adapter packages, and the Rust crates, all stamped with one version, with the MIT license retained in every distributed form. A stable, append-only error-code registry means every runtime failure exits with exactly one `error[<code>]` diagnostic, and published codes never change meaning.

**Scratchpad** — `inferlab scratchpad note`/`show`, an append-only, file-first operator journal, with entries optionally tied to a specific record.

### Changed

- Multi-node placements express machine facts (hosts, devices, launch access, endpoints, execution-visible paths) in local, machine-specific bindings, keeping recipes themselves machine-independent.
- Runtime cache storage roots are configurable per machine, with a workspace-local default when none is bound.

### Removed

- The old adapter-mediated Eval and Bench request/response path is gone in favor of direct typed runner requests; framework integrations no longer lower ordinary Bench operations, only server-specific control.
- The binary no longer embeds or materializes workspace-side Python packages — those now ship as ordinary wheels from the package index, so a binary upgrade can no longer silently change adapter behavior under an unchanged workspace commit.

### Fixed

- Numerous image-build correctness and caching fixes: cache keys now cover every input that actually affects the built content (source-set paths, Pixi manifest digest, target platform, build-procedure identity), export archives are named so concurrent or repeated builds can't clobber each other's evidence, cache publication is safe under concurrent builds, and a workspace mutated during a build now fails the build loudly instead of shipping silently-wrong output.
- Container launches no longer leak: adapter containers are terminated and removed through an owned handle rather than left to the docker client, removal is bounded by a deadline instead of hanging, and an unconfirmed removal is reported honestly rather than assumed clean.
- Activation values that could break shell quoting or leak credentials are rejected at render time; container pass-through environment variables are validated and passed by name only, never by value.
- A disaggregated-serving streaming correctness fix: prefill no longer sends more tokens than the forced single-token prefill maximum allows.

### Security

- Model weights, weight locations, credentials, and any other undeclared private workspace content never enter a built image, its OCI output, or a shareable manifest.
