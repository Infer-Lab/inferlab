# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] - 2026-07-18

### Added

- `inferlab tui` provides a persistent, strictly view-only console for one
  discovered or explicitly selected workspace. Its Overview, Operations,
  Records, and Workspace views combine declared definitions, concurrent CLI
  observations, records, referenced logs, and scratchpad context without
  starting or changing an experiment.
- A static product and documentation website publishes the selected public
  guides together with the current RFC and ADR corpus through GitHub Pages,
  with search, locked local preview and production-build tasks, and
  revision-matched CI deployment.
- Public SGLang reference workflows and support documentation cover
  disaggregated prefill/decode serving with Model Gateway, Mooncake, and NIXL,
  while distinguishing execution-qualified pairings from supported but
  unqualified cross-pairings.

### Changed

- The TUI uses a responsive infrastructure-console hierarchy, typed Global Find
  and contextual log search, stable object navigation, and a record-local
  Metrics surface that compares one selected metric across authoritative case
  loads with horizontal bars and explicit missing or failed states.
- The complete Records catalog and Global Find scale to at least 1,000 records
  through source-aware disposable projections, tiered observation cadence, one
  fair refresh-wide active-server probe budget, and redraws driven by observable
  presentation changes rather than the input-poll loop.
- The final capless IL mark and InferLab Blue brand color are applied
  consistently across the website, favicon, plugin identity, and TUI loading
  and accent surfaces; constrained terminals retain a compact text fallback.
- The adapter SDK and each framework integration own independent package
  versions and package-scoped releases. InferLab product releases continue to
  version the Cargo workspace, embedded plugin, and internal measurement
  runners; exact workspace pins preserve artifacts and the adapter protocol
  version remains the runtime compatibility authority.
- Published framework workspace baselines are clean and reproducible, with
  generic local-binding examples and without machine-local state, credentials,
  model locators, or cross-framework package drift.

### Fixed

- TUI rows and details keep recorded lifecycle, observed process liveness, and
  refresh health separate: stopped servers no longer appear live, a dead process
  behind a recorded-running server becomes explicit attention, and a failed
  observation retains its prior value only as stale.
- Healthy automatic refresh shows a stable cadence instead of oscillating
  between `now` and elapsed ages. The indicator reports waiting before its
  first generation, becomes overdue only after two missed intervals, measures
  receipt age monotonically, and recovers after the next completed generation.
- Global Find no longer combines unrelated typed fields into false matches,
  keyboard selection opens the chosen result, referenced-log search retains its
  owning object and selected log, and responsive layouts preserve navigation
  and visible overflow down to the supported minimum terminal size.
- Human-facing data-age labels remain `now` for their complete first second;
  elapsed-duration fields retain subsecond precision.
- Website routes accept either local trailing-slash form while publishing one
  canonical form, human-facing brand text consistently uses InferLab, and
  projected Markdown renders one semantic page title instead of duplicating its
  leading heading.

## [0.4.0] - 2026-07-16

### Added

- Long-running commands now report phases, bounded item progress, lock contention, readiness failures, heartbeats, elapsed time, record directories, and durable log paths on stderr while keeping machine-readable stdout clean.
- lm-eval definitions can select built-in tasks or workspace-local task YAML, including task-owned datasets, splits, prompting, output type, and scoring. The resolved model locator supplies the default tokenizer, and likelihood tasks receive a bounded prompt-logprob and tokenizer-alignment probe before evaluation.
- Eval results now normalize task and metric identity deterministically, preserve raw native output and failure artifacts, and expose explicit transport, endpoint, response-shape, metric-selection, and tokenizer-alignment failures instead of silently changing task semantics.
- The release-owned Eval runtime includes an offline long-context, single-sample generation task with strict terminal-answer scoring. Eval definitions can repeat an eligible task with deterministic per-trial seeds, existing request concurrency, incremental per-trial evidence, and pass rate over issued trials.
- Eval and Bench definitions can carry task-specific OpenAI request parameters, including sampling, logprobs, reasoning effort, and chat-template arguments, with invocation-time nested overrides. Generate and serving workloads use named chat-completions routes, likelihood and smoke workloads use named completions routes, and Inferlab's built-in vLLM, SGLang, and TensorRT-LLM proxies support the corresponding chat path.
- Concurrency Benches can run AIPerf-native warmup before profiling. Normalized results include request latency, TTFT, and TPOT mean/min/max/stddev/p50/p90/p95/p99 plus prompt-cache read ratio when reported.
- Static and adaptive Benches support aggregate SLOs, per-request latency SLOs, minimum good-request ratio, goodput, and an automatically expanding and bisecting highest-feasible-rate search.
- Serving Bench can materialize a release-pinned ShareGPT snapshot into deterministic, tokenizer-bounded single-request populations with content-verified caching and complete acquisition, truncation, population, and native-request identity evidence.

### Changed

- The release-owned Bench environment now uses AIPerf 0.11.0; lm-eval remains pinned at 0.4.12.
- Adapter protocol version 6 makes completions and chat-completions routes explicit and carries the revised Eval and Bench request shapes. The 0.4.0 integration wheels are the tested lockstep set; workspaces pinned to 0.3.0 integrations must bump and relock with the binary.
- Eval and Bench cases now consume one end-to-end case budget beginning at their first case-owned action. Readiness, adapter invocation, container operations, and profiler control likewise have explicit owning deadlines, while cleanup and finalization use separate grace periods and never consume or rewrite the preceding business-operation budget.
- Serving Bench definitions now use the closed `request_source` union. The former flat random-token fields and adaptive `target_metric`, `target_threshold`, and `max_refinement_steps` fields have no compatibility aliases; use `aggregate_slos` and `max_search_steps`.
- Agent commands again delegate orchestration to the agent-plugin-installer batch API and remain outside the long-running-command progress contract.
- Workload plans, typed records, local process groups, measurement-runner operations, Python integration helpers, and server launch mechanisms now have explicit single owners. These internal boundary refactors are intended to preserve existing serving, placement, readiness, interruption, cleanup, and record behavior.

### Fixed

- Official static release binaries now detect the runtime Linux architecture and glibc compatibility from host facts, so measurement toolchain installation works on supported x86_64 and aarch64 glibc hosts instead of inheriting the musl build target.
- TPOT applicability, stable Bench metric names, lm-eval metric selection, operation terminal causes, and cleanup elapsed time now come from one typed authority at each boundary instead of duplicated string or boolean inference.
- A failed or timed-out measurement, adapter, profiler, container, or cleanup operation preserves the established business result, terminal cause, native command evidence, partial artifacts, and verified cleanup outcome without granting nested attempts fresh timeout budgets.

### Security

- Backend qualification remains scoped to the exact demonstrated topology, route, integration revision, and hardware baseline; public support documentation excludes credentials, private hosts, model locators, local paths, record identifiers, and private downstream revisions.

## [0.3.0] - 2026-07-14

### Changed

- Workspace schema version 2 replaces source sets, serving environments, serve profiles, and recipe-owned cases with stacks, independently launchable servers, server-owned cases, and recipes that compose one server with one workload suite. Local placements can bind explicit role/replica/rank allocations, including zero-device proxy ranks and machine-specific model locators.
- The matching CLI uses `workspace lock`, `stack status [STACK]`, `run --stack`, and `serve start <SERVER>`; serve, recipe, and image workflows accept explicit local placement selection.
- Adapter protocol version 5 plans canonical role/replica/rank hierarchies. Each role is the sole authority for effective settings and parallelism; concrete process allocations carry only placement and launch facts. Dry-run, execution, and records now share concrete resolved types, and lifecycle commands reload their complete authority from records without consulting current workspace configuration.
- Ad-hoc `run` resolves only committed stack and image declarations and no longer loads machine-local bindings. Server overrides mirror the typed workspace shape, keep backend values under explicit common or role `settings` paths, and accept quoted TOML key segments including setting names containing literal dots.
- Record identifiers include the workflow and selected server, recipe, Bench, or image name, plus the selected case where applicable and the creating process ID.

### Fixed

- Device hardware evidence now invokes NVIDIA's native `nvidia-smi --query-gpu` spelling while keeping Inferlab's public resource terminology consistently device-based.

## [0.2.0] - 2026-07-13

### Added

- TokenSpeed is now a supported serving integration through the new `inferlab-integration-tokenspeed` package, covering aggregated `ts serve` launches and SMG-routed prefill/decode serving over Mooncake with explicit attention, dense, and MoE parallelism.
- SGLang prefill/decode serving can use either the Inferlab built-in proxy or SGLang Router, independently of Mooncake or NIXL KV transfer.
- TensorRT-LLM prefill/decode serving can use its native disaggregated frontend or the Inferlab built-in proxy over NIXL.
- Framework integrations can render content-addressed launch files; the control plane validates, records, and atomically materializes them for local, SSH, and container launches.
- TensorRT-LLM is now a supported serving framework: the new `inferlab-integration-tensorrt-llm` package plans and renders `trtllm-serve` launches (declare `integration = "tensorrt-llm"` in a serve profile). It maps the shared parallelism vocabulary onto TensorRT-LLM's semantics — attention data parallelism is all-or-nothing (`--enable_attention_dp`), expert parallelism divides the tensor-parallel world — and rejects shapes TensorRT-LLM cannot serve (context parallel, MoE data parallel, dense tensor parallel) at planning time. Framework knobs only reachable through TensorRT-LLM's extra-LLM-API-options YAML (MoE backends, attention-DP balancing, KV block-reuse control) pass through the `extra_llm_api_options` path setting. Note: TensorRT-LLM exposes no prefix-cache flush endpoint, so benches against it cannot request `reset_prefix_cache`; disable KV block reuse at launch instead when a case needs cache isolation. The adapter boundary was smoke-validated against the official release image; the maintained DeepSeek-V4 SM120 baseline is source-built.
- `inferlab run [--environment ID] [--image RECORD | --external-image ID] [--mount PATH[:rw]]... [--gpus SPEC] -- CMD...` runs one ad-hoc command inside a serving environment — a local Pixi install, a built image, or an external image — attached to your terminal and exiting with the command's own status. There are no default mounts; `--mount` binds an absolute host path read-only unless suffixed `:rw`, and `--gpus` exposes an explicit GPU selection to a container.
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
