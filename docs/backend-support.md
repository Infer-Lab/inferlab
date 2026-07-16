# Backend Support Matrix

This public document is the authority for the operator-visible backend
capabilities on the current Inferlab main branch. It describes workflows that
Inferlab plans, executes, and records; it is not a list of every feature offered
by the upstream frameworks.

Status meanings:

- **Qualified**: implemented and demonstrated by a real downstream execution
  record for the baseline named below.
- **Supported**: implemented and covered by deterministic integration tests,
  but not qualified for every relevant hardware and model shape.
- **Limited**: implemented only under the conditions stated in the cell.
- **Unsupported**: a real probe demonstrated a specific conformance failure for
  the exact integration revision, route, model, and baseline named in the cell.
- **Inconclusive**: a real probe ran but its transport or HTTP outcome could not
  establish endpoint capability.
- **Unqualified**: no complete real public-route record establishes the
  capability for the exact path in the cell.
- **—**: rejected by the integration or not exposed by Inferlab.

A qualified baseline is evidence for that concrete shape, not blanket
certification of every framework version, model, device, or parallel configuration.

## Serving And Control

| Capability | vLLM | SGLang | TensorRT-LLM | TokenSpeed |
| --- | --- | --- | --- | --- |
| Integration package | `inferlab-integration-vllm` | `inferlab-integration-sglang` | `inferlab-integration-tensorrt-llm` | `inferlab-integration-tokenspeed` |
| Single-node `single` topology | Qualified | Qualified | Qualified | Qualified |
| Multi-node replica | Supported | — | — | — |
| Disaggregated prefill/decode | Qualified | Qualified for the pairing-specific baselines below | Qualified: built-in proxy and native `trtllm-disaggregated` | Qualified for the maintained 1P1D pairing below |
| KV-transfer backend | Qualified: Mooncake and NIXL | Qualified: Mooncake and NIXL in the pairing-specific baselines below | Qualified: NIXL with the built-in proxy and native `trtllm-disaggregated` | Qualified: Mooncake for the maintained 1P1D pairing below |
| Request routing | Qualified: direct single endpoint, built-in P/D proxy, and vLLM Router | Qualified: direct single endpoint; built-in P/D proxy and SGLang Router in the pairing-specific baselines below | Qualified: direct single endpoint, built-in proxy, and `trtllm-disaggregated` | Qualified: direct single endpoint and native `tokenspeed-smg` P/D routing for the maintained 1P1D pairing below |
| Declared public workload paths | `/v1/completions`; `/v1/chat/completions` | `/v1/completions`; `/v1/chat/completions` | `/v1/completions`; `/v1/chat/completions` | `/v1/completions`; `/v1/chat/completions` |
| Completion request used by Inferlab | Qualified: scalar prompt | Qualified: scalar prompt | Qualified: scalar prompt | Qualified: scalar prompt |
| Chat-completions execution | Supported: direct serving lowers a selected vLLM reasoning parser, and the built-in Mooncake and NIXL P/D proxies preserve the route; no exact public-route record qualifies these paths | Supported by deterministic built-in P/D proxy coverage; the native router and every exact public route remain Unqualified | Supported by deterministic built-in P/D proxy coverage for context-first streaming and non-streaming handoff; the native router and every exact public route remain Unqualified | Unqualified: the named path is preserved; native P/D routing requires separate qualification |
| Prefix-cache reset between cases | Limited: `POST /reset_prefix_cache` for `single`; no reset control on the P/D endpoint | `POST /flush_cache` for `single`; Qualified for the demonstrated P/D pairings below | —; P/D enforces block reuse off at launch | Qualified for `single` and the maintained P/D pairing below through `POST /flush_cache` |
| Framework profiling capture | Supported | — | — | — |

The two named paths are endpoint-contract facts, not route qualification. Chat
execution becomes Qualified only after an Inferlab workflow produces a real
record through the exact integration, route, topology, routing backend, and
model being claimed. Optional upstream API extensions such as embeddings and
batched prompt arrays remain outside this matrix. A pending upstream pull
request or an unreleased dependency does not count as current support.

## lm-eval Loglikelihood Routes

Loglikelihood support is an observed property of the complete public serving
route, not a consequence of scalar completion support. The entries below are
bounded to the published `0.4.0` integration packages and the named maintained
DeepSeek-V4 SM120 baselines. The qualification task was the built-in `hellaswag`
multiple-choice task with the model-directory Hugging Face tokenizer and text
requests. A worker-only probe cannot qualify a prefill/decode public endpoint.

| Public route or qualification boundary | vLLM (`inferlab-integration-vllm==0.4.0`) | SGLang (`inferlab-integration-sglang==0.4.0`) | TensorRT-LLM (`inferlab-integration-tensorrt-llm==0.4.0`) | TokenSpeed (`inferlab-integration-tokenspeed==0.4.0`) |
| --- | --- | --- | --- | --- |
| Direct `single` endpoint | **Qualified**: DeepSeek-V4 SM120 TP2/EP2; the prompt-logprob probe passed tokenizer alignment and `hellaswag` completed with its selected metric gate. | **Unsupported for the probed baseline**: DeepSeek-V4 SM120 TP2/EP2 returned a conforming response shape, but the public endpoint exposed 14 prompt positions for the tokenizer's 13 tokens. | **Unsupported for the probed baseline**: DeepSeek-V4 SM120 TP2/EP2 returned a conforming response shape, but the public endpoint exposed one prompt position for the tokenizer's 13 tokens. | **Inconclusive for the probed baseline**: DeepSeek-V4 SM120 TP2/EP2 returned HTTP 400 to the prompt-logprob probe, so the probe did not establish support or unsupported prompt scoring. |
| Prefill/decode public endpoint, aggregate | **Unqualified**: no complete route-level prompt-logprob record; worker behavior is not qualification evidence. | **Unqualified**: no complete route-level prompt-logprob record; worker behavior is not qualification evidence. | **Unqualified**: no complete route-level prompt-logprob record; worker behavior is not qualification evidence. | **Unqualified**: no complete route-level prompt-logprob record; worker behavior is not qualification evidence. |
| Built-in P/D proxy | **Unqualified**: `0.4.0` declares Mooncake and NIXL cases, but no public-endpoint record establishes a concrete placement or per-role shape. | **Unqualified** for the maintained single-machine 1P1D TP2/EP2 Mooncake and NIXL pairings. | **Unqualified** for the maintained 1P1D TP2/EP2 NIXL path. | —; the maintained P/D path uses the native TokenSpeed router. |
| Upstream or native P/D router | **Unqualified** for the maintained vLLM Router Mooncake and NIXL paths. | **Unqualified** for the maintained SGLang Router Mooncake and NIXL pairings. | **Unqualified** for the maintained native `trtllm-disaggregated` NIXL path. | **Unqualified** for the maintained native `tokenspeed-smg` Mooncake path. |

These direct-route failures do not establish the behavior of another model,
integration release, built-in proxy, or router revision. Requalification must
run the tokenizer-alignment probe and representative task through the exact
public endpoint being claimed.

## Parallelism

The rows below describe which user-requested parallel dimensions the integration
can lower. “Derived” means the effective kernel dimension is calculated from the
declared outer world and the other accepted dimensions rather than configured as
an independent public setting.

| Capability | vLLM | SGLang | TensorRT-LLM | TokenSpeed |
| --- | --- | --- | --- | --- |
| Outer tensor parallelism | Qualified | Qualified | Qualified | Qualified |
| Outer pipeline parallelism | Supported | Supported | Supported | — |
| Attention data parallelism | Supported | Supported | Limited: `1` or the outer TP size | Supported |
| Attention context parallelism | — | Supported | — | — |
| MoE expert parallelism | Qualified | Qualified | Qualified | Qualified |
| MoE data parallelism | — | Supported with topology constraints | — | Supported |
| Independent dense tensor parallelism | — | Supported | — | Qualified |
| Effective expert tensor parallelism | Derived | Derived | Derived | Derived; cannot be greater than `1` together with expert parallelism greater than `1` |

Backend-specific constraints remain validated by the integration and are
reported during planning. This table records the public capability boundary; it
does not duplicate every arithmetic constraint enforced by each adapter.

## Maintained Qualification Baselines

| Backend | Real-hardware baseline | Important boundary |
| --- | --- | --- |
| vLLM | Source-built DeepSeek-V4 SM120 TP2/EP2 serving; real two-machine 1P1D vLLM Router serving with Mooncake and NIXL | Multi-node replica lowering is supported but unqualified; the maintained cross-machine baseline is 1P1D. |
| SGLang | Source-built DeepSeek-V4 SM120 TP2/EP2 serving and pairing-specific single-machine 1P1D serving | P/D qualification is pairing-specific below; TP4 is outside the maintained baseline. |
| TensorRT-LLM | Source-built DeepSeek-V4 SM120 TP2/EP2 serving and 1P1D NIXL serving with built-in and native routing | SM120 DeepSeek-V4 serving requires the source integration's FlashInfer sparse-MLA path; the stock NGC image through 1.3.0rc21 is not sufficient. |
| TokenSpeed | Source-built DeepSeek-V4 SM120 TP2/EP2/dense-TP2 serving; single-machine 1P1D serving with TP2/EP2/dense-TP2 per role, native `tokenspeed-smg` routing, and Mooncake KV transfer | P/D qualification is limited to that concrete routing/transfer pairing; the source-built framework baseline includes its required kernel fixes. |

### SGLang P/D Pairings

The qualified entries use source-built DeepSeek-V4 on SM120 in a
single-machine 1P1D topology with TP2/EP2 per role. Qualification is per
pairing; Supported cells are implemented but have not been separately
qualified on real hardware.

| Routing backend | Mooncake | NIXL |
| --- | --- | --- |
| Built-in P/D proxy | Qualified | Supported |
| SGLang Router | Supported | Qualified |

## Maintenance Rules

Update this document in the same change as an integration when the change
affects any matrix row or qualification statement. In particular:

- use **Supported** for deterministic implementation coverage and **Qualified**
  only after a real record proves the exact workflow and shape;
- name material limitations, including dependencies on downstream framework
  patches, instead of presenting them as general support;
- remove or downgrade capabilities when the integration stops exposing them;
- retain the underlying execution evidence internally, but cite it here only
  when a public qualification artifact exists;
- never expose unpublished internal identifiers, machine-local record paths, or
  private downstream revisions; and
- do not list pending upstream pull requests or future releases as current
  support.
