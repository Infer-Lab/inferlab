# View-only workspace TUI

`inferlab tui` opens a persistent, read-only view of one InferLab workspace.
It uses the same upward discovery as other workspace commands. Select a
workspace explicitly with the global option when needed:

```sh
inferlab tui
inferlab --workspace /path/to/workspace tui
inferlab tui --refresh-interval 2s
```

The refresh override applies only to that invocation; the default is one
second. Opening, refreshing, searching, and closing the TUI do not write a UI
session or launch an InferLab workflow.

## Views and sources

- **Overview** summarizes active Operations and top-level Workflows in plain
  language, then groups visible rows into Attention, Active, Recent, and
  Workspace. Top-level workflow failures appear before abnormal child servers;
  running and abnormal recipe-owned servers remain visible without being
  counted again as top-level workflows. The canonical workspace root is shown
  in the header, with an explicit ellipsis when the terminal cannot fit it. The
  refresh indicator shows the configured automatic cadence while complete
  generations arrive normally. It shows `WAITING` before the first complete
  generation and switches to an elapsed `LAST REFRESH` age only when no new
  generation has completed for two configured refresh intervals.
- **Operations** shows concurrent workspace CLI invocations. Each command owns
  one atomic JSON observation under `.inferlab/runtime/observations/`. A normal
  completion removes it; abrupt residue stays visible and is classified from
  its host, boot, PID, and process-start identity.
  A workspace command fails with `E5002` before continuing when it cannot
  establish this required observation; update failures remove the previous
  live observation and are reported rather than silently degrading.
- **Records** shows top-level records. Recipe-owned server and measurement
  records remain explicit children of the recipe rather than being flattened.
  Details lead with outcome and human-readable start, finish, duration, and age,
  followed by metrics and case state. Exact record identifiers, paths, child
  references, scratchpad references, and explicitly referenced logs remain
  available farther down the detail surface; case stdout and stderr artifacts
  remain mapped to their owning case under **Case Artifacts**. For a workload
  record with case metrics, `m` opens a
  record-local comparison surface. It selects one available metric at a time
  and draws horizontal bars for every case in that record; it does not combine
  records or add an SLO interpretation.
- **Workspace** groups declared models, stacks, servers, Evals, Benches,
  workload suites, recipes, images, and external images by kind, followed by
  scratchpad entries.

Every fact is labeled by its authority: **declared** workspace configuration,
**recorded** evidence or journal text, **ephemeral** command progress, or an
**observed** live check. The TUI does not convert an observation into evidence
or reconstruct a record from definitions.

List rows keep these authorities as subdued `DECL`, `REC`, `EPH`, and `OBS`
badges. A successful read is intentionally implicit. Only exceptional read
health is added to a row, for example `refresh stale` or `refresh unavailable`.
A record row therefore presents its authoritative recorded lifecycle
(`running`, `stopped`, `failed`, or a workload outcome) without making a
successful file read look like another lifecycle. Detail surfaces keep source
authority and read health under **Source Health**, separate from Progress,
Outcome, Process Liveness, Metrics, Context, and Technical References.

For a server record whose recorded status is `running`, each refresh uses the
existing bounded server-status observation to report `OBS process alive`, `OBS
process dead`, `OBS process stale`, or `OBS process unavailable`. A successfully
observed dead process is Attention rather than Active. One failed probe does
not affect other objects; after an earlier successful probe its last value
remains visible as stale.
Terminal server records such as `stopped` are not probed because their recorded
lifecycle is already authoritative and no running process is claimed.
Recipe-owned server records remain children in Records, but a running or
abnormal child server is still observed and surfaced in Overview.

The Metrics selector groups recorded names into throughput, request latency,
TTFT, TPOT, cache, and `OTHER` families. It defaults to request throughput when
that metric exists and retains unknown recorded metric names under `OTHER`.
Concurrency and request-rate cases form separate chart regions and are ordered
by their typed effective load values, including the explicit request evidence
for adaptive probes; case identifiers are not parsed to recover load. A case
whose selected metric is absent remains visible as `—`, and its recorded state
remains distinct from a numeric zero. Wide terminals place the selector beside
the chart. Narrow terminals devote the body to the chart while keeping the
selected metric, metric position, and visible case window in view. Metric
values use compact, stable human precision rather than raw floating-point
serialization.

Source Health distinguishes a current read from `stale`, `unavailable`, or
`incompatible` data. A failed refresh after an earlier success retains the last
successful value as stale with its observation time and age. A malformed or
newer operation schema is isolated to that operation. Refresh generations do
not overlap; missed ticks coalesce, and the prior complete generation remains
navigable while the next one is read. The header's refresh indicator describes
that display schedule only; workspace and object observation health remains on
the corresponding row and detail surface.

## Navigation and search

| Key | Effect |
| --- | --- |
| `1`–`4` | Select Overview, Operations, Records, or Workspace |
| `Tab`, arrows, `j`/`k` | Move views and selection; scroll detail, select a Find result, or select a metric while Metrics is open |
| `PageUp` / `PageDown` | Scroll detail or a referenced log by ten lines, or advance the Metrics case window |
| `Enter` / `Esc` | Open detail / return or close search |
| `m` | Open Metrics for the selected workload record, or return to its record detail |
| `[` / `]` | Select the previous or next explicitly referenced log |
| `Ctrl+K` | Find operations, top-level records, definitions, and scratchpad entries by typed fields |
| `/` | Filter the current list, or search the selected explicitly referenced log tail |
| `r` | Request one immediate non-overlapping refresh |
| `q`, `Ctrl+C` | Exit and restore the terminal |

Wide terminals show list and detail together. Intermediate and narrow terminals
show one pane at a time so neither pane is squeezed into unreadable columns.
List and detail titles retain the selected object and list position while the
content scrolls. Long labels and workspace paths use a visible ellipsis instead
of silent clipping. Empty collections, an empty filter result, and an
unavailable source have different messages. Refresh and quit remain visible
down to the supported minimum; below it, the TUI reports both required and
current dimensions.

Global Find is a dedicated cross-view result surface. Results are labeled by
object kind and authority, with contiguous kinds sectioned visually. The arrow
keys select a result while the query remains open, and `Enter` opens that
selected object. Matching
is case-insensitive and ranks exact, prefix, and contains matches before bounded
fuzzy matches. Each typed field is matched independently, so characters from an
identifier cannot combine with characters from a status or another field to
create a false result. Find never searches raw JSON, environments, artifacts,
arbitrary files, or logs.

`/` filters the current list. After a record or operation detail has loaded an
explicit log reference, `/` opens a search scoped only to the currently selected
log. The sticky detail title identifies the log number and filename; `[` and `]`
switch among references. Search results retain original tail line numbers and a
match count, and searching never changes the selected record or operation.

## Strict view-only boundary

The TUI cannot start or stop servers, launch Eval or Bench work, execute a
recipe, build an image, install tools, edit definitions or notes, clean state,
download data, synchronize logs, or reset caches. Run every such action as a
separate explicit `inferlab` command. This keeps the CLI as the action authority
and records as the durable evidence authority.
