use super::app::App;
use super::collector::Collector;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::Terminal;
use ratatui::backend::TestBackend;
use serde_json::{Value, json};
use std::fs;
use std::time::{Duration, Instant};

const TOP_LEVEL_RECORDS: usize = 1_000;
const CHILD_SERVERS: usize = 20;
const RECORD_PADDING_BYTES: usize = 21_000;

#[test]
#[ignore = "manual scale qualification; prints environment-dependent aggregate timings"]
fn thousand_record_catalog_reports_aggregate_scale_evidence()
-> Result<(), Box<dyn std::error::Error>> {
    let workspace = tempfile::tempdir()?;
    let metadata = workspace.path().join(".inferlab");
    fs::create_dir_all(&metadata)?;
    fs::write(metadata.join("workspace.toml"), "schema_version = 2\n")?;
    let records = metadata.join("records");
    fs::create_dir_all(&records)?;

    let mut top_level_bytes = 0u64;
    for index in 0..TOP_LEVEL_RECORDS {
        let id = format!("record-{index:04}");
        let child = (index < CHILD_SERVERS).then(|| format!("child-server-{index:04}"));
        let body = top_level_record(&id, child.as_deref(), index);
        let encoded = serde_json::to_vec(&body)?;
        top_level_bytes = top_level_bytes.saturating_add(encoded.len() as u64);
        write_record(&records, &id, &encoded)?;
    }
    for index in 0..CHILD_SERVERS {
        let id = format!("child-server-{index:04}");
        let encoded = serde_json::to_vec(&json!({
            "id": id,
            "status": "stopped",
            "started_unix_ms": 1_000 + index,
            "finished_unix_ms": 2_000 + index,
            "process_evidence": {"worker": {}}
        }))?;
        write_record(&records, &id, &encoded)?;
    }
    write_journal(&metadata)?;

    let mut collector = Collector::new(Duration::from_secs(60));
    let mut app = App::default();
    let refresh_status = super::RefreshStatus::Healthy {
        interval: Duration::from_secs(60),
    };
    let mut terminal = Terminal::new(TestBackend::new(120, 40))?;
    let first_started = Instant::now();
    let first = collector.collect(workspace.path(), true);
    let collected_records = first.records.len();
    app.accept(first);
    app.select_view(2);
    terminal.draw(|frame| super::ui::render(frame, &mut app, refresh_status))?;
    let first_frame = first_started.elapsed();

    let steady_started = Instant::now();
    let steady = collector.collect(workspace.path(), false);
    app.accept(steady);
    terminal.draw(|frame| super::ui::render(frame, &mut app, refresh_status))?;
    let steady_refresh = steady_started.elapsed();

    let redraw_started = Instant::now();
    for _ in 0..100 {
        terminal.draw(|frame| super::ui::render(frame, &mut app, refresh_status))?;
    }
    let average_redraw = redraw_started.elapsed() / 100;

    let mut searches = Vec::new();
    for index in 0..20 {
        app.start_global_find();
        let query = format!("record-{:04}", index * 10);
        let started = Instant::now();
        for character in query.chars() {
            let _ = app.handle_key(KeyEvent::new(KeyCode::Char(character), KeyModifiers::NONE));
        }
        terminal.draw(|frame| super::ui::render(frame, &mut app, refresh_status))?;
        searches.push(started.elapsed());
        let _ = app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
    }
    searches.sort_unstable();
    let p95_index = searches
        .len()
        .saturating_mul(95)
        .div_ceil(100)
        .saturating_sub(1);
    let search_p95 = searches.get(p95_index).copied().unwrap_or_default();
    let mean_record_bytes = top_level_bytes / TOP_LEVEL_RECORDS as u64;
    let rss_kib = resident_set_kib().unwrap_or(0);

    assert_eq!(collected_records, TOP_LEVEL_RECORDS);
    assert_eq!(app.visible_len(), TOP_LEVEL_RECORDS);
    println!(
        "records={collected_records} mean_record_bytes={mean_record_bytes} \
         first_frame_ms={} steady_refresh_ms={} search_p95_ms={} \
         average_redraw_us={} rss_kib={rss_kib}",
        first_frame.as_millis(),
        steady_refresh.as_millis(),
        search_p95.as_millis(),
        average_redraw.as_micros(),
    );
    Ok(())
}

fn top_level_record(id: &str, child: Option<&str>, index: usize) -> Value {
    let cases = (0..8)
        .map(|case| {
            json!({
                "id": format!("case-{case:02}"),
                "status": "succeeded",
                "stdout": format!("cases/{case:02}/stdout.log"),
                "stderr": format!("cases/{case:02}/stderr.log"),
                "metrics": {
                    "request_throughput": 10.0 + case as f64,
                    "p50_ttft_ms": 20.0 + case as f64,
                    "p95_ttft_ms": 30.0 + case as f64,
                    "p50_tpot_ms": 4.0 + case as f64,
                    "p95_tpot_ms": 6.0 + case as f64
                }
            })
        })
        .collect::<Vec<_>>();
    let mut body = json!({
        "id": id,
        "status": "succeeded",
        "kind": "bench",
        "definition_id": "qualification-bench",
        "started_unix_ms": 1_000 + index,
        "finished_unix_ms": 2_000 + index,
        "resolved": {
            "execution": {"padding": "x".repeat(RECORD_PADDING_BYTES)}
        },
        "cases": cases
    });
    if let Some(child) = child
        && let Some(object) = body.as_object_mut()
    {
        object.insert("server".to_owned(), json!({"id": child}));
        object.insert("evals".to_owned(), json!([]));
        object.insert("benches".to_owned(), json!([]));
    }
    body
}

fn write_record(records: &std::path::Path, id: &str, encoded: &[u8]) -> Result<(), std::io::Error> {
    let directory = records.join(id);
    fs::create_dir_all(&directory)?;
    fs::write(directory.join("record.json"), encoded)
}

fn write_journal(metadata: &std::path::Path) -> Result<(), std::io::Error> {
    let scratchpads = metadata.join("scratchpads");
    fs::create_dir_all(&scratchpads)?;
    let lines = (0..100)
        .map(|index| {
            json!({
                "timestamp": format!("2026-01-01T00:{:02}:{:02}Z", index / 60, index % 60),
                "author": "operator",
                "text": format!("qualification note {index}"),
                "topic": "qualification",
                "records": [format!("record-{:04}", index * 10)]
            })
            .to_string()
        })
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(scratchpads.join("journal.jsonl"), format!("{lines}\n"))
}

fn resident_set_kib() -> Option<u64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    status.lines().find_map(|line| {
        line.strip_prefix("VmRSS:")?
            .split_whitespace()
            .next()?
            .parse()
            .ok()
    })
}
