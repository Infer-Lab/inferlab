use std::error::Error;
use std::fs;
use std::process::{Command, Output};
use tempfile::TempDir;

const RECORD_A: &str = "2026-07-04T00-00-00.000Z-bench-1";
const RECORD_B: &str = "2026-07-04T01-00-00.000Z-recipe-2";

struct JournalWorkspace {
    root: TempDir,
}

impl JournalWorkspace {
    fn new() -> Result<Self, Box<dyn Error>> {
        let root = tempfile::tempdir()?;
        fs::create_dir_all(root.path().join(".inferlab"))?;
        fs::write(root.path().join(".inferlab/workspace.toml"), "")?;
        Ok(Self { root })
    }

    fn write_record(&self, id: &str, body: &str) -> Result<(), Box<dyn Error>> {
        let dir = self.root.path().join(".inferlab/records").join(id);
        fs::create_dir_all(&dir)?;
        fs::write(dir.join("record.json"), body)?;
        Ok(())
    }

    fn run(&self, args: &[&str]) -> Result<Output, Box<dyn Error>> {
        Ok(Command::new(env!("CARGO_BIN_EXE_inferlab"))
            .current_dir(self.root.path())
            .args(args)
            .output()?)
    }

    fn ok(&self, args: &[&str]) -> Result<String, Box<dyn Error>> {
        let output = self.run(args)?;
        if !output.status.success() {
            return Err(format!(
                "inferlab {args:?} failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )
            .into());
        }
        Ok(String::from_utf8(output.stdout)?)
    }

    fn fails(&self, args: &[&str]) -> Result<String, Box<dyn Error>> {
        let output = self.run(args)?;
        if output.status.success() {
            return Err(format!("inferlab {args:?} unexpectedly succeeded").into());
        }
        Ok(String::from_utf8_lossy(&output.stderr).into_owned())
    }

    fn entries(&self) -> Result<Vec<serde_json::Value>, Box<dyn Error>> {
        let path = self.root.path().join(".inferlab/scratchpads/journal.jsonl");
        if !path.is_file() {
            return Ok(Vec::new());
        }
        fs::read_to_string(&path)?
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| Ok(serde_json::from_str(line)?))
            .collect()
    }
}

fn assert_record_axis_timestamp(timestamp: &str) {
    // The record-identifier prefix shape: YYYY-MM-DDTHH-MM-SS.mmmZ.
    assert_eq!(
        timestamp.len(),
        24,
        "timestamp {timestamp:?} has wrong width"
    );
    for (index, byte) in timestamp.bytes().enumerate() {
        match index {
            4 | 7 | 13 | 16 => assert_eq!(byte, b'-', "timestamp {timestamp:?} byte {index}"),
            10 => assert_eq!(byte, b'T', "timestamp {timestamp:?} byte {index}"),
            19 => assert_eq!(byte, b'.', "timestamp {timestamp:?} byte {index}"),
            23 => assert_eq!(byte, b'Z', "timestamp {timestamp:?} byte {index}"),
            _ => assert!(
                byte.is_ascii_digit(),
                "timestamp {timestamp:?} byte {index} is not a digit"
            ),
        }
    }
}

#[test]
fn notes_require_no_setup_and_render_with_record_summaries() -> Result<(), Box<dyn Error>> {
    let workspace = JournalWorkspace::new()?;
    workspace.write_record(
        RECORD_A,
        r#"{"schema_version":7,"kind":"bench","definition_id":"random-8k1k","status":"succeeded","cases":[],"summary":{"policy":"highest-feasible-rate-v1"}}"#,
    )?;

    // The very first command against a fresh workspace is a note: no thread
    // to name, nothing to activate.
    workspace.ok(&[
        "scratchpad",
        "note",
        "tp1 hits CUDA OOM at prefill readiness",
        "--record",
        "last",
        "--author",
        "tester",
    ])?;

    let shown = workspace.ok(&["scratchpad", "show"])?;
    assert!(shown.contains("# journal"));
    assert!(shown.contains("— tester"));
    assert!(shown.contains("tp1 hits CUDA OOM at prefill readiness"));
    assert!(
        shown.contains(&format!("- {RECORD_A} — bench random-8k1k: succeeded")),
        "missing one-line record summary in:\n{shown}"
    );
    assert!(!shown.contains("omitted"));
    Ok(())
}

#[test]
fn topics_route_explicitly_and_filter_on_read() -> Result<(), Box<dyn Error>> {
    let workspace = JournalWorkspace::new()?;
    workspace.ok(&["scratchpad", "note", "flash tp2 is up", "--topic", "flash"])?;
    workspace.ok(&[
        "scratchpad",
        "note",
        "pro weights still copying",
        "--topic",
        "pro",
    ])?;
    workspace.ok(&["scratchpad", "note", "both machines reserved tonight"])?;

    let flash = workspace.ok(&["scratchpad", "show", "--topic", "flash"])?;
    assert!(flash.contains("# journal — topic flash"));
    assert!(flash.contains("flash tp2 is up"));
    assert!(!flash.contains("pro weights still copying"));
    assert!(!flash.contains("both machines reserved tonight"));

    let merged = workspace.ok(&["scratchpad", "show"])?;
    assert!(merged.contains("· flash"));
    assert!(merged.contains("· pro"));
    assert!(merged.contains("flash tp2 is up"));
    assert!(merged.contains("pro weights still copying"));
    assert!(merged.contains("both machines reserved tonight"));

    let empty_topic = workspace.fails(&["scratchpad", "note", "text", "--topic", " "])?;
    assert!(
        empty_topic.contains("topic must not be empty"),
        "{empty_topic}"
    );
    Ok(())
}

#[test]
fn record_references_are_validated_without_mutating_records() -> Result<(), Box<dyn Error>> {
    let workspace = JournalWorkspace::new()?;
    let body = r#"{"kind":"recipe","definition_id":"rdma-pd-smoke","status":"succeeded"}"#;
    workspace.write_record(RECORD_B, body)?;

    let missing = workspace.fails(&[
        "scratchpad",
        "note",
        "points nowhere",
        "--record",
        "2026-01-01T00-00-00.000Z-bench-404",
    ])?;
    assert!(missing.contains("does not exist"), "{missing}");
    assert!(
        workspace.entries()?.is_empty(),
        "failed note must append nothing"
    );

    workspace.write_record(
        RECORD_A,
        r#"{"kind":"bench","definition_id":"random-8k1k","status":"succeeded"}"#,
    )?;
    workspace.ok(&["scratchpad", "note", "latest evidence", "--record", "last"])?;

    let entries = workspace.entries()?;
    assert_eq!(entries.len(), 1);
    assert_eq!(
        entries[0]["records"][0].as_str(),
        Some(RECORD_B),
        "`last` must pick the lexically newest record id"
    );
    let after = fs::read_to_string(
        workspace
            .root
            .path()
            .join(".inferlab/records")
            .join(RECORD_B)
            .join("record.json"),
    )?;
    assert_eq!(after, body, "referenced record must not be mutated");
    Ok(())
}

#[test]
fn entries_are_appended_jsonl_on_the_record_id_time_axis() -> Result<(), Box<dyn Error>> {
    let workspace = JournalWorkspace::new()?;
    workspace.ok(&["scratchpad", "note", "first"])?;
    std::thread::sleep(std::time::Duration::from_millis(5));
    workspace.ok(&["scratchpad", "note", "second"])?;

    let entries = workspace.entries()?;
    assert_eq!(entries.len(), 2);
    let first = entries[0]["timestamp"].as_str().ok_or("timestamp")?;
    let second = entries[1]["timestamp"].as_str().ok_or("timestamp")?;
    assert_record_axis_timestamp(first);
    assert_record_axis_timestamp(second);
    assert!(first < second, "append order must be chronological");
    // Record ids carry this exact shape as their prefix — one time axis.
    assert_record_axis_timestamp(&RECORD_A[..24]);
    assert_eq!(entries[0]["text"].as_str(), Some("first"));
    assert_eq!(entries[1]["text"].as_str(), Some("second"));
    Ok(())
}

#[test]
fn default_view_names_omitted_entries_and_all_shows_everything() -> Result<(), Box<dyn Error>> {
    let workspace = JournalWorkspace::new()?;
    for index in 1..=12 {
        workspace.ok(&["scratchpad", "note", &format!("entry-{index}")])?;
    }

    let tail = workspace.ok(&["scratchpad", "show"])?;
    assert!(
        tail.contains("(2 earlier entries omitted; --all shows the full journal)"),
        "{tail}"
    );
    assert!(tail.contains("entry-12"));
    assert!(tail.contains("entry-3"));
    assert!(
        !tail.contains("entry-1\n"),
        "oldest entries must be omitted:\n{tail}"
    );
    assert!(!tail.contains("entry-2\n"));

    let full = workspace.ok(&["scratchpad", "show", "--all"])?;
    assert!(full.contains("entry-1\n"));
    assert!(full.contains("entry-12"));
    assert!(!full.contains("omitted"));
    Ok(())
}

#[test]
fn show_waits_for_a_concurrent_appender_instead_of_reading_a_torn_line()
-> Result<(), Box<dyn Error>> {
    use fs2::FileExt;
    use std::io::Write;
    use std::process::Stdio;

    let workspace = JournalWorkspace::new()?;
    let dir = workspace.root.path().join(".inferlab/scratchpads");
    fs::create_dir_all(&dir)?;
    let path = dir.join("journal.jsonl");
    let mut writer = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)?;
    writer.lock_exclusive()?;
    // Half an entry on disk while the exclusive lock is held: an unlocked
    // reader would decode exactly this torn line.
    writer.write_all(br#"{"timestamp":"2026-07-05T00-00-00.000Z","au"#)?;
    writer.flush()?;

    let child = Command::new(env!("CARGO_BIN_EXE_inferlab"))
        .current_dir(workspace.root.path())
        .args(["scratchpad", "show"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    std::thread::sleep(std::time::Duration::from_millis(800));
    writer.write_all(br#"thor":"op","text":"whole line"}"#)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    drop(writer);

    let output = child.wait_with_output()?;
    assert!(
        output.status.success(),
        "the reader must wait for the appender, not decode the torn line: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout)?;
    assert!(stdout.contains("whole line"), "{stdout}");
    Ok(())
}
