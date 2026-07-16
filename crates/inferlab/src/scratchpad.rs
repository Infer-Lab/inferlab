//! Operator scratchpad journal: one workspace-local append-only stream per
//! workspace ([[RFC-0005:C-SCRATCHPAD-JOURNAL]]). Records own execution
//! facts; the journal owns the operator narrative. Nothing here feeds
//! resolution, execution, or record content, and no workspace-level mode
//! routes entries — an entry's topic is stated on the entry itself.

use crate::InferlabError;
use crate::progress::{Phase, Progress};
use crate::record::{RECORD_FILE, RECORDS_DIR, now_unix_ms, utc_timestamp, validate_record_id};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{Read, Write};
use std::path::Path;

pub(crate) const SCRATCHPADS_DIR: &str = ".inferlab/scratchpads";
const JOURNAL_FILE: &str = "journal.jsonl";

/// Entries the default view renders before pointing at `--all`.
const DEFAULT_TAIL: usize = 10;

#[derive(Debug, Serialize, Deserialize)]
struct Entry {
    timestamp: String,
    author: String,
    text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    topic: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    records: Vec<String>,
}

/// Minimal tolerant projection of a referenced record for one-line summaries.
#[derive(Debug, Default, Deserialize)]
struct RecordSummary {
    #[serde(default)]
    kind: Option<String>,
    #[serde(default)]
    definition_id: Option<String>,
    #[serde(default)]
    status: Option<String>,
}

pub(crate) fn note_with_progress(
    root: &Path,
    text: &str,
    topic: Option<&str>,
    record_refs: &[String],
    author: Option<&str>,
    progress: &Progress,
) -> Result<String, InferlabError> {
    let topic = match topic.map(str::trim) {
        Some("") => return fail("topic must not be empty".to_owned()),
        other => other.map(str::to_owned),
    };
    let records = resolve_record_refs(root, record_refs)?;
    let dir = root.join(SCRATCHPADS_DIR);
    fs::create_dir_all(&dir).map_err(|source| io_fail(&dir, source))?;
    let path = dir.join(JOURNAL_FILE);
    // An exclusive lock keeps concurrent entries whole and ordered: bare
    // O_APPEND guarantees neither on network filesystems, and the timestamp
    // must be taken inside the lock so the journal's time axis matches
    // append order ([[RFC-0005:C-SCRATCHPAD-JOURNAL]]). The lock releases
    // when the file closes.
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .map_err(|source| io_fail(&path, source))?;
    progress.phase(Phase::named("journal-lock waiting").lock(&path))?;
    file.lock_exclusive()
        .map_err(|source| io_fail(&path, source))?;
    let entry = Entry {
        timestamp: utc_timestamp(now_unix_ms()?)?,
        author: author.map_or_else(default_author, str::to_owned),
        text: text.to_owned(),
        topic,
        records: records.clone(),
    };
    let mut line =
        serde_json::to_vec(&entry).map_err(|source| InferlabError::RecordEncode { source })?;
    line.push(b'\n');
    file.write_all(&line)
        .map_err(|source| io_fail(&path, source))?;
    let mut message = format!("noted {}", entry.timestamp);
    if let Some(topic) = &entry.topic {
        message.push_str(&format!(" · {topic}"));
    }
    for record in &records {
        message.push_str(&format!("\n  -> {record}"));
    }
    Ok(message)
}

pub(crate) fn show_with_progress(
    root: &Path,
    topic: Option<&str>,
    all: bool,
    progress: &Progress,
) -> Result<String, InferlabError> {
    let entries: Vec<Entry> = read_entries(root, progress)?
        .into_iter()
        .filter(|entry| topic.is_none() || entry.topic.as_deref() == topic)
        .collect();
    let mut output = match topic {
        Some(topic) => format!("# journal — topic {topic}\n"),
        None => "# journal\n".to_owned(),
    };
    if entries.is_empty() {
        output.push_str("\nno entries\n");
        return Ok(output);
    }
    let omitted = if all {
        0
    } else {
        entries.len().saturating_sub(DEFAULT_TAIL)
    };
    if omitted > 0 {
        output.push_str(&format!(
            "\n({omitted} earlier entries omitted; --all shows the full journal)\n"
        ));
    }
    for entry in &entries[omitted..] {
        output.push_str(&format!("\n## {} — {}", entry.timestamp, entry.author));
        if let Some(topic) = &entry.topic {
            output.push_str(&format!(" · {topic}"));
        }
        output.push_str(&format!("\n\n{}\n", entry.text));
        for record in &entry.records {
            output.push_str(&format!("\n- {}\n", record_line(root, record)));
        }
    }
    Ok(output)
}

fn read_entries(root: &Path, progress: &Progress) -> Result<Vec<Entry>, InferlabError> {
    let path = root.join(SCRATCHPADS_DIR).join(JOURNAL_FILE);
    if !path.is_file() {
        return Ok(Vec::new());
    }
    // Readers share the lock protocol: the shared lock waits for a
    // concurrent appender to finish its line, so a torn entry is never
    // observed ([[RFC-0005:C-SCRATCHPAD-JOURNAL]]).
    let mut file = fs::File::open(&path).map_err(|source| io_fail(&path, source))?;
    progress.phase(Phase::named("journal-lock waiting").lock(&path))?;
    file.lock_shared()
        .map_err(|source| io_fail(&path, source))?;
    let mut text = String::new();
    file.read_to_string(&mut text)
        .map_err(|source| io_fail(&path, source))?;
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            serde_json::from_str(line).map_err(|source| InferlabError::RecordDecode {
                path: path.clone(),
                source,
            })
        })
        .collect()
}

/// Resolve `--record` references: the keyword `last` selects the newest local
/// record; every reference must name an existing record and is never mutated.
fn resolve_record_refs(root: &Path, refs: &[String]) -> Result<Vec<String>, InferlabError> {
    let records_root = root.join(RECORDS_DIR);
    let mut resolved = Vec::with_capacity(refs.len());
    for reference in refs {
        let id = if reference == "last" {
            newest_record(&records_root)?
        } else {
            reference.clone()
        };
        validate_record_id("record", &id).map_err(|_| InferlabError::Scratchpad {
            message: format!("invalid record reference {id:?}"),
        })?;
        if !records_root.join(&id).join(RECORD_FILE).is_file() {
            return fail(format!("record {id:?} does not exist in {RECORDS_DIR}"));
        }
        resolved.push(id);
    }
    Ok(resolved)
}

fn newest_record(records_root: &Path) -> Result<String, InferlabError> {
    let mut newest: Option<String> = None;
    if records_root.is_dir() {
        let entries = fs::read_dir(records_root).map_err(|source| io_fail(records_root, source))?;
        for entry in entries {
            let entry = entry.map_err(|source| io_fail(records_root, source))?;
            if !entry.path().join(RECORD_FILE).is_file() {
                continue;
            }
            let id = entry.file_name().to_string_lossy().into_owned();
            // Record ids start with the fixed-width UTC timestamp, so the
            // lexically greatest id is the newest record.
            if newest
                .as_deref()
                .is_none_or(|current| id.as_str() > current)
            {
                newest = Some(id);
            }
        }
    }
    newest.map_or_else(
        || fail("no records exist yet; `--record last` has nothing to reference".to_owned()),
        Ok,
    )
}

fn record_line(root: &Path, id: &str) -> String {
    let path = root.join(RECORDS_DIR).join(id).join(RECORD_FILE);
    let summary = fs::read(&path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<RecordSummary>(&bytes).ok());
    match summary {
        Some(summary) => format!(
            "{id} — {} {}: {}",
            summary.kind.as_deref().unwrap_or("record"),
            summary.definition_id.as_deref().unwrap_or("?"),
            summary.status.as_deref().unwrap_or("?"),
        ),
        None => format!("{id} — (record unavailable)"),
    }
}

fn default_author() -> String {
    std::env::var("USER")
        .ok()
        .filter(|user| !user.is_empty())
        .unwrap_or_else(|| "operator".to_owned())
}

fn fail<T>(message: String) -> Result<T, InferlabError> {
    Err(InferlabError::Scratchpad { message })
}

fn io_fail(path: &Path, source: std::io::Error) -> InferlabError {
    InferlabError::RecordIo {
        path: path.to_path_buf(),
        source,
    }
}
