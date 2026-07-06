use inferlab_protocol::protocol_schema;
use std::error::Error;
use std::io::{Error as IoError, ErrorKind};
use std::path::PathBuf;

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let output = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or_else(|| IoError::new(ErrorKind::InvalidInput, "missing output path"))?;
    let mut rendered = serde_json::to_string_pretty(&protocol_schema())?;
    rendered.push('\n');
    std::fs::write(output, rendered)?;
    Ok(())
}
