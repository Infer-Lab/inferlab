use std::error::Error;
use std::process::Command;

#[test]
fn help_is_a_runnable_minimal_surface() -> Result<(), Box<dyn Error>> {
    let output = Command::new(env!("CARGO_BIN_EXE_inferlab"))
        .arg("--help")
        .output()?;

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout)?;
    assert!(stdout.contains("Usage: inferlab"));
    assert!(stdout.contains("Inference optimization control plane"));
    // Pin the visible subcommand surface derived from `Command` in
    // `crates/inferlab/src/cli.rs`: every non-hidden top-level subcommand must
    // be advertised, so a rename or a dropped command fails this test.
    for subcommand in [
        "env",
        "toolchain",
        "serve",
        "recipe",
        "bench",
        "run",
        "image",
    ] {
        assert!(
            stdout
                .lines()
                .any(|line| line.split_whitespace().next() == Some(subcommand)),
            "help must advertise the {subcommand:?} subcommand: {stdout}"
        );
    }
    // The internal command is declared `hide = true`; surfacing it in help
    // would leak an implementation-only entrypoint.
    assert!(
        !stdout.contains("__internal"),
        "help must not advertise the hidden __internal command: {stdout}"
    );
    Ok(())
}
