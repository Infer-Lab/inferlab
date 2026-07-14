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
    for subcommand in [
        "workspace",
        "stack",
        "toolchain",
        "serve",
        "recipe",
        "bench",
        "run",
        "image",
        "scratchpad",
        "agent",
        "license",
    ] {
        assert!(
            stdout
                .lines()
                .any(|line| line.split_whitespace().next() == Some(subcommand)),
            "help must advertise the {subcommand:?} subcommand: {stdout}"
        );
    }
    assert!(
        !stdout.contains("__internal"),
        "help must not advertise the hidden __internal command: {stdout}"
    );
    Ok(())
}
