use clap::Parser;
use inferlab::{Cli, run};

fn main() {
    if let Err(error) = run(Cli::parse()) {
        eprintln!("error[{}]: {error}", error.code());
        std::process::exit(1);
    }
}
