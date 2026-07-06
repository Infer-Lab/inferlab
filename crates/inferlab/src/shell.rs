//! POSIX shell text encoding for rendered scripts and ssh argv.

use std::path::Path;

pub(crate) fn shell_quote_path(path: &Path) -> String {
    shell_quote(&path.to_string_lossy())
}

pub(crate) fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\"'\"'"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shell_quote_preserves_single_quotes() {
        assert_eq!(shell_quote("a'b"), "'a'\"'\"'b'");
    }
}
