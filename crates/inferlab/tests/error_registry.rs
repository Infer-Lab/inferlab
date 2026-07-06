use std::error::Error;

fn is_error_code(text: &str) -> bool {
    text.len() == 5 && text.starts_with('E') && text[1..].chars().all(|c| c.is_ascii_digit())
}

/// The registry table shipped in the rendered specification and the codes
/// error.rs emits must agree exactly. A code added in either place alone
/// fails here, so amending [[RFC-0001:C-ERROR-CODES]] before the
/// implementation is machine-forced, not convention. This lives outside
/// src/ so the published crate carries no test that reads outside the
/// package. Coverage boundary: remapping the sole variant of a code empties
/// that code and fails here; moving one variant between multi-variant codes,
/// or giving a new variant a semantically wrong existing code, keeps the
/// sets equal and is review's job under the clause's append-only and
/// MAY-join rules.
#[test]
fn the_shipped_registry_and_the_emitted_codes_agree() -> Result<(), Box<dyn Error>> {
    let implementation = include_str!("../src/error.rs");
    let emitted: std::collections::BTreeSet<&str> = implementation
        .split('"')
        .filter(|segment| is_error_code(segment))
        .collect();

    let documented: std::collections::BTreeSet<&str> =
        include_str!("../../../docs/rfc/RFC-0001.md")
            .lines()
            .filter_map(|line| line.strip_prefix('|'))
            .filter_map(|rest| rest.split('|').next())
            .map(str::trim)
            .filter(|cell| is_error_code(cell))
            .collect();

    assert!(
        !emitted.is_empty() && !documented.is_empty(),
        "extraction found nothing; the registry table or code() moved"
    );
    assert_eq!(
        emitted, documented,
        "the registry in docs/rfc/RFC-0001.md and error.rs disagree"
    );
    Ok(())
}
