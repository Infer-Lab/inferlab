#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) enum MatchRank {
    Exact,
    Prefix,
    Contains,
    Fuzzy,
}

pub(super) fn match_rank(query: &str, value: &str) -> Option<MatchRank> {
    let query = query.to_lowercase();
    let value = value.to_lowercase();
    match_rank_normalized(&query, &value)
}

fn match_rank_normalized(query: &str, value: &str) -> Option<MatchRank> {
    if value == query || value.split_whitespace().any(|part| part == query) {
        Some(MatchRank::Exact)
    } else if value.split_whitespace().any(|part| part.starts_with(query)) {
        Some(MatchRank::Prefix)
    } else if value.contains(query) {
        Some(MatchRank::Contains)
    } else if value
        .split_whitespace()
        .any(|part| is_bounded_subsequence(query, part))
    {
        Some(MatchRank::Fuzzy)
    } else {
        None
    }
}

#[cfg(test)]
pub(super) fn match_rank_fields(query: &str, fields: &[String]) -> Option<(MatchRank, usize)> {
    let query = query.to_lowercase();
    let fields = fields
        .iter()
        .map(|field| field.to_lowercase())
        .collect::<Vec<_>>();
    match_rank_normalized_fields(&query, &fields)
}

pub(super) fn match_rank_normalized_fields(
    query: &str,
    fields: &[String],
) -> Option<(MatchRank, usize)> {
    fields
        .iter()
        .enumerate()
        .filter_map(|(index, field)| match_rank_normalized(query, field).map(|rank| (rank, index)))
        .min()
}

fn is_bounded_subsequence(query: &str, value: &str) -> bool {
    let query = query.chars().collect::<Vec<_>>();
    if query.len() < 3 {
        return false;
    }
    let mut matched = 0usize;
    let mut first = None;
    let mut last = 0usize;
    for (index, character) in value.chars().enumerate() {
        if query.get(matched) == Some(&character) {
            first.get_or_insert(index);
            last = index;
            matched += 1;
            if matched == query.len() {
                break;
            }
        }
    }
    let Some(first) = first else {
        return false;
    };
    if matched != query.len() {
        return false;
    }
    let span = last.saturating_sub(first) + 1;
    let skipped = span.saturating_sub(query.len());
    skipped <= query.len().div_ceil(2).max(2)
}

#[cfg(test)]
mod tests {
    use super::{MatchRank, match_rank, match_rank_fields};

    #[test]
    fn exact_and_prefix_rank_before_fuzzy() {
        assert_eq!(match_rank("bench", "bench"), Some(MatchRank::Exact));
        assert_eq!(
            match_rank("bench", "bench random-8k1k"),
            Some(MatchRank::Exact)
        );
        assert_eq!(
            match_rank("ben", "bench random-8k1k"),
            Some(MatchRank::Prefix)
        );
        assert_eq!(match_rank("rndm", "random-8k1k"), Some(MatchRank::Fuzzy));
        assert_eq!(match_rank("missing", "bench random-8k1k"), None);
    }

    #[test]
    fn fuzzy_matching_is_bounded_to_one_typed_field_and_one_word() {
        assert_eq!(
            match_rank_fields("brm", &["bench".to_owned(), "random".to_owned()]),
            None
        );
        assert_eq!(match_rank("b81", "bench random-8k1k"), None);
        assert_eq!(
            match_rank_fields("random", &["bench".to_owned(), "random-8k1k".to_owned()]),
            Some((MatchRank::Prefix, 1))
        );
    }
}
