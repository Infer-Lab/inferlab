use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

const ELLIPSIS: &str = "…";

pub(super) fn display_width(value: &str) -> usize {
    UnicodeWidthStr::width(value)
}

pub(super) fn ellipsize_end(value: &str, max_width: usize) -> String {
    ellipsize(value, max_width, false)
}

pub(super) fn ellipsize_start(value: &str, max_width: usize) -> String {
    ellipsize(value, max_width, true)
}

pub(super) fn ellipsize_middle(value: &str, max_width: usize) -> String {
    if display_width(value) <= max_width {
        return value.to_owned();
    }
    let ellipsis_width = display_width(ELLIPSIS);
    if max_width < ellipsis_width {
        return String::new();
    }
    let available = max_width - ellipsis_width;
    let left_width = available / 2;
    let right_width = available - left_width;
    format!(
        "{}{}{}",
        take_width(value, left_width, false),
        ELLIPSIS,
        take_width(value, right_width, true)
    )
}

pub(super) fn pad_right(value: &str, width: usize) -> String {
    let value = ellipsize_end(value, width);
    format!(
        "{value}{}",
        " ".repeat(width.saturating_sub(display_width(&value)))
    )
}

pub(super) fn pad_left(value: &str, width: usize) -> String {
    let value = ellipsize_start(value, width);
    format!(
        "{}{value}",
        " ".repeat(width.saturating_sub(display_width(&value)))
    )
}

fn ellipsize(value: &str, max_width: usize, from_start: bool) -> String {
    if display_width(value) <= max_width {
        return value.to_owned();
    }
    let ellipsis_width = display_width(ELLIPSIS);
    if max_width < ellipsis_width {
        return String::new();
    }
    let retained = take_width(value, max_width - ellipsis_width, from_start);
    if from_start {
        format!("{ELLIPSIS}{retained}")
    } else {
        format!("{retained}{ELLIPSIS}")
    }
}

fn take_width(value: &str, max_width: usize, from_end: bool) -> String {
    let graphemes = UnicodeSegmentation::graphemes(value, true);
    if from_end {
        let mut retained = Vec::new();
        let mut width = 0usize;
        for grapheme in graphemes.rev() {
            let grapheme_width = display_width(grapheme);
            if width.saturating_add(grapheme_width) > max_width {
                break;
            }
            retained.push(grapheme);
            width += grapheme_width;
        }
        retained.into_iter().rev().collect()
    } else {
        let mut retained = String::new();
        let mut width = 0usize;
        for grapheme in graphemes {
            let grapheme_width = display_width(grapheme);
            if width.saturating_add(grapheme_width) > max_width {
                break;
            }
            retained.push_str(grapheme);
            width += grapheme_width;
        }
        retained
    }
}

#[cfg(test)]
mod tests {
    use super::{display_width, ellipsize_end, ellipsize_middle, ellipsize_start};

    #[test]
    fn ellipsis_respects_terminal_cells_and_grapheme_boundaries() {
        let end = ellipsize_end("模型评测结果", 7);
        let start = ellipsize_start("模型评测结果", 7);
        let middle = ellipsize_middle("模型评测结果", 7);

        assert!(display_width(&end) <= 7 && end.ends_with('…'));
        assert!(display_width(&start) <= 7 && start.starts_with('…'));
        assert!(display_width(&middle) <= 7 && middle.contains('…'));
    }
}
