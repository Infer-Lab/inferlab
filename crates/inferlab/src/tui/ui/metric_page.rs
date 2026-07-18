use super::text::{display_width, ellipsize_end, pad_left, pad_right};
use super::theme::{ACCENT, ACCENT_SOFT, CRITICAL, MUTED, SECONDARY, WARNING};
use crate::tui::metrics::{self, MetricDescriptor, MetricPoint, MetricUnit};
use crate::tui::{App, WIDE_WIDTH};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

pub(super) fn render(frame: &mut ratatui::Frame<'_>, app: &App, area: Rect) {
    let Some(page) = app.metric_page() else {
        frame.render_widget(
            Paragraph::new("Selected record metrics are no longer available")
                .style(Style::default().fg(WARNING)),
            area,
        );
        return;
    };
    let Some(metric) = page.catalog.get(page.selected) else {
        return;
    };
    if area.width >= WIDE_WIDTH {
        let columns = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Length(32),
                Constraint::Length(1),
                Constraint::Min(20),
            ])
            .split(area);
        render_selector(frame, columns[0], &page.catalog, page.selected);
        render_chart(frame, columns[2], &page, metric);
    } else {
        render_chart(frame, area, &page, metric);
    }
}

fn render_selector(
    frame: &mut ratatui::Frame<'_>,
    area: Rect,
    catalog: &[MetricDescriptor],
    selected: usize,
) {
    let mut lines = Vec::new();
    let mut selected_line = 0usize;
    let mut family = None;
    for (index, metric) in catalog.iter().enumerate() {
        if family != Some(metric.family) {
            if !lines.is_empty() {
                lines.push(Line::from(""));
            }
            lines.push(Line::from(Span::styled(
                format!("  {}", metric.family.label()),
                Style::default()
                    .fg(ACCENT_SOFT)
                    .add_modifier(Modifier::BOLD),
            )));
            family = Some(metric.family);
        }
        if index == selected {
            selected_line = lines.len();
        }
        lines.push(Line::from(vec![
            Span::styled(
                if index == selected { "▸ " } else { "  " },
                Style::default().fg(ACCENT),
            ),
            Span::styled(
                ellipsize_end(&metric.label, usize::from(area.width).saturating_sub(2)),
                if index == selected {
                    Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(SECONDARY)
                },
            ),
        ]));
    }
    let visible_height = area.height.saturating_sub(1) as usize;
    let scroll = selected_line.saturating_sub(visible_height.saturating_sub(2));
    frame.render_widget(
        Paragraph::new(lines)
            .block(
                Block::default()
                    .borders(Borders::TOP)
                    .border_style(Style::default().fg(MUTED))
                    .title(Span::styled(
                        " METRIC SELECTOR ",
                        Style::default().fg(ACCENT_SOFT),
                    )),
            )
            .scroll((u16::try_from(scroll).unwrap_or(u16::MAX), 0)),
        area,
    );
}

fn render_chart(
    frame: &mut ratatui::Frame<'_>,
    area: Rect,
    page: &crate::tui::app::MetricPage<'_>,
    metric: &MetricDescriptor,
) {
    let record_context = compact_record_context(page.record);
    let context_width = usize::from(area.width)
        .saturating_sub(display_width(" METRICS · ") + display_width(" · REC "));
    let block = Block::default()
        .borders(Borders::TOP)
        .border_style(Style::default().fg(MUTED))
        .title(Line::from(vec![
            Span::styled(" METRICS ", Style::default().fg(ACCENT_SOFT)),
            Span::styled("· ", Style::default().fg(MUTED)),
            Span::styled(
                ellipsize_end(&record_context, context_width),
                Style::default().fg(SECONDARY),
            ),
            Span::styled(" · REC ", Style::default().fg(ACCENT_SOFT)),
        ]));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(1)])
        .split(inner);
    let points = metrics::points(&page.record.cases, &metric.name);
    let case_start = app_scroll(page, points.len());
    let case_end = visible_case_end(&points, case_start, usize::from(rows[1].height));
    render_metric_header(
        frame,
        rows[0],
        page,
        metric,
        case_start,
        case_end,
        points.len(),
    );
    let scroll = chart_line_offset(&points, case_start);
    frame.render_widget(
        Paragraph::new(chart_lines(&points, metric, rows[1].width))
            .scroll((u16::try_from(scroll).unwrap_or(u16::MAX), 0))
            .wrap(Wrap { trim: false }),
        rows[1],
    );
}

fn render_metric_header(
    frame: &mut ratatui::Frame<'_>,
    area: Rect,
    page: &crate::tui::app::MetricPage<'_>,
    metric: &MetricDescriptor,
    case_start: usize,
    case_end: usize,
    case_count: usize,
) {
    let unit = metric.unit.label();
    let suffix = if unit.is_empty() {
        String::new()
    } else {
        format!("  {unit}")
    };
    let position = format!("{}/{}", page.selected + 1, page.catalog.len());
    let heading_width = usize::from(area.width)
        .saturating_sub(display_width(&position) + display_width(&suffix) + display_width("  "));
    let case_suffix = format!(
        "  ·  cases {}–{}/{}",
        case_start.saturating_add(1).min(case_count),
        case_end,
        case_count
    );
    let name_width = usize::from(area.width).saturating_sub(display_width(&case_suffix));
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(vec![
                Span::styled(
                    format!("{}  ", ellipsize_end(&metric.heading(), heading_width)),
                    Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
                ),
                Span::styled(position, Style::default().fg(MUTED)),
                Span::styled(suffix, Style::default().fg(SECONDARY)),
            ]),
            Line::from(vec![
                Span::styled(
                    ellipsize_end(&metric.name, name_width),
                    Style::default().fg(MUTED),
                ),
                Span::styled(case_suffix, Style::default().fg(MUTED)),
            ]),
            Line::from(scale_line(page, metric, area.width)),
        ]),
        area,
    );
}

fn scale_line(
    page: &crate::tui::app::MetricPage<'_>,
    metric: &MetricDescriptor,
    width: u16,
) -> Vec<Span<'static>> {
    let maximum = scale_maximum(
        &metrics::points(&page.record.cases, &metric.name),
        metric.unit,
    );
    let label = value_text(maximum, metric.unit);
    let rule_width = usize::from(width).saturating_sub(display_width(&label) + 12);
    vec![
        Span::styled("scale  0 ", Style::default().fg(MUTED)),
        Span::styled("─".repeat(rule_width), Style::default().fg(ACCENT_SOFT)),
        Span::styled(format!(" {label}"), Style::default().fg(MUTED)),
    ]
}

fn chart_lines(
    points: &[MetricPoint],
    metric: &MetricDescriptor,
    width: u16,
) -> Vec<Line<'static>> {
    let maximum = scale_maximum(points, metric.unit);
    let values = points
        .iter()
        .map(|point| {
            point
                .value
                .map(|value| value_text(metric.unit.display_value(value), metric.unit))
                .unwrap_or_else(|| "—".to_owned())
        })
        .collect::<Vec<_>>();
    let available_width = usize::from(width);
    let label_limit = (available_width / 3).clamp(4, 16);
    let label_width = points
        .iter()
        .map(|point| display_width(&point.label))
        .max()
        .unwrap_or(4)
        .clamp(4, label_limit);
    let value_limit = (available_width / 4).clamp(6, 24);
    let value_width = values
        .iter()
        .map(|value| display_width(value))
        .max()
        .unwrap_or(1)
        .clamp(1, value_limit);
    let status_limit = (available_width / 5).clamp(6, 12);
    let status_width = points
        .iter()
        .filter(|point| point.status != "succeeded")
        .map(|point| display_width(&point.status))
        .max()
        .unwrap_or(0)
        .min(status_limit);
    let bar_width = available_width
        .saturating_sub(label_width + value_width + status_width + 5)
        .max(1);
    let mut lines = Vec::new();
    let mut group = None;
    for (index, point) in points.iter().enumerate() {
        if group != Some(point.group) {
            lines.push(Line::from(Span::styled(
                point.group.label(),
                Style::default()
                    .fg(ACCENT_SOFT)
                    .add_modifier(Modifier::BOLD),
            )));
            group = Some(point.group);
        }
        let displayed = point.value.map(|value| metric.unit.display_value(value));
        let fill = displayed.map_or(0, |value| {
            if maximum <= 0.0 || value <= 0.0 {
                0
            } else {
                ((value / maximum) * bar_width as f64)
                    .round()
                    .clamp(0.0, bar_width as f64) as usize
            }
        });
        let status = if point.status == "succeeded" {
            String::new()
        } else {
            point.status.clone()
        };
        lines.push(Line::from(vec![
            Span::styled(
                format!("{} ", pad_right(&point.label, label_width)),
                Style::default().fg(SECONDARY),
            ),
            Span::styled("█".repeat(fill), Style::default().fg(ACCENT)),
            Span::raw(" ".repeat(bar_width.saturating_sub(fill))),
            Span::styled(
                format!(" {}", pad_left(&values[index], value_width)),
                Style::default().fg(if point.value.is_some() {
                    SECONDARY
                } else {
                    MUTED
                }),
            ),
            Span::styled(
                format!(" {}", pad_left(&status, status_width)),
                Style::default().fg(status_color(&point.status)),
            ),
        ]));
    }
    lines
}

fn app_scroll(page: &crate::tui::app::MetricPage<'_>, point_count: usize) -> usize {
    page.case_scroll.min(point_count.saturating_sub(1))
}

fn chart_line_offset(points: &[MetricPoint], case_start: usize) -> usize {
    if case_start == 0 {
        return 0;
    }
    let mut lines = 0usize;
    let mut group = None;
    for (index, point) in points.iter().enumerate() {
        if index == case_start {
            return lines;
        }
        if group != Some(point.group) {
            lines += 1;
            group = Some(point.group);
        }
        lines += 1;
    }
    lines
}

fn visible_case_end(points: &[MetricPoint], case_start: usize, visible_lines: usize) -> usize {
    if points.is_empty() || visible_lines == 0 {
        return case_start.min(points.len());
    }
    let mut used = usize::from(case_start == 0);
    let mut end = case_start.min(points.len());
    let mut previous_group = case_start
        .checked_sub(1)
        .and_then(|index| points.get(index))
        .map(|point| point.group);
    for point in points.iter().skip(case_start) {
        if previous_group.is_some() && previous_group != Some(point.group) {
            used += 1;
        }
        if used >= visible_lines {
            break;
        }
        used += 1;
        end += 1;
        previous_group = Some(point.group);
    }
    end
}

fn compact_record_context(record: &crate::tui::RecordView) -> String {
    let id = record.id.as_deref().unwrap_or("unreadable-record");
    let without_time = id.split_once("Z-").map_or(id, |(_, remainder)| remainder);
    let label = without_time
        .strip_prefix(&record.kind)
        .and_then(|remainder| remainder.strip_prefix('-'))
        .unwrap_or(without_time);
    format!("{} · {label}", record.kind)
}

fn scale_maximum(points: &[MetricPoint], unit: MetricUnit) -> f64 {
    if unit == MetricUnit::Ratio {
        return 100.0;
    }
    points
        .iter()
        .filter_map(|point| point.value)
        .map(|value| unit.display_value(value))
        .filter(|value| value.is_finite() && *value > 0.0)
        .max_by(f64::total_cmp)
        .unwrap_or(0.0)
}

fn value_text(value: f64, unit: MetricUnit) -> String {
    let number = human_number(value);
    if unit.label().is_empty() {
        number
    } else {
        format!("{number} {}", unit.label())
    }
}

fn human_number(value: f64) -> String {
    if !value.is_finite() {
        return value.to_string();
    }
    if value == 0.0 {
        return "0".to_owned();
    }
    let absolute = value.abs();
    if absolute > 0.0 && absolute < 0.001 {
        return format!("{value:.3e}");
    }
    let decimals = if absolute >= 1_000.0 {
        0
    } else if absolute >= 100.0 {
        1
    } else if absolute >= 10.0 {
        2
    } else if absolute >= 1.0 {
        3
    } else if absolute >= 0.1 {
        4
    } else if absolute >= 0.01 {
        5
    } else {
        6
    };
    let formatted = format!("{value:.decimals$}");
    if decimals == 0 {
        return formatted;
    }
    formatted
        .trim_end_matches('0')
        .trim_end_matches('.')
        .to_owned()
}

fn status_color(status: &str) -> ratatui::style::Color {
    match status {
        "failed" => CRITICAL,
        "succeeded" => MUTED,
        _ => WARNING,
    }
}

#[cfg(test)]
mod tests {
    use super::{chart_line_offset, value_text};
    use crate::tui::metrics::{self, MetricUnit};
    use crate::tui::{CaseLoad, CaseView};
    use std::collections::BTreeMap;

    #[test]
    fn metric_values_use_stable_human_precision() {
        assert_eq!(
            value_text(7.412500701361551, MetricUnit::RequestsPerSecond),
            "7.413 req/s"
        );
        assert_eq!(value_text(47.8, MetricUnit::Milliseconds), "47.8 ms");
        assert_eq!(value_text(1000.4, MetricUnit::None), "1000");
        assert_eq!(value_text(0.0123456, MetricUnit::Ratio), "0.01235 %");
        assert_eq!(value_text(-0.0, MetricUnit::None), "0");
    }

    #[test]
    fn case_window_keeps_the_heading_at_a_load_group_boundary() {
        let case = |id: &str, load| CaseView {
            id: Some(id.to_owned()),
            load,
            status: Some("succeeded".to_owned()),
            stdout: None,
            stderr: None,
            error: None,
            metrics: BTreeMap::from([("request_throughput".to_owned(), 1.0)]),
        };
        let points = metrics::points(
            &[
                case("concurrency", CaseLoad::Concurrency(1)),
                case("rate", CaseLoad::RequestRate(1.0)),
            ],
            "request_throughput",
        );

        assert_eq!(chart_line_offset(&points, 1), 2);
    }
}
