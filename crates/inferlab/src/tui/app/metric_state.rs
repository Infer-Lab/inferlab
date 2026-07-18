use super::{App, AppAction, View};
use crate::tui::presentation::EntryIdentity;
use crate::tui::{EntryKind, RecordView, metrics};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

pub(in crate::tui) struct MetricSelection {
    record_key: String,
    metric_name: String,
    case_scroll: usize,
}

pub(in crate::tui) struct MetricPage<'a> {
    pub(in crate::tui) record: &'a RecordView,
    pub(in crate::tui) catalog: Vec<metrics::MetricDescriptor>,
    pub(in crate::tui) selected: usize,
    pub(in crate::tui) case_scroll: usize,
}

impl App {
    pub(super) fn handle_metric_key(&mut self, key: KeyEvent) -> AppAction {
        match (key.code, key.modifiers) {
            (KeyCode::Char('q'), KeyModifiers::NONE) => AppAction::Quit,
            (KeyCode::Char('r'), KeyModifiers::NONE) => AppAction::Refresh,
            (KeyCode::Char('k'), modifiers) if modifiers.contains(KeyModifiers::CONTROL) => {
                self.start_global_find();
                AppAction::Continue
            }
            (KeyCode::Esc | KeyCode::Left, _) | (KeyCode::Char('m'), KeyModifiers::NONE) => {
                self.metric_selection = None;
                self.detail = true;
                AppAction::Continue
            }
            (KeyCode::Down | KeyCode::Char('j'), _) => {
                self.move_metric(1);
                AppAction::Continue
            }
            (KeyCode::Up | KeyCode::Char('k'), _) => {
                self.move_metric(-1);
                AppAction::Continue
            }
            (KeyCode::PageDown, _) => {
                self.move_metric_cases(10);
                AppAction::Continue
            }
            (KeyCode::PageUp, _) => {
                self.move_metric_cases(-10);
                AppAction::Continue
            }
            (KeyCode::Char(value @ '1'..='4'), KeyModifiers::NONE) => {
                self.select_view((value as usize) - ('1' as usize));
                AppAction::Continue
            }
            (KeyCode::Tab, _) => {
                let current = View::ALL
                    .iter()
                    .position(|view| *view == self.view)
                    .unwrap_or(0);
                self.select_view((current + 1) % View::ALL.len());
                AppAction::Continue
            }
            _ => AppAction::Continue,
        }
    }

    pub(super) fn open_metrics(&mut self) {
        let selected = self
            .selected_entry()
            .and_then(|entry| (entry.kind == EntryKind::Record).then(|| entry.key.clone()));
        let Some(record_key) = selected else {
            return;
        };
        let catalog = self
            .snapshot
            .as_ref()
            .and_then(|snapshot| {
                snapshot
                    .records
                    .iter()
                    .find(|record| record.id.as_deref() == Some(record_key.as_str()))
            })
            .map_or_else(Vec::new, |record| metrics::catalog(&record.cases));
        let Some(metric) = default_metric(&catalog) else {
            self.status = "selected record has no case metrics".to_owned();
            return;
        };
        self.view = View::Records;
        self.rebuild_visible();
        self.reanchor(&EntryIdentity::new(EntryKind::Record, record_key.clone()));
        self.metric_selection = Some(MetricSelection {
            record_key,
            metric_name: metric.name.clone(),
            case_scroll: 0,
        });
        self.detail = true;
        self.status.clear();
    }

    pub(super) fn reconcile_metric_selection(&mut self) {
        let Some(selection) = self.metric_selection.as_ref() else {
            return;
        };
        let record_key = selection.record_key.clone();
        let selected_name = selection.metric_name.clone();
        let Some((catalog, case_count)) = self.snapshot.as_ref().and_then(|snapshot| {
            snapshot
                .records
                .iter()
                .find(|record| record.id.as_deref() == Some(record_key.as_str()))
                .map(|record| (metrics::catalog(&record.cases), record.cases.len()))
        }) else {
            self.metric_selection = None;
            return;
        };
        self.view = View::Records;
        self.rebuild_visible();
        self.reanchor(&EntryIdentity::new(EntryKind::Record, record_key.clone()));
        if catalog.is_empty() {
            self.metric_selection = None;
        } else if !catalog.iter().any(|metric| metric.name == selected_name)
            && let Some(metric) = default_metric(&catalog)
        {
            self.metric_selection = Some(MetricSelection {
                record_key,
                metric_name: metric.name.clone(),
                case_scroll: 0,
            });
        } else if let Some(selection) = &mut self.metric_selection {
            selection.case_scroll = selection.case_scroll.min(case_count.saturating_sub(1));
        }
    }

    pub(in crate::tui) fn metric_page(&self) -> Option<MetricPage<'_>> {
        let selection = self.metric_selection.as_ref()?;
        let record = self
            .snapshot
            .as_ref()?
            .records
            .iter()
            .find(|record| record.id.as_deref() == Some(selection.record_key.as_str()))?;
        let catalog = metrics::catalog(&record.cases);
        let selected = catalog
            .iter()
            .position(|metric| metric.name == selection.metric_name)?;
        Some(MetricPage {
            record,
            catalog,
            selected,
            case_scroll: selection.case_scroll,
        })
    }

    pub(in crate::tui) fn selected_record_has_metrics(&self) -> bool {
        self.selected_entry()
            .filter(|entry| entry.kind == EntryKind::Record)
            .and_then(|entry| {
                self.snapshot.as_ref().and_then(|snapshot| {
                    snapshot
                        .records
                        .iter()
                        .find(|record| record.id.as_deref() == Some(entry.key.as_str()))
                })
            })
            .is_some_and(|record| !metrics::catalog(&record.cases).is_empty())
    }

    fn move_metric(&mut self, delta: isize) {
        let Some(page) = self.metric_page() else {
            self.metric_selection = None;
            return;
        };
        let next = if delta.is_negative() {
            page.selected.saturating_sub(delta.unsigned_abs())
        } else {
            page.selected
                .saturating_add(delta as usize)
                .min(page.catalog.len().saturating_sub(1))
        };
        if let Some(metric) = page.catalog.get(next)
            && let Some(selection) = &mut self.metric_selection
        {
            selection.metric_name = metric.name.clone();
            selection.case_scroll = 0;
        }
    }

    fn move_metric_cases(&mut self, delta: isize) {
        let maximum = self
            .metric_page()
            .map_or(0, |page| page.record.cases.len().saturating_sub(1));
        if let Some(selection) = &mut self.metric_selection {
            selection.case_scroll = if delta.is_negative() {
                selection.case_scroll.saturating_sub(delta.unsigned_abs())
            } else {
                selection
                    .case_scroll
                    .saturating_add(delta as usize)
                    .min(maximum)
            };
        }
    }
}

fn default_metric(catalog: &[metrics::MetricDescriptor]) -> Option<&metrics::MetricDescriptor> {
    catalog
        .iter()
        .find(|metric| metric.name == "request_throughput")
        .or_else(|| catalog.first())
}
