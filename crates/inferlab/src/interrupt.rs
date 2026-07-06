use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

static INTERRUPTED: AtomicBool = AtomicBool::new(false);
static CTRL_C_HANDLER: OnceLock<Result<(), String>> = OnceLock::new();

pub(crate) fn prepare() -> Result<(), String> {
    INTERRUPTED.store(false, Ordering::SeqCst);
    CTRL_C_HANDLER
        .get_or_init(|| {
            ctrlc::set_handler(|| INTERRUPTED.store(true, Ordering::SeqCst))
                .map_err(|error| format!("failed to install Ctrl-C handler: {error}"))
        })
        .clone()
}

pub(crate) fn received() -> bool {
    INTERRUPTED.load(Ordering::SeqCst)
}
