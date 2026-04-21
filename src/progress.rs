// =============================================================================
//  progress.rs — ETA tracker + indicatif progress bar
// =============================================================================

use indicatif::{ProgressBar, ProgressStyle};
use std::time::{Duration, Instant};

pub struct Progress {
    bar:     ProgressBar,
    started: Instant,
    total:   u64,
    verbose: bool,
}

impl Progress {
    pub fn new(total: u64, verbose: bool) -> Self {
        let bar = ProgressBar::new(total);

        bar.set_style(
            ProgressStyle::with_template(
                "  {spinner:.cyan} [{bar:44.green/dim}] {pos}/{len}  {msg}",
            )
            .unwrap()
            .progress_chars("█▉▊▋▌▍▎▏ ")
            .tick_strings(&["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]),
        );

        bar.enable_steady_tick(Duration::from_millis(80));
        Self { bar, started: Instant::now(), total, verbose }
    }

    /// Advance by `count` candidates and refresh the ETA message.
    pub fn advance(&self, count: u64) {
        self.bar.inc(count);
        let done    = self.bar.position();
        let elapsed = self.started.elapsed().as_secs_f64();
        if done == 0 || elapsed < 0.5 { return; }

        let rate      = done as f64 / elapsed;
        let remaining = self.total.saturating_sub(done);
        let eta_secs  = (remaining as f64 / rate) as u64;

        self.bar.set_message(format!(
            "│ {:.0}/s │ ETA {}",
            rate,
            fmt_duration(eta_secs)
        ));
    }

    /// Print a single-candidate result line (--verbose only).
    pub fn log_verbose(&self, index: usize, passphrase: &str, matched: bool) {
        if !self.verbose { return; }
        let icon = if matched { "\x1b[32m✓\x1b[0m" } else { "\x1b[31m✗\x1b[0m" };
        self.bar.println(format!("  [{:>7}]  {}  {}", index + 1, icon, passphrase));
    }

    /// Print an informational line above the bar (visible in all modes).
    pub fn log(&self, msg: &str) {
        self.bar.println(msg.to_string());
    }

    /// Call when the search is complete.
    pub fn finish(&self) {
        self.bar.finish_and_clear();
    }

    /// Returns elapsed time since progress tracker was created. Currently unused but available for diagnostics.
    #[allow(dead_code)]
    pub fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }
}

/// Format seconds into a human-readable string.
///   3661 → "1h 01m 01s"
///    90  → "1m 30s"
///    45  → "45s"
pub fn fmt_duration(secs: u64) -> String {
    let h = secs / 3600;
    let m = (secs % 3600) / 60;
    let s = secs % 60;
    if h > 0 {
        format!("{}h {:02}m {:02}s", h, m, s)
    } else if m > 0 {
        format!("{}m {:02}s", m, s)
    } else {
        format!("{}s", s)
    }
}