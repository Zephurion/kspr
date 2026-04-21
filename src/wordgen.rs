// =============================================================================
//  wordgen.rs — Phase 2 brute-force generator  (STUB)
//
//  This module is intentionally unimplemented.  The contract below is the
//  full specification so it can be filled in without touching any other file.
//
//  ── What it should do ────────────────────────────────────────────────────────
//
//  After the wordlist phase fails, Phase 2 generates candidates algorithmically
//  and streams them directly into the GPU batcher.  No pre-built list is kept
//  in memory — each thread produces one candidate at a time on demand.
//
//  ── Parallelism model ────────────────────────────────────────────────────────
//
//  The total search space is split into `num_threads` equal contiguous slices.
//  Thread i owns the slice  [i * (space/N) .. (i+1) * (space/N)).
//  Each thread runs a WordgenIter that tracks its own position with no
//  shared mutable state — iteration is purely arithmetic, no locks needed.
//
//  A channel (std::sync::mpsc or crossbeam) collects batches from all
//  generator threads and feeds them to the GPU uploader thread.
//
//  ── Character sets ───────────────────────────────────────────────────────────
//
//  LOWER   = b"abcdefghijklmnopqrstuvwxyz"              (26)
//  UPPER   = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ"              (26)
//  DIGITS  = b"0123456789"                              (10)
//  SYMBOLS = b"!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"    (32)
//
//  Flags are OR-able: LOWER | DIGITS is the default charset.
//  The combined alphabet is sorted so slice boundaries are deterministic.
//
//  ── Mask mode (--mask) ───────────────────────────────────────────────────────
//
//  Hashcat-compatible position masks:
//    ?l  any lowercase letter
//    ?u  any uppercase letter
//    ?d  any digit
//    ?s  any symbol
//    ?a  any of the above
//    <literal>  fixed character in output
//
//  Example: "Pass?d?d?d" → Pass000 … Pass999  (1 000 candidates)
//
//  ── Dictionary mangling (--mangle) ───────────────────────────────────────────
//
//  For every word W loaded from the original wordlist, emit all transforms:
//    W                   (original)
//    capitalize(W)       (first char upper)
//    W.to_uppercase()
//    W + common_suffixes  where suffixes = ["1","12","123","!","@","2024",…]
//    leet(W)             (a→4, e→3, i→1, o→0, s→5, t→7)
//    leet(capitalize(W))
//
//  ── ETA display ──────────────────────────────────────────────────────────────
//
//  Total candidates for length L over an alphabet of size A: A^L.
//  Total for a length range [min..max]: sum of A^L for L in min..=max.
//  After 5 seconds of running, measure actual GPU throughput (candidates/s)
//  and compute ETA = remaining / throughput.  Display via the same
//  Progress bar used in Phase 1.
//
//  ── Integration point ────────────────────────────────────────────────────────
//
//  In main.rs, Phase 2 currently calls `wordgen::print_stub(num_threads)`.
//  Replace that call with the real generator once implemented:
//
//      let found = wordgen::run(
//          WordgenConfig { charsets: LOWER|DIGITS, min_len: 6, max_len: 10,
//                          mask: None, mangle: false, num_threads },
//          &key, gpu_ctx.as_ref(), batch_size, verbose,
//      );
//
// =============================================================================

pub const LOWER:   u32 = 1 << 0;
pub const UPPER:   u32 = 1 << 1;
pub const DIGITS:  u32 = 1 << 2;
pub const SYMBOLS: u32 = 1 << 3;

#[allow(dead_code)]
pub struct WordgenConfig {
    pub charsets:    u32,
    pub min_len:     usize,
    pub max_len:     usize,
    pub mask:        Option<String>,
    pub mangle:      bool,
    pub num_threads: usize,
}

/// Prints a summary of what Phase 2 would do and exits cleanly.
/// Replace this function body with the real generator when ready.
pub fn print_stub(num_threads: usize) {
    eprintln!();
    eprintln!("  ┌─ Phase 2: Word Generator  (not yet implemented)");
    eprintln!("  └──────────────────────────────────────────────────────────");
    eprintln!();
    eprintln!("  When implemented this phase will:");
    eprintln!("  • Walk the full charset × length search space on {} CPU thread(s)", num_threads);
    eprintln!("  • Stream candidates directly to the GPU — no RAM spike");
    eprintln!("  • Support mask mode  e.g.  --mask \"Pass?d?d?d\"");
    eprintln!("  • Support dictionary mangling  --mangle");
    eprintln!("  • Show live ETA once throughput is measured");
    eprintln!();
    eprintln!("  See src/wordgen.rs for the full implementation contract.");
    eprintln!();
}