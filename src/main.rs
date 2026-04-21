// =============================================================================
//  main.rs — SSH key passphrase finder  (Rust + wgpu)
//
//  Phase 1: Wordlist scan
//    CPU (Rayon)  → bcrypt_pbkdf for every candidate in the current batch
//    GPU (wgpu)   → crack.wgsl checks check-bytes for all N candidates at once
//    CPU confirm  → any GPU hit is re-verified with RustCrypto before we win
//
//  Phase 2: Word generator stub  (see src/wordgen.rs)
// =============================================================================

mod cpu_verify;
mod gpu;
mod kdf;
mod keyparser;
mod progress;
mod wordgen;

use anyhow::{bail, Context, Result};
use clap::Parser;
use progress::fmt_duration;

// ─────────────────────────────────────────────────────────────────────────────
// CLI
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(
    name    = "kspr",
    version = "2.0.0",
    about   = "GPU-accelerated SSH key passphrase finder",
    long_about = "\
Recovers a forgotten SSH key passphrase by decrypting the check-byte block
of the OpenSSH private key file — no network access, no ssh-keygen subprocess.

Supported key types : RSA · ED25519 · ECDSA (nistp256 / nistp384 / nistp521)
Supported ciphers   : chacha20-poly1305@openssh.com · aes256-ctr
GPU backend         : wgpu  (Vulkan / Metal / DX12 / WebGPU)
CPU fallback        : Rayon parallel threads (automatic if no GPU found)"
)]
struct Args {
    /// Path to the SSH private key  (id_rsa / id_ed25519 / id_ecdsa)
    #[arg(short = 'k', long = "key")]
    key: String,

    /// Path to the wordlist file  (one passphrase per line; # = comment)
    #[arg(short = 'w', long = "wordlist")]
    wordlist: String,

    /// Number of CPU threads used for bcrypt_pbkdf key derivation
    #[arg(short = 't', long = "threads", default_value_t = default_threads())]
    threads: usize,

    /// Number of candidates per GPU dispatch (tune for your VRAM)
    #[arg(long = "batch-size", default_value_t = 4096)]
    batch_size: usize,

    /// Print every candidate and its ✓/✗ result
    #[arg(short = 'v', long = "verbose")]
    verbose: bool,

    /// Skip GPU init and run on CPU only
    #[arg(long = "cpu-only")]
    cpu_only: bool,
}

fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

// ─────────────────────────────────────────────────────────────────────────────
// Wordlist loader — dynamic capacity, no hard cap
// ─────────────────────────────────────────────────────────────────────────────

fn load_wordlist(path: &str) -> Result<Vec<Vec<u8>>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("cannot open wordlist: {}", path))?;

    // Start small; Vec will grow as needed via amortised doubling
    let mut list: Vec<Vec<u8>> = Vec::with_capacity(4096);

    for line in text.lines() {
        // Strip trailing CR/LF (already handled by .lines(), but defensive)
        let s = line.trim_end_matches(['\r', '\n']);
        if s.is_empty() || s.starts_with('#') {
            continue;
        }
        list.push(s.as_bytes().to_vec());
    }
    Ok(list)
}

// ─────────────────────────────────────────────────────────────────────────────
// Banner
// ─────────────────────────────────────────────────────────────────────────────

fn banner() {
    println!();
    println!("  ╔════════════════════════════════════════════════════════╗");
    println!("  ║    SSH Key Passphrase Finder  v2.0  (Rust + wgpu)      ║");
    println!("  ║   RSA · ED25519 · ECDSA  ·  legal personal use only    ║");
    println!("  ╚════════════════════════════════════════════════════════╝");
    println!();
}

// ─────────────────────────────────────────────────────────────────────────────
// GPU-accelerated batch scan
// ─────────────────────────────────────────────────────────────────────────────

async fn gpu_scan(
    ctx:        &gpu::GpuContext,
    passwords:  &[Vec<u8>],
    key:        &keyparser::ParsedKey,
    batch_size: usize,
    threads:    usize,
    verbose:    bool,
) -> Option<String> {
    use keyparser::Kdf;

    // Unencrypted key — nothing to derive
    if matches!(key.cipher, keyparser::Cipher::None) {
        if key.encrypted_blob.len() >= 8
            && key.encrypted_blob[0..4] == key.encrypted_blob[4..8]
        {
            return Some("(empty — key has no passphrase)".to_string());
        }
        return None;
    }

    let (salt, rounds) = match &key.kdf {
        Kdf::Bcrypt { salt, rounds } => (salt.clone(), *rounds),
        Kdf::None => return None,
    };

    let output_len = key.cipher.key_material_len();
    let total      = passwords.len() as u64;
    let prog       = progress::Progress::new(total, verbose);

    // Set Rayon thread pool size for the KDF derivation phase
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .ok(); // ignore error if already initialised

    let mut global_i = 0usize;

    for chunk in passwords.chunks(batch_size) {
        // ── CPU: derive key material for this batch ───────────────────────────
        let km_u32 = kdf::derive_batch(chunk, &salt, rounds, output_len);

        // ── GPU: dispatch compute shader ──────────────────────────────────────
        let gpu_hits = match ctx
            .run_batch(&km_u32, &key.cipher, &key.encrypted_blob)
            .await
        {
            Ok(v) => v,
            Err(e) => {
                prog.log(&format!("  [warn] GPU batch error: {} — skipping", e));
                prog.advance(chunk.len() as u64);
                global_i += chunk.len();
                continue;
            }
        };

        // ── CPU confirm: verify any GPU hits precisely ────────────────────────
        for (local_i, &hit) in gpu_hits.iter().enumerate() {
            let pw     = &chunk[local_i];
            let pw_str = String::from_utf8_lossy(pw).into_owned();

            prog.log_verbose(global_i + local_i, &pw_str, hit);

            if hit {
                let confirmed = cpu_verify::verify(pw, &key.kdf, &key.cipher, &key.encrypted_blob)
                    .unwrap_or(false);
                if confirmed {
                    prog.advance((local_i + 1) as u64);
                    prog.finish();
                    return Some(pw_str);
                }
            }
        }

        prog.advance(chunk.len() as u64);
        global_i += chunk.len();
    }

    prog.finish();
    None
}

// ─────────────────────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args = Args::parse();

    banner();

    // ── Parse the private key ─────────────────────────────────────────────────
    println!("  Key      : {}", args.key);
    let key = keyparser::parse_key_file(&args.key)?;

    println!("  Type     : {}", key.key_type);
    println!("  Cipher   : {}", key.cipher);
    println!("  KDF      : {}", key.kdf);

    if !key.key_type.is_supported() {
        bail!(
            "\n  ✗ Key type '{}' is not supported.\n\
               \n  Supported: RSA, ED25519, ECDSA\
               \n  Rejected : DSA (deprecated — weak cryptography)\n",
            key.key_type
        );
    }
    if !key.cipher.is_supported() {
        bail!(
            "\n  ✗ Cipher '{}' is not supported.\n\
               \n  Supported: chacha20-poly1305@openssh.com, aes256-ctr\n",
            key.cipher
        );
    }

    // ── Load wordlist ─────────────────────────────────────────────────────────
    println!("  Wordlist : {}", args.wordlist);
    let passwords = load_wordlist(&args.wordlist)?;
    println!("  Found    : {} candidates", passwords.len());

    if passwords.is_empty() {
        bail!("wordlist is empty — add at least one passphrase to try");
    }

    // ── Initialise GPU ────────────────────────────────────────────────────────
    let gpu_ctx: Option<gpu::GpuContext> = if args.cpu_only {
        println!("  Backend  : CPU only (--cpu-only flag)");
        None
    } else {
        print!("  Backend  : ");
        match pollster::block_on(gpu::GpuContext::new()) {
            Ok(ctx) => {
                println!("GPU  [{} / {}]", ctx.adapter_name, ctx.adapter_type);
                Some(ctx)
            }
            Err(e) => {
                println!("CPU fallback (GPU unavailable: {})", e);
                None
            }
        }
    };

    println!("  Threads  : {} CPU (KDF derivation)", args.threads);
    if gpu_ctx.is_some() {
        println!("  Batch    : {} candidates/dispatch", args.batch_size);
    }
    if args.verbose {
        println!("  Verbose  : on — printing every candidate");
    }
    println!();

    // ── Phase 1: wordlist scan ────────────────────────────────────────────────
    println!("  ┌─ Phase 1: Wordlist scan");
    println!("  └──────────────────────────────────────────────────────────");
    println!();

    let t0 = std::time::Instant::now();

    let found: Option<String> = if let Some(ref ctx) = gpu_ctx {
        pollster::block_on(gpu_scan(
            ctx,
            &passwords,
            &key,
            args.batch_size,
            args.threads,
            args.verbose,
        ))
    } else {
        // Full CPU fallback
        let prog = progress::Progress::new(passwords.len() as u64, args.verbose);
        let result = cpu_verify::cpu_scan(
            &passwords,
            &key.kdf,
            &key.cipher,
            &key.encrypted_blob,
            &prog,
        );
        prog.finish();
        result
    };

    // ── Also try the empty passphrase explicitly ───────────────────────────────
    // (not in the wordlist — handle separately so it's always tested)
    let found = found.or_else(|| {
        print!("  Trying empty passphrase … ");
        let ok = cpu_verify::verify(b"", &key.kdf, &key.cipher, &key.encrypted_blob)
            .unwrap_or(false);
        if ok {
            println!("\x1b[32m✓\x1b[0m");
            Some("(empty — no passphrase set)".to_string())
        } else {
            println!("\x1b[31m✗\x1b[0m");
            None
        }
    });

    let elapsed = t0.elapsed();

    // ── Print result ──────────────────────────────────────────────────────────
    println!();
    println!("  ┌─ Result");
    println!("  └──────────────────────────────────────────────────────────");
    println!();

    if let Some(ref passphrase) = found {
        println!(
            "  \x1b[32m✓ Passphrase found!\x1b[0m  (after {})",
            fmt_duration(elapsed.as_secs())
        );
        println!();
        println!("  ┌────────────────────────────────────────────────────┐");
        println!("  │  {}", passphrase);
        println!("  └────────────────────────────────────────────────────┘");
        println!();
        println!("  Connect to your eBPF VM:");
        println!("    ssh -i {} <user>@<vm-ip>", args.key);
        println!();
        return Ok(());
    }

    println!(
        "  \x1b[31m✗ Not found\x1b[0m across {} candidates  ({})",
        passwords.len(),
        fmt_duration(elapsed.as_secs())
    );
    println!();

    // ── Phase 2 stub ──────────────────────────────────────────────────────────
    wordgen::print_stub(args.threads);

    std::process::exit(1);
}