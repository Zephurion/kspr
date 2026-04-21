// =============================================================================
//  kdf.rs — bcrypt_pbkdf key derivation
//
//  OpenSSH uses bcrypt_pbkdf (NOT standard bcrypt) to stretch a passphrase
//  into cipher key material.  We run this on the CPU via Rayon so all cores
//  are busy deriving while the GPU runs the previous batch.
//
//  Memory layout per candidate (64 bytes, always, regardless of cipher):
//    bytes  0-31  → cipher key   (32 bytes)
//    bytes 32-47  → IV / nonce   (16 bytes, aes256-ctr only)
//    bytes 48-63  → key2         (chacha20-poly1305 only, second key)
//
//  On the GPU side this is uploaded as 16 × u32 (little-endian).
// =============================================================================

use anyhow::{bail, Result};
use bcrypt_pbkdf::bcrypt_pbkdf;
use rayon::prelude::*;
use zeroize::Zeroizing;

/// 64 bytes per candidate, packed as 16 × u32.
pub const KM_BYTES: usize = 64;
pub const KM_U32S:  usize = KM_BYTES / 4; // 16

// ─────────────────────────────────────────────────────────────────────────────
// Single derivation
// ─────────────────────────────────────────────────────────────────────────────

/// Derive key material for one passphrase.
/// `output_len` is cipher-specific: 48 (aes256-ctr) or 64 (chacha20-poly1305).
/// Returns a Zeroizing 64-byte buffer (zeroed past output_len).
pub fn derive_one(
    passphrase: &[u8],
    salt:       &[u8],
    rounds:     u32,
    output_len: usize,
) -> Result<Zeroizing<[u8; KM_BYTES]>> {
    if output_len > KM_BYTES {
        bail!("output_len {} exceeds KM_BYTES ({})", output_len, KM_BYTES);
    }
    let mut buf = Zeroizing::new([0u8; KM_BYTES]);
    bcrypt_pbkdf(passphrase, salt, rounds, &mut buf[..output_len])
        .map_err(|_| anyhow::anyhow!("bcrypt_pbkdf failed for given salt/rounds"))?;
    Ok(buf)
}

// ─────────────────────────────────────────────────────────────────────────────
// Batch derivation (Rayon parallel)
// ─────────────────────────────────────────────────────────────────────────────

/// Derive key material for a slice of passphrases in parallel.
///
/// Returns a flat `Vec<u32>` of length `passwords.len() × KM_U32S`.
/// Values are packed little-endian so the GPU shader can consume them
/// directly as `array<u32>` without endian swaps in the hot path.
///
/// Failed derivations (should never happen for valid salt/rounds) produce
/// an all-zero block so the GPU simply skips them (0 != valid check_bytes).
pub fn derive_batch(
    passwords:  &[Vec<u8>],
    salt:       &[u8],
    rounds:     u32,
    output_len: usize,
) -> Vec<u32> {
    passwords
        .par_iter()
        .flat_map_iter(|pw| {
            let km = derive_one(pw, salt, rounds, output_len)
                .unwrap_or_else(|_| Zeroizing::new([0u8; KM_BYTES]));

            // Pack each group of 4 bytes into a LE u32
            km.chunks_exact(4)
              .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
              .collect::<Vec<u32>>()
        })
        .collect()
}