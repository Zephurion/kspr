// =============================================================================
//  cpu_verify.rs — authoritative CPU check-byte verification
//
//  Used in two situations:
//    1. No GPU available → full Rayon-parallel wordlist scan on CPU
//    2. GPU returns a hit → confirm with RustCrypto before declaring success
//       (the GPU shader is a faithful implementation but the CPU path uses
//        the well-tested RustCrypto crates as the ground truth)
//
//  The OpenSSH check-byte test
//  ──────────────────────────
//  Before encryption, OpenSSH writes two identical random u32 values at
//  bytes [0..4] and [4..8] of the plaintext private-key blob.
//  After decryption, if check_bytes[0..4] == check_bytes[4..8] the
//  passphrase is correct.  The probability of a false positive is 1/2^32.
// =============================================================================

use anyhow::Result;
use cipher::{KeyIvInit, StreamCipher, StreamCipherSeek};
use zeroize::Zeroizing;

use crate::kdf;
use crate::keyparser::{Cipher, Kdf};
use crate::progress::Progress;

// Concrete cipher types using RustCrypto
// ctr::Ctr128BE: AES-256 in CTR mode, big-endian counter (as OpenSSH uses)
type Aes256Ctr128BE = ctr::Ctr128BE<aes::Aes256>;
// chacha20::ChaCha20: original RFC 7539 / ChaCha20 with 96-bit nonce
type ChaCha20 = chacha20::ChaCha20;

// ─────────────────────────────────────────────────────────────────────────────
// Single passphrase check
// ─────────────────────────────────────────────────────────────────────────────

/// Derive key material and attempt to decrypt the first 8 bytes of `enc_blob`.
/// Returns `true` if the check-bytes test passes (passphrase is correct).
pub fn verify(
    passphrase: &[u8],
    kdf:        &Kdf,
    cipher:     &Cipher,
    enc_blob:   &[u8],
) -> Result<bool> {
    if enc_blob.len() < 8 {
        return Ok(false);
    }

    // ── Unencrypted key (cipher: none) ────────────────────────────────────────
    if matches!(cipher, Cipher::None) {
        return Ok(enc_blob[0..4] == enc_blob[4..8]);
    }

    // ── Derive key material ───────────────────────────────────────────────────
    let (salt, rounds) = match kdf {
        Kdf::Bcrypt { salt, rounds } => (salt.as_slice(), *rounds),
        Kdf::None => {
            // Should not happen when cipher != None, but handle gracefully
            return Ok(enc_blob[0..4] == enc_blob[4..8]);
        }
    };

    let output_len = cipher.key_material_len();
    let km: Zeroizing<[u8; kdf::KM_BYTES]> =
        kdf::derive_one(passphrase, salt, rounds, output_len)?;

    // Copy the first 8 bytes of the encrypted blob into a local buffer
    let mut block = [0u8; 8];
    block.copy_from_slice(&enc_blob[0..8]);

    // ── Decrypt with the appropriate cipher ───────────────────────────────────
    match cipher {
        Cipher::Chacha20Poly1305 => {
            // OpenSSH chacha20-poly1305@openssh.com:
            //   key   = km[0..32]  (32 bytes)
            //   nonce = [0u8; 12]  (SSH sequence number 0 for key file)
            //   The *data* keystream starts at ChaCha20 block counter 1
            //   (block 0 is used to derive the Poly1305 authentication key).
            //   seek(64) advances the stream by exactly one 64-byte block.
            let key   = chacha20::Key::from_slice(&km[0..32]);
            let nonce = chacha20::Nonce::from_slice(&[0u8; 12]);
            let mut c = ChaCha20::new(key, nonce);
            c.seek(64u64); // skip block 0
            c.apply_keystream(&mut block);
        }

        Cipher::Aes256Ctr => {
            // key = km[0..32], iv = km[32..48]
            // ctr::Ctr128BE counts upward from the IV in big-endian order,
            // which matches the way OpenSSH emits the IV from bcrypt_pbkdf.
            let key = aes::cipher::generic_array::GenericArray::from_slice(&km[0..32]);
            let iv  = aes::cipher::generic_array::GenericArray::from_slice(&km[32..48]);
            let mut c = Aes256Ctr128BE::new(key, iv);
            c.apply_keystream(&mut block);
        }

        // None already handled above; Unsupported never reaches here
        _ => {}
    }

    Ok(block[0..4] == block[4..8])
}

// ─────────────────────────────────────────────────────────────────────────────
// Rayon parallel scan (CPU fallback when no GPU is available)
// ─────────────────────────────────────────────────────────────────────────────

/// Try every password in parallel.  Stops as soon as a match is found.
/// Returns the matching passphrase as a String, or None.
pub fn cpu_scan(
    passwords: &[Vec<u8>],
    kdf:       &Kdf,
    cipher:    &Cipher,
    enc_blob:  &[u8],
    progress:  &Progress,
) -> Option<String> {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    let stop = Arc::new(AtomicBool::new(false));

    // par_iter + find_any: Rayon will return the *first* match found
    // across all threads (order not guaranteed, but we only need one).
    let result = passwords
        .par_iter()
        .enumerate()
        .find_any(|(i, pw)| {
            if stop.load(Ordering::Relaxed) {
                return false;
            }
            let matched = verify(pw, kdf, cipher, enc_blob).unwrap_or(false);
            progress.log_verbose(*i, &String::from_utf8_lossy(pw), matched);
            progress.advance(1);
            if matched {
                stop.store(true, Ordering::Relaxed);
            }
            matched
        })
        .map(|(_, pw)| String::from_utf8_lossy(pw).into_owned());

    result
}