// =============================================================================
//  crack.wgsl  —  SSH key passphrase check-byte validator
//
//  Each GPU thread handles ONE password candidate.
//  Input : derived key material (64 raw bytes per candidate, packed as LE u32)
//  Output: 1 if check_bytes[0..4] == check_bytes[4..8] after decryption, 0 otherwise
//
//  Supports:
//    cipher_type == 0  → ChaCha20-poly1305@openssh.com  (legacy 64-bit counter)
//    cipher_type == 1  → AES-256-CTR
// =============================================================================

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

struct Params {
    num_candidates : u32,
    cipher_type    : u32,   // 0 = chacha20-poly1305, 1 = aes256-ctr
    // First 16 bytes of the encrypted blob (packed as LE u32, GPU byte order)
    enc_w0 : u32,
    enc_w1 : u32,
    enc_w2 : u32,
    enc_w3 : u32,
}

// key_material: 16 u32 per candidate (64 bytes).
//   [0..7]   = 32-byte key     (LE u32 packing)
//   [8..11]  = 16-byte AES IV  (LE u32 packing, used only for aes256-ctr)
//   [12..15] = reserved / extra bytes from bcrypt_pbkdf
@group(0) @binding(0) var<storage, read>       key_material : array<u32>;
@group(0) @binding(1) var<uniform>             params       : Params;
@group(0) @binding(2) var<storage, read_write> results      : array<u32>;
@group(0) @binding(3) var<storage, read>       sbox         : array<u32>;
@group(0) @binding(4) var<storage, read>       rcon         : array<u32>;

// ---------------------------------------------------------------------------
// AES S-box and RCON: provided as storage buffer bindings (gpu.rs initialization)

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

// Reverse byte order in a u32 (LE ↔ BE)
fn bswap(x: u32) -> u32 {
    return ((x & 0x000000ffu) << 24u) | ((x & 0x0000ff00u) << 8u)
         | ((x >> 8u)  & 0x0000ff00u) | ((x >> 24u) & 0x000000ffu);
}

// ---------------------------------------------------------------------------
// AES-256-CTR helpers
// ---------------------------------------------------------------------------

// Apply S-box substitution to each byte of a u32 (big-endian column)
fn sub_word(w: u32) -> u32 {
    return (sbox[(w >> 24u) & 0xffu] << 24u)
         | (sbox[(w >> 16u) & 0xffu] << 16u)
         | (sbox[(w >>  8u) & 0xffu] <<  8u)
         |  sbox[ w         & 0xffu];
}

// Rotate u32 left by 8 bits (RotWord in AES spec)
fn rot_word(w: u32) -> u32 {
    return (w << 8u) | (w >> 24u);
}

// GF(2^8) multiply by 2 (xtime)
fn xtime(x: u32) -> u32 {
    return ((x << 1u) ^ (select(0u, 0x1bu, (x & 0x80u) != 0u))) & 0xffu;
}

// AES MixColumns on one big-endian column word
// col = (b0<<24)|(b1<<16)|(b2<<8)|b3  where b0=row0, b3=row3
fn mix_col(col: u32) -> u32 {
    let b0 = (col >> 24u) & 0xffu;
    let b1 = (col >> 16u) & 0xffu;
    let b2 = (col >>  8u) & 0xffu;
    let b3 =  col         & 0xffu;
    let xb0 = xtime(b0); let xb1 = xtime(b1);
    let xb2 = xtime(b2); let xb3 = xtime(b3);
    let r0 = xb0 ^ (xb1 ^ b1) ^ b2 ^ b3;
    let r1 = b0 ^ xb1 ^ (xb2 ^ b2) ^ b3;
    let r2 = b0 ^ b1 ^ xb2 ^ (xb3 ^ b3);
    let r3 = (xb0 ^ b0) ^ b1 ^ b2 ^ xb3;
    return (r0 << 24u) | (r1 << 16u) | (r2 << 8u) | r3;
}

// AES-256 encrypt one 16-byte block.
//   km_base : index (in u32) of this candidate's key material in key_material[].
//             key  = key_material[km_base .. km_base+8]   (32 bytes, LE packed)
//             iv   = key_material[km_base+8 .. km_base+12] (16 bytes, LE packed)
//   Returns 4 x u32 output in big-endian column format.
fn aes256_encrypt_block(km_base: u32) -> array<u32, 4> {
    // ---- Key expansion ----
    var rk: array<u32, 60>;

    // Load key words in BE format (byteswap from LE buffer)
    for (var i = 0u; i < 8u; i++) {
        rk[i] = bswap(key_material[km_base + i]);
    }

    // Expand: 8 initial + 52 derived = 60 round key words
    for (var i = 8u; i < 60u; i++) {
        var tmp = rk[i - 1u];
        if i % 8u == 0u {
            tmp = sub_word(rot_word(tmp)) ^ rcon[i / 8u - 1u];
        } else if i % 8u == 4u {
            tmp = sub_word(tmp);
        }
        rk[i] = rk[i - 8u] ^ tmp;
    }

    // ---- Initialize state from IV (bytes 32-47 of key material) ----
    var s: array<u32, 4>;
    for (var i = 0u; i < 4u; i++) {
        s[i] = bswap(key_material[km_base + 8u + i]);
    }

    // ---- Round 0: AddRoundKey ----
    for (var i = 0u; i < 4u; i++) { s[i] ^= rk[i]; }

    // ---- Rounds 1-13 (SubBytes + ShiftRows + MixColumns + AddRoundKey) ----
    for (var r = 1u; r <= 13u; r++) {
        // SubBytes
        for (var i = 0u; i < 4u; i++) { s[i] = sub_word(s[i]); }

        // ShiftRows  (new col c gets row r from old col (c+r)%4)
        let t0 = (s[0] & 0xff000000u) | (s[1] & 0x00ff0000u) | (s[2] & 0x0000ff00u) | (s[3] & 0x000000ffu);
        let t1 = (s[1] & 0xff000000u) | (s[2] & 0x00ff0000u) | (s[3] & 0x0000ff00u) | (s[0] & 0x000000ffu);
        let t2 = (s[2] & 0xff000000u) | (s[3] & 0x00ff0000u) | (s[0] & 0x0000ff00u) | (s[1] & 0x000000ffu);
        let t3 = (s[3] & 0xff000000u) | (s[0] & 0x00ff0000u) | (s[1] & 0x0000ff00u) | (s[2] & 0x000000ffu);
        s[0] = t0; s[1] = t1; s[2] = t2; s[3] = t3;

        // MixColumns
        for (var i = 0u; i < 4u; i++) { s[i] = mix_col(s[i]); }

        // AddRoundKey
        let b = r * 4u;
        for (var i = 0u; i < 4u; i++) { s[i] ^= rk[b + i]; }
    }

    // ---- Round 14: SubBytes + ShiftRows + AddRoundKey (no MixColumns) ----
    for (var i = 0u; i < 4u; i++) { s[i] = sub_word(s[i]); }

    let t0 = (s[0] & 0xff000000u) | (s[1] & 0x00ff0000u) | (s[2] & 0x0000ff00u) | (s[3] & 0x000000ffu);
    let t1 = (s[1] & 0xff000000u) | (s[2] & 0x00ff0000u) | (s[3] & 0x0000ff00u) | (s[0] & 0x000000ffu);
    let t2 = (s[2] & 0xff000000u) | (s[3] & 0x00ff0000u) | (s[0] & 0x0000ff00u) | (s[1] & 0x000000ffu);
    let t3 = (s[3] & 0xff000000u) | (s[0] & 0x00ff0000u) | (s[1] & 0x0000ff00u) | (s[2] & 0x000000ffu);
    s[0] = t0 ^ rk[56u];
    s[1] = t1 ^ rk[57u];
    s[2] = t2 ^ rk[58u];
    s[3] = t3 ^ rk[59u];

    return s;  // big-endian columns
}

// ---------------------------------------------------------------------------
// ChaCha20 (original / legacy: 64-bit counter + 64-bit nonce)
// ---------------------------------------------------------------------------
//
// OpenSSH chacha20-poly1305@openssh.com uses the original 64-bit counter
// variant.  For private-key file decryption:
//   key     = key_material[0..32]
//   counter = 1  (block 0 reserved for Poly1305 key derivation)
//   nonce   = 0  (SSH sequence number is 0 for key file)
//
// We produce block 1 of the keystream and XOR with the first 8 bytes
// of the encrypted blob to recover check1 and check2.
// ---------------------------------------------------------------------------

fn rotl(v: u32, n: u32) -> u32 { return (v << n) | (v >> (32u - n)); }

fn qr(s: ptr<function, array<u32, 16>>, a: u32, b: u32, c: u32, d: u32) {
    (*s)[a] = (*s)[a] + (*s)[b]; (*s)[d] ^= (*s)[a]; (*s)[d] = rotl((*s)[d], 16u);
    (*s)[c] = (*s)[c] + (*s)[d]; (*s)[b] ^= (*s)[c]; (*s)[b] = rotl((*s)[b], 12u);
    (*s)[a] = (*s)[a] + (*s)[b]; (*s)[d] ^= (*s)[a]; (*s)[d] = rotl((*s)[d], 8u);
    (*s)[c] = (*s)[c] + (*s)[d]; (*s)[b] ^= (*s)[c]; (*s)[b] = rotl((*s)[b], 7u);
}

// Returns block 1 of the ChaCha20 keystream (16 x u32, little-endian).
// key_material[km_base .. km_base+8] = 32-byte key (already LE, no swap needed)
fn chacha20_block(km_base: u32) -> array<u32, 16> {
    var s: array<u32, 16>;
    // Constants: "expa nd 3 2-by te k"
    s[0]  = 0x61707865u; s[1]  = 0x3320646eu;
    s[2]  = 0x79622d32u; s[3]  = 0x6b206574u;
    // Key (LE u32, already correct for ChaCha20)
    for (var i = 0u; i < 8u; i++) { s[4u + i] = key_material[km_base + i]; }
    // Counter = 1, Nonce = 0  (legacy layout: [ctr_lo, ctr_hi, nonce_lo, nonce_hi])
    s[12] = 1u; s[13] = 0u; s[14] = 0u; s[15] = 0u;

    var init = s;

    // 20 rounds (10 double rounds)
    for (var i = 0u; i < 10u; i++) {
        qr(&s, 0u, 4u,  8u, 12u);
        qr(&s, 1u, 5u,  9u, 13u);
        qr(&s, 2u, 6u, 10u, 14u);
        qr(&s, 3u, 7u, 11u, 15u);
        qr(&s, 0u, 5u, 10u, 15u);
        qr(&s, 1u, 6u, 11u, 12u);
        qr(&s, 2u, 7u,  8u, 13u);
        qr(&s, 3u, 4u,  9u, 14u);
    }

    for (var i = 0u; i < 16u; i++) { s[i] = s[i] + init[i]; }
    return s;
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.num_candidates { return; }

    let km_base = idx * 16u;  // 64 bytes / 4 per u32

    var plain_w0: u32;
    var plain_w1: u32;

    if params.cipher_type == 0u {
        // ── ChaCha20-poly1305 ──────────────────────────────────────────────
        // Keystream block 1 is in LE byte order (natural for ChaCha20).
        // Encrypted blob is also stored as LE u32 in the params.
        let ks = chacha20_block(km_base);
        plain_w0 = params.enc_w0 ^ ks[0];
        plain_w1 = params.enc_w1 ^ ks[1];

    } else {
        // ── AES-256-CTR ────────────────────────────────────────────────────
        // aes256_encrypt_block() returns BE column words.
        // params.enc_w* are LE-packed.
        // We byteswap the keystream to LE before XOR.
        let ks_be = aes256_encrypt_block(km_base);
        plain_w0 = params.enc_w0 ^ bswap(ks_be[0]);
        plain_w1 = params.enc_w1 ^ bswap(ks_be[1]);
    }

    // check1 == check2 means we found the right passphrase.
    // Both are 4-byte raw values written by OpenSSH; equal only for correct key.
    results[idx] = select(0u, 1u, plain_w0 == plain_w1);
}
