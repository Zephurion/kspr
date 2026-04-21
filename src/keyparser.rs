// =============================================================================
//  keyparser.rs — OpenSSH private key parser
//
//  Parses "BEGIN OPENSSH PRIVATE KEY" PEM files and extracts everything
//  the passphrase finder needs:
//    • key type   (RSA / ED25519 / ECDSA; DSA rejected immediately)
//    • cipher     (chacha20-poly1305 / aes256-ctr / none)
//    • KDF params (bcrypt salt + rounds)
//    • raw encrypted blob  (the bytes we will attempt to decrypt)
//
//  Binary layout after base64 decode:
//    "openssh-key-v1\0"     15 bytes — magic
//    string  ciphername
//    string  kdfname
//    string  kdfoptions     for bcrypt: (string salt)(uint32 rounds)
//    uint32  number_of_keys
//    string  public_key     first inner string = key-type ("ssh-ed25519" …)
//    string  encrypted_blob
//
//  "string" = big-endian uint32 length followed by that many bytes.
// =============================================================================

use anyhow::{bail, Context, Result};
use base64::{engine::general_purpose::STANDARD as B64, Engine};
use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// Public enums
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum KeyType {
    Rsa,
    Ed25519,
    Ecdsa(String),   // curve name e.g. "nistp256"
    Dsa,
    Unknown(String),
}

impl KeyType {
    pub fn is_supported(&self) -> bool {
        matches!(self, Self::Rsa | Self::Ed25519 | Self::Ecdsa(_))
    }
}

impl fmt::Display for KeyType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rsa        => write!(f, "RSA"),
            Self::Ed25519    => write!(f, "ED25519"),
            Self::Ecdsa(c)   => write!(f, "ECDSA ({})", c),
            Self::Dsa        => write!(f, "DSA"),
            Self::Unknown(s) => write!(f, "Unknown ({})", s),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum Cipher {
    None,
    Aes256Ctr,
    Chacha20Poly1305,
    Unsupported(String),
}

impl Cipher {
    /// Bytes of key material bcrypt_pbkdf must produce for this cipher.
    ///   aes256-ctr        → 32 (key) + 16 (iv)  = 48
    ///   chacha20-poly1305 → 32 (key) + 32 (key2) = 64
    ///   none              → 0
    pub fn key_material_len(&self) -> usize {
        match self {
            Self::None             => 0,
            Self::Aes256Ctr        => 48,
            Self::Chacha20Poly1305 => 64,
            Self::Unsupported(_)   => 0,
        }
    }

    pub fn is_supported(&self) -> bool {
        !matches!(self, Self::Unsupported(_))
    }
}

impl fmt::Display for Cipher {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None             => write!(f, "none (unencrypted)"),
            Self::Aes256Ctr        => write!(f, "aes256-ctr"),
            Self::Chacha20Poly1305 => write!(f, "chacha20-poly1305@openssh.com"),
            Self::Unsupported(s)   => write!(f, "{} (unsupported)", s),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum Kdf {
    None,
    Bcrypt { salt: Vec<u8>, rounds: u32 },
}

impl fmt::Display for Kdf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None                   => write!(f, "none"),
            Self::Bcrypt { rounds, .. }  => write!(f, "bcrypt ({} rounds)", rounds),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Everything the finder needs to crack a key.
#[derive(Debug, Clone)]
pub struct ParsedKey {
    pub key_type:       KeyType,
    pub cipher:         Cipher,
    pub kdf:            Kdf,
    /// Raw encrypted private-key blob; first 8 bytes are the check-bytes.
    pub encrypted_blob: Vec<u8>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Binary cursor
// ─────────────────────────────────────────────────────────────────────────────

struct Cursor<'a> {
    data: &'a [u8],
    pos:  usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos + n;
        if end > self.data.len() {
            bail!(
                "truncated key blob: need {} bytes at offset {}, only {} remain",
                n, self.pos, self.data.len() - self.pos
            );
        }
        let slice = &self.data[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn read_u32_be(&mut self) -> Result<u32> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// Read a length-prefixed binary string.
    fn read_string(&mut self) -> Result<&'a [u8]> {
        let len = self.read_u32_be()? as usize;
        self.read_bytes(len)
    }

    /// Read a length-prefixed UTF-8 string.
    fn read_str(&mut self) -> Result<&'a str> {
        let bytes = self.read_string()?;
        std::str::from_utf8(bytes).context("non-UTF-8 string in key blob")
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn parse_key_type(s: &str) -> KeyType {
    match s {
        "ssh-rsa"    => KeyType::Rsa,
        "ssh-ed25519"=> KeyType::Ed25519,
        "ssh-dss"    => KeyType::Dsa,
        other => {
            if let Some(curve) = other.strip_prefix("ecdsa-sha2-") {
                KeyType::Ecdsa(curve.to_string())
            } else {
                KeyType::Unknown(other.to_string())
            }
        }
    }
}

/// Fast path: read first word of the companion .pub file.
fn detect_from_pub(private_path: &str) -> Option<KeyType> {
    let pub_path = format!("{}.pub", private_path);
    let s = std::fs::read_to_string(pub_path).ok()?;
    let first = s.split_ascii_whitespace().next()?;
    Some(parse_key_type(first))
}

// ─────────────────────────────────────────────────────────────────────────────
// Core parser
// ─────────────────────────────────────────────────────────────────────────────

const MAGIC: &[u8] = b"openssh-key-v1\0"; // 15 bytes

fn parse_blob(blob: &[u8]) -> Result<ParsedKey> {
    let mut c = Cursor::new(blob);

    // Magic
    let magic = c.read_bytes(MAGIC.len())?;
    if magic != MAGIC {
        bail!("not an OpenSSH private key (bad magic header)");
    }

    // ciphername
    let cipher = match c.read_str()? {
        "none"                          => Cipher::None,
        "aes256-ctr"                    => Cipher::Aes256Ctr,
        "chacha20-poly1305@openssh.com" => Cipher::Chacha20Poly1305,
        other                           => Cipher::Unsupported(other.to_string()),
    };

    // kdfname + kdfoptions
    let kdf = match c.read_str()? {
        "none" => {
            let _ = c.read_string()?; // consume empty options
            Kdf::None
        }
        "bcrypt" => {
            let opts = c.read_string()?;
            let mut oc = Cursor::new(opts);
            let salt   = oc.read_string()?.to_vec();
            let rounds = oc.read_u32_be()?;
            Kdf::Bcrypt { salt, rounds }
        }
        other => bail!("unsupported KDF: {}", other),
    };

    // number of keys (modern OpenSSH always writes 1)
    let _n = c.read_u32_be()?;

    // public key blob — dive inside to extract the key-type string
    let pub_blob  = c.read_string()?;
    let mut pc    = Cursor::new(pub_blob);
    let kt_str    = pc.read_str().unwrap_or("unknown");
    let key_type  = parse_key_type(kt_str);

    // encrypted private blob
    let encrypted_blob = c.read_string()?.to_vec();

    Ok(ParsedKey { key_type, cipher, kdf, encrypted_blob })
}

// ─────────────────────────────────────────────────────────────────────────────
// Public entry point
// ─────────────────────────────────────────────────────────────────────────────

pub fn parse_key_file(path: &str) -> Result<ParsedKey> {
    let pem = std::fs::read_to_string(path)
        .with_context(|| format!("cannot read key file: {}", path))?;

    // Reject legacy PEM formats — they use different encryption
    if pem.contains("BEGIN RSA PRIVATE KEY") {
        bail!(
            "Legacy PKCS#1 RSA key (not OpenSSH format).\n\
             Convert with:  ssh-keygen -p -f \"{}\" -m OpenSSH", path
        );
    }
    if pem.contains("BEGIN EC PRIVATE KEY") {
        bail!(
            "Legacy SEC1 EC key (not OpenSSH format).\n\
             Convert with:  ssh-keygen -p -f \"{}\" -m OpenSSH", path
        );
    }
    if pem.contains("BEGIN DSA PRIVATE KEY") {
        bail!(
            "DSA keys are NOT supported.\n\
             DSA (ssh-dss) was deprecated in OpenSSH 7.0 (2015) and disabled\n\
             by default since OpenSSH 8.8 (2021) due to weak cryptography.\n\
             Generate a modern key:  ssh-keygen -t ed25519"
        );
    }
    if !pem.contains("BEGIN OPENSSH PRIVATE KEY") {
        bail!("not a recognised SSH private key: {}", path);
    }

    // Strip PEM headers, base64-decode
    let b64: String = pem
        .lines()
        .filter(|l| !l.starts_with("-----"))
        .collect();
    let blob = B64.decode(b64.trim())
        .context("base64 decode of key body failed")?;

    let mut key = parse_blob(&blob)
        .with_context(|| format!("malformed key blob in: {}", path))?;

    // Companion .pub is more reliable for key-type detection — override
    if let Some(kt) = detect_from_pub(path) {
        key.key_type = kt;
    }

    // Final DSA check (could have come from the blob path)
    if key.key_type == KeyType::Dsa {
        bail!(
            "DSA keys are NOT supported.\n\
             Generate a modern key:  ssh-keygen -t ed25519"
        );
    }

    Ok(key)
}