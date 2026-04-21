
# kspr

Fast SSH private key passphrase finder written in Rust.

## Build

```bash
cd kspr
cargo build --release
```

## Usage

```bash
./target/release/kspr -k <key_path> -w <wordlist>
```

### Examples

```bash
./target/release/kspr -k ~/.ssh/id_ed25519 -w passwords.txt

./target/release/kspr -k ~/.ssh/id_rsa -w passwords.txt \
  --threads 8 --batch-size 8192 --verbose

./target/release/kspr -k ~/.ssh/id_ecdsa -w passwords.txt --cpu-only
```

## Options

```
-k <path>        Path to SSH private key
-w <file>        Wordlist file

--threads <n>    Number of threads
--batch-size <n> Work per iteration
--cpu-only       Disable GPU/acceleration
--verbose        Show progress
```

## Notes

* Supports common SSH key types (RSA, ECDSA, ED25519)
* Optimized for speed and parallel execution
* Wordlist must be newline separated
