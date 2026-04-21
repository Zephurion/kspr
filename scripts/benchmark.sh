#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# ------------------------------------------------------------------------------
# SSH private key password cracker benchmark (John the Ripper vs. KSPR)
# ------------------------------------------------------------------------------

# --- Configuration ------------------------------------------------------------
KEY_JOHN="$HOME/.ssh/id_rsaebpf"
KEY_KSPR="$HOME/.ssh/id_edebpf"
WORDLIST="$HOME/PWD/passwords.txt"
RUNS=15
JOHN_THREADS=8
KSPR_THREADS=8

# Paths to binaries
JOHN_BIN="john"
SSH2JOHN_PY="/usr/share/john/ssh2john.py"
KSPR_BIN="$HOME/kspr/target/release/kspr"

# Temporary file (will be set with mktemp)
TEMP_HASH_FILE=""

# --- Helper Functions ---------------------------------------------------------
cleanup() {
    # Remove temporary hash file
    if [[ -n "$TEMP_HASH_FILE" && -f "$TEMP_HASH_FILE" ]]; then
        rm -f "$TEMP_HASH_FILE"
    fi
    # Remove John's pot file and recovery files (only those from current session)
    rm -f "$HOME/.john/john.pot" "$HOME/.john/"*.rec 2>/dev/null || true
}
trap cleanup EXIT INT TERM

die() {
    echo "ERROR: $*" >&2
    exit 1
}

check_dependencies() {
    local deps=("bc" "$JOHN_BIN" "$KSPR_BIN")
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" >/dev/null 2>&1; then
            die "Required program not found: $dep"
        fi
    done

    if [[ ! -f "$SSH2JOHN_PY" ]]; then
        die "ssh2john.py not found at $SSH2JOHN_PY"
    fi

    if [[ ! -f "$KEY_JOHN" ]]; then
        die "John SSH key not found: $KEY_JOHN"
    fi

    if [[ ! -f "$KEY_KSPR" ]]; then
        die "KSPR SSH key not found: $KEY_KSPR"
    fi

    if [[ ! -f "$WORDLIST" ]]; then
        die "Wordlist not found: $WORDLIST"
    fi
}

# Time a command using 'time' builtin or external, but we use date +%s.%N for precision
run_timed() {
    local start end duration
    start=$(date +%s.%N)
    "$@" >/dev/null 2>&1
    end=$(date +%s.%N)
    duration=$(echo "$end - $start" | bc)
    echo "$duration"
}

# ------------------------------------------------------------------------------
# Pre-flight checks
# ------------------------------------------------------------------------------
check_dependencies

# Create a temporary file for John's hash
TEMP_HASH_FILE=$(mktemp) || die "Failed to create temporary file"

echo "Running benchmark ($RUNS runs each)..."
echo "====================================="

# ------------------------------------------------------------------------------
# John the Ripper benchmark
# ------------------------------------------------------------------------------
echo ""
echo ">>> JOHN TEST <<<"
JOHN_TOTAL=0

for i in $(seq 1 "$RUNS"); do
    echo "[John Run $i]"

    # Generate hash file
    python3 "$SSH2JOHN_PY" "$KEY_JOHN" > "$TEMP_HASH_FILE" 2>/dev/null || {
        echo "Warning: ssh2john.py failed for run $i" >&2
        continue
    }

    # Benchmark John
    duration=$(run_timed "$JOHN_BIN" --wordlist="$WORDLIST" --fork="$JOHN_THREADS" "$TEMP_HASH_FILE")
    echo "Time: $duration s"

    JOHN_TOTAL=$(echo "$JOHN_TOTAL + $duration" | bc)

    # Clean John's state between runs
    rm -f "$HOME/.john/john.pot" "$HOME/.john/"*.rec 2>/dev/null || true
done

JOHN_AVG=$(echo "scale=6; $JOHN_TOTAL / $RUNS" | bc)

# ------------------------------------------------------------------------------
# KSPR benchmark
# ------------------------------------------------------------------------------
echo ""
echo ">>> KSPR TEST <<<"
KSPR_TOTAL=0

for i in $(seq 1 "$RUNS"); do
    echo "[KSPR Run $i]"

    duration=$(run_timed "$KSPR_BIN" \
        -k "$KEY_KSPR" \
        -w "$WORDLIST" \
        --threads "$KSPR_THREADS" \
        --cpu-only)

    echo "Time: $duration s"
    KSPR_TOTAL=$(echo "$KSPR_TOTAL + $duration" | bc)
done

KSPR_AVG=$(echo "scale=6; $KSPR_TOTAL / $RUNS" | bc)

# ------------------------------------------------------------------------------
# Final results
# ------------------------------------------------------------------------------
echo ""
echo "====================================="
echo "FINAL RESULTS"
echo "====================================="
echo "John avg : $JOHN_AVG seconds"
echo "KSPR avg : $KSPR_AVG seconds"

# Calculate speed ratio
ratio=$(echo "scale=3; $JOHN_AVG / $KSPR_AVG" | bc)
echo ""
echo "Speed ratio (John / KSPR): $ratio"
