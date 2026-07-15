#!/usr/bin/env bash
# Queue both sides of the genuine phone-tier MFA job. This script performs no
# feature/audio warping and does not start X-VC training.
set -euo pipefail
cd "$(dirname "$0")/.."

MFA_ROOT="${MFA_ROOT:-data/mfa_hindi_phoneaware_asi}"
SOURCE_CORPUS="${SOURCE_CORPUS:-$MFA_ROOT/mfa_corpus/source}"
TARGET_CORPUS="${TARGET_CORPUS:-$MFA_ROOT/mfa_corpus/target}"
SOURCE_ALIGN_DIR="${SOURCE_ALIGN_DIR:-$MFA_ROOT/mfa_align/source}"
TARGET_ALIGN_DIR="${TARGET_ALIGN_DIR:-$MFA_ROOT/mfa_align/target}"
VALIDATION_ROOT="${VALIDATION_ROOT:-$MFA_ROOT/mfa_validation}"
DICTIONARY="${MFA_DICTIONARY:-english_us_mfa}"
ACOUSTIC_MODEL="${MFA_ACOUSTIC_MODEL:-english_mfa}"
MFA_BIN="${MFA_BIN:-mfa}"
PYTHON_BIN="${PYTHON_BIN:-python}"
REUSE_MFA="${REUSE_MFA:-0}"
MFA_NUM_JOBS="${MFA_NUM_JOBS:-4}"
MFA_TEST_TRANSCRIPTIONS="${MFA_TEST_TRANSCRIPTIONS:-0}"

for corpus in "$SOURCE_CORPUS" "$TARGET_CORPUS"; do
  [[ -d "$corpus" ]] || { echo "[error] missing MFA corpus: $corpus" >&2; exit 1; }
  n_wav=$(find "$corpus" -type f -name '*.wav' | wc -l)
  n_lab=$(find "$corpus" -type f \( -name '*.lab' -o -name '*.txt' \) | wc -l)
  [[ "$n_wav" -gt 0 && "$n_wav" -eq "$n_lab" ]] || {
    echo "[error] $corpus has wav=$n_wav transcript=$n_lab" >&2
    exit 1
  }
done
command -v "$MFA_BIN" >/dev/null || {
  echo "[error] '$MFA_BIN' not found; activate the MFA environment first" >&2
  exit 1
}

if [[ "$REUSE_MFA" != "1" ]] && \
   { [[ -d "$SOURCE_ALIGN_DIR" ]] || [[ -d "$TARGET_ALIGN_DIR" ]]; }; then
  echo "[error] MFA output already exists. Use a new MFA_ROOT, or set " \
       "REUSE_MFA=1 only after verifying it is complete." >&2
  exit 1
fi

mkdir -p "$VALIDATION_ROOT"
echo "=== PHONE-AWARE MFA QUEUE START $(date) ==="
echo "MFA version: $($MFA_BIN version)"
echo "dictionary: $DICTIONARY"
echo "acoustic model: $ACOUSTIC_MODEL"
echo "num jobs: $MFA_NUM_JOBS"
echo "test transcriptions: $MFA_TEST_TRANSCRIPTIONS"

validate_extra=(--clean --num_jobs "$MFA_NUM_JOBS")
if [[ "$MFA_TEST_TRANSCRIPTIONS" == "1" ]]; then
  validate_extra+=(--test_transcriptions)
fi

if [[ "$REUSE_MFA" != "1" ]]; then
  echo "=== VALIDATE SOURCE $(date) ==="
  "$MFA_BIN" validate "$SOURCE_CORPUS" "$DICTIONARY" \
    --acoustic_model_path "$ACOUSTIC_MODEL" \
    "${validate_extra[@]}" \
    --output_directory "$VALIDATION_ROOT/source"

  echo "=== ALIGN SOURCE $(date) ==="
  "$MFA_BIN" align "$SOURCE_CORPUS" "$DICTIONARY" "$ACOUSTIC_MODEL" \
    "$SOURCE_ALIGN_DIR" --clean --num_jobs "$MFA_NUM_JOBS"

  echo "=== VALIDATE TARGET $(date) ==="
  "$MFA_BIN" validate "$TARGET_CORPUS" "$DICTIONARY" \
    --acoustic_model_path "$ACOUSTIC_MODEL" \
    "${validate_extra[@]}" \
    --output_directory "$VALIDATION_ROOT/target"

  echo "=== ALIGN TARGET $(date) ==="
  "$MFA_BIN" align "$TARGET_CORPUS" "$DICTIONARY" "$ACOUSTIC_MODEL" \
    "$TARGET_ALIGN_DIR" --clean --num_jobs "$MFA_NUM_JOBS"
else
  echo "=== REUSING EXISTING MFA OUTPUTS ==="
fi

echo "=== AUDIT GENUINE PHONE TIERS $(date) ==="
"$PYTHON_BIN" scripts/audit_mfa_phone_tiers.py \
  --source-align-dir "$SOURCE_ALIGN_DIR" \
  --target-align-dir "$TARGET_ALIGN_DIR" \
  --out "$MFA_ROOT/phone_tier_audit.json"

archive="${MFA_ARCHIVE:-${MFA_ROOT%/}_phone_textgrids.tgz}"
archive_parent=$(dirname "$archive")
mkdir -p "$archive_parent"
tar -czf "$archive" -C "$MFA_ROOT" \
  mfa_align mfa_validation phone_tier_audit.json
sha256sum "$archive" > "$archive.sha256"

echo "=== PHONE-AWARE MFA QUEUE DONE $(date) ==="
echo "archive: $archive"
cat "$archive.sha256"
echo "Next: while mfa_corpus and mfa_align still coexist, build the canonical"
echo "pristine dataset with scripts/build_pristine_parallel_dataset.py."
echo "See docs/mfa_phone_alignment.md and docs/datasets.md."
