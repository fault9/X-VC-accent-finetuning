#!/bin/bash
# Repair data/distill_sources_asi after the latent_400 cleanup: the carrier
# dir was built as symlinks into data/crosspair_hindi_latent_400/wavs/src,
# which no longer exists. Replace every dangling link with a REAL copy of the
# same clip from the wide_v2 source pool (same raw clips, same preprocessing),
# so no future dataset cleanup can break the carriers again.
set -e
cd "$(dirname "$0")/.."
pool=data/crosspair_hindi_latent_wide_v2/wavs/src

missing=0
repaired=0
for f in data/distill_sources_asi/*.wav; do
  [ -e "$f" ] && continue            # still resolves -> leave untouched
  b=$(basename "$f")
  if [ -f "$pool/$b" ]; then
    rm "$f"
    cp "$pool/$b" "$f"
    repaired=$((repaired+1))
  else
    echo "MISSING from pool: $b"
    missing=$((missing+1))
  fi
done

if [ "$missing" -gt 0 ]; then
  echo "[repair] $missing clip(s) not in $pool -- dir NOT fully repaired" >&2
  exit 1
fi

dangling=$(find -L data/distill_sources_asi -type l | wc -l)
echo "[repair] replaced $repaired dangling link(s); $dangling still dangling"
echo "[repair] carrier count: $(ls data/distill_sources_asi/*.wav | wc -l)"

echo "[repair] other load-bearing data dirs:"
ls -d data/stackdistill_hindi_asi_l10/wavs data/stackdistill_hindi_asi_s15/wavs \
      data/accentbridge_pairs data/asi_selfpairs_wide70 2>&1 || true
