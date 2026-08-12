#!/usr/bin/env bash
#SBATCH --job-name=sci-align-event-pilot
#SBATCH --partition=toltec-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/work/toltec/wilson/sci-align-event-pilot_%j.out
#SBATCH --error=/work/toltec/wilson/sci-align-event-pilot_%j.err
set -euo pipefail

: "${SCI_REPO:?set SCI_REPO}"
: "${SCI_CAMPAIGN_ROOT:?set SCI_CAMPAIGN_ROOT}"

obsnum=150818
protocol="$SCI_CAMPAIGN_ROOT/frozen/timestream_protocol.json"
crossing_protocol="$SCI_REPO/validation/sci_align_001_lissajous_timestream_2026-08-10/crossing_support_protocol.json"
centroid_protocol="$SCI_REPO/validation/sci_align_001_lissajous_timestream_2026-08-10/event_centroid_protocol.json"
selection="$SCI_CAMPAIGN_ROOT/frozen/selected_pointings.json"
map_root="$SCI_CAMPAIGN_ROOT/map_results"
fit_root="$SCI_CAMPAIGN_ROOT/event_centroid_fit_results_v1/o${obsnum}"
review_root="$SCI_CAMPAIGN_ROOT/event_centroid_review_results_v1/o${obsnum}"
mplconfig="$SCI_CAMPAIGN_ROOT/_runtime_cache/event_pilot_matplotlib"
xdg_cache="$SCI_CAMPAIGN_ROOT/_runtime_cache/event_pilot_xdg"
mkdir -p "$mplconfig" "$xdg_cache"
export MPLBACKEND=Agg
export MPLCONFIGDIR="$mplconfig"
export XDG_CACHE_HOME="$xdg_cache"

if test -f "$fit_root/fit_gate.json"; then
  (cd "$fit_root" && shasum -a 256 -c FIT_GATE_SHA256SUMS)
else
  python "$SCI_REPO/tools/diagnostics/run_sci_align_001_lissajous_event_fit_gate.py" \
    --protocol "$protocol" \
    --crossing-protocol "$crossing_protocol" \
    --centroid-protocol "$centroid_protocol" \
    --selection "$selection" \
    --map-root "$map_root" \
    --obsnum "$obsnum" \
    --output "$fit_root" \
    --maximum-wall-seconds 1200
fi

if test -f "$review_root/manifest.json"; then
  (cd "$review_root" && shasum -a 256 -c SHA256SUMS)
else
  python "$SCI_REPO/tools/diagnostics/render_sci_align_001_lissajous_event_centroid_fit_gate.py" \
    --protocol "$protocol" \
    --crossing-protocol "$crossing_protocol" \
    --centroid-protocol "$centroid_protocol" \
    --selection "$selection" \
    --fit-gate-root "$fit_root" \
    --obsnum "$obsnum" \
    --selected-count 16 \
    --output "$review_root"
fi

printf 'event pilot complete: obs=%s fit=%s review=%s\n' \
  "$obsnum" "$fit_root" "$review_root"
