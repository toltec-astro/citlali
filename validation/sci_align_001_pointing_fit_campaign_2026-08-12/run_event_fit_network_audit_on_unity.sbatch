#!/usr/bin/env bash
#SBATCH --job-name=sci-align-event-network-audit
#SBATCH --partition=toltec-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/work/toltec/wilson/sci-align-event-network-audit_%j.out
#SBATCH --error=/work/toltec/wilson/sci-align-event-network-audit_%j.err
set -euo pipefail

: "${SCI_REPO:?set SCI_REPO}"
: "${SCI_CAMPAIGN_ROOT:?set SCI_CAMPAIGN_ROOT}"

output="$SCI_CAMPAIGN_ROOT/event_centroid_fit_network_audit_v1"
test ! -e "$output"
mkdir -p "$SCI_CAMPAIGN_ROOT/_runtime_cache/network_audit/matplotlib"
mkdir -p "$SCI_CAMPAIGN_ROOT/_runtime_cache/network_audit/xdg"

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLBACKEND=Agg \
MPLCONFIGDIR="$SCI_CAMPAIGN_ROOT/_runtime_cache/network_audit/matplotlib" \
XDG_CACHE_HOME="$SCI_CAMPAIGN_ROOT/_runtime_cache/network_audit/xdg" \
python "$SCI_REPO/tools/diagnostics/audit_sci_align_001_lissajous_event_fit_networks.py" \
  --centroid-protocol "$SCI_REPO/validation/sci_align_001_lissajous_timestream_2026-08-10/event_centroid_protocol.json" \
  --campaign-audit "$SCI_CAMPAIGN_ROOT/event_centroid_fit_campaign_audit_v1" \
  --fit-root "$SCI_CAMPAIGN_ROOT/event_centroid_fit_results_v1" \
  --output "$output" \
  --cadence-ms 8.192
