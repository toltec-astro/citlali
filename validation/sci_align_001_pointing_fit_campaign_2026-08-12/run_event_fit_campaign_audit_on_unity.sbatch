#!/usr/bin/env bash
#SBATCH --job-name=sci-align-event-audit
#SBATCH --partition=toltec-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/work/toltec/wilson/sci-align-event-audit_%j.out
#SBATCH --error=/work/toltec/wilson/sci-align-event-audit_%j.err
set -euo pipefail

: "${SCI_REPO:?set SCI_REPO}"
: "${SCI_CAMPAIGN_ROOT:?set SCI_CAMPAIGN_ROOT}"

output="$SCI_CAMPAIGN_ROOT/event_centroid_fit_campaign_audit_v1"
test ! -e "$output"
python "$SCI_REPO/tools/diagnostics/audit_sci_align_001_lissajous_event_fit_campaign.py" \
  --protocol "$SCI_CAMPAIGN_ROOT/frozen/timestream_protocol.json" \
  --selection "$SCI_CAMPAIGN_ROOT/frozen/selected_pointings.json" \
  --fit-root "$SCI_CAMPAIGN_ROOT/event_centroid_fit_results_v1" \
  --review-root "$SCI_CAMPAIGN_ROOT/event_centroid_review_results_v1" \
  --output "$output" \
  --require-complete
