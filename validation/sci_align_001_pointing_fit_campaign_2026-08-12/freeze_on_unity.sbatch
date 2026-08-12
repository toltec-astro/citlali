#!/usr/bin/env bash
#SBATCH --job-name=sci-align-freeze-66
#SBATCH --partition=toltec-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/work/toltec/wilson/sci-align-pointing-freeze_%j.out
#SBATCH --error=/work/toltec/wilson/sci-align-pointing-freeze_%j.err
set -euo pipefail

: "${SCI_REPO:?set SCI_REPO}"
: "${SCI_SCHEMA_AUDIT:?set SCI_SCHEMA_AUDIT}"
: "${SCI_CAMPAIGN_ROOT:?set SCI_CAMPAIGN_ROOT}"

python \
  "$SCI_REPO/tools/diagnostics/prepare_sci_align_001_pointing_fit_campaign.py" \
  freeze \
  --schema-audit "$SCI_SCHEMA_AUDIT" \
  --quality-package \
    "$SCI_REPO/validation/sci_align_001_pointing_source_quality_2026-08-12" \
  --map-protocol-template \
    "$SCI_REPO/validation/sci_align_001_lissajous_pointing_2026-08-10/frozen_protocol.json" \
  --timestream-protocol-template \
    "$SCI_REPO/validation/sci_align_001_lissajous_timestream_2026-08-10/frozen_protocol.json" \
  --repo-root "$SCI_REPO" \
  --output-root "$SCI_CAMPAIGN_ROOT" \
  --python python \
  --array-concurrency 4
