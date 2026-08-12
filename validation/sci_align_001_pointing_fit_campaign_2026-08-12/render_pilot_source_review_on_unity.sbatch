#!/usr/bin/env bash
#SBATCH --job-name=sci-align-source-review
#SBATCH --partition=toltec-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/work/toltec/wilson/sci-align-source-review_%j.out
#SBATCH --error=/work/toltec/wilson/sci-align-source-review_%j.err
set -euo pipefail

: "${SCI_REPO:?set SCI_REPO}"
: "${SCI_CAMPAIGN_ROOT:?set SCI_CAMPAIGN_ROOT}"

runtime_cache="$SCI_CAMPAIGN_ROOT/_runtime_cache"
mkdir -p "$runtime_cache/matplotlib" "$runtime_cache/xdg"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="$runtime_cache/matplotlib"
export XDG_CACHE_HOME="$runtime_cache/xdg"

python \
  "$SCI_REPO/tools/diagnostics/render_sci_align_001_lissajous_fit_gate_source_review.py" \
  --protocol "$SCI_CAMPAIGN_ROOT/frozen/timestream_protocol.json" \
  --selection "$SCI_CAMPAIGN_ROOT/frozen/selected_pointings.json" \
  --fit-gate-root "$SCI_CAMPAIGN_ROOT/fit_results/o150818" \
  --obsnum 150818 \
  --selected-count 16 \
  --output "$SCI_CAMPAIGN_ROOT/fit_results/o150818_source_review_v1"
