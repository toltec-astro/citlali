#!/bin/bash
set -euo pipefail
repo=/Users/gwilson/.codex/worktrees/ptc-rank-response/citlali-refactor
root=/Users/gwilson/work_toltec/local_data/citlali-validation/successor-point152420-amplitude-20260930
cd "$repo"
export PYTHONDONTWRITEBYTECODE=1
/Users/gwilson/tolteca/bin/python "$repo/validation/point152420_amplitude_investigation_2026-09-30/frozen_calibration.py" run 2>&1 | tee "$root/batch.log"
