#!/bin/bash
set -euo pipefail
repo=/Users/gwilson/.codex/worktrees/ptc-rank-response/citlali-refactor
cd "$repo"
export PYTHONDONTWRITEBYTECODE=1
/Users/gwilson/tolteca/bin/python "$repo/validation/point152420_amplitude_investigation_2026-09-30/frozen_calibration.py" run
