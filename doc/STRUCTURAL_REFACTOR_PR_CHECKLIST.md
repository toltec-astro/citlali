# Citlali Structural Refactor PR Checklist

Use this checklist for each focused refactor branch or pull request.

## Scope

- Branch:
- Base branch:
- Goal:
- Files/modules touched:
- Affected modes: science / pointing / beammap / polarimetry / TOD output
- Hot paths touched: yes / no
- Expected behavior change: none / documented below

## Behavior

- User-facing CLI behavior unchanged:
- YAML compatibility unchanged:
- Output directory layout unchanged:
- Product names/formats unchanged:
- Science behavior change, if any:

## Validation

- Local checks run:
- Unity compile run by maintainer:
- Unity reductions run by maintainer:
- Baseline manifest:
- Candidate manifest:
- Comparator command:
- Comparator result:

## Products Compared

- FITS maps:
- netCDF TOD:
- ECSV/CSV tables:
- Logs:
- Science metrics:

## Performance

- Wall-time baseline:
- Wall-time candidate:
- Peak RSS baseline:
- Peak RSS candidate:
- Stage timings, if available:
- Regression within 3-5% budget: yes / no / not measured

## Risk

- Residual risk:
- Follow-up tests:
- Follow-up refactor work:
- Reviewer notes:
