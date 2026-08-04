# SCI-MAP-001 Unity successor package (MAP-UNITY-ED2)

This is the bounded successor package for request `SCI-MAP-001-UNITY-001`,
revision `repair-sha-ed28dafb-ed1-2026-08-02`. It prepares, but does not run,
the owner-approved full/all PTC evidence route for the unchanged application
candidate `ed28dafb37f9113c0d3c95297148157129a90886` (tree
`cf75c36557178f351fb62781108a6f4b41b19225`).

The two auxiliary captures are separate from the unchanged seven acceptance
cases. CAP-POINT contains observation 152389. CAP-SCIENCE contains ordered
science observations 152390 and 152392; 152389, 152391, and 152393 are only
pointing support. One ordinary exact-candidate binary is built once and bound
for both captures and all seven cases. No application, estimator, build,
fixed-case, product, tolerance, or scientific-gate change belongs to this
package.

## Package surfaces

- `campaign.json` freezes identities, the seven cases, the two captures, the
  three-leaf merged-config allowlist, nine groups, and resource/retention
  policy.
- `tolproj-*-source.json` are the exact lightweight Point and Science project
  specifications. `processed-time-chunk-full-overlay.yaml` is the sole
  capture-only output overlay.
- `scripts/ed2-capture.py` generates and stages individual canonical raw-file
  links, copies only the named APT/PPT ECSV authorities, checks fully merged
  configs, binds the one binary and native/effective rates, inventories
  governed roots, and enforces the cumulative byte ceiling.
- `source-selection.schema.json` types the five factual, run-specific source
  roles consumed by the automatic raw-manifest generator; PTC, projection,
  rate, FWHM, and target roles are added from realized capture authorities.
- `scripts/compact-evidence.py` streams full PTC primitives into nine bounded
  compact groups, deterministic every-active-network traces, and separately
  requested focused expansion.
- `SCI-MAP-001-analysis.py` independently reconstructs the requested map,
  noise, F010, coadd, WCS, provenance, sequential/OpenMP, and F011 gates from
  the compact groups rather than final FITS payloads.
- `scripts/unity-campaign.py` retains the fail-closed one-build/seven-case
  orchestration and emits human commands without contacting Unity or
  submitting Slurm work.
- `resource-report.json` records local metadata measurements and the Unity
  planning estimate. `resource-projection.schema.json` binds that estimate to
  before-stage records, but it is not a full/all-PTC serialization upper bound
  or a guarantee. The owner reviews measured Unity-root use and capacity before
  choosing each later stage.
- `SHA256SUMS.ed2` is the exhaustive active package inventory. The older
  `SHA256SUMS` and the three files it inventories are immutable ED1 stop-return
  history.

## Scientific authority

Processed-TOD `SAMPRATE` remains the native `telescope.fsmp` value. Every
sample interval, exposure, and mapmaking reconstruction instead uses a
separate finite-positive realized `telescope.d_fsmp` record, preserved in
decimal and C99 hexadecimal binary64 form and checked against the realized
scan/sample plan. The raw-timestream provenance must also carry a positive
integral downsample factor, and `telescope.d_fsmp` must be bit-equal to
`telescope.fsmp / downsample_factor`. Compact groups bind raw-timestream and
mapmaking provenance
with distinct SHA-256 fields, so rate authority cannot be substituted by the
projection authority. Missing primitive or effective-rate authority is a
stop; it cannot be inferred from final FITS products.

The producer emits exactly nine observation/array groups. Each group contains
domain-separated complete-order/population digests, scan-first binary64
sufficient statistics, pinned 64-realization signs, and a fixed trace set for
first/middle/last scans and valid/flagged representatives of every active
network (or typed class-absence facts). Schemas reject a full primitive-term
axis. Focused expansion is a separately invoked, bounded two-pass operation.

## Resource and retention boundary

The combined governed Unity roots have an inclusive limit of 214,748,364,800
bytes; this does not constrain local package work. Before and after every
future material stage, every root, directory, regular file, and symlink is
inventoried without following symlink payloads. The record binds current
logical and allocated use, selected-filesystem capacity, and the frozen local
planning estimate. That estimate is operational context only—not a
source-derived full/all-PTC byte proof. The owner reviews each pre-record and
stops before the next stage if observed use approaches or exceeds the cap, or
capacity is inadequate. Capture, all nine compact groups, each focused
expansion pass, analysis, and final-bundle construction require pre/post
records. No record authorizes deletion or a larger ceiling.

Individual symlink targets under the owner-verified canonical raw root are
pre-existing inputs and excluded from generated usage; the links, copied
ECSV authorities, captures, logs, intermediates, compact artifacts, and all
analysis/manifest/return outputs under `compact-groups/_campaign` are
included. Both full captures remain retained through fresh MAP re-audit and
any requested focused expansion. Nothing in this package automatically cleans
them up.

The future return tar excludes the exact PTC paths bound by both capture
records and carries a separate exhaustive return manifest. The full remote
inventory still binds the retained payloads without transferring them.

## Use boundary

Read `OWNER_RUNBOOK.md`, `LAUNCH_CHECKLIST.md`, and `EVIDENCE_BOUNDARIES.md`
before any future execution. Every remote command is human-only and uses
`unity_toltec`. This implementation did not push, contact Unity, transfer
files, build or reduce there, submit Slurm, fill owner values, integrate the
repair, supply external evidence, close findings or dependencies, launch
re-audit, expand production, or execute cleanup.
