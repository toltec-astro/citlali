# SCI-MAP-001-UNITY-001 repair-candidate campaign

This directory is the frozen, human-run Unity campaign package for Citlali
candidate `ed28dafb37f9113c0d3c95297148157129a90886`. It prepares evidence; it
does not contain external evidence and does not authorize Codex to access
Unity.

The request identity remains `SCI-MAP-001-UNITY-001`, with revision
`repair-sha-ed28dafb-2026-08-01`. All seven cases now have an expected exit
status of zero. In particular, `S-X-SEQ` keeps its historical
`sci_map_001_science_expected_failure` jobkey only as an identity anchor; the
former publication error must be absent and both observation and coadd output
families must complete.

## Package contents

- `campaign.json`: immutable candidate, authority, case, resource, tolerance,
  dependency-state, and numbered-config contract.
- `owner-values.template.json` and schema: every deployment choice the owner
  must supply. Nulls are deliberate stop conditions, not defaults.
- `raw-input-manifest.{point,science}.template.json` and schema: exact frozen
  producer/source, scan, detector, sample-cardinality, projection, target,
  FWHM, and sample-rate authority the owner must supply before case setup.
- `result-collection.template.json` and schema: seven run roots, complete job
  records, and the nine independent observation reconstruction ledgers.
- `sample-ledger-contract.json`: exact processed-term interface required to
  reconstruct observation F010 facts without using output facts as their own
  authority.
- `SCI-MAP-001-analysis.py`: frozen generator, preflight, collector, and
  numerical verifier, including contract-derived diagnostic inventories and
  lossless per-map residual NPZs with a digest manifest.
- `scripts/unity-campaign.py`: human-run preparation driver. It never submits
  a job; it emits the submission plan for owner review.
- `scripts/case-job-wrapper.sh`: allocation-local identity/resource/log
  wrapper.
- `scripts/analysis-job-wrapper.sh`: bounded analysis allocation and immutable
  analyzer/registry/input wrapper.
- `scripts/hash-tree.py`: complete deterministic path/size/mtime/digest
  inventory utility.
- `scripts/verify-package.sh`: exhaustive package digest, self-check, syntax,
  and compilation preflight.
- `OWNER_RUNBOOK.md`: exact preparation, execution, collection, analysis, and
  return procedure.
- `LAUNCH_CHECKLIST.md`: short copy/paste handoff.
- `SHA256SUMS`: package digest authority.

## Fail-closed boundaries

The owner must provide all values in `owner-values.template.json`; none is
guessed. Missing source, observation, array/network, APT, calibration,
pointing, dependency, site, module, sample-ledger, or Slurm identity is an
evidence gap. The program must not infer sample-origin F010 values from the
published F010 planes.

Deployment paths are canonical absolute POSIX paths with no trailing slash.
Owner values, raw manifests, independent ledgers, generated plans, and returned
bundles are single-shot artifacts: existing or symlinked state is preserved
for inspection and never overwritten by the campaign.

The native TolProj directory has nine runtime-recognized numbered sources:
TolTECA-owned `40_setup.yaml`, the seven ordered mode files, and generated
`99_zz_tolproj_submission_runtime.yaml`. The original audit appendix counted
only the eight post-setup files. This package preserves and hashes all nine,
requires exactly eight after `40_setup.yaml`, proves deployed `RuntimeContext`
order, and rejects a tenth source.

## Status boundary

F009 and F010 remain `addressed_pending_reaudit`; F012 remains the human-run
exact-repair-SHA evidence gate. F013 remains conditioned on the named upstream
audits. ALIGN-OD1 through ALIGN-OD8 and ALIGN-C001 are owner-approved at record
commit `4f905f4f353e91847a303f4f3959654f3f03c302`, with canonical identity
correction at `35cc8ce246e8e70c569e650be6c1eae2c91b80ef`; the bounded handoff is integrated
at `0309fd48a973a6e7e136224906ac49c02f0171be`, with coordination
ledger HEAD `846128c8ee6dc27851bd6c71aeecbe4739e1d24a`. No ALIGN application-repair
commit or re-audit exists: implementation remains nonconformant, validation is
in progress, and production remains `existing_use_only`. This campaign cannot
close MAP findings or any ALIGN, CAL, AST, PTC, or VAL dependency, and cannot
promote production use.

Run `scripts/verify-package.sh` locally before transfer. Then follow
`OWNER_RUNBOOK.md` exactly.
