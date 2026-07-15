# KIDs External Config Authority

This document fixes the bounded Phase 2 contract at the Citlali-to-Kidscpp
boundary. It does not move KIDs fitting into Citlali or redesign Kidscpp.

## Ownership

Kidscpp remains the execution authority for fitting raw KIDs data. Citlali
selects one supported solved TOD representation and supplies the existing
`kids.*` request to that dependency. Citlali records the boundary but does not
maintain a second implementation of its numerical policy.

All four current solved TOD types are supported:

- `xs`
- `rs`
- `is`
- `qs`

An unknown type remains a configuration error. This support declaration is
independent of the future R-analysis auxiliary-channel work, which remains
structure-only until its measured-channel contract is approved.

## Recorded Identity

`KidsExternalConfigPlan` preserves the requested KIDs fitter model, weight
window, fit-report directory, solver parallel policy, and optional
`solver.extra_output`. It separately records the effective values passed
through the established bridge, the selected TOD type, the TolTEC KIDs data
schema, and the exact Kidscpp build identity.

Historical Citlali behavior ignored `kids.solver.extra_output` and always ran
with extra solver output disabled. The effective plan records that resolution
explicitly instead of silently claiming the requested value took effect. No
numerical behavior changes in this checkpoint.

Successful reductions require an atomically published
`kids_external_provenance.yaml` using schema
`citlali-kids-external-provenance-v1`. A missing, incomplete, or unwritable
record fails the CLI success path.

## Stop Rule

Do not migrate Kidscpp fitting algorithms or workspaces into Citlali as part of
Phase 2. Further work at this boundary requires a concrete missing identity,
unsupported data schema, or measured integration defect.
