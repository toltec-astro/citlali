# SCI-FRUIT EL-F10-R1 registration manifest r0.2

Test ID: `SCI-FRUIT-EL-F10-R1-COMPATIBILITY-NORMALIZATION-R0.1`

Status: **frozen after the replay and before accounting values were opened**

The scientific owner authorized the exact no-replay repair in
`EL_F10_R1_BUNDLE_MANIFEST_R0.1.md`. The authorization is recorded in
`SCIENTIFIC_OWNER_EL_F10_R1_AUTHORIZATION_2026-09-04.md`.

`REGISTRATION_R0.2.yaml` is 9,486 bytes with SHA-256
`8ac3f36d75bfacf718aac3220277c224d6adf3e055880da1e55782a7e7a43b46`.
All 21 files registered there passed size and SHA-256 validation before this
manifest was written. They include the immutable r0.1 registration and test
definition, exact frozen analyzer, authorization packet, r0.1 failure record,
retained receipt and target ledger, replay checkpoint and map products, log,
and historical comparison products.

No additional Citlali replay is authorized or required. The exact retained
receipt and target ledger remain unchanged from the hashes recorded before
approval.

## Sole repaired gate

The checkpoint comparison may now admit `learning_policy_yaml` as a changed
container only when:

- the observed checkpoint differences are exactly `creator_version` and
  `learning_policy_yaml`;
- both learning policies parse as scalar key/value maps;
- historical absence of only
  `map_pixel_outlier_detector_exclusion_application` is normalized to
  `pre_cleaning`; and
- the normalized maps then match exactly.

Every other checkpoint value, structure, ordinary-map bitwise comparison,
total-accumulator closure, sample ledger, forward-error formula, safety factor,
support rule, region, trigger pixel, descriptive output, and claim limit is
unchanged from r0.1.

The exact required checkpoint-difference set will be checked before the
frozen analyzer opens either accounting file. A failure stops the analysis.
