# FRUIT EL-F10-R1 result manifest r0.2

Test ID: `SCI-FRUIT-EL-F10-R1-COMPATIBILITY-NORMALIZATION-R0.1`

Status: **valid repaired-compatibility pass followed by analysis-reader stop;
no accounting result**

## R1 authorization and registration

| Object | Bytes | SHA-256 |
|---|---:|---|
| `SCIENTIFIC_OWNER_EL_F10_R1_AUTHORIZATION_2026-09-04.md` | 1,030 | `e80646757b629a2ce67e39946c070c97656c22373dcf3a144cab585ff1e05d17` |
| `EL_F10_R1_BUNDLE_MANIFEST_R0.1.md` | 4,256 | `ee353ac67630a77ab1727084475eb6db86ca46d583bd43c16d06e3fe52d6a217` |
| `REGISTRATION_R0.2.yaml` | 9,486 | `8ac3f36d75bfacf718aac3220277c224d6adf3e055880da1e55782a7e7a43b46` |
| `REGISTRATION_MANIFEST_R0.2.md` | 1,851 | `fa1f09d02e1100235e835469c1c59a01e17e30a5e53c663bd87aabd078ba4969` |
| `tools/fruit_loops/analyze_jinc_accounting.py` | 27,141 | `f78d033af1d7fb68b8c5a73197cbcf1b1d936b1cecaa92b4241dc76693da31e4` |

The owner authorization and bundle are under
`doc/scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane`; unqualified
registration and result names are in this validation directory.

## Result records

| Object | Bytes | SHA-256 |
|---|---:|---|
| `R1_ANALYSIS_ABORT_R0.2.json` | 1,100 | `35deb1cc0c666c98b7ee7d4de10caf7e16ff3c6fcefc9f9f0da6c53d49296ffd` |
| `EXECUTION_RESULT_R0.2.md` | 2,934 | `4ff2926beba01a53d757e9157866836ea115e919b7cc4ae5605249f88b2fcf13` |

## Verification and disposition

- all 21 R1-registered files passed their frozen sizes and SHA-256 identities;
- the observed checkpoint difference set was exactly `creator_version` and
  `learning_policy_yaml`;
- the old and new learning policies matched exactly after only the authorized
  missing-key-to-`pre_cleaning` normalization;
- all nine ordinary science planes and all three formal-coefficient planes
  again passed their bitwise comparisons;
- the analyzer then failed while reading the receipt's `schema_identity`
  because it attempted `.item()` on a native Python `str`;
- no total or target `N`, `C`, or `Q` plane was read, the target ledger was not
  opened, and no analysis product was written;
- no Citlali replay, external-file change, or Unity activity occurred; and
- the retained external files remain bound by `REGISTRATION_R0.2.yaml`.

This is not a scientific accounting result. A separately approved parser-only
repair and newly frozen analyzer are required before another analysis attempt.
