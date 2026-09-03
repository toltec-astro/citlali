# FRUIT EL-F8 analysis repair manifest r0.4

Test ID: `SCI-FRUIT-EL-F8-PENALTY-PLACEMENT-DECOMPOSITION-R0.1`

Status: **frozen before replacement analysis**

The complete R0.4 replay set is unchanged.  Analysis r0.4 adds only the
explicit built-in scalar conversion documented in `ANALYSIS_ABORT_R0.3.md`
and writes to the fresh `analysis-r0.4` directory.

| Object | Bytes | SHA-256 |
|---|---:|---|
| `REGISTRATION_R0.4.yaml` | 3975 | `107f47bd3a616b7add5a3163288605936e334d86345824454c539b2ef6234446` |
| `ANALYSIS_ABORT_R0.3.md` | 1388 | `d37d2dd2565f1abe04e68b8b5d936f93068acf5a5d1640b712c52fcd48cec14d` |
| `ANALYSIS_MANIFEST_R0.4_ANALYSIS_R0.4.yaml` | 3886 | `7228a6dd68f697b98341a140b2fabe8002b4472168b8ed6c104d25b17805ab67` |
| `tools/fruit_loops/analyze_penalty_placement.py` | 36959 | `0023b28338657b08f69647330f7b559565ed4b4f468b9eb74390f1c2ec189ec4` |
| `tools/fruit_loops/test_analyze_penalty_placement.py` | 7635 | `d4aaf298c677b05dcb8d710291713d8f434fb5b8ae5f30d357f9cf90797d16a3` |

The repaired analyzer is exact commit
`7d5d2f647ceecfd37ee01e7b804436681f709c1d`.  All nine focused analyzer
tests, Ruff, byte compilation, and a dry full analysis through successful JSON
serialization passed.  All three array closure checks passed in that dry run.
The replacement output directory did not exist when this manifest was
written.

Every scientific gate, component, region, metric, and claim limit remains as
registered.  Only the in-memory scalar representation used for JSON output
changed.
