# FRUIT EL-F8 analysis repair manifest r0.2

Test ID: `SCI-FRUIT-EL-F8-PENALTY-PLACEMENT-DECOMPOSITION-R0.1`

Status: **frozen before replacement analysis**

The complete R0.4 replay set is unchanged.  Analysis r0.2 corrects only the
application-accounting condition documented in `ANALYSIS_ABORT_R0.1.md` and
writes to the fresh `analysis-r0.2` directory.

| Object | Bytes | SHA-256 |
|---|---:|---|
| `REGISTRATION_R0.4.yaml` | 3975 | `107f47bd3a616b7add5a3163288605936e334d86345824454c539b2ef6234446` |
| `ANALYSIS_ABORT_R0.1.md` | 1528 | `8e5609bae0a6520d8dae327495b5947aab1725f00b72f7249a7e96af9f9e8b72` |
| `ANALYSIS_MANIFEST_R0.4_ANALYSIS_R0.2.yaml` | 3886 | `42bc5cabf84574c64d11716cc812442d471faf6731031667baa0928c719f51bc` |
| `tools/fruit_loops/analyze_penalty_placement.py` | 36792 | `43222586233f2143479236cf5fbda4476e7116afe0616b4eccdc8c3cf3033b9d` |
| `tools/fruit_loops/test_analyze_penalty_placement.py` | 6715 | `bb5d0b7d7426bc8d441c02ab937d9718c384e204cbcb9caef137c16c751d926d` |

The repaired analyzer is exact commit
`9ee9cfc98d1e0ba62b0d0e05bdbc88fe79bad518`.  All seven focused analyzer
tests, Ruff, byte compilation, and the repository whitespace check passed.
The replacement output directory did not exist when this manifest was
written.

The analysis continues to require the exact legacy-control compatibility
gate, both checkpoint-policy intervention re-audits, paired realized
configuration, units/WCS/grid/support compatibility, exact `Q` closure,
continuous component measurements, trigger-pixel evidence, and successful
execution logs.  No scientific criterion or claim limit changed.
