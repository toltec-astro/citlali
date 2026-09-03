# FRUIT EL-F8 analysis repair manifest r0.3

Test ID: `SCI-FRUIT-EL-F8-PENALTY-PLACEMENT-DECOMPOSITION-R0.1`

Status: **frozen before replacement analysis**

The complete R0.4 replay set is unchanged.  Analysis r0.3 adds the tested
macOS `/usr/bin/time -l` parser correction documented in
`ANALYSIS_ABORT_R0.2.md` and writes to the fresh `analysis-r0.3` directory.

| Object | Bytes | SHA-256 |
|---|---:|---|
| `REGISTRATION_R0.4.yaml` | 3975 | `107f47bd3a616b7add5a3163288605936e334d86345824454c539b2ef6234446` |
| `ANALYSIS_ABORT_R0.2.md` | 988 | `f84f1a35854731dd9302b2101c02c3dc074706748e7bae5d2702834743d1726f` |
| `ANALYSIS_MANIFEST_R0.4_ANALYSIS_R0.3.yaml` | 3886 | `939102ba31cab08d494c74ae92f0653a0d3c27cb2009e3d8d485320508b93183` |
| `tools/fruit_loops/analyze_penalty_placement.py` | 36833 | `9197da222e565021999159d2bb128b8fbbcca172bc9cda7e10829d0e9fb95335` |
| `tools/fruit_loops/test_analyze_penalty_placement.py` | 7359 | `34e2a00faf5440c10d5c8994b640b55340669b01c1798ea21b7f91a6cb52fe6a` |

The repaired analyzer is exact commit
`90b97cc9595b8a4ccfea6f421c72a55c9698242b`.  All eight focused analyzer
tests, Ruff, byte compilation, the real four-log execution reader, and the
repository whitespace check passed.  The replacement output directory did
not exist when this manifest was written.

Every scientific gate, component, region, metric, and claim limit remains as
registered.  Only successful-run resource parsing changed.
