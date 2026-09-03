# FRUIT EL-F7 result manifest r0.1

Test ID: `SCI-FRUIT-EL-F7-SHARED-START-RESPONSE-DECOMPOSITION-R0.1`

Status: **valid completed development result**

## Repository evidence

| Object | Bytes | SHA-256 |
|---|---:|---|
| `REGISTRATION_MANIFEST_R0.1.md` | 1858 | `e90d2d8bba29500d381c891b502171557b9c36e5ace088e3b4e8514a753090de` |
| `FROZEN_INPUTS_R0.1.md` | 4368 | `0ff4d18c0c5ad15b8c2e722eb37c1e4e6661e507bc415b7fd239a7e672b4be05` |
| `EXECUTION_RESULT_R0.1.md` | 8750 | `4914b64b829a34b8aa0f7d2813e57a755f1b5b13364fd49681b550d300c7955b` |
| `DECOMPOSITION_RESULT_R0.1.json` | 3735 | `e0b57f21d00c6094f3142eee6291831c59bba41d686c0ba77a53af0ad9e5e355` |
| `COMPONENT_METRICS_R0.1.csv` | 5023 | `acd565087c81d342ae49d57b9e97ce545e0377b171762abd8b901773908985f3` |
| `CROSS_TERMS_R0.1.csv` | 7742 | `26127bf3ef60870b04b1a330e92b9ca861365dd4e365a7de93368643955ff051` |
| `CHECKPOINT_DIFFERENCES_R0.1.json` | 1400 | `e8bbc9e0a504e75d5e11888f91125ce091056d4e602962820206ebc7e80a1de1` |
| `PRIMARY_EXECUTION_R0.1.csv` | 219 | `9b5df86c3864cd4eb1f63c8b3179ff3e409f27cde574a4aa375c08223c7657b2` |
| `RESPONSE_DECOMPOSITION_R0.1.png` | 963366 | `1c5e6f4a980bd11f7760cb3b1d72a04fd7575103c877af82313d79fa9f7f89c8` |
| `ANALYSIS_PROVENANCE_R0.1.yaml` | 10787 | `c8bcbd5129714618c875b374de1fba58d6e9d7488bdeb83add039830628605f2` |

## Complete external component products

The complete component and fixed-kernel-residual maps are retained outside
Git under the registered EL-F7 analysis directory.

| Array | Bytes | SHA-256 |
|---|---:|---|
| a1100 | 8182080 | `4a04ac832c2e51a4b62189ec8a150272ad8988d71a9a84626d401f0641221c29` |
| a1400 | 8182080 | `0d6d0caa11eb994869812131bb962316da9ddc0851709e97497c78075fd8c732` |
| a2000 | 8182080 | `1173ae376ff15f15445d0eb59eb817da279805c7c7ebc4439b8189fb5eb1338c` |

Each FITS file contains `T5`, `S5`, `H5`, and `D5` signal components and their
fixed-`P5`-kernel residual maps on the unchanged 355 by 357 map grid.

## Verification

- all registered sham, configuration, identity, WCS/grid, support, and closure
  gates passed;
- the original frozen executable and control iteration-4 checkpoint retain
  their registered SHA-256 identities after execution;
- all 618 enabled CTest cases passed; one unrelated case remains disabled;
- all 228 baseline and FRUIT-loop Python tests passed;
- the complete required configuration preflight passed;
- the changed Python files passed Ruff and byte compilation;
- the retained result structures and FITS extensions were reopened and
  checked; and
- the repository whitespace check passed.

The result records no qualification, method selection, safeguard selection,
production change, Gate-D launch, Stage B authoring, or Unity activity.
