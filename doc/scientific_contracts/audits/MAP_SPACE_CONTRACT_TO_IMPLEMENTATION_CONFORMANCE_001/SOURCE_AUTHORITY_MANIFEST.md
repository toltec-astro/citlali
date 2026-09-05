# MAP-SPACE-CONTRACT-TO-IMPLEMENTATION-CONFORMANCE-001 Source Authority Manifest

Status: **candidate manifest bound to exact base; owner review required**

This manifest distinguishes the scientific oracle from implementation,
configuration, test, validation, and manager evidence.  Only `ORACLE` sources
may control scientific meaning.  `IMPLEMENTATION`, `CONFIG`, `TEST`, and
`VALIDATION` sources may establish only what the exact tree contains.
`MANAGER` sources govern process and sequencing, not science.

## Repository identity

| Field | Exact value |
| --- | --- |
| Base commit | `9f42d348298d76c5d5145aaf0c3eace1f3e154c1` |
| Base tree | `e51f22760c64454ce7233c45dd740aa710777bae` |
| Initial status | clean |
| Inspection mode | read-only source study |

## Admitted sources

<!-- BEGIN-ADMITTED-SOURCES -->
| ID | Class | Path | SHA-256 | Admitted role |
| --- | --- | --- | --- | --- |
| CTI-S001 | ORACLE | `doc/SCIENTIFIC_CONVENTIONS.md` | `c29ad515eb84aa2ee2d13b04245ebacf99ba8a790ea2ccb86ff568bcb84284d7` | repaired shared conventions only |
| CTI-S002 | ORACLE | `doc/scientific_contracts/audits/MAP_SPACE_HORIZONTAL_AUDIT_001/WORK_ORDER.md` | `b5cfdc0d2e9b72984b48bbe46e6d5750699828e47370e36996f72fc0b7196d4f` | accepted scope and owner disposition |
| CTI-S003 | ORACLE | `doc/scientific_contracts/audits/MAP_SPACE_HORIZONTAL_AUDIT_001/SOURCE_AUTHORITY_MANIFEST.md` | `d21d1446ebcdda8597cf08a4568be91906e3cc22e97f9e7f5544a5fa590b2cd5` | frozen-authority binding |
| CTI-S004 | ORACLE | `doc/scientific_contracts/audits/MAP_SPACE_HORIZONTAL_AUDIT_001/PRODUCT_AND_BOUNDARY_GRAPH.md` | `c5f256496c891925bb90c73a79e7af68e1d5966bc68f9c930b41a4690db7145c` | exact 17 products and 32 edges |
| CTI-S005 | ORACLE | `doc/scientific_contracts/audits/MAP_SPACE_HORIZONTAL_AUDIT_001/CROSS_PACKAGE_CONFORMANCE_MATRIX.md` | `106585cb838adc176c9faa5a89d19d0fed261ef136adfd2d98148271b38ce307` | horizontal dispositions |
| CTI-S006 | ORACLE | `doc/scientific_contracts/audits/MAP_SPACE_HORIZONTAL_AUDIT_001/FINDINGS_REPAIRS_AND_OWNER_DECISIONS.md` | `b8fad111974d79dcb48ee9977353d2e8958d74701a07ce7f507cd80962ff9310` | repaired finding lineage |
| CTI-S007 | ORACLE | `doc/scientific_contracts/audits/MAP_SPACE_HORIZONTAL_AUDIT_001/HORIZONTAL_AUDIT_REPORT.md` | `b32c0cf2249e74ef35b177cd016e5d03854e725466719f35fa62d9425270fe96` | exact 16 trace scenarios |
| CTI-S008 | ORACLE | `doc/scientific_contracts/audits/MAP_SPACE_HORIZONTAL_AUDIT_001/verify_horizontal_audit.py` | `bdd7013b61c254ae6bb8d2e6c900b8fd0764f6ee87280c014d37f50e7d33fc3a` | original packet integrity |
| CTI-S009 | ORACLE | `doc/scientific_contracts/audits/MAP_SPACE_SHARED_CONVENTIONS_REPAIR_001/CANDIDATE_REPORT.md` | `0e132d53fd379f2db585568a240caaa8c89d95a041dd40e226b1e9c9416217be` | exact bounded repair |
| CTI-S010 | ORACLE | `doc/scientific_contracts/audits/MAP_SPACE_SHARED_CONVENTIONS_REPAIR_001/SCIENTIFIC_OWNER_ACCEPTANCE_AND_INTEGRATION_2026-09-04.md` | `f8517a404c47757bcbd282164547822c5f9515011da312339d4945738e16390d` | owner acceptance and canonical integration |
| CTI-S011 | ORACLE | `doc/scientific_contracts/packages/SCI-MAP/v0.1/SOURCE_MANIFEST_R0.7.md` | `bd3f172f8bb1e17bf95fde034ad76632439c766ef89b0a992361b9e6d79ada0a` | SCI-MAP r0.7.1 set binding |
| CTI-S012 | ORACLE | `doc/scientific_contracts/packages/SCI-JINC/v0.1/FREEZE_AUTHORITY_MANIFEST_R0.3.md` | `ff4b79e7cca3950831eda95a16ec6a535597f543c4676378d2fc2f01d50faed2` | SCI-JINC r0.3 set binding |
| CTI-S013 | ORACLE | `doc/scientific_contracts/packages/SCI-FLT/v0.1/stage_b/AUTHORITY_MANIFEST.json` | `69e6766f26396ba843ee29cfb89a48efd91b7e1b517ed90d3d93c87a63e55778` | SCI-FLT-FIXED set binding |
| CTI-S014 | ORACLE | `doc/scientific_contracts/packages/SCI-FLT-MATCHED/v0.1/STAGE_B_DRAFT_MANIFEST.md` | `6b0231a7e9d34f028eda9cce48f62de1fc9e594348aa1448a2d182d732f78688` | frozen SCI-FLT-MATCHED set binding |
| CTI-S015 | ORACLE | `doc/scientific_contracts/packages/SCI-NOI/v0.1/stage_b/r0.5/PROPOSED_FREEZE_MANIFEST.json` | `b6915186424dd52d7c94fb0df47db91654d3c20cf4b3fa6ab98c3554626d8bfc` | frozen SCI-NOI set binding |
| CTI-S016 | ORACLE | `doc/scientific_contracts/packages/SCI-POINT/v0.1/STAGE_B_SOURCE_MANIFEST.json` | `76811b925834c7572b422aba3b23820b041307348bdde3da2fb1300263bf1828` | frozen SCI-POINT set binding |
| CTI-S017 | ORACLE | `doc/scientific_contracts/packages/SCI-VAL/v0.1/PROFILE_REGISTRY.md` | `95b2ddeca2039aa2b3614e3dfede530cb7e024e6db33ca22baed867f19d2709d` | exact registered and unavailable profiles |
| CTI-S018 | IMPLEMENTATION | `include/citlali/core/pipeline/timestream_native_science_projection.h` | `f222d32b590631ef1489b8e56aca7fa23b6c6cce1334639177b9143dcf1326ee` | processed-sample coordinate bridge |
| CTI-S019 | IMPLEMENTATION | `include/citlali/core/timestream/ptc/ptcproc.h` | `dab67354e1cb2ae40b3fd7e0dc619f768935adb5d27544d370e2761487d31396` | legacy coefficient production |
| CTI-S020 | IMPLEMENTATION | `include/citlali/core/mapmaking/naive_mm.h` | `f23a1b534690e80ee2ab859c5f4e021bb02ab84eab75c325e539551b639fa74d` | ordinary MAP ingress/accumulation/noise |
| CTI-S021 | IMPLEMENTATION | `include/citlali/core/mapmaking/science_map_contract.h` | `90afef645fafdd8077e564b007d913396c1cb33a72a65232c8ea9f43067f035c` | predecessor MAP contract types |
| CTI-S022 | IMPLEMENTATION | `include/citlali/core/mapmaking/map.h` | `f5ef1e05f48faef041cc797709c97fc1acd0cb47be1f17a3323ff534fc96525b` | shared map-buffer product storage |
| CTI-S023 | IMPLEMENTATION | `include/citlali/core/pipeline/observation_coadd_accumulation.h` | `c4f351b72800e25ca543dd6a15ddbe6b9679cec5a362b5e696e8d0e1d5a0cb29` | ordinary coadd admission/arithmetic |
| CTI-S024 | IMPLEMENTATION | `include/citlali/core/mapmaking/jinc_contract.h` | `19ad23ada0def48f0fa68c106c0108665e90ea286c5385b70910569db118e158` | predecessor JINC contract/types |
| CTI-S025 | IMPLEMENTATION | `include/citlali/core/mapmaking/jinc_mm.h` | `d621bda7f6f5070cfef522e5e5f50343fc7d68a8d4a2a83d1be966ff12dbabf6` | JINC accumulation |
| CTI-S026 | IMPLEMENTATION | `src/citlali/core/mapmaking/map.cpp` | `23fcb1e318d95cd0780e102c3d381003ca326b8b6e7f6bf20e7384ff4cc2d2a6` | MAP/JINC finalization and NOI products |
| CTI-S027 | CONFIG | `include/citlali/core/config/post_processing_config.h` | `efc2ab1321069acda3438249a2da3fe0fffbe8e93641c7c66ae5da910e3fe73f` | legacy filter/template/source controls |
| CTI-S028 | IMPLEMENTATION | `include/citlali/core/pipeline/map_filter_config_policy.h` | `9018877074d42182249864b2bca192b33af5c61af330921c4ee61459536cb143` | one-way legacy filter adaptation |
| CTI-S029 | IMPLEMENTATION | `include/citlali/core/mapmaking/wiener_filter.h` | `ffab049f189561197fc35425a925254208c0fc3796148c9bd4a79a39a3701a5f` | convolution/Wiener implementation |
| CTI-S030 | IMPLEMENTATION | `include/citlali/core/pipeline/filtered_observation_outputs.h` | `07e1aca69dc2777b7f055e8284d7028963b4e7f6b123de54d4c58eb94673acb3` | filtered observation lifecycle |
| CTI-S031 | CONFIG | `include/citlali/core/config/noise_config.h` | `4cc7faa6f0c83556c222780495d346ab94321851f6d98ab2c2405ab4700e9795` | legacy NOI controls |
| CTI-S032 | IMPLEMENTATION | `include/citlali/core/pipeline/noise_execution_plan.h` | `66ead2c026171a275ca394e15c6ccd720f3daf136803f299280c078100f70bb6` | predecessor NOI identities/lifecycle |
| CTI-S033 | IMPLEMENTATION | `include/citlali/core/pipeline/native_noise_assignment.h` | `5c3235a4b1dcca79e784b69b14097c763e8b5992d99525de31ad9eb2deba6b5a` | sign-assignment summary |
| CTI-S034 | IMPLEMENTATION | `include/citlali/core/pipeline/noise_weight_policy.h` | `e8f43607bc0dcdb28d8c05f9cd3249e2c5902de833c7263ee58cd763c7bc7226` | empirical coefficient mutation policy |
| CTI-S035 | IMPLEMENTATION | `include/citlali/core/pipeline/pointing_execution_plan.h` | `96175b06def2726632325300d4341cfab14f3ee09bb9cc3a4e057b74bac09b74` | pointing fit lifecycle/census |
| CTI-S036 | IMPLEMENTATION | `include/citlali/core/engine/detail/pointing_fit_maps_impl.h` | `236b6718ee05dc8875a00a62f344a0b3636bafd7d87c5ff9a2fd419f05be00d1` | direct per-array Gaussian fits |
| CTI-S037 | IMPLEMENTATION | `include/citlali/core/engine/detail/pointing_output_impl.h` | `291613c4c0a0fafeb7e48f961c4d5c6d473ddfde88817825817c38c041199256` | legacy pointing output schema |
| CTI-S038 | IMPLEMENTATION | `include/citlali/core/engine/detail/source_finding_execution_impl.h` | `8ee8066832e6afe65889740357b2e8b0fbda150df1fa016ca8c066f52f8775ad` | separate detection/catalog path |
| CTI-S039 | TEST | `tests/test_science_map_contract.cpp` | `4e8e84a2210f5715434e06c1f1473c23098aaa6651ca3821d4dd8ff5ca1b0f52` | predecessor MAP unit inventory |
| CTI-S040 | TEST | `tests/test_science_map_truth_suite.cpp` | `1f61839877ffb287cdbe9f293997ba4f703445446532349e13a959e48cc9db26` | predecessor MAP truth fixtures |
| CTI-S041 | TEST | `tests/test_jinc_map_contract.cpp` | `699000f41aa041c17d87afc2ccaedc1b3a7932aaec4effeab07d61b7e9e5242c` | predecessor JINC unit inventory |
| CTI-S042 | TEST | `tests/test_sci_align_native_science_projection.cpp` | `7dbe028054a926296473824fab357472db4d08795d55bb7cba5b5ea015e3c504` | processed-coordinate bridge tests |
| CTI-S043 | TEST | `tests/test_utils.cpp` | `09382e85f2c9d9065c842e488e0013a71a22bedc38bf75021ca7ba7d92be46eb` | legacy filter unit inventory |
| CTI-S044 | TEST | `tests/test_config_scaffold.cpp` | `7e99ed39152c2166b09de836cbed2f77d5e927d63416a22ef311bec31efd10a5` | config/lifecycle unit inventory |
| CTI-S045 | TEST | `tests/test_map_fitter_lifecycle.cpp` | `7dc51b306930197ba5a811de98305e6c9ab63abac2e9878b9ca15d6eb20c1d26` | Gaussian fitter lifecycle tests |
| CTI-S046 | TEST | `tests/test_pointing_fit_table_metrics.cpp` | `47f912ad1fc0775ca43b8b859c929df5e888803081b530d36e3acfbd26f7b132` | pointing diagnostic semantics tests |
| CTI-S047 | VALIDATION | `validation/product_contracts.json` | `34d5f6cf2be99c206f136ebb98d80932fd7087bf5e45f5209b711aad0f5ca565` | checked-in predecessor product registry |
| CTI-S048 | VALIDATION | `validation/accepted_runs.json` | `4a134dcdd14e0444d96875547f628a3353574cc66574dd9a559bcf59dafb94bb` | accepted-run inventory; no frozen-oracle conformance inference |
<!-- END-ADMITTED-SOURCES -->

## Exclusions

No active FRUIT ref, worktree, implementation source, validation payload, or
untracked artifact was inspected.  No historical ALIGN worktree was
inspected.  OOF is represented only by a future attachment envelope.  The
source inventory is deliberately bounded to files necessary to classify the
accepted graph; it is not a whole-repository correctness review.

## Verification limitation

There is no configured local `build/` directory at the exact base, so CTest
was unavailable without configuring or installing dependencies.  This is
recorded as a limitation rather than repaired.  Checked-in tests and
validation records are inventories only; no runtime-conformance conclusion
depends on them.
