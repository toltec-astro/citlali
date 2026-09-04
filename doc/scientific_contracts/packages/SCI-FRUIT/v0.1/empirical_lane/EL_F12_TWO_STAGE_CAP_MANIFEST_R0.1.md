# SCI-FRUIT EL-F12 — Two-stage cap review manifest r0.1

Date: `2026-09-04`

Decision: `SCI-FRUIT-EL-F12-CAP-001-R0.1`

Status: **exact proposed amendment; owner decision pending**

The owner has already approved Choice A against proposal commit
`d78d4d94a3ab67746c46e7519fc5e32ddbcb1844` and its original manifest digest
`a587db8af538856ce5bd966727b20a55369a12f335105657f3bf3069dc6a021f`.
That packet remains unchanged. This manifest binds only the subsequent
two-stage cap amendment, authorization record and supporting evidence.
Source files are implementation references; their inclusion is not approval
of additional actions or an EL-F12 executable registration.

| Repository-relative file | Bytes | SHA-256 |
| --- | ---: | --- |
| `doc/scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane/EL_F12_TWO_STAGE_CAP_OWNER_REVIEW_R0.1.md` | 5736 | `e34ccc2334eee19b1fe2a816c0fc29cdd5e67a4db32c8d1e47873f9b3823c752` |
| `doc/scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane/SCIENTIFIC_OWNER_EL_F12_AUTHORIZATION_2026-09-04.md` | 1447 | `b567435eb39661269db061815839725afa484d4d3ea852b040e32022bf2d9fc0` |
| `doc/scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane/EL_F12_BUNDLE_MANIFEST_R0.1.md` | 7610 | `a587db8af538856ce5bd966727b20a55369a12f335105657f3bf3069dc6a021f` |
| `doc/scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane/EL_F12_RESPONSE_AWARE_INTERVENTION_DESIGN_R0.1.md` | 23036 | `7504bdfc8492bc0aac8249cb0c4862d01a582e3a0325205244d0eb6b1d0570ec` |
| `validation/fruit_loop_el_f12_preimplementation_2026-09-04/CAP_BOUNDARY_TEST_RESULT_R0.1.md` | 3862 | `69708063eb75d8836c108efe8ca12580103fd5376ae1c85879895ee160e8416d` |
| `validation/fruit_loop_el_f12_preimplementation_2026-09-04/INPUT_IDENTITY_PREFLIGHT_R0.1.json` | 25969 | `07e80afe91b1fe220e22699684b8e3e1cae4e7bb68b2c334434276c925a73802` |
| `validation/fruit_loop_el_f12_preimplementation_2026-09-04/cap_boundary_build_r0.1.log` | 896 | `259a77c2e2371173a183e23213e62e72acaf34bfbde097a2911be2569cc80a03` |
| `validation/fruit_loop_el_f12_preimplementation_2026-09-04/cap_boundary_ctest_r0.1.log` | 2527 | `a0bf0fbb4eec05994edb3b8f95255206391dc48aff26427270d40a51e2e71a59` |
| `tests/test_learning_target_application.cpp` | 13279 | `1e702600e2084e8131bc92ce1adb9b6964a2b40549821da78a361bdd662f779e` |
| `tests/CMakeLists.txt` | 5677 | `9f7d282a8239ce2e1e67d6f307455e63ac30feaaf7a8e52f36fc3ad50575be3f` |
| `include/citlali/core/engine/detail/learning_detector_exclusion_apply_impl.h` | 12461 | `c848a62be82bd5329c8fca3d5241d8cbc50831158ea31dbcdd4bc22798a7acb5` |
| `include/citlali/core/engine/detail/learning_detector_apply_impl.h` | 1825 | `a658f1ed9d1878445c2d8a0735925f53c5b782bb3ce31c1b5b64f4dae236f11c` |
| `include/citlali/core/pipeline/learning_detector_exclusion_stage.h` | 861 | `11aba40af84689ec6af9ae88714f2d6f69d1432d49875703f89ff72114567cf6` |
| `include/citlali/core/engine/detail/pointing_run_impl.h` | 7821 | `68fdea619ea94b127f11fd45934932f3fc3506f1a8594a09ad5e4489b8d7be2f` |

Approval of CAP-001 would supply the missing gate-composition rule and allow
the already authorized EL-F12 work to resume. It would not add a candidate,
change a numerical cutoff or extend any input, run, resource, qualification
or production boundary. Its exact bytes are bound by the containing commit;
there is no self-referential digest.
