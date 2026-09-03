# FRUIT EL-F7 registration manifest r0.1

Test ID: `SCI-FRUIT-EL-F7-SHARED-START-RESPONSE-DECOMPOSITION-R0.1`

Status: **frozen before external staging or execution**

| Object | Bytes | SHA-256 |
|---|---:|---|
| `doc/scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane/SCIENTIFIC_OWNER_EL_F7_AUTHORIZATION_2026-09-03.md` | 873 | `05ffeff2e6dc19da501b727380157012c47720e65673e6ec69953cca49748331` |
| `TEST_DEFINITION.md` | 3755 | `156b020a01e96f0b82e95a8bcee56310d07524a87e0452ac96ca06b8dadbdaf3` |
| `REGISTRATION_R0.1.yaml` | 3597 | `57a63f41e45766851b3f5fe1f9a261ee836486b0662833167612c8e369c558ff` |
| `ANALYSIS_MANIFEST_R0.1.yaml` | 2398 | `60f8e6dd57b87636a106b170a150fcdad4b1c255b586f0c75e1b6ab29ac38f4d` |
| `EL_F7_CONTROL_SHAM.yaml` | 557 | `681b615757c4aa06ce929ae953a6f74c176f8bccba55438381559e1c2e5a7138` |
| `EL_F7_SHARED_START_PROBE.yaml` | 568 | `4caa01dd1ec14da8c8a1c84dd97500f1400578b4f500b236bd8bf9ae44d5942f` |
| `tools/fruit_loops/analyze_shared_start_response.py` | 40313 | `ce3ca607fe4f5a3540f6d6193194377c8c0be39e035a8557d68ab86a3a4442c7` |
| `tools/fruit_loops/test_analyze_shared_start_response.py` | 2843 | `8c13d87703bd6b6d68f0f7572a0bb42632e41bf0027a93bd886fcc17dd566bb1` |

The exact approved owner-review bundle remains
`EL_F7_BUNDLE_MANIFEST_R0.1.md` at preparation commit `c24c81304`. The
registered control iteration-4 checkpoint remains pinned to SHA-256
`a77505ab0637c1f257016ee0d9e801b3bba17ed52ab88d52f417a5c1513b451f`,
and the executable remains pinned to SHA-256
`6431c6653ed46ff6e1dfa5512cd27e8169525f7a110207b0b24505786f39dbbe`.

Before freeze, all 228 baseline and FRUIT-loop Python tests passed. The two
new files also passed Ruff and byte-compilation checks, all registered YAML
parsed, and the repository whitespace check passed. No EL-F7 external root or
output existed when this manifest was created.
