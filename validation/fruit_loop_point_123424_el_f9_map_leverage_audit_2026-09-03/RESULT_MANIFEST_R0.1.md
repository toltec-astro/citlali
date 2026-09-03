# FRUIT EL-F9 result manifest r0.1

Test ID: `SCI-FRUIT-EL-F9-MAP-LEVERAGE-FLAGGING-AUDIT-R0.1`

Status: **valid completed read-only audit with registered availability stop**

## Direction, registration, and correction chain

| Object | Bytes | SHA-256 |
|---|---:|---|
| `SCIENTIFIC_OWNER_EL_F9_AUTHORIZATION_2026-09-03.md` | 831 | `8a71f98b31f8073f1209723f26f1ec17ba6081de984ed36524cdd933daf353f3` |
| `OWNER_DIRECTION_2026-09-03.md` | 639 | `2d2e65bd16aade2154802dcd2e136bd317cb16d8c8c3f1fc970ca58c42b70660` |
| `TEST_DEFINITION.md` | 5590 | `a23979463c8f47f8f4278f70d72b1c6af7cc8e1eef65d118014131afdd44cabf` |
| `REGISTRATION_R0.1.yaml` | 6926 | `0fea360923e4ae7ca10a1c623d62cf6dce7c46ffb27f6d362bf3beef70ff6b29` |
| `PRE_ANALYSIS_SEMANTICS_R0.2.md` | 1552 | `b9f8d4d22ee9f2f48a4eb9b03c877b7e7a848e3b121cc92d90941b22934e9f61` |
| `REGISTRATION_R0.2.yaml` | 1460 | `933d665ebd95ed06193cf07818bd5728217a9d846a7ebbfc2757bdc4adb18cf0` |
| `REGISTRATION_R0.3.yaml` | 2802 | `c59eb2a5ff6bf6583fefac8fc95adc023eb553d154d44862b0c106efbe32564f` |
| `AVAILABILITY_STOP_R0.3.md` | 3299 | `6cb57eb23e208371a28903e481ca7923932ebf459d61bf1749e5fb37d3be8249` |

The scientific-owner authorization file is in
`doc/scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane`. All other
objects in this manifest table are in this validation directory unless a
different path is shown.

`PRE_ANALYSIS_SEMANTICS_R0.2.md` is preserved because it was part of the
registered chronological record. Its generic additive interpretation is
explicitly superseded by `AVAILABILITY_STOP_R0.3.md` after the JINC source and
registered paired values demonstrated that the published coefficient is
nonlinear.

## Result products

| Object | Bytes | SHA-256 |
|---|---:|---|
| `AVAILABILITY_RESULT_R0.1.json` | 8433 | `6c60cd61871cfa2e6efcfeb66e3756aab5f6217ceb9174b8bb446dcad66e17f4` |
| `DETECTOR_METRICS_R0.1.csv` | 1272 | `0b9ee5d7a467fa110744d45c97b894a9962638fdeea2685491afba7942819f1f` |
| `TRIGGER_RESPONSE_R0.1.csv` | 775 | `2a76a6b201f79e2b63d627bc109452c844d21641d24643d715bccb8ed0276c49` |
| `MAP_RESPONSE_AND_COEFFICIENT_DIAGNOSTIC_R0.1.png` | 169051 | `9ca5bc42561e007ed6e7d787974ddb2408f5a1659b5938457fa9cd53928c2811` |
| `ANALYSIS_PROVENANCE_R0.1.yaml` | 9364 | `bcacbe9aef5a39a7dcd4861afe8991e72fe36d1a3fe9bef2f5d5d42e24fb5d71` |
| `EXECUTION_RESULT_R0.1.md` | 8774 | `05cf29f7c6722f02a1181d4b3787c575bcfd686fe6423f3dc0ec5100929d861f` |
| `tools/fruit_loops/audit_map_leverage_availability.py` | 32082 | `d6ad6018ab9f8178b700580cb772c41e1d1a8fbc2073838941882189b7318f88` |
| `tools/fruit_loops/test_audit_map_leverage_availability.py` | 5285 | `6c224a3d17157d60f6be683a57344f5ac69ad5394330fd063da8dcb2e86006d3` |

## Result and validity

- all 30 hash-registered evidence and implementation files passed size and
  SHA-256 validation before the analysis result was written;
- the paired products are both JINC maps with matching signal and formal-
  coefficient units, WCS/grid, shapes, and support conventions;
- the published formal-coefficient difference is materially negative in
  2,019 pixels and positive in 2,448, so it is not an additive UID weight;
- exact UID leverage, contribution contrast, hit count, and unique-detector
  count were recorded as unavailable rather than replaced with proxies;
- the A5-map/N5 scan-5 RMS, standard deviation, median, flagged fraction, and
  detector weight are identical for all 445 accepted a1400 detectors;
- the four iteration-4 trigger pixels and the one iteration-5 application row
  reproduce the registered UID, scan, factor, count, and sample accounting;
- the direct next-map response is exactly zero at all four trigger pixels;
- no external input was changed, no Citlali reduction was run, and no Unity
  activity occurred; and
- all 244 baseline and FRUIT-loop Python tests passed;
- the complete required configuration preflight passed, including its 127
  unit tests and all eight compact-compatibility fixtures;
- Ruff and byte compilation passed for the analysis tool and tests; and
- every result JSON/YAML file reopened, all 30 registered hashes and the
  analysis-script hash were rechecked, the final image was inspected, and the
  repository whitespace check passed.

The result does not judge UID 4460, establish a generic detector mechanism,
select a safeguard, change a production policy, qualify a recurrence, launch
Gate D or Stage B, or authorize diagnostic instrumentation or another replay.
