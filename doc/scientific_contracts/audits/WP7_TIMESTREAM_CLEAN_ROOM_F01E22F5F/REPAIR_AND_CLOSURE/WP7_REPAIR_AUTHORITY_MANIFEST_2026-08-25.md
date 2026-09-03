# WP-7 Repair Authority Manifest

Status: **exact authority-publication binding for the bounded clean-room
successor; this manifest does not hash itself**

Prepared: `2026-08-25`

## 1. Owner authority

| Object | Role | SHA-256 |
| --- | --- | --- |
| `WP7_SCIENTIFIC_OWNER_DECISION_PACKET.md` | Exact approved recommendations and recorded owner responses | `916ff4df02af90af1ccf745b8ba5de4065a8ba1ed53fa02bf0197f3c562d6d07` |
| `WP7_SCIENTIFIC_OWNER_DISPOSITION_2026-08-25.md` | Final D001--D004 authority and precedence | `cee7f2445dd0bbad9b1925e82e6d9f757ed158237a481fbccee1e83263e72833` |
| `WP7_APPROVED_SCIENTIFIC_AUTHORITY_ADDENDUM_2026-08-25.md` | Sanitized readable view of the approved scientific content | `7a0a92a411f4d93f321257fba5cdbc561a4249f5d49cc49b8fe974f87e77d577` |
| `verify_repair_authority.py` | Standalone publication-set verifier | `233895f2c89f40cf46b148dd6cb067ba143ddf045739695917b9912a86c3e965` |

The disposition governs. The sanitized addendum introduces no independent
authority; it is admitted so a fresh auditor can read the approved content
without consulting repair history. If its content differs from the exact
approved decision packet, the decision packet bound above governs and the
successor packet fails preparation.

## 2. Native paired-readout authority set

All five objects below shall be readable together. The source manifest and
approval record promote the retained interface bytes and supersede only their
embedded pre-promotion status and candidate-only wording.

| Repository path | Role | SHA-256 |
| --- | --- | --- |
| `doc/scientific_contracts/producer_interfaces/v0.1/README.md` | Approved interface-set entry point | `b0946488a6d903c5423eccb0b72c7061242c3a075627b7355f9e3a97fa19435b` |
| `doc/scientific_contracts/producer_interfaces/v0.1/SOURCE_MANIFEST.md` | Exact source and precedence binding | `a417fb3d22aa46ad7d7f1134b6d804b9d3c3f5a7f601dbb53c19f10a23e72912` |
| `doc/scientific_contracts/producer_interfaces/v0.1/WP2_FOLLOWUP_D011_OWNER_DECISION_2026-08-23.md` | Scientific-owner decision record | `9826d144187e65b838e6b97cf9c08708bfa08808b5b608757cb163ad23b69c28` |
| `doc/scientific_contracts/producer_interfaces/v0.1/SCIENTIFIC_OWNER_APPROVAL_2026-08-24.md` | Exact-byte promotion and precedence | `4f14ed83f1d1625553d95ad259c8fb0b0f8628d5ef59bb04851b7e5763899da8` |
| `doc/scientific_contracts/producer_interfaces/v0.1/TUNE_READOUT_NATIVE_XR_PRODUCER_INTERFACE.md` | Approved native paired-`x/r` interface bytes | `f9659b34a49a07d4287c4a70db798cdd2ec30049531da603fcca1e9d1fdd5969` |

## 3. Recovered CAL numerical authority

The staging root is
`RECOVERED_CAL_NUMERICAL_AUTHORITY_2026-08-25/`. The recovery manifest records
source commits, object identities, schemas, and redistribution notices. The
checksum list binds the exact admitted inventory, and the verifier checks both
the individual objects and canonical passband-set aggregation.

| Relative path under the staging root | Role | SHA-256 |
| --- | --- | --- |
| `RECOVERY_MANIFEST.md` | Recovery provenance and admission boundary | `41ab2917fa490c63f64182f2d43c631b517624356871fd4f66bd17a4a045511a` |
| `SOURCE_OBJECT_SHA256SUMS.txt` | Exact staged-source inventory | `0c113c1592f26a1c1cee4b0cb21aa63830b3304850ae1036ef9f0ea9f4033e57` |
| `verify_recovery.py` | Standalone exact-byte and aggregation verifier | `5b01cddcc023aaeba601cdbce54ed23ef28dac83e6c85925cf70d57cc1e5e2c8` |
| `sources/citlali/licenses/LICENSE` | Citlali source notice | `8f46574eb73aa5ca78636c21f83a5cc2bbdf32793a6f563d6463b4103ca2df9b` |
| `sources/citlali/validation/sci_cal_001_atmosphere_operator_2026-08-01/SCI_CAL_001_FIXED_DJF25_FULL_DOMAIN_OWNER_DECISION.md` | Original atmosphere-operator owner decision | `c43aa932c633e336497547730f73278d3a5cf70d2a5fcfb19049d967c79dd469` |
| `sources/citlali/validation/sci_cal_001_atmosphere_operator_2026-08-01/sci_cal_001_fixed_djf25_full_domain_operator_contract.json` | Atmosphere machine contract | `7a064ff768a3de4f427f1338d94ef6cb9026d248f3c3c816fc3dfc96d156e36a` |
| `sources/citlali/validation/sci_cal_001_atmosphere_operator_2026-08-01/sci_cal_001_fixed_djf25_full_domain_operator_nodes.csv` | Atmosphere node table | `fd688a4cd3f46585b08631bc63a562aed482feb9b24ec9ee0071b70db7eb8a5f` |
| `sources/tolteca/licenses/LICENSE.rst` | TolTECA source notice | `9959c063acc1cb1030dc86d7efbd6591d9a759c7787721dee6320ebd4d20a41e` |
| `sources/tolteca/tolteca/data/cal/toltec_passband/index.yaml` | Passband-set index | `74465637294e536c44818099e4858a916fc6b9acbb1ea21b40427d15fb6532d5` |
| `sources/tolteca/tolteca/data/cal/toltec_passband/data/a1100_passband.ecsv` | 1.1-mm passband | `13b8fd009bb8d7c375d3c46d21e26d0a779f7f00a949a2a5ccd619d1fe56fd72` |
| `sources/tolteca/tolteca/data/cal/toltec_passband/data/a1400_passband.ecsv` | 1.4-mm passband | `a7b671d9f659cbc98dad99d3015ce81a3d7a3486c702819d9b3305703e7c682e` |
| `sources/tolteca/tolteca/data/cal/toltec_passband/data/a2000_passband.ecsv` | 2.0-mm passband | `77e4b33c7bbc2c345ef94d41480d5fee5cb096d789f4fe78e1b4f80a37e0d6ff` |

The canonical passband-set identity is
`5e6f38f14bcae93a29ffe8362c52b15209f51aee4e48373b23aaa5ec2f8a6433`
over `1,297,803` member bytes.

## 4. Admission and non-substitution rule

The successor packet shall admit the exact objects above without byte changes.
It shall not regenerate the atmosphere table, substitute another passband
object, rewrite the approved native interface, infer an implementation default,
or treat a readable sanitized view as stronger than its governing disposition.
Any missing object, hash mismatch, inventory difference, or authority-content
disagreement fails packet construction.
