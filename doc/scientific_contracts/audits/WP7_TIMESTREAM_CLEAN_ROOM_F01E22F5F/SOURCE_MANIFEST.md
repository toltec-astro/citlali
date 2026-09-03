# WP-7 Clean-Room Source Manifest

Prepared: `2026-08-24`

Status: launch packet bound to one immutable source commit; independent audit
not yet run

Source commit:
`f01e22f5f8d8d92e49ae70312bdc59a81c1540ec`

This manifest does not hash itself. It separates readable scientific content
from integrity-only administrative bindings. The fresh auditor receives the
readable objects listed in `READABLE_SOURCE_ALLOWLIST.md`; the coordinator's
nested manifests below prove which frozen package and composition generations
were used to assemble them.

## Frozen Package Bindings

| Integrity object | Authority role | SHA-256 |
| --- | --- | --- |
| `doc/scientific_contracts/packages/SCI-ALIGN/v0.1/README.md` | Frozen SCI-ALIGN v0.1/r0.3 status | `be51b2347f04237ed5ae5773efb6978405f76666b3a92647721a482d25f7f9e0` |
| `doc/scientific_contracts/packages/SCI-ALIGN/v0.1/SOURCE_MANIFEST.md` | Exact SCI-ALIGN source/PDF generation | `26285329635c722cb9161d383ad1b95f56a03b782c101bcd89d8785a3575faac` |
| `doc/scientific_contracts/packages/SCI-AST/v0.1/README.md` | Frozen SCI-AST v0.1/r0.3 status | `f722589fb39df1d75c12c6f5a99797ee9bd1f304088edada8cf4788311b8b257` |
| `doc/scientific_contracts/packages/SCI-AST/v0.1/SOURCE_MANIFEST.md` | Exact SCI-AST source/PDF generation | `b54b6013750540f28aad02339a60bf36078980dc53b132beab73069d66ef3601` |
| `doc/scientific_contracts/packages/SCI-RTC/v0.1/SCIENTIFIC_OWNER_FREEZE_R0.12.md` | Frozen SCI-RTC v0.1/r0.12 generation | `0cac4396df225c1f2808ee1055e063c9a4e72a02549557c5e997f54d72dac0bf` |
| `doc/scientific_contracts/packages/SCI-CAL/v0.1/SCIENTIFIC_OWNER_FREEZE_R0.5.md` | Frozen SCI-CAL v0.1/r0.5-r0.4 generation | `413426f49edf1249f751a05bb8c6e9fd907b11e8da0530fe2da39814885efb22` |
| `doc/scientific_contracts/packages/SCI-PTC/v0.1/SCIENTIFIC_OWNER_FREEZE_R0.5.md` | Frozen SCI-PTC v0.1/r0.5 generation | `8357961a49272adc40e27a8aa9e760e0d01ff2419ae2c88a62c0f93c9f959e66` |
| `doc/scientific_contracts/packages/SCI-VAL/v0.1/SCIENTIFIC_OWNER_FREEZE_R0.3.md` | Frozen SCI-VAL v0.1/r0.3 Core and continuing-registry generation | `2fc3b3ad329fe3035d442b43d1e564a74fc86ab49f85f56e87322d8553fad9a6` |

## Composition Bindings

| Integrity object | Authority role | SHA-256 |
| --- | --- | --- |
| `doc/scientific_contracts/boundaries/v0.1/SOURCE_MANIFEST.md` | Exact coordinate, geometry, and exposure boundary generation | `ce813e0adab8270daf713b30db8a271185227048fb79a71abe4b9e4a6ae2ab4a` |
| `doc/scientific_contracts/producer_interfaces/v0.1/SOURCE_MANIFEST.md` | Exact native readout interface generation | `a417fb3d22aa46ad7d7f1134b6d804b9d3c3f5a7f601dbb53c19f10a23e72912` |
| `doc/scientific_contracts/audits/SIX_PACKAGE_TIMESTREAM_CLOSURE_PROGRAM_55EFD8A/WP3_SOURCE_MANIFEST.md` | Reference-first processed-timestream binding generation | `d407228bfbbdbe8be994e7e84e4945fc6868365c2d045c18ac7ce1e5c40ae9aa` |
| `doc/scientific_contracts/audits/SIX_PACKAGE_TIMESTREAM_CLOSURE_PROGRAM_55EFD8A/WP4_SOURCE_HYGIENE_MANIFEST.md` | Frozen-notation parity generation | `57dacf3a5847a24a85b754e878306bd5efb088f571c354f650d0961bdd3ca9a0` |
| `doc/scientific_contracts/audits/SIX_PACKAGE_TIMESTREAM_CLOSURE_PROGRAM_55EFD8A/WP5_SOURCE_MANIFEST.md` | Frozen VAL source/profile binding generation | `365de9715c7b0fb3ef7390a07caf53a8b7c89d1bb6939f2fad36db0a816261cd` |

## Sanitized View Parent Bindings

| Parent object | Retained semantic role | SHA-256 |
| --- | --- | --- |
| `doc/scientific_contracts/audits/SIX_PACKAGE_TIMESTREAM_CLOSURE_PROGRAM_55EFD8A/WP3_REFERENCE_BINDING_REGISTER.md` | Reference-first handoff and failure semantics | `96338b56aa57211d2e59664f6eeb9514d0846d51aca94bcb9cbf11d076402efe` |
| `doc/scientific_contracts/audits/SIX_PACKAGE_TIMESTREAM_CLOSURE_PROGRAM_55EFD8A/WP4_RTC_NOTATION_SEMANTIC_MAPPING.md` | RTC role-sensitive notation parity | `c5466abe6386715c52edbf1632972b2ec5c1f39ae684612a5b47813a1d919a09` |
| `doc/scientific_contracts/audits/SIX_PACKAGE_TIMESTREAM_CLOSURE_PROGRAM_55EFD8A/WP4_AST_NOTATION_SEMANTIC_MAPPING.md` | AST compact semantic crosswalk | `514e05c2b4e3d7053ee23e9e1a11f269252d97c63c8be69354e330b0a1310636` |
| `doc/scientific_contracts/audits/SIX_PACKAGE_TIMESTREAM_CLOSURE_PROGRAM_55EFD8A/PTC_NAMED_USE_COMMON_SEMANTICS_R0.1.md` | PTC named-use common semantics | `c1fc8370007b65307769fb966c8523251695924aaff84f3e5b4c89b6d3380b8c` |

The first three parent objects above are not readable during independent
extraction because their administrative sections expose repair history. Their
approved scientific content is reproduced without those labels in
`SANITIZED_COMPOSITION_NOTES.md`. The PTC named-use artifact contains no prior
finding labels and is admitted directly.

## Packet File Hashes

| Packet file | SHA-256 |
| --- | --- |
| `AUDITOR_START_HERE.md` | `b0956b8706c3308c9f693513520066354bfd4eb1803ca1dceb8ff60230b3d854` |
| `AUDIT_THREAD_LAUNCH_PROMPT.md` | `fc7249fa4d2d92c3c95a1965ae8c4dd43a935fdc6ca48eed55a6f40023972868` |
| `CLEAN_ROOM_CHARTER.md` | `7d040f96613b193c5ff14108422ba2dd6160f0d6b7b87056961c16de31cce2f0` |
| `READABLE_SOURCE_ALLOWLIST.md` | `fedaa674a5e343214214ec5d97acc7325d70a66ed90bc1d350ec2552c9082957` |
| `SANITIZED_COMPOSITION_NOTES.md` | `9f9ed24fedb20bf4f28bfe51de849d00f7cf1e6dcf4e8fe564ade65ef19f35c4` |
| `verify_packet.py` | `ddd21821c8d0125c1e3442317a0757823f8d5307ec1c9fd226a8a9691c524336` |
| `build_handoff.py` | `ea1d2966bf278e59e875554c5307fb4467566f6c2eeb25e41e9c0a3984ddc0b8` |

## Handoff Rule

The deterministic archive contains the seven packet files above, this manifest,
an extracted `sources/` tree containing exactly the readable source objects,
and `SOURCE_OBJECT_SHA256SUMS.txt`. The archive digest is recorded outside the
archive in `HANDOFF_ARCHIVE.md`, avoiding a self-hash cycle.

No source beyond the allowlist is admitted. Missing authority remains missing;
the archive must not be supplemented from a repository checkout during the
independent phase.
