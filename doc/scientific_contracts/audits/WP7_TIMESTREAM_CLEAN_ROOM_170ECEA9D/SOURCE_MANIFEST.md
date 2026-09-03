# WP-7 Clean-Room Successor Source Manifest

Prepared: `2026-08-25`

Status: launch packet bound to one immutable successor source commit;
independent successor audit not yet run

Source commit:
`170ecea9de1ee810da7d7e45a489a4545ccd623d`

This manifest does not hash itself. It separates readable scientific content
from integrity-only administrative bindings. The fresh auditor receives the
readable objects listed in `READABLE_SOURCE_ALLOWLIST.md`; the integrity tree
and nested manifests below prove which frozen, corrected, composition, and
repair-authority generations were used to assemble them.

## Frozen And Corrected Package Bindings

| Integrity object | Authority role | SHA-256 |
| --- | --- | --- |
| `doc/scientific_contracts/packages/SCI-ALIGN/v0.1/README.md` | Frozen SCI-ALIGN v0.1/r0.3 status | `be51b2347f04237ed5ae5773efb6978405f76666b3a92647721a482d25f7f9e0` |
| `doc/scientific_contracts/packages/SCI-ALIGN/v0.1/SOURCE_MANIFEST.md` | Exact SCI-ALIGN source/PDF generation | `26285329635c722cb9161d383ad1b95f56a03b782c101bcd89d8785a3575faac` |
| `doc/scientific_contracts/packages/SCI-AST/v0.1/README.md` | Frozen SCI-AST v0.1/r0.3 status | `f722589fb39df1d75c12c6f5a99797ee9bd1f304088edada8cf4788311b8b257` |
| `doc/scientific_contracts/packages/SCI-AST/v0.1/SOURCE_MANIFEST.md` | Exact SCI-AST source/PDF generation | `b54b6013750540f28aad02339a60bf36078980dc53b132beab73069d66ef3601` |
| `doc/scientific_contracts/packages/SCI-RTC/v0.1/SOURCE_MANIFEST_CORRECTED_2026-08-25.md` | Frozen SCI-RTC v0.1/r0.12 plus approved explanatory correction | `a5c06bd46cd8514e67ea77a7a728e3decb8c415cf486c4ec121927212bf22994` |
| `doc/scientific_contracts/packages/SCI-CAL/v0.1/SCIENTIFIC_OWNER_FREEZE_R0.5.md` | Frozen SCI-CAL v0.1/r0.5-r0.4 generation | `413426f49edf1249f751a05bb8c6e9fd907b11e8da0530fe2da39814885efb22` |
| `doc/scientific_contracts/packages/SCI-PTC/v0.1/SCIENTIFIC_OWNER_FREEZE_R0.5.md` | Frozen SCI-PTC v0.1/r0.5 generation | `8357961a49272adc40e27a8aa9e760e0d01ff2419ae2c88a62c0f93c9f959e66` |
| `doc/scientific_contracts/packages/SCI-VAL/v0.1/SCIENTIFIC_OWNER_FREEZE_R0.3.md` | Frozen SCI-VAL v0.1/r0.3 Core and continuing-registry generation | `2fc3b3ad329fe3035d442b43d1e564a74fc86ab49f85f56e87322d8553fad9a6` |

## Composition Bindings

| Integrity object | Authority role | SHA-256 |
| --- | --- | --- |
| `doc/scientific_contracts/boundaries/v0.1/SOURCE_MANIFEST.md` | Exact coordinate, geometry, and exposure boundary generation | `ce813e0adab8270daf713b30db8a271185227048fb79a71abe4b9e4a6ae2ab4a` |
| `doc/scientific_contracts/producer_interfaces/v0.1/SOURCE_MANIFEST.md` | Exact approved native readout interface and precedence generation | `a417fb3d22aa46ad7d7f1134b6d804b9d3c3f5a7f601dbb53c19f10a23e72912` |
| `doc/scientific_contracts/audits/SIX_PACKAGE_TIMESTREAM_CLOSURE_PROGRAM_55EFD8A/WP3_SOURCE_MANIFEST.md` | Reference-first processed-timestream binding generation | `d407228bfbbdbe8be994e7e84e4945fc6868365c2d045c18ac7ce1e5c40ae9aa` |
| `doc/scientific_contracts/audits/SIX_PACKAGE_TIMESTREAM_CLOSURE_PROGRAM_55EFD8A/WP4_SOURCE_HYGIENE_MANIFEST.md` | Frozen-notation parity generation | `57dacf3a5847a24a85b754e878306bd5efb088f571c354f650d0961bdd3ca9a0` |
| `doc/scientific_contracts/audits/SIX_PACKAGE_TIMESTREAM_CLOSURE_PROGRAM_55EFD8A/WP5_SOURCE_MANIFEST.md` | Frozen VAL source/profile binding generation | `365de9715c7b0fb3ef7390a07caf53a8b7c89d1bb6939f2fad36db0a816261cd` |

## Approved Repair-Authority Binding

| Integrity object | Authority role | SHA-256 |
| --- | --- | --- |
| `doc/scientific_contracts/audits/WP7_TIMESTREAM_CLEAN_ROOM_F01E22F5F/REPAIR_AND_CLOSURE/WP7_REPAIR_AUTHORITY_MANIFEST_2026-08-25.md` | Exact D001--D004 owner authority, native-interface set, CAL numerical bytes, and sanitized readable addendum | `f5f9f903e52a979339c04cd741686d267c739e5530e364ab49e3bb612b37b26f` |

The repair manifest binds the exact owner disposition and decision packet as
integrity-only authority. Independent extraction reads only the sanitized
approved authority addendum, exact native-interface approval set, and exact
numerical objects listed by the readable-source allowlist.

## Sanitized View Parent Bindings

| Parent object | Retained semantic role | SHA-256 |
| --- | --- | --- |
| `doc/scientific_contracts/audits/SIX_PACKAGE_TIMESTREAM_CLOSURE_PROGRAM_55EFD8A/WP3_REFERENCE_BINDING_REGISTER.md` | Reference-first handoff and failure semantics | `96338b56aa57211d2e59664f6eeb9514d0846d51aca94bcb9cbf11d076402efe` |
| `doc/scientific_contracts/audits/SIX_PACKAGE_TIMESTREAM_CLOSURE_PROGRAM_55EFD8A/WP4_RTC_NOTATION_SEMANTIC_MAPPING.md` | RTC role-sensitive notation parity | `c5466abe6386715c52edbf1632972b2ec5c1f39ae684612a5b47813a1d919a09` |
| `doc/scientific_contracts/audits/SIX_PACKAGE_TIMESTREAM_CLOSURE_PROGRAM_55EFD8A/WP4_AST_NOTATION_SEMANTIC_MAPPING.md` | AST compact semantic crosswalk | `514e05c2b4e3d7053ee23e9e1a11f269252d97c63c8be69354e330b0a1310636` |
| `doc/scientific_contracts/audits/SIX_PACKAGE_TIMESTREAM_CLOSURE_PROGRAM_55EFD8A/PTC_NAMED_USE_COMMON_SEMANTICS_R0.1.md` | PTC named-use common semantics | `c1fc8370007b65307769fb966c8523251695924aaff84f3e5b4c89b6d3380b8c` |

The first three parent objects above remain integrity-only because their
administrative sections expose prior repair history. Their approved scientific
content appears without those labels in `SANITIZED_COMPOSITION_NOTES.md`. The
PTC named-use artifact contains no prior finding labels and is admitted
directly.

## Packet File Hashes

| Packet file | SHA-256 |
| --- | --- |
| `AUDITOR_START_HERE.md` | `06a5756c88c091acea482236826b3d8b2427229f02a0af529ea63e7e84e0d3d6` |
| `AUDIT_THREAD_LAUNCH_PROMPT.md` | `eb983850815aa5536a8e70f3700f7fa6dcff715c2d1df3918220b371a35a22d4` |
| `CLEAN_ROOM_CHARTER.md` | `a732f340640b00ea319fa355ee1ddd030488061901a3a4ef2f2c677f0581e6b8` |
| `READABLE_SOURCE_ALLOWLIST.md` | `d44d632a293d302867d812c8ba65ed21fe58789e34db1231fc27aedb166ea630` |
| `SANITIZED_COMPOSITION_NOTES.md` | `29fb24f7657f0e67fa9d8b51680867e73f90b41754b954e3b0920f964f299298` |
| `verify_packet.py` | `faed1a4db49f41c498abb4831e3cea4111484363719831e400234569443241d1` |
| `build_handoff.py` | `dfd14c1f31e94b98353b493767c191c0287389cb280171e56d3037b03aa84d91` |

## Handoff Rule

The deterministic archive contains the seven packet files above, this
manifest, an extracted `sources/` tree containing exactly the readable source
objects, a separate `bindings/` tree containing the eighteen integrity objects,
and checksum lists for both trees. The archive digest is recorded outside the
archive in `HANDOFF_ARCHIVE.md`, avoiding a self-hash cycle.

No source beyond the allowlist is readable during independent extraction.
Missing authority remains missing; the archive must not be supplemented from a
repository checkout.
