# WP-3 CAL And External-Producer Source Manifest

Prepared: `2026-08-24`

Status: WP-3 packet-control manifest; scientific-owner decisions and mechanical
reference-binding preparation complete; WP-7 clean-room re-audit pending

Scientific owner: Grant Wilson

## Bound Authority

| Authority | Bound identity |
| --- | --- |
| Consolidated clean-room audit baseline | `55efd8a54464636a24e621f6d1b60486d235b20e` |
| SCI-CAL | frozen v0.1 science-rationale r0.5 / engineering-conformance r0.4 at `a8b57587a02bba309677cda5267bf394167ec146` |
| SCI-PTC consumer boundary | frozen v0.1/r0.5 at `a18defe701bc824879a18cc6adafa6631fd22391` |
| WP-2 timestream boundaries | exact packet approved at `0b3cfb24070c1eda04dbda7633accf40e2e8b852` |
| Native Tune/readout interface | `WP2-FOLLOWUP-D011` and exact v0.1/r0.1 artifact approved `2026-08-24` |
| WP-3 owner decisions | `WP3-OWNER-D001--D008`, approved or approved with recorded owner correction through `57d221d253d00186ab37fa5ccb2552be9dbbda8c` |

The CAL freeze, upstream clarification, and WP-3 decision sequence are retained
in commits `a8b57587a`, `49ea86123`, `67d3f955f`, `7f4108a9f`, `537781568`,
`b3faeb13a`, `9e82081f3`, `de6868e87`, and `57d221d25`. These commit references
preserve review history; the artifact digests below bind the assembled current
packet.

## WP-3 Artifact Digests

This manifest does not hash itself.

| Artifact | SHA-256 |
| --- | --- |
| `WP3_CAL_EXTERNAL_PRODUCER_SCIENTIFIC_OWNER_DECISION_PACKET.md` | `26d1e75094a5cf465199f87f33332326e5d196d5c66926c8868b8928ce2fff4c` |
| `WP3_REFERENCE_BINDING_REGISTER.md` | `96338b56aa57211d2e59664f6eeb9514d0846d51aca94bcb9cbf11d076402efe` |

## Referenced Authority Digests

These entries bind the exact repository records referenced by the WP-3
register. They do not copy the scientific contents of those records.

| Referenced artifact | SHA-256 |
| --- | --- |
| `packages/SCI-CAL/v0.1/SCIENTIFIC_OWNER_FREEZE_R0.5.md` | `413426f49edf1249f751a05bb8c6e9fd907b11e8da0530fe2da39814885efb22` |
| `packages/SCI-CAL/v0.1/FREEZE_VERIFICATION_R0.5.md` | `4516e4b123daf33e15ca5d74dcb506a65de51e6c204c01698a350f770f9efc16` |
| `packages/SCI-PTC/v0.1/SCIENTIFIC_OWNER_FREEZE_R0.5.md` | `8357961a49272adc40e27a8aa9e760e0d01ff2419ae2c88a62c0f93c9f959e66` |
| `packages/SCI-PTC/v0.1/FREEZE_VERIFICATION_R0.5.md` | `da0a8dc8c5059449afd860abd06955322b3c887188983d4d4b145daea58b6860` |
| `boundaries/v0.1/SOURCE_MANIFEST.md` | `ce813e0adab8270daf713b30db8a271185227048fb79a71abe4b9e4a6ae2ab4a` |
| `producer_interfaces/v0.1/SOURCE_MANIFEST.md` | `a417fb3d22aa46ad7d7f1134b6d804b9d3c3f5a7f601dbb53c19f10a23e72912` |

## Mechanical Verification

The assembled register was checked for the following approved properties:

- one reference-first rule governs every listed boundary;
- static authority is distinct from observation-instance realization;
- no combined APT/Beammap/WVR/TEL/CAL provenance payload is introduced;
- CAL-to-PTC passes the calibrated signal, identity, required flags/validity,
  and parent references;
- CAL produces no calibrated \(r\), while RTC \(r\) remains parent-reachable;
- unrecoverable RTC-grid pointing retains the approved observation-level hard
  stop;
- unavailable response and uncertainty remain typed and claim-local unless a
  requested operation requires them; and
- MAP-only roles remain excluded.

The exact Tune/readout v0.1/r0.1 interface digest and approval record are bound
by its own source manifest.

## Claim Boundary

This manifest establishes the identity and internal reference structure of the
WP-3 closure packet. It does not establish implementation conformity,
observation-instance validity, numerical validation, achieved performance,
science qualification, total uncertainty, production readiness, MAP
availability, or closure of `F-016` or `F-017`. Final finding disposition
belongs to WP-7 clean-room re-audit.
