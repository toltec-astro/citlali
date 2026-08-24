# SCI-VAL v0.1 — Adjacent Scientific Source-Binding Register

Status: continuing r0.3 source-binding authority updated under approved
`WP5-OWNER-D001`; availability limits preserved

Last updated: `2026-08-24`

## Purpose

This register binds every adjacent meaning used by SCI-VAL to the exact
approved package or sanitized boundary authority available to this revision.
It does not import a full adjacent contract, upgrade an open package to
frozen authority, or create a missing policy.

The source tables rendered in the r0.3 rationale and engineering companion
are snapshots of this register. This file remains the continuing authority;
an adjacent source update is recorded here and in affected immutable profile
bindings without requiring a rewrite of SCI-VAL Core narrative.

| Producer or use owner | Exact source/version binding | Imported meaning | Compatibility and change consequence |
| --- | --- | --- | --- |
| SCI-ALIGN | Frozen SCI-ALIGN v0.1/r0.3. Packet-manifest SHA-256 `26285329635c722cb9161d383ad1b95f56a03b782c101bcd89d8785a3575faac`; joint-freeze-record SHA-256 `dda3767559ab3a7f8801ae2589e99c1d54ea582924c9b366b44b0bf6eda4b6e1`; WP-2 boundary-manifest SHA-256 `ce813e0adab8270daf713b30db8a271185227048fb79a71abe4b9e4a6ae2ab4a` | Stable original occurrence, physical acquisition and valid-original facts, origin/synthesis state, exact representative relation, typed causes, time/support, and immutable lifecycle | Compatible with the direct representative-origin proposition in `SCI-VAL:independent_exposure@1`. This does not make a coordinate, signal, or downstream use eligible by itself |
| SCI-AST | Frozen SCI-AST v0.1/r0.3. Packet-manifest SHA-256 `b54b6013750540f28aad02339a60bf36078980dc53b132beab73069d66ef3601`; same joint freeze record; WP-2 manifest above; WP-4 source-hygiene-manifest SHA-256 `57dacf3a5847a24a85b754e878306bd5efb088f571c354f650d0961bdd3ca9a0` | Exact ALIGN-grid and RTC-grid coordinate roles, coordinate validity/causes, parentage, and typed coordinate-response/uncertainty availability | Coordinate validity remains distinct from signal validity and independent exposure. The WP-4 notation map changes no source meaning and authorizes no MAP projection |
| SCI-RTC | Frozen SCI-RTC v0.1/r0.12. Freeze-record SHA-256 `0cac4396df225c1f2808ee1055e063c9a4e72a02549557c5e997f54d72dac0bf`; WP-2 and WP-4 manifests above | Representative source, original/synthesized/replaced origin, typed causes, operator controls, support, influence precision, response/uncertainty availability, and immutable lifecycle | Compatible with `SCI-VAL:independent_exposure@1`: direct representative synthesis/replacement remains disqualifying only for that proposition. Nonrepresentative influence remains use-owner governed |
| SCI-CAL | Frozen SCI-CAL v0.1 science-rationale r0.5 / engineering-conformance r0.4. Freeze-record SHA-256 `413426f49edf1249f751a05bb8c6e9fd907b11e8da0530fe2da39814885efb22`; WP-3 source-manifest SHA-256 `d407228bfbbdbe8be994e7e84e4945fc6868365c2d045c18ac7ce1e5c40ae9aa` | Calibration availability/domain, detector binding, applied-calibration state, engineering-only/science-qualification classification, response, uncertainty, and typed causes | CAL classifications remain producer facts rather than universal eligibility decisions. Each PTC use profile must state its own consequence; no identity response or missing numerical policy is inferred |
| SCI-PTC | Frozen SCI-PTC v0.1/r0.5. Freeze-record SHA-256 `8357961a49272adc40e27a8aa9e760e0d01ff2419ae2c88a62c0f93c9f959e66` | Distinct basis/loading fit, application, output, coefficient/QC, response, empirical/simulation, support, and staged lifecycle roles | Source binding does not register a policy. Every reserved PTC profile remains unavailable until an exact PTC-owned profile record is registered. A new binding never rewrites an earlier VAL decision |
| Tune/readout and telescope inputs | Exact approved `TUNE_READOUT_NATIVE_XR_PRODUCER_INTERFACE v0.1/r0.1`, SHA-256 `f9659b34a49a07d4287c4a70db798cdd2ec30049531da603fcca1e9d1fdd5969`; producer-interface-manifest SHA-256 `a417fb3d22aa46ad7d7f1134b6d804b9d3c3f5a7f601dbb53c19f10a23e72912`. Observation telescope records remain runtime parents referenced through ALIGN/AST/RTC/CAL facts | Native paired-\(x/r\) identity and producer-owned mapping validity; telescope/time/observing facts only through their owning package facts | VAL does not reinterpret Tune/readout or telescope records and does not copy their payloads into a profile. A missing upstream owner fact remains unavailable at the dependent scope |
| SCI-MAP | Explicitly deferred and unbound for the processed-timestream packet | Stable ownership separation may be cited only as a boundary: MAP upstream admission, projection, contribution, exposure, support, response/covariance, coaddition, and final validity remain MAP-owned | No `SCI-MAP:map_upstream_admission` profile is registered or evaluable. MAP source/profile work requires a separately approved future packet |

## Binding Rule

An exact source binding is part of policy/profile identity and replay. If the
authoritative source changes, VAL must either resolve a newly registered
compatible binding or return the owner-declared unavailable result. It must
not silently substitute “current” adjacent meaning. In particular, this
register contains no MAP predicate, threshold, or exception and does not make
`SCI-MAP:map_upstream_admission` evaluable.

The adjacent-source tables embedded in the r0.3 rationale and engineering view
remain historical snapshots. This continuing register supersedes their source
versions without modifying VAL Core or retroactively changing a prior
evaluation identity.
