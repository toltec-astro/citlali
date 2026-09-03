# SCI-VAL v0.1 — Adjacent Scientific Source-Binding Register

Register identity: `SCI-VAL_SOURCE_BINDING_REGISTER v0.1/r0.3-map-r0.7.1-jinc-stage-a-q002-noi-stage-a-r0.18-2026-08-30`

Status: exact manifest-bound r0.3 source-binding successor for SCI-MAP r0.7.1,
owner-approved SCI-JINC Stage A Q002, and SCI-NOI Stage A r0.18;
availability limits preserved

Last updated: `2026-08-30`

## Purpose

This register binds every adjacent meaning used by SCI-VAL to the exact
approved package or sanitized boundary authority available to this revision.
It does not import a full adjacent contract, upgrade an open package to
frozen authority, or create a missing policy.

The source tables rendered in the r0.3 rationale and engineering companion
are historical snapshots. This exact revision is the authority for the
manifest-bound SCI-MAP r0.7.1 evaluation generation, the separately manifest-
bound SCI-JINC Stage A Q002 admission profile, and the four separately bound
SCI-NOI Stage A r0.18 profiles. A later adjacent-source update requires a new
immutable register revision and affected profile bindings; it does not rewrite
SCI-VAL Core narrative or this revision.

| Producer or use owner | Exact source/version binding | Imported meaning | Compatibility and change consequence |
| --- | --- | --- | --- |
| SCI-ALIGN | Frozen SCI-ALIGN v0.1/r0.3. Packet-manifest SHA-256 `26285329635c722cb9161d383ad1b95f56a03b782c101bcd89d8785a3575faac`; joint-freeze-record SHA-256 `dda3767559ab3a7f8801ae2589e99c1d54ea582924c9b366b44b0bf6eda4b6e1`; WP-2 boundary-manifest SHA-256 `ce813e0adab8270daf713b30db8a271185227048fb79a71abe4b9e4a6ae2ab4a` | Stable original occurrence, physical acquisition and valid-original facts, origin/synthesis state, exact representative relation, typed causes, time/support, and immutable lifecycle | Compatible with the direct representative-origin proposition in `SCI-VAL:independent_exposure@1`. This does not make a coordinate, signal, or downstream use eligible by itself |
| SCI-AST | Frozen SCI-AST v0.1/r0.3. Packet-manifest SHA-256 `b54b6013750540f28aad02339a60bf36078980dc53b132beab73069d66ef3601`; same joint freeze record; WP-2 manifest above; WP-4 source-hygiene-manifest SHA-256 `57dacf3a5847a24a85b754e878306bd5efb088f571c354f650d0961bdd3ca9a0` | Exact ALIGN-grid and RTC-grid coordinate roles, coordinate validity/causes, parentage, and typed coordinate-response/uncertainty availability | Coordinate validity remains distinct from signal validity and independent exposure. The WP-4 notation map changes no source meaning and authorizes no MAP projection |
| SCI-RTC | Frozen SCI-RTC v0.1/r0.12. Freeze-record SHA-256 `0cac4396df225c1f2808ee1055e063c9a4e72a02549557c5e997f54d72dac0bf`; WP-2 and WP-4 manifests above | Representative source, original/synthesized/replaced origin, typed causes, operator controls, support, influence precision, response/uncertainty availability, and immutable lifecycle | Compatible with `SCI-VAL:independent_exposure@1`: direct representative synthesis/replacement remains disqualifying only for that proposition. Nonrepresentative influence remains use-owner governed |
| SCI-CAL | Frozen SCI-CAL v0.1 science-rationale r0.5 / engineering-conformance r0.4. Freeze-record SHA-256 `413426f49edf1249f751a05bb8c6e9fd907b11e8da0530fe2da39814885efb22`; WP-3 source-manifest SHA-256 `d407228bfbbdbe8be994e7e84e4945fc6868365c2d045c18ac7ce1e5c40ae9aa` | Calibration availability/domain, detector binding, applied-calibration state, engineering-only/science-qualification classification, response, uncertainty, and typed causes | CAL classifications remain producer facts rather than universal eligibility decisions. Each PTC use profile must state its own consequence; no identity response or missing numerical policy is inferred |
| SCI-PTC | Frozen SCI-PTC v0.1/r0.5. Freeze-record SHA-256 `8357961a49272adc40e27a8aa9e760e0d01ff2419ae2c88a62c0f93c9f959e66` | Distinct basis/loading fit, application, output, coefficient/QC, response, empirical/simulation, support, and staged lifecycle roles | Source binding does not register a policy. Every reserved PTC profile remains unavailable until an exact PTC-owned profile record is registered. A new binding never rewrites an earlier VAL decision |
| Tune/readout and telescope inputs | Exact approved `TUNE_READOUT_NATIVE_XR_PRODUCER_INTERFACE v0.1/r0.1`, SHA-256 `f9659b34a49a07d4287c4a70db798cdd2ec30049531da603fcca1e9d1fdd5969`; producer-interface-manifest SHA-256 `a417fb3d22aa46ad7d7f1134b6d804b9d3c3f5a7f601dbb53c19f10a23e72912`. Observation telescope records remain runtime parents referenced through ALIGN/AST/RTC/CAL facts | Native paired-\(x/r\) identity and producer-owned mapping validity; telescope/time/observing facts only through their owning package facts | VAL does not reinterpret Tune/readout or telescope records and does not copy their payloads into a profile. A missing upstream owner fact remains unavailable at the dependent scope |
| SCI-MAP | SCI-MAP v0.1/r0.7.1 under owner directive SHA-256 `f7747eea28710d524e12c818b872ac3fcc49f413271f83c0644ae129949a8c8c` and `FREEZE_ONLY_ERRATA_R0.7.1.md`; shared r0.7.1 authority; exact `SCI-PTC_TO_SCI-MAP v0.1/r0.1`; original-footprint-coordinate boundary; occurrence profile `SCI-MAP:map_upstream_admission@2`; aggregate/coefficient profiles; frozen PTC/CAL/AST sources above; every exact digest bound by `SCI-MAP_SOURCE_MANIFEST v0.1/r0.7.1` | MAP-owned occurrence admission; exact PTC/AST signal join; ordered typed contribution gates; stable-original AST ALIGN-grid exposure coordinate; one exact MAP operator; fixed-state, PTC full-procedure, and PTC+MAP re-resolved response roles; coadd compatibility; response/covariance disclosure; and MAP-local base-product validity | `SCI-MAP:map_upstream_admission@1` is immutable historical r0.5 authority, not a compatibility alias. `@2` and `SCI-MAP:observation_coadd_admission@1` are evaluable only under these exact compatible r0.7.1 records. VAL evaluates MAP-authored rules and preserves all four named decision fields, causes, and immutable reasons; it does not select the missing PTC MAP coefficient, grant MAP use through a PTC QC decision, classify coefficient numerics, place pixels, accumulate maps, allocate exposure, or author MAP policy |
| SCI-JINC | Owner-approved SCI-JINC v0.1 Stage A successor packet at commit `88dcce8b0f7b1d78053b25831b39cf370afd47cc`; author-packet manifest SHA-256 `52a8e843456a8cb033b7593d9b9f67fb83b0ee565c91c141d8e16d46b906140e`; admission profile `SCI-JINC:jinc_map_contribution@1`, SHA-256 `2db95da7e5d1b980df79993907d45ac0ababc3aa05c189bfb62dcf04ff2c2e8a`; `SCI-PTC_TO_SCI-JINC v0.1/r0.3`, SHA-256 `5769d413460e931745e0d401ea432b12d1077c15466247c49caa71b997d4ab1e`; `SCI-AST_TO_SCI-JINC v0.1/r0.2`, SHA-256 `efffa7059b59c89793fa1d523fb3bb48235f1ab55f7d55060af1600cbfd470a5`; Q002 approval record at commit `ebc0e907fe96163e48818fec99e42cc272b2cfb4`, SHA-256 `c70e8216e816a7f98486b4c61236acc49713a5ce1d6f5ba722ad6e015e0c7e9f`; frozen PTC/AST and their upstream parents as bound above | JINC-owned occurrence-level upstream admission before rounded-center, pixel-support, signed-kernel, accumulator, conditioning and bundle-validity gates; exact PTC transformed-signal/output-retention/coefficient-family permission and QC binding; exact same-processed-sample AST coordinate association; stable TolTEC array and complete immutable parent/lifecycle identity | `SCI-JINC:jinc_map_contribution@1` is evaluable only under this exact source/profile binding. VAL evaluates the JINC-authored atomic admission rule and preserves all four decision fields and causes; it does not select or authorize a PTC family, infer a TolTEC parameter set, place a JINC footprint, apply `kappa_ip`, accumulate a product, create response/covariance/exposure roles, or author JINC policy. A missing authorized coefficient family or TolTEC parameter set remains the JINC-approved typed-unavailability state |
| SCI-NOI | Owner-approved SCI-NOI v0.1 Stage A r0.18 packet at commit `2f7076e0c7a51320413a86cc6be74c2d3e8f1537`; author-packet manifest `SCI-NOI_AUTHOR_PACKET_MANIFEST v0.1/r0.18`, SHA-256 `b6f8e7252e7f61f4506899cb3e8e26cf939887bb48464852713f8ce81ac77ca0`; four owner-approved NOI profile policy/action records in `SCI-NOI_VAL_PROFILE_DRAFTS.md`, SHA-256 `c89883d8c20f72aea05f0ae62464daee3a3ee6e81543ff313888ac318a192d6b`; sanitized owner decisions `SCI-NOI_OWNER_DECISIONS v0.1/r0.18`, SHA-256 `272ac939b8a7109a123073b1a39fcdd7ac4129c603683ee81257b94ab2f55a0b`; ODQ-111 approval SHA-256 `4e377dba46f8aead91ce14291ff6ae41de46476ff6cf3eab732d3aa29b503e67`; `SCI-PTC_TO_SCI-NOI-GEN v0.1/r0.6`, SHA-256 `0a6484058569930cee62e80e04ca2045c107fde67603f662473ae471406f905c`; `SCI-MAP_TO_SCI-NOI v0.1/r0.3`, SHA-256 `4273c5a75ff10d00506e5aa8732690cd3f398ff5afbaa561af8f1434ec467e29`; `SCI-JINC_TO_SCI-NOI v0.1/r0.3`, SHA-256 `7bf0ff489957943cee5abcd581b6b6b1fea0840969d62ced4d73072cff8b51f8`; frozen SCI-PTC/MAP/JINC and upstream authorities as bound above | NOI-owned generation-input, uncertainty-member, uncertainty-ensemble, and standardization admission policies; exact owner-selected fixed-state PTC-to-frozen-MAP route and typed numerical-unavailability gates; GEN producer facts; immutable MAP/JINC parents; separate GEN/UNC/STD identities; exact consumer actions and four named decision fields | The four `SCI-NOI:*@1` profiles are evaluable only under this exact source/profile binding. VAL binds/evaluates the NOI-authored rules and preserves producer facts, causes, and separate request/applicability/eligibility/realization fields; it does not assign signs, apply MAP arithmetic, complete members, estimate uncertainty, construct standardized signal, author NOI policy, or make an unavailable numerical parent available. Missing finite-design mechanics, PTC MAP coefficient, admitted `coverage_cut`, or another named required authority preserves the owner-approved typed-unavailability state |

## Binding Rule

An exact source binding is part of policy/profile identity and replay. If the
authoritative source changes, VAL must either resolve a newly registered
compatible binding or return the owner-declared unavailable result. It must
not silently substitute “current” adjacent meaning. In particular, this
register supplies no MAP, JINC, or NOI predicate, threshold, or exception of its
own. MAP predicates are evaluable only through the separately owner-authored
and registered `SCI-MAP:map_upstream_admission@2` or
`SCI-MAP:observation_coadd_admission@1` records. The JINC predicate is
evaluable only through the separately owner-authored and registered
`SCI-JINC:jinc_map_contribution@1` record.
The NOI predicates are evaluable only through the four separately owner-
authored and registered `SCI-NOI:*@1` records. No profile evaluation realizes
the next GEN, UNC, or STD action.

The adjacent-source tables embedded in the r0.3 rationale and engineering view
remain historical snapshots. This exact register revision binds the MAP
r0.7.1, SCI-JINC Stage A Q002, and SCI-NOI Stage A r0.18 source versions without
modifying VAL Core or retroactively changing a prior evaluation identity. A
successor requires a new revision identity and digest.
