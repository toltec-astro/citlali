# WP-7 AST Route-Family Motion Owner Decision Packet (2026-08-30)

Status: **approved bounded scientific-owner authority 2026-08-30;
implementation and exact-SHA conformance review pending**

Authority identifier: `wp7-ast-scan-motion-v2`

This packet records the approved smallest extension of the accepted
[`wp7-ast-scan-motion-v1`](WP7_AST_SCAN_MOTION_OWNER_DECISION_PACKET_2026-08-30.md)
authority needed to measure the WP-7 filtering/downsampling witness matrix.
The scientific owner approved it exactly as requested on 2026-08-30. Approval
authorizes the bounded implementation but does not promote the exploratory
values below to F0 results until a conforming implementation, clean
representative execution, and fresh exact-SHA review pass.

## 1. Decision requested

Approve one bounded successor authority that:

1. preserves the complete accepted v1 trajectory, continuity, defect,
   derivative, scalar-speed, validity, compact-summary, and network-mapping
   semantics;
2. admits the exact supported `Science/Lissajous`, `Oof/Lissajous`, and
   `Pointing/Lissajous` route profiles to that unchanged numerical operator;
3. adds one exact rectilinear continuous `BeamMap/Map` profile and defines its
   physical-scan membership from producer hold state and the realized corrected
   telescope trajectory inside the configured map footprint;
4. assigns compact physical-segment identity before applying the unchanged
   v1 continuity and eleven-record rules; and
5. returns typed causes for unsupported profiles and non-member records.

This decision does **not** select an RTC factor, filter, or velocity ceiling.
It does not authorize nonidentity RTC, alter native network timing, construct a
common analysis grid, or change a sampling rate.

## 2. Existing authority retained unchanged

The following accepted `wp7-ast-scan-motion-v1` facts remain authoritative:

- telescope record identity and exact `TelTime`;
- realized J2000 `SourceRaAct` and `SourceDecAct` trajectory;
- the inclusive `dt <= 30 ms` continuity boundary;
- the strictly greater than `2 arcsec` robust local-position residual defect
  boundary;
- the centered eleven-record quadratic derivative on a valid contiguous
  physical segment;
- scalar on-sky speed in `arcsec/s`;
- typed validity and local causes;
- the raw valid-scan maximum as a compact diagnostic, never as whole-scan RTC
  admission;
- immutable raw AST ownership and network-specific mapped views that retain
  native occurrence/time identity; and
- no cross-network common-grid dependency.

Approval of this packet widens only route-profile admission and the exact
physical-scan membership needed by the newly admitted Beammap profile. It does
not reopen the accepted numerical estimator.

## 3. Supported route profiles

The successor recognizes only these profiles:

Every profile additionally requires the v1 exact matched observation scope,
`Header.ScanFile.Valid=1`, `Header.Source.Epoch=2000.0`,
`Header.Source.CoordSys=0`, the exact v1 realized J2000 field registry, and the
declared nominal 50 Hz producer cadence. Those predicates are not relaxed by
this proposal.

| Profile | Required producer identity | Physical-scan membership |
| --- | --- | --- |
| Science Lissajous | `Header.Dcs.ObsGoal=Science`, `Header.Dcs.ObsPgm=Lissajous` | Every native telescope record in the exact observation scope is a member; accepted v1 continuity and defect rules govern estimator support |
| OOF Lissajous | `Header.Dcs.ObsGoal=Oof`, `Header.Dcs.ObsPgm=Lissajous` | Same as Science Lissajous |
| Pointing Lissajous | `Header.Dcs.ObsGoal=Pointing`, `Header.Dcs.ObsPgm=Lissajous` | Same as Science Lissajous |
| Rectilinear continuous Beammap | `Header.Dcs.ObsGoal=BeamMap`, `Header.Dcs.ObsPgm=Map`, `Header.Map.ExecMode=0`, `Header.Map.MapCoord=Az`, `Header.Map.MapMotion=Continuous`, `Header.Map.MapPath=Rectilinear`, `Header.Map.HoldDuringTurns=0`, zero `Header.Map.XOffset` and `Header.Map.YOffset`, finite positive `Header.Map.XLength` and `Header.Map.YLength`, finite `Header.Map.ScanAngle`, and exact horizon realization fields | Exact predicate in Section 4 |

`Pointing/Lissajous` is included because it exercises the same bounded producer
family and unchanged operator. It is supporting coverage, not a substitute for
the required Beammap, OOF, and Science witnesses.

Any other observation goal, program, map execution mode, coordinate mode,
motion mode, path, or missing required field is unsupported. The implementation
must fail closed for that record or observation with a typed cause; it must not
guess a membership rule.

## 4. Exact Beammap physical-scan membership

The added native per-record field registry is exactly:

```text
Data.TelescopeBackend.Hold
Data.TelescopeBackend.TelAzAct
Data.TelescopeBackend.TelElAct
Data.TelescopeBackend.SourceAz
Data.TelescopeBackend.SourceEl
Data.TelescopeBackend.TelAzCor
Data.TelescopeBackend.TelElCor
```

The fields are immutable producer facts. AST may reference them through a
bounded typed source view; it does not duplicate them into the derived motion
plane.

For each native telescope record, first require the producer hold field to be
exactly zero. Any nonzero hold bit pattern is not a physical Beammap scan
member.

For a non-hold record, define the realized corrected horizontal tangent-plane
offset, in angular units consistent with the producer fields, as

```text
dx = cos(TelElAct - TelElCor) * wrap_pi(TelAzAct - SourceAz) - TelAzCor
dy = TelElAct - SourceEl - TelElCor
```

where `wrap_pi` returns the unique shortest signed azimuth difference on
`[-pi, pi)`. Rotate this realized offset by the configured `ScanAngle`:

```text
x_map =  cos(ScanAngle) * dx + sin(ScanAngle) * dy
y_map = -sin(ScanAngle) * dx + cos(ScanAngle) * dy
```

The record is a physical-scan member exactly when

```text
abs(x_map) <= XLength / 2
and
abs(y_map) <= YLength / 2
```

The footprint boundary is inclusive. A record with a non-finite required input
is invalid with a typed cause, not merely outside the footprint.

This uses the realized actual/correction fields. `TelAzMap` and `TelElMap` are
not substituted as the authority because the local representative Beammap
shows measurable differences from the corrected realized trajectory, including
boundary-membership differences.

## 5. Physical-segment identity and derivative support

A physical segment is a maximal native-contiguous run of records for which the
applicable profile membership predicate is true. Segment identity is the
compact tuple

```text
(observation identity, route-profile identity, first telescope-record identity)
```

or an exactly equivalent compact stable representation.

The accepted v1 continuity, defect, and eleven-record derivative rules are then
applied within each segment. A derivative window may not cross a membership
boundary, hold interval, footprint excursion, native-record gap, accepted
continuity break, or telemetry defect.

The successor does **not** adopt two legacy raster implementation choices as
AST science authority:

- it does not discard the first record of every physical segment merely to
  reproduce an existing range convention; and
- it does not reject a physical segment solely because it contains fewer than
  two seconds of 50 Hz records.

Those choices may remain in an unchanged legacy Beammap product path until a
separate consumer decision is made. AST reports the physical facts and exact
operator support. It does not erase short physical segments to imitate a
consumer's historical container construction.

## 6. Validity and typed causes

The successor preserves all v1 derivative, continuity, telemetry-defect, edge,
and mapping causes. It adds or makes explicit these route-membership causes:

| Cause | Meaning |
| --- | --- |
| `unsupported_observation_goal` | The observation goal is outside the exact supported profile registry |
| `unsupported_observation_program` | The program is outside the exact supported profile registry |
| `unsupported_beammap_profile` | A `BeamMap/Map` field such as execution mode, map coordinate, motion, or path is not the approved bounded profile |
| `nonfinite_membership_field` | A required trajectory, correction, source, footprint, or angle value is non-finite |
| `producer_hold_active` | The Beammap producer hold field is nonzero |
| `outside_scan_footprint` | A non-hold finite Beammap record lies outside the inclusive realized footprint |

The implementation may retain the v1 serialized cause name
`not_science_observation` for the v1 product contract. A v2 product must expose
the more exact route-profile distinction above or a typed backward-compatible
encoding with the same meaning. It must not collapse local membership causes
into a generic invalid bit.

The accepted RTC lower-speed and later per-mode upper-speed causes remain RTC
authority. They are not AST causes and are not added here.

## 7. Network-specific mapped views

The accepted v1 network mapping remains unchanged. For every originating
network occurrence, the mapped view retains:

- network identity;
- exact network occurrence and reconstructed network time;
- corresponding telescope-record and physical-segment identity;
- raw AST validity, speed, and local cause; and
- the exact ALIGN association used for that network occurrence.

No observation-wide slot lattice is introduced. A gap or absent occurrence in
one network cannot manufacture a slot, absence, support fact, or velocity in
another network.

## 8. Representative evidence supporting the proposal

This section is design evidence only. It was produced by an independent
read-only implementation of the unchanged accepted v1 operator over the exact
D0 telescope files. The verifier first reproduced the accepted observation
152390 result to floating-point roundoff:

| Quantity | Accepted C++ result | Independent evidence-only result |
| --- | ---: | ---: |
| valid derivative records | 62,067 | 62,067 |
| defect record identities | 2504, 12971 | 2504, 12971 |
| raw maximum record | 16973 | 16973 |
| raw maximum speed (`arcsec/s`) | 221.40490828695155 | 221.4049082869514 |

Applying that unchanged operator counterfactually to the proposed Lissajous
profiles gives:

| Observation | Route | Telescope records | Valid derivative records | Defects | Raw maximum (`arcsec/s`) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 152385 | OOF Lissajous | 3,271 | 3,251 | 0 | 135.38178144220348 |
| 152386 | OOF Lissajous | 3,353 | 3,333 | 0 | 158.40684078738656 |
| 152387 | OOF Lissajous | 3,266 | 3,246 | 0 | 140.54776730507834 |
| 152391 | Pointing Lissajous | 3,309 | 3,289 | 0 | 171.69626792458715 |

For Beammap 148670, alternate membership hypotheses demonstrate why the
membership decision must be explicit:

| Evidence-only membership | Member records | Segments | Valid derivative records | Raw maximum (`arcsec/s`) |
| --- | ---: | ---: | ---: | ---: |
| whole observation | 157,219 | 1 | 157,199 | 116.40231518714249 |
| `Hold == 0` only | 113,383 | 222 | 109,182 | 116.40231518714249 |
| proposed non-hold and inclusive realized footprint | 61,816 | 252 | 57,563 | 75.29115202170506 |
| evidence-only analogue of current legacy range construction | 61,100 | 198 | 57,140 | 75.29115202170506 |

The proposed predicate is the physical definition, not a fit to the last row.
The legacy analogue trims one record from retained starts and drops 54 short
fragments totaling 518 raw records; those are consumer conventions rather than
evidence that the records were not physical scan occurrences.

Exact telescope-input SHA-256 values used for the exploratory evidence are:

| Observation | SHA-256 |
| ---: | --- |
| 148670 | `a4eccc358433aded7bf21e36d513502ad34f6e5e9b234d003f7fd43e1db0c36c` |
| 152385 | `eea40ee5443022370c4c66ba289ce2ba566aaf0e645abe72985a53788a9e2926` |
| 152386 | `35468feccdb9d07d0a370e8fca2d3b3bee379f983786fb4bce330a44dc7e96d8` |
| 152387 | `e9fdcdf66c45e517fa4e3cf773e7ccff9173e1bc9c9ff611d48cd9419e775c3c` |
| 152390 | `2845455a620635955c00a4731e0d9720cfa456fece79d1729cf755a366a1ad6b` |
| 152391 | `63973b7a7cb0668d89378639a92fd76c80ab12f334dab4b4d81e3c05d5cdc375` |

These values are not yet accepted F0 evidence because the proposed route
profiles were not authority when they were computed.

## 9. Required implementation and verification after approval

Approval authorizes only a bounded AST authority/implementation repair:

1. supersede the affected v1 route-profile clauses with v2 while preserving the
   v1 numerical operator and all unaffected contracts;
2. implement an explicit route-profile registry and the exact membership rules
   above without changing RTC filtering/downsampling code;
3. add focused below/exact/above boundary tests for the inclusive Beammap
   footprint, nonzero hold, unsupported profile fields, short segments, segment
   isolation, continuity, defect, and derivative support;
4. add representative exact-SHA tests for Science, OOF, Pointing, and Beammap;
5. prove network-mapped views preserve independent native time axes and do not
   invoke common-slot association;
6. rerun focused AST/ALIGN tests, the full repository gates, and the clean D0/F0
   census; and
7. obtain a fresh independent exact-SHA conformance review before using the
   resulting Beammap/OOF motion values to advance F1.

The clean post-approval census, not the exploratory table, becomes the first
eligible F0 evidence.

## 10. Explicit nonclaims

This proposal does not:

- change the accepted estimator or any numerical boundary;
- assert that every future Beammap, OOF, or Pointing producer profile is
  supported;
- authorize a percentile velocity, scan maximum, or global worst sample as an
  RTC admission input;
- select a decimation factor or filter family;
- authorize nonidentity numerical RTC or terminal publication;
- define a persistent AST or RTC TOD schema;
- construct a common analysis grid;
- alter CAL, PTC, MAP/JINC, or legacy Beammap science behavior; or
- promote the counterfactual measurements to accepted evidence.

## 11. Scientific-owner disposition

The scientific owner supplied the following exact approval on 2026-08-30:

> I approve the bounded WP-7 AST route-family motion authority proposed in
> `WP7_AST_ROUTE_FAMILY_MOTION_OWNER_DECISION_PACKET_2026-08-30.md` as
> `wp7-ast-scan-motion-v2`, including the preserved v1 numerical operator, the
> Science/OOF/Pointing Lissajous profiles, the exact rectilinear Beammap
> profile, the non-hold realized in-footprint membership predicate and
> inclusive boundaries, compact physical-segment identity, typed causes, and
> unchanged network-specific mapped-view semantics.
