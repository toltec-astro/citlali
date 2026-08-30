# WP-7 AST Scan-Motion Velocity And Validity Owner Decision Packet

Date: 2026-08-30

Scientific owner: Grant Wilson

Status: approved bounded scientific-owner authority 2026-08-30; bounded AST
implementation passes local gates, with representative-data conformance and
exact-SHA review pending; no nonidentity RTC implementation authorized by this
packet alone

Authority identity: `wp7-ast-scan-motion-v1`

## Approved decision

The scientific owner approved the bounded rules below on 2026-08-30 for the
AST product that supplies science-scan motion to the already-approved WP-7 RTC
scan/array planner. The decision closes only the missing trajectory-field,
physical-scan membership, scalar-velocity, derivative-support, and telescope-
telemetry validity authority needed by the first nonidentity RTC increment. It
does not reopen the RTC filter-bank policy, ordinary AST coordinate
realization, pointing corrections, CAL, VAL, PTC, MAP/JINC, or a common
analysis grid.

The approved disposition establishes `wp7-ast-scan-motion-v1` as written and
authorizes its bounded implementation as a compact immutable AST role. The
numerical defect threshold and local derivative operator are versioned v1
assignments. They may be recalibrated only through a named successor authority
and new representative evidence; calibration may not silently change an
existing plan.

## Authority gap being closed

Frozen SCI-AST v0.1/r0.3 defines coordinate realization, topology, role-local
validity, exact ALIGN parentage, and dependency-limited failure. It does not
define a scan-velocity estimator, a velocity smoothing or differentiation
operator, a per-record telescope-defect rule, or the exact producer fields that
constitute the realized science-scan trajectory. Its exact telescope field
registry and producer `Hold`/state semantics remain explicit owner questions.

The accepted WP-7 RTC planning authority subsequently requires:

```text
v(q) = norm(d theta(q) / dt)
v_max,s = max(v(q)) over valid admitted science-scan motion
```

and assigns the realized trajectory, scalar velocity, derivative validity,
science-scan membership, and telemetry-defect facts to AST. Therefore a legacy
coordinate name, direct finite difference, configured scan-rate header,
processing chunk, percentile, clipping rule, or convenient smoothing window
cannot fill the gap without this owner decision.

## Bounded v1 input family

The first role supports real TolTEC telescope products satisfying all of:

- exact matched observation scope `(observation, subobservation, scan)` across
  the telescope and participating detector inputs;
- `Header.Dcs.ObsGoal = Science`;
- `Header.Dcs.ObsPgm = Lissajous`;
- `Header.ScanFile.Valid = 1`;
- `Header.Source.Epoch = 2000.0` and `Header.Source.CoordSys = 0`; and
- finite telescope records on the declared nominal 50 Hz producer cadence.

Other programs, simulation encodings, source coordinate-system values, invalid
scan files, or absent exact header identity are typed unavailable in v1. This
is a bounded first authority, not a claim that raster `Hold`, simulation, or
another telescope product has the same semantics.

## Physical science-scan identity

The physical science scan is the exact producer scope
`(observation, subobservation, scan)`. The science window is the intersection
of that telescope scan's valid native support with the exact admitted detector
observation support. The matched input headers, not a local row number, bind
the scope.

Legacy `scan_indices` subdivisions used for duration chunking, filter context,
PSD context, parallel work, or storage do not create or renumber physical
science scans. `Header.Lissajous.TScan` is descriptive producer metadata and
does not trim support without an exact producer start/stop relation. The
constant `Hold = 0`, `BufPos = 0`, and `ScanPos = 0` series in observation
152390 do not independently establish finer membership.

Every AST motion record retains physical-scan identity separately from any
processing-chunk identity. Chunk partitioning cannot change membership,
defect classification, derivative support, speed, validity, cause, or the
scan maximum.

## Realized trajectory field registry

For this bounded input family, assign the exact pair

```text
Data.TelescopeBackend.SourceRaAct
Data.TelescopeBackend.SourceDecAct
```

as the producer's realized boresight direction in equatorial J2000, with both
values in radians and bound to the exact occurrence time. The pair is one
spherical direction; right ascension is circular with period `2 pi`,
declination is its latitude, and the resulting unit vector must be finite and
normalizable.

This owner assignment is the v1 authority for the fields' scientific meaning;
their similar names or numeric values are not sufficient by themselves.
`TelAzAct/TelElAct` are realized horizon-state inputs used by other roles but
are not differentiated for this celestial scan-speed product because doing so
would include sidereal tracking in a rotating frame. `TelAzDes/TelElDes`,
`TelAzCmd/TelElCmd`, `TelAzMap/TelElMap`, and
`Header.Lissajous.ScanRate` are not substitutes for realized celestial
motion. Command, desired, and redundant encoder fields may be retained as
diagnostic witnesses but do not replace the realized direction numerically.

All circular differences use the frozen shortest signed interval
`[-pi, pi)`. An exact antipode remains typed unavailable under SCI-AST; no
local unwrap choice resolves it.

## Native telescope continuity

Let raw telescope records have stable identities `k` and exact producer times
`t_k`. A v1 continuity run requires:

- finite, strictly increasing `t_k`;
- finite, normalizable realized direction;
- equal physical-scan identity and field-registry identity; and
- every adjacent interval satisfying `0 < t_(k+1) - t_k <= 0.030 s`.

The 30 ms upper boundary is inclusive. A larger interval is a telescope gap
and splits the run; it is never bridged by the velocity estimator. The v1
bound deliberately admits measured 152390 cadence jitter around 20 ms while
detecting a missing 50 Hz producer record. A different nominal producer
cadence requires a new compatible profile or successor authority rather than
rescaling this threshold from a processing chunk.

## Per-record telemetry-defect classification

AST performs one deterministic position-domain defect test before forming a
derivative. For raw record `k`, use the eleven records `k-5 ... k+5` only when
they are structurally valid members of one continuity run. Express their
spherical logarithms in the canonical local J2000 east/north tangent basis at
record `k`, using their exact time offsets from `t_k`.

For each tangent component:

1. form every pairwise slope within the eleven-record window;
2. take the component-wise median slope;
3. take the component-wise median intercept after removing that slope; and
4. evaluate the radial intercept residual at `t_k`.

Record `k` is a telescope telemetry defect when that radial residual is
strictly greater than `2.0 arcsec`. Equality is valid. The classification does
not replace, clip, winsorize, or move the direction. A defect retains its raw
producer value and exact local cause while making the realized AST direction
and any dependent derivative unavailable for ordinary astronomical use.

The first and last five records of a run lack this symmetric defect-test
support and carry `telemetry_quality_support_unavailable`; they are not assumed
valid. Nonfinite or topology-unavailable windows retain their more specific
causes.

This rule is intentionally versioned. The 2 arcsec boundary is a v1
assignment, not a universal telescope constant. Representative Lissajous,
raster, pointing, and simulation evidence is required before a successor
widens the supported input family or changes the threshold.

## Scalar derivative and velocity

For a nondefective raw record `k`, the derivative window is the same eleven
stable records `k-5 ... k+5`. Every record in the window must be structurally
valid, defect-free, and in the same continuity run and physical science scan.
Use exact time offsets and the canonical J2000 east/north tangent coordinates
at `k`. Fit each coordinate by unweighted least squares to

```text
p(t) = b0 + b1 (t - t_k) + b2 (t - t_k)^2.
```

The realized tangent velocity is the pair of fitted `b1` coefficients and the
scalar speed is their Euclidean norm, converted exactly from radians per
second to arcseconds per second. No additional smoothing, percentile,
clipping, commanded-rate substitution, or detector-derived correction is
permitted. The exact eleven source record identities and time interval are the
derivative support.

The derivative is unavailable if the window intersects a gap, scan boundary,
invalid direction, telemetry defect, unavailable topology, or field-registry
change, or if the fit is rank-deficient or nonfinite. No one-sided endpoint
derivative is introduced in v1.

## Network-specific occurrence view

The raw AST motion product is a producer-time role. ALIGN owns its mapping to
each network's exact reconstructed occurrence time. For a network occurrence,
ALIGN may linearly interpolate the scalar velocity only between two adjacent,
valid AST motion records in the same continuity run; it may not extrapolate,
cross a defect/gap/scan boundary, or upgrade validity. The mapped view retains
both source AST record identities, weights, source times, network occurrence
identity and time, validity, cause, and support.

Paired `x/r` remain bound to that one network occurrence. Different networks
retain independent occurrence and time axes. A gap or unavailable motion view
in one network does not manufacture a slot or absence in another. This role
does not request or construct a cross-network common analysis grid.

## Scan maximum versus occurrence admission

Two related reductions remain distinct:

1. AST forms a compact physical-scan summary from its raw telescope motion
   records. Its candidate set contains only science-scan records with valid
   derivative and scalar speed `v >= 1 arcsec/s`. `v_max,s` is the actual
   maximum of that set, with the exact maximizing record identity and speed.
2. RTC applies the already-approved `1 arcsec/s` inclusive admission rule to
   each network-native occurrence using its mapped AST view. A valid mapped
   speed below the threshold receives the RTC cause
   `below_minimum_science_scan_speed`; that state is not an AST defect.

The raw scan summary prevents the plan from depending on which detector
network happened to sample the continuous telescope path nearest its maximum.
The existing approved 5% planning-velocity margin remains an RTC/filter-bank
policy and is applied after AST publishes the uninflated actual maximum.

If the AST candidate set is empty, the maximum is unavailable with an exact
cause. If a required portion of the physical scan lacks motion authority, AST
may publish partial local facts and diagnostics, but it may not label an
incomplete maximum as the scan maximum. RTC then forms no ordinary
astronomical plan for that scan.

## Typed validity and causes

The v1 role distinguishes at least:

- `not_science_observation`;
- `invalid_scan_file`;
- `unsupported_scan_program`;
- `unsupported_source_frame`;
- `observation_scope_mismatch`;
- `nonfinite_telescope_time`;
- `nonmonotonic_telescope_time`;
- `telescope_gap`;
- `nonfinite_or_unnormalizable_direction`;
- `spherical_topology_unavailable`;
- `telemetry_defect`;
- `telemetry_quality_support_unavailable`;
- `derivative_support_intersects_invalidity`;
- `rank_deficient_derivative_fit`;
- `nonfinite_derivative`;
- `network_mapping_support_unavailable`; and
- `scan_maximum_incomplete`.

These AST causes do not overwrite ALIGN mapping causes, `x/r` member-local
validity or causes, the RTC slow-motion cause, or a downstream eligibility
decision. When a pair-wide RTC action follows from one mapped motion fact, the
local AST cause remains inspectable.

## Product, ownership, and lifecycle

`SCI-AST:scan_motion_planning@1` is a compact immutable role. It owns only
genuinely derived defect state, tangent derivative, scalar speed, exact compact
support references, validity/cause, and the scan summary. It references the
immutable raw telescope axis, direction fields, physical-scan identity, and
ALIGN network views through bounded typed handles. It does not duplicate the
input axes or pointing planes, content-hash full timestream planes, attach an
identity object per detector cell, or create generalized provenance.

Requested, effective, observation-resolved, and realized identities bind the
field registry, continuity rule, defect operator and threshold, derivative
operator, frame, unit conversion, physical scan, source handle, and network
mapping profile. The realized record is a compact statement of what was
formed; it does not duplicate the product or support planes.

## Observation 152390 evidence

The retained raw telescope file
`tel_toltec_2026-02-19_152390_00_0002.nc` has SHA-256
`2845455a620635955c00a4731e0d9720cfa456fece79d1729cf755a366a1ad6b`
and contains 62,109 records. Its exact producer identity is
`(152390, 0, 2)`, `ObsGoal=Science`, `ObsPgm=Lissajous`,
`ScanFile.Valid=1`, `Source.Epoch=2000.0`, and `Source.CoordSys=0`.
`TelTime` is finite and strictly increasing; its intervals range from
approximately 18.9998 ms to 21.1163 ms.

Direct adjacent great-circle differences of `SourceRaAct/SourceDecAct`
produce a maximum above 1,400 arcsec/s. Inspection identifies two isolated
one-record discontinuities near raw indices 2504 and 12971, each approximately
28--29 arcsec in one 20 ms record and reversed immediately. One is contradicted
by the independent elevation encoder; the other is contradicted by the smooth
commanded trajectory. They are telemetry-defect evidence, not permission to
clip high speeds.

The approved eleven-record robust defect test separates those two events from
the retained trajectory in this file. The approved quadratic derivative then
retains sustained, locally corroborated speeds above 200 arcsec/s and gives an
evidence-only candidate maximum of approximately `221.405 arcsec/s`. This is
materially different from treating the mislabeled
`Header.Lissajous.ScanRate` value as `50 arcsec/s` and demonstrates why neither
the header nor a percentile can be planning authority.

The approximate candidate maximum is diagnostic evidence, not an accepted
152390 AST product. Exact implementation arithmetic, source mapping, product
identity, typed causes, and complete scenario gates must pass before 152390
can close the representative AST gate.

## Required implementation and acceptance gates

Before nonidentity RTC consumes this role:

1. implement the immutable raw motion product and network-specific mapped
   views without changing the accepted identity RTC route;
2. prove exact synthetic constant, uniform great-circle, accelerated,
   below/exact/above-threshold, wrap, antipode, gap, endpoint, rank-failure,
   and nonfinite behavior;
3. prove an isolated position spike is typed as a telemetry defect while a
   sustained valid speed above 200 arcsec/s remains valid and can set the
   maximum;
4. prove processing chunks and order/parallel schedules do not change any
   scientific identity, speed, validity, cause, support, or maximum;
5. prove two networks with distinct native times retain distinct mapped motion
   views and that no common-grid constructor is invoked;
6. prove a gap in one network does not create an occurrence or absence in
   another;
7. run focused AST/ALIGN/RTC tests, the full repository gates, and a clean
   observation-152390 evidence package;
8. extend empirical coverage to representative raster, pointing, and
   simulation data before extending the v1 supported input family; and
9. obtain a fresh independent exact-SHA conformance review before any
   nonidentity RTC terminal publication or production activation.

The existing certified-filter-bank, representative broadband PSD, naive/JINC,
and OOF/fruitloops gates remain separate and pending.

## Explicit nonclaims

This packet does not claim that:

- the approved v1 threshold is a universal LMT hardware specification;
- the 152390 candidate maximum is already authoritative;
- command or desired pointing is realized motion;
- direct horizon-coordinate velocity is celestial scan speed;
- every telescope program shares Lissajous membership semantics;
- ordinary AST coordinate realization has passed implementation conformity;
- a common analysis grid is required;
- nonidentity RTC, CAL, VAL, PTC, MAP/JINC, or a persistent TOD schema is
  implemented or activated; or
- production readiness or observational performance is established.

## Scientific-owner disposition

The scientific owner supplied the following exact approval on 2026-08-30:

```text
I approve the bounded WP-7 AST scan-motion velocity and validity authority
proposed in WP7_AST_SCAN_MOTION_OWNER_DECISION_PACKET_2026-08-30.md as
wp7-ast-scan-motion-v1, including the exact field registry, physical-scan
membership, 30 ms continuity boundary, 2 arcsec telemetry-defect boundary,
eleven-record quadratic derivative, typed causes, raw scan maximum, and
network-specific mapped-view semantics.
```
