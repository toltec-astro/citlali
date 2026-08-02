# SCI-ALIGN-001 phase-zero D004 owner decision — 2026-08-01

Status: `ALIGN-P0-D004` resolved for bounded active-registry design; exact
`Hold` transition-side and scan-selection evidence routed to
`ALIGN-P0-D005`; `D005` pending; phase one unauthorized

Package: `SCI-ALIGN-001`

## Authority and evidence boundary

The project owner approved a restricted telescope/HWPR registry and output
contract for the exact admitted nonpolarimetric intensity profiles. This
decision deliberately does not assign scientific contracts to all 383 rows in
the discovery registry. It approves 20 canonical active internal fields, with
22 admitted raw source-name forms because RA and Dec each have one legacy
alias, and makes exact-only/zero-span behavior the universal fail-closed
default for every inactive or semantically unresolved field.

This record binds, but does not rewrite:

- immutable phase-zero evidence commit
  `53c7154a3633dfe19dc036cfb5a6250f729a897d`, whose exact application parent
  is `9aae0e669384c5c0c0dda93debc194d6b8dac787`;
- `REPORT.md` SHA-256
  `4ac7c1bb9c67da3ce99ddfe4f96e42799a704bcb5acf89e3fa17cdfda1ef31c8`;
- the complete 383-row `field_registry.csv` SHA-256
  `5ac211f7f21e8a7547ceb4a3db8c37491711e06a5359e9b53aec01ed3115d6f3`;
- `SCI-ALIGN-001_PHASE_ZERO_COORDINATOR_REVIEW_2026-08-01.md`; and
- the D001, D002, and D003 decisions at content commits
  `86434df2cfb5b85d0ccd306150cb428321abdbb9`,
  `10981b29c1870e745b7f3c9cabed3c634a46427f`, and
  `d500e33da1869bc1e20383a49484daddca9e7ea7`.

The frozen census contains 337 LMT and 46 HWPR rows. Twenty of the 22
configured telescope data names occur in the surveyed files;
`TelRaAct`/`TelDecAct` do not, while accepted inputs provide
`SourceRaAct`/`SourceDecAct`. All 31 surveyed HWPR files report
`Header.Toltec.HwpInstalled=0`, none provides the `Data.Hwp.*` schema expected
by the application, and the packed raw matrices do not establish valid-count,
ordering, angle, time, cadence, or support semantics. Telescope time fields
also have unresolved clock/epoch identities. These facts make the full
registry the authoritative frozen discovery/evidence inventory, not 383
approved interpolation contracts.

The owner-provided operational evidence is also binding: accepted Pointing,
Beammap, and large maps establish that current end-to-end astrometric signs,
handedness, and composition are self-consistent, including correct point-source
locations across roughly square-degree maps. The bounded repair must preserve
that behavior and may not introduce a new frame, epoch, sign, handedness, or
projection convention under D004.

## Citlali 4.x `Hold` forensic amendment

The released/reference 4.x sources do not prove that `Hold` was a Boolean
producer field or that a left-continuous state operator was used. Tag
`v4.0.0` (`a398581f48200dcd0cf41e1e09d33b5b7922a06f`), local `v4.x`
(`bbf14b57f22e9a1e30f2c156f66b199d64d52f95`), `origin/v4.x`
(`61dfdc0492ef56bc88e572e295369f5b63f7d91d`), and
`origin/v4.x_gcc13` (`0ac3411bf6b4e2455a34ae3081e1210fa1c91910`)
linearly interpolate the full numeric word with every other telescope field
and later cast the result to Boolean, so any nonzero interpolated value is
true. The later `v4.x` branch heads also force the derived scan/window
condition true outside the configured map box; that spatial exclusion is
separate from the raw word and is not a `Hold` bit meaning.

Repository history nevertheless supplies a bounded candidate: the initial
Citlali commit `1c9f76235b16b75e726e539a5b00b675565a6a1e` says the telescope
is turning when `Hold & 8` is set and implements that predicate. Commit
`8753667f860aa200666a7fa8aa5e804b7d07d86f` temporarily restored it. Commit
`c58a688731dfaf641d9a2a761a32d538a6f7616e` returned to whole-word Boolean
conversion without a recorded scientific rationale. The observed Beammap
values `{0,2,8,10,64,66,72,74}` show that multiple bits exist. This evidence
supports named compatibility hypotheses; it does not establish producer bit
meanings or a transition side.

## ALIGN-P0-D004 — restricted registry and output contract

Questions: `Q06`, `Q07`, `Q08`, `Q09`, `Q10`, `Q11`, `Q12`, `Q14`

Decision: owner-approved for bounded existing-use-only design with explicit
unavailability where producer authority is absent.

### Active registry allowlist

ALIGN may activate only the following canonical fields for the admitted
nonpolarimetric profiles. It preserves their current numerical sign,
handedness, composition, and admitted frame use and performs no coordinate
frame conversion.

1. The native `Data.TelescopeBackend.TelTime` series is the admitted legacy
   telescope bracketing coordinate in seconds. Its epoch, physical event, and
   absolute precision remain unproved. It is a coordinate, not a scalar field
   to interpolate. ALIGN publishes the D001/D002 detector-reference
   `common_time` separately in seconds and retains the native coordinate in
   provenance.

2. These full angular coordinates may use bracketed shortest-arc interpolation
   with period `2*pi`:

   - `ActGalAng` and `ActParAng`;
   - `SourceAz`;
   - `SourceLAct` as canonical internal `TelL`;
   - `SourceRaAct` as canonical internal `TelRa`;
   - `TelAzAct`; and
   - `TelAzDes`.

3. These bounded signed/angular scalars may use ordinary bracketed linear
   interpolation and must not be wrapped as full longitudes:

   - `SourceBAct` as canonical internal `TelB`;
   - `SourceDecAct` as canonical internal `TelDec`;
   - `SourceEl`;
   - `TelAzCor` and `TelAzMap`; and
   - `TelElAct`, `TelElCor`, `TelElDes`, and `TelElMap`.

4. `Data.TelescopeBackend.Hold` is admitted only under the typed raw-word and
   named-view policy below.

5. Native `TelUtc` and `PpsTime` are admitted only as exact diagnostic or
   provenance identities. They are never generic scalar/circular
   interpolation inputs. Existing aligned `TelTime`/`TelUTC` output names may
   survive only as explicit one-way compatibility aliases for `common_time`
   where accepted consumers require them; they are not claims about the
   native clock. `TelLst`, `AcuTime`, `BackendTime`, counters, modes,
   categorical values, other state words, and every other unlisted data field
   remain exact-only inventory or unavailable.

An allowlisted interpolated field is available only when the source exists
with its approved unit and shape, both adjacent native rows are finite and
valid, the target is bracketed without extrapolation, and the bracket does not
cross a D003/OD4 gap. Structural support is at most one adjacent valid native
interval; D005 must preregister the numerical maximum bracket duration before
successor results are inspected. Exact `pi` shortest-arc ambiguity is
unavailable. An unknown sentinel, frame, topology, unit, or validity condition
produces typed unavailability, never a zero/default or generic linear fallback.

### RA/Dec source aliases

`SourceRaAct` and `SourceDecAct` are canonical because they occur in the
surveyed accepted inputs and underpin the demonstrated astrometric behavior.
`TelRaAct` and `TelDecAct` are permitted only as a schema-versioned legacy
source pair for the same canonical internal roles. If both pairs occur, unit,
shape, declared identity, and elementwise values in the same declared
representation must agree exactly. Disagreement, including a merely
wrap-equivalent but nonidentical alternative, fails closed pending separate
authority; container or map insertion order never selects silently.

### Typed `Hold` policy

1. Preserve the complete raw numeric word and its native source identity.
   Before any bit test, require every used value to be finite, nonnegative,
   exactly integral, and losslessly representable by the selected typed word.
   The producer's logical width remains unavailable.

2. Expose `legacy_4x_linear_any_nonzero` only as a named existing-use
   compatibility view reproducing the released-4.x sequence: linearly align
   the numeric word, then test the aligned value for nonzero. This is not a
   producer-authoritative Boolean or turnaround meaning and must not become a
   generic operator for other state fields.

3. Expose `turnaround_candidate_0x08 = ((raw_word & 0x08) != 0)` only as a
   separately named, repository-history-supported candidate. It may be
   evaluated diagnostically but may not silently replace the existing-use scan
   predicate before the D005 preregistered comparison.

4. Preserve every other set bit. In particular, `0x02` and `0x40` remain
   uninterpreted and confer no scan, hardware, validity, flagging, eligibility,
   or exposure meaning. Do not discard them by reducing the stored raw word to
   one bit.

5. Keep the outside-map-box scan/window condition separate from both raw and
   derived `Hold` identities. `Header.M2.Hold` and
   `Header.Map.HoldDuringTurns` are separate exact-only observation/header
   snapshots, not aliases, bit definitions, or authorities for
   `Data.TelescopeBackend.Hold`; the surveyed Citlali path does not use either
   header to construct scans.

D004 does not select left- versus right-continuous state placement, declare
which native bit is producer-authoritative, or authorize a new scan boundary.
Until D005 freezes and evaluates the alternatives, non-native `Hold` is
available only through the explicitly named released-4.x compatibility
adapter for the exact admitted existing-use path. Stronger scientific
scan-turnaround state remains unavailable. Raw words, transitions, and
optional diagnostic views use compact run/exception representation; dense
per-sample provenance or duplicated state products are not required.

### Header, shape, unit, and output contract

- Publish a versioned, digest-bound active registry with canonical identity,
  native source name, actual unit, source dtype/shape, topology/operator,
  availability, frame qualification, support rule, and compatibility status.
- Aligned angular coordinates are radians at ALIGN's interface. Existing
  Pointing/OOF/Beammap tangent-plane arcseconds and Science WCS degrees remain
  governed by their own accepted boundaries. D004 introduces no frame
  conversion.
- State output is the exact raw typed word plus only explicitly named derived
  views; it is never labeled radians. Time/common-axis output is seconds with
  its qualified clock and semantics source; an unavailable epoch/event is not
  silently asserted.
- File/observation headers are not sample-aligned fields. Preserve native
  dtype, identity, and complete scalar/vector shape. Do not generically take
  element zero or broadcast an observation-level vector per sample. A scalar
  consumer may require a scalar source; an unreviewed header remains opaque or
  scientifically unavailable.
- A missing or ambiguous required active field fails the affected product.
  Optional absent fields are explicitly unavailable or `not_applicable`.
  Ordinary mappings/provenance remain implicit or generative, with compact
  transition/exception records and expanded forensic detail only as requested
  under `ALIGN-C001`.

### HWPR disposition

No HWPR field is active in this bounded repair. Absent or disabled HWPR is
`not_applicable` and nonfatal for admitted intensity processing and does not
trim detector support. Packed raw inputs may be retained only as optional
diagnostic evidence; ALIGN does not infer their angle transform, unit, epoch,
counter width, rollover, cadence, valid ordering, support, or offset stage.

Enabled HWPR, a requested nonzero HWPR offset, an aligned HWPR scientific
output, a mode requiring HWPR, or any polarization use fails closed until a
versioned producer schema/ICD, angle/time contract, and separate validation are
approved.

### Proportionality and downstream boundary

Implement the registry as bounded typed setup plus the existing alignment
pass. Do not add a second avoidable full-data pass, a generalized telemetry
framework, or routine per-sample/per-detector/per-pixel field registries,
provenance arrays, response matrices, or covariance products. ALIGN supplies
typed time-aligned pointing coordinates. AST retains authority for its
owner-approved single inverse-TAN response at map center. The repair must show
no repeatable measurable Pointing/Beammap timing regression and preserve
accepted source crossings, centroids, PSFs, and end-to-end astrometric
self-consistency.

## D005 validation obligation

D005 must preregister `ALIGN-D004-HOLD-VALIDATION-001` before inspecting any
successor result. The frozen protocol must compare:

- exact governing-SHA `9aae0e669384c5c0c0dda93debc194d6b8dac787`
  `Hold`-derived state and final scan outputs as the direct old/new baseline;
- released-4.x whole-word linear interpolation followed by nonzero conversion;
- native raw-word nonzero and native `raw_word & 0x08` hypotheses;
- candidate transition-side placements;
- raw per-bit transitions and the separate outside-map-box condition; and
- resulting aligned states, scan counts/windows, first-post-turn sample,
  source crossings, centroids, and per-array major/minor PSFs.

It must freeze input/cohort digests, metrics, numerical limits, and the maximum
allowed telescope bracket duration. If `0x02` or `0x40` correlates materially
with turns, the hypotheses differ materially, no transition side is supported,
or an accepted astrometric/scan boundary would regress, stop for a separate
owner scope amendment. Do not select a predicate or tune a limit after viewing
successor results.

## Explicit non-approvals and remaining authority

This decision does not establish producer meanings for any `Hold` bit, an
enabled-HWPR schema, polarization science, producer-authoritative telescope
time epochs/events/precision, a new equatorial frame/epoch, or scientific
semantics for unallowlisted fields. It does not authorize a new profile or
consumer, off-center AST response, application code, phase one, Unity
evidence, re-audit, production expansion, or finding closure.

`SCI-ALIGN-001-F001`, `F004`, `F005`, `F006`, `F007`, `F008`, `F009`,
`F010`, `F012`, and `F014` remain open pending complete implementation, D005,
local validation, exact-repair-SHA human evidence, and fresh re-audit.

`ALIGN-P0-D005` is the sole remaining substantive phase-zero owner decision.
Until D005 is recorded and the coordinator explicitly advances the repair,
`phase_one_authorization` remains `none`; application edits, Unity evidence,
and re-audit remain unauthorized.
