# SCI-ALIGN-001 acquisition-boundary event-semantics audit

Date: 2026-08-08

## Stopping result

**STOP — essential producer evidence is unavailable.** The physical event
represented by a detector integration row cannot be identified from the
retained evidence. The accepted producer clarification says that the FPGA
source is not presently available, and no authoritative event-level
specification binds integration accumulation, counter capture, PPS ISR update,
UDP assembly, and NetCDF row insertion. This is the required stopping result,
not an authorization to retrieve source, contact a producer, run a reduction,
or design a correction.

No timing correction is proposed or authorized. The accepted modal labels
remain descriptive only.

## Frozen identity and inspected evidence

- entry commit: `92cfa670a33255250895d68aaf26e8b01aa057bd`;
- parent: `77c8a1a71cc79eb3aeacbd596c42b6dae33b3aa4`;
- tree: `908825af674e3ea19c03cbb54441680dd4d6ad12`;
- frozen group: `roach-t0:44cf69da97d473965ef6`;
- maps: 148670, 150819, and 151126;
- networks: 0, 1, 2, 3, 4, 5, 7, 8, 9, 11, 12.

All 11 retained Beammap 148670 raw detector files were inspected directly and
their full-file SHA-256 values were reverified. `Is`, `Qs`, `Ts`, and
`RecvTime` share the NetCDF `time` dimension in every file. PacketCount and
ClockCount have zero increment mismatches. `RecvTime` is strictly increasing;
its per-network median minus the reconstructed detector coordinate spans
0.175953--0.239611 ms, with rare maxima up to
40.210962 ms. This is host-delivery/packaging evidence only:
Citlali does not use `RecvTime` to construct detector time, and its physical
event and clock remain undocumented.

For all three maps, the checksum-bound compact evidence retains 33/33
map/network records with one-to-one delivered raw linkage, same-row-only
PpsCount/PpsTime transition association, no association-class changes, and no
reported variable PpsCount/PpsTime transition latency. The exact campaign
per-map raw-counter artifact files are represented locally only by
owner-returned paths and SHA-256 digests; the accepted aggregate projection is
local. Direct detector raw is local only for 148670. This audit did not retrieve
any missing source file.

## Event mapping

The strongest supported mapping is:

1. Producer authority says each UDP packet carries T0, the incremental PPS
   counter, and the internal-clock counter. PPS does not restart the detector
   integration cadence.
2. The delivered NetCDF places `Is[n,*]`, `Qs[n,*]`, and the six `Ts[n,*]`
   fields on one row. This proves delivered packaging, not FPGA atomicity.
3. Citlali reconstructs the native coordinate from the first-row integer
   anchor plus PpsCount and `(ClockCount-PpsTime)/FpgaFreq`; it does not use
   RecvTime.
4. The same delivered detector row is carried through native-time
   reconstruction and nearest-slot assignment without a demonstrated
   permutation.
5. Which integration event the row represents—start, end, effective centroid,
   counter-capture instant, or another event—remains unresolved upstream of
   the delivered pair.

`semantic_boundary_matrix.csv` records the field-by-field boundary.

## Integer- and half-cadence test

A whole detector-row reassociation changes time by 8.192 ms, exactly two
4.096-ms half-step labels. It therefore preserves half-step parity. The frozen
pair modes include odd changes, 148670->150819 = -3 and 150819->151126 = +1.
No common additive offset changes those pair parities. Consequently a
whole-row integer reassociation **alone** cannot reproduce the exact three-map
modal label system. It may still contribute an even component and is not
excluded upstream.

The transitive half-step states
`{148670: 0, 150819: -3, 151126: -2}` reproduce all three modal pair labels,
so a half-cadence association or map-varying start/end/centroid convention is
descriptively compatible. It is not physically identified: the half lattice
nests the full lattice, only three maps are present, and producer event
semantics are unavailable.

The 1,437 retained PpsTime increment-anomaly rows each perturb the delivered
timestamp step by at most one 256-MHz tick (3.90625 ns), while anomaly-free
controls retain the timing bands. They are strongly disfavored as the primary
cause and do not authorize a repair or mask.

## Hypothesis disposition

Falsified or strongly disfavored within this boundary:

- a Citlali delivered-row permutation or off-by-one;
- a change in delivered same-row versus adjacent-row PpsCount/PpsTime class;
- whole-row integer reassociation as the sole exact generator of all labels;
- retained PpsTime increment anomalies as the primary cause;
- stable native detector-frame phase alone;
- host RecvTime as the coordinate consumed by Citlali.

Surviving without preference:

- acquisition-hardware integration phase or capture state;
- upstream non-atomic or adjacent integration/metadata packaging;
- map-varying timestamp start/end/effective-centroid semantics;
- map fitting.

`hypothesis_disposition.csv` gives the evidence and limit for each statement.

## Evidence limits and smallest next step

Unavailable essential evidence includes the exact FPGA/packetizer source or an
equivalent authoritative event specification, direct local raw detector files
for 150819 and 151126, and an independent physical integration-event marker.
The raw absence is not repaired by using telescope data, map fitting, or a new
reduction.

The smallest next step is an owner decision on owner-mediated acquisition of
the exact producer FPGA/packetizer source revision or an authoritative
event-level specification. That later material must bind integration
accumulation, counter capture, PPS ISR update, UDP assembly, and NetCDF row
insertion. This audit stops before retrieval, execution, external contact,
scientific policy choice, or correction.

The orthogonal 20-ms telescope-file ingress boundary was not evaluated or
reinterpreted. `SCI_TEL_INPUT_001_HANDOFF.md` contains only the bounded facts
relevant to that separate audit.
