# SCI-ALIGN-001 same-T0 cadence-lattice evidence

Date: 2026-08-08

This note records a local, read-only comparison of the three successful
enhanced 3C273 maps in frozen ROACH-initialization group
`roach-t0:44cf69da97d473965ef6`. It is a bounded diagnostic of delivered compact
evidence, not a timing-correction prescription or a claim about all TolTEC
observations.

## Frozen identity

The protocol was written before interpreting the pairwise results. It fixes
the 8.192-ms detector cadence, its 4.096-ms half cadence, pair order, formulas,
all three maps, and the exact common network set. It permits no
result-dependent exclusion:

- ObsNum 148670: `map:912d0ccf8b3539501f6c`;
- ObsNum 150819: `map:1971b4dfddbc99932afb`;
- ObsNum 151126: `map:d5fec4dcd0f16b424fb6`;
- networks: 0, 1, 2, 3, 4, 5, 7, 8, 9, 11, and 12.

Identity digests are:

- aggregate `SHA256SUMS` SHA-256:
  `bb6d821d9103930e94d425b4531a63c4f4b65fcb1aa5f7000afcb416a3e92388`;
- frozen protocol SHA-256:
  `4ba36fada3ae0dde2df93c8850734a2da3603b726937891eddc6722f6a4c4b7a`;
- diagnostic tool SHA-256:
  `9c464ff2bb489fadcf43fc8ee7f873bb7a967d5ed1d82d59a42cb1c46fdae93a`;
- result `SHA256SUMS` SHA-256:
  `ab77361e5f330e6ace5bc76ca5827df7d65bfab7bd6da137421cfc8a65a5f50c`.

`same_t0_cadence_lattice_protocol.json` contains the seven individual input
digests. `same_t0_cadence_lattice_result_2026-08-08/input_identity.json`
records the resolved local paths, sizes, roles, and verified digests. The
diagnostic independently requires successful enhanced-map status, primary
role, the common T0-vector group, cadence identity, stable map/network keys,
and exact agreement among the retained timing/phase tables before writing an
output directory.

## Comparison pre-specified before result inspection

For each stable network and ordered map pair, the diagnostic computes:

- measured timing change;
- native detector-frame phase change, wrapped to one 8.192-ms cadence;
- change in native-to-assigned-slot residual;
- the fixed minus-one-slot prediction, `-delta_slot`;
- its residual, `delta_timing + delta_slot`;
- nearest 8.192-ms and 4.096-ms lattice indices and remainders;
- delivered PPS/PpsTime transition-association class;
- PPS-time increment-anomaly class.

Timing standard errors are combined diagonally only. Cross-map and
cross-network covariance is unavailable. The comparison contains only three
maps, and the half-cadence lattice nests the full-cadence lattice. Therefore
the envelope counts and RMS values below are descriptive checks, not formal
model selection or independent-record inference.

## Results

All 33 map/network records and all 33 pairwise records were retained.

| Obs A | Obs B | median timing change (ms) | median absolute native-phase change (ms) | median fixed -1-slot residual (ms) | modal half step | modal support | half-lattice RMS (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 148670 | 150819 | -14.797973 | 0.017701 | -12.237806 | -3 | 11/11 | 1.000774 |
| 148670 | 151126 | -8.444458 | 0.023753 | -8.270742 | -2 | 11/11 | 0.657737 |
| 150819 | 151126 | +6.843495 | 0.014610 | +3.248104 | +1 | 10/11 | 0.908390 |

The modal nearest-half-cadence labels are transitive. Up to a common additive
offset, their unique transitive assignment is represented by taking ObsNum
148670 as zero:

- 148670: 0 half steps;
- 150819: -3 half steps;
- 151126: -2 half steps.

Those assigned labels reproduce all three pair modes: -3, -2, and +1. The
first two modes have 11/11 network support; the last has 10/11, with network
12 assigned to the neighboring zero index. Across all pairwise records, the
half-cadence remainder RMS is 0.868 ms, compared with 2.733 ms on the
full-cadence lattice. Only 23/33 records lie within the 1.96 diagonal-SE
half-cadence envelope. Because covariance is unavailable and the half-cadence
lattice contains every full-cadence point, the lower RMS is descriptive and
does not by itself prefer a physical half-cadence model.

Native phase is nearly unchanged compared with the timing displacement: the
overall median absolute pairwise native-phase change is 0.0167 ms, while the
median absolute timing change is 8.477 ms. The fixed minus-one-slot residual
RMS is 8.669 ms, so measured native-to-assigned-slot residual alone does not
predict the map-to-map timing bands.

All 33 delivered PPS/PpsTime transition associations are `same_row_only`,
with no association-class change and no retained variable-latency flag.
Increment-anomaly classes do change: 24 records are anomaly-free, eight are
mixed isolated/consecutive, and one is consecutive-only. Each pair has six
networks that are anomaly-free in both maps. The fixed anomaly-free
intersection across all three maps is five networks: 0, 4, 7, 11, and 12.
Four of those five retain the modal +1 label in the 150819-to-151126 pair;
network 12 has the neighboring zero label. The persistence in these controls
strongly disfavors the retained increment anomalies as the primary cause, but
does not make the records statistically independent.

## Falsified or strongly disfavored within this scope

- A stable network-dependent native detector-frame phase, by itself, cannot
  explain the several-millisecond map-to-map timing changes in this same-T0
  group.
- The fixed minus-one-slot relation pre-specified before result inspection,
  using the measured native-to-assigned-slot residual, is not sufficient; its
  residual is not centered near zero.
- Delivered PPS-time increment anomalies are strongly disfavored as the
  primary cause because anomaly-free controls retain the same common bands.
- A change in the delivered same-row versus adjacent-row PPS/PpsTime
  transition class cannot explain these three maps; that class is unchanged.
- A single full-cadence-only lattice is descriptively poorer than the
  half-cadence lattice for the pairs involving ObsNum 150819; this is not
  formal model selection.

## Retained hypotheses and limits

The unique transitive assignment of pairwise modal nearest-half-cadence labels,
up to a common additive offset, is compatible with and favors a common
map-level acquisition or timestamp-semantics state after accounting for the
measured network slot residual. It does not identify such a state or its
physical origin. Non-hardware alternatives, including producer/software
association and map-varying timestamp semantics, remain in scope. In
particular:

- Stage-A lineage begins at the delivered raw detector-data/timestamp pair and
  cannot exclude FPGA-level metadata-to-integration association;
- same-row delivered transition pairing does not prove atomic association of
  detector data, internal-clock counter, and PPS counter;
- a common start/end/centroid convention cancels in these pairwise
  differences, while a map-varying convention or half-step association state
  remains possible;
- the compact aggregate cannot prove the physical event represented by each
  timestamp or counter field;
- the result neither proves a physical clock error nor supports a fixed clock
  correction, a universal half-cadence correction, or extrapolation beyond
  this frozen three-map T0 group.

## Smallest next owner decision

Decide whether to authorize one bounded, read-only acquisition-boundary event
semantics audit for these exact three maps and networks. The audit should test
whether detector samples, internal-clock counter, PPS counter, and their
delivered timestamp can change association by an integer or half cadence
between initializations while preserving the already frozen identities. It
should test hardware and non-hardware explanations without privileging either:
FPGA metadata-to-integration association if evidence exists, host/producer
counter association, timestamp construction, and map-varying start/end/
centroid semantics. It should begin with existing raw/counter evidence and
producer documentation; it should not launch a reduction, modify Citlali, or
fit a correction. If the required acquisition-level artifact or source is
unavailable, that absence should be the explicit stopping result.

The orthogonal bounded 20-ms TolTECA/telescope-ingress audit remains pending
owner authorization. It should retain its separate raw-to-recomputed row,
timestamp-semantics, and provenance questions and is neither launched nor
resolved by this note.
