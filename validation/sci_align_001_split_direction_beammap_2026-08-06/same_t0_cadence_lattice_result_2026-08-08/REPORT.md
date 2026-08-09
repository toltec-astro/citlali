# SCI-ALIGN-001 same-T0 cadence-lattice comparison

## Frozen identity

- Group: `roach-t0:44cf69da97d473965ef6`
- Observations: 148670, 150819, 151126
- Networks: 0, 1, 2, 3, 4, 5, 7, 8, 9, 11, 12
- Protocol SHA-256: `4ba36fada3ae0dde2df93c8850734a2da3603b726937891eddc6722f6a4c4b7a`
- Joined records: 33
- Pairwise records: 33

## Descriptive results

| Obs A | Obs B | median delta timing (ms) | median abs phase delta (ms) | median -1-slot residual (ms) | full-lattice RMS (ms) | half-lattice RMS (ms) | modal half step | modal support | full within 1.96 SE | half within 1.96 SE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 148670 | 150819 | -14.797973 | 0.017701 | -12.237806 | 3.334107 | 1.000774 | -3 | 11/11 | 0/11 | 8/11 |
| 148670 | 151126 | -8.444458 | 0.023753 | -8.270742 | 0.657737 | 0.657737 | -2 | 11/11 | 8/11 | 8/11 |
| 150819 | 151126 | 6.843495 | 0.014610 | 3.248104 | 3.295119 | 0.908390 | +1 | 10/11 | 0/11 | 7/11 |

## Transitive assignment of modal nearest-half-cadence labels

Reference observation: 148670.
Unique transitive assignment, up to a common additive offset, in 4.096-ms label units: `{"148670": 0, "150819": -3, "151126": -2}`.
All pair modes unique: true.
Pair modes transitive: true.

This label assignment is compatible with a common map-level acquisition or timestamp-semantics state; it does not identify one.

Association classes: `{"same_row_only": 33}`.

Increment-anomaly classes: `{"consecutive_only": 1, "mixed_isolated_and_consecutive": 8, "none": 24}`.

Delivered association-class changes across pairs: 0.
Increment-anomaly-class changes across pairs: 12.
Records with observed variable metadata latency: 0.

Each map pair has six networks that are anomaly-free in both maps. The fixed three-map intersection is networks 0, 4, 7, 11, and 12; four of those five retain the +1 label for 150819 to 151126, while network 12 has label 0.

Only 23/33 pairwise records lie within the 1.96 diagonal-SE half-cadence envelope. Covariance is unavailable, only three maps are in scope, and the half-cadence lattice nests the full-cadence lattice. The lower half-lattice RMS is therefore descriptive rather than formal model selection.

## Interpretation boundary

This is a descriptive comparison of delivered compact evidence. Diagonal timing SE is not cross-map covariance. Same-row delivered PPS/PpsTime pairing does not prove FPGA metadata-to-integration association. Hardware and non-hardware producer, timestamp-construction, and map-varying event-semantics explanations remain open. A common start/end/centroid convention cancels in pairwise differences. The orthogonal 20-ms TolTECA/telescope-ingress audit remains pending. No result authorizes a timing correction.
