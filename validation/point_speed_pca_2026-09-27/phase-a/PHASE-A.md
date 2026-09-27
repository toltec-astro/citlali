# Phase A: measured unchanged POINT latency

Scope: POINT 152430/0/2, a2000 networks 11/12, Apple M4 Pro, 14 cores, 48 GiB RAM. Two upstream network workers, serial feedback segments, one thread per numerical library. Fresh process and scientific state each time; filesystem cache not flushed, preflight hashing may warm inputs. All 14 trajectories / 78 feedback calls passed strict unchanged-science or exact-prefix gates. No science replacement.
| Rank | N | Prefix readiness median [range], s (3 runs) | Actual terminating analysis / exit, s |
|---|---:|---:|---:|
| 10 | 0 | 33.07 [31.78, 36.89] | 33.98 / 34.12 (single) |
| 10 | 1 | 42.27 [40.36, 46.35] | 41.57 / 41.68 (single) |
| 10 | 3 | 59.76 [57.74, 64.35] | 64.28 / 64.42 (single) |
| 10 | 5 | 77.83 [75.48, 82.83] | 84.36 / 84.46 (single) |
| 10 | 10 | 123.58 [119.24, 128.96] | 123.64 / 123.75 (medians) |
| 15 | 0 | 35.92 [35.11, 36.89] | 36.19 / 36.27 (single) |
| 15 | 1 | 46.53 [44.83, 47.35] | 47.44 / 47.59 (single) |
| 15 | 3 | 67.33 [64.87, 67.74] | 69.55 / 69.69 (single) |
| 15 | 5 | 88.12 [84.88, 88.66] | 95.98 / 96.10 (single) |
| 15 | 10 | 140.37 [135.00, 141.05] | 140.46 / 140.59 (medians) |
READY files contain actual map, measurements and current accepted/failed status and are atomically published. Parent polling independently recorded caller detection. Internal timestamps and process exit are separately retained. All shorter commands execute exactly N applications and one terminal endpoint; the saved m_N is unapplied.

## Leading bottleneck findings before alternative implementation

Rank 10, predetermined first T10 run: feedback native wall totals 79.55 s; PTC fits 8.75 s inclusive, of which initialization 4.24 s and eigendecomposition 3.11 s. Native adapter time not attributed to segment work, MAP or final serialization/verification totals 46.49 s. J ranges 2–85 over network/segments/cycles. Nested fit/initialization/decomposition times must not be added. The large adapter residual prevents identifying PCA as the leading overall opportunity from solver cost alone.

Rank 15, predetermined first T10 run: feedback native wall totals 80.95 s; PTC fits 9.83 s inclusive, of which initialization 4.23 s and eigendecomposition 3.09 s. Native adapter time not attributed to segment work, MAP or final serialization/verification totals 46.57 s. J ranges 2–80 over network/segments/cycles. Nested fit/initialization/decomposition times must not be added. The large adapter residual prevents identifying PCA as the leading overall opportunity from solver cost alone.

The upstream per-network trace separately reports ingress, RTC stages, CAL, PTC and publication. Network intervals overlap and are not summed into end-to-end wall time. Process CPU and sampled aggregate process-tree RSS are in the run table. Python/native startup, synchronous validation and cleanup remain on the measured path; no endpoint stage was omitted. Full-instrument operational latency and Unity performance remain unmeasured. No operational deadline or automatic stopping criterion has been invented.

Phase A timing gate is complete. The authorized next step is B1/B2 implementation and matched-rank comparison. The final report will add scientific prefix/response comparisons from the retained controls and directly terminated endpoint scales.
