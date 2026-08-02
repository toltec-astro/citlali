# SCI-ALIGN-001 ALIGN-P0-D005 sky-domain amendment

Date: 2026-08-02

Verdict: **OWNER RETURN REQUIRED — PHASE ONE UNAUTHORIZED**

## Scope and identity

- Repair branch `codex/repair-sci-align-001`; additive task base `5a0d64b8f1b9b246b1b5d575c548269823203d22`.
- Governing application `9aae0e669384c5c0c0dda93debc194d6b8dac787`; frozen phase-zero evidence `53c7154a3633dfe19dc036cfb5a6250f729a897d`.
- Frozen D005 `SHA256SUMS` `149ef430af3223562d9e69b7224703b831f6f56629b2f3c513bf44c40a567bbb` was verified and not rewritten.
- Read-only coordination authority is the immutable `6785152c2a2d4113c9ba89073de00cb454aa70c4` snapshot, verified clean by the frozen parent D005 package. The live coordination checkout was deliberately excluded; no live HEAD, status, or working-tree bytes enter this package.
- This package reads every canonical selected Pointing/Beammap timing row and records compact digests/strata. It does not execute Citlali, TolProj, AST changes, SCI-MAP, or Unity; inspect a successor; edit application code or sibling repositories; or authorize phase one.

## Three timing quantities are not interchangeable

1. `raw_timestamp_minus_candidate_slot` is the signed D001 reconstructed timestamp residual after the exactly-once requested offset. The strict half-cell rule is an engineering unique-slot invariant.
2. `candidate_minus_governing_assigned_time` is the differential time that can move a sample on the governing telescope trajectory.
3. `assigned_time_minus_physical_integration_centroid` is absolute physical timing. It is unavailable for every selected row because no producer authority identifies start, end, centroid, capture, or packet event.

The Pointing 152389 `34.062668 us` margin is only distance to the half-cell decision boundary. It is not sky-placement accuracy and is not converted into an acceptance tolerance.

## Measured differential sky result

The six canonical observations contain **4,645,586** native detector rows. All **4,645,476** governing-supported ordinary rows retain the same integer slot and assigned time, so their candidate-minus-governing AltAz tangent coordinates and signed along/cross components are exactly **0 arcsec**. All **110** union-edge rows have no governing assigned-time baseline and are `unavailable`, not zero. The full-slot reassignment rate is exactly `0/4645476`.

For each requested time the diagnostic mirrors governing order: periodic-fix and interpolate `TelAzAct`, `TelElAct`, `TelAzCor`, `TelElCor`, `SourceAz`, and `SourceEl`, then form the AltAz tangent coordinate. Velocity and acceleration use the digest-bound symmetric half-cadence trajectory estimator. Hold raw words, left/right views, bit `0x08`, outside-map state, and composite compatibility state remain separate; none is called a physical turn.

| Obs | Mode | Native rows | Paired ordinary | Edge N/A | Speed p50 | Speed p95 | Speed max | +half radial p50 | +half radial p99 | +full radial p50 | +full radial p99 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 148669 | point | 85,082 | 85,063 | 19 | 30.641924 | 101.953907 | 183.754242 | 0.127484 | 0.635857 | 0.252437 | 1.268554 |
| 148670 | beammap | 4,220,705 | 4,220,689 | 16 | 40.656612 | 70.325765 | 164.011607 | 0.165653 | 0.411743 | 0.332849 | 0.818045 |
| 148671 | point | 83,614 | 83,600 | 14 | 27.520567 | 112.901768 | 212.128687 | 0.114727 | 0.627597 | 0.226773 | 1.253961 |
| 152389 | point | 84,689 | 84,667 | 22 | 30.198921 | 102.908635 | 173.655499 | 0.124651 | 0.559374 | 0.246367 | 1.121566 |
| 152391 | point | 85,661 | 85,646 | 15 | 26.444257 | 104.225681 | 193.081457 | 0.109462 | 0.607528 | 0.216267 | 1.197650 |
| 152393 | point | 85,835 | 85,811 | 24 | 30.027044 | 96.458213 | 180.034649 | 0.124088 | 0.530100 | 0.244906 | 1.055254 |

Speeds are digest-bound symmetric finite-difference estimates from the actual tangent trajectory, and the shift columns are exact curved-trajectory sensitivity calculations, row-weighted over selected detector rows. They are descriptive, not producer-authoritative kinematics or limits.

## Half/full-slot dimensional references

At 1x, half slot is `4.096 ms` and full slot is `8.192 ms`:

| Reference speed | Half slot | Full slot | Half in 1/2 arcsec pixels | Full in 1/2 arcsec pixels |
| ---: | ---: | ---: | ---: | ---: |
| 50 arcsec/s | 0.2048 arcsec | 0.4096 arcsec | 0.2048 / 0.1024 | 0.4096 / 0.2048 |
| 100 arcsec/s | 0.4096 arcsec | 0.8192 arcsec | 0.4096 / 0.2048 | 0.8192 / 0.4096 |
| 200 arcsec/s | 0.8192 arcsec | 1.6384 arcsec | 0.8192 / 0.4096 | 1.6384 / 0.8192 |

These are constant-speed dimensional examples only. Actual comparisons use the trajectory table above and the full populated-stratum artifact.

## Physical timestamp authority

All selected detector files expose `FpgaFreq=256000000 Hz`, `AccumLen=2097152`, and `SampleFreq=122.0703125 Hz`, proving the 8.192-ms cadence relation but not the integration event or a contiguous 8.192-ms exposure duration. The timestamp/receive variables contain no start/end/centroid, bounds, exposure, or cell-method authority. There are 2 producer compile identities and no local versioned ICD for either. `RecvTime - reconstructed_time` is nonnegative in the selected cohort (median `219.107 us`), but its clock and row association are undocumented, so it cannot select end/completion semantics.

Counterfactual start/centroid/end formulas are recorded in `physical_timestamp_scenarios.csv`; they are not acceptance evidence. Absolute assigned-time-to-integration-centroid error and absolute physical sky-placement correctness remain unresolved even though the differential slot result is exact.

## Science-facing gates and ownership

- Sample coordinate, systematic/scatter, 1/2-arcsec pixel sensitivity, source crossing, centroid, major/minor FWHM, ellipticity, full-slot rate, and wing metrics are preregistered in `preregistration_protocol.json`.
- Current Point/Beammap policy remains exact complete-product/TOD equality for unaffected behavior. The historical `0.0001 arcsec` OG/refactor Beammap profile is not promoted into this successor gate.
- ALIGN owns row identity, offsets, assigned time, aligned support/validity, and aligned telescope fields. AST retains the frozen sky transform. SCI-MAP owns WCS projection, JINC/naive coefficients, map weights, coverage, and accumulation.
- Exact map-level Pointing/Beammap comparison may be a bounded downstream sentinel. No gridding or mapmaking implementation is in this repair.
- No successor was run, so source-crossing, centroid, PSF, ellipticity, and map products are preregistered downstream gates rather than claimed executed non-degradation results.

## Owner return

Engineering invariants, measured angular equivalence, and unresolved physical timestamp authority are separated in `owner_decision_brief.json`. The owner must answer SKY-Q1 through SKY-Q5, and the frozen parent D005-Q1 through D005-Q8 remain unchanged and incorporated by reference. In particular, no nonzero angular tolerance can be derived from cadence, half-cell margin, or the measured residual maximum.
