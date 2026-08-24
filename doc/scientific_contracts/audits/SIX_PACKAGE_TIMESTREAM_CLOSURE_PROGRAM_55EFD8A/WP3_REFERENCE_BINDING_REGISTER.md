# WP-3 Processed-Timestream Reference-Binding Register

Prepared: `2026-08-24`

Status: mechanical realization of approved `WP3-OWNER-D001--D008` and
`WP2-FOLLOWUP-D011`; clean-room re-audit pending

Scope: ordinary processed-timestream route through CAL and into PTC. MAP,
coaddition, implementation conformity, observation-instance validation, and
achieved-performance claims are excluded.

## Governing Rule

> **Pass the data needed by the next stage; reference the history rather than
> retelling it.**

Each stage records only the new scientific facts it owns. A downstream product
keeps resolvable parent references and does not copy the full contents of
upstream products, sidecars, manifests, APTs, or telescope records.

This register binds authorities and reference behavior. It is not a runtime
payload schema and creates no observation data product.

## Static Authority And Runtime Realization

| Role | Static authority | Observation-instance realization |
| --- | --- | --- |
| Native readout and paired \(x/r\) meaning | `producer_interfaces/v0.1/TUNE_READOUT_NATIVE_XR_PRODUCER_INTERFACE.md`; approved `WP2-FOLLOWUP-D011` disposition | Exact Tune/readout record and detector/tone occurrence association. The v0.1/r0.1 interface bytes remain candidate authority pending exact artifact approval. |
| ALIGN occurrence identity and physical acquisition | Frozen SCI-ALIGN v0.1/r0.3 plus the WP-2 exposure boundary | ALIGN product or sidecar referenced by occurrence key. |
| RTC conditioned pair, output grid, decisions, support, diagnostics, and response state | Frozen SCI-RTC v0.1/r0.12 plus the approved WP-2 RTC-to-AST boundary | RTC product or compact RTC sidecar referenced by product identity and detector/sample key. |
| RTC-grid pointing and coordinate realization | Frozen SCI-AST v0.1/r0.3 plus both approved WP-2 coordinate boundaries | AST product or sidecar referencing the RTC parent, telescope/observing inputs, and matched APT. |
| Detector geometry and selected absolute factor | Frozen SCI-BEAM authority, approved WP-2 geometry boundary, and `WP3-OWNER-D003/D005` | Exact SHA-bound matched APT plus artifact-local detector row or UID. AST consumes coordinate fields; CAL consumes the selected finite nonzero `flxscale`. |
| Beammap, source, calibrator, matching, and transformation ancestry | Matched-APT manifest and referenced SCI-BEAM products under `WP3-OWNER-D005` | Resolved only when a dependent calibration, response, or validation role requires it; not copied into CAL. |
| WVR opacity and telescope state | Frozen SCI-CAL atmosphere meaning and `WP3-OWNER-D004` | Existing observation telescope records, with exact observation/time association. |
| Atmosphere operator and passband | Frozen SCI-CAL v0.1/r0.5-r0.4 content-bound authority | No separate observation payload. CAL references the frozen operator/passband identity while evaluating the observation inputs. |
| Applied calibration and CAL classification | Frozen SCI-CAL authority and `WP3-OWNER-D002--D006` | CAL product or compact CAL sidecar referencing RTC, matched APT, and telescope inputs. |
| PTC science-signal parent | Frozen SCI-PTC v0.1/r0.5 plus `WP3-OWNER-D006` | \(Y^{\rm CAL}_{dn}\), detector/sample identity, required flags/validity, and references to the CAL and RTC parents. |

## Minimum Boundary References

| Boundary | Data passed directly | References retained | Data not repeated |
| --- | --- | --- | --- |
| ALIGN to RTC | Aligned paired \(x/r\) occurrences and required validity | Native Tune/readout and ALIGN parent | Native acquisition history beyond what ALIGN itself owns |
| RTC to AST/CAL | Conditioned RTC product, detector/sample identity, required validity | RTC sidecar and ALIGN parent | RTC decision history, dense response, and exposure history |
| Matched APT to AST/CAL | Exact detector-row binding and fields each consumer uses | APT artifact identity, SHA, and manifest | Beammap fits, matching history, and source/calibrator ancestry |
| CAL to PTC | \(Y^{\rm CAL}_{dn}\), detector/sample identity, and required flags/validity | CAL and RTC products or sidecars | APT, Beammap, WVR, operator, exposure, and full provenance contents |

CAL produces no calibrated \(r\) quantity. The paired \(r^{\rm RTC}\) remains
reachable through RTC parentage; SCI-PTC owns any auxiliary diagnostic or
learning use. CAL failure does not authorize an uncalibrated RTC-\(x\) fallback
on the ordinary PTC route.

## Reference And Failure Rules

1. A reference identifies its owning product or artifact and the local key
   needed by the consumer. An existing content digest or manifest remains the
   integrity authority; it is not copied into every sample.
2. Unrecoverable RTC-grid pointing is an observation-level hard stop under the
   approved WP-2 authority. No CAL, PTC, science, or ML handoff is produced.
3. Missing or invalid matched-APT integrity, detector-row binding, or finite
   nonzero selected `flxscale` blocks calibration at the affected scope.
4. Missing or invalid required WVR, telescope state, time association, or
   frozen operator support blocks CAL at the affected sample scope as already
   governed by SCI-CAL and D004.
5. A typed unavailable response or uncertainty remains claim-local unless an
   exact requested operation requires it. It is never replaced with zero or a
   reconstructed default.
6. `engineering-only` remains a CAL-owned classification. It does not itself
   prohibit PTC mathematics; each named scientific use owns admission.
7. When a required owner fact is absent, correct the owning stage, leave the
   dependent use unavailable, or return the bounded omission for owner review.
   Do not construct a generic CAL provenance bundle.

## Verification State

The references above correspond to existing frozen package authority,
approved WP-2 boundaries, the frozen CAL packet, the frozen PTC successor, and
the approved WP-3 decisions. The Tune/readout interface disposition is
approved, but exact approval of the current v0.1/r0.1 candidate artifact bytes
remains pending and must not be inferred from this register.

This register establishes neither observation-instance validity nor closure of
`F-016` or `F-017`. Final finding disposition belongs to the unchanged
clean-room re-audit under WP-7.
