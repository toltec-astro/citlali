# SCI-CAL v0.1 Rationale r0.3 Scientific/Engineering Consistency Report

Status: complete for the rationale r0.3 correction pass

Date: 2026-08-16

The engineering contract remains unchanged and normative for formal
conformance. Rationale r0.3 changes scientific explanation and ownership
clarity without changing requirements, assumptions, edge predictions, or
machine states.

| Consistency axis | r0.3 result |
| --- | --- |
| Ownership | Consistent under producer--transformer--consumer: Beammap/source APT owns source flxscale meaning; TolProj owns target/source association and approved child transformation lineage; SCI-CAL applies selected child flxscale and target atmosphere once; MAP/FLT owns realized response. |
| Factor direction | Consistent: the abstract generative response is R in input units per mJy/beam and selected flxscale is R inverse. R is explicitly not the APT responsivity field. |
| Atmosphere orientation | Consistent: both main text and Appendix A interpolate neutral H first, then define (T,C) from the declared transmission-or-correction orientation. No numerical orientation is inferred. |
| Units and output plane | Consistent: the intended result is top-of-atmosphere, point-source-equivalent, beam-peak-normalized amplitude; literal mJy/beam peak additionally depends on realized downstream response. Unsupported surface-brightness, temperature, extended-source, and integrated-photometry meanings remain excluded. |
| Ordering | Consistent: fixed per-detector scalar commutation is distinguished from detector mixing and sample-dependent atmosphere. The exact pipeline order remains open decision Q02. |
| Claim layers | Consistent: conformance, atmosphere-representation fidelity, observational performance, and production readiness remain separate. The exact engineering state science-qualification-eligible is retained only in the formal appendix; main text uses criteria for validated science use. |
| Version axes | Consistent: v0.1 identifies the governed scientific contract; r0.3 identifies this science-rationale revision. No prose revision is represented as a contract-version change. |

## Unresolved consistency dependencies

- Q01 blocks complete physical units and interpretation of ordinary xs.
- Q02 blocks a unique governed pipeline order.
- Q03--Q05 block complete scientific derivation and broadband transfer
  meaning.
- Q06 blocks contract-supported numerical calibration and associated
  atmosphere-evaluation claims.
- Q07--Q09 block policy rationale, total uncertainty, and achieved validation
  claims.

No unresolved item was converted into a factual claim. No numerical
calibration, representation-fidelity, observational-performance, or
production-readiness validation was performed in this pass.
