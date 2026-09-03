# SCI-ALIGN v0.1 Stage B r0.3 Semantic Change Map

Status: implementation-blind traceability for the bounded r0.3 author draft.
No stable requirement or prediction identifier is added, removed, or
renumbered.

| r0.3 repair | Canonical authority amended | Stable requirements | Stable predictions | Semantic result |
| --- | --- | --- | --- | --- |
| Cross-package notation | Notation; EQ-002, 003, 018, 020; rationale figures and provenance examples | REQ-003, 015, 030, 034, 045-047 | PRED-009, 014, 019, 023-024, 026 | `s` is stable ALIGN slot, `j` local row, `n` RTC sample, `d` detector occurrence/identity, `p` map pixel, and `x/r` paired readout coordinates. ALIGN no longer uses `p` for a detector. |
| Exposure taxonomy | EQ-018; notation; definitions; slot-anatomy figure; transfer boundary | REQ-027, 030, 034, 046, 050, 054 | PRED-007, 014, 024, 026 | `e^acq_sd` is physical acquired integration independent of payload validity; `e^vo_sd` is the valid-original subset. Original-invalid may have nonzero physical exposure; synthesized and missing add zero. Later guard/use facts do not rewrite acquisition. |
| Exact occurrence time | EQ-002; observing-state definition; shared boundary | REQ-015, 020-022, 046-048 | PRED-022, 025 | `t_s` is exact grid/time identity, not an interpolated observing-state scalar. Boresight/elevation/azimuth and registered state are evaluated or mapped at `t_s`; producer timestamps remain native metadata only. |
| Shared profile typography | Boundary, rationale, definitions, requirements, predictions | REQ-045 | PRED-023 | Profile identity is shown exactly as `SCI-ALIGN_TO_SCI-AST v0.1/r0.1`; digest remains manifest-only. |
| Metadata | Both LaTeX covers and PDF metadata | none | none | Revision/date/title and owner/author are consistent; deterministic creation date is documented in QA. |

EQ-018 retains its stable equation ID while now defining both exposure facts.
All 20 equation identities, 55 requirement identities, and 26 prediction
identities remain in their established order. No listed prediction is reported
as executed or passed.
