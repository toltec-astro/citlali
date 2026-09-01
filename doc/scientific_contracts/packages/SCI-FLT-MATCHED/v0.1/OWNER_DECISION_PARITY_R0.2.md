# SCI-FLT-MATCHED v0.1 r0.2 Owner-Decision and View-Parity Report

Date: `2026-08-31`

Result: `PASS` for source/view parity and directive coverage; this is not
scientific approval, freeze, implementation conformity, response/covariance
fidelity, observational validation, or route authorization

## Input parity

The exact eight admitted author-packet objects remain present and byte-identical
to the approved manifest. `AUTHOR_PACKET_MANIFEST.md` reproduces SHA-256
`255c66da880fc7664a57635b28a98d874fc024490d04528f802635c0382a57c8`.
The r0.2 revision adds only the owner-supplied targeted closure directive,
recorded by
`SCIENTIFIC_OWNER_R0.2_DIRECTIVE_2026-08-31.md`; no implementation-informed
object entered the revision channel.

## Directive coverage

| Directive part | r0.2 authoritative location |
| --- | --- |
| collision-free notation | `src/common/notation.tex`; `NOTATION_CROSSWALK_R0.2.md` |
| measure/covariance algebra | notation Section 7.3; equations Section 9; `SEMANTIC_CHANGE_MAP_R0.2.md` |
| local restriction/inversion and singular GLS | definitions Sections 8.5; equations Section 9.3; `REQ-012/041/050` |
| exact self-adjoint PSD weighting | `REQ-009`; `AO-001` family rule; edge cases |
| output anchor disposition | definitions Section 8.2; `REQ-040`; owner ledger `R0.2-OD-001` |
| general sky and fit boundary | equations Section 9.2; `REQ-030/043`; `PRED-019`--`022` |
| five support roles | definitions Section 8.4; `REQ-042` |
| Learn--Resolve--Apply | definitions Section 8.6; `REQ-015/044`; NOI boundary |
| fixed versus full-procedure response | definitions/equations response sections; `PRED-023` |
| exact science versus numerical profile | equations Section 9.6; `AO-002`; owner ledger `R0.2-OD-002` |
| realized versus reference products | equations Section 9.5; `REQ-020/045`; `PRED-024` |
| AO-003--006 refactor | `src/common/requirements.tex`; `REPRESENTATION_CROSSWALK_R0.2.md` |
| lifecycle and SCI-VAL | definitions Sections 8.9--8.10; `REQ-026/035/049`; owner ledger `R0.2-OD-003` |
| empirical covariance repair | `PRED-008`; NOI boundary |
| radial spectral repair | `AO-001-C`; `SODL-004`; edge cases |
| template representation | definitions Section 8.3; template boundary; `REQ-005` |
| exact boundaries and routes | four `*-v0.1-r0.1.md` boundary drafts; `ROUTE_STATUS_R0.2.md` |
| source/view/manifest parity | this report; `SOURCE_BYTE_REPORT_R0.2.md`; `build/consistency-report.json`; active draft manifest |

## Stable-ID parity

- Requirements: r0.1 `REQ-001`--`039` preserved; r0.2 adds `REQ-040`--`050`.
- Predictions: r0.1 `PRED-001`--`018` preserved; `PRED-008` and `PRED-010`
  receive explicit repairs; r0.2 adds `PRED-019`--`024`.
- Authored options: all 21 r0.1 alternative IDs across six families remain
  present. No alternative is owner-selected by implication.
- Owner ledger: all 17 SODL IDs remain present: 14 open, 2 decided by the exact
  r0.2 invariant directions, and 1 superseded after removal of privileged
  core/tail thresholds.
- `CROSSWALK.md` contains all 95 requirement, prediction, and alternative IDs.

## Two-view parity

Both LaTeX views import the same six shared modules in the same order. The
mechanical verifier found 50/50 requirements, 24/24 predictions, and 21/21 AO
alternatives in each rendered PDF. Scientist-facing narrative and engineering
audit protocol add no conflicting science. The Scientific Rationale and
Engineering Conformance Specification respectively render 32 and 30 pages.

## Unavailability closure

Generic mathematics is defined, but no ordinary-MAP parent realization,
template realization, `AO-001` weight, numerical profile, response-qualified
use, covariance-qualified use, NOI companion, FRUIT consumer query, SCI-VAL
profile, or implementation assessment is available. This matches
`ROUTE_STATUS_R0.2.md` and prevents a source or view parity claim from
manufacturing a scientific route.
