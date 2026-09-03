# SCI-RTC v0.1/r0.11 to r0.12 clause and equation analysis

Date: 2026-08-21

Status: Implementation-blind surgical supersession map for owner review. The
approved r0.11 architecture is preserved at comparison commit
`85e1e6c6865f74f1a97e99fab465714f43877c3d`.

## Exact correction boundary

R0.12 corrects seven consistency defects identified after approval of the
r0.11 scientific architecture. It does not reopen that architecture, change
conditioned-$x$ numerical behavior, authorize $x\leftrightarrow r$ numerical
mixing, introduce an $r$ donor, calibrate $r$, or choose PTC/SCI-VAL policy.

| R0.11 locus | R0.12 disposition | R0.12 authority |
| --- | --- | --- |
| Mapping prose that could conflate native and aligned coordinates | Native IQ maps to $(x^{\rm acq},r^{\rm acq})$ before exactly one ALIGN produces $(x^A,r^A)$ | REQ-139; PRED-104 |
| Iterative equations and prose projected onto $x$ | Learn, evaluate, and replay complete pair plans from original admitted $u^{(0)}$; output projection remains conditional | DEF-027--028; EQ-028--029; REQ-074/081--082/140; PRED-040/042/105 |
| Ambiguous “common support” language | Common index grid, pair-action/operator support, coordinate-local availability, and joint covariance support are distinct | DEF-052; EQ-042; REQ-093/116/135/141; PRED-096/106 |
| Local-only unavailable affine correction | Unavailability propagates through full downstream FIR, IIR-to-reset, notch, and sampling influence for that coordinate | DEF-049; REQ-128/142; PRED-091--092/107 |
| $x$-only composition/prefilter shorthand | Ordinary prefilter is paired, admitted by $x$-domain science budgets; conditioned $r$ is published only when requested and available | REQ-074/116/136; PRED-105 |
| Implicit r0.11 decision summary | OWNER-090--096 are enumerated in the rationale | Rationale §1; OWNER-097--102 |
| Diagram/prose ambiguity after conditioned-$r$ correction | Immutable raw-$r$ parent and original raw-$r$ event evidence remain unchanged | REQ-143; PRED-108; OWNER-103 |

The formal inventory grows only to name those distinctions: DEF-052, EQ-042,
REQ-139--143, PRED-104--108, and OWNER-097--103. Every unrelated owner-ledger
state and package boundary is preserved.
