# SCI-PTC v0.1 — Stage A Scope Review R0.1

Status: manager disposition of two scientific review responses supplied by
Grant on `2026-08-19`; owner approval still pending

Historical note: the later bounded review and owner resolution are recorded in
[`SCOPE_REVIEW_R0.2.md`](SCOPE_REVIEW_R0.2.md); its Q001 disposition supersedes
the open state recorded here.

## Review Outcome

The original Stage A governance structure is retained, but scientific
authorship remains held. The reviews identified missing scientific objects,
not editorial preferences. The revised Scope Brief adds decisions
`PTC-SCOPE-D007--D017` and leaves one explicit owner decision open,
`PTC-OWNER-Q001`.

## Disposition Matrix

| Review issue | Disposition | Revised authority |
| --- | --- | --- |
| Physical estimand and warning that correlation does not identify origin | adopted | Scope §§1, 10; D004/D014 |
| Removed subspace, additive reference, astronomical null modes | adopted | Scope §§1, 4, 6; D014 |
| Centering/scaling axis, support, estimator, reversibility, and null space | adopted | Scope §§3--4, 6; D017 |
| Coefficient-family taxonomy and gauge | adopted | Scope §§1, 4, 6; D015 |
| PTC sample response versus map-center diagnostic functional | adopted | Scope §§4--5; D016 |
| Point-source-equivalent mJy per fixed nominal beam terminology | adopted | Scope §§2--3, 6; D002 |
| Retain raw-`r` parent and causal lineage without a raw numerical branch | adopted | Scope §§2--3, 6; D008 |
| Within-array hierarchy and separately authorized cross-array modes | adopted | Scope §§3, 6; D011 |
| Route-specific disabled PTC/MAP behavior | adopted | Scope §§2, 5; D005 |
| Per-cause, per-stage support mapping; no zero-fill PCA | adopted | Scope §§3, 6; D007 |
| Conditioned-`r` diagnostic PCA | adopted subject to a separate conditioner providing an exact compatible parent | Scope §§2--3, 5; D008 |
| `r`-derived temporal template subtraction from calibrated `x` | owner choice required | `PTC-OWNER-Q001`; unavailable pending decision |
| Unconstrained joint `x/r` PCA | deferred | Scope §§2, 8; D008 |
| Bounded fit-diagnose-classify-refit detector assessment | adopted | Scope §§3--4, 6; D009 |
| Base estimator families and adjacent alternatives | adopted | Scope §§7--8; D010 and method boundary |
| Science-objective rank selection rather than universal threshold | adopted | Scope §§3, 6; D012 |
| Fixed-state response companion versus end-to-end injected response | adopted | Scope §§3--4, 6; D013 |
| Source mask protects only declared model/support | adopted | Scope §§3, 7; D004/D010 |
| Internal iteration, PTC pass, and FRUIT recurrence terminology | adopted | Scope §§3--6; D009 |
| Science-rationale narrative order | adopted | Scope §1 and author deliverables |

## Literature Disposition

The five cited primary papers were verified, and one primary robust-PCA paper
was added to support the review's low-rank-plus-sparse distinction. Their
bounded claims and explicit non-authorities are recorded in
[`AUTHOR_METHOD_REFERENCE_BOUNDARY.md`](AUTHOR_METHOD_REFERENCE_BOUNDARY.md).
The Stage B author may use only that summary, not the full papers.

The references demonstrate available method families and model-dependent
limits. They do not select TolTEC algorithms, ranks, thresholds, covariance,
response, or performance.

## Remaining Launch Block

Stage B must not launch until Grant:

1. approves or amends `PTC-SCOPE-D001--D017`;
2. resolves `PTC-OWNER-Q001` as diagnostic-only `r` or authorized
   cross-channel template regression; and
3. approves the recomputed content-bound author packet.
