# MAP-SPACE-HORIZONTAL-SCIENTIFIC-CONTRACT-AUDIT-001 Cross-Package Conformance Matrix

Status: completed under owner disposition `MSP-OD-001`, with shared-source
repair outstanding.  The matrix retains every admitted-source contradiction
as a repository-documentation finding while applying the frozen MAP r0.7.1
and JINC r0.3 meanings to the scientific/package-coherence result.

## Dimensions and controlled statuses

| Code | Audit dimension |
| --- | --- |
| A | product identity, route identity, parentage and ancestry |
| B | units, calibration, normalization and reference state |
| C | coordinates, geometry, support, coverage and no-data semantics |
| D | signal, residual, model, diagnostic and randomization semantics |
| E | coefficients, weights, coverage, variance and covariance distinctions |
| F | response, transfer function, kernel and effective source shape |
| G | uncertainty and statistical claims |
| H | lifecycle, completeness, partial success and failure |
| I | eligibility, named use and policy ownership |
| J | deterministic, estimator, inferential, diagnostic and evaluation category |
| K | available, conditional, unavailable and not-applicable classification |

The cells use only `PASS`, `CONDITIONAL`, `CONTRADICTION`, `UNAVAILABLE`, and
`NOT_APPLICABLE`.  `PASS` means the frozen contracts express a coherent
contract-level rule on that dimension; it is not a realized-product or
implementation result.  Every other cell maps in the final column to a
finding or an explicit unavailable/not-authorized state.

`CONTRADICTION` cells remain deliberately present after the owner disposition.
They record conflicting clauses that still exist in
`doc/SCIENTIFIC_CONVENTIONS.md`; they do not mean that the audit is still free
to choose between scientific interpretations.  `MSP-OD-001` selects the frozen
package meaning, and each affected MAJOR finding is therefore
`OWNER-RESOLVED / SHARED-SOURCE-REPAIR-REQUIRED` rather than downgraded or
closed as repaired.

## Explicit unavailable and negative states

| State ID | Exact state at the audit base |
| --- | --- |
| MSP-U-001 | Ordinary numerical MAP ingress is unavailable: no exact MAP-facing PTC coefficient family and no admitted numerical `coverage_cut`. |
| MSP-U-002 | A MAP coadd type and policy are defined, but no source-closed numerical observation route exists at this base; the response/covariance-qualified variants also require their exact companions. |
| MSP-U-003 | Numerical JINC ingress is unavailable: no selected JINC-permitted coefficient family, TolTEC parameter set, and required numerical-adequacy profile/certificate. |
| MSP-U-004 | The edge is scientifically not authorized, not merely unimplemented: MAP-to-JINC serialization, JINC ordinary coadd, implicit filter cascades, NOI-to-mapmaking coefficient promotion, and POINT detection/catalog expansion. |
| MSP-U-005 | FLT-FIXED's type-level parent routes are defined, but ordinary numerical parents remain unavailable and its policy records are unregistered. |
| MSP-U-006 | FLT-MATCHED A/C method families are frozen, but no concrete numerical weighting, registered role profile, or representation/response/covariance fidelity is established. |
| MSP-U-007 | Frozen NOI r0.5 changes require `@2`; those profiles are unapproved, Registry-unbound and unevaluable, and the needed numerical MAP/JINC parents are unavailable.  The older `@1` records cannot substitute. |
| MSP-U-008 | Every POINT numerical parent route is unbound; compatibility, formal-error and full-map-RMS method authorities remain unavailable. |
| MSP-U-009 | POINT's four named-use profiles are draft, absent from the exact Registry, and unevaluable. |
| MSP-U-010 | FRUIT is an excluded deferred attachment envelope; no FRUIT route realization is applicable to this audit. |
| MSP-U-011 | Where a contract permits a base signal without complete response or covariance, the missing companion remains explicitly limited/partial/symbolic/lineage-resolvable/unavailable—never zero or independence. |

## Product and producer-consumer matrix

<!-- BEGIN-CONFORMANCE-ROWS -->
| Edge | A | B | C | D | E | F | G | H | I | J | K | Finding/state mapping |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MSP-E001 | CONTRADICTION | PASS | PASS | CONTRADICTION | PASS | PASS | PASS | PASS | PASS | PASS | UNAVAILABLE | MSP-F-001; MSP-U-001 |
| MSP-E002 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | UNAVAILABLE | MSP-U-001 |
| MSP-E003 | PASS | PASS | CONTRADICTION | CONTRADICTION | CONTRADICTION | PASS | PASS | CONTRADICTION | PASS | PASS | CONTRADICTION | MSP-F-003 |
| MSP-E004 | PASS | CONTRADICTION | PASS | PASS | CONTRADICTION | PASS | PASS | PASS | PASS | PASS | CONDITIONAL | MSP-F-002; MSP-U-002; MSP-U-011 |
| MSP-E005 | CONTRADICTION | PASS | PASS | PASS | CONTRADICTION | CONTRADICTION | CONTRADICTION | PASS | PASS | CONTRADICTION | CONTRADICTION | MSP-F-004; MSP-U-003 |
| MSP-E006 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | UNAVAILABLE | MSP-U-003 |
| MSP-E007 | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | MSP-U-004 |
| MSP-E008 | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | MSP-U-004 |
| MSP-E009 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | CONDITIONAL | PASS | UNAVAILABLE | MSP-U-005; MSP-U-011 |
| MSP-E010 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | CONDITIONAL | PASS | UNAVAILABLE | MSP-U-005; MSP-U-011 |
| MSP-E011 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | CONDITIONAL | PASS | UNAVAILABLE | MSP-U-003; MSP-U-005; MSP-U-011 |
| MSP-E012 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | CONDITIONAL | PASS | UNAVAILABLE | MSP-U-006; MSP-U-011 |
| MSP-E013 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | CONDITIONAL | PASS | UNAVAILABLE | MSP-U-006; MSP-U-011 |
| MSP-E014 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | CONDITIONAL | PASS | UNAVAILABLE | MSP-U-006 |
| MSP-E015 | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | MSP-U-004 |
| MSP-E016 | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | MSP-U-004 |
| MSP-E017 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | UNAVAILABLE | PASS | UNAVAILABLE | MSP-U-001; MSP-U-007 |
| MSP-E018 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | CONDITIONAL | UNAVAILABLE | PASS | UNAVAILABLE | MSP-U-007 |
| MSP-E019 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | UNAVAILABLE | PASS | UNAVAILABLE | MSP-U-007; MSP-U-011 |
| MSP-E020 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | UNAVAILABLE | PASS | UNAVAILABLE | MSP-U-003; MSP-U-007; MSP-U-011 |
| MSP-E021 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | CONDITIONAL | UNAVAILABLE | PASS | UNAVAILABLE | MSP-U-005; MSP-U-007; MSP-U-011 |
| MSP-E022 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | CONDITIONAL | UNAVAILABLE | PASS | UNAVAILABLE | MSP-U-006; MSP-U-007; MSP-U-011 |
| MSP-E023 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | UNAVAILABLE | PASS | UNAVAILABLE | MSP-U-001; MSP-U-008; MSP-U-011 |
| MSP-E024 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | UNAVAILABLE | PASS | UNAVAILABLE | MSP-U-003; MSP-U-008; MSP-U-011 |
| MSP-E025 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | UNAVAILABLE | PASS | UNAVAILABLE | MSP-U-005; MSP-U-008; MSP-U-011 |
| MSP-E026 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | UNAVAILABLE | PASS | UNAVAILABLE | MSP-U-006; MSP-U-008; MSP-U-011 |
| MSP-E027 | PASS | PASS | PASS | PASS | PASS | PASS | CONDITIONAL | CONDITIONAL | UNAVAILABLE | PASS | UNAVAILABLE | MSP-U-008; MSP-U-011 |
| MSP-E028 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | UNAVAILABLE | PASS | UNAVAILABLE | MSP-U-009 |
| MSP-E029 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | UNAVAILABLE | PASS | UNAVAILABLE | MSP-U-009 |
| MSP-E030 | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | MSP-U-010 |
| MSP-E031 | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | MSP-U-004 |
| MSP-E032 | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | MSP-U-004 |
<!-- END-CONFORMANCE-ROWS -->

## Owner-disposition overlay for the four contradiction rows

| Edge | Scientific/package result under `MSP-OD-001` | Repository-documentation result |
| --- | --- | --- |
| MSP-E001 | `PASS` for frozen MAP physical identity; numerical ingress remains `UNAVAILABLE` under MSP-U-001. | `CONTRADICTION`: MSP-F-001 is MAJOR and `OWNER-RESOLVED / SHARED-SOURCE-REPAIR-REQUIRED`. |
| MSP-E003 | `PASS` for frozen unique-original AST-coordinate exposure semantics. | `CONTRADICTION`: MSP-F-003 is MAJOR and `OWNER-RESOLVED / SHARED-SOURCE-REPAIR-REQUIRED`. |
| MSP-E004 | `PASS` for the frozen observation-level dimensionless `u_op=1` rule; an actual source-closed coadd remains `CONDITIONAL`/`UNAVAILABLE` under MSP-U-002 and MSP-U-011. | `CONTRADICTION`: MSP-F-002 is MAJOR and `OWNER-RESOLVED / SHARED-SOURCE-REPAIR-REQUIRED`. |
| MSP-E005 | `PASS` for exact five-role JINC base-bundle closure; numerical ingress remains `UNAVAILABLE` under MSP-U-003. | `CONTRADICTION`: MSP-F-004 is MAJOR and `OWNER-RESOLVED / SHARED-SOURCE-REPAIR-REQUIRED`. |

The MAP coadd result is limited to one dimensionless coefficient for each
admitted observation row.  It does not replace or flatten authorized sample-,
pixel-, numerator-, denominator-, validity-, or coverage-level information,
and it does not authorize a JINC coadd.  Base SCI-JINC v0.1 has no
cross-observation coaddition rule.

The JINC closure result is also negative: weight, support, response,
covariance, exposure, diagnostic, and generalized-provenance numerical roles
are not implicit, optional, or downstream-inferable base-v0.1 products.
Representative routes that need them remain `NOT_AUTHORIZED`, `UNAVAILABLE`,
or `NOT_APPLICABLE` as mapped above.

## Horizontal conclusions that remain safe

- The frozen package authorities themselves preserve MAP/JINC sibling
  identity, FIXED/MATCHED distinction, NOI/mapmaking separation, POINT's
  no-detection ceiling, per-array atomicity, and named-use ownership.
- The rationale and engineering-conformance representations of each package
  are faithful to their respective frozen core.  That is representation
  fidelity only.
- All present numerical routes fail closed through explicit unavailable states;
  no missing response/covariance is treated as zero or independence.
- Under `MSP-OD-001`, the frozen package meanings are horizontally coherent.
  The four retained contradiction rows show that repository documentation is
  not yet coherent and prevent an unqualified `PASS` until the cited shared-
  conventions clauses receive a separately authorized repair.
