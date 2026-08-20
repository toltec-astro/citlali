# RTC–CAL–PTC Cross-Package Owner Decision Ledger

Status: draft for scientific-owner review. This ledger contains only unresolved decisions whose consequences cross a package boundary. It does not close or replace any package ledger.

## CHAIN-OD-001 — Exact RTC product admitted by CAL

- **Question:** Is CAL’s ordinary `xs` input exactly the SCI-RTC conditioned-`x` product and atomic bundle, and what exact coordinate, unit, sign, reference, preprocessing history, valid domain, grid, and parent identity does CAL admit?
- **Alternatives:** (A) exact identity with SCI-RTC output; (B) a named intervening transformation with complete response/support/identity; (C) no composable RTC→CAL route for this role.
- **Scientific consequences:** A closes the direct parent chain; B changes the response and lineage composition; C terminates the proposed chain at RTC.
- **Conservative state while open:** RTC→CAL numerical handoff identity and signal-role continuity are unavailable; similarly named `xs` is insufficient.
- **Affected profile invariants:** CHAIN-INV-001–003/005–007/017.
- **Affected source clauses:** `SCI-RTC-DEF-004/018`, `SCI-RTC-REQ-001–005/103–105`; `SCI-CAL-REQ-001–004/020`; `CAL-OWNER-Q01`.
- **Authority needed:** CAL scientific owner, with an RTC owner concurrence if the bound meaning would alter RTC’s frozen product claim.

## CHAIN-OD-002 — Complete RTC/CAL/PTC operation order

- **Question:** Does CAL operate after complete RTC and before PTC, or is another ordering scientifically intended?
- **Alternatives:** (A) RTC→CAL→PTC; (B) a different order with explicitly redefined quantities, factors, responses, baselines, and parents; (C) separate product roles with different authorized routes.
- **Scientific consequences:** A matches RTC/PTC’s stated chain; B or C changes the signal domain, once-only accounting, response composition, and possibly the meaning of the PTC estimator.
- **Conservative state while open:** The profile’s three-stage order remains proposed and no complete numerical chain operator is authoritative.
- **Affected profile invariants:** CHAIN-INV-001/008–017/035.
- **Affected source clauses:** `SCI-RTC-EQ-004`, `SCI-RTC-REQ-013`; `CAL-OWNER-Q02`, CAL rationale §2; `SCI-PTC-REQ-001/046/063`; PTC `CROSS_PACKAGE_FOLLOWUP.md` CAL-order entry.
- **Authority needed:** CAL scientific owner, with RTC/PTC owner review for any nonstandard order.

## CHAIN-OD-003 — Complete upstream response and RTC-filter/CAL-atmosphere composition

- **Question:** Which authority composes, publishes, and binds the complete admitted upstream response ending on the CAL detector-time grid; how is the generally noncommuting RTC temporal-filter/CAL sample-dependent-atmosphere relation represented; and how is typed unavailability recorded without double application?
- **Alternatives:** (A) CAL composes and carries one cumulative object with the exact operation order; (B) a separate cross-package response authority composes it; (C) PTC consumes an ordered bundle of domain-qualified local responses under an exact composition rule and owner-approved noncommutation bound where needed.
- **Scientific consequences:** A centralizes the CAL-parent guarantee; B creates a separate parent/owner; C preserves local ownership but requires a normative composition and application-count contract.
- **Conservative state while open:** RTC-local, CAL-local, and PTC-local response claims may remain available separately; complete-chain response-dependent use is unavailable.
- **Affected profile invariants:** CHAIN-INV-005/009–010/015–016/029.
- **Affected source clauses:** `SCI-RTC-DEF-004/010`, `SCI-RTC-EQ-006/012–015`, `SCI-RTC-REQ-037–041`; CAL rationale §2 noncommutation identity, `CAL-OWNER-Q02`, `SCI-CAL-EQ-008–009`, `SCI-CAL-REQ-039–043/047`; `SCI-PTC-DEF-041`, `SCI-PTC-REQ-061–066/087`; `PTC-OWNER-OD-007`.
- **Authority needed:** Joint RTC, CAL, and PTC scientific-owner decision; an adjacent response/beam owner if their object becomes part of the normative cumulative parent.

## CHAIN-OD-004 — Numerical CAL atmosphere operator

- **Question:** What exact content-bound atmosphere operator, node set, ordinate orientation, passband identity, support, interpolation/extrapolation rule, and quality policy authorizes CAL numerical evaluation?
- **Alternatives:** (A) supply the record anticipated by CAL; (B) approve a different fully specified operator; (C) retain numerical calibration as unavailable.
- **Scientific consequences:** A or B can create the mandatory numerical PTC parent but does not by itself establish performance or qualification; C prevents the numerical chain.
- **Conservative state while open:** Explicit no-calibrated-output state; no numerical PTC product on this route.
- **Affected profile invariants:** CHAIN-INV-009/011–020/034/037.
- **Affected source clauses:** `SCI-CAL-ASM-011`, `SCI-CAL-REQ-021–031/045–046`, `CAL-OWNER-Q06`; `SCI-PTC-REQ-001`, `PTC-OWNER-OD-007`.
- **Authority needed:** CAL scientific owner and the designated atmosphere-data/operator authority.

## CHAIN-OD-005 — CAL endpoint terminology and PTC fixed-nominal-beam identity

- **Question:** Is the CAL output intrinsically “point-source-peak,” or point-source-equivalent/beam-peak-normalized with literal peak conditional on complete realized response; and how, if at all, does CAL’s originating Beammap response basis become PTC’s fixed nominal beam?
- **Alternatives:** (A) amend CAL to the narrower point-source-equivalent claim and make PTC preserve the exact CAL response-basis identity; (B) retain literal peak and/or fixed nominal beam by adding the exact conversion/renormalization and response guarantees; (C) define distinct output/input roles.
- **Scientific consequences:** A aligns endpoint meaning but changes PTC's current fixed-beam precondition; B strengthens CAL's response and uncertainty obligations; C requires explicit role identity at the handoff.
- **Conservative state while open:** Use only the narrower point-source-equivalent wording; withhold literal peak and fixed-nominal-beam equivalence.
- **Affected profile invariants:** CHAIN-INV-012/019/041.
- **Affected source clauses:** `SCI-CAL-REQ-002/040–043`, CAL response-basis definition, CAL engineering abstract, CAL active rationale §§1–2/5; `SCI-PTC-ASM-001`, `SCI-PTC-REQ-001/010/061/087`.
- **Authority needed:** CAL scientific owner; PTC owner concurrence if the PTC admitted signal wording changes.

## CHAIN-OD-006 — RTC causes and representative-state carriage through CAL

- **Question:** Which RTC cause, support, direct representative-synthesis/replacement, and transitive-influence axes must the CAL product carry, and does CAL own any use-specific narrowing?
- **Alternatives:** (A) transparent carriage with CAL-local validity separate; (B) explicit CAL named-use narrowing while retaining causes; (C) PTC binds the RTC bundle separately and CAL is prohibited from claims depending on undispositioned axes.
- **Scientific consequences:** A maximizes lineage continuity; B introduces a CAL policy identity; C makes joint CAL+RTC parent binding mandatory at PTC.
- **Conservative state while open:** CAL and downstream cause-dependent eligibility/independence claims are unavailable; RTC causes may not be erased or converted to generic validity.
- **Affected profile invariants:** CHAIN-INV-006/017–018/023–024.
- **Affected source clauses:** `SCI-RTC-DEF-011–013/018`, `SCI-RTC-EQ-020/022`, `SCI-RTC-REQ-019–020/046–052/103–105`; `SCI-CAL-REQ-003–004/045–048`; `SCI-PTC-REQ-002–003/011–017/089`.
- **Authority needed:** CAL scientific owner with RTC/PTC owner review.

## CHAIN-OD-007 — CAL→PTC uncertainty and correlation handoff

- **Question:** What minimum uncertainty products and correlation scopes must accompany an admitted CAL parent into PTC, and which stronger covariance/total claims remain optional or deferred?
- **Alternatives:** (A) conditional measurement covariance only with complete nuisance ledger; (B) selected factor/atmosphere/response covariance families; (C) a complete cross-stage covariance package including selection and cross terms.
- **Scientific consequences:** A permits limited signal use but withholds total errors; B enables specified weighted/uncertainty-dependent uses; C supports broader claims at substantially stronger authority/evidence cost.
- **Conservative state while open:** Truthfully limited conditional uncertainty may be available; complete covariance and total uncertainty are unavailable, and missing terms are not zero.
- **Affected profile invariants:** CHAIN-INV-014/019/025–026/039.
- **Affected source clauses:** `SCI-RTC-EQ-016–019`, `SCI-RTC-REQ-042–045`; `SCI-CAL-EQ-010–013`, `SCI-CAL-REQ-032–039`, `CAL-OWNER-Q08`; `SCI-PTC-REQ-057–060/066/088`, `PTC-OWNER-OD-009`.
- **Authority needed:** CAL and PTC scientific owners; other nuisance owners for any promoted component.

## CHAIN-OD-008 — Optional conditioned-`r` producer and grid relation

- **Question:** If the optional PTC `r` diagnostic is desired, which package produces conditioned `r`, and what are its unit, operator, response, support, validity, optical-leakage state, uncertainty, provenance, and exact relation to the CAL detector-time grid?
- **Alternatives:** (A) authorize a dedicated conditioned-`r` producer; (B) omit the branch in base v0.1; (C) define a successor PTC mode with a different coupled `x/r` authority.
- **Scientific consequences:** A enables the inert/advisory diagnostic only; B leaves the main `x` chain unchanged; C changes PTC scope and cannot be inferred from base v0.1.
- **Conservative state while open:** Optional `r` branch unavailable; raw `r` remains uncalibrated RTC lineage and cannot affect PTC `x` membership, subtraction, output, or coefficients.
- **Affected profile invariants:** CHAIN-INV-031–033.
- **Affected source clauses:** `SCI-RTC-REQ-092/103/108`, RTC owner entries `071/074`; `SCI-PTC-REQ-078`, `PTC-OWNER-Q001`, `PTC-OWNER-OD-005`.
- **Authority needed:** RTC/PTC scientific owners and the newly designated conditioned-`r` producer owner.
