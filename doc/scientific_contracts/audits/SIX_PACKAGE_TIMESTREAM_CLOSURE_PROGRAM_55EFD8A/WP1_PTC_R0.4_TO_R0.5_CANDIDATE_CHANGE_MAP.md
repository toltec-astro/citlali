# WP-1 SCI-PTC r0.4 to r0.5 Candidate Change Map

Status: bounded scientific-owner review candidate; not frozen

Date: `2026-08-23`

Baseline authority: consolidation commit
`55efd8a54464636a24e621f6d1b60486d235b20e`, SCI-PTC `v0.1/r0.4`

Candidate authority: none. This change map records authoring performed under
approved `WP1-OWNER-D001`--`WP1-OWNER-D009`. The canonical r0.4 PDFs remain
byte-for-byte unchanged. No audit finding is closed until separate owner
approval, freeze, and clean-room re-audit.

## Owner-decision realization

| Owner decision | Candidate realization | Principal formal locations | Audit target |
| --- | --- | --- | --- |
| `WP1-OWNER-D001` | Makes the complete application operator primary, defines total removal by input-output identity, separates additive-reference and correlated-subspace removal, and propagates fixed-state kernel semantics. | Definitions 002--005, 007, 014, 032, 041--043; Equations 2--3, 8--9, 15--18; Requirements 019--025, 062--065, 083, 090, 093, 097; Predictions 039--043, 050, 056, 058. | Candidate repair for `F-001`. |
| `WP1-OWNER-D002` | Replaces the incomplete eligibility pair by a total T/F/U/C decision rule after request, binding, and applicability checks; all causes and conflicts remain visible. | Definition 013; Equation 4; Assumption 029; Requirements 011--017; Predictions 005--011. | Candidate repair for `F-002`. |
| `WP1-OWNER-D003` | Keeps learning, application, retention, QC, response, and other uses as distinct propositions; preserves CAL classification without globally prohibiting PTC mathematics or requiring an engineering route. | Definitions 013 and 037; Requirements 012--017 and 098; Predictions 005--011. | PTC/VAL congruence and classification preservation. |
| `WP1-OWNER-D004` | Reserves `mathcal C` for causes and uses `mathcal M_cand` for candidate-model specifications while retaining candidate symbol `c`. | Notation; Equations 4 and 11; Definition 029; Requirements 034--040 and 084. | Candidate repair for `F-009`. |
| `WP1-OWNER-D005` | Makes explicit PTC-disabled mean complete RTC publication followed by successful Citlali termination before CAL/PTC/MAP; route publication failure fails, and no CAL-to-MAP fallback exists. | Definition 044; Equation 25; Assumption 035; Requirements 076--077, 088, 095; Predictions 001, 053. | Candidate repair for formal role-map defect `F-011`. |
| `WP1-OWNER-D006` | Selects exactly one configured array- or network-level grouping and scopes support, masks, fit, state, application, and kernel to that group. | Definitions 004, 007, 014, 030--031, 042--043; Equations 5, 8, 10, 15; Assumptions 011, 023--024, 030; Requirements 008, 029, 062, 083, 089--093, 097; Predictions 012--013, 051--052, 056--058. | Bounded ordinary-route completion. |
| `WP1-OWNER-D007` | Requires an explicit integer rank at least one, independently feasible per group; zero, noninteger, and unrealizable ranks fail without clipping, substitution, centering-only output, route conversion, or map. | Definitions 026, 029--030, 042--043, 045; Equation 11; Assumption 032; Requirements 034--040, 073--074, 084--085, 090, 094--095, 099; Predictions 017--020, 046, 053--055. | Bounded ordinary-route completion. |
| `WP1-OWNER-D008` | Selects one immutable-CAL-parent fit per group, zero support-changing refinements, advisory diagnostics, inert conditioned `r`, same-operator fixed-state kernel propagation, and explicit first-route exclusions. | Definitions 019--020 and 042--043; Equations 12 and 15; Assumptions 010, 013, 034; Requirements 030, 041--048, 062, 085, 096--097, 099; Predictions 025--026, 021--024, 058--059. | Bounded ordinary-route completion. |
| `WP1-OWNER-D009` | Preserves r0.4 as frozen authority, places candidate PDFs separately, and requires another scientific-owner decision before freeze. | Candidate status blocks in both audience views; author ledger decision 036; this change map; candidate PDF README and verifier. | Authoring authorization only; no closure claim. |

## Normative register delta

The candidate preserves every existing normative identifier and appends:

- Definitions `SCI-PTC-DEF-042`--`045`;
- Assumptions `SCI-PTC-ASM-030`--`035`;
- Requirements `SCI-PTC-REQ-090`--`099`; and
- Predictions `SCI-PTC-PRED-051`--`059`.

Existing entries are revised only where necessary to make the approved route
consistent with those additions. The candidate register contains 45
definitions, 25 numbered equations, 35 assumptions, 99 requirements, and 59
predictions. `CROSSWALK.md` contains 158 exact sequential requirement and
prediction rows.

## Material equation delta

1. Equation 2 now defines `Z` through `mathcal A_Theta` and defines
   `U_total,Theta` as the input-output difference; it no longer makes the
   fitted correlated component alone the subtraction identity.
2. Equation 3 and its linear specialization retain nonrestoring centering and
   explicitly decompose total removal into learned location plus correlated
   removal.
3. Equation 4 is a total three-outcome decision over T/F/U/C restrictions.
4. Equation 8 supplies the exact candidate mask-aware detector-right,
   time-local coefficient recomputation for the ordinary PCA/SVD route.
5. Equation 9 gives the separately named frozen-total-component affine family
   and its identity derivative; it is not the ordinary PCA/SVD rule.
6. Equations 10--11 make grouping alternatives and strictly positive explicit
   rank part of the route identity.
7. Equations 15--18 propagate the group-local fixed operator to compatible
   kernels while keeping fixed-state and full-procedure response distinct.
8. Equation 25 replaces fabricated downstream disabled roles with the
   RTC-terminal workflow transition.

## Candidate packet identity

Candidate source SHA-256 digests are:

| Candidate file | SHA-256 |
| --- | --- |
| `src/common/notation.tex` | `94082ef51a1702a5637eac29fcdd3798114343eeccb18e3343985a950c725543` |
| `src/common/definitions.tex` | `a79bfa0ac0f6fbfef1bc6264dc95389cedfc31b77723fe12ab881f28691c6bd1` |
| `src/common/equations.tex` | `9d1806e281d53501ea9042de6332e06addd4ad92502666f4c0dd89deaebf7410` |
| `src/common/assumptions.tex` | `74dce0b34991d450c952eb742e5079f116676de188a26fabc41f2269d2f0451b` |
| `src/common/requirements.tex` | `da66a9d6cc90c6656a80d1ac3659a6afbd4874525beef6977104c72b929da066` |
| `src/common/edge_cases.tex` | `ca526796ca1ea7781b208a248ebe9b6a76bf4b95bbe91f131ab3214e9173e323` |
| `src/scientific-rationale.tex` | `89b08cb564da174bde1b9400521e77211d747c186b68d36d9930e4e4a2dd475c` |
| `src/engineering-conformance.tex` | `4c8ec10bef5243f3d67e2a0a6ee14af7f3ce56ebe44b626b51ed51fa4f5b91cc` |
| `AUTHOR_DRAFT_DECISIONS.md` | `b31fc6c9654245d78353468d19aa3d1917158c1b3b3c63177fd51e0906270ed0` |
| `CROSSWALK.md` | `bc0bd3f9d428a72531c74e03f8c545bcb993c3aade74c732dc329ef51fb61d1f` |
| `src/generate_crosswalk.py` | `e022f0463a215f81dbd731e5dea27fdcd253fffedd2bc0b6e216486ffa64b797` |
| `src/verify_contract.py` | `1bbdea1a091ceda9b12dafede31cc66aeb324d285c22fe6899c5de8b14b1f2ff` |

The separate candidate PDFs are:

- scientific rationale, 12 letter pages, SHA-256
  `69dc2c86f6193434fcbba3737def8395f5d650c30eb6858ab6b85e47e2a5b7d0`;
- engineering conformance, 25 letter pages, SHA-256
  `cfee3e2f691289ef6bb6a81100051277f65a2e8a6bface962afc8477aa03d9a6`.

The canonical r0.4 PDF hashes remain:

- scientific rationale
  `7cb358eec6633e06ca2559741d4f32ca2cf62607fac2fe6efb73365863832fd0`;
- engineering conformance
  `1e73d3e001dafce4dd6a9025553af95da58075fb49ea2b4eb41222431d658b85`.

The package verifier confirms exact metadata/crosswalk regeneration, source
packet hashes, normative sequences, audience separation, candidate PDF
hashes, complete formal-ID coverage, letter page size, and absence of PDF
forms, encryption, or JavaScript. All candidate pages were rendered with
Poppler and inspected for clipping, overlap, and legibility.

## Remaining owner review before freeze

The following is deliberately not inferred by the author:

1. the exact detector-wise time-axis location estimator and its boundary rule
   for `lambda_g`; and
2. final acceptance of the Equation 8 mask-aware coefficient realization as
   the exact ordinary-route operator text.

Until those items are resolved and the full candidate receives a separate
owner approval, r0.5 remains non-authoritative. This packet makes no claim of
implementation conformity, numerical validation, achieved performance,
science qualification, production readiness, or MAP availability.
