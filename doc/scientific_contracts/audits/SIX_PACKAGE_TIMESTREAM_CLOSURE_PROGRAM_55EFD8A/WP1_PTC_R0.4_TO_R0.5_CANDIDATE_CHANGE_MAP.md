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
- Predictions `SCI-PTC-PRED-051`--`060`.

Existing entries are revised only where necessary to make the approved route
consistent with those additions. The candidate register contains 45
definitions, 25 numbered equations, 35 assumptions, 99 requirements, and 60
predictions. `CROSSWALK.md` contains 159 exact sequential requirement and
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
| `src/common/notation.tex` | `7d614c2eda17eb88af1c7f8b885e59f2fbd08b430108a9a4426f6226c7f3d08a` |
| `src/common/definitions.tex` | `096dbdb863202bbec06ac547d2484cc1aef4fba7b85570a58c8dcb78463bdd3d` |
| `src/common/equations.tex` | `134959bb9ed3035d750bdc03ee86e9cddf3ce8cd13c16534946e255a0661be49` |
| `src/common/assumptions.tex` | `6c1ed4bcecdbf36817578bd44be0dced27b6fed6856f9c08f4d6e15996096496` |
| `src/common/requirements.tex` | `5953a12e985b8bc1d564807ff94613a8ef048c8cb67883108da43d93660f44f1` |
| `src/common/edge_cases.tex` | `9ed5ff843e2dff5cb5509fe1f90962be1a9acff4f216c663185d180bf2f3b698` |
| `src/scientific-rationale.tex` | `ea9d9ba0aa9515ed1ca1a22c98a18243e0204ffcd8b949608ee33dbf8627b977` |
| `src/engineering-conformance.tex` | `ab6d5c573a79bcf8834e42592b426e8f9346b65dbdacccd0b0b737bdef8af7bc` |
| `AUTHOR_DRAFT_DECISIONS.md` | `695c8c875f693f37f673475fb5ef20bd184af356271c06698b0bfdc306a351cd` |
| `CROSSWALK.md` | `36a6b1ac15f69e66e33a9a622c3a1da8e247cec14b9b0eb9651ed975c82e65a2` |
| `src/generate_crosswalk.py` | `e022f0463a215f81dbd731e5dea27fdcd253fffedd2bc0b6e216486ffa64b797` |
| `src/verify_contract.py` | `3bf4c6503991254728e5d3c2372174742ae3da54dc9a22086fb8b86fcfd2c29b` |

The separate candidate PDFs are:

- scientific rationale, 13 letter pages, SHA-256
  `eb881ba6d85193d01b3c5f3cc387e5e59d70d498f5210914cfdf0041a4671703`;
- engineering conformance, 26 letter pages, SHA-256
  `23941bb70fb24a0e46f24a41409d0f94ba5de9c4b14b35126b239f6452ff4dfc`.

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

The first owner-review gate was resolved on `2026-08-23`: the ordinary route
uses the support-normalized arithmetic mean per detector over finite
basis-fit-admitted occurrences within one immutable PTC segment, with binary
centering influence, no numerical reweighting or cross-boundary borrowing,
and fail-closed invalid-support behavior. This resolution updates Definition
007, Equation 3, Assumption 031, Requirements 023 and 093, Prediction 005, and
the two audience views.

The second owner-review gate was resolved on `2026-08-23`: Equation 8's
group-local detector-right mask-aware coefficient realization is accepted
with an explicit finite time-local full-rank guard under frozen tolerance. A
deficient group-time is unavailable for both data and kernel, without
partial-rank subtraction, rank reduction, interpolation, masked numerical-zero
admission, or cross-group borrowing. This resolution updates Definitions
030--031, Equation 8, Assumption 023, Requirements 029, 083, 089, and 097,
Prediction 060, and the two audience views.

No substantive candidate equation remains open. Complete-packet owner
approval and freeze are still required before r0.5 becomes authoritative.

Until the full candidate receives a separate owner approval and freeze, r0.5
remains non-authoritative. This packet makes no claim of
implementation conformity, numerical validation, achieved performance,
science qualification, production readiness, or MAP availability.
