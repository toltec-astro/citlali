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
| `src/common/notation.tex` | `4ed9c44386c3723f72f381c409171de1d558f69bf35581cd7aa3ca5b600dc5c2` |
| `src/common/definitions.tex` | `e9b6cd6dcc7aa6efbfec9b73f72506c6c896103b10bfae82b2536795d1dcd003` |
| `src/common/equations.tex` | `97917f4cb805bd566f91a81f40b12afc917f884cbb860ebc53f707c92a0693dc` |
| `src/common/assumptions.tex` | `8d4281cc62fdcbb231b4cf7b582f1f272a8c0fee7f6287da3a2ba0271eb28724` |
| `src/common/requirements.tex` | `a74b12fe762d7248ae7ddff91208561a829e7b4213cf8dc409365b284d5e2508` |
| `src/common/edge_cases.tex` | `d74901e1fc8f67e52c166d8a3f0933c1906651ee7588b6afad545b12872177ae` |
| `src/scientific-rationale.tex` | `3adc73b8c9f8ca257a78a9ea97859bdef0f3bcbb94e86081a1454cf5623cf474` |
| `src/engineering-conformance.tex` | `159b4e74672a35b021e7c4a1b42bc7a0e8de93d45c41dd0190bc63a57edfb9cd` |
| `AUTHOR_DRAFT_DECISIONS.md` | `f260879cf004b9c37ce58ef763eabc7c1a485117d6e9d4e5ad3de377dec85114` |
| `CROSSWALK.md` | `378298f296b436c89bfbdb3d3e53741f93301332d291b65e18c62e9ec6e4b969` |
| `src/generate_crosswalk.py` | `e022f0463a215f81dbd731e5dea27fdcd253fffedd2bc0b6e216486ffa64b797` |
| `src/verify_contract.py` | `264fa001c788cbfc189143df1289bbc87d563993235f958351bfe690ad8adc29` |

The separate candidate PDFs are:

- scientific rationale, 12 letter pages, SHA-256
  `ce5ccaed4c570533e2d6e96a3230e6eda3b99555aff411494bce0acb0a56cdec`;
- engineering conformance, 25 letter pages, SHA-256
  `bbdb32535511395925b8b85c1529b94365a8e077bd8dd5ccbdd32ae46d5f47e0`.

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

The remaining substantive item deliberately not inferred by the author is
final acceptance of the Equation 8 mask-aware coefficient realization as the
exact ordinary-route operator text.

Until those items are resolved and the full candidate receives a separate
owner approval, r0.5 remains non-authoritative. This packet makes no claim of
implementation conformity, numerical validation, achieved performance,
science qualification, production readiness, or MAP availability.
