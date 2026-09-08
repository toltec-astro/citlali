# Core-declared products and branched lifecycle - r0.3

Complete sole current Stage B review-candidate normative core: SCI-FRUIT-NORMATIVE-CORE v0.1/r0.3. Canonical source-inventory SHA-256: `e6c3a536246e75739a50741de7862d664cd1ffd52ac877d5ef6b9ead45558c17`. Owner approval, freeze, Registry registration and activation are not established. Numerical methods/routes remain unavailable_pending_separate_owner_approval.

Canonical definitions: [current core](src/common/definitions.tex); response classification: [PTC crosswalk](PTC_RESPONSE_CROSSWALK.md). These are candidate declarations, not an actual Registry or activation record.

| Product | Meaning |
| --- | --- |
| `SCI-FRUIT/ITERATION-RESULT` | Child payload requires exact bundle/reference for scientific use. |
| `SCI-FRUIT/ITERATION-BUNDLE` | Restated universal roles and hard dependencies. |
| `SCI-FRUIT/SUCCESSOR-STATE` | Produced causal state, distinct from sufficiency. |
| `SCI-FRUIT/CONTINUATION-STATE` | Exact causal query/reconstruction and failure contract. |
| `SCI-FRUIT/PATH-SUPPORT` | All eight separately bound. |
| `SCI-FRUIT/INFORMATION-ORIGIN` | Four occurrence classes and distinct model/prior mode origin. |
| `SCI-FRUIT/TERMINAL-SELECTION` | Nine fields, exact comparison and open-world use references. |
| `SCI-FRUIT/TERMINAL-PRODUCT` | Selector plus immutable bundle/reference. |
| `SCI-FRUIT/MODEL-PATH-TRANSFER` | Exact operator, separate from output shift. |
| `SCI-FRUIT/MODEL-INDUCED-OUTPUT-SHIFT` | Deterministic output difference, not covariance or automatic bias. |
| `SCI-FRUIT/MODEL-MISMATCH` | Model/reference discrepancy separately addressable. |
| `SCI-FRUIT/MODEL-INDUCED-BIAS` | Exact estimand/reference/error-law/conditioning required. |
| `SCI-FRUIT/RESPONSE/RF-01` | Input-only with complete defining state/map and other operand fixed; internal coordinates evaluate on perturbed input. |
| `SCI-FRUIT/RESPONSE/RF-02` | Applied-model-only with complete defining state/maps and input fixed; required internal coordinates evaluate on that operand. |
| `SCI-FRUIT/RESPONSE/RF-03` | One iteration: prescribed defining-map/consequential-state resolution rerun; internal fixed-map evaluation alone excluded; prior/upstream fixed. |
| `SCI-FRUIT/RESPONSE/RF-04` | Fixed recursive sequence/index with causal propagation. |
| `SCI-FRUIT/RESPONSE/RF-05` | Selected terminal with stopping/resource/selector dependence. |
| `SCI-FRUIT/RESPONSE/RF-06` | Whole chain reruns included upstream producers. |
| `SCI-FRUIT/RESPONSE/RF-07` | FRUIT-only parent-domain query with upstream procedure/state/generation fixed; explicit component references. |

The thirteen uncertainty identities use the closed prefix/token grammar in [closed uncertainty-role crosswalk](MODEL_UNCERTAINTY_CROSSWALK.md). No unexplained wildcard role exists. Every instance binds its exact scientific identity, parent/bundle/method/state/generations, availability/outcome/cause and provenance.

Request state is immutable `not_requested | requested`; later request creates a new request generation. For a requested generation the successful producer path is:

`requested -> effective -> application_scope_resolved -> applied -> iteration_realized -> complete_iteration_candidate -> completion_decided -> completed_iteration -> published`

Scientific availability (`available | unavailable`) and producer outcome (`realized | failed | not_produced`) are independent dimensions attached to the stage actually reached. No early stop fabricates later invocation, realization, completion or publication. An unavailable optional companion permits a narrower complete product only when core and method allow it. Supersession adds an immutable replacement relation. Terminal selection consumes completed bundles without mutation. Downstream use is open-world child accounting; only an explicitly requested narrower FRUIT role may require a specified child, never the base roles by implication.
