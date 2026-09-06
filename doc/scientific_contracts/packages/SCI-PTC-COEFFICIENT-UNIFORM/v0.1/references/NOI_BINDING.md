# NOI derived coefficient parent and profile limitations

## Source cover

N01 approves the exact N05-bound SCI-NOI v0.1/r0.5 snapshot at
2303daf7061a19945a6333099d33dd559cf2abf8. That freeze controls historical
proposed/not-frozen wording. The earlier N02 boundary supplies the derived
MAP-balance role only; later r0.5 exact-parent and profile states control.
No historical delegation to choose an NOI design is imported into this PTC
task. The notation B_d in the earlier boundary and beta_d in the final
parent-binding record denotes the NOI-owned detector balance role.

The following science is already selected and must be preserved, not
re-derived. NOI's exact rational representation governs design identity only;
it does not select the PTC coefficient or change MAP arithmetic. PTC work may
state a conditional compatibility requirement. It cannot approve the four
NOI r0.5 profile successors, redefine the randomization method, or admit a
JINC-hosted NOI route.

## Derived balance definition

Source: N02; whole-source SHA-256 `0a6484058569930cee62e80e04ca2045c107fde67603f662473ae471406f905c`.

<!-- EXCERPT NOI_BINDING-Derived balance definition -->
## Approved Ordinary Balance Family

ODQ-102C selects a network-stratified, coefficient-balanced randomized sign
family. For each exact observation and stable readout network, NOI derives

```text
B_d = sum_p sum_{i in C_p, detector(i)=d} a_pi,
a_pi = G_pi gamma_i > 0,
```

from the exact frozen MAP-admitted contribution population. The sign design
balances `B_d` separately within each network. No cross-network, cross-array,
or cross-observation cancellation may satisfy that balance rule.

<!-- END EXCERPT NOI_BINDING-Derived balance definition -->

## Derived meaning and changed identity

Source: N02; whole-source SHA-256 `0a6484058569930cee62e80e04ca2045c107fde67603f662473ae471406f905c`.

<!-- EXCERPT NOI_BINDING-Derived meaning and changed identity -->
`B_d` is an NOI-owned design coefficient derived from the exact frozen host
route. It is not precision, inverse variance, empirical NOI weight, exposure,
validity, or a replacement for the PTC-owned `gamma_i`. Changed coefficient,
admitted population, network, observation, or MAP plan changes the design
identity.
<!-- END EXCERPT NOI_BINDING-Derived meaning and changed identity -->

## Frozen exact rational parent

Source: N03; whole-source SHA-256 `ff2d3ca458bfab8d2eb3cf76fc639fa0619100eda0c97c508ccec4faa06e0763`.

<!-- EXCERPT NOI_BINDING-Frozen exact rational parent -->
The selected representation is a canonical reduced arbitrary-precision
rational pair `(n,q)`: `q>0`, `gcd(|n|,q)=1`, and zero is only `(0,1)`.
Positive `a_pi` has `n>0`; `tau_h` satisfies `0<=n<q`. Canonical external
serialization is ASCII `n/q` with base-ten integers, no leading plus sign, no
leading zeros except zero, and no whitespace.

Each design contribution satisfies `decode(n_pi/q_pi)=a_pi^MAP` exactly and
binds the frozen MAP product/generation, exact `G_pi`, exact `gamma_i`
family/generation/payload, exact parent `a_pi`, representation source, and
source digest. `beta_d`,
imbalance numerators, and totals are accumulated exactly in stable scientific-
identity order with arbitrary-precision integer arithmetic and canonical
reduction. For nonnegative `L_h=n_L/q_L`, `T_h=n_T/q_T`, and
`tau_h=n_tau/q_tau`, admission is exactly
`n_L q_T q_tau <= n_tau n_T q_L`.

The rational is not rounded from display text, independently quantized,
re-estimated, converted through undocumented decimal precision, or taken from
another generation. It governs NOI balance/design identity only and never
replaces independently governed MAP numerical arithmetic. Missing lossless
canonical rational identity makes design resolution unavailable.
Traversal-dependent floating reduction, undocumented epsilon, post-failure
tolerance relaxation, and best-failed candidate selection are prohibited.
<!-- END EXCERPT NOI_BINDING-Frozen exact rational parent -->

## Freeze and profile nonapproval

Source: N01; whole-source SHA-256 `dba66966ed7082ea55b756c23f4cc9de6205022f1362ff0a69da63bc85190d2c`.

<!-- EXCERPT NOI_BINDING-Freeze and profile nonapproval -->
## Disposition

The implementation-independent SCI-NOI v0.1 r0.5 scientific contract snapshot
identified above is scientifically frozen. This post-snapshot record supplies
approval provenance; it is not part of, and does not modify, the frozen byte
set.

The freeze preserves the manifest's exact authority rule and claim ceiling.
In particular, it freezes the four r0.5 profile-successor records in their
stated proposed, unapproved, Registry-unbound, and unevaluable status. It does
not approve or bind those profiles and does not authorize their evaluation.

The freeze establishes no implementation conformity, random-generator
validation, calibration, physical-noise validity, covariance completeness,
Gaussian significance, achieved performance, readiness, production
suitability, or production authorization.
<!-- END EXCERPT NOI_BINDING-Freeze and profile nonapproval -->

## Exact profile successor status

Source: N04; whole-source SHA-256 `38d4e66613a9c290d470948a2e9b550384338b9362f4aca76fa1bd38cb29cec7`.

<!-- EXCERPT NOI_BINDING-Exact profile successor status -->
| Use | Immutable old profile/digest and old action | Changed scientific action in r0.5 | Revised successor/digest | Policy owner | Approval / Registry binding | Source compatibility | Current evaluation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Generation input | `SCI-NOI:generation_input_admission@1`; `373b8496f6553527da937ae44422734b00db16517964ca239dbbefba6f59c5b3`; admit one exact candidate occurrence only. | Requires distinct enabled lifecycle and a lossless rational lift bound to exact frozen MAP/coefficient generations, representation source, and digest; later actions remain separate. | `SCI-NOI:generation_input_admission@2`; `4d62a7c6469f708fc21a6f8a0ae89a97b9e75940eeb9ec1c55fc842d52369a41`. | Grant Wilson | Proposed bytes; approval absent; Registry/source binding absent. | Not byte- or action-compatible with `@1`; r0.4 draft `53361ec1df6006e8b0d80b6c59391165fd1df678a777811680d24856cc18a697` superseded before approval. | Decision/evaluation unavailable. |
| UNC member | `SCI-NOI:uncertainty_member_admission@1`; `85c6641060ebb21f4246995d8f471888992b2b29b013457254ba1cc5938cfaa3`; admit one exact GEN member candidate; ensemble separate. | Binding completion/use predicates must be fixed before assignment or assignment-invariant; output-dependent diagnostics are advisory, and assignment-dependent failure makes the method unavailable. | `SCI-NOI:uncertainty_member_admission@2`; `5abb1353efe1b424021ab2dbc842232c9eb00c0d44692638c48e46ea19fd824f`. | Grant Wilson | Proposed bytes; approval absent; Registry/source binding absent. | Not byte- or action-compatible with `@1`; r0.4 draft `554e608d3eb3f5a89a3bc9d07504f8f48673ed768ae85ad1970a59ff51a240b0` superseded before approval. | Decision/evaluation unavailable. |
| UNC ensemble | `SCI-NOI:uncertainty_ensemble_admission@1`; `61e3da9349f55110ee8cd46114c980dbda6b173cc24514e79404db3ede9d18d6`; admit one complete all-members-successful ensemble to the named estimator/domain. | Requires EQ-017 ignorability, conditional realized set/count equality, ideal pre-key `Uniform(product_h A_h)` target separated from deterministic replay, no favorable regeneration, and explicit/unavailable estimator uncertainty. | `SCI-NOI:uncertainty_ensemble_admission@2`; `abbfc097d9b652101d6c4a1400ce15d72c0576a56320cfb07c93b1941c0949db`. | Grant Wilson | Proposed bytes; approval absent; Registry/source binding absent. | Not byte- or action-compatible with `@1`; r0.4 draft `b19e0a19782cd42cb24203fd96fe67b89e014aa999c34decf49e09412f2d4e3c` superseded before approval. | Decision/evaluation unavailable. |
| Standardization | `SCI-NOI:standardization_admission@1`; `f6979949b907ffebe97c1a807381398dfb342dafcfe3def477f1c08dbcd77021`; permit the old unit-one conditional-scale action. | Retains the r0.4 independently governed numerator, dependence/support/cause, and fixed-scale/full-response boundary; source and upstream UNC authority references are revised. | `SCI-NOI:standardization_admission@2`; `2cdc10762664469647d113269263f90a78f800fabdd790f00cad901430850d27`. | Grant Wilson | Proposed bytes; approval absent; Registry/source binding absent. | Not byte-compatible with `@1`; r0.4 draft `d825f97c3d89f6f6f73a83bdf2d3fc2034c949f881b29533e8eff8353df419b4` superseded before approval. | Decision/evaluation unavailable. |
| Reciprocal use | No dedicated r0.18 profile; no portable consumer action. | Reciprocal, inverse variance, precision, and weight remain outside the ordinary base bundle. | No active successor. | Grant Wilson | Separate future decision/profile/binding required. | Not applicable. | Unavailable. |

<!-- END EXCERPT NOI_BINDING-Exact profile successor status -->
