# SCI-NOI v0.1 r0.5 Randomization-Law and Replay Disposition

Scientific owner: Grant Wilson. Date: 2026-08-30. Status: proposed final; not
frozen.

Owner disposition: B. For each active stratum, exact balance defines `A_h`; one
canonical concatenation defines `A=product_h A_h`. Before key resolution and
conditional on complete assignment-design resolution, the declared scientific
ideal model is `epsilon_1,...,epsilon_N iid ~ Uniform(A)`. Ideal active signs
are independent symmetric Rademacher values before conditioning. Draws are
with replacement; duplicates and complements remain valid distinct draws.

The scientific target measure, ideal candidate-bit/key-selection law, and
deterministic replay map are separate objects. After exact key binding,
`epsilon_1,...,epsilon_N=F(K,plan)`. The plan binds generator identity/version,
opaque key bytes, namespace, serialization, observation, array, stratum,
member, attempt counter, stable detector order, and domain separation. Replay
is invariant to scheduling, worker count, traversal, layout, and persistence.

Replay equality establishes realized identity, not statistical independence,
random-generator adequacy, or estimator uncertainty. Generator/randomness
adequacy belongs to a separately named validation layer. Any future retained
dependence requires a new joint law and propagated uncertainty/information
semantics. D-007, EQ-011, ASM-004/012, REQ-045/049, and PRED-024 carry this
disposition.
