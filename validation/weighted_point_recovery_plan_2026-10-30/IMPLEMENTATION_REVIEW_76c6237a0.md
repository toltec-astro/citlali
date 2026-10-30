# First exact-source implementation review and repair disposition

Independent fresh-context reviewer `/root/recovery_implementation_review` reviewed
candidate76c6237a0b7d82b95a59a3e71349eb13e60a79ea,
tree8d80073a76f32f6928ae4bf6cb91cbb0c7e78834,
parent/basee42b0368cb3ffe3d8d44488e749a39f3b8ab8d65.
Verdict: **repair required**. No launch readiness follows from this candidate.

P2: final native stage/terminal accounting could cross resource ceilings while
RUN reported completed. The reviewer reproduced a last-stage sparse-file
overshoot using the fully mocked runner fixture, without reducing observations.
Repair: check after every completed cycle and before/after terminal accounting,
including after final RUN publication. Persist global stopped status on terminal
failure. Regression fixtures cover SEED M10 and accounting overshoots.

P2: metadata, separate FITS and mocked-runner seed tests did not exercise the
actual native M1/M2 switch. Repair: route native intervention through a shared
private boundary delegating arithmetic to existing FRUIT. Tiny matrix tests call
that exact boundary and prove unchanged SEED parent, M1 ordinary-plus-shoulder,
M2 ordinary replacement without forced seed, signed parent addition and no
additional restoration term. No scientific algorithm or policy is changed.

Scientific/behavioral disposition: repair required for those two issues; otherwise
signed parents, independent M0 contexts, fixed weighting, native weighted reference,
measurement geometry and descriptive interpretation conformed on inspection.
Architecture: pass. Evidence/repository: pass with limitations. Reviewer inspected
all changed source/tests and reran27 synthetic tests, all passing. No reduction,
saved-observation fit, Unity, push or edit. A new exact-SHA verdict is required.
