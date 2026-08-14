# Prompt for the Citlali scientific-audit manager

You are the Citlali scientific-audit manager/coordinator. Grant has directed
the program to add one new durable MAP audit -> bounded repair -> independent
re-audit product for production of an OOF residual transfer function. This is
owner authority to prepare and register the product described below. It is
not authority to launch its independent audit, edit application code, run a
reduction, request or use Unity, change LMTOOF, authorize repair/re-audit,
integrate, expand production, merge, rebase, push, or contact an external
party.

Read completely before acting:

1. repository `AGENTS.md`;
2. the TolTEC context skill and the authorities to which it routes this task;
3. the current canonical versions of `doc/audits/README.md`,
   `doc/audits/AUDIT_MANAGER_INSTRUCTIONS.md`,
   `doc/audits/NUMERICAL_PROPORTIONALITY_AND_COST_CONTROL_POLICY.md`, and
   `doc/audits/templates/PACKAGE_AUDIT_PROMPT_TEMPLATE.md`; and
4. `handoff/HANDOFF_2026-08-14.md`, section
   "proposed `SCI-MAP-003` OOF residual-transfer product."

Do not treat the current working branch or a historical branch named in that
handoff as canonical audit authority. Resolve and report the present
coordination branch, coordination HEAD/parent/tree, pushed application
mainline and exact application HEAD, current audit-ledger state, relevant
package states, worktrees, and dirty state. Do not alter another checkout or
any pre-existing dirty/untracked files.

## Mandatory intake checkpoint

Before editing, return a concise manager checkpoint and stop for direction if
any material mismatch exists. The checkpoint must state:

- whether `SCI-MAP-003` is unused on all current canonical/local audit refs;
- the proposed exact package ID, name, tier, queue/dependency position, and
  governing-source selection rule;
- the exact documentation-only branch/worktree you will use;
- exact files you propose to add or modify;
- the pre-core authority candidates and post-core evidence candidates, with a
  one-line authority rationale for each;
- whether the package can remain Tier B or has already met a Tier A promotion
  trigger;
- planned resource allocation under `FRAMEWORK-EFFORT-001`;
- numerical-study cost classification and why no execution is needed for
  registration;
- first viable artifact; and
- the next mandatory return point.

Silence prohibits a capability. Do not create a new schema, executable helper,
runner, verifier, validation campaign, delegation, independent review, local
reduction, or evidence request during this intake.

## Product to register

Provisional package identity:

- package ID: `SCI-MAP-003`, subject to verified availability;
- package name: `OOF residual transfer-function estimation and product`;
- proposed tier: Tier B, interface and response;
- lifecycle: independent audit -> separately authorized bounded repair ->
  separately launched fresh independent re-audit;
- production status at registration: `existing_use_only` for existing OOF,
  with the new transfer product and LMTOOF use not yet authorized;
- initial contract/implementation/validation/verdict:
  `not_started` / `not_assessed` / `not_started` / `pending`.

The bounded transformation is:

```text
exact discrete APT-Gaussian reference g
    + final normalized OOF/JINC kernel k
    + exact parent/configuration/iteration identity
    -> complex two-dimensional residual transfer H and validity domain
    -> declared LMTOOF forward-model consumer interface
```

The claimed starting estimator is:

```text
G = FFT2(g)
K = FFT2(k)
H = conj(G) * K / abs(G)^2   only on an admitted denominator domain
P_pred(theta) = IFFT2(H * FFT2(P_opt(theta)))
```

This is system identification from a known tracer and its final response. It
does not divide or otherwise deconvolve the observed OOF map. `H` is produced
once per OOF observation/reduction, held fixed throughout that LMTOOF solve,
and recomputed from a later observation/reduction after a telescope
correction. Trial aberrations are never inputs to transfer construction.

The ordinary proposed claim is only the map-center, linearized residual
transfer for the exact OOF/JINC product cell. The product retains two-
dimensional complex phase and measured DC/amplitude unless the future audit
and owner decisions explicitly authorize a narrower representation. The
validity domain, denominator conditioning, frequency/WCS axes, FFT centering,
padding/windowing, units, parentage, and failure/status semantics are part of
the estimator, not implementation details to choose silently.

## Package split and primary ownership

Register this as a new MAP package rather than silently reopening or
relabelling `SCI-MAP-002`. The transfer quotient, validity band, persisted
representation, and LMTOOF admission are an independently meaningful
estimator/product/consumer gate.

Preserve these ownership boundaries:

- `SCI-MAP-001` owns shared map-product identity, WCS/unit/validity, parentage,
  required-output, and publication vocabulary.
- `SCI-MAP-002` owns the JINC response and the final `K/C` kernel identity
  under `SCI-MAP-002-D003-KERNEL-001`. It supplies the numerator; this package
  does not reopen JINC deposition.
- `SCI-RTC-001` owns exact source-kernel construction and RTC response parity.
- `SCI-PTC-001` owns conditioned/local cleaning response and its response-
  family restrictions. Its optional transfer-characterization plan is
  relevant context but is neither launched nor made primary owner here.
- `SCI-VAL-001` owns applicable validity/eligibility facts.
- `SCI-AST-001` owns map-frame/WCS/axis/centering validity.
- `SCI-FRUIT-001` owns the final iteration/pass relation between feedback
  state and the delivered map/kernel.
- `SCI-MODE-001` owns the OOF product association and LMTOOF consumer policy.

Create stable inbound/outbound handoff IDs as needed. Only owner-approved
contract facts may be `pre_core_authority`; implementation observations,
unaccepted package results, current source traces, and the informal Python
prototypes are `post_core_evidence` unless a specific approved abstraction
authorizes otherwise.

## Prototype identity

Record but do not modify or import the informal prototypes unless Grant later
requests a durable fixture copy:

- `/Users/gwilson/foo/transfer_function/test_kernel_jinc.py`, SHA-256
  `65d5f1ee95f09ca290594bbf239950bce932e56d6186f9818102beee6311d1dd`;
- `/Users/gwilson/foo/transfer_function/test_kernel_2.py`, SHA-256
  `b16ba174cbd75dc6685332f7187f33038cad537a845befd4c52ef98d08f91824`.

Their cross-spectrum, JINC/Hankel, and box/sinc mechanics are design evidence.
Their hard-coded parameters, synthetic convolution and normalization,
analytic denominator shortcut, real-only projection, and radial plots are not
production authority.

## Required registration deliverables

After the intake checkpoint is accepted, prepare only the documentation and
coordination artifacts needed to make a later audit independently
dispatchable:

1. a product-registration record under `doc/audits/packages/` stating exact
   identity, included scope, exclusions, tier/promotion triggers,
   dependencies, consumer restrictions, lifecycle, status axes, and explicit
   non-authorizations;
2. the corresponding canonical `audit-ledger.yaml` package record and any
   bounded queue/dependency update;
3. proposed cross-audit handoff records and a frozen inbox authority manifest
   separating `pre_core_authority` from `post_core_evidence`;
4. a fully populated future independent-audit prompt derived from
   `PACKAGE_AUDIT_PROMPT_TEMPLATE.md`, with exact source SHA, paths, resource
   profile, scope checkpoint, studies/cost classification, outputs, and stop
   rule; and
5. a concise coordinator return recording exact commits, trees, file
   digests, unresolved owner choices, and whether the packet is ready for
   Grant's separate launch decision.

The audit prompt should quarantine at least the implementation paths that
construct the Gaussian tracer, filter/clean the kernel, deposit and normalize
the JINC kernel, execute fruit-loop feedback, publish map products, and route
OOF outputs. Resolve exact paths at the selected governing SHA; do not trust
line numbers from the handoff.

## Required future audit questions

The frozen audit packet must require an independent core before source or
post-core evidence inspection. At minimum it must decide or identify owner
decisions for:

1. the exact discrete reference `g`, including APT/detector mixture,
   amplitude, truncation, centering, pixelization, and relation to the LMTOOF
   nominal optical model;
2. the exact final kernel `k` and its map/JINC/fruit-loop parent and validity;
3. complex versus real and two-dimensional versus radial product identity;
4. FFT centering, axes, units, normalization, padding/cropping/windowing, and
   Hermitian/real-inverse rules;
5. denominator threshold, admitted spatial-frequency band, taper/exclusion,
   and invalid/unavailable state;
6. DC/amplitude preservation and any separately represented LMTOOF nuisance
   amplitude;
7. approximation domain in position, morphology, aberration magnitude,
   linearity, band/mode, and iteration;
8. required product representation, association, parent digests,
   provenance, status, and failure policy;
9. LMTOOF consumer behavior, including fixed-within-solve and
   recomputed-between-OOF-cycles identity; and
10. local, retained, external, and telescope-gain evidence needed for
    validation and production adoption.

The future audit should preregister identity, scalar, shift/phase, box/sinc,
JINC/Hankel, asymmetric-filter, forward-closure, denominator-boundary,
shape/centering/WCS, Hermitian, and deliberate-identity-mismatch fixtures.
These are proposed inexpensive analytic/synthetic cases. Any local Citlali
reduction, broad injection campaign, LMTOOF execution, Unity evidence,
telescope correction, or iteration-to-gain study requires a later checkpoint,
the applicable `FRAMEWORK-NUM-001` controls, and separate human/external
authorization.

## Explicit exclusions and stop rule

Exclude observed-map deconvolution; LMTOOF optics/aberration implementation;
trial-aberration-dependent transfer construction; changes to mature RTC/PTC/
JINC/fruit algorithms; automatic real/radial/unity-normalized projection;
unmeasured high-frequency extrapolation; cross-band/mode substitution; and
any new production default or consumer authorization.

Stop after committing and verifying the registration and frozen audit packet.
Return the exact coordinator branch, commit(s), parent/tree, clean state,
artifact paths and SHA-256 values, proposed package axes, dependency/handoff
inventory, resource allocation, cost classification, unresolved choices, and
the explicit statement that no audit was launched and no application,
validation, external, repair, re-audit, integration, production, merge,
rebase, or push action occurred.
