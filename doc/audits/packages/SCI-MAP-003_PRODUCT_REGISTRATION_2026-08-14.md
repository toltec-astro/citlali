# SCI-MAP-003 OOF residual-transfer product registration

Record ID: `SCI-MAP-003-REG-D001`

Date: 2026-08-14

Status: registered; not dispatched

Owner decision: create one durable Tier A independent audit -> separately
authorized bounded repair -> separately launched fresh independent re-audit
product for an observation/reduction-specific OOF residual transfer function.
This record does not launch any lifecycle phase.

## Product identity and governing source

- Package ID: `SCI-MAP-003`
- Name: OOF residual transfer-function estimation and product
- Tier: A -- full scientific contract
- Queue position:
  `registered_unlaunched_map_product_after_SCI-MAP-002_before_SCI-MODE-001_LMTOOF_consumer_admission`
- Canonical coordination base:
  `codex/coordinate-rtc-ptc-queue` commit
  `8fc3b3dc549532f254bc814ef76f9a606c2a8059`, parent
  `87af719885fe73c1f292c97efb92026c4405ba05`, tree
  `401c0b0e7c9da0577c7164d636030cfffed1fdf7`
- Governing application source selected at registration:
  `origin/codex/refactor-mainline` commit
  `46ad23888a40f5102cdfd50c06e49a549bdf8a20`, parent
  `4846fa4db39bd2f7d4ddc41f693836834cbc5ff4`, tree
  `ab230a93b8fb310d58aefd7ac5da92e5d5e0f408`
- Registration authority: the documentation-only scientific-audit
  coordination commit containing this record; its exact commit, parent, tree,
  and artifact digest are reported at the mandatory registration checkpoint.

The governing-source rule is exact-SHA and fail-closed. A future dispatch must
revalidate the ref and SHA. If the application authority has changed, the
coordinator must issue an explicit amendment and refreeze the packet; an
auditor may not silently substitute another source. The unpushed historical
handoff branch and the owner-accepted but unintegrated `SCI-MAP-002` successor
are not application-mainline authority for this registration.

## Tier decision

Tier A is required because the unresolved reference mixture and amplitude,
complex phase and centering, denominator conditioning and admitted response
domain, morphology and linearity limits, spatial/iteration dependence, and
fruit-loop/LMTOOF consumer identity can change scientific meaning. These are
already promotion triggers under the audit framework; they are not narrowed
in advance merely to retain Tier B.

This tier does not reopen mature RTC, PTC, or JINC numerical internals by
default. Source inspection may open them only where the independent contract
and interface evidence identify a specific scientific conformity question.

## Bounded transformation and starting estimator

For one exact OOF/JINC product cell, the bounded transformation is:

```text
exact discrete APT-Gaussian reference g
    + final normalized OOF/JINC kernel k
    + exact parent/configuration/iteration identity
    -> complex two-dimensional residual transfer H and validity domain
    -> declared LMTOOF forward-model consumer interface
```

The owner-approved starting estimator is

\[
G = \operatorname{FFT2}(g), \qquad K = \operatorname{FFT2}(k),
\]

\[
H = \frac{G^*K}{|G|^2}
\]

only on a separately admitted denominator domain, with the forward prediction

\[
P_{\mathrm{pred}}(\theta)
  = \operatorname{IFFT2}\!\left(H\,\operatorname{FFT2}
    (P_{\mathrm{opt}}(\theta))\right).
\]

This is system identification from a known tracer and its final response. It
is not observed-map deconvolution. `H` is constructed once for an exact OOF
observation/reduction, held fixed throughout that LMTOOF solve, and
recomputed from a later observation/reduction after a telescope correction.
Trial aberrations are never inputs to transfer construction.

The ordinary proposed claim is only the map-center, linearized residual
transfer for the exact OOF/JINC product cell. The initial product retains
two-dimensional complex phase and measured DC/amplitude unless a later audit
and owner decision explicitly authorize a narrower derived representation.

## Included audit scope

The future independent audit must derive, assess, or identify owner decisions
for:

1. the exact discrete reference `g`, including APT rows or detector mixture,
   amplitude, truncation, centering, pixelization, and its relation to the
   LMTOOF nominal optical model;
2. the exact final kernel `k`, including map/JINC/fruit-loop parentage,
   normalization, processing state, and validity;
3. complex versus real and two-dimensional versus radial product identity;
4. FFT centering, frequency/WCS axes, units, normalization, padding, cropping,
   windowing, and Hermitian/real-inverse rules;
5. denominator reference, threshold, admitted spatial-frequency band,
   taper/exclusion, and invalid/unavailable state;
6. DC/amplitude preservation and any separately represented LMTOOF nuisance
   amplitude;
7. approximation limits in position, morphology, aberration magnitude,
   linearity, band, mode, and iteration;
8. persisted representation, association, parent digests, provenance,
   status, required/optional publication, and failure policy;
9. LMTOOF fail-closed consumer behavior, including fixed-within-solve and
   recomputed-between-OOF-cycles identity; and
10. local, retained, external, and telescope-gain evidence needed for
    validation and production adoption.

Validity domain, denominator conditioning, axes, centering, padding/windowing,
units, parentage, and failure/status semantics are estimator facts. They may
not be selected silently as implementation details.

## Ownership boundaries and dependencies

- `SCI-MAP-001` owns shared map-product identity, WCS/unit/validity,
  parentage, required-output, and publication vocabulary. Its approved
  contract is usable authority; its current implementation and open
  dependencies are not silently declared conformant here.
- `SCI-MAP-002` owns JINC deposition and the final `K/C` kernel identity under
  `SCI-MAP-002-D003-KERNEL-001`. It supplies the numerator. Its accepted third
  successor remains unintegrated, so conformity of the selected application
  SHA is not inferred from that acceptance.
- `SCI-RTC-001` owns exact source-kernel construction and RTC response parity.
  Its approved D002/D003 response and identity rules may be authority; its
  package implementation remains nonconformant.
- `SCI-PTC-001` owns conditioned/local cleaning response and response-family
  restrictions. D003 is bounded authority; the optional transfer-
  characterization plan is unlaunched context, not primary ownership.
- `SCI-VAL-001` owns applicable validity and eligibility facts. Its package is
  not started, so the dependency remains open.
- `SCI-AST-001` owns map-frame, WCS, axis, orientation, pixel-scale, and
  centering validity. Its approved bounded coordinate contract may be
  authority while implementation and ALIGN-dependent placement remain open.
- `SCI-FRUIT-001` owns final iteration/pass identity and the relation between
  feedback state and the delivered map/kernel. Its package is not started.
- `SCI-MODE-001` owns OOF product association and LMTOOF consumer policy. Its
  package is not started; this registration authorizes no LMTOOF use.

Every dependency must state its exact required fact and consequence in the
future frozen packet. Open dependencies do not prevent an independent audit,
but they keep the new product and LMTOOF consumer fail-closed.

## Product identity, status, and consumer restrictions

The future product must bind at least the exact observation/reduction,
application and algorithm identities, TolTEC band and map slot, APT/detector
selection, RTC/PTC/JINC requested/effective/observation-resolved/realized
state, source and validity masks, fruit-loop iteration/pass and terminal
state, parent map/kernel identity and digest, denominator policy, valid-
frequency domain, representation, and completion status.

Response status must distinguish at least:

- `computed_published`;
- `not_computed_or_not_requested_for_this_product`;
- `invalid`; and
- `unavailable`.

Kernel-disabled, missing-parent, identity-mismatch, invalid-denominator, and
incomplete-product cases must fail closed according to an explicit required
or optional publication policy.

Registration production status is `existing_use_only` for existing OOF.
The residual-transfer product and every LMTOOF use are new consumers and are
not authorized. Existing no-transfer OOF use is not made dependent on this
future refinement.

## Explicit exclusions

- observed-map deconvolution;
- LMTOOF optical, aperture, illumination, passband, or aberration-model
  implementation;
- trial-aberration-dependent transfer construction;
- changes to mature RTC, PTC, JINC, mapmaking, or fruit-loop algorithms;
- automatic real projection, radialization, unity normalization, or removal
  of measured DC/amplitude;
- unmeasured high-frequency extrapolation;
- cross-band or cross-mode substitution;
- off-center, spatially varying, arbitrary-morphology, extended-source, or
  nonlinear claims without separately admitted evidence;
- a new production default, consumer authorization, or LMTOOF adoption; and
- modification or import of the informal prototype files.

## Lifecycle and role separation

### Independent audit

A separate owner launch must create a fresh audit branch, worktree, and task.
The auditor must freeze an independent scientific core before opening
quarantined implementation paths or post-core evidence. Only exact approved
contract facts named in the frozen manifest may be opened as
`pre_core_authority`. Current source traces, unaccepted audit conclusions,
optional plans, historical implementation observations, and informal
prototypes remain `post_core_evidence`.

### Bounded repair

Repair requires a separate owner disposition after the audit. Its dispatch
must name the exact accepted contract, base SHA, paths, findings, tests,
evidence, consumer restrictions, and stop rule. Registration supplies no
repair authority and does not preselect a representation, threshold, format,
or implementation.

### Fresh independent re-audit

Re-audit requires a new role-separated task and worktree after an authorized
repair. It must assess the exact successor, repeat every applicable admitted
fixture, disposition all handoffs and open findings, and preserve the
consumer fail-closed boundary until a separate production decision.

## Proposed evidence and cost classification

The future audit should preregister identity, signed-scalar, integer/subpixel
shift and phase, box/sinc, JINC/Hankel, asymmetric-filter, forward-closure,
denominator-boundary, odd/even shape, FFT centering, WCS orientation,
Hermitian/real-inverse, and deliberate identity-mismatch fixtures.

Those bounded analytic and synthetic cases are provisionally `not_costly`
under `FRAMEWORK-NUM-001`: they are expected to use small fixed arrays with no
Citlali reduction, external scheduling, scarce data, or material storage.
This is a planning classification, not execution authority. Scope expansion
or material resource exposure requires reclassification before execution.

Any local Citlali reduction, broad injection campaign, LMTOOF execution,
Unity evidence, telescope correction, or iteration-to-gain study is costly
and held. It requires a later scope checkpoint, the complete applicable
`FRAMEWORK-NUM-001` controls, and separate human/external authorization.

The two informal prototypes are recorded only as post-core design evidence:

| File | SHA-256 |
| --- | --- |
| `/Users/gwilson/foo/transfer_function/test_kernel_jinc.py` | `65d5f1ee95f09ca290594bbf239950bce932e56d6186f9818102beee6311d1dd` |
| `/Users/gwilson/foo/transfer_function/test_kernel_2.py` | `b16ba174cbd75dc6685332f7187f33038cad537a845befd4c52ef98d08f91824` |

Their cross-spectrum, JINC/Hankel, and box/sinc mechanics are design evidence.
Hard-coded parameters, synthetic normalization, analytic denominator
shortcut, real-only projection, and radial plots are not production authority.

## Resource plan

Under `FRAMEWORK-EFFORT-001`:

- registration and future dispatch preparation: Terra High;
- Tier A launch checkpoint and source inventory: Terra High;
- independent scientific core: Sol Max;
- implementation, product, metadata, and consumer tracing: Sol XHigh;
- scientific decision synthesis: Sol XHigh, with Sol Max reserved for one
  remaining coherent contradiction;
- mechanical validation, if later authorized: Terra Medium; and
- validation interpretation: Terra High or Sol High according to whether it
  can change scientific meaning.

Ultra, delegation, and parallel workstreams are not authorized by this
registration. A later change requires its own written trigger, independence
boundaries, synthesis owner, and stop rule.

## Phase 2 inbound packet checkpoint

The following eight inbound handoffs are prepared in the accepted dependency
order. They are uncommitted coordination artifacts for review, not audit
dispatch authority:

1. `SCI-MAP-003-XAUD-001` -- `SCI-MAP-001` -- `pre_core_authority`;
2. `SCI-MAP-003-XAUD-002` -- `SCI-MAP-002` -- `pre_core_authority`;
3. `SCI-MAP-003-XAUD-003` -- `SCI-RTC-001` -- `pre_core_authority`;
4. `SCI-MAP-003-XAUD-004` -- `SCI-PTC-001` -- `pre_core_authority`;
5. `SCI-MAP-003-XAUD-005` -- `SCI-AST-001` -- `pre_core_authority`;
6. `SCI-MAP-003-XAUD-006` -- `SCI-VAL-001` -- `post_core_evidence`;
7. `SCI-MAP-003-XAUD-007` -- `SCI-FRUIT-001` -- `post_core_evidence`; and
8. `SCI-MAP-003-XAUD-008` -- `SCI-MODE-001` -- `post_core_evidence`.

The frozen inbox authority manifest is
`doc/audits/handoffs/SCI-MAP-003/SCI-MAP-003_INBOX_AUTHORITY_MANIFEST_2026-08-14.yaml`,
SHA-256
`a6c36b7c0416e1f03ce88b8004712db666c625a95cf521787cfad5ad28d27603`.
Its status is
`phase_2_uncommitted_frozen_for_coordinator_review_not_dispatch_authority`.
It contains 9 `pre_core_authority` objects and 13
`post_core_evidence` objects. Only the exact owner-approved abstractions
explicitly admitted by the manifest may be opened before an independent core
freeze. Open or conditioned dependency consequences remain unchanged and
must not be promoted into accepted implementation authority.

No independent-audit prompt or coordinator-return artifact exists. The
packet is not dispatchable and no audit has been launched.

## Registration disposition and next checkpoint

- contract status: `not_started`
- implementation status: `not_assessed`
- validation status: `not_started`
- production status: `existing_use_only`
- verdict: `pending`
- remediation branch/commit: null; repair not authorized
- re-audit status: `not_started`
- independent audit: not launched

This uncommitted Phase 2 packet is the next viable artifact. Stop after
verifying this record, the ledger bindings, the eight inbound handoffs, and
the frozen authority manifest. Obtain coordinator acceptance before any
commit and before creating an independent-audit prompt or coordinator-return
artifact.

This record does not authorize application, test, configuration, validation,
or LMTOOF edits; numerical execution; a local or Unity reduction; an external
request or contact; audit launch; repair; re-audit; integration; production
adoption; merge; rebase; or push.
