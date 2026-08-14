# SCI-MAP-003 registration coordinator return

Date: 2026-08-14

Status: final packet prepared; not launched

This return closes documentation preparation for Grant's separate launch
decision. It records reproducible Phase-1 and Phase-2 identities and the
Phase-3 identities available before the final commit. It is not an audit
launch, scientific decision, numerical authorization, evidence request,
repair disposition, integration decision, or production approval.

## Coordination identity

- branch: `codex/coordinate-sci-map-003-registration`
- worktree: `/private/tmp/citlali-coordinate-sci-map-003-registration`
- Phase-3 preparation base:
  `b3e500117ab273d38d5055e82683a49015d028cf`
- governing application ref: `origin/codex/refactor-mainline`
- governing application SHA:
  `46ad23888a40f5102cdfd50c06e49a549bdf8a20`
- governing application parent:
  `4846fa4db39bd2f7d4ddc41f693836834cbc5ff4`
- governing application tree:
  `ab230a93b8fb310d58aefd7ac5da92e5d5e0f408`
- governing-source rule: a separate launch must revalidate the ref and exact
  SHA; any change requires an explicit coordinator amendment and refrozen
  packet, never silent substitution.

## Phase 1: accepted registration

- commit: `5c5a7fad052ac433207c1c6c61355b53c61a476c`
- parent: `8fc3b3dc549532f254bc814ef76f9a606c2a8059`
- tree: `3caafdaef58941758772d41df256f8c5f6cb13e0`
- standard binary commit-patch SHA-256:
  `e3d90fbfbff295eeb268cbdf5684d85b28df8e3dda411b71de4d50d86b275bbf`
- diffstat: 3 files changed, 539 insertions, 1 deletion

| Artifact | SHA-256 |
| --- | --- |
| `doc/audits/README.md` | `3274bbdd8ccc92cd14dab3c4b4556b7b2a0a620afd3530c2a31b9ffe62be082f` |
| `doc/audits/audit-ledger.yaml` | `28449cefead083a8b9867a3f4efd7efafd3430a35ab6a803b27fabf5aecb3d26` |
| `doc/audits/packages/SCI-MAP-003_PRODUCT_REGISTRATION_2026-08-14.md` | `b3c81a9186e3fb68e0209654427b14dcf81793a832bba47e8357d262c0072ae8` |

Phase 1 registered the durable Tier A product, governing SHA and fail-closed
source rule, queue position, eight dependency consequences, status axes,
resource/cost classifications, exclusions, lifecycle, and non-authorizations.
It did not create or launch a handoff packet or audit.

## Phase 2: accepted manifest and handoffs

- commit: `b3e500117ab273d38d5055e82683a49015d028cf`
- parent: `5c5a7fad052ac433207c1c6c61355b53c61a476c`
- tree: `9e79dde73df9c3655e43f35e1beebb5d1413603a`
- standard binary commit-patch SHA-256:
  `5b0ba32cebc76726c255fd78b7a30ca67f1f0e8dd053a64bd07e074440431f8e`
- diffstat: 11 files changed, 1,158 insertions, 12 deletions

| Artifact | SHA-256 |
| --- | --- |
| `doc/audits/audit-ledger.yaml` | `69129a80c7b74a27708b01150657fa492c04f9f0215fe4d8693f2337ed2d7d3e` |
| `doc/audits/packages/SCI-MAP-003_PRODUCT_REGISTRATION_2026-08-14.md` | `d87d3b9e2252c7ade3cbd1c978c835575d14d2038efcb732171103d383dad281` |
| `doc/audits/handoffs/SCI-MAP-003/SCI-MAP-003_INBOX_AUTHORITY_MANIFEST_2026-08-14.yaml` | `a6c36b7c0416e1f03ce88b8004712db666c625a95cf521787cfad5ad28d27603` |
| `doc/audits/handoffs/SCI-MAP-003/SCI-MAP-003-XAUD-001.yaml` | `02601475819b1e64df0975098c70898f0db7b4d3b975eedaa5f3125692f8f809` |
| `doc/audits/handoffs/SCI-MAP-003/SCI-MAP-003-XAUD-002.yaml` | `9708a2ddd0a1b95079037cee110a2c51c0b26bc20f349b09fd817a48a024035d` |
| `doc/audits/handoffs/SCI-MAP-003/SCI-MAP-003-XAUD-003.yaml` | `1db43ab766d4f80ec691fdffac4f32b064b76f196a2ad25df030a62b69b2f884` |
| `doc/audits/handoffs/SCI-MAP-003/SCI-MAP-003-XAUD-004.yaml` | `20d73f69f9c4413da970e96b6d3c4e85a3426550482df6e37b06f164db3c1893` |
| `doc/audits/handoffs/SCI-MAP-003/SCI-MAP-003-XAUD-005.yaml` | `27f36350feb0a126f5045d6b64a98afc44ccaf3c339f102c30cc390daf6779ac` |
| `doc/audits/handoffs/SCI-MAP-003/SCI-MAP-003-XAUD-006.yaml` | `877f9cec7f7f9fd698a0226289effbc6ab46315aa280b60cba283e05aedba6ed` |
| `doc/audits/handoffs/SCI-MAP-003/SCI-MAP-003-XAUD-007.yaml` | `801928c460745b748163027191d20f5873110db937aa508e95542206e5623498` |
| `doc/audits/handoffs/SCI-MAP-003/SCI-MAP-003-XAUD-008.yaml` | `c84cb142813333c0499bde7275d61f940f593fd92875d527bf420bb4003d0087` |

The accepted manifest closes exactly 22 objects: 9 pre-core authority and 13
post-core evidence, disjoint and complete. All 34 referenced Git/local source
objects were verified by exact ref/path/digest. Phase-2 handoff bytes remain
the immutable authority/evidence freeze; Phase 3 does not change any handoff.

## Phase 3: uncommitted final-packet identity

Authorized path ceiling:

1. `doc/audits/handoffs/SCI-MAP-003/SCI-MAP-003_INBOX_AUTHORITY_MANIFEST_2026-08-14.yaml`;
2. `doc/audits/packages/SCI-MAP-003_PRODUCT_REGISTRATION_2026-08-14.md`;
3. `doc/audits/audit-ledger.yaml`;
4. `doc/audits/prompts/SCI_MAP_003_AUDIT_PROMPT.md`; and
5. `doc/audits/packages/SCI-MAP-003_REGISTRATION_COORDINATOR_RETURN_2026-08-14.md`.

The audit prompt was finalized before the registry bindings:

- path: `doc/audits/prompts/SCI_MAP_003_AUDIT_PROMPT.md`
- uncommitted SHA-256:
  `306b201c81526da203ae671e189f70b489d8c8a595cdc6aa1e502fab479564bc`
- status: prepared for separate owner launch decision; not launched.

This coordinator return's own SHA-256 cannot appear inside its own bytes.
Likewise, its digest is needed before the manifest, registration, and ledger
can bind it, so their final Phase-3 digests and the full five-path patch digest
cannot be recorded here without creating a digest cycle. At the uncommitted
checkpoint, the external coordinator response must report:

- this return's exact SHA-256;
- the final manifest, registration, and ledger SHA-256 values after binding
  the prompt and return;
- the normative temporary-index full-patch SHA-256 and diffstat; and
- confirmation that the real index stayed empty and only these five paths
  changed.

After a separately authorized commit, the external coordinator response—not
this self-referential artifact—must add the exact Phase-3 commit, parent
`b3e500117ab273d38d5055e82683a49015d028cf`, tree, standard binary commit-
patch SHA-256, committed paths, artifact hashes, and clean state. These values
are `pending_final_commit`; they are not guessed here.

## Unresolved scientific decisions

The independent audit must resolve or return explicit owner choices for:

1. exact discrete reference `g`, including APT/detector mixture, amplitude,
   truncation, centering, pixelization, and nominal-optical-model relation;
2. exact final `k` and map/JINC/fruit-loop parent, normalization, processing
   state, and validity;
3. complex/real and two-dimensional/radial product identity;
4. FFT centering, axes, units, normalization, padding/cropping/windowing, and
   Hermitian/real-inverse rules;
5. denominator threshold/domain, band, taper/exclusion, and invalid or
   unavailable state;
6. DC/amplitude preservation and any LMTOOF nuisance amplitude;
7. position, morphology, aberration, linearity, band/mode, and iteration
   approximation limits;
8. persisted representation, association, parent digests, provenance, typed
   status, publication requirement, and failure policy;
9. LMTOOF fixed-within-solve and recomputed-between-OOF-cycles behavior; and
10. local, retained, external, and telescope-gain validation evidence.

No choice above is silently resolved by packet preparation.

## Axes, dependencies, resources, and cost

Initial package axes remain:

```text
contract_status: not_started
implementation_status: not_assessed
validation_status: not_started
production_status: existing_use_only
verdict: pending
```

Dependency states remain `SCI-MAP-001 open`, `SCI-MAP-002 conditioned`,
`SCI-RTC-001 open`, `SCI-PTC-001 conditioned`, `SCI-AST-001 conditioned`,
`SCI-VAL-001 open`, `SCI-FRUIT-001 open`, and `SCI-MODE-001 open`, with the
exact required facts and fail-closed consequences frozen in the registration,
ledger, manifest, and prompt.

The `FRAMEWORK-EFFORT-001` plan remains Terra High for registration/launch
checkpoint, Sol Max for the independent core, Sol XHigh for source/product/
consumer tracing and ordinary synthesis, Terra Medium for separately
authorized mechanical validation, and Terra High or Sol High for validation
interpretation. Sol Max is reserved for one coherent remaining contradiction.
Ultra, delegation, and parallelism are not authorized.

The small fixed analytic/synthetic fixture plan is provisionally
`not_costly`, proposed and not executed. The reduction/LMTOOF/Unity/telescope-
gain plan is `costly`, held, and requires all applicable `FRAMEWORK-NUM-001`
controls plus separate human/external authority. Registration and prompt
preparation execute neither plan.

## Readiness and non-launch truth

The documentation packet is ready for Grant's separate launch decision only
after the five-path checkpoint is independently accepted and committed, and
after the launch-time coordinator revalidates the exact packet commit,
manifest digest, application ref/SHA, worktrees, and clean state. A launch
must create a fresh role-separated audit branch/worktree and begin with the
prompt's mandatory scope checkpoint.

Packet readiness does not mean scientific readiness or production adoption.
Existing OOF remains `existing_use_only`. The new transfer product and every
LMTOOF use remain fail-closed and unauthorized.

No audit was launched. No independent core was started or source/post-core
evidence opened for an audit. No numerical study, fixture, Citlali or LMTOOF
execution, local or Unity reduction, telescope correction, external request
or contact, application/test/configuration/validation change, repair,
re-audit, integration, production action, merge, rebase, or push occurred.
