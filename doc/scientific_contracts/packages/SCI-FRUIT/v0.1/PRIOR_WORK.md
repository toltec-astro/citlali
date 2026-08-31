# SCI-FRUIT v0.1 — Prior-Work Recovery

This recovery record follows the
[Citlali Scientific Contract Library Program](../../../README.md). It is an
internal Stage A artifact and is not part of a scientific-author packet.

Status: **review candidate**

Investigator/date: Codex, `2026-08-31`

Scope revision examined: launch commit
`7f9307ff4e1cda0f112f2398bb72f52a3f4f01d5`; provisional matched-filter
snapshot `faff97565ee27e375e1337febe5a0a6681507c3b`; historical coordination
snapshot `8c581bfb26f01b187f4f1e0565f4457bcc25f099`; historical fruit-calibration
reference `f70701ad488444f3e2528c6bbe3e798863c9e301`.

## Program Adherence And Prior-Work Recovery

Recovery started from the program
[`PRIOR_WORK_REGISTRY.md`](../../../PRIOR_WORK_REGISTRY.md), then searched
current conventions, ADRs, frozen or approved adjacent packages, historical
investigations, audit handoffs, diagnostic tooling, current implementation,
configuration, validation products, and exact topic refs. New scientific
derivation has not been commissioned.

## Search Coverage

- Program: `README.md`, `PILOT_PROCESS_REVIEW_2026-08-16.md`,
  `DOWNSTREAM_CONTRACT_ROADMAP_2026-08-26.md`, `PRIOR_WORK_REGISTRY.md`, and
  package templates.
- Governing repository conventions: `doc/SCIENTIFIC_CONVENTIONS.md`,
  `doc/ARCHITECTURE.md`, and `doc/adr/0006-fruit-loop-restart-contract.md`.
- Adjacent science: frozen SCI-MAP v0.1/r0.7.1, frozen SCI-JINC v0.1/r0.3,
  approved SCI-NOI Stage A including ODQ-110C, frozen SCI-PTC v0.1/r0.5, and
  conditionally frozen SCI-FLT-FIXED v0.1.
- Provisional adjacent work: the exact SCI-FLT-INF/FLT-MATCHED holding study at
  `faff97565...`; no other ref or later state was inspected for its content.
- Historical FRUIT studies: feedback, convergence, convergence-criteria,
  calibration-reference, and population-extension documents; associated
  handoffs; `validation/fruit_loop_*`; and `tools/fruit_loops/`.
- Historical audit/coordination: exact `SCI-FRUIT-001` ledger and cross-package
  handoffs at `8c581bfb...`.
- Current code at the launch base: FRUIT iteration policy, engine feedback
  paths, map loading/projection, reduction learning lifecycle, PTC weight
  validation, and restart checkpoint structures. These are quarantined in the
  internal dossier.
- No Unity execution, external reduction, new validation run, or unpublished
  external source was used.

## Recovered Materials

| Material and exact reference | Classification | Scientific content already available | Limitations or conflicts | Disposition |
| --- | --- | --- | --- | --- |
| `doc/SCIENTIFIC_CONVENTIONS.md` at launch base | governing convention | Absolute zero-based fruit iteration; completed `N` resumes at `N+1`; requested/effective/realized separation; significance and morphology distinctions | Does not define the FRUIT estimator, model, update, or stop law | **adopt** exact identity and distinctions |
| `doc/adr/0006-fruit-loop-restart-contract.md` at launch base | accepted architecture decision | `path` is map-only seeding; `restart_path` is exact continuation; absolute exclusive `max_iters`; v2 checkpoint includes PTC validation state | Operational completeness is implementation-informed, not a scientific proof that every future state is complete | **adopt** restart identity; **cite** v2 history |
| SCI-MAP v0.1/r0.7.1 freeze | frozen authority | Ordinary observation/coadd product roles, response and validity boundaries, typed numerical unavailability | A normalized map is not automatically a feedback model; numerical route remains gated | **cite** as candidate-parent authority |
| SCI-JINC v0.1/r0.3 freeze | frozen authority | Signed JINC observation-map estimator, grouping, response/covariance roles, validity | Base route has no generally admitted numerical realization and does not make the map a sky model | **cite** as separate candidate route |
| SCI-PTC v0.1/r0.5 authority and MAP/JINC handoffs | frozen authority/owner decision | PTC owns cleaning and coefficient families; FRUIT owns recurrence/add-back and its state interaction | FRUIT must not redefine PTC science or infer unavailable coefficient families | **adopt** ownership boundary |
| SCI-NOI Stage A owner approvals and `FILTER_AND_FRUIT_SCOPE.md` | approved adjacent authority | FRUIT owns model, recurrence, selection, response, stopping/restart; fixed-state, successor-generation, and replay uncertainty routes are distinct | No fixed or replayed FRUIT numerical uncertainty route is admitted before FRUIT authority/parity exists | **adopt** ownership and generation boundary |
| SCI-FLT-FIXED conditional freeze at `7f9307ff...` | conditional frozen authority | Fixed deterministic transformation and exact typed unavailable parent/profile states | Does not admit ordinary MAP/JINC numerical parents, profiles, Registry route, or inference filter | **cite** without modification |
| SCI-FLT-INF holding study at exact `faff97565...` | provisional owner-decision study | Historical method separated as matched-template amplitude filtering; ordinary MAP observation/coadd roles distinguished; learned-state/NOI graph still open | Not an approved package, author packet, Stage B launch, or scientific authority; numerical route unavailable | **defer** and label provisional; never promote |
| `FRUIT_LOOP_FEEDBACK_INVESTIGATION_2026-07-24.md` | historical implementation/validation evidence | Controlled ablations separated projection, PTC recovery, support, and restart effects; v2 restart equality observed | Current algorithm behavior is not science authority; v1 trajectory was invalid because state was incomplete | **abstract** questions; **exclude** from authorship |
| `FRUIT_LOOP_CONVERGENCE_STUDY_2026-07-23.md` | historical validation evidence | Map-only continuation resets learning and cannot extend the primary sequence; multiple change metrics were measured | No approved stopping rule; historical sample and implementation specific | **abstract** metric taxonomy; **exclude** thresholds |
| `FRUIT_LOOP_CONVERGENCE_CRITERIA_DISCUSSION_2026-07-27.md` | historical scientific discussion/evidence | Amplitude, PSF, centroid, map, support, learning, and noise convergence are not interchangeable; candidate 3% unresolved-source criterion explored | Population contains measurement-limited and unresolved trajectories; no owner-approved universal rule | **abstract** separations; **defer** thresholds |
| `FRUIT_LOOP_CALIBRATION_REFERENCE_INVESTIGATION_2026-07-26.md` | historical evidence | Stable endpoint supported qualified relative astrometry/effective-PSF use; no universal photometric correction; science response unmeasured | One position/amplitude/observation cannot define calibration or stopping | **cite** limitations internally; **exclude** calibration claims |
| `FRUIT_LOOP_POPULATION_EXTENSION_PLAN_2026-07-26.md` and handoffs | historical operational evidence | Quality-stratified population and point/planet separation were recognized | Unity-specific execution and empirical results do not define authority | **abstract** validation strata; **exclude** operations |
| `validation/fruit_loop_*` and `tools/fruit_loops/` | validation/tooling evidence | Reproducible diagnostic categories and trajectory bookkeeping | Not scientific authority; some products reflect obsolete or invalid checkpoints | **exclude** from authorship; retain for later conformance planning |
| debug notes dated 2026-03-05 and 2026-04-22 | historical debug evidence | Earlier hypotheses about gain, weighting, masks, add-back, and projection coupling | Implementation-specific; some hypotheses were later superseded or bounded | **supersede/defer**; exclude from authorship |
| `SCI-FRUIT-001` ledger and handoffs at `8c581bfb...` | historical coordination/audit evidence | FRUIT owns terminal pass, restart, parent-state relation, and final delivered map/kernel identity; dependency was open | Post-core evidence, explicitly not pre-core science authority | **abstract** closure questions; **exclude** audit prose |
| Current code/config at `7f9307ff...` | implementation evidence | Hard maximum iteration, model subtraction/PTC/add-back/mapmaking sequence, learning and weight-validation state, checkpoint fields | Behavior may be incomplete or scientifically wrong and cannot enter author packet | **quarantine** in `INTERNAL_DOSSIER.md` |

## Recovery Synthesis

### Questions already answered

- Iteration labels are zero-based and absolute across exact restart; exact
  restart continues after the completed iteration. Source: governing scientific
  conventions and ADR 0006.
- Map-only seeding and exact restart are different lifecycle operations.
  Source: ADR 0006.
- Requested configuration, effective plan, observation-resolved state, and
  realized state are distinct. Source: scientific conventions and architecture.
- FRUIT—not NOI—owns its model, recurrence, state, stopping, restart, response,
  support, validity, and failure. Source: approved SCI-NOI ODQ-110C boundary.
- Fixed-state conditional uncertainty, an NOI-informed successor generation,
  and per-realization replay are non-equivalent methods. Source: approved
  SCI-NOI boundary.
- A formal standardized quantity, a formal fit significance, empirical
  blank-sky point-source S/N, and legacy dynamic range are non-equivalent.
  Source: scientific conventions.
- No recovered authority approves a universal FRUIT photometric correction or
  stopping threshold.

### Reusable definitions, equations, and reasoning

The absolute iteration/restart definitions, requested/effective/realized state
separation, and adjacent-package ownership can be cited or copied into a future
sanitized conventions extract after owner review. The exact NOI successor graph
may be cited as an adjacent boundary. MAP/JINC/FLT scientific products must be
referenced through exact frozen authority, not restated from implementation.

The historical studies may justify asking separate questions about amplitude,
morphology, centroid, map change, support/learning, and noise health. Their
numerical trajectories and thresholds are not suitable author inputs because
they reveal implementation/validation behavior and do not carry scientific
authority.

### Conflicts and unresolved choices

- Historical code subtracts/adds a loaded map, while the scientific boundary
  requires a separately typed feedback-model product. The owner must define the
  model construction and accumulation law.
- Historical runs terminate at a hard maximum; discussion documents explored
  change metrics. Neither establishes a scientific stop rule or terminal
  selection policy.
- Exact restart v2 restores the current operational state known to affect
  output, but future scientific choices may add state that changes the
  completeness set.
- Existing outputs use terms such as signal-to-noise for non-equivalent
  quantities. FRUIT must not import these as one convergence metric.
- The four candidate parent families have different estimands, grouping,
  response, covariance, support, lifecycle, and availability. A single generic
  `type` choice would erase scientific identity.
- Historical audit inventory placed FRUIT last and made filtered inputs fail
  closed; the owner has now changed sequencing to launch FRUIT after filtering.
  That sequencing change does not itself admit any filtered route.

### Material excluded from authorship

All code, schemas, configuration names/defaults, debug notes, audits, repairs,
tests, validation outputs, empirical trajectory values, Unity handoffs, and
production-history claims are excluded. The exact provisional FLT-MATCHED
study is not an author authority; at most a future owner decision could admit a
sanitized, source-bound boundary extract after that study itself is approved.

### Genuinely new work

The smallest remaining scientific work is to choose and define:

1. the primary feedback-model estimand and update/accumulation law;
2. admitted parent route or routes and the model-construction operator for each;
3. forward projection, subtraction/add-back semantics, response, and null space;
4. fixed versus learned/relearned state and its generation graph;
5. support, selection, validity, failure, and terminal product bundle;
6. diagnostic convergence versus stopping and terminal-selection rules;
7. uncertainty/covariance scope, including fixed-state and replay variants; and
8. exact checkpoint completeness and compatibility rules implied by those
   choices.

## Proposed Author Reference Packet

No author packet is approved. The proposed sanitized set and its exclusions are
listed in [`PROPOSED_SANITIZED_AUTHOR_INPUTS.md`](PROPOSED_SANITIZED_AUTHOR_INPUTS.md).
It must not be dispatched until exact owner decisions, byte identities, and a
supersession cover are approved.

## Investigator Attestation

Prior work was recovered before new derivation was commissioned. Existing
authority is cited rather than rewritten. Implementation and empirical behavior
have not been promoted to scientific authority. Conflicts and unavailable
states remain explicit rather than being filled from current behavior.
