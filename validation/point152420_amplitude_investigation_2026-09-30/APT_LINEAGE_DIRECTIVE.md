# Codex directive: explain the APT-v2 calibration divergence in POINT152420

**Owner:** Grant Wilson  
**Date:** September 30, 2026  
**Scope:** POINT152420, a1100; successor B, historical redu09, and their actual calibration-production, selection, association, and consumption paths.  
**Priority:** Establish the origin and legitimacy of the calibration-factor difference before further map-level attribution. Q remains parked; neither reduction's photometry is accepted by this directive.

> We invested substantial effort in APT v2 to make calibration identity, ancestry, and use unambiguous. Explain how these two reductions acquired different calibration factors, whether each followed the workflow appropriate to its role, and why the existing safeguards did or did not identify the difference. “The APT factors differ” is the starting observation, not the explanation.

## 1. Immediate direction and current-state recovery

This supplements the September 30 amplitude-investigation directive and preserves the September 29 bounded-development authority. It changes the immediate ordering: **trace calibration origin and integration first; measure downstream consequences second.** Do not substitute a calibration-swap experiment for this investigation.

Recover current HEAD, worktree status, applicable repository instructions, the existing investigation ledger, and actual process/batch state. Do not reset to a previously reported revision or overwrite subsequent work. Discovery anchors are:

```text
~/.codex/worktrees/ptc-rank-response/citlali-refactor
validation/point152420_amplitude_investigation_2026-09-30/
validation/point152420_map_comparison_2026-09-29/
validation/rtc_point_transfer_2026-09-28/
~/work_toltec/local_data/citlali-validation/successor-point152420-amplitude-20260930/
~/work_toltec/local_data/citlali-validation/successor-rtc-point-transfer-completion-20260930/
```

The last supplied report described three prepared frozen-M0 controls and zero starts at that snapshot. **Verify present status rather than assuming that remains true.** If unstarted, defer those controls while performing this audit and record the deferral without rewriting their frozen identity. If already completed, preserve and account for them; use their findings without mistaking consequence for origin. If running, do not interrupt, overlap, or edit their bound code/inputs; continue independent read-only lineage work. Use an isolated local worktree for necessary development while an active batch's source must remain unchanged. Do not poll.

Reuse `ASSESSMENT.md`, `BATCH1.md`, `RESUME.md`, the calibration/provenance audits, receipts, and compact measurements already produced. Reported starting evidence is successor/historical common-detector factor medians around 0.40–0.45, a separately reported atmospheric gain around 1.0333, and final fitted peaks around 291.81/574.59 mJy per nominal beam. Verify what each factor represents; do not treat these summaries as the causal answer.

## 2. Recover the intended APT-v2 path and the role of this POINT reduction

Read the actual accepted APT-v2 specifications, implementation, acceptance evidence, and relevant changes through the revisions used by each run. Do not substitute an old scientific-rationale draft, current default, or previous assistant summary for implementation evidence.

An August 24 handoff reported this sequence: pointing reductions estimate a flux correction; TolProj assembles calibration evidence; Citlali issues/verifies the canonical calibrated APT-v2; TolProj binds it to the reduction. Locate and verify that work. Historical discovery leads include TolProj revision `2fe2af8fae681608893a0b491bd4b3e0b340ba03`, Citlali revision prefix `07b60205`, and Stage-7 validation. These are leads, not instructions to check out those revisions or assertions that either POINT152420 run used them.

Distinguish schema/version, artifact identity, detector-matching state, and calibration state using their actual definitions. **A v2-format table, an observation-matched table, and a calibrated child are not automatically interchangeable.** Recover which states are permitted for the specific POINT role in each workflow.

Determine whether POINT152420 was being reduced to derive calibration evidence, validate an existing calibration independently, or supply another downstream use. Trace any correction's evidence dependencies. Do not “repair” the difference by applying a correction derived from this same POINT measurement and then present agreement as independent validation. Conversely, do not reject a legitimate earlier-state POINT input merely because later science reductions require a calibrated child.

Establish the expected chain for each run before judging conformance. A historical run may legitimately predate or bypass v2 under its then-declared workflow; that must be demonstrated, not presumed. The question is whether its difference from the successor is explained and appropriate, not whether every historical product satisfies today's schema.

## 3. Trace requested, resolved, and consumed calibration side by side

Resolve the actual historical map path and follow its parent observation/reduction/project directories, configurations, scripts, logs, and dependency references. The owner states that redu09 provenance is upstream in that hierarchy. Do not stop at a copied comparison map or the redu09 leaf. For the successor, trace the full validation runner and adapters, not only the native CAL function.

Produce one compact side-by-side lineage table, with exact evidence references, covering:

| Boundary | Required evidence for each reduction |
|---|---|
| Observation and role | Observation/tune context, acquisition identity, POINT purpose, relevant cohort or explicit absence of one, and permitted APT state. |
| Source calibration | Selected beammap/source-APT identity, calibration evidence, factor meaning, units/reference plane, and selection rule or explicit override. |
| Matching and transformations | Target-to-source association, parent/child identities, matcher and transformation versions, correction evidence, and numerical operations. |
| Issuance and selection | Whether a canonical artifact was issued and verified; exact selected artifact/state; requested versus effective configuration and selection reason. |
| Runtime consumption | Actual executable/build and loaded Python implementation, invoked adapter/reader, resolved input, sample-column-to-row binding, and realized multiplier. |
| Admission safeguards | Checks that actually executed, checks only documented/tested elsewhere, explicit compatibility/diagnostic exceptions, and missing or bypassed checks. |

Recover selection from the effective configuration and invocation, including inherited templates, overrides, cached products, and resolved paths. An available new implementation is not proof that a run called it. Identify the actual module/executable used where environment or checkout differences could matter.

Locate the **earliest demonstrable divergence**. Possibilities include different source calibration, different child transformations, different row association, different selected state, or different interpretation/application of the same row. These are hypotheses to discriminate, not faults to assume. Follow an unexplained difference upstream until its generating measurement, selection decision, transformation, or precise evidence gap is identified.

Use existing canonical provenance pointers; follow them when necessary. Do not duplicate the entire beammap provenance into CAL, build a parallel APT registry, or introduce a second calibration authority merely to perform this audit.

## 4. Reconstruct the numerical difference from its actual lineage

Recheck the audit's definition of “common detector.” Establish correspondence through observation/tune-scoped identities and recorded associations. Equal network/channel numbers or row indices across different source observations are not sufficient by themselves. Separate genuinely common valid associations from missing, ambiguous, or unsupported ones; do not force a match to enlarge the comparison.

For a small deterministic representative set, preferably reusing the saved examples, trace the source-row factor through every embodied transformation to the selected row and the multiplier actually applied to representative native samples. Independently calculate the expected values from recorded inputs rather than calling the same unchecked conversion helper as both implementation and oracle.

Separate the source factor, any child correction, unit/native-signal conversion, and target-atmosphere term. Establish whether similarly named fields have the same role and direction in both implementations. Record which corrections are already embodied and which are applied at runtime. Check omissions, duplication, reciprocal conventions, and compatibility aggregates only against their declared meanings.

If the lineage says `F_child = P × F_parent`, reconstruct `P` from its actual recorded evidence and generating rule—not by fitting a multiplier to the observed discrepancy. Verify the resulting selected factors across the full supported common population, reporting per-network distributions and unexplained residual differences. Use a precision tolerance justified by the stored numbers and arithmetic, not a loose tolerance chosen to declare agreement.

Answer quantitatively: **Does the identified selection or transformation explain the observed factor ratios, for which detectors, and with what residual?** If not, continue upstream or identify the remaining specific gap. A median near 0.42 does not establish a universal factor, and a reconstructed factor difference does not by itself allocate the final map discrepancy.

Treat hashes as integrity evidence, not evidence of appropriate scientific selection. Two intact artifacts may represent different legitimate states, or one may be inappropriate for the requested role. Establish which case applies.

## 5. Explain the safeguard outcome and repair demonstrated defects

For the implicated boundary, compare the intended APT-v2 guarantee, the implemented check, and evidence that the check actually ran. Explain why the difference was not surfaced earlier: an explicitly different valid lineage; a historical route outside that guarantee; a canonical route bypassed by the successor harness; a producer/selection/association/application defect; an unenforced requirement; or a genuinely unspecified scientific choice. Support the classification with code and runtime evidence. Do not claim “APT v2 failed” or “APT v2 worked” from table format, checksums, or test counts alone.

If a defect is demonstrated and the intended behavior is established, **repair it under the existing bounded authority**. Relevant local changes in Citlali, TolProj, adapters, and tests are authorized; inspect each affected repository's instructions and preserve unrelated work. Restore the canonical producer–transformer–consumer path rather than adding an observation-specific multiplier or maintaining a second divergent resolver.

Do not modify authoritative APTs, source calibration evidence, archives, or frozen historical inputs. When a repair requires a new derived artifact, generate an explicitly identified local child through the appropriate canonical machinery, retaining its parent and evidence. Do not overwrite or retrospectively bless the original product. A scientific ambiguity is not permission to choose the state that best reproduces 574.59.

Add focused regression coverage at the implicated integration boundary, not only a unit test of multiplication. Reproduce the actual failure or allowed-difference case and verify the intended outcome. Include an appropriate positive control and negative case: for example, allowed POINT state versus wrong-role state; the expected child versus a wrong-but-checksummed artifact; stable keyed association after row reordering versus ambiguous association; or once-only correction versus duplicated application. Choose tests from the demonstrated mechanism rather than building a new testing framework.

Do not convert every numerical disagreement or calibration warning into a hard veto. Enforce established identity, role, validity, and application requirements; preserve scientifically permitted variation with explicit provenance. Unsupported calibration must not silently become accepted output, but valid alternative calibration lineages need not have identical values.

Use existing manifests and configuration references to identify which previous successor claims/products depend on the implicated path. Distinguish implemented capability from validated scientific use and absolute photometry from conditional incremental-response claims. This is a targeted dependency assessment, not authorization to rerun other observations or sweep the archive.

## 6. Numerical authority, launch behavior, and resources

Source/configuration/history audits, table comparisons, representative sample checks, canonical-artifact verification, and genuinely small regression fixtures may proceed immediately. Prefer those to a full observation run. This ordering is not a new owner-approval gate for routine implementation.

Only after documenting the lineage finding and any necessary repair should you decide whether the prepared map controls remain relevant. Retire or revise an unstarted control if it no longer answers the actual question; do not spend runs simply because a batch was prepared. Preserve its old freeze and record a new identity for revised work. If a remaining numerical test is needed, state its discriminator and use the smallest adequate comparison under the existing authority. Keep frozen-operator effects separate from fully relearned PTC/FRUIT consequences.

**No additional numerical allowance is created.** Continue the existing investigation-wide allowance of six planned starts plus at most two diagnosed replacements, eight maximum, less actual starts already used. The three prepared frozen-M0 roles remain charged as already specified when run; changing runners or calling them diagnostics does not make them free. Check the ledger. Do not blindly retry or repeat the eight-role B/Q campaign.

Preserve the current manual-resume workflow. For long numerical work, preflight and freeze, use the already verified launch mechanism, make at most one bounded startup check, and end the active turn without polling. Where the current route is owner foreground execution, supply one exact command after readiness checks rather than claiming a launch or inventing a new launcher. Do not modify source bound to active work. Resume assessment on the owner's message.

Respect existing worker/memory/storage limits, including the latest reported shared 30 GB ceiling unless the governing record has changed. Retain compact lineage tables, tests, receipts, and reproduction inputs rather than new timestream copies. Local evidence comes first; the prior permission for targeted read-only recovery of identified missing dependencies through existing Unity access remains, with exact paths and sizes established before retrieval. No remote computation, archive writes, bulk downloads, credential changes, production activation, push, or merge.

Ask only for a substantive unresolved scientific decision or an existing authority/resource boundary. Continue independent in-scope work; do not ask permission for each helper edit, regression test, local commit, or established-contract repair.

## 7. Deliverable and completion criterion

Extend the existing investigation directory and handoff with one clearly named APT-v2 lineage assessment and compact machine-readable evidence. Reuse existing audit/report structure where practical. Preserve original executed histories and distinguish later corrected artifacts, source revisions, and claims.

Lead the owner report with direct answers to:

1. **How did the two reductions acquire different factors?** Name the exact origin, decision/transformation, and numerical reconstruction—not just the differing filenames or medians.
2. **Was each calibration state appropriate to its POINT role, and did the relevant APT-v2 path actually execute?** Explain the applicable guarantee and why the safeguards accepted, missed, or did not govern the difference.
3. **What was repaired or legitimately left unchanged?** Give the reproducing test, affected evidence/claims, and any remaining specific dependency or scientific decision.
4. **What remains of the amplitude/profile investigation?** Separate calibration-lineage resolution from map-response attribution and absolute-calibration validation; report actual starts and the smallest justified next discriminator, if any.

Continue the existing focused independent milestone review for the substantive lineage conclusion and repair. The reviewer should independently trace representative factors and challenge the claimed safeguard explanation, including POINT circularity where relevant. A prior prelaunch review is not this conclusion; a fresh review is not required for every edit.

**Completion means a verified explanation of the calibration divergence and its workflow legitimacy, plus a tested repair where warranted—or an explicitly open finding narrowed to a precise unresolved link.** Do not stop with another generic audit proposal, a list of passed checks, “different calibration files,” or a brighter map. Do not close the broader photometric discrepancy merely because its upstream calibration component is understood.

---

### Source context

This directive supplements `CODEX_POINT152420_REDU09_AMPLITUDE_INVESTIGATION.md` (September 30) and carries forward `CODEX_POINT152420_COMPLETION_AND_BOUNDED_AUTONOMY.md` (September 29), subject to the explicit lineage-first ordering above. Numerical values, worktree paths, prepared-batch status, and the August 24 implementation leads are reported context to verify against the actual repositories and run evidence. This document does not assert that new runs have occurred, identify an established APT-v2 defect, or authorize production changes.
