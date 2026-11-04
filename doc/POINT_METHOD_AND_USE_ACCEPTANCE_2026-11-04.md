# Draft POINT method and use acceptance record

Recorded: November 4, 2026. Scientific owner: Grant Wilson.

**Disposition: owner direction recorded; complete method and named-use
acceptance pending.** This record preserves the POINT decisions from the
owner discussion preceding this request, reconciles them with existing
authority, and identifies the evidence still needed. It does not certify
SCI-POINT conformance or change the application. The
[team summary](POINT_TEAM_SUMMARY_2026-11-04.md) is the short sharing version.

## Scope and preflight

The owner requested saving the summary and preparing this record, with no
new branches or worktrees and no implementation changes. Initial clean base:
e425ab715c044a99340941818f92a1009bf2e300. During preparation, a separate
documentation update selected configured POINT plus retained-standard SCIENCE
at e988f7be19891bb114d77e757bd756c568792ee0. A subsequent owner clarification
deferred SCIENCE until the successor connection. Both updates were inspected
and preserved; the final documentation base is
**dfaf5efb813690f887dccea4176d7386973d79e0** on
`codex/refactor-mainline` in
`/Users/gwilson/.codex/worktrees/pointing-engineering-candidate-20261030/citlali-refactor`.
Staged, unstaged, and untracked state was empty at the initial and e988f7be1
inspections. The final base preserves the separately committed qualification
updates; the remaining dirty paths belong only to this documentation task.
No reset, merge, branch, or worktree operation reconciled the moving base.
Only the new POINT policy navigation entries in the status documents belong
to this change; the linked plan owns the execution sequence.

The accepted application remains **19b6fc462930af6f0a7bf2ca5b5350dd3c72af6e**,
tree **3b1d9e0ae067e502faa1e02895be3695f5166a66**. Later changes through the
documentation base are documentation and acceptance evidence. The
[stage-3 acceptance](../validation/review_stage3_integration_2026-11-03/OWNER_ACCEPTANCE.json)
is bounded software acceptance, not POINT scientific qualification.

Preflight read [Engineering Governance](governance/ENGINEERING_GOVERNANCE.md),
[Timestream Successor Governance](governance/TIMESTREAM_SUCCESSOR_GOVERNANCE.md),
and [Review and Conformance](governance/REVIEW_AND_CONFORMANCE.md). Their
file digests match the accepted identities in the
[integration ledger](INTEGRATION_LEDGER.md). Sequencing follows
[current state](CURRENT_STATE.md) and [refactor status](REFACTOR_STATUS.md).

This is Tier 1 documentation preparation: recording owner intent and a draft
acceptance boundary without amending frozen science or implementing a schema.
Complete method adoption, substantive conformance conclusions, and cross-package
implementation require the applicable Tier 2 evidence and independent review.
No application WIP slot is opened here.

**Grant executes all builds, tests, probes, benchmarks, reductions, and
validation runners on Mac and Unity.** This task performs file/Git inspection,
editing, and static review only. It authorizes no Unity access, push, runtime
replacement, cleanup, or sibling-repository edit.

## Authority reconciliation and prior work

This is a preparatory acceptance record, not a new scientific-contract package
or an implementation-blind author packet. The
[contract-library charter](scientific_contracts/README.md),
[prior-work registry](scientific_contracts/PRIOR_WORK_REGISTRY.md), and recovered
[SCI-POINT prior work](scientific_contracts/packages/SCI-POINT/v0.1/PRIOR_WORK.md)
remain entry points for a separately authorized successor package. Its reviewed
recovery record and sanitized Scope Brief must precede new derivation; the
implementation and validation material below cannot become Stage B authority.

| Source and status | Disposition for this record |
| --- | --- |
| [SCI-POINT v0.1 r0.4](scientific_contracts/packages/SCI-POINT/v0.1/README.md), unchanged r0.3 scientific core | **Cite and preserve.** Per-array scientific atoms, measured displacement, route-specific identity, processed-map shape meaning, separate use owners, and explicit unavailable uncertainty remain established boundaries. |
| [ODQ-004 approval](scientific_contracts/packages/SCI-POINT/v0.1/SCIENTIFIC_OWNER_ODQ_004_APPROVAL_2026-09-02.md) and [method-authority disposition](scientific_contracts/packages/SCI-POINT/v0.1/SCIENTIFIC_OWNER_METHOD_AUTHORITY_RESPONSE_2026-09-02.md) | **Preserve.** Frozen compatibility uses six parameters and zero background; its complete numerical method remains unavailable pending separate approval. Another estimator requires a separate versioned identity. |
| [September 25 expected-source directive](../validation/point_152389_expected_source_guard_2026-09-25/OWNER_DIRECTIVE.md) and [September 30 fitter repair directive](../validation/point152420_amplitude_investigation_2026-09-30/OWNER_DIRECTIVE.md) | **Cite within scope.** Explicit development authority for Gaussian plus constant, the source guard, conditional amplitude uncertainty, and repaired periodic-angle handling. These do not rewrite frozen SCI-POINT. |
| [Configured POINT work order](../validation/configured_point_2026-10-28/WORK_ORDER.md), [optical-response policy](OPTICAL_RESPONSE_POLICY_2026-10-29.md), and [weighted implementation](../validation/weighted_point_2026-10-29/IMPLEMENTATION.md) | **Cite within scope.** Configured development and weighted parent processing already have owner direction. This does not qualify every configuration or the new array-selection policy. |
| Owner decisions recorded below | **Adopt as intended use policy.** Preserve their thresholds, qualifications, and precedence. Complete method/profile acceptance and unresolved cases remain distinct. |
| Historical fits, replays, response experiments, and consumer checks below | **Cite as evidence only.** Successful execution cannot select missing science or establish new accuracy. |

The frozen [source manifest](scientific_contracts/packages/SCI-POINT/v0.1/STAGE_B_SOURCE_MANIFEST.json)
binds normative core SHA-256
`c0ca71bd457b8e6d37a425eb3ead76400dba3a5e29c869420807928201cdcdbd`.
Requirements 010 and 033 prohibit a fitted background on the compatibility
route; requirement 012 requires its approved numerical method. The development
Gaussian-plus-constant route cannot be certified unchanged against those
requirements. This is a method-authority reconciliation, not a finding that
the explicitly authorized development fitter should be rolled back.

The frozen compatibility, formal-error, and full-map-RMS methods remain
separately unavailable. The four
[named-use profiles](scientific_contracts/packages/SCI-POINT/v0.1/AUTHOR_POLICY_PROFILE_RECORDS.md)
remain drafts. A software `pointing_eligible` flag is not registration of a
scientific profile. This record does not supply their final approval.

## Recovered development method

This describes the implementation and its scoped owner directions, not a newly
approved numerical specification. Source links are read at the exact
documentation base above.

| Aspect | Recovered behavior and limit |
| --- | --- |
| Estimator | `POINT-EXPECTED-v1`: amplitude, two centroid coordinates, two Gaussian sigmas, orientation, and fitted constant. Bind exact source as well as this label: historical and repaired implementations share it. |
| Objective | Uniform fit weights, W = I, on the finite, supported 40 arcsec radius disk. Inverse-noise cleaning/map weights do not make the Gaussian fit inverse-noise weighted. |
| Search and solve | Central initialization within 20 arcsec, median fit-region background, nominal-beam width seed; one optimizer attempt, at most 3,000 function evaluations, no global fallback or restart ladder. The repaired angle is periodic and not artificially bounded. |
| Existing guard | Finite convergence, positive peak above the constant, centroid radius at most 20 arcsec, both sigmas strictly above 1.5 arcsec without an active lower bound, and conditional A/sigma_A greater than 10. These source-selection rules are distinct from the new 1.5 arcsec accuracy target. |
| Conditional uncertainty | Amplitude sensitivity includes all seven fitted-parameter directions, including background, and contracts against frozen M0 covariance. This is a source guard, not exact ten-sigma significance, a calibrated final centroid error, or a photometric uncertainty. |
| Parent and lifecycle | Current configured POINT is observation-local ordinary MAP in delta-source Alt/Az. Final-map measurements feed publication; internal feedback guards are separate uses, not frozen-contract terminal POINT products. |

Implementation anchors: [guard](../tools/timestream_successor/point_expected.py),
[solver](../tools/timestream_successor/inspect_point_source.py),
[fit publication](../tools/timestream_successor/configured_point_products.py), and
[table handoff](../tools/timestream_successor/configured_point_table.py).
No new fit model, weighting, feedback rule, optical threshold, or optimizer
change is selected here. Carrying this method into a formally accepted successor
method with an exact parent route/configuration and identity remains open.

## Owner decisions and their priority

These statements paraphrase Grant's direction in the discussion leading to
the November 4 documentation request. They supersede the earlier suggestion
that all-array disagreement always defeats the beam-quality override.

- **Primary use:** measure azimuth and elevation pointing offsets. The target
  is approximately **1.5 arcsec combined two-dimensional error**, not 1.5 arcsec
  independently on each axis. Its statistical acceptance interpretation remains
  open. Telescope-gain monitoring is optional and separately qualified.
- **Preserve all three measurements:** reduce a1100, a1400, and a2000 and retain
  each fit result or explicit failure/unavailability. One outlier does not erase
  the other results or automatically fail the observation's pointing use.
- **First priority, beam-quality override:** if a1100 fitted FWHM is **greater
  than 8 arcsec in either dimension**, use the **usable a2000 solution alone**,
  even if the three offsets disagree. The owner offered 8 arcsec provisionally.
  Fitted shape is evidence of possible deformation/defocus, not a unique causal
  diagnosis or an intrinsic optical beam measurement.
- **Otherwise, with roughly nominal beam shapes:** adopt the arithmetic mean of
  the consistent array solutions. Exclude a clear outlier separated by
  **5 arcsec or more** from an otherwise agreeing pair, retaining its values
  and reason. Typical agreement near or below 1 arcsec is descriptive, not an
  independently approved agreement cutoff.
- **Nominal-beam failure:** if all three pairwise offset separations are
  **5 arcsec or more**, report no accepted pointing solution. This use-level
  failure preserves per-array measurements, maps, and diagnostics. The usable
  a2000 beam-quality override takes precedence over this failure rule.
- **Observer feedback:** report beam assessments and selection/failure to LMT
  monitor and control. The intended output contract leads; TolTECA may be
  adapted instead of constraining the policy to its present mean.

These rules are not yet a total executable decision function. The discussion
did not authorize a closest-pair tie breaker, a median, uncertainty weighting,
or acceptance of a lone remaining array outside the explicit a2000 override.

## Product ownership and proposed reporting

POINT owns per-array fitted measurements. A named telescope-QC policy assesses
beam condition; a named pointing-support policy selects and combines results.
Publishing those decisions together does not turn the aggregate into a base
SCI-POINT fit component. Request, fit realization, beam assessment, selected
use, and application remain separate facts.

Use measured displacement in the established ordered tangent-azimuth/elevation
basis, in arcseconds. Preserve the expected-source reference, map/fit ancestry,
and sign. The selected displacement is **not an already applied correction**.
TolTECA retains correction construction, User/Paddle composition, and temporal
record selection; AST retains application. Selection must not add another
negation or azimuth cosine factor. The
[compatibility assessment](../validation/pointing_og_compatibility_2026-10-30/REPORT.md)
and later [handoff report](../validation/point_handoff_2026-10-30/REPORT.md)
bind these conventions and repaired interfaces.

Proposed delivery extends `point-table.json`, retains
`ppt_successor_pointing_<obs>_citlali.ecsv` per-array measurements, and references
the existing `point-fit-<array>.json` diagnostics. The owner permits consumer
adaptation; exact schema/version and publication semantics remain to be specified.
The report should carry:

- observation, source, epoch, coordinate reference, and input/map/method identities;
- all three offsets, fitted widths/orientation with units and availability,
  fit status, and beam assessment with its supporting measurements;
- policy identity, actual thresholds, reasons, selected/excluded arrays,
  adopted measured displacement or explicit absence, and uncertainty availability;
- independent pointing-use and gain/photometric-use dispositions.

Today the ECSV requires all three qualified fits and zero applied POINT
corrections; its uncertainty columns are NaN and calibration eligibility is
false. A withholding record survives a normal table hold. Proposed partial-array
selection/reporting must be reconciled with these rules, not assumed implemented.
The pinned operational TolTECA consumer still averages ECSV rows: adding JSON
fields alone will not change its behavior. Adaptation needs a separately scoped
work order and exact consumer source binding.

Saving this proposal does not authorize already-corrected POINT residuals as
uncorrected measurements, new flux errors, or telescope commands. Calibration
holds remain in force.

## Existing evidence and focused scorecard

Links resolve to tracked records at the documentation base. Their own source,
environment, input manifests, and limitations govern their claims; old passes
do not transfer to the proposed selection/reporting policy.

| Subject | Evidence and supported finding | Acceptance disposition |
| --- | --- | --- |
| Guard behavior | [September 25 assessment](../validation/point_152389_expected_source_guard_2026-09-25/ASSESSMENT.md) and [closure](../validation/point_152389_expected_source_guard_2026-09-25/COMPLETION.json): per array, 1,024 calibration and 1,024 replayed-validation fields retained all bright cores and admitted no blanks in the named scenes. | Historical support using reused fields, an earlier method realization, and unapplied saved-map diagnostics. No new policy or absolute-accuracy pass. |
| Repaired fit and conditional errors | [October 1 completion](../validation/point_quality_2026-10-01/COMPLETION.md), executed source a6d2c4bb2b1ccce671c17ab29e723da3b57bc8db: corrected 152420 reference and compact perturbations completed. | Conditional error and within-trajectory stability evidence; independent pointing accuracy explicitly unestablished. |
| Compact and broad response | [152418 recovery report](../validation/weighted_point_recovery_assessment_2026-10-30/REPORT.md), [input bindings](../validation/weighted_point_recovery_assessment_2026-10-30/INPUTS.json), and [review](../validation/weighted_point_recovery_assessment_2026-10-30/REVIEW.md), implementation d0d3eeb7981975a2f43f1a55878668410aa6c20b. M10 compact projection gains span 99.46–101.41%; a2000 shoulder gain is 84.37%, distortion 21.72%. | Narrow post-CAL injection response, not flux recovery or telescope-to-centroid accuracy. Broad-source qualification remains open. |
| Iteration stability | Same 152418 report: REF M5-to-M10 centroid shifts are 0.0000167, 0.0000260, and 0.0021921 arcsec for a1100/a1400/a2000. | Stability on one observation is not independent accuracy or justification to retune iteration limits. |
| Coordinates and consumer handoff | [Handoff report](../validation/point_handoff_2026-10-30/REPORT.md), [consumer results](../validation/point_handoff_2026-10-30/consumer_results.json), and [integration acceptance](../validation/pointing_handoff_integration_2026-10-31/OWNER_ACCEPTANCE.json). Actual pinned TolTECA 2791e6a1e6349ad1d3ac549a648f41cbc51b98c7 and TolProj 6309a5a852a0412cee867992aff01341a56b0ebf exercised the former interface. | Supports existing signs, discovery, correction composition, and unavailable errors. Does not exercise outlier selection or the beam override. |
| Gain and calibration | [Calibration audit](../validation/point_calibration_audit_2026-10-30/REPORT.md) and handoff report. Conditional sigma_A is not a calibrated amplitude error; the actual flux reader rejects NaN amp_err. | Gain monitoring is a separate proposed use; photometric transfer stays held. Shape and nominal-beam peak alone do not establish gain change. |
| Engineering baseline | [Stage-3 acceptance](../validation/review_stage3_integration_2026-11-03/OWNER_ACCEPTANCE.json) and [owner results](../validation/review_stage3_integration_2026-11-03/OWNER_RESULTS-19b6fc462.json). | Accepted Mac Spack software evidence with limits; no new configured POINT scientific or Unity qualification. |
| New owner policy | Decisions above; no execution of the new policy is asserted. | Direction recorded. Implementation, named-use registration, threshold qualification, and the 1.5 arcsec accuracy claim remain pending. |

## Unresolved cases and required decisions

| Open case | Required disposition before dependent acceptance |
| --- | --- |
| Formal method and route | Bind a separately versioned successor estimator and intended parent/configuration, including weighting, grid, source class, and feedback recipe. Decide its formal relationship to frozen SCI-POINT without replacing the compatibility result. A successful historical trial does not select those settings. |
| Accuracy target | Specify bias/repeatability measure, confidence or coverage, population, per-array versus selected-solution obligations, and acceptable failure rate for the combined 1.5 arcsec target. Decide whether initial use may retain unavailable per-observation errors. |
| Separation definition | Confirm that 5 arcsec means physical two-dimensional distance in the common tangent plane, and whether an outlier is compared with each other array or their mean. Later owner wording includes equality at 5; it does not define that operator. |
| Consistency and ambiguous groups | Define an agreeing pair and handle chained/tied groups. Offsets (0,0), (4,0), (8,0) arcsec have two close pairs and one distant pair; no exclusion rule was selected. Typical 1 arcsec agreement and the 1.5 arcsec accuracy goal are not automatic cutoffs. |
| Beam criteria | Confirm whether either FWHM dimension means principal axes, define roughly nominal shapes for all arrays, and finalize the provisional strict greater-than-8 trigger. Define the role of a1400 shape, asymmetric/non-Gaussian structure, uncertain widths, and missing a1100 shape. Broadening does not uniquely diagnose deformation. |
| Missing or unusable fits | Define usable a2000 for the override and the outcome if it fails. Decide cases with only two or one usable array, missing metadata, nonfinite results, bound-active fits, or failed source association. No fallback from an unavailable preferred fit is selected here. |
| Output and failure behavior | Bind schema/version, required fields, QC/selection owner, partial-result publication, and machine-visible failure. Reconcile all-three ECSV publication with partial measurements and selected-use status. An unusable pointing does not erase valid maps or excuse an I/O failure. |
| Consumer and correction eligibility | Bind TolTECA adaptation and legacy behavior, preserving sign, User/Paddle, time, and source identity. Already-corrected POINT stays held pending residual-to-measurement qualification. LMT consumption is intended capability, not a verified deployed interface. |
| Gain monitoring | Define relative monitoring versus calibration transfer and its source, atmosphere/calibration controls, response, time comparison, and uncertainty requirements. It is not a prerequisite for initial astrometry. |

No defaults are assigned to these unresolved cases. The usable a2000 override's
priority over nominal-beam disagreement failure is settled.

## Acceptance sequence and current disposition

Review the team summary and resolve the open policy cases, then bind one complete
method/use record and the smallest implementation scope for selection, observer
reporting, and consumer adaptation. Separate fit completeness, pointing admission,
telescope QC, and gain/photometric use; approve/register the profiles actually needed.

Future owner-run evidence must cover threshold boundaries, override priority,
all-array disagreement, one outlier, ambiguous groups, failed preferred fits,
and preservation of results. Actual-consumer fixtures must establish that the
selected displacement is used once with existing correction conventions,
including unavailable/error states. These are pending gate requirements, not
tests prepared or run by this documentation task.

The [individual POINT and celestial SCIENCE qualification plan](POINT_SCIENCE_QUALIFICATION_PLAN_2026-11-03.md)
owns the qualification sequence. Its subsequent owner clarification selects
individual configured successor POINT now and defers SCIENCE/coadd/known-offset
qualification until successor SCIENCE is connected. Retained-driver preparation
is on hold and is not a prerequisite to POINT. This record does not reopen that
choice or supply launch authority. The existing stage-3 executable cannot
exercise the unimplemented selection/reporting policy; its qualification and a
later policy candidate need distinct evidence bindings. POINT remains
observation-local, with coadds in SCIENCE. Independent reference comparisons,
rather than iteration stability or reuse of the same POINT-derived correction,
must support an eventual accuracy claim. Selecting that evidence need not turn
the deferred SCIENCE implementation into an automatic prerequisite to all POINT
qualification.

Current preparation disposition:

- **Scientific and behavioral:** owner intent and evidence limits are recorded;
  frozen science and behavior are unchanged. Complete method/use acceptance,
  formal errors, and achieved accuracy remain pending.
- **Architecture and ownership:** per-array measurement, QC, pointing-support,
  TolTECA, and AST boundaries are preserved in the proposal. No interface or
  sibling implementation was edited.
- **Repository and evidence:** two new documents and navigation updates form
  one documentation change in the existing checkout. Historical evidence,
  runtimes, branches, worktrees, and frozen packages are preserved.
- **Review and execution:** static editorial/source/diff review only. No builds,
  tests, probes, reductions, or validation runners were executed. Independent
  method review and future exact-candidate implementation review remain pending;
  this is not a Tier 2 conformance pass.

The next decision is completion of the open method/use policy, not approval to
launch a reduction, activate production, or publish to GitHub.
