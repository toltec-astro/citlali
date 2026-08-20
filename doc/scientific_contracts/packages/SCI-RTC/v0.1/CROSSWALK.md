# SCI-RTC v0.1/r0.9 exact authority crosswalk

Status: implementation-blind author crosswalk. It maps every normative ID in
the shared core to explanatory and conformance loci and to the approved author
packet. It reports no implementation or validation result.

## Approved source keys

| Key | Exact approved author input | Approved SHA-256 |
| --- | --- | --- |
| `SB` | `SCOPE_BRIEF.md` | `c8cac0b8ae731919622d7b696c60946685b5eba9b16a5cd830c01a2f6f28e013` |
| `SC` | `AUTHOR_SUPERSESSION_COVER.md` | `f183c8fb083c3a851fda5d77a0944405cc41650ced29bd0162cffba832f25575` |
| `CO` | `AUTHOR_CONVENTIONS_AND_OWNERSHIP.md` | `a26220dc827330e30ca8e4c75e82600e6cc2f05358887bbaa0c6da93f98ecb5b` |
| `RC` | Exact retained core obtained as `git show 3319d7424c732c1c9fc300c336e4d428e6f91068:doc/audits/packages/SCI-RTC-001_INDEPENDENT_CORE.tex` | `d6cf49d1a5e17754c55cc4f2c8f4b4f5e276755f247496df888581d890be80b7` |
| `R4` | Supplied r0.3 scientific-owner review plus the two explicit r0.4 owner approvals recorded in `SCIENTIFIC_OWNER_REVIEW_R0.4.md` | `2298d9b801a2213bf8327f83abe5e0a6aeb3eca2c1398bafb4ec106a9972eba4` |
| `R5` | Supplied r0.5 scientific-owner directive, content-bound and summarized in `SCIENTIFIC_OWNER_DIRECTIVE_R0.5.md` | `7469fd327d9465904a4e59c287577bab0dcd9f93fd2cc555cdee6680e89714a6` |
| `R6` | Supplied r0.6 scientific-owner review plus the three explicit owner confirmations recorded in `SCIENTIFIC_OWNER_REVIEW_R0.6.md` | `2a4163d1ed0775e83ef981573d1a3a1f65fe2d89860bd92b0ad456e61fa8e266` |
| `R7` | Supplied r0.7 scientific-owner review and surgical correction request recorded in `SCIENTIFIC_OWNER_REVIEW_R0.7.md` | `01ec886e6d1dad89835463a1cee39dd0da067cf7532608698f90262cb41a9937` |
| `R8` | Binding scientific-owner Decision 9 recorded in `SCIENTIFIC_OWNER_DECISION_R0.8.md` | `8862e3d4caf3fdd695fa66cbc0af58d40725375444f145525c4393f3859095b1` |
| `R9` | Binding scientific-owner Decisions 1--8 recorded in `SCIENTIFIC_OWNER_DECISIONS_R0.9.md` | `90cad00151d975e0bb2a432c907f4a2198a1f3645f52c645c7e71cfa58ac57cb` |

The supersession cover controls wherever `RC` is broader, older, or
ambiguous. In particular, crosswalk entries that cite the retained calibration,
replacement, sampling, influence, or signal-domain derivation always include
the applicable `SC` correction.
The later `R5` owner directive controls wherever the approved packet or
retained core assumes a single measured coordinate or excludes the paired
$r$ coordinate.  It does not alter the recorded hashes of those earlier
inputs.
The later `R6` review and confirmations supersede the two false r0.5 owner
attributions and control the atmospheric-template, shift-learning/replacement,
required-output, general-mapping, and bounded scientific-correction questions.
They do not alter the recorded hashes of earlier inputs.
The later `R7` review accepts the r0.6 narrative and architecture, controls the
ALIGN, fixed-state covariance, leakage-normalization, event-time, reset/carry,
operation-inventory, atmosphere/support, and output-split corrections, and
adds no implementation or validation evidence.
The later `R8` Decision 9 controls the additive-only stable-plateau model,
finite physical-time transition support, transition-unmodeled boundary,
optional valid additive correction, insufficient-support behavior, compact
production state, and exclusion of gain/responsivity-change modeling.
The later `R9` decisions control operation availability across application
contexts, the context--plan--record lifecycle, the consumer-neutral bundle,
mapping/pair/coordinate authority, typed non-finite handling, covariance-claim
disclosure, and actual despiking with compact normal-run population reporting.

## View inclusion crosswalk

| Shared-core component | Scientist-facing view | Engineering/formal view | Published inclusion |
| --- | --- | --- | --- |
| `src/common/notation.tex` | Explained and ID-crosswalked; not duplicated | Shared normative authority in `engineering-conformance.tex` | engineering exactly 1; rationale 0 |
| `src/common/definitions.tex` | same explanatory relationship | same shared-core sequence | engineering exactly 1; rationale 0 |
| `src/common/equations.tex` | same explanatory relationship | same shared-core sequence | engineering exactly 1; rationale 0 |
| `src/common/assumptions.tex` | same explanatory relationship | same shared-core sequence | engineering exactly 1; rationale 0 |
| `src/common/requirements.tex` | same explanatory relationship | same shared-core sequence | engineering exactly 1; rationale 0 |
| `src/common/edge_cases.tex` | same explanatory relationship | same shared-core sequence | engineering exactly 1; rationale 0 |

The engineering wrapper contains no displayed equation outside those six
imports. The separate scientific rationale's sections 1--12 are explanatory,
contain no independent displayed normative equation, and do not duplicate or
modify the formal authority.

## Definitions

| Normative ID | Shared-core locus | Rationale locus | Engineering use | Packet authority |
| --- | --- | --- | --- | --- |
| `SCI-RTC-DEF-001` | Definitions: admitted aligned stream | §1 | Routing REQ-001--009 | SB §§2--3; CO Identity/Time |
| `SCI-RTC-DEF-002` | Definitions: RTC application context and raw boundary | §§1--2 | Routing REQ-001--005 | R9 decisions 1/4; R4 signal decision; SB D003; SC 1 |
| `SCI-RTC-DEF-003` | Definitions: exact detector occurrence | §§1, 4 | Routing REQ-006--012 | SB §3.2; CO Identity |
| `SCI-RTC-DEF-004` | Definitions: downstream CAL handoff | §§1, 10 | Routing REQ-001--005 | R4 calibration-order decision; SB §§3.5, 5 |
| `SCI-RTC-DEF-005` | Definitions: compatible `flxscale` pair | §3 | Routing REQ-013--018 | SB D004; SC 2; CO Signal labels |
| `SCI-RTC-DEF-006` | Definitions: selected conditioning policy and admitted classes | §§1, 4--8 | Routing REQ-006--012 | R9 decisions 1/8; SB §§3.6, 10 D014; SC 13 |
| `SCI-RTC-DEF-007` | Definitions: one-way context--plan--record lifecycle | §§1--2, 10 | Routing REQ-006--012 | R9 decision 2; SB D007/D010; SC 5, 8; CO State |
| `SCI-RTC-DEF-008` | Definitions: realized conditioning operator | §§3--7 | Response checks | SB D006/D009; RC operator derivation; SC corrections |
| `SCI-RTC-DEF-009` | Definitions: phase-zero sampling | §7 | Routing REQ-028--031 | SB D008; SC 7; CO State |
| `SCI-RTC-DEF-010` | Definitions: RTC-local and optional end-to-end response | §§4--5 | Response checks | R7 blocker 1; SB D006; SC 4; CO Response |
| `SCI-RTC-DEF-011` | Definitions: direct source cell | §4 | State/failure checks | SB D005; SC 3; CO Influence |
| `SCI-RTC-DEF-012` | Definitions: transitive influence | §§4, 7 | Routing REQ-019--020 | SB D005; SC 3; CO Influence |
| `SCI-RTC-DEF-013` | Definitions: direct exclusion and consumer eligibility | §§4, 10 | Routing REQ-019--020/052 | R4 influence decision; SB D005; CO Influence |
| `SCI-RTC-DEF-014` | Definitions: coordinate-dependent control | §6 | Routing REQ-024--027 | R9 decision 5; SB D013; SC 12; CO Coordinates |
| `SCI-RTC-DEF-015` | Definitions: fixed sampling mode | §8 | Routing REQ-028--031 | SB D010; SC 8; CO State |
| `SCI-RTC-DEF-016` | Definitions: learned sampling mode | §8 | Routing REQ-032--036 | SB D010--D012; SC 8--11; CO State |
| `SCI-RTC-DEF-017` | Definitions: conditional covariance and claim disclosure | §12 | Routing REQ-042--045 | R9 decision 7; SB §§3.8, 7.7; RC covariance; CO Statistics |
| `SCI-RTC-DEF-018` | Definitions: consumer-neutral atomic RTC bundle | §§1, 10 | Routing REQ-046--051 | R9 decisions 3/8; SB D015; SC 14; CO RTC transformer |
| `SCI-RTC-DEF-019` | Definitions: scientifically named diagnostic | §10 | State/failure checks | SB §4.10; SC 14 |
| `SCI-RTC-DEF-020` | Definitions: claim layer | §10 | Completion checklist | SB D016; SC 15; CO Claim Layers |
| `SCI-RTC-DEF-021` | Definitions: declared learning population | §2 | Lifecycle stop check | r0.2 directive §1 |
| `SCI-RTC-DEF-022` | Definitions: learned evidence | §2 | Evidence/uncertainty trace | r0.2 directive §1 |
| `SCI-RTC-DEF-023` | Definitions: application-context-bound resolved RTC plan | §§1--2, 12 | Plan completeness check | R9 decision 2; r0.2 directive §§1, 14 |
| `SCI-RTC-DEF-024` | Definitions: apply state | §2 | Plan-mutation rejection | r0.2 directive §1 |
| `SCI-RTC-DEF-025` | Definitions: online-adaptive estimator | §§2, 5 | Adaptive separation check | r0.2 directive §§1, 4 |
| `SCI-RTC-DEF-026` | Definitions: scientific filter design | §§3--9 | Design-register review | r0.2 directive §§2--10, 14 |
| `SCI-RTC-DEF-027` | Definitions: bounded attempts over accepted plans | §§2, 5 | Attempt/plan and immutability trace | R4 correction; r0.3 directive §§1--2 |
| `SCI-RTC-DEF-028` | Definitions: complete cumulative successor proposal | §§2, 5 | Full-proposal reconstruction | R4 correction; r0.3 directive §§2--3 |
| `SCI-RTC-DEF-029` | Definitions: final accepted plan and termination | §§2, 5, 12 | Stop/nonconvergence audit | r0.3 directive §§5--8 |
| `SCI-RTC-DEF-030` | Definitions: exact paired raw occurrence | §§1, 4 | Pair-admission checks | R5 §§II--III |
| `SCI-RTC-DEF-031` | Definitions: upstream IQ-to-$x/r$ mapping | §4 | Mapping reconstruction | R5 §III |
| `SCI-RTC-DEF-032` | Definitions: coordinate-qualified optical leakage | §§4, 6 | Paired response checks | R7 clarification 3; R5 §IV |
| `SCI-RTC-DEF-033` | Definitions: atmospheric leakage diagnostic | §§3, 6 | Atmosphere estimator routing | R5 §§IV--V |
| `SCI-RTC-DEF-034` | Definitions: bright-source leakage diagnostic | §6 | Source estimator routing | R5 §VI |
| `SCI-RTC-DEF-035` | Definitions: additive-only network level shift | §§5, 10 | Event/model checks | R8 Decision 9; R7 clarification 4; R5 §§VII--VIII |
| `SCI-RTC-DEF-036` | Definitions: finite physical transition support and propagated influence | §§5, 10 | Timing-vector/mask/boundary checks | R8 Decision 9; R7 clarification 5; R5 §§VII--VIII |
| `SCI-RTC-DEF-037` | Definitions: stable additive-baseline plateau | §§5, 10 | Support/admission checks | R8 Decision 9; R5 §§VII--VIII |
| `SCI-RTC-DEF-038` | Definitions: response-changing successor boundary | §§4--6 | Gain/two-coordinate forbidden-route checks | R8 Decision 9; R5 §§IV, IX |

## Equations and identities

| Normative ID | Shared-core locus | Rationale locus | Engineering use | Packet authority |
| --- | --- | --- | --- | --- |
| `SCI-RTC-EQ-001` | Equations: detector-static comparison coordinate | §4 | Donor vectors | R4 signal decision; SB D004; SC 2 |
| `SCI-RTC-EQ-002` | Equations: raw donor transfer | §3 | Donor direction/availability | SB D004; SC 2; CO Signal labels; supersedes RC responsivity equation |
| `SCI-RTC-EQ-003` | Equations: distinct downstream CAL operator | §10 | CAL handoff checks | R4 calibration-order decision; SCI-CAL boundary |
| `SCI-RTC-EQ-004` | Equations: upstream ALIGN relation, RTC/CAL order, and raw-$r$ disposition | §§1, 4, 11 | Composition trace | R7 blocker 1; R6 decisions 1/3; R5 §§I, IX |
| `SCI-RTC-EQ-005` | Equations: RTC-local conditioned-$x$ operator from aligned input | §§1, 3--8 | Factorization reconstruction | R7 blocker 1; R6 decisions 2--3; R4 signal decision; SB D001/D006/D008; RC RTC-16 |
| `SCI-RTC-EQ-006` | Equations: local affine operator, zero $r$ branch, optional end-to-end response | §§4--5, 9 | Response/covariance checks | R7 blockers 1--2; R6 decision 3; RC RTC-17; SC 1--7 |
| `SCI-RTC-EQ-007` | Equations: post-segmentation $x$ replacement | §§4--5 | Donor/boundary fixtures | R6 decision 2; RC RTC-08 specialized by SC 2--3 |
| `SCI-RTC-EQ-008` | Equations: FIR response | §5 | Impulse/DC checks | RC RTC-10; SC 6, 13 |
| `SCI-RTC-EQ-009` | Equations: IIR state | §§5--6 | Split-state checks | RC RTC-11; SC 6, 13 |
| `SCI-RTC-EQ-010` | Equations: example mask operator | §6 | Mask distinction checks | RC RTC-12; SC 12--13 |
| `SCI-RTC-EQ-011` | Equations: conditioned-$x$ phase-zero selection and raw-pair representative occurrence | §§7--8 | Factor/length and parent-occurrence checks | R6 decision 3; R4 symbol correction; SB D008; SC 7 |
| `SCI-RTC-EQ-012` | Equations: local response | §5 | Jacobian fixtures | SB D006; SC 4; RC RTC-26 |
| `SCI-RTC-EQ-013` | Equations: restricted LTI response | §5 | Interior frequency checks | RC RTC-24--25; CO Response |
| `SCI-RTC-EQ-014` | Equations: phase-zero alias identity | §§7--8 | Folded-band checks | SC 7, 9, 11; RC RTC-27 at phase zero |
| `SCI-RTC-EQ-015` | Equations: RTC-local and optional end-to-end support | §§4, 6--7 | Support expansion | R7 blocker 1; RC RTC-28; SC 3--4, 7 |
| `SCI-RTC-EQ-016a` | Equations: fixed-state paired-input conditional mean | §12 | Statistical checks | R7 blocker 2; RC RTC-18 |
| `SCI-RTC-EQ-016b` | Equations: fixed-state $[L^x\ 0]$ covariance | §12 | Bounded matrix checks | R7 blocker 2; RC RTC-19; CO Statistics |
| `SCI-RTC-EQ-017` | Equations: unconditional covariance identity | §12 | Selector-availability and disclosure check | R9 decision 7; RC RTC-21; SC transcription context |
| `SCI-RTC-EQ-018` | Equations: nuisance propagation | §9 | Nuisance record check | RC RTC-22 |
| `SCI-RTC-EQ-019` | Equations: possible complete covariance decomposition | §12 | Claim-scope and availability audit | R9 decision 7; RC RTC-23 corrected by SC Corrections |
| `SCI-RTC-EQ-020a` | Equations: influence closure | §§4, 7 | Noncenter-cause fixture | SB D005; SC 3; CO Influence |
| `SCI-RTC-EQ-020b` | Equations: direct exclusion and consumer policy | §§4, 11 | Downstream handoff | R4 influence decision; SB D005; CO Influence |
| `SCI-RTC-EQ-021` | Equations: maximum-safe learned plan | §8 | Candidate decision table | SB D010--D012; SC 8--11; CO State |
| `SCI-RTC-EQ-022` | Equations: consumer-neutral conditioned-$x$/raw-$r$ atomic bundle | §§1, 10 | Atomic output checks | R9 decision 3; R6 decision 3; SB D015; SC 14; RC RTC-30 specialized |
| `SCI-RTC-EQ-023` | Equations: projected Gaussian crossing time | §3 | Scan/beam domain calculation | r0.2 directive §3 |
| `SCI-RTC-EQ-024` | Equations: generic notch response | §5 | Notch response/state checks | r0.2 directive §4 |
| `SCI-RTC-EQ-025` | Equations: constrained FIR order | §7 | Candidate design table | r0.2 directive §6 |
| `SCI-RTC-EQ-026` | Equations: linear-phase FIR delay | §7 | Delay fixture | r0.2 directive §6 |
| `SCI-RTC-EQ-027` | Equations: scan displacement | §7 | Coordinate/centroid fixture | r0.2 directive §§6, 9 |
| `SCI-RTC-EQ-028` | Equations: learn--resolve--apply | §2 | State and mutation trace | r0.2 directive §§1, 14 |
| `SCI-RTC-EQ-029` | Equations: attempts, accepted plans, and replay | §§2, 5 | Attempt/plan/replay trace | R4 correction; r0.3 directive §§2--3, 10 |
| `SCI-RTC-EQ-030` | Equations: native Tune-dependent IQ-to-$x/r$ mapping before ALIGN | §4 | Nonlinear/local mapping and pair round trip | R7 blocker 1; R6 mapping correction; R5 §III |
| `SCI-RTC-EQ-031` | Equations: local paired optical response with distinct residual | §§4, 6 | Paired injection response | R6 symbol correction; R5 §IV |
| `SCI-RTC-EQ-032` | Equations: coordinate-qualified leakage ratio and metric-qualified angle | §6 | Atmosphere/source leakage, rescaling, and bias fixtures | R7 clarification 3; R6 estimator/symbol corrections; R5 §§IV--VI |
| `SCI-RTC-EQ-033` | Equations: additive plateaus around finite physical-time transition support | §§5, 10 | Cross-cadence support/model fixtures | R8 Decision 9; R7 clarification 4; R5 §§VII--VIII |
| `SCI-RTC-EQ-034` | Equations: additive plateau correction, transition exclusion, reset/carry | §§5, 10 | Offset/reference/support/state checks | R8 Decision 9; R7 clarification 5; R5 §§VII--VIII |
| `SCI-RTC-EQ-035` | Equations: actual attempts, maximum, and accepted plans | §§2, 7 | Early-stop/no-no-op check | R5 bounded-iteration correction |

## Assumptions

| Normative ID | Rationale locus | Engineering use | Packet authority |
| --- | --- | --- | --- |
| `SCI-RTC-ASM-001` | §§1--2 | Boundary review | SB D002; CO Capability |
| `SCI-RTC-ASM-002` | §§1, 7 | Identity/grid review | SB §§2--3, 6; CO Time |
| `SCI-RTC-ASM-003` | §6 | Mask failure review | R9 decision 5; SB D013; SC 12; CO Coordinates |
| `SCI-RTC-ASM-004` | §§2--3 | CAL boundary review | SB D004; SC 2; CO Producers |
| `SCI-RTC-ASM-005` | §§1, 3--8 | Admission-versus-selection review | R9 decision 1; SB D014; SC 13 |
| `SCI-RTC-ASM-006` | §§4--5, 9 | Response/statistics review | RC conditional operator/covariance; CO Response |
| `SCI-RTC-ASM-007` | §12 | Statistical claim disclosure | R9 decision 7; SB §3.8; CO Statistics |
| `SCI-RTC-ASM-008` | §8 | Learned-plan lifecycle | SB D011--D012; SC 9--10 |
| `SCI-RTC-ASM-009` | §8 | Learned analytical checks | SC 11 |
| `SCI-RTC-ASM-010` | §§5, 10 | Compact spike-summary and optional-manifest fidelity | R9 decision 8; SB §4.3--4; SC 4, 14; RC compact provenance |
| `SCI-RTC-ASM-011` | §§1, 10 | Consumer routing | SB §5; CO Consumers |
| `SCI-RTC-ASM-012` | §10 | Claim checklist | SB D016; SC 15; CO Claim Layers |

## Requirements

| Requirement | Rationale locus | Engineering routing/check | Packet authority |
| --- | --- | --- | --- |
| `SCI-RTC-REQ-001` | §1 | Raw boundary/state fixture | R4 signal decision; SB §§1--3 |
| `SCI-RTC-REQ-002` | §1 lifecycle table | Application-context and label-neutrality inspection | R9 decisions 1--2; R4 signal decision; SB §§3.9, 6 |
| `SCI-RTC-REQ-003` | §1 lifecycle table | Consumer-neutral conditioned-$x$/raw-$r$ bundle boundary | R9 decision 3; R6 decision 3; R4 signal decision; SB D003 |
| `SCI-RTC-REQ-004` | §§1, 10 | Distinct downstream CAL boundary | R4 calibration-order decision |
| `SCI-RTC-REQ-005` | §§1, 3, 10 | CAL handoff and diagnostic-only atmosphere trace | R6 decisions 1/3; R4 calibration-order decision; SB §5 |
| `SCI-RTC-REQ-006` | §1 | Distinct-identity fixture | SB §§3.2, 4.2; CO Identity |
| `SCI-RTC-REQ-007` | §§1, 7 | Index/mapping round trip | SB §§3.2, 4.1--2; CO Identity |
| `SCI-RTC-REQ-008` | §§1, 7 | Grid-claim audit | SB §6; CO Time |
| `SCI-RTC-REQ-009` | §10 | Frame/binding/inference failure injection | R9 decision 5; SB D013; SC 12; CO Coordinates |
| `SCI-RTC-REQ-010` | §§1--2, 10 | One-way context--plan--record trace | R9 decision 2; SB D007/D010; SC 5, 8 |
| `SCI-RTC-REQ-011` | §§1, 10 | Stage/parent identity | SB D007; SC 5 |
| `SCI-RTC-REQ-012` | §§1--10 | All-class admission versus explicit selection/execution | R9 decision 1; R7 clarification 6; SB D014; SC 13 |
| `SCI-RTC-REQ-013` | §§4--5, 11 | Single upstream ALIGN, original-pair shift learning, transition/plateau resolution, optional valid additive correction, post-segmentation replacement, conditioned-$x$-then-CAL trace | R8 Decision 9; R7 blocker 1; R6 decisions 2--3; R4 calibration-order decision |
| `SCI-RTC-REQ-014` | §3 | Raw donor direction matrix | SB D004; SC 2; CO Signal labels |
| `SCI-RTC-REQ-015` | §3 | Dependency inspection | SB D004; SC 2; CO Signal labels |
| `SCI-RTC-REQ-016` | §3 | Invalid-transfer failure | SB D004; SC 2; CO Transformer |
| `SCI-RTC-REQ-017` | §5 | Actual accepted-target modification, compact spike summary, and optional event/donor-manifest toggle | R9 decision 8; SB §§3.6, 7.3; SC 13; RC replacement |
| `SCI-RTC-REQ-018` | §§4, 9 | Donor link/covariance fixture | SB §§4.4--6; SC 3; CO Statistics |
| `SCI-RTC-REQ-019` | §§4, 11 | Direct/noncenter synthesis distinction | R4 influence decision; SB D005 |
| `SCI-RTC-REQ-020` | §§5, 12 | Direct/noncenter replacement cause and influence distinction | R9 decision 8; R4 influence decision; SB D005 |
| `SCI-RTC-REQ-021` | §5 | Coefficient/state serialization | SB D009; SC 6, 13 |
| `SCI-RTC-REQ-022` | §5 | FIR impulse/DC checks | RC RTC-10; SC 6 |
| `SCI-RTC-REQ-023` | §§5, 10 | Reset-by-default and authorized-continuity checks | R7 clarification 5; RC RTC-11; SC 6, 13 |
| `SCI-RTC-REQ-024` | §6 | Mask-operator distinction | SB D013--D014; SC 12--13 |
| `SCI-RTC-REQ-025` | §6 | Invalid-coordinate matrix | SB D013; SC 12; CO Coordinates |
| `SCI-RTC-REQ-026` | §10 | Typed-cause non-finite injection and generic-NaN rejection | R9 decision 6; SB §§4.4--5, 6; CO Missing State |
| `SCI-RTC-REQ-027` | §6 | Edge/short-scan fixtures | SB D009/D014; SC 6, 13 |
| `SCI-RTC-REQ-028` | §§7--8 | Conditioned-$x$ point selection and raw-pair representative occurrence | R6 decision 3; R4 symbol correction; SB D008; SC 7 |
| `SCI-RTC-REQ-029` | §7 | Cardinality/time/support check | SB §§4.2, 7.8; SC 7 |
| `SCI-RTC-REQ-030` | §7 | Folded-band calculation | SB §§4.3, 7.5; SC 7, 9 |
| `SCI-RTC-REQ-031` | §8 | Fixed-plan state trace | SB D010; SC 8 |
| `SCI-RTC-REQ-032` | §8 | Learned lifecycle trace | SB D010; SC 8 |
| `SCI-RTC-REQ-033` | §8 | Candidate decision table | SB D011; SC 9, 11 |
| `SCI-RTC-REQ-034` | §8 | Common-plan identity | SB D011--D012; SC 9--10 |
| `SCI-RTC-REQ-035` | §8 | Immutable apply/fallback | SC 8--10; CO State |
| `SCI-RTC-REQ-036` | §8 | Restart mismatch fixture | CO State and Sampling |
| `SCI-RTC-REQ-037` | §§4--5, 12 | RTC-local and optional end-to-end response audit | R7 blocker 1; SB D006/D015; SC 4, 14 |
| `SCI-RTC-REQ-038` | §§4--5, 12 | Zero fixed-state $r$ branch, selector dependence, and CAL composition | R7 blockers 1--2; R4 calibration-order decision; SB D006 |
| `SCI-RTC-REQ-039` | §5 | LTI-domain proof | RC temporal response; CO Response |
| `SCI-RTC-REQ-040` | §§5, 7 | Response-component inspection | SB §§4.3, 7.5; RC response |
| `SCI-RTC-REQ-041` | §§4, 6--7 | RTC-local and optional end-to-end support expansion | R7 blocker 1; SB D005--D006; SC 3--4, 7 |
| `SCI-RTC-REQ-042` | §12 | Fixed-state numerical covariance plus included/excluded effect disclosure | R9 decision 7; R7 blocker 2; SB §§3.8, 7.7; RC covariance |
| `SCI-RTC-REQ-043` | §12 | Selector component/correlation disclosure audit | R9 decision 7; RC RTC-20--21; CO Statistics |
| `SCI-RTC-REQ-044` | §12 | Included/excluded component and correlation-scope record | R9 decision 7; SB §§4.6, 7.7; RC RTC-22--23 |
| `SCI-RTC-REQ-045` | §12 | Unknown/unavailable and qualified-partial-claim audit | R9 decision 7; SB §3.8; CO Statistics |
| `SCI-RTC-REQ-046` | §§6, 10 | Orthogonal-state fixture | SB §4.5; CO Validity |
| `SCI-RTC-REQ-047` | §§6, 10 | Flag/cause aggregation | SB §§4.4--5, 7.6; CO Validity |
| `SCI-RTC-REQ-048` | §§1, 10 | Consumer-neutral conditioned-$x$/raw-$r$ bundle inspection | R9 decision 3; R6 decision 3; SB D015; SC 14 |
| `SCI-RTC-REQ-049` | §§1, 10 | Required-write failure injection | SB §§4.9, 6; CO Missing State |
| `SCI-RTC-REQ-050` | §§5, 10--12 | Native/ALIGN/RTC/CAL reconstruction plus compact spike treatment/population summary | R9 decisions 2/4/8; R7 blocker 1; R4 calibration-order decision; SB §§4.2, 4.7--8 |
| `SCI-RTC-REQ-051` | §§5, 10 | Optional event/donor-detail inertness and diagnostic classification | R9 decision 8; SB §4.10; SC 14 |
| `SCI-RTC-REQ-052` | §§4, 11 | Consumer influence-policy audit | R4 influence decision; SB §5; CO Consumers |
| `SCI-RTC-REQ-053` | §10 | Disabled-PTC terminal path | CO Consumers |
| `SCI-RTC-REQ-054` | §10 | Claim-label audit | SB D016; SC 15; CO Claim Layers |
| `SCI-RTC-REQ-055` | §2 | Learning-population identity | r0.2 directive §1 |
| `SCI-RTC-REQ-056` | §§2, 12 | Resolved-plan completeness | r0.2 directive §§1, 14 |
| `SCI-RTC-REQ-057` | §§2, 12 | Missing-predicate stop matrix | r0.2 directive §1 |
| `SCI-RTC-REQ-058` | §§2, 5 | Adaptive-estimator separation | r0.2 directive §§1, 4 |
| `SCI-RTC-REQ-059` | §§1--10 | Complete ten-part operation register | R7 clarification 6; r0.2 directive §2 |
| `SCI-RTC-REQ-060` | §3 | Exact scan/beam/source response | r0.2 directive §3 |
| `SCI-RTC-REQ-061` | §5 | Notch specification and sweep | r0.2 directive §4 |
| `SCI-RTC-REQ-062` | §6 | Broad-band design register | r0.2 directive §5 |
| `SCI-RTC-REQ-063` | §7 | Constrained tap/order selection | r0.2 directive §6 |
| `SCI-RTC-REQ-064` | §4 | Donor meaning separation | r0.2 directive §8 |
| `SCI-RTC-REQ-065` | §§4, 10 | Beammap circular-factor rejection | r0.2 directive §8 |
| `SCI-RTC-REQ-066` | §8 | Decimation scientific justification | r0.2 directive §9 |
| `SCI-RTC-REQ-067` | §§7--8 | Timing/coordinate registration | r0.2 directive §§6, 9 |
| `SCI-RTC-REQ-068` | §10 | Complete-plan calibration compatibility | r0.2 directive §11 |
| `SCI-RTC-REQ-069` | §12 | Scientific design studies | r0.2 directive §12 |
| `SCI-RTC-REQ-070` | §12 | Claim-layer stop rule | r0.2 directive §§12, 15 |
| `SCI-RTC-REQ-071` | §§2, 5 | Finite-attempt and accepted-plan policy | R4 correction; r0.3 directive §6 |
| `SCI-RTC-REQ-072` | §2 | Accepted-plan apply immutability | R4 correction; r0.3 directive §§1--2 |
| `SCI-RTC-REQ-073` | §§2, 5 | Complete cumulative proposal | R4 correction; r0.3 directive §2 |
| `SCI-RTC-REQ-074` | §2 | Original-input replay | r0.3 directive §3 |
| `SCI-RTC-REQ-075` | §2 | Explicit cascade authority | r0.3 directive §3 |
| `SCI-RTC-REQ-076` | §§5, 12 | Successor-attempt evaluation | R4 correction; r0.3 directive §4A |
| `SCI-RTC-REQ-077` | §§5, 12 | Artifact-aware candidate admission | r0.3 directive §4B |
| `SCI-RTC-REQ-078` | §§5, 12 | Cumulative scientific budgets | r0.3 directive §§5--6 |
| `SCI-RTC-REQ-079` | §§5, 12 | Complete-plan stability | r0.3 directive §7 |
| `SCI-RTC-REQ-080` | §§5, 12 | Typed attempt disposition and stop | R4 correction; r0.3 directive §§5--7 |
| `SCI-RTC-REQ-081` | §§2, 5 | Attempt/accepted-plan one-way provenance | R4 correction; r0.3 directive §8 |
| `SCI-RTC-REQ-082` | §§2, 12 | Final-plan restart reproducibility | r0.3 directive §8 |
| `SCI-RTC-REQ-083` | §§1, 4 | Missing aligned-partner and upstream-lineage admission failure | R7 blocker 1; R5 §II |
| `SCI-RTC-REQ-084` | §§4, 10 | Exact pair identity with asymmetric numerical treatment | R6 decision 3; R5 §II |
| `SCI-RTC-REQ-085` | §4 | Distinct native IQ mapping, ALIGN relation, and local-input reconstruction | R7 blocker 1; R6 mapping correction; R5 §III |
| `SCI-RTC-REQ-086` | §§4, 12 | Independent member validity | R5 §II |
| `SCI-RTC-REQ-087` | §6 | Nonzero coordinate-qualified optical-response check | R7 clarification 3; R5 §IV |
| `SCI-RTC-REQ-088` | §§3, 6 | Atmosphere leakage estimator and shared-data bias accounting | R6 estimator correction; R5 §V |
| `SCI-RTC-REQ-089` | §6 | Bright-source leakage estimator | R5 §VI |
| `SCI-RTC-REQ-090` | §6 | Separate diagnostic parentage | R5 §§V--VI |
| `SCI-RTC-REQ-091` | §6 | Scalar/frequency status and coordinate-comparison compatibility | R7 clarification 3; R5 §§IV--VI |
| `SCI-RTC-REQ-092` | §§4, 11 | Forbidden correction/calibration/donor routes | R5 §§IV, IX |
| `SCI-RTC-REQ-093` | §§4, 12 | Joint $x/r$ selection dependence without a fixed-state numerical branch | R7 blocker 2; R5 §§III--V |
| `SCI-RTC-REQ-094` | §§5, 10 | Compact event/treatment state, physical-time transition bound, and population-summary boundary | R8 Decision 9; R7 clarification 4; R5 §§VII--VIII |
| `SCI-RTC-REQ-095` | §§5, 8 | Spike-aware original-pair shift learning before replacement | R6 decision 2 |
| `SCI-RTC-REQ-096` | §§5, 10 | Timing-vector-derived physical transition mask, unmodeled state, and distinct propagated influence | R8 Decision 9; R5 §§VII--VIII |
| `SCI-RTC-REQ-097` | §§5, 10 | Ordinary reset, authorized carry, and no-cross-boundary donor rule | R7 clarification 5; R6 decision 2; R5 §§VII--VIII |
| `SCI-RTC-REQ-098` | §§5, 10 | Plateau/additive-offset estimator, support, reference, and state | R8 Decision 9; R5 §§VII--VIII |
| `SCI-RTC-REQ-099` | §§5, 10 | Multiple-shift identity/conflict | R5 §§VII--VIII |
| `SCI-RTC-REQ-100` | §§5--6 | Pre/post operation leakage identity | R6 response-comparison correction; R5 §§V--VI |
| `SCI-RTC-REQ-101` | §§5, 10 | Selected additive stable-plateau correction and no-gain boundary | R8 Decision 9; R5 §VIII superseded for additive correction |
| `SCI-RTC-REQ-102` | §§1, 5, 10 | Physical support/influence, additive correction, segmentation, and replacement lifecycle states | R8 Decision 9; R6 decision 2; R5 §§II--VIII |
| `SCI-RTC-REQ-103` | §11 | Conditioned-$x$-only SCI-CAL handoff | R5 §§I, IX |
| `SCI-RTC-REQ-104` | §§3, 6, 11 | Diagnostic-only atmospheric-template boundary | R6 decision 1 |
| `SCI-RTC-REQ-105` | §§1, 11 | All-class context admission, explicit plan selection, neutral bundle, and consumer routing | R9 decisions 1--3; R6 decisions 1/3; R5 §§II, IX |
| `SCI-RTC-REQ-106` | §§5, 10 | Application-context plateau support, no-invented-offset, and retain/reject disposition | R8 Decision 9; R6 plateau-support correction |
| `SCI-RTC-REQ-107` | §§5--6, 10 | Response-change block and no fitted gain-change model | R8 Decision 9; R6 response-comparison correction |
| `SCI-RTC-REQ-108` | §§1, 4, 11 | Separately authorized application-context-bound conditioned-$r$ product | R9 decisions 1/3; R6 decision 3 |

## Falsifiable predictions

| Prediction | Rationale locus | Engineering method | Packet authority |
| --- | --- | --- | --- |
| `SCI-RTC-PRED-001` | §§1--2 | Raw application-context identity/no-filter vector | R9 decisions 2/4; R4 signal decision; RC identity limit |
| `SCI-RTC-PRED-002` | §1 lifecycle table | Context-label operation and bundle neutrality fixture | R9 decisions 1/3; R6 decision 3; R4 signal decision |
| `SCI-RTC-PRED-003` | §4 | Exact raw donor comparison coordinate | R4 signal decision; SB D004; SC 2 |
| `SCI-RTC-PRED-004` | §3 | Invalid factor/domain matrix | SB D004; SC 2 |
| `SCI-RTC-PRED-005` | §§5, 8, 10 | Shift/replace/filter/CAL order and no-atmosphere-subtraction trace | R6 decisions 1--3; R4 calibration-order decision |
| `SCI-RTC-PRED-006` | §§4--5 | Target/donor impulses | RC impulse and donor cases; SC 2--4 |
| `SCI-RTC-PRED-007` | §5 | Constant vector | RC constant case |
| `SCI-RTC-PRED-008` | §5 | Step vector | RC step case |
| `SCI-RTC-PRED-009` | §5 | Ramp/moment vector | RC ramp case |
| `SCI-RTC-PRED-010` | §§5, 7 | Sinusoid grid | RC sinusoid/phase/alias cases; SC 7 |
| `SCI-RTC-PRED-011` | §5 | Notch grid/state | RC notch case |
| `SCI-RTC-PRED-012` | §10 | Invalid-coordinate/no-inference matrix | R9 decision 5; RC invalid-coordinate case; SC 12 |
| `SCI-RTC-PRED-013` | §6 | Mask boundary/dilation | RC mask-boundary case; SC 12--13 |
| `SCI-RTC-PRED-014` | §§5--6 | FIR edge vector | RC FIR-edge case |
| `SCI-RTC-PRED-015` | §§5--6 | IIR split-state vector | RC IIR state/edge case |
| `SCI-RTC-PRED-016` | §6 | Short/empty length grid | RC short-scan case |
| `SCI-RTC-PRED-017` | §10 | Typed-cause non-finite injection | R9 decision 6; RC nonfinite case |
| `SCI-RTC-PRED-018` | §§4, 11 | Direct/noncenter influence fixture | R4 influence decision; SB D005 |
| `SCI-RTC-PRED-019` | §7 | Phase-zero length/factor enumeration | SB D008; SC 7; RC odd/even specialized |
| `SCI-RTC-PRED-020` | §7 | Two-tone alias fixture | RC anti-alias case; SC 7, 9 |
| `SCI-RTC-PRED-021` | §§5, 12 | Reused-donor covariance disclosure matrix | R9 decision 7; RC donor/covariance case; SC 3 |
| `SCI-RTC-PRED-022` | §5 | Partial-response availability fixture | SB D006; SC 4 |
| `SCI-RTC-PRED-023` | §8 | Learned candidate decision table | SB D011; SC 9--11 |
| `SCI-RTC-PRED-024` | §§1--2, 8 | Context/consumer/plan restart compatibility matrix | R9 decision 2; SC 8--10; CO State |
| `SCI-RTC-PRED-025` | §§6, 8, 10 | Two-observation reset | RC observation-reset case |
| `SCI-RTC-PRED-026` | §§5, 10 | Actual-despike plus optional spike/donor-detail toggle | R9 decision 8; SB §6; RC optional-provenance case; SC 14--15 |
| `SCI-RTC-PRED-027` | §2 | Same-observation learning comparison | r0.2 directive §1 |
| `SCI-RTC-PRED-028` | §§2, 12 | Missing-predicate stop fixture | r0.2 directive §1 |
| `SCI-RTC-PRED-029` | §2 | Resolved-plan mutation fixture | r0.2 directive §1 |
| `SCI-RTC-PRED-030` | §§5, 12 | Notch design sweep | r0.2 directive §§4, 12 |
| `SCI-RTC-PRED-031` | §3 | Scan speed/direction source transfer | r0.2 directive §3 |
| `SCI-RTC-PRED-032` | §§7, 12 | Tap-count/precision sweep | r0.2 directive §§6, 12 |
| `SCI-RTC-PRED-033` | §§6, 12 | High-/band-pass source transfer | r0.2 directive §§5, 12 |
| `SCI-RTC-PRED-034` | §4 | Donor sky-mismatch fixture | r0.2 directive §8 |
| `SCI-RTC-PRED-035` | §§4, 10 | Circular Beammap-factor rejection | r0.2 directive §8 |
| `SCI-RTC-PRED-036` | §§7--8 | Delay/coordinate-direction fixture | r0.2 directive §§6, 9 |
| `SCI-RTC-PRED-037` | §10 | Cross-plan calibration comparison | r0.2 directive §11 |
| `SCI-RTC-PRED-038` | §12 | Algebra-pass/qualification-stop | r0.2 directive §§12, 15 |
| `SCI-RTC-PRED-039` | §5 | No-successor attempt without new plan | R4 correction; r0.3 directive §10.1 |
| `SCI-RTC-PRED-040` | §§2, 5 | Accepted-plan advancement and replay | R4 correction; r0.3 directive §10.2 |
| `SCI-RTC-PRED-041` | §5 | Notch-edge artifact rejection | r0.3 directive §10.3 |
| `SCI-RTC-PRED-042` | §2 | Reapply-versus-successor distinction | r0.3 directive §10.4 |
| `SCI-RTC-PRED-043` | §5 | Rejected attempt retains accepted index | R4 correction; r0.3 directive §10.5 |
| `SCI-RTC-PRED-044` | §5 | Oscillating/nonconvergent attempts | R4 correction; r0.3 directive §10.6 |
| `SCI-RTC-PRED-045` | §5 | Maximum-attempt nonconvergence | R4 correction; r0.3 directive §10.7 |
| `SCI-RTC-PRED-046` | §§2, 12 | Final-plan restart identity | r0.3 directive §10.8 |
| `SCI-RTC-PRED-047` | §4 | Missing-pair-member failure | R5 §II |
| `SCI-RTC-PRED-048` | §4 | Native IQ mapping, single ALIGN, and RTC-local round trip | R7 blocker 1; R6 mapping correction; R5 §III |
| `SCI-RTC-PRED-049` | §4 | Independent $x/r$ validity | R5 §II |
| `SCI-RTC-PRED-050` | §6 | Leakage recovery under explicit coordinate rescaling and metric | R7 clarification 3; R6 estimator correction; R5 §§IV--VI |
| `SCI-RTC-PRED-051` | §6 | Compatible versus incompatible leakage comparison | R7 clarification 3; R5 §§V--VI |
| `SCI-RTC-PRED-052` | §6 | Scalar versus frequency leakage | R5 §§IV--VI |
| `SCI-RTC-PRED-053` | §§4, 11 | Forbidden $r$ routes | R5 §§IV, IX |
| `SCI-RTC-PRED-054` | §§4, 12 | Fixed-state zero $r$ branch and joint selector uncertainty | R7 blocker 2; R5 §§III--V |
| `SCI-RTC-PRED-055` | §§5, 10 | Cross-cadence finite-transition support, original-pair learning, then correction/replacement | R8 Decision 9; R7 clarification 4; R6 decision 2; R5 §§VII--VIII |
| `SCI-RTC-PRED-056` | §5 | Donor-invariant spike/shift learning and post-segmentation replacement | R6 decision 2 |
| `SCI-RTC-PRED-057` | §§5, 10 | Ordinary reset versus separately authorized carry | R7 clarification 5; R5 §§VII--VIII |
| `SCI-RTC-PRED-058` | §§5, 10 | Multiple/overlapping shifts | R5 §§VII--VIII |
| `SCI-RTC-PRED-059` | §§5, 10 | Selected additive correction, reference/sign, transition exclusion, and no gain | R8 Decision 9; R5 §VIII superseded for additive correction |
| `SCI-RTC-PRED-060` | §6 | Pre/post leakage response | R5 §§V--VI |
| `SCI-RTC-PRED-061` | §§3, 6, 11 | Diagnostic atmosphere inertness to conditioned $x$ | R6 decision 1 |
| `SCI-RTC-PRED-062` | §§2, 7 | Early stop without no-op attempts | R5 bounded-iteration correction |
| `SCI-RTC-PRED-063` | §§1, 11 | Multi-context consumer-neutral bundle and conditioned-$x$/raw-$r$ disposition | R9 decisions 1--3; R6 decisions 1/3; R5 §IX |
| `SCI-RTC-PRED-064` | §§4, 6 | Ideal zero-leakage mapping fixture | R6 scientific falsifier |
| `SCI-RTC-PRED-065` | §§4, 6 | Known local mapping-rotation fixture | R6 scientific falsifier |
| `SCI-RTC-PRED-066` | §6 | Noisy/shared-coordinate estimator-bias fixture | R6 estimator-bias falsifier |
| `SCI-RTC-PRED-067` | §§5--6 | Optical-source versus abrupt-shift discrimination | R6 source-discrimination falsifier |
| `SCI-RTC-PRED-068` | §5 | Gradual-drift versus step discrimination | R6 drift-discrimination falsifier |
| `SCI-RTC-PRED-069` | §§5, 8 | Unsegmented-step PSD/notch contamination | R6 spectral-contamination falsifier |
| `SCI-RTC-PRED-070` | §§5--6 | Response-change additive block and no gain fit | R8 Decision 9; R6 response-comparison falsifier |
| `SCI-RTC-PRED-071` | §§5, 10 | Insufficient offset support, no invention, and application-context plateau disposition | R8 Decision 9; R6 plateau-support falsifier |

## Decision-register coverage

- `AUTHOR_DRAFT_DECISIONS.md` maps every `SCI-RTC-AUTHOR-D001` through
  `SCI-RTC-AUTHOR-D024` to its packet or later-directive basis and consequence.
- `SCIENTIFIC_OWNER_DECISION_LEDGER.md` maps every
  `SCI-RTC-OWNER-001` through `SCI-RTC-OWNER-024` to affected requirement
  IDs and an exact unavailable consequence, and records successor exclusions
  `SCI-RTC-OWNER-025` through `SCI-RTC-OWNER-028` as deferred and adds the
  r0.2 design choices `SCI-RTC-OWNER-029`--`036` and the r0.3 bounded-cycle
  choices `SCI-RTC-OWNER-037`--`050` and the r0.5 numerical/methodological
  choices `SCI-RTC-OWNER-051`--`071` and the r0.6 bounded decisions
  `SCI-RTC-OWNER-072`--`074`, the r0.8 Decision 9 entry
  `SCI-RTC-OWNER-075`, and the r0.9 clarifications
  `SCI-RTC-OWNER-076`--`083`.
- R0.4 resolves OWNER-010 and OWNER-024 by explicit scientific-owner approval;
  r0.6 resolves OWNER-072--074 and defers OWNER-068 by explicit owner
  confirmation; r0.8 resolves OWNER-075; and r0.9 resolves OWNER-076--083.
  No other open entry is silently resolved in either PDF. In particular, the
  learned safe set is undefined until OWNER-011 through OWNER-020 are resolved.

## Mechanical completeness invariant

For revision r0.9 the exact expected normative inventory is:

- 38 definitions: `SCI-RTC-DEF-001`--`038`;
- 37 displayed equation tags: `001`--`015`, `016a`, `016b`, `017`--`019`,
  `020a`, `020b`, and `021`--`035`;
- 12 assumptions: `SCI-RTC-ASM-001`--`012`;
- 108 requirements: `SCI-RTC-REQ-001`--`108`;
- 71 predictions: `SCI-RTC-PRED-001`--`071`;
- 24 author-draft decisions: `SCI-RTC-AUTHOR-D001`--`D024`;
- 63 open, 1 conditional, 14 resolved, and 5 deferred owner entries:
  `SCI-RTC-OWNER-001`--`083`.

Every ID in that inventory appears exactly once as an authority-row key in
this crosswalk or, for the two decision registers, in the explicitly named
self-crosswalk table. Any missing, duplicated, out-of-range, or additional ID
requires synchronized revision of the shared core, both outputs, crosswalk, and
decision files.
