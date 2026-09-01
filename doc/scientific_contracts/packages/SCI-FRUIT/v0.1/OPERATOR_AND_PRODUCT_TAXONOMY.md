# SCI-FRUIT v0.1 — Operator, State, And Product Taxonomy

Status: Stage A vocabulary candidate; no equation or method is approved

| Candidate term | Bounded meaning | Must not be conflated with |
| --- | --- | --- |
| Upstream reduction parent | Exact immutable timestream/reduction state on which one FRUIT generation operates | Candidate map-family parent or mutable work buffer |
| Candidate map-family parent | Exact ordinary MAP, JINC, FLT-FIXED, or provisional FLT-MATCHED product considered for a route | Feedback model merely because it is map-shaped |
| Complete route predecessor `Q_k` | Complete raw/filtered observation/coadd map bundle selected from completed iteration `k` in the historical recurrence | Selected projected model, residual, or accumulated increments |
| Accepted feedback state `F_k` | Versioned state admitted to determine iteration `k+1`, including exact predecessor/model identity, selection/support, response, grid, and lineage | `Q_k` alone unless the contract explicitly equates them |
| Feedback-model estimand | Scientific sky/model quantity FRUIT intends to project and update | Map signal, catalog, fitted source list, or terminal science product by default |
| Model-construction operator | Exact selection/synthesis mapping from admitted parent/state to feedback model | Forward projector or map estimator |
| Selection state | Exact support/source/eligibility decisions used to build a model | Formal source catalog or mere validity mask |
| Forward projector | Mapping from the feedback-model identity into the exact timestream sample domain | Interpolation choice alone, inverse mapmaker, or response correction |
| Projected model | Realized timestream-domain signal predicted by the feedback model/projector | Observed timestream or residual |
| Residual input | Exact upstream timestream minus the projected model under declared order | Noise realization or source-free truth |
| Residual processing | Admitted RTC/PTC operation on the residual input with exact policy/state | FRUIT update law or evidence that signal is absent |
| Processed residual | Output of residual processing before any owner-selected add-back/update operation | A pure physical-noise sample |
| Update contribution | Optional object associated with one transition; status may be diagnostic difference, equivalence witness, causal update, or admitted science product | Independently calibrated sky map or required permanent product by default |
| Add-back/rejoin semantics | Normative declaration of which model bypasses which residual-only operators, where it rejoins, what later operators act on it, and its next-map response | Literal array addition, inversion of cleaning, or proof of unbiasedness |
| Update/accumulation operator | Owner-selected recurrence producing immutable next model/state from prior state and new information | File overwrite or loop counter increment |
| Iteration product | Exact product/state realized at one absolute iteration | Terminal product |
| Terminal selector | Rule selecting a terminal iteration/product from realized states | Convergence diagnostic, hard maximum, or last file |
| Fixed-state response | Response conditional on exact frozen model/selection/learned/apply state | Full-procedure response including selection/learning/stopping |
| Procedure response | Response of the complete iterative/selection/stopping method for a named target population | A single fixed-state Jacobian or injection ratio |
| Conditional uncertainty | Uncertainty for an exact frozen state/product under a named method | Complete learning/stopping/restart variation |
| Replay uncertainty | Separate member-specific partial/complete FRUIT replay method | Fixed-state conditional uncertainty |
| Diagnostic | Causally inert measurement reported for review | Stop criterion unless explicitly admitted |
| Checkpoint | Complete causal state for exact continuation | Map seed, QA archive, or diagnostic bundle |
| Science profile | Versioned FRUIT recovery objective, metric priorities, validity domain, and qualified claim boundary | Observation-mode name alone, downstream OOF/SZE inference, or permission for ad hoc tuning |
| Parameter policy | Exact fixed parameters or deterministic bounded diagnostic-to-parameter mapping | Unrecorded manual choices or implementation defaults |
| Qualification identity `K` | Exact `(method, parameter policy, stopping policy, profile, domain, historical control, protocol, evidence)` tuple | Universal statement that “FRUIT is qualified” |
| Development population | Identified inputs permitted to influence hypotheses, tuning, adaptation, and prospective thresholds | Untouched qualification evidence |
| Qualification population | Held-out population opened only after the complete method and decision protocol freeze | Additional development data after exposure |
| Challenge population | Predeclared edge/near-boundary population used under a frozen characterization or decision role | Post-hoc replacement for failed qualification cases |
| Experimental override product | Realized output with explicit nonstandard method/policy identity and downgraded claim class | Ordinary qualified product or silent nearest-profile fallback |
| Applicability population | Prospectively defined finite population or superpopulation target, with exact sampling, condition distribution/weights, signal/nuisance domain, and missing/failure rules | Convenient available observations or post-hoc favorable subset |
| Oriented paired improvement | Metric-specific signed candidate-minus-historical difference defined so positive favors the candidate | Absolute truth accuracy, unpaired cohort difference, or scalar global quality |
| Protected guardrail | Prospectively frozen non-inferiority, safety, adverse-tail, stratum, failure, or catastrophic-regression limit | Primary improvement target or development-time preference |
| Pareto specialization | Material profile/domain benefit that justifies a separately qualified exact method identity while protected guardrails pass | Small benchmark fluctuation, universal dominance, or permission for run-by-run choice |
| Historical fallback | Exact historical Citlali method/control selected when no narrower qualified identity applies or a candidate fails its guardrails | Proof of historical correctness, hidden fallback, or automatic numerical route availability |

## Naming Rule

The future author should assign stable product and method identities only after
the owner selects the estimand and recurrence. A stable identity is compatible
with bounded retention, exact reconstruction, or compaction when the approved
causal and reproducibility rules allow it. Current configuration terms and file
labels are implementation evidence and must not determine the taxonomy.
