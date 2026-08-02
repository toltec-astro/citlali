# Scientific-Audit Manager Instructions

These instructions govern coordinator and audit-manager actions that can
authorize computation, external evidence, repair, re-audit, or production.
They supplement the package lifecycle in `README.md`; they do not transfer
scientific authority from the project owner to the manager.

## Before package dispatch

1. Freeze the package ID, tier, included scope, exclusions, governing source
   SHA, dependencies, consumer restrictions, and inbound-handoff manifest.
2. Confirm that the audit thread and worktree are fresh and role-separated
   from implementation, repair, and re-audit work.
3. Route only approved `pre_core_authority` handoffs before the independent
   core freeze. Quarantine `post_core_evidence` until afterward.
4. State explicitly what the dispatch does not authorize: application edits,
   integration, external execution, repair, production, and subsequent audit
   launches remain separate actions.
5. Apply `FRAMEWORK-SCOPE-001` before substantive work. The task's first
   return is a concise scope checkpoint stating exact allowed paths and
   deliverables; required/prohibited local Citlali reduction and Unity
   evidence; planned tests; permitted delegation/review; first viable
   artifact; and the next return point. Silence prohibits a capability.
6. Do not authorize a local Citlali reduction by implication. Unity is the
   default reduction-evidence lane. A local reduction requires a written
   local-only rationale and a named scientific or engineering purpose.
7. Bound deliverables by their named path and class. A task must return before
   adding a new executable helper, schema, wrapper, verifier, validation
   campaign, delegation, independent review, or generalized control system.
   Do not treat a broad phrase such as “provenance,” “validation,” or
   “evidence package” as standing authority for such expansion.
8. Review the task at the scope checkpoint, after its first viable artifact,
   and before costly or local execution. The first viable artifact proves the
   named output exists; it is not permission for a perfection or framework
   construction pass.

## Before any costly numerical study

Apply the canonical
[numerical proportionality and cost-control policy](NUMERICAL_PROPORTIONALITY_AND_COST_CONTROL_POLICY.md).
A study is costly when repeating it materially exposes wall time, compute,
memory, storage, external scheduling, scarce data, or human operational cost;
there is no universal resource threshold.

The manager must obtain and freeze:

1. the scientific protocol, candidates, case set, metric, domain, and
   acceptance gates;
2. a complete Tolerance-and-Stop-Condition Register covering every abort,
   invalidation, and scientific-failure route;
3. a source-level guard-site census;
4. a model-free preflight report covering every frozen tuple and every
   deterministic conversion, boundary, dispatch, branch, and formatting path;
5. a salvage plan that separates raw-model, parser/admission, evaluator, and
   scientific-decision validity;
6. an independent review by a role or task that did not author the runner or
   register; and
7. an execution-readiness certificate bound to the exact artifacts by
   SHA-256.

Run the mechanical gate exactly as documented by
`tools/audits/validate_expensive_study_controls.py --help`. Schema validity is
not launch authority. A launch requires a successful `--launch-gate` result,
the manager's `ready` certificate, and any separate human/external execution
authorization required by the package.

The manager may not:

- accept an unregistered source assertion as an invalidating condition;
- accept a source abort relabeled as a registered warning without changing and
  preflighting its implemented route;
- treat equivalent floating-point construction paths as exactly identical
  unless exact identity is itself the protected requirement;
- approve an aborting numerical tolerance without a derived bound or
  quantified propagation into the final scientific metric;
- let a deterministic guard first become reachable during costly model work;
- classify a study as cheap merely to bypass the gate; or
- discard valid raw evidence merely because a conservative predecessor
  protocol assumed a fresh cache.

## While a costly study executes

- Preserve raw outputs and immutable sidecars independently of evaluator
  state and summary products.
- Record every condition by stable ID, operands, action, affected artifact,
  and branch.
- A Class C scientific failure is a valid result, not corrupt evidence.
- An unknown hard guard is a framework incident. Stop safely, preserve the
  evidence, and do not infer that already written raw artifacts are invalid.
- Do not inspect partial decisive metrics, candidate rankings, or maximum
  error locations when the protocol reserves them for a complete
  confirmatory evaluation.

## After a stop or harness failure

Perform the salvage review before authorizing regeneration. Keep raw-model,
parser/evaluator, and scientific-decision validity separate. Bind reusable
artifacts and their execution context by digest, identify the exact anti-join
of missing evidence, and freeze a successor evaluator before calculating any
decisive metric. A full repeat requires a concrete scientific, provenance,
warning-admission, completeness, parsing, or independence reason.

## Review and escalation

Bring the project owner only choices that can change scientific meaning,
accepted risk, production use, resource allocation, or governance policy.
Engineering implementation details that merely realize an already approved
contract stay with the manager and repairer. Record any proposed expansion of
scope separately; do not let numerical-control work become an application or
scientific redesign.
