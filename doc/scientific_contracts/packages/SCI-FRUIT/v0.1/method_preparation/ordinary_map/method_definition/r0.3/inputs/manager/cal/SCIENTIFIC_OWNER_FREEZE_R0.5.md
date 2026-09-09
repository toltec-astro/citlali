# SCI-CAL v0.1/r0.5-r0.4 Scientific-Owner Freeze

Date: `2026-08-23`

Owner: Grant Wilson

Approval record:

> The scientific owner answered “I approve” to `WP3-OWNER-D001`, which asked
> whether SCI-CAL v0.1 science-rationale r0.5 and engineering-conformance
> r0.4 should be frozen as the exact scientific authority while observational
> validation and achieved-performance acceptance remain pending.

Normalized freeze directive:

> Freeze SCI-CAL v0.1 at science-rationale r0.5 and
> engineering-conformance r0.4. Keep implementation conformity, observational
> validation, achieved accuracy, and achieved-performance acceptance
> unestablished until their separate evidence exists.

Status: Scientific authority frozen; implementation conformity and achieved
performance not assessed under this contract.

## Frozen Authority

This owner statement promotes the content-bound candidate at commit
`0b3cfb24070c1eda04dbda7633accf40e2e8b852` without scientific change. It
freezes SCI-CAL v0.1/r0.5-r0.4 as the active scientific authority for:

- the standalone science-team rationale and the shared notation,
  definitions, assumptions, equations, requirements, and edge predictions;
- the engineering conformance view of the same scientific authority;
- the approved `SCI-CAL-OWNER-Q01--Q09` dispositions;
- the exact atmosphere operator, photometric convention, CAL-before-PTC
  ordering, source/child-APT lineage, and claim-layer boundaries;
- all 11 assumptions, 50 requirements, 30 edge predictions, and 50 crosswalk
  rows; and
- the two canonical frozen PDF renderings recorded in `pdf/README.md`.

The status promotion removes active draft labels and uses
“achieved-performance acceptance” for the evidence-dependent Q09 gate. That
terminology distinguishes contract-authority approval from later acceptance
of achieved observational performance; it changes no equation, requirement,
prediction, benchmark, scientific policy, or unavailable-state consequence.

## Exact Frozen Hashes

| Artifact | SHA-256 |
| --- | --- |
| `README.md` | `c6db96f6011d8c6f3c8f8ac0b217e650fac39df6b6ee2422c8a681e5eb3be067` |
| `CROSSWALK.md` | `d8e99a105f832d21f536137b931ec6ae84e41a10f66df74926bfbcb7814e432a` |
| `DECISION_LOG.md` | `1aafd3989123a015e23893856efde37745013743fbb1f5573db95887a8eda9a5` |
| `SCIENTIFIC_OWNER_DECISIONS_R0.5.md` | `32810d617c0df166415c1fe172054dc32ea89e3489137ff0ee1ed29280410506` |
| `SCIENTIFIC_OWNER_DECISION_LEDGER.md` | `39b0f3b495e61699090d30eb497d09dcb003ab5c6a40a82d0c46178b79386ca1` |
| `SCIENTIFIC_ENGINEERING_CONSISTENCY_R0.5.md` | `e649cb6a7d84b8ff1e098a7e1b42b49722381755d49e70dc1eea961d7dd2a37f` |
| `SCIENTIFIC_ENGINEERING_R0.5_R0.4_BUILD_REVIEW.md` | `e47599858e1b0e95b09fa78b695013bf9c9be79887ac7dda62a784bbf32548fd` |
| `pdf/README.md` | `24245914000a9bae674a079cece8445defb94097a0e053b29e78d446a766f800` |
| `src/common/preamble.tex` | `beb69a3e05260db7561be87aca04f6bf3eee4ea36f80411d1b0baddb5e0ac7a9` |
| `src/common/notation.tex` | `c5d4ec103d6a01eaec15bcb816d019d78b6aaf8700998e563e0849122421f4db` |
| `src/common/definitions.tex` | `2a9c91f485ea7d41ba6d5b13c77f77b8314612da4bc4b59eb1228235374b71b5` |
| `src/common/assumptions.tex` | `6da85c4a44d5b20b222f5796dae8922594f1b1d043a9ac993f5fb6f12059eea9` |
| `src/common/equations.tex` | `b8027f5e0b787a95708be6cc51018bb993d32f4863c34c4b1b55dd71bd2d3322` |
| `src/common/requirements.tex` | `80054fbd526d6a0878f6724c620024955062d41fca1273b85836ead3ee9b5f74` |
| `src/common/edge_cases.tex` | `45ff8dc1befe04216f9e93cb8f2713f2bfdb3799459714e71187d7202dc084c0` |
| `src/scientific-rationale.tex` | `f780cef579cb39ac1ed748f021a0024d9f1576d7960fe3b4363557c10bbff318` |
| `src/engineering-conformance.tex` | `4d807192adabf0dc0fc8dc20505c528eca865d484dd5891e37cb5913bf138f7a` |
| `pdf/SCI-CAL-SCIENTIFIC-RATIONALE-v0.1.pdf` | `fa6a11a359bcc4f54cb75ab5057ba56cf76fa8eef8d28ac3bcb7954963a12034` |
| `pdf/SCI-CAL-ENGINEERING-CONFORMANCE-v0.1.pdf` | `07dd88895b21ee02eca611bba8d1adcf90f9c37439f791437c4f46e351700101` |

## Claim Boundary

This freeze establishes document identity and scientific authority. It does
not establish implementation conformity, atmosphere-model fidelity,
executed validation, observational repeatability, achieved accuracy,
achieved-performance acceptance, total calibrated uncertainty, total
significance, science qualification, production readiness, MAP availability,
or clean-room audit-finding closure.

The 1%, 5%, and 5--10% figures remain reporting benchmarks rather than
automatic pass/fail ceilings. Q09 authorizes the validation workflow and its
truthful reporting requirements; it does not manufacture achieved evidence.

This artifact supplies the CAL freeze authority required by WP-3. `F-016`
remains open until the final source-binding and clean-room re-audit work
defined for WP-7.

## Change Control

The freeze promotion is status-only. Any future substantive scientific
correction, newly available scientific mechanism, changed boundary, or
changed policy requires explicit owner authority and a versioned successor or
formally reopened revision. Later implementation or observational evidence
may be attached under its own identity without silently editing frozen
r0.5/r0.4 authority.
