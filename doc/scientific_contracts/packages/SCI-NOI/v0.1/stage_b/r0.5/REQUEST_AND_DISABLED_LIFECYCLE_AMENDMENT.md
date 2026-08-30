# SCI-NOI v0.1 r0.5 Request and Disabled-Lifecycle Amendment

Scientific owner: Grant Wilson. Date: 2026-08-30. Status: proposed final; not
frozen.

| State | Request/effective identity | Exact cardinality and products |
| --- | --- | --- |
| `not_requested` | Request axis is `not_requested`; no GEN method, eligibility proposition, request identity, effective-plan identity, or assignment design exists. | `N_requested=0`; no resolved design, member, UNC, or STD. |
| `explicitly_disabled` | Exact request/effective-plan identity exists; effective state is `disabled`; disabling owner, policy, and cause are recorded. | `N_requested=0`; no resolved design, member, UNC, or STD. |
| Enabled and completely resolved | Exact request/effective-plan identity exists. | `N_requested` is positive, every assignment resolves within cap, and `N_resolved=N_requested>0`; later operations remain separate. |
| Enabled and incompletely resolved | Exact request/effective-plan identity exists. | Complete design fails; no smaller successful ensemble, UNC, or STD. |

Candidate rejection is an attempt inside design construction, not a member or
member failure. The two zero-cardinality states are not aliases. D-005,
REQ-043/051, and PRED-018/026 carry this amendment without renumbering any
stable ID.
