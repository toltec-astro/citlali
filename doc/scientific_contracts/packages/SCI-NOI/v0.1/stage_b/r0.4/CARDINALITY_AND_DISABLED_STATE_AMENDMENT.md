# SCI-NOI v0.1 r0.4 Cardinality and Disabled-State Amendment

Scientific owner: Grant Wilson. Date: 2026-08-30. Status: proposed final; not
frozen.

| State | Exact cardinality | Design/members | UNC/STD disposition |
| --- | --- | --- | --- |
| Disabled/not requested | `N_requested=0` | No design and no members; never a successful empty ensemble. | No UNC and no STD. |
| Enabled and completely resolved | `N_requested` is a positive integer and `N_resolved=N_requested>0`. | Every requested member assignment resolves within its preregistered cap. | May proceed only through separate admission and realization gates. |
| Enabled and incompletely resolved | At least one required member exhausts its cap. | Complete design fails; no smaller successful ensemble is published. | UNC and STD unavailable. |

Candidate rejection is an attempt inside design construction, not a member or
member failure. This amendment is represented normatively by D-005,
ASM-006, REQ-043, and PRED-018 without renumbering r0.3 IDs.
