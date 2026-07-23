# Phase 5 Integration Record Template

Use this document only after the TolTECA build review has selected an adoption
path and the final candidate has been frozen. Copy it to a dated closeout
record and replace every `TBD`. Do not edit the record, branch, or candidate
while the same-SHA validation matrix is running.

## Candidate Identity

| Field | Value |
| --- | --- |
| Frozen full commit SHA | `TBD` |
| Embedded `citlali --version` | `TBD` |
| Source branch | `TBD` |
| Destination branch | `TBD` |
| Integration operation | `TBD` |
| Proposed forensic tag | `TBD` |
| Pre-integration rollback commit | `TBD` |
| Candidate worktree clean | `TBD: yes/no` |

Any source, generated-input, dependency, or documentation commit after the
frozen SHA invalidates this record and restarts the local and Unity candidate
gates.

## Build Review

| Field | Value |
| --- | --- |
| Reviewed v4.x repository/branch/SHA | `TBD` |
| Decision | `TBD: adopt/adapt/defer` |
| Canonical configure/build command or preset | `TBD` |
| Compiler, C++ standard, and build type | `TBD` |
| Direct dependency identity | `TBD` |
| Supported standalone/TolTECA modes | `TBD` |
| Clean/no-op/incremental timing evidence | `TBD` |
| Generated version/default-config evidence | `TBD` |
| Criteria 6, 7, and 10 disposition | `TBD` |
| Known cluster-specific configuration | `TBD` |

The detailed evidence belongs in the build-review report. This table records
the selected outcome against
[`TOLTECA_BUILD_INTEGRATION_REQUIREMENTS_2026-07-23.md`](TOLTECA_BUILD_INTEGRATION_REQUIREMENTS_2026-07-23.md).

## Source Disposition

| Decision | Result |
| --- | --- |
| Historical/experimental mains | `TBD: delete/retain with reason` |
| Empty or commented placeholder sources | `TBD: delete/retain with reason` |
| `wiener_filter.cpp` placeholder | `TBD: compile coherently/delete/retain with reason` |
| Public/private header boundary | `TBD` |
| Unsupported external C++ API/ABI claim remains absent | `TBD: yes/no` |

No source cleanup is accepted merely because a file appears unused. Record the
review decision, local gate, and Unity point smoke that support the final
disposition.

## Frozen-SHA Local Gates

| Gate | Required result | Candidate result |
| --- | --- | --- |
| Supported clean configure/build | exit 0 | `TBD` |
| CTest | all discovered tests pass | `TBD: count/report` |
| Config preflight | all required checks pass | `TBD: count/report` |
| Baseline-tool tests | all discovered tests pass | `TBD: count/report` |
| Validation ledger | valid | `TBD` |
| Validation profile registry | four expected profiles | `TBD` |
| Intended-science-change ledger | valid | `TBD` |
| Session-exit audit | zero supported library exits/growth | `TBD` |
| Worktree after gates | clean | `TBD` |

## Unity Same-SHA Matrix

Build the frozen SHA once. Every row must identify that same binary version and
retain the exact numbered TolTECA YAML inputs used for the run.

| Mode | Required profile | Reduction path | Config digest | Report | Verdict |
| --- | --- | --- | --- | --- | --- |
| Point | `phase4-point-152389-v1` | `TBD` | `TBD` | `TBD` | `TBD` |
| OOF | `phase4-oof-152385-152387-v1` | `TBD` | `TBD` | `TBD` | `TBD` |
| Science | `phase4-science-152390-152392-v1` | `TBD` | `TBD` | `TBD` | `TBD` |
| Beammap | `phase4-beammap-148670-v1` | `TBD` | `TBD` | `TBD` | `TBD` |

Each verdict requires:

- completed reduction and valid required provenance;
- zero unexpected error-level records;
- exact requested low-level config;
- requested product inventory and contract;
- no skipped required comparisons; and
- numerical results accepted by the immutable profile.

A failure repaired at a new commit creates a new candidate. Do not mix passing
rows from different candidate SHAs.

## Scientific And Capability Disposition

| Field | Value |
| --- | --- |
| Accepted intended-science changes | `TBD: ledger IDs` |
| Successor validation epoch, if any | `TBD or none` |
| Supported modes/capabilities | `TBD` |
| Explicitly unsupported/experimental capabilities | `TBD` |
| Performance/RSS result or approved retained-debt disposition | `TBD` |
| Remaining retained-debt items material to integration | `TBD` |

The integration record does not redefine scientific conventions or relax a
profile. Link any new intentional behavior to the checked science-change
ledger and successor evidence.

## Canonical References

- [`ARCHITECTURE.md`](ARCHITECTURE.md)
- [`SCIENTIFIC_CONVENTIONS.md`](SCIENTIFIC_CONVENTIONS.md)
- [`adr/README.md`](adr/README.md)
- [`RETAINED_DEBT.md`](RETAINED_DEBT.md)
- [`../validation/accepted_runs.json`](../validation/accepted_runs.json)
- [`../validation/validation_profiles.json`](../validation/validation_profiles.json)
- [`../validation/product_contracts.json`](../validation/product_contracts.json)
- [`../validation/intended_science_changes.json`](../validation/intended_science_changes.json)

## Integration Authorization

| Decision | Value |
| --- | --- |
| All required fields complete with no `TBD` | `TBD: yes/no` |
| Build-review disposition accepted | `TBD: yes/no, owner/date` |
| Four-mode same-SHA matrix accepted | `TBD: yes/no, owner/date` |
| Exact candidate approved for integration | `TBD: yes/no, owner/date` |
| Actual integration commit | `TBD after integration` |
| Forensic tag created | `TBD after integration` |

Integration copies or merges the exact validated candidate. It does not
recreate the result by selecting a subset of branch commits.
