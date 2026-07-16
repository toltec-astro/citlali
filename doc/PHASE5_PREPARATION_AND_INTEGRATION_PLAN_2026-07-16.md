# Phase 5 Preparation And Integration Plan - 2026-07-16

## Purpose And Boundary

The TolTECA build owner is unavailable until the week following 2026-07-16.
That external schedule does not justify either idle work or an unreviewed waiver
of the three deferred compilation criteria in the
[`Phase 4 closeout census`](PHASE4_CLOSEOUT_CENSUS_2026-07-16.md).

The project owner therefore authorizes **Phase 5 preparation**, not final Phase
5 integration. Preparation may assemble the source disposition, candidate
discipline, validation runbook, and review packet. It may not:

- change CMake, dependency, preset, CI-build, install/export, or cluster-helper
  behavior;
- claim criteria 6, 7, or 10 are closed;
- delete or activate placeholder source entries before the build review;
- tag a final candidate or integrate it into the destination branch; or
- resume open-ended structural or scientific refactoring.

The build review remains the entry gate for final candidate freeze. It must
either close the three criteria or support explicit project-owner dispositions
that are recorded before integration.

## Work That Can Finish Now

1. Classify every production, generated, historical, placeholder, and deferred
   source family.
2. Define the one-commit cleanup decision that follows the build review.
3. Define the exact local and Unity evidence required from one frozen commit.
4. Identify the information needed to tag, review, and integrate that commit.
5. Keep post-refactor projects separate from the closeout candidate.

This document completes those preparation tasks. No production or build file
changes are required for preparation.

## Source Disposition Census

The authoritative production target remains the root `CMakeLists.txt`. The
following dispositions are based on that target, the canonical
[`architecture map`](ARCHITECTURE.md), and
`tools/refactor/refactor_inventory.py`.

| Source family | Current state | Prepared disposition |
| --- | --- | --- |
| `src/citlali/cli/main.cpp` and the seven compiled `citlali` library sources | Active production target | Retain. Any later movement requires ordinary build and affected-mode evidence. |
| Headers reached by the active target, including transitional `engine/detail` fragments | Active implementation graph | Retain for the candidate. Public/private physical reorganization is criterion 6 and remains deferred. |
| `include/citlali/core/pipeline/config_leaf_schema_generated.h` | Checked-in generated startup schema | Retain and continue guarding with `generate_config_schema_header.py --check`. |
| `src/citlali/main_old.cpp`, `mpi_main.cpp`, `kids_main.cpp`, and `lali_main.cpp` | Unbuilt historical or experimental mains | Recommend deletion in the bounded post-review cleanup. Git preserves their forensic history; they are not supported entry points. |
| Empty/commented engine sources: `todproc.cpp`, `kidsproc.cpp`, `engine.cpp`, `lali.cpp`, `pointing.cpp`, and `beammap.cpp` | Unbuilt placeholders referenced only by commented CMake entries | Recommend deleting the files and their comments together after the build review. Do not present them as module boundaries. |
| `src/citlali/core/mapmaking/wiener_filter.cpp` | Unbuilt one-include placeholder referenced by a commented CMake entry | Decide during the compiled-boundary review: either make it part of a measured coherent boundary or delete it and its comment. Do not retain an ambiguous placeholder. |
| `src/citlali/core/utils/utils.cpp` and `src/citlali/dummy.cpp` | Unbuilt empty/placeholder sources with no active target role | Recommend deletion with the bounded cleanup. |
| Configure-time headers under `build/config_header` | Generated build artifacts | Never edit or commit as source. Reproducibility and version identity belong to criterion 10. |

The proposed cleanup is intentionally one decision after the build review. It
is not a series of speculative deletion commits. A local build and tests prove
that the selected files are not part of the active graph; a Unity build plus
point gate confirms the cluster path before the candidate is frozen.

## Final Candidate Discipline

The current branch is a **provisional integration tree**, not yet a frozen
candidate. The final candidate is selected only after:

1. the TolTECA build approach has been reviewed;
2. criteria 6, 7, and 10 have a recorded closure or final disposition;
3. the source cleanup above has been accepted or explicitly declined;
4. all parallel test-suite work intended for this refactor has landed; and
5. `git status --short` is empty.

Once selected, record the full commit SHA before compiling. Do not merge,
rebase, amend, regenerate checked-in artifacts, or make documentation-only
commits while the final validation matrix is running. Citlali embeds Git
identity, so even a documentation commit creates a different candidate SHA.

## Local Gate At The Frozen SHA

Run the supported build command determined by the TolTECA review, followed by:

```bash
ctest --test-dir build --output-on-failure -j 8
$HOME/tolteca/bin/python tools/config/run_config_preflight.py --require-all
$HOME/tolteca/bin/python -m unittest discover -s tools/baseline -p 'test_*.py'
$HOME/tolteca/bin/python tools/baseline/validate_validation_ledger.py
$HOME/tolteca/bin/python tools/baseline/validation_profiles.py --list
$HOME/tolteca/bin/python tools/baseline/validate_science_change_ledger.py
$HOME/tolteca/bin/python tools/refactor/audit_session_exits.py --fail-on-growth
```

Required result: every command exits zero, all expected tests are discovered,
the config gate reports no drift or review-required reads, and the session
audit reports no supported library exits.

## Unity Same-SHA Matrix

Build the frozen SHA once on Unity. Before starting reductions, save:

- `git rev-parse HEAD`;
- `./build/bin/citlali --version`;
- the build command/preset and dependency environment identity determined by
  the build review; and
- the exact numbered TolTECA YAML files used for each mode.

Run point, OOF, science, and Beammap with that same executable and without a
commit, config, or dependency change between modes. Runs may be scheduled in
the most practical order; same-SHA identity matters more than wall-clock
proximity. A failed or incomplete mode is repaired at a new commit, which
invalidates the candidate and restarts the four-mode matrix.

After downloading each candidate reduction, run:

```bash
$HOME/tolteca/bin/python tools/baseline/validate_reduction.py \
  /path/to/candidate/reduNN \
  --profile PROFILE_ID \
  --output-dir /tmp/citlali-PROFILE_ID \
  --json-out /tmp/citlali-PROFILE_ID.json \
  --report-out /tmp/citlali-PROFILE_ID.md
```

Use these immutable profiles:

| Mode | Profile ID |
| --- | --- |
| Point | `phase4-point-152389-v1` |
| OOF | `phase4-oof-152385-152387-v1` |
| Science | `phase4-science-152390-152392-v1` |
| Beammap | `phase4-beammap-148670-v1` |

Acceptance requires all four delegated gates for every mode: completed-run and
provenance audit, exact merged low-level config, requested product contract,
and the profile's numerical comparison. There are no unexpected error-level
records, missing required products, skipped comparisons, or silent profile
changes. Existing accepted profiles are never loosened to admit the candidate.

## Integration Packet

Before tagging or integration, assemble one short closeout record containing:

- frozen commit and embedded binary version;
- source/destination branches and the chosen integration operation;
- build/dependency identity and the disposition of criteria 6, 7, and 10;
- local gate counts and reports;
- the four Unity reduction paths, profile reports, and config digests;
- accepted intended-science changes and any successor validation epoch;
- supported and explicitly unsupported capabilities;
- links to `ARCHITECTURE.md`, `SCIENTIFIC_CONVENTIONS.md`, the ADR index,
  validation ledger, product contracts, and `RETAINED_DEBT.md`; and
- the proposed forensic tag name and rollback commit.

The integration operation copies or merges the exact validated commit. It does
not recreate the refactor by selecting a subset of commits. The granular
branch and its tag remain available for forensic history.

## Decisions Needed After The Build Owner Returns

The review must answer only the questions that are actually blocking closeout:

1. What C++ build and dependency path will TolTECA support on the collaborating
   clusters?
2. Which headers are intended public interfaces, and what isolation matrix is
   practical for that path?
3. Is a broader compiled boundary useful in that topology, or should criterion
   7 receive an explicit scope disposition based on the neutral measurement?
4. What clean, pinned test lane is proportionate for four clusters?
5. Should `wiener_filter.cpp` become part of a coherent compiled boundary or be
   removed with the other placeholders?

The project owner must also name the destination branch and preferred
integration operation before final freeze. Those decisions do not block the
preparation completed here.

## Stop Rule

Until the build review occurs, the correct state is **prepared and waiting**.
Do not fill the interval with more config migration, header splitting,
numerical cleanup, compact-config rollout, R execution, or unrelated feature
work on the closeout branch.
