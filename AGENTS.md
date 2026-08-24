# Citlali Refactor Agent Guide

Read these documents before making architectural changes:

1. `doc/REFACTOR_STATUS.md` - current phase, gates, and next actions.
2. `doc/PHASE5_PREPARATION_AND_INTEGRATION_PLAN_2026-07-16.md` - active
   compilation-independent closeout lane and final-candidate discipline.
3. `doc/TOLTECA_BUILD_INTEGRATION_REQUIREMENTS_2026-07-23.md` - required
   outcomes and evidence for the build/integration review.
4. `doc/TOLTECA_BUILD_INTEGRATION_REVIEW_2026-07-26.md` - reviewed Conan 2
   evidence, Adapt decision, compatibility gaps, and bounded integration path.
5. `doc/PHASE4_1_TOLTECA_CONFIG_STRUCTURE_PLAN_2026-07-16.md` - completed
   four-mode numbered-YAML authoring work and gates.
6. `doc/PHASE4_2_TECHNIQUE_PERFORMANCE_REVIEW_PLAN_2026-07-16.md` - completed
   whole-code review rubric, evidence labels, priorities, and stop rules.
7. `doc/ARCHITECTURE.md` - active components, ownership, dependencies,
   compatibility boundaries, and extension rules.
8. `doc/SCIENTIFIC_CONVENTIONS.md` - identities, units, frames, validity, and
   validation routing.
9. `doc/RETAINED_DEBT.md` - deliberate limitations, owners, triggers, and exit
   conditions.
10. `handoff/EXTERNAL_REFACTOR_ARCHITECTURE_REVIEW_2026-07-10.md` - adopted
   independent review and completion criteria.
11. `doc/STRUCTURAL_REFACTOR_PLAN_2026-06-29.md` - original goals and history.

The living status document governs sequencing when these documents differ.
Durable architecture decisions are indexed in `doc/adr/README.md`.

## Current Direction

The refactor follows the roadmap in `doc/REFACTOR_STATUS.md`. Project-owner
additions Phase 4.1 and Phase 4.2 are complete. The accepted four-mode TolTECA
numbered-config kits remain opt-in through TolPROJ's `--refactor` path, and the
whole-code review's untriggered optimization candidates remain governed by the
retained-debt register. The TolTECA Conan 2 implementation has now been
reviewed and the project selected the bounded **Adapt** path. Phase 5 build
integration may proceed under
`doc/TOLTECA_BUILD_INTEGRATION_REVIEW_2026-07-26.md`: preserve the full
refactored CLI and validation surface, keep the existing build available until
the successor passes its gates, and do not combine build integration with
numerical algorithm changes. Open-ended header subdivision remains paused.

Do not broaden mature RTC, PTC, JINC, or Wiener-filter algorithms while fixing
their contracts. Preserve numerical behavior unless a change is named,
measured, and recorded as intentional.

## Architectural Rules

- YAML is accepted at the application boundary, not throughout core logic.
- Requested config, effective plans, and realized observation state are
  distinct objects.
- Each migrated fact has one authority and, when needed, a one-way legacy
  adapter. Do not add bidirectional typed/legacy synchronization.
- Required output failures propagate to the CLI. Library code does not call
  `exit()` and does not log-and-forget required failures.
- Reduction, observation, and scan state have explicit lifecycle owners. Do
  not add process-lifetime statics or singletons.
- Treat the current `Engine` as a compatibility boundary. Do not add new
  cross-cutting public state to it.
- New interfaces state scientific identity, units, coordinate frame, shape,
  indexing, and missing/non-finite policy where applicable.
- Public headers must compile in isolation. Context-dependent implementation
  fragments remain private or are grouped behind a coherent interface.
- Optimize hot paths only from measurements. Keep allocations and virtual
  dispatch out of established inner loops unless evidence justifies them.
- Validation must cover the behavior touched. A successful run has zero
  unexpected error-level messages.
- R integration remains structure-only until its measured-channel contract is
  explicitly approved.

## Working Protocol

- Use `$HOME/tolteca/bin/python` for Python.
- Build locally with:

  `cmake --build build --target citlali_cli -j 8`

- The build refreshes `citlali_config/gitversion.h` from the current Git
  checkout before compiling. After changing commits, verify that
  `build/bin/citlali --version` reports the same short revision as
  `git rev-parse --short HEAD`; treat a mismatch as a failed build and do not
  use that executable for validation or reduction provenance.

- Run the config gate with:

  `$HOME/tolteca/bin/python tools/config/run_config_preflight.py --require-all`

- Run focused tests for the behavior touched. CTest, baseline-tool tests, and
  the full config preflight are active gates; skipped required data is not a
  successful validation.
- The user performs Unity builds and reductions. Do not attempt to use Unity.
- Do not push. The user controls pushes to GitHub.
- Leave unrelated dirty files unchanged.
- Commit coherent changes after local verification; do not create commits only
  to satisfy a numerical cadence.
- Update `doc/REFACTOR_STATUS.md` when a phase gate, governing decision, or
  validated snapshot changes. Use dated handoff notes when needed to preserve
  detailed continuity.
