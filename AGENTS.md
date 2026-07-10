# Citlali Refactor Agent Guide

Read these documents before making architectural changes:

1. `doc/REFACTOR_STATUS.md` - current phase, gates, and next actions.
2. `handoff/EXTERNAL_REFACTOR_ARCHITECTURE_REVIEW_2026-07-10.md` - adopted
   independent review and completion criteria.
3. `doc/STRUCTURAL_REFACTOR_PLAN_2026-06-29.md` - original goals and history.

The living status document governs sequencing when these documents differ.

## Current Direction

The refactor follows the five-phase roadmap in `doc/REFACTOR_STATUS.md`.
Safety stabilization is the active phase. Pause additional typed
analysis-control migration, compact-config production rollout, and open-ended
header subdivision until the active phase gates pass.

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

- Run the config gate with:

  `$HOME/tolteca/bin/python tools/config/run_config_preflight.py --require-all`

- Activate and run focused tests as the safety phase adds them. The current
  absence of discovered tests is a known release blocker, not an acceptable
  success condition.
- The user performs Unity builds and reductions. Do not attempt to use Unity.
- Do not push. The user controls pushes to GitHub.
- Leave unrelated dirty files unchanged.
- Commit coherent changes after local verification; do not create commits only
  to satisfy a numerical cadence.
- Update `doc/REFACTOR_STATUS.md` when a phase gate, governing decision, or
  validated snapshot changes. Use dated handoff notes when needed to preserve
  detailed continuity.

