# SCI-FRUIT v0.1 — EL-G0 Access Incident 001

Incident ID: `SCI-FRUIT-EL-G0-ACCESS-INCIDENT-001-2026-09-01`

Detected: `2026-09-01`

Status: **recorded; historical validation outcome scope quarantined from future
qualification; Gate 0 may continue under narrowed searches**

## Event

During the first read-only control-recovery pass, a broad repository search
included the shell path expression `validation/fruit_loop*`. The command was
intended to locate historical-control identity and configuration references,
but it printed rows from at least:

`validation/fruit_loop_population_stage_a_analysis_2026-07-26/iteration_metrics.csv`

The visible output contained historical per-iteration scientific metrics and
observation identities. The output was truncated, so the safe incident scope
is the complete repository-local `validation/fruit_loop*` evidence family and
every exact, near-duplicate, source, injection, nuisance, observation, or
descendant identity represented by it.

## Immediate Containment

1. No further content search or read of `validation/` is permitted during Gate
   0.
2. All future repository searches explicitly exclude `validation/**` unless a
   later owner record changes the access plan.
3. No numerical method, parameter, threshold, candidate, or population choice
   is inferred from the exposed values.
4. The exposed evidence remains historical/implementation-informed and outside
   any implementation-blind Stage B packet.

## Qualification Consequence

No development, qualification, or challenge population had yet been created,
so no frozen held-out population was opened. Nevertheless, the conservative
lineage rule applies: the complete incident scope is ineligible for untouched
qualification evidence for this lane. It may enter a development or historical
descriptive role only after an exact later Gate-D decision identifies that
role; it receives no automatic admission.

A future population custodian must exclude the incident scope and its lineage
descendants from qualification, or return a replacement-population/no-go
decision. Unknown overlap fails closed.

## Gate Disposition

Gate 0 continues because the incident did not open a frozen qualification
population, did not tune or rank a method, and can be contained by a complete
family-level quarantine. This incident must appear in every later Gate-D,
Gate-F, Gate-Q, evidence, and qualified-method record.
