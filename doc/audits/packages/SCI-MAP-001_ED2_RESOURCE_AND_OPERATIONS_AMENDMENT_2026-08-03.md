# SCI-MAP-001 ED2 resource and human-operations amendment — 2026-08-03

Status: owner approved bounded clarification; no Unity launch or external evidence
is authorized.

Decision ID: `MAP-UNITY-ED2-OPS-RESOURCE-001`

Authority: project owner

## Resource ceiling

The existing ED2 temporary 200-GiB (`214748364800` byte) ceiling is accepted
as practical and remains a hard, inclusive limit. It covers both retained
full-PTC captures, capture intermediates, logs, compact evidence, analysis
outputs, and return construction under the declared governed roots.

This does not raise the ceiling, permit deletion to fit, or accept an aggregate
planning estimate as a sufficient live bound. Before every material stage, the
successor must still make a component-wise conservative projection from
digest-bound metadata, account for every write location, check both logical and
allocated use, and verify available filesystem space.

## Human-managed Unity operation

The owner will operate Unity using the established naive-reduction pattern and
takes responsibility for reviewing each printed pre-stage record before
manually starting the next command. The ED2 runbook must therefore be a
human-managed sequence:

1. a standalone preflight/resource-record command that does not submit work;
2. human inspection of its pass record; and
3. a separate, explicitly invoked staging/capture/submission command.

No local driver or compound runbook block may contact Unity, invoke `sbatch`,
or automatically continue from a preflight command. This replaces the need to
treat shell fail-fast behavior within a combined preflight-and-submit block as
an evidence blocker; it does not weaken the resource gate or make failed
preflight results ignorable.

## Remaining bounded package work

The successor may correct only the resource-control findings from its stop:

- place all resource records, projections, analysis, manifests, and return
  products under the declared governed roots;
- account for and gate fresh-project creation, raw/ECSV staging, duplication,
  and configuration preparation before their first write;
- derive conservative component-wise bounds for capture, compact production,
  analysis, and return stages from typed/digest-bound metadata; and
- revise the runbook into the human-managed sequence above.

No change to the repaired Citlali application, MAP estimator, observations,
arrays, cases, products, numerical gates, 200-GiB ceiling, retained-capture
policy, CAP transfer boundary, or scientific claim is authorized. On any
resource projection above the ceiling or any unproved component bound, the
task stops and returns to the coordinator.

