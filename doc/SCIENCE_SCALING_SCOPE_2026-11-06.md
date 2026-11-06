# SCIENCE workload and single-node planning scope

Date: 2026-11-06. Status: owner-provided planning constraints; application design,
implementation and qualification remain separate. This records the next planning
direction after the accepted POINT assessment foundation. Grant executes all
builds, tests, probes, benchmarks, reductions and validation runners. No direct
Unity access, push, runtime replacement or execution is part of this record.

## Owner direction and available evidence

Grant reports twenty-minute timestreams with a largest available collection of
approximately twenty observations, rather than one hundred. Approximately ten
one-square-degree map test cases are also available. These are owner-reported
inventory estimates, not a bound observation manifest; no assumption is made
that the long-timestream and wide-field cases are the same collection.

Grant reports seven Unity nodes, each with two sixteen-core CPUs and roughly
370 GB of memory. Available resources may be used for a reduction, with a
one-node baseline first to avoid inter-node communication. These are planning
figures supplied by the owner, not a live hardware or scheduler inspection.
The eventual owner-run packet must bind the actual allocated cores, memory,
CPU topology, storage and build realization. No SMT multiplier or exact usable
memory amount is inferred from the approximate node description.

| Dimension | First empirical scope | Remaining boundary |
| --- | --- | --- |
| Timestream duration | Real twenty-minute observations | Bind sampling, channels, validity, segments and actual input sizes. |
| Coadd population | Distinct compatible observations up to the available approximately twenty | One hundred remains a capacity target, not a demonstrated scientific coadd. |
| Map area | Approximately ten available one-square-degree cases | Bind raw inputs versus saved products, overlap, WCS, pixel scale, arrays and companion products before choosing cases. |
| Compute | One Unity node: nominally 32 physical cores and roughly 370 GB | Bind the real allocation and demonstrate scaling within the node. |
| Multiple nodes | Available for later consideration | Distributed execution is deferred until one-node measurements establish a need. |

Repeating or synthesizing observations could support a separately prepared
capacity stress experiment, but would not establish genuine hundred-observation
science, independent noise statistics or sensitivity. No such experiment is
selected or launched by this record.

## Requirements for the architecture plan

- Carry forward the accepted POINT observation-memory handoffs, reusable
  processing state, bounded parallelism and detective-only intermediate exports.
  Recover the actual producer/consumer interfaces before deciding what is shared.
- Use the available node memory to retain expensive reusable data where the
  measured working set permits. Frozen input identities do not require every
  observation to be resident simultaneously. Size observation concurrency and
  caches explicitly; avoid automatic per-worker copies of entire campaigns or
  full-size maps. Specify controlled storage only where capacity or restart
  requirements justify it.
- Resolve one CPU and memory budget at setup across observation, network,
  segment and array work. Account for the two CPU sockets and memory locality;
  using one node avoids inter-node communication but not cross-socket memory
  costs. Worker placement and numerical-library limits need owner-run evidence.
- Preserve scientific segment boundaries and declared accumulation/equivalence
  policies when choosing batches. Parallelism must not silently change cleaning,
  coadd weights, validity, normalization, coordinate placement or feedback.
- Bind the common RA/Dec grid, calibration inputs and coadded-map feedback
  lifecycle before connecting the route. Required outputs and failure
  propagation remain explicit. POINT-derived flux calibration is still held.
- Estimate memory from samples, detectors, live copies, map pixels, array and
  companion-plane counts, iteration state and worker concurrency. Area alone
  does not determine map memory. Leave measured headroom for libraries, the OS
  and storage caching within the actual allocation.

## Proposed qualification sequence

Grant selected **152389**, the existing small POINT fixture, as the first
SCIENCE case. Connect the successor route for this one observation in RA/Dec
with coaddition disabled. Recover the bound original raw/telescope, APT and
calibration inputs from the accepted POINT evidence; record all required
mode/configuration differences rather than silently selecting new inputs or
treating POINT-derived flux calibration as qualified.

Use this small case to establish successful end-to-end publication, celestial
coordinates, units, validity and required product metadata. Compare shared
upstream behavior where its inputs and contracts remain identical. Alt/Az and
RA/Dec maps are not expected to be pixel-identical; any positional or map
comparison must account for the declared frame and gridding, and bind the
source reference/epoch as appropriate. Exact scientific criteria and a later
known-offset treatment still belong in the owner-run packet.

This fixture selection does not make the configured SCIENCE route executable:
the current POINT parser admits only POINT and disables coadd. The first
implementation milestone is the missing connected single-observation SCIENCE
route, using recovered accepted contracts and reusable POINT services.

After that small case, qualify a genuine two-observation coadd and the
known-offset check, then use the available twenty-minute inputs and progressively
larger compatible subsets up to the real collection limit. No second observation
is selected here. Exercise the wide-field cases separately before selecting any
combined long-duration, large-population, large-map case the inventory supports.

Measure stage wall/CPU time, peak memory, allocation failures, input/intermediate
I/O and scientific product equivalence. Compare within-node worker counts,
including one socket and both sockets; exact counts and run budgets belong in
the later owner-run packet. Evidence up to twenty real observations supports
that tested scope only. Larger resource projections must remain labeled as
projections until measured.

POINT remains a regression control. Its scientific rule design stays with
“Assess contract compliance”; source amplitude, shoulder recovery and absolute
accuracy remain open scientific uses. Carry relevant recovery questions into
SCIENCE qualification without treating them as accepted or inventing thresholds.

## Authority and record disposition

Documentation base: `ae57f9b991efac5e4f15300b85cecf442d194958`, clean
`codex/refactor-mainline` in the pointing-engineering-candidate checkout.
The subsequent owner selection of 152389 is recorded from clean documentation
base `4e139f48a4af50018f6c3915e6a364a890e90a99` on the same branch. The same
engineering, successor and review governance applies to this text-only update.
Risk Tier 1: durable owner direction and planning scope only. No application,
scientific contract, active runtime, architecture implementation or dependency
is changed; no runtime gate is needed for this text-only record.

Read engineering, Timestream Successor and review/conformance governance, with
incorporation recorded in [INTEGRATION_LEDGER](INTEGRATION_LEDGER.md).
[CURRENT_STATE](CURRENT_STATE.md), [ARCHITECTURE](ARCHITECTURE.md),
[SCIENTIFIC_CONVENTIONS](SCIENTIFIC_CONVENTIONS.md#science-map-bundle-identity-and-coaddition)
and the [qualification route record](POINT_SCIENCE_QUALIFICATION_PLAN_2026-11-03.md#execution-routes-must-remain-explicit)
retain their subject authority. Connected successor SCIENCE is unfinished;
retained-standard SCIENCE or historical V2 evidence does not qualify that route.

Scientific/behavioral disposition: unchanged; no new qualification claim.
Architecture/ownership disposition: constraints recorded for the subsequent
plan; no implementation design is adopted here. Repository/evidence disposition:
documentation only, original POINT application and evidence preserved. No
scientific or performance runner has been executed by Codex.
