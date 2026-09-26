# Codex directive: inventory local_data and purge saved Citlali RTC/PTC timestreams

**Owner:** Grant Wilson  
**Date:** September 26, 2026  
**Status:** Execute the inventory, bounded local purge, verification, and report; then stop.  
**Scope:** `~/work_toltec/local_data`, including all of its local subdirectories. No new scientific experiment.

> Unity is the authoritative archive for TolTEC raw data and the data products maintained by other team members. `local_data` is a disposable local repository for Grant's testing. Inventory what is actually here before choosing a resident observation set or downloading anything else. Purge all saved Citlali-produced RTC and PTC timestream products: we will not use those saved files again, or will regenerate the needed processing from upstream inputs.

## 1. Authority and boundaries

This instruction supersedes earlier requirements to retain saved Citlali RTC/PTC timestreams as shared processing parents, immutable checkpoints, historical campaign dependencies, or unique-product exceptions. It also supersedes any requirement to demonstrate numerical regeneration before deleting those particular derived products.

**After identifying the eligible files and completing a bounded safety review, execute their removal in this task. Do not return another routine request for owner approval.** A historical manifest reference, lack of an identical surviving copy, or untested regeneration recipe is not a reason to retain an identified, inactive Citlali RTC/PTC timestream.

This is not permission to delete raw acquisition files, tune inputs/products, APTs or other calibration inputs, team-produced inputs, unrelated personal testing data, source code, or independently useful final maps and scientific records. It authorizes no deletion or modification on Unity, no filesystem-snapshot or backup cleanup, and no broad `git clean` or worktree pruning.

Recover applicable repository instructions and current status. The September 26 importance/compaction closeout at reported local commit `84eb4d919` is a discovery anchor, not an assumption about current HEAD. Preserve unrelated changes. Keep CUT3, the second-observation comparison, and other new science work parked.

## 2. Inventory the whole local_data tree

Resolve the actual root and enumerate all local contents, not just the recent POINT campaign. Report inaccessible or unreadable locations rather than treating them as empty. Do not follow symlinks outside the root, traverse mounted remote storage, or infer ownership from a directory name alone. Record symlink targets and separately account for hard-linked files where present.

Provide top-level and useful product-class totals, with enough detail to identify the largest contributors. Classify at least:

- Original raw acquisition `.nc` files and their observation groupings; raw and processed tune products; APTs, matching records, and calibration/metadata inputs.
- Citlali RTC outputs, PTC outputs, their saved copies/checkpoints, and associated processing caches.
- Maps, figures, compact measurements, configuration/provenance records, other generated products, and temporary files.
- Other testing projects and products from other team members, plus genuinely unknown contents.

Report logical bytes and allocated-file-block accounting separately, noting limitations from filesystem sharing. Measure free space before and after deletion. Do not call logical size a physical-space saving. Do not perform a whole-tree payload rehash or load full timestream arrays merely to inventory them; reuse existing identities and inspect headers or producer metadata where sufficient.

This task inventories the entire root but deletes only the class authorized below. Do not move or rename observation directories, consolidate every historical campaign, download new datasets, or create replacement bulk archives during this pass.

## 3. Identify complete raw observation packages for testing

Discover observations from the actual local holdings. Do not limit the search to previously discussed obsnums or assume that those obsnums are the best resident set.

For each distinct observation/acquisition grouping, record:

| Field | Required information |
|---|---|
| Identity | Obsnum and any relevant sub-observation, scan, date, source, and observing-mode identifiers. |
| Raw inventory | All available acquisition parts, networks/arrays, time coverage, and required telescope/pointing streams. |
| Expected coverage | Evidence for the expected acquisition inventory; missing, duplicated, truncated, or conflicting parts. |
| Tune association | Exact associated tune obsnums/identities, locally available raw tune data and processed tune products, and the evidence for the association. |
| Other dependencies | Exact known APT, detector matching, calibration, and metadata dependencies; local availability and gaps. |
| Locations and sizes | Actual local paths; raw, tune, and other-input byte totals; shared dependencies counted once in combined totals. |
| Archive provenance | Known Unity location and retrieval command/recipe, with verified versus inferred locations distinguished. |
| Testing role | What the observation could usefully exercise, based on its metadata and established history. |
| Readiness | Explicit completeness/readiness classification and remaining uncertainty. |

Use classifications such as **complete raw package with identified local tune dependencies**, **complete raw package but tune/calibration dependency incomplete**, **partial raw package**, and **completeness unverified**. Distinguish raw completeness from readiness for a particular reduction: a missing APT need not mean the raw acquisition itself is incomplete.

A collection is not complete merely because a few `.nc` files open or its directory is named with an obsnum. Compare actual parts, network coverage, and required supporting streams with acquisition metadata or an authoritative inventory when available. Do not impose a fixed network count on every historical observation; distinguish networks absent during acquisition from files missing locally.

Use lightweight header/structure checks, acquisition metadata, saved manifests, and established metadata tools. An inventory-complete package is not a claim that all detector samples are scientifically good. No new reductions or comprehensive data-quality campaign are requested.

Resolve tune associations from metadata and established processing records, not numerical proximity between obsnums. Account for network-specific associations when present. If an association cannot be established, report it as unknown rather than inventing one.

Existing authorized read-only Unity access may be used to resolve expected inventories, archival locations, and dependency identities. Do not launch bulk transfers or modify remote content. If that access is unavailable, complete the local inventory and mark the affected conclusions unverified.

Finish with a compact list of the **complete local packages we can work with now**, followed by nearly complete packages and their specific missing pieces. Recommend a small possible resident test corpus based on actual readiness, scientific coverage, and measured size. Do not automatically download, select a new experiment, or discard raw observations to implement that recommendation.

## 4. Purge saved Citlali RTC/PTC timestream products

Identify products through their actual producer, contents, manifests, or processing lineage—not extensions alone. Eligible products include saved Citlali RTC-stage and PTC-stage detector-versus-time outputs, their x/r counterparts where present, and copies of those outputs inside:

- Shared processing caches and purported indispensable processing parents.
- Per-observation, per-network, per-cycle, replay, injection, and validation directories.
- Checkpoints, restart packages, scratch directories, and packaged/compressed copies under `local_data`.

Changing a filename or calling a saved PTC output a native parent does not put it outside this purge. Include redundant sample axes/masks that belong solely to a discarded saved-timestream package; preserve independently useful small configuration, fit, summary, and provenance records.

Do not sweep every `.nc`, `.npy`, `.npz`, binary, or archive file. Raw TolTEC streams and raw/tune/calibration inputs are protected even when their formats resemble derived output. Large basis matrices or other numerical products that are not themselves part of the identified saved-timestream package should be inventoried for a later decision, not silently added to this deletion authority.

For a mixed container holding both eligible timestreams and protected or independently useful products, preserve the non-target contents before removing the eligible payload if a narrow, verifiable separation is practical. Otherwise retain and report that exact mixed container as an exception. Do not delete non-target data merely to complete the purge.

Before deletion:

1. Produce an exact machine-readable target manifest with path, product classification, size, identifying file metadata, and the reason it is eligible. Identify the raw/tune/input exclusions explicitly.
2. Check for running writers or consumers. Do not kill jobs or delete files in active use. Defer only those live targets, with an explicit reason. A reference from an inactive historical run is not active use.
3. Check resolved paths and file stability, and perform the existing bounded independent safety review of the target list. Review is for scope and safety, not a new scientific replay campaign.

Then remove only the enumerated eligible files or exclusively eligible packages. Stop on changed or ambiguous targets; complete independent safe portions where possible and report exceptions. Remove only resulting empty directories. Do not use broad age-based deletion or move material to a same-disk quarantine and call it reclaimed space.

**No byte-identical replacement, numerical replay, end-to-end rebuild, or per-producer regeneration proof is a prerequisite for this authorized purge.** Do not preserve an identified inactive RTC/PTC output solely because a previous cleanup could not prove its regeneration.

## 5. Keep useful recovery information outside disposable storage

Before deleting the target files, preserve existing compact scientific records and the available information needed to understand or reproduce useful work. The durable copy must be outside `local_data` and must not exist only as uncommitted material in `/private/tmp`.

Use the existing permanent Citlali repository and normal owned-worktree practice for compact records. If the working checkout is temporary, ensure the records are committed into the durable repository's reachable history before removing their only local-data copies. Do not touch unrelated changes or create a new archival platform.

For useful processing routes, retain or reference:

- Raw observation and tune identities, known Unity retrieval locations/commands, and required APT/calibration/matching dependencies.
- Exact available code revisions, source changes, configurations, runtime/dependency identities, seeds where relevant, and existing execution commands.
- Compact results, final maps already retained, measurement definitions, assessments, and small failure or diagnostic records.

Preserve existing recipes rather than reconstructing a separate detailed manual for every discarded array. State missing dependencies or unverified recovery steps plainly. **Documenting a recovery gap does not block deletion of the explicitly authorized derived timestream.** A hash is an identity check, not a replacement for unavailable inputs.

Distinguish two dispositions: **useful result, recover by reprocessing from documented upstream inputs**, and **retired experiment, saved restart state deliberately discarded**. Do not promise that every abandoned trajectory remains exactly replayable. No need to preserve instant restart at every historical iteration.

Keep original manifests and scientific conclusions unchanged as historical records. Add a concise purge record identifying intentionally absent products. Update current status and relevant existing indexes so they no longer advertise deleted RTC/PTC caches as available. Old full-payload verifiers may correctly report the removed paths missing; do not claim their original seals still validate the current layout. Reports and status readers must not silently regenerate deleted data.

## 6. Verification, disk budget, and deliverables

After deletion, verify target absence, inspect any failed/deferred operations, and confirm that the protected raw/tune inventories remain unchanged. Reuse inventory identities and existing checks where appropriate; do not require a fresh full-byte hash of every protected observation.

Produce one compact owner-facing report and machine-readable supporting inventories, not a proliferating set of large evidence bundles. Include:

- Before/after totals for `local_data`, the largest directories and product classes, deleted logical bytes, and measured free-space change.
- The observation/tune inventory, complete packages ready for testing, and exact incompleteness or association gaps.
- The executed deletion manifest/receipt, failures or active-use exceptions, and confirmation that Unity and non-target data were not modified.
- Where the durable inventory and recovery records live, which old processing routes now require rebuilding from raw, and which experiments are deliberately retired.
- A proposed single working-data root under `local_data` and an evidence-based total disk budget covering raw/tune inputs, caches, retained results, active jobs, and peak scratch usage.

The previously suggested resident obsnum list and numerical disk limits were proposals, not established facts or newly approved quotas. Use this inventory to propose an achievable bound; do not delete raw data to hit an assumed number. Identify known bulk-output paths outside the intended future root if encountered, but do not extend this sweep or deletion authority outside `local_data`.

Commit the reviewed compact records and necessary narrow status/index changes locally, using established practice. Report the commit and exact report location. No merge, push, new observation campaign, new bulk download, remote modification, production/default change, or broad directory reorganization.

**Stop after the inventory, authorized RTC/PTC purge, verification, and report. The next owner decision will be the single workspace root, hard disk budget, and which complete raw/tune packages to keep resident.**
