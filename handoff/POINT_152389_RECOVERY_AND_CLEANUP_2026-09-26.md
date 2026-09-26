# POINT 152389 recovery, cleanup, and next steps — 2026-09-26

Owner request: recover project state after the interrupted session, remove old
generated products that are no longer useful, and plan the next work. This is a
maintenance/recovery lane. No scientific continuation was launched during it.

## Recovered checkpoint

The scientific worktree is `/private/tmp/citlali-point-additive-20260923`, branch
`codex/map-successor-first-forward-001`, starting at
`11b4d2c07a81cc5876bd3da833c4168731e34aea`. The main checkout is on a different
branch and contains unrelated changes; it is not the scientific working tree.

The last completed task is the independently reviewed and sealed a2000
saved-product diagnosis dated September 25. The preceding guarded comparison
completed 45 trajectories and 450 feedback applications across three arrays,
three policies and five branches. The current experimental policy is not a
production default. The uncertainty calculations remain qualified conditional
diagnostics, including the unresolved source-location transfer question.

The completed diagnosis found that CORR improves shoulder response, but retains
a2000 seed dependence and compact-response distortion. At T10 the seed branch
changes fitted peak by -0.103601 mJy/nominal beam (-0.0127448% of its REF peak)
and the large aperture by +0.203416% of its REF aperture. Separately, its
compact-plus large-aperture discrepancy is +9.284% of the ideal injected compact
increment. These are different comparisons. The diagnosis did not establish a
common cause or new uncertainty/significance.

Eight checkpoint bindings were freshly authenticated: the two campaigns' root
ledgers, reviews and assessments, the diagnosis detached seal, and the exact
native binary required for continuation. All matched. This is a bounded identity
check, not a rehash of every payload in the large campaigns.

## Where the interrupted task stopped

The authorized three-pixel m1 experiment has not run. Inspection found no CUT3
output root and no live Citlali/Python scientific process. The worktree contains
only a 20-line optional operand-logging addition in `point_corr_admission.py`
and the untracked directive/preflight directory
`validation/point_152389_a2000_m1_cut3_2026-09-25`.

The continuation command, focused tests, final prospective bindings and
independent pre-execution review are unfinished. The draft preflight is not a
completed review or execution receipt. The recorded task state is zero new
science applications out of the ceiling of 18. The unfinished patch and both
draft documents have been copied into this maintenance evidence directory so
another interrupted session cannot obscure them. Their preservation is not
approval of the unfinished implementation.

The current checks establish the project state; they do not establish the root
cause of the connectivity problem.

## Disk inventory and retention

At inspection the data volume had about 172 GiB free. The validation area uses
about 398 GiB, including about 300 GiB of development runs. The guarded comparison
uses about 117 GiB and response/robustness C+D about 61.5 GiB. Both are retained:
they contain inputs and evidence for the pending experiment. The frozen guard,
correlation inputs, original maps, network rescue, first/ten-cycle FRUIT,
additive uncertainty and post-FRUIT covariance parents also remain retained.
Older dates alone do not establish that a product is expendable.

The first cleanup proposal removes redundant copies only, with individual
source-to-retained SHA256 bindings. It retains all scientific content. The
exact targets, limitations, verification and approval state are in
`validation/point_152389_recovery_cleanup_2026-09-26/CLEANUP_PLAN.md`.

Additional inventory targets, not approved deletion targets:

- Approximately 3.99 GiB of generated comparison fingerprint JSON expansions
  in older performance runs. Check regeneration and record which historical
  seals would become incomplete before considering deletion.
- Approximately 32.6 GiB in `local_data/fruit-development`: older FRUIT work
  needs a separate product/dependency disposition; none is marked expendable
  solely because the current experiment supersedes its method.
- Approximately 104.9 GiB in `local_data/2026-refactor`: contains baseline
  reductions and input/project assets. It is not a blanket cleanup target.
- Build trees, Git worktrees and branches are retained. Ref/worktree/status
  inventories were saved, but unique-commit and full preservation reviews for
  deleting them have not been performed. The many already-missing `prunable`
  worktree registrations are not large disk payloads.

Actual recovered space must be measured after approved deletion; APFS shared
storage and snapshots can make it smaller than the logical/allocated totals.

## Next work, in order

1. Approve and execute the exact duplicate-only cleanup batch after the bounded
   independent review. Recheck hashes immediately before removal, record
   deleted paths and retained identities, and measure free space afterward.
   Do not rewrite old scientific seals to conceal retirement of copied files.
2. Finish the existing a2000 three-pixel continuation implementation and tests.
   Bind the original m1, immutable input, frozen guard/thresholds, exact binary,
   attempt ceilings and inherited replay tolerances; obtain the required
   independent exact-source review before any scientific application.
3. Run the unchanged m1 control through T2–T10 first (nine applications). If its
   replay fails, stop; do not start the intervention or retry a failed science
   application automatically.
4. After successful replay, zero only the three authorized m1 feedback members
   790, 882 and 1003, preserving data validity and all other model values. Run
   T2–T10 with normal admission from m2 onward (nine more applications). Cache
   the one required fit per map. No persistent veto, new injection or T11.
5. Compare intervention, replay and original REF with the required accounting,
   pixel re-entry and practical source measurements. Review, seal, commit
   locally and stop. Decide the subsequent scientific test from those results;
   do not assume that removing seed dependence repairs compact response.

No new full RTC/PTC ensemble, parameter tuning, JINC campaign, FRUIT production
adoption, Unity access, merge or push is part of this recovery operation.

## Evidence and authority

Evidence lives in `validation/point_152389_recovery_cleanup_2026-09-26`:
checkpoint authentication, allocated-size inventories, direct dependency-root
counts, worktree/ref/status inventories, preserved unfinished work, exact
duplicate ledgers, independent review and the cleanup plan. Dependency counts
are occurrences in named records, not a complete transitive dependency proof.

The owner has requested cleanup generally. The exact deletion batch remains
pending approval under `doc/governance/ENGINEERING_GOVERNANCE.md:173`:
“Deletion is allowed only after preservation and reachability are proved and
the owner approves the exact operation.” This does not revoke the current
cleanup request; it identifies the precise final approval still needed for the
listed paths. No file has been deleted during this inventory.
