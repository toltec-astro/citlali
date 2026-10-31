# Proposed next pointing and handoff integration

2026-10-31. **Proposal for owner approval; no application merge has occurred.**
This follows the completed integration of candidate `44171fd3a` and successful
publication of mainline `8078ba456`. Current documentation updates are local
maintenance, not acceptance of the source cutoffs proposed below.

## Decision in plain English

Bring the latest completed pointing assessments and the already implemented
pointing-table handoff into one mainline candidate. This improves the connection
between pointing measurements and the existing correction reader. It does not
make the tables usable for flux calibration or finish the SCIENCE/coadd pipeline.

Approve the exact cutoffs and bounded scope below, preparation of one isolated
candidate, and local mainline integration only after its required checks and fresh
independent review pass. No push, reduction, source-branch deletion, working-folder
removal, new scientific policy, or production activation is included.

## Exact inputs

| Role | Exact source | Disposition |
| --- | --- | --- |
| Published and hosted-validated mainline | `8078ba4567b3096b74ee378a99acfca270e27c3f` | Application/build/workflow base; preserve its full-history checkout and version-header fix |
| Local documentation closure | Commit containing this proposal and refreshed workspace map | Use the current clean mainline descendant when preparing the candidate; bind its full SHA in the execution record |
| Pointing assessment cutoff | `6969dcf8168bcffbdfdbf7ebe56a22730545e086` on `codex/configurable-point-20261028` | Seven newer commits; documentation, saved-evidence analysis and planning only |
| Handoff cutoff | `1f3bf6db744050102bc78e06917613e5dbe7fe3d` on `codex/point-handoff-20261030` | Two newer commits; implementation plus reviewed closure |
| Handoff implementation | `8fa064d1169cb7d37ccfea4a046c66a78fe675e1` | Previously reviewed source; closure does not change its application files |
| Handoff original base | `b090da0458271c618568802fde1cf5c39f5dd5d2` | Historical combined-candidate base, already an ancestor of current mainline |

These are frozen import cutoffs, not a freeze on the original pointing workstream.
Any later source-branch commits stay separate. Preserve the source branches,
original working folders, frozen trial executable `331ff29ea`, inputs, products,
recovery artifacts and all Citlali 4.x/5.x/legacy references.

## Acceptance table

| Proposed item | Value | Acceptance boundary |
| --- | --- | --- |
| Saved shoulder-feedback assessment and review (`a3e7bc3b5`, `50c6321ba`) | Keeps the interpretation of already saved measurements with the source history | Evidence only; no new observations, reruns, refits, threshold or feedback changes |
| Connected-pilot priority, plan and review (`02ff39ffe`, `323f1fba1`, `7f960a377`) | Gives POINT–SCIENCE–POINT and eventual coadds a concrete next-work plan | Import as a dated proposal. Its old base/build status is historical; product-scope, source-policy and calibration decisions remain separate |
| OG compatibility assessment and review (`28061144d`, `6969dcf81`) | Records established correction signs, units and time support | Prior-work evidence for the handoff; no new coordinate model or scientific acceptance |
| CAL correction ingress in the handoff | Accepts single-date constant, undated observation-span and dated-bracket corrections | Preserve OG whole-Unix-second MJD conversion and existing AST equations; no extra sign or cosine; invalid/collapsed support fails |
| Saved CAL/MAP correction support | Preserves the selected timing interpretation across saved products | Old constant-offset receipts remain compatible. Old variable-offset receipts without an explicit timing convention are rejected, not silently reinterpreted; any fresh CAL preparation remains owner-run |
| Guarded all-array pointing table | Publishes qualified final measurements through the existing TolTECA pointing interface | All three final fits and required metadata must qualify; signed arcsec displacement and existing mJy/nominal-beam amplitude are retained; no second fit |
| Unavailable and corrected-POINT cases | Avoids misleading partial tables or double-applied corrections | Errors remain unavailable (`NaN`), flux eligibility remains false, and nonzero already-applied POINT corrections withhold the table; valid maps remain available |
| Package/runtime inventory and tests | Makes the new module available in the installed application | Verify packaged imports and CLI staging; propagate I/O failures; run the new table tests |
| Hosted test-list completion | Covers the new publication safeguards on GitHub | Add `tools/timestream_successor/test_configured_point_table.py` to the explicit application-test list; preserve `fetch-depth: 0`, existing tests and matrix |

No uncertainty estimator, calibration threshold, optical-response cut, FRUIT
policy, weighting, grid setting, native sample scheduling, physical coordinate
formula or SCIENCE/coadd algorithm change is authorized by this proposal.

## Review findings and merge treatment

Read-only comparisons used the exact tips above. Pairwise merge simulation in a
temporary object directory found only `doc/REFACTOR_STATUS.md` conflicts for the
pointing cutoff, and `doc/CURRENT_STATE.md` plus `doc/REFACTOR_STATUS.md` conflicts
for the handoff cutoff. Application paths changed by the handoff do not overlap
mainline changes since its original combined base. This is pairwise inspection,
not a passing combined-candidate merge or test result.

Keep the refreshed mainline front page as current authority. Preserve incoming
dated assessment/implementation history, but do not restore obsolete claims that
Spack is pending or old mainline/candidate branches are current. Route the incoming
connected-pilot plan to the current candidate for any later implementation;
retain its original source identities and distinguish completed handoff work from
its remaining proposed batches. Never mark all of that plan accepted by importing
its document.

Independent scope review found no scientific or architectural source blocker.
It identified the missing GitHub table-test entry as an integration-readiness gap;
the existing mainline workflow cannot currently exercise that new test module.
The bounded test-list addition above must accompany the import. The reviewer
reproduced 164 focused config/input/table/CLI tests and actual pinned consumer
checks, including rejection of unavailable flux uncertainty. Prior handoff native
qualification used AppleClang/Homebrew and remains supplemental.

## Required candidate checks

After owner cutoff/scope approval:

1. Record the current clean mainline full SHA and protected-state snapshot. Prepare
   one candidate branch, `codex/pointing-handoff-integration-20261031`, reusing the
   existing mainline working folder after preserving its accepted runtime bindings.
   Keep the mainline reference unchanged during assembly. Do not repurpose or
   alter either original pointing checkout or add another permanent working folder.
2. Combine the two exact cutoffs, resolve current-state/status documentation as
   described above, and add the missing hosted table-test entry. Record the final
   changed-path inventory against the approved base; do not import later commits.
3. Use the already qualified isolated Mac Spack dependencies and a separate build
   directory. Run focused correction-ingress/saved-receipt tests, all enabled CTests,
   baseline-tool tests and full configuration preflight. Bind source SHA, version,
   lock/DAG and executable hash; the binary must report the candidate revision.
4. Run the complete configured POINT Python suite including table tests, static
   checks, packaging/installed-runtime checks and synthetic consumer roundtrips.
   The fixtures bind read-only operational TolTECA
   `2791e6a1e6349ad1d3ac549a648f41cbc51b98c7` and TolProj
   `6309a5a852a0412cee867992aff01341a56b0ebf`; verify current consumer authority
   before claiming current-environment compatibility. No sibling edits are included.
5. Obtain fresh independent review of the exact combined source and all three
   conformance axes. Prior review of `8fa064d11` is useful input, not approval of
   the combined candidate. Any unresolved new scientific or ownership question
   stops that portion for owner disposition.
6. If the owner approves the full operation proposed here and all gates pass,
   advance local mainline by guarded fast-forward, verify preservation and record
   acceptance. Keep source branches and recovery evidence. Publication remains
   owner-run; new hosted CI follows that publication. Branch/folder retirement is
   a separate later batch, including disposition of the temporary candidate name.

Native tests are development qualification, not observation/scientific acceptance.
Unity and the exact connected observational pilot remain separate owner-run work.
No observed maps are regenerated by these checks.

## What remains after this integration

- Flux uncertainty and calibration eligibility need their own scientific decision
  and qualification; conditional fit uncertainty is not a substitute.
- Corrected POINT runs need a reviewed residual-to-measurement convention before
  their tables can be published for the established correction reader.
- SCIENCE input/configuration, celestial mapping, calibration propagation and real
  multi-observation coadds remain separate implementation/qualification work.
- The proposed reduced base-map pilot product set and SCIENCE source/training
  policy need explicit disposition before execution; importing its plan grants
  neither policy nor observing authority.
- Actual supported-environment and owner-run POINT–SCIENCE–POINT acceptance remain
  required before release or production claims.

## Evidence and current disposition

Engineering, successor, and review/conformance governance were read and retain
their accepted exact sources in `INTEGRATION_LEDGER.md`. This work edits only
navigation/status/proposal documents; it changes no scientific authority.

The external packet contains initial Git/worktree/stash snapshots, live GitHub
checks, exact unique-commit/path lists, pairwise merge-inspection output and the
independent scope review:

`/Users/gwilson/.codex/visualizations/2026/10/29/01a188f2-4d30-7e60-88a8-956b50bb08cf/citlali-workspace-pointing-review-20261031`

Current disposition: **documentation refreshed; integration scope reviewed;
owner cutoff/scope approval pending; application source unchanged.**
