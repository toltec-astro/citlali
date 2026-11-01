# Older scientific workspace: inventory and proposed cleanup

Prepared on 2026-11-01. **Batch C awaits owner approval. Nothing in the older
scientific workspace has been deleted, moved, hidden or edited by this work.**
The five refactor folders and two protected legacy/performance folders remain.

## Which workspace this concerns

`/Users/gwilson/GitHub/citlali-refactor` is a retained historical scientific
checkout on `codex/sci-align-001-lissajous-timestream-fit`, at
`dd4a3599279a0d09eeccaf3c63a4ac2782198d57`. It is not the accepted development
mainline. Use the [workspace map](WORKSPACE_MAP.md) to find mainline and ongoing
pointing work. This folder has 78 commits outside mainline; its tracked source,
branch and ignored build dependencies remain protected. Cleanup is not a merge
or scientific disposition of that work.

## Where its 465 changed/untracked paths belong

| Material | Paths | Disposition |
| --- | ---: | --- |
| Local `AGENTS.md` edit | 1 | Keep in place; owner-local command guidance, not application code |
| Original Git recovery bundles | 41 | Keep every original and backup |
| Two WP7 audit-comparison files | 2 | Keep historical audit evidence in place |
| Event-centroid/pointing analysis, results and associated renders/caches | 183 | Keep in place; not part of this cleanup |
| ALIGN/AST draft documents and document verification output | 213 | Proposed batch C; preserve historical copies in verified archives |
| ALIGN r0.2 document page renders | 25 | Proposed batch C; preserve historical copies in verified archives |
| **Total** | **465** | **238 proposed removals; 227 paths would remain** |

The count is a Git inventory, not a count of unfinished coding tasks. The only
modified tracked file among these paths is `AGENTS.md`; the other 464 files are
untracked. Existing tracked science and unmerged branch history require their
own review. The ignored dependency sources are outside this 465-path count.

## Accepted specifications and historical drafts

Use the canonical [contract index](scientific_contracts/README.md) and the
[SCI-ALIGN / SCI-AST r0.3 owner freeze record](scientific_contracts/audits/ALIGN_AST_HORIZONTAL_R0.3/SCI-ALIGN_AST_STAGE_B_R0.3_FREEZE_RECORD.md).
That record identifies the exact accepted package manifests, sources, PDFs and
shared boundary. The proposed output files include earlier r0.2 drafts and r0.3
rendering/check products. Their archive classification does not revise scientific
content or assert implementation/observational acceptance. Some files match
canonical bytes; that fact alone is not the reason for scientific authority.

## Batch C: exact proposal for approval

Remove only the **238 enumerated untracked regular files**, totaling
**63,680,360 bytes (about 64 MB)**, from the older workspace:

| Location relative to that workspace | Files |
| --- | ---: |
| `output/pdf/` — historical ALIGN/AST draft PDFs | 8 |
| `output/scientific_contracts/SCI-ALIGN/` — draft sources and document checks | 54 |
| `output/scientific_contracts/SCI-AST/` — draft sources and document checks | 150 |
| `output/scientific_contracts/SCI-ALIGN_AST_HORIZONTAL_AUDIT_STAGE_B_R0.2.md` | 1 |
| `tmp/pdfs/align-r02-final/` — page renders | 25 |

These are summaries of an explicit file list, **not recursive directory-deletion
instructions**. No directory removal is proposed. Any extra or changed file
requires reassessment. The benefit is removing over half the distracting status
entries; disk-space recovery is modest and the archives remain on this Mac.

The exact manifest is `batch-C.json` in the preparation packet below. Its SHA-256:

`f2e8ffb33ad0441cf14a3e2bf19110d0cb41347687e4f4e833bc81beab0b5f14`

Approval would cover this manifest only. Before execution, recheck its digest,
source identity, each file's bytes/mode/untracked status, all preservation targets,
and live process/chat/dependency use. Independently review the bounded execution
script. Stop on drift or a live dependency; do not use `git clean`, force removal,
reset, pruning or broad recursive deletion. Preserve all branches, stashes,
seven workspaces, pointing work, Citlali 4.x/5.x and every backup.

## Recovery and continuity

Preparation packet on this Mac:

`/Users/gwilson/.codex/visualizations/2026/10/29/01a188f2-4d30-7e60-88a8-956b50bb08cf/citlali-science-documentation-preparation-20261101`

Start with `batch-C-paths.txt`, `batch-C.json`, `restore-verification.json` and
`RESTORE.txt`. `batch-C-document-history.tar.gz` contains every proposed file at
its original relative path. Archive SHA-256:

`5df108160241fb67d6e6a6b73febfd1789d9ba54cad5a58f97eb9b743fd3d2f5`

Every listed file was restored into a new scratch folder and checked for exact
bytes, permissions and relative path. Each also matches the earlier complete
science archive and organized catalog in `citlali-workspace-consolidation-20261031`
beside this packet. That earlier independent review and all 535 bound artifacts
were authenticated again. No prior backup was rewritten.

To recover after a later approved removal: authenticate the manifest/archive,
extract into a new empty scratch folder, verify all file digests and modes, then
copy only the needed listed paths back when absent. Stop on an existing different
file. Never restore an old `.git` pointer. OS-managed extended attributes,
ownership and timestamps are not restoration guarantees; their original metadata
is recorded in the earlier physical manifest. These are local recovery copies,
not an off-machine disaster-recovery claim or runtime qualification.

Historical status inventories and document workflows can still name these paths.
Restore the needed files before reproducing an old document check. No chat is
closed, archived or redirected by this proposal. The retained scientific folder
continues to support its older chats and dependency sources. The separately parked
contract chat retains the restoration requirement in the [A+B retirement record](WORKSPACE_RETIREMENT_2026-10-31.md).

## Preparation scope and checks

Canonical base: `8e8a0581d1cfe6b1edffe5f63aac7efbdd69cee9`. The owner requested
documentation clarity and safe preparation before any deletion. Engineering,
review/conformance and successor governance were read against the accepted
digests in the [integration ledger](INTEGRATION_LEDGER.md). This is Tier 2 evidence
preparation with no application implementation branch, new worktree or scientific
authority change. The documentation edits clarify current entry points, completed
integration status and dated handoffs; historical records remain intact.

The packet records source preservation, backup authentication, restoration,
bounded reference/chat checks, documentation checks and independent exact-source
review. Reference searches do not prove absence of every possible external
consumer; fresh checks remain mandatory before removal. Native, Spack, Unity and
observational tests are not triggered by these documentation and archive-copy
operations. No hosted test result, production acceptance or permission to delete
is inferred. The final preparation receipt binds the documentation commit and
review; publication remains owner-controlled.
