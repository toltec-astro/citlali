# Citlali refactor — current state

As of 2026-10-31. **The shared development baseline is `codex/refactor-mainline`.**
The owner approved incorporating pointing assessments through `6969dcf81` and
pointing handoff through `1f3bf6db7`, following the previously integrated
application `44171fd3a` and workflow repair `8078ba456`. The
[acceptance record](../validation/pointing_handoff_integration_2026-10-31/OWNER_ACCEPTANCE.json)
binds both full cutoffs, scope, preservation and required gates. It takes effect
as an integration receipt only after the exact independently reviewed candidate
is on local mainline; the external packet's `COMPLETION.json` records that move,
full source identity, Spack executable and passing checks.

The latest published/hosted-validated source at preparation is
`8078ba4567b3096b74ee378a99acfca270e27c3f` ([successful run](https://github.com/toltec-astro/citlali/actions/runs/41201517670)).
The new local integration needs owner publication and its own hosted run.
No hosted, Unity or observational qualification is inferred from local integration.

Use the [workspace map](WORKSPACE_MAP.md) to find the correct checkout and
protected work. The [approved pointing integration scope](POINTING_HANDOFF_INTEGRATION_2026-10-31.md)
explains compatibility limits and remaining release work. The
[integration ledger](INTEGRATION_LEDGER.md) retains exact acceptance/evidence routes.

## Start here

| What you need | Read |
| --- | --- |
| Find the current working folder or retained branch | [Workspace map](WORKSPACE_MAP.md) |
| Prepare the configured POINT application | [Python deployment and launcher](../requirements/README.md) |
| Build or test native Citlali | [Supported Spack workflow](../spack/README.md), then [this Mac's qualification](SPACK_QUALIFICATION_2026-10-30.md) |
| Find the code owner or execution route | [Architecture and source map](ARCHITECTURE.md) |
| Interpret scientific meaning or limitations | [Scientific conventions](SCIENTIFIC_CONVENTIONS.md), [contracts](scientific_contracts/README.md), [retained debt](RETAINED_DEBT.md) |
| Understand an integration or previous decision | [Integration ledger](INTEGRATION_LEDGER.md), [dated status](REFACTOR_STATUS.md), [application landmarks](APPLICATION_BASELINES.md) |

## One baseline, separate work in progress

| Role | Identity and purpose |
| --- | --- |
| Previous accepted application foundation | `44171fd3a6ba5924f8f4dba0324defe0922c597d`; retained pointing/engineering and Mac Spack qualification |
| New accepted input scope | Handoff `1f3bf6db744050102bc78e06917613e5dbe7fe3d` plus pointing evidence `6969dcf8168bcffbdfdbf7ebe56a22730545e086`; final combined source is bound by the integration receipt |
| Development mainline | `codex/refactor-mainline`; local combined handoff integration follows published `8078ba456` |
| Previous mainline | `e32c4584dec07617d6caa857d39bf13047b6c4b0`; retained in ancestry and recovery evidence |
| Pointing evidence included here | `6969dcf8168bcffbdfdbf7ebe56a22730545e086`; includes the earlier `797d013f` cutoff and later saved assessment, compatibility audit and proposed pilot plan |
| Engineering input included here | `5c46201057d6b6d1048aac2949a64e9a34e9bad3` |
| Protected pointing source workspaces | `codex/configurable-point-20261028` and `codex/point-handoff-20261030`; only the approved cutoffs enter this integration; any subsequent work stays separate |
| Frozen weighted trial source | `331ff29ea1720b1454e671cbbab93814d8df8321`; runtime, inputs and products stay preserved |
| Legacy 4.x/5.x and historical evidence | Protected separate lineages; integration grants no cleanup or replacement permission |

The existing working folder is reused for mainline:
`/Users/gwilson/.codex/worktrees/pointing-engineering-candidate-20261030/citlali-refactor`.
Its older folder name is a locator, not a separate integration authority. The
shared Git store is `/Users/gwilson/GitHub/citlali/.git`. The folder named
`/Users/gwilson/GitHub/citlali-refactor` is an older protected checkout.

The owner subsequently retired the local engineering and combined-candidate
branch names and the GitHub engineering branch. Their committed work remains in
mainline and recovery references/bundles. The engineering folder is retained at
its saved commit without an active branch. The integration adds one retained candidate name to the previous inventory:
11 local branches, 29 GitHub branches, nine working folders and three protected
legacy stashes. The workspace map records this dated inventory and its recovery packet;
it is not permission for further deletion.

## What is implemented and checked

Engineering cleanup unifies native source/strict-floating-point declarations,
builds the complete ordinary test aggregate, gives output/FITS/FFTW resources
explicit owners, extracts shared native input binding, and packages a curated
Python application. [The engineering record](ENGINEERING_CLEANUP_2026-10-29.md)
and [combined-candidate record](POINTING_ENGINEERING_CANDIDATE_2026-10-30.md)
retain exact implementation and reconciliation details.

The accepted application source passes the supported local Mac Spack developer
gate: all 44 dependencies installed, 1,388 enabled CTests, 207 baseline tests,
full configuration preflight, installed synthetic consumers, standalone headers,
and runtime dependency identity. One inherited lifecycle test remains disabled.
Independent review passed with recorded limitations. [Qualification evidence](SPACK_QUALIFICATION_2026-10-30.md)
states the exact LLVM20/SDK26.5/macOS27 realization and preserves initial failures.
Earlier AppleClang/Homebrew checks remain supplemental. Integration exposed a
Ninja dependency defect: a commit change could refresh the version header but
leave the binary stale for one build. The closure declares the header producer
and runs the existing regression with the actual build generator. This changes
build provenance, not scientific execution. Exact source and final closure
executable identities remain separately bound in their evidence packets.

## Current handoff behavior and qualification

The handoff preserves established correction signs and User/Paddle metadata,
accepts constant and bracketed pointing inputs, and publishes one discoverable
three-array pointing table only when all final fits and metadata qualify. It
reuses the final fits rather than fitting again. Missing metadata or a failed
array withholds the whole table while preserving valid maps. Publication/I/O
failures propagate to the caller.

Uncertainties remain unavailable (`NaN`), and the actual TolProj reader rejects
these tables for flux calibration. Nonzero already-applied POINT corrections
also withhold the table to prevent double application. Old constant-offset saved
CAL receipts remain compatible; old variable-offset receipts without a declared
timing convention require fresh owner preparation. This integration regenerates
no saved products and changes no frozen trial runtime.

Supported candidate qualification uses the existing isolated Mac Spack graph and
a separate `build/spack-pointing-handoff-20261031` directory. Its exact source,
executable/version, dependency identity, tests and review are bound in the external
integration packet named by the acceptance record. The old `build/spack-native`
executable remains preserved at its earlier qualified source. Native Spack and
synthetic consumer checks do not qualify a scientific observing campaign.

## Application and release boundaries

Configured POINT uses the explicit Python/TolProj launcher and uniform v1 or
weighted v2 profile. The ordinary native `citlali` development entry is a separate
explicit request interface; its omitted terminal still defaults to PTC. Neither
interface silently selects the other's scientific route.

The collaborator release target remains POINT and SCIENCE on a1100, a1400 and
a2000 through TolProj configuration, validation and guarded submission, with
TolTECA preparing and launching the selected runtime. Grant supplies beammap/APT
products. Beammap production and OOF remain outside that release target.

| Acceptance area | Current disposition |
| --- | --- |
| Local application integration | Owner-approved reviewed development baseline |
| Configured POINT engineering | Implemented and synthetically tested; scientific assessment remains in the pointing workstream |
| Pointing-table handoff | Correction compatibility and guarded astrometry publication included; flux uncertainty/eligibility and corrected-POINT table conversion remain held |
| Pointing-derived calibration consumed by SCIENCE | Flux policy, downstream connection and owner acceptance still required |
| All-array SCIENCE and genuine multi-observation coadds | Required release milestone; not established by POINT checks |
| Downstream products and consumers | Product/units/layout qualification still required |
| Build and publication | New combined source requires its recorded Mac Spack gates before local integration; new hosted CI follows owner publication. Published `8078ba456` passes hosted checks; Unity and release/export qualification remain separate |
| Production | No new authorization from integration or passing local tests |

General v2 defaults remain 1 arcsec / coverage 0.1 / M10. The frozen trials used
2 arcsec / 0.15 / M10; prospective M5 diagnostics require their own coherent
preparation. The retained optical-response minimum is tunable 0.10. Integration
changes none of these policies and launches no reduction.

## Evidence and ongoing maintenance

[Engineering](governance/ENGINEERING_GOVERNANCE.md),
[successor](governance/TIMESTREAM_SUCCESSOR_GOVERNANCE.md), and
[review](governance/REVIEW_AND_CONFORMANCE.md) governance retain their exact
accepted sources and digests in the integration ledger. This integration
incorporates the existing September 30 owner amendment without rewriting it.
Historical scientific router statuses and accepted-run ledgers retain their
original scope; a development integration does not relabel them as new science.

The [documentation guide](README.md) separates user guidance, architecture,
contracts and historical evidence. Keep concise summaries/manifests in Git and
retrievable bulk evidence in its recorded artifact location. Existing bundles,
recovery refs and sealed packets remain intact. The dated
[cleanup](REPOSITORY_CLEANUP_PROPOSAL_2026-10-29.md),
[working-folder consolidation](REPOSITORY_CONSOLIDATION_2026-10-30.md), and
[branch consolidation](BRANCH_CONSOLIDATION_2026-10-30.md) records describe their
own checkpoints, not live counts. Some dated ledger links retain temporary
report paths that are no longer present; current integration evidence is bound
to the sealed packets, and older claims require their named evidence to be
recovered before reuse. New cleanup still requires an exact approved batch with
recovery verification.
