# Citlali refactor — current state

As of 2026-10-30. **The shared development baseline is `codex/refactor-mainline`.**
Grant approved integration of application candidate
`44171fd3a6ba5924f8f4dba0324defe0922c597d`, followed only by documentation closure.
The [owner acceptance record](../validation/mainline_integration_2026-10-30/OWNER_ACCEPTANCE.json)
binds the exact source, earlier admission table, governance and checks. This
record becomes the completed integration receipt when its independently reviewed
closure is on the literal local mainline; the external `COMPLETION.json` records
that operation's exact old and new tips. Local integration does not publish GitHub.

## Start here

| What you need | Read |
| --- | --- |
| Prepare the configured POINT application | [Python deployment and launcher](../requirements/README.md) |
| Build or test native Citlali | [Supported Spack workflow](../spack/README.md), then [this Mac's qualification](SPACK_QUALIFICATION_2026-10-30.md) |
| Find the code owner or execution route | [Architecture and source map](ARCHITECTURE.md) |
| Interpret scientific meaning or limitations | [Scientific conventions](SCIENTIFIC_CONVENTIONS.md), [contracts](scientific_contracts/README.md), [retained debt](RETAINED_DEBT.md) |
| Understand an integration or previous decision | [Integration ledger](INTEGRATION_LEDGER.md), [dated status](REFACTOR_STATUS.md), [application landmarks](APPLICATION_BASELINES.md) |

## One baseline, separate work in progress

| Role | Identity and purpose |
| --- | --- |
| Accepted application source | `44171fd3a6ba5924f8f4dba0324defe0922c597d`; combined pointing cutoff plus engineering cleanup and Mac Spack qualification |
| Local integration branch | `codex/refactor-mainline`; contains that application source and its reviewed documentation closure |
| Previous mainline | `e32c4584dec07617d6caa857d39bf13047b6c4b0`; retained in ancestry and recovery evidence |
| Pointing input included here | `797d013f115e403a30663eb92254082e8e742826`; approved cutoff with saved five-case assessment |
| Engineering input included here | `5c46201057d6b6d1048aac2949a64e9a34e9bad3` |
| Ongoing pointing and handoff | `codex/configurable-point-20261028` and `codex/point-handoff-20261030`; later work stays separate until its own review/integration |
| Frozen weighted trial source | `331ff29ea1720b1454e671cbbab93814d8df8321`; runtime, inputs and products stay preserved |
| Legacy 4.x/5.x and historical evidence | Protected separate lineages; integration grants no cleanup or replacement permission |

The existing working folder is reused for mainline:
`/Users/gwilson/.codex/worktrees/pointing-engineering-candidate-20261030/citlali-refactor`.
Its older folder name is a locator, not a separate integration authority. The
shared Git store is `/Users/gwilson/GitHub/citlali/.git`. The folder named
`/Users/gwilson/GitHub/citlali-refactor` is an older protected checkout.

The engineering and combined-candidate branch names remain as retained evidence
of work now included in mainline. They are not additional application mainlines.
No branch, working folder, stash, bundle or backup is retired by this integration.

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
Earlier AppleClang/Homebrew checks remain supplemental. Exact source and final
closure executable identities remain separately bound in their evidence packets.

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
| Pointing-derived calibration consumed by SCIENCE | Compatibility and owner acceptance still required |
| All-array SCIENCE and genuine multi-observation coadds | Required release milestone; not established by POINT checks |
| Downstream products and consumers | Product/units/layout qualification still required |
| Build and publication | Local Mac developer gate passes; eligible hosted CI follows owner publication; Unity and immutable release/export qualification remain separate |
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
