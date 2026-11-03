# Stage 3: native configuration setup and reporting

Grant accepted stage 2 and authorized proceeding with stage 3 on 2026-11-03.
The canonical acceptance receipt is `fba31ae909bbaf6358390937ec0845fe1716dcf3`,
following accepted application `3d537e978ac0a8e067c93af7804eadf4723f86ff`.
This is one Tier-2 spine increment on `codex/review-stage3-20261103`, in
`citlali-stage3-20261103/source` beside the preserved stage-1 and stage-2
evidence packets. The initial base and worktree were clean.

Grant alone executes all builds, tests, probes, benchmarks, reductions and
validation runners. Codex prepares changes, reviews static source and prepares
owner commands. No Unity access, push, deployment, runtime replacement or cleanup.
Stage-3 integration needs separate owner acceptance.

## Authority and scope

Engineering, Timestream Successor and review/conformance governance were read;
their accepted exact identities/digests are recorded in the canonical integration
ledger. CURRENT_STATE and REFACTOR_STATUS govern sequencing; ARCHITECTURE owns
the application adapter and one-way config flow; SCIENTIFIC_CONVENTIONS requires
requested/effective/observation-resolved/realized distinctions. APPLICATION_BASELINES
and RETAINED_DEBT retain separate software, scientific and production acceptance.
The WP-7.1 contract source `170ecea9de1ee810da7d7e45a489a4545ccd623d`,
closure `20ba6ae5dcf0b90a24ac3e778a75eff0a1bbe2aa` and later accepted scientific
owners remain unchanged. No scientific-contract package or new policy is created.

The concrete gap is in `execute_connected_ptc`: it read and rejected unsupported
cleaning configuration only after RTC and CAL. Move those same checks to setup.
The private `NativePtcConfiguration` snapshots YAML at the application boundary,
checks supported cleaning choices and resolves a typed `NativePtcSetup` using
verified APT array identity. Do not guess array from network number. An explicit
override still requires the existing single positive configured rank. Preserve
method mapping, scope restrictions, solver defaults, transfer/OOF iteration
limits and the experimental masked-covariance boundary.

For a single network, resolve PTC before raw ingress; for an observation,
resolve all selected networks before shared preparation and worker launch.
Workers recheck the previously bound network request digest before loading it,
rejecting changes since setup. Per-network PTC setup values outlive all workers.
PTC validates the prepared array against its CAL source and passes the prepared
solver request to its existing Learn/Consider/Apply owner.

The application owns a required `setup.yaml` per native network, including
source revision, terminal, destination, effective-config binding, configured
rank, explicit override, effective rank and solver controls when PTC is selected.
It is explicitly setup-only, never a completion record. Startup text also shows
requested/effective worker counts for observations. Final text reads existing
RTC exclusions/replacement influence, CAL availability, and PTC availability and
failed-fit counts from actual receipts. Top-level reporting retains nonzero exit
status and distinguishes exceptions from completed calls. Existing detailed
Learn/Consider/Apply receipts and reassessment reports remain authoritative.

## Reassessment and limits

- New persistent setup fields trigger Tier 2. They are application diagnostics;
  no scientific product values, units, validity rules or numerical calculations
  change. Existing PTC receipts gain additive configured-rank/override provenance.
- Setup failure order intentionally changes: unsupported PTC choices fail before
  output claims; array-dependent choices fail after APT binding but before
  processing. Claimed attempts remain preserved under the accepted stage-2 owner.
- Earlier APT verification for single-network PTC is needed to resolve rank.
  The verified relation is reused. No performance improvement is claimed.
- This is bounded native PTC preflight, not a claim that every legacy or
  experiment configuration check has moved to startup. Data-dependent validation
  stays with its scientific owner.
- Configured POINT's Python reservation, source feedback and publication,
  retained standard reduction/validation driver, saved MAP/FRUIT, comparison
  science, and RTC-only CAL preparation remain unchanged. Native service logs
  improve configured POINT visibility without changing its orchestration.
- Broad-source recovery, calibration policy, complete response/covariance,
  and scientific/production acceptance remain outside this repair.

## Gates and dispositions

Prepared focused cases: six C++ tests for array-specific rank resolution and
snapshot ownership, rejected cleaning choices, explicit-rank requirements,
special-profile scope/limits, setup-report distinctions and partial-result
reporting; three actual-CLI tests for early single-network/observation rejection
and unused PTC under RTC-only. Existing stage-2 CLI fixtures now include valid
PTC configuration to reach their intended output-ownership failure boundaries.

Owner gates: Mac Spack build/version identity; focused C++ and CLI tests;
full enabled CTest; baseline-tool tests; config preflight `--require-all`.
All are pending. No test, build, probe or validation runner was executed by Codex.
Full numerical reduction, Unity and configured POINT scientific qualification
are not inferred from these setup tests. Later representative runs may assess
the complete new reporting in real reductions; this packet does not claim them.

Scientific/behavioral: intended policy and numerical behavior preserved; earlier
configuration rejection and new diagnostics require owner-run evidence.
Architecture/ownership: application-owned setup snapshot and prepared typed
request; no Engine growth or science in reporting; writer static assessment only.
Repository/evidence: isolated base descendant; prior accepted branches, runtimes,
pointing work, stashes and backups preserved. Exact candidate and changed-file
digests will be in the external CANDIDATE.json. Independent fresh-context
exact-SHA review is pending, after owner evidence; the writer does not self-approve.

Stop for unrelated dirt, moving sources, scientific-policy ambiguity, new owners
or scope expansion. No new integration or production authority follows from
preparing this candidate.

## Owner-run reporting correction

At exact candidate `6d6a9f9afaf126375aed5162b2cca75093df2750`, Grant's attempt
`owner-run-20261103T204225.788859Z` passed configure/build/version and all
13 focused C++ cases (six stage-3 and seven stage-2). The actual-CLI gate
failed one of seven cases: `test_rtc_only_does_not_activate_or_validate_unused_ptc`
did not see the expected setup summary. CTest, baseline and config gates were
not started. The failed log and original candidate binding remain preserved.

Static diagnosis: `configure_run_loggers` deliberately sets the default
console sink to critical-only and provides a separate `citlali_logger` for
application messages. The new default-logger info calls were suppressed,
affecting both setup and final summaries. Repair within scope: use one small
reporting helper to reuse the existing named logger, with ordinary default
logging only when no application logger exists. Do not change logger lifetime,
sinks or verbosity. All three call sites use the helper. The existing actual-CLI
visibility assertion is retained without weakening; no new scientific policy
or numerical behavior is introduced. All gates for the corrected source and
independent exact-SHA review remain pending.
