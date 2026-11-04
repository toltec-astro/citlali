# Stable optical bootstrap seed identity

The defect was a data-dependent bootstrap seed built from the CAL receipt hash.
CAL receipts include execution timings and config-derived output provenance;
identical learning data could therefore receive different random draws. In the
owner's profile, network 3 detector occurrence row/association 1678 changed
retained to unresolved while its declared CAL payload digests and measured
response were unchanged. The lower interval bound crossed the existing 0.10
threshold. This repair intentionally changes seed binding, not that threshold.

## Ownership and encoding

`PointOpticalEvidence::learn` computes `seed_identity()` once after the existing
source-protected binning/centering step. The scientific owner fingerprints the
inputs it actually learns from; adapters do not hash selected YAML fields and
guess their scientific meaning. No new input cache or Engine state is added.

Policy: `point-optical-learning-input-seed-v1`. The SHA-256 preimage contains:

- length-prefixed policy, statistical method, units/domain and observation ID;
- absolute UNIX-second learning-grid origin, bin width, confidence level and
  number of bootstrap draws;
- ordered bin/block keys;
- network and array identity in canonical network order;
- exact detector occurrence identities in canonical detector order;
- centered time-bin values in mJy/nominal-beam, with explicit unavailable markers.

Integers use eight-byte little-endian encoding; finite binary64 values have an
explicit tag, canonical positive zero and explicit bit encoding. Unavailable
values use one fixed tag instead of a platform-dependent NaN payload. Arrays
and strings are length-framed. No native structs or memory-layout bytes enter
the preimage. The encoding is part of the named policy and cannot be changed
silently. Source-protected and otherwise excluded payload values do not enter
the binned learning data. Changes that leave the actual learning inputs exactly
identical need not change their fingerprint.

The Consider-only response threshold is excluded. The existing statistical
method, robust weights, source protection, peer construction, normalization,
resampling count, generator, quantiles and six interval checks are unchanged.
Shared-network and cross-array coupling seeds already use stable observation
and network identities and remain unchanged. Detector interval seeds combine
the new input fingerprint with the exact detector ID and panel/window scope.
The full cohort is included because peer references and normalization depend
on other detectors/networks.

## Provenance and publication

`cal_identity` and `source_mask_identity` remain exact receipt/producer bindings.
`PointOpticalPlan::for_network` still rejects foreign CAL, source masks or
detector order; the seed fingerprint is not a replacement replay identity.

Health schema `citlali-point-optical-health-v2` adds `bootstrap_seed_policy` and
`bootstrap_seed_identity`. Native restoration checks schema/policy, validates
the identity shape and checks each evaluated interval's seed against its
declared learning identity and detector/window/panel. The Python summary
validates and exposes the same fields. Older health schema v1 is rejected on
the new route rather than silently reinterpreted; historical products and
executables remain preserved. No stored output migration is performed.

## Validation prepared, not executed

New native cases cover metadata-only receipt/mask changes with exact equality
of seeds, intervals and decisions; relevant learning-data/support/domain/cohort
and request changes; invariance under cohort/detector ordering; and rejection
of absent, foreign or tampered restored seed identity. Existing source-mask
protection coverage now also changes the CAL receipt identity, proving that
an excluded source signal cannot alter the interval via a receipt-derived seed.
Python coverage checks the publication fields and rejects old/invalid bindings.

Grant's first command builds the candidate using the existing Mac Spack
realization, runs focused native/Python tests, full CTest, baseline-tool tests
and config preflight. No tests, probes, builds or reductions were run by Codex.
The initial owner-run gates can expose compiler or numerical-fixture failures;
none is silently assumed to pass. Independent static review binds the exact
candidate commit. Actual POINT behavior requires a subsequent explicitly
prepared owner-run M0--M5 reproducibility check, comparing new-seed runs with
one another and reporting changed historical borderline decisions.

Scientific/behavioral: candidate only, intentional seed realization change;
method and policy retained. Architecture: learning identity owned by the typed
optical evidence module, provenance/replay authority retained at the adapter.
Repository/evidence: isolated clean-base branch, prior evidence preserved;
no mainline integration, push, cleanup, Unity or runtime replacement.
