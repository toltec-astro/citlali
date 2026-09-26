# Independent input-selection preflight — 2026-09-26

Reviewed base: `43844bfe07b81c3f70bd01f370f46820982f5bba`, owned checkout `/private/tmp/citlali-point-additive-20260923`.
Read owner independent-observation directive, AGENTS.md, ENGINEERING_GOVERNANCE.md, TIMESTREAM_SUCCESSOR_GOVERNANCE.md, REVIEW_AND_CONFORMANCE.md, current status, Toltec skill and authority maps, completed local-data OBSERVATIONS.json.gz, accepted `prepare_point_forward.py` and canonical APT bindings. Read-only inspection only: no new reductions/fits, no Unity, no source or input modification.

## Verdict

**PASS with stated readiness qualifications for selecting POINT 152430 / 0 / 2, a2000 networks 11 and 12.** Selection is based on dependencies and existing metadata/summary; no new-policy result was examined. Freeze selection prospectively, then retain it even if the expected-source guard rejects the new reduction. Execution still requires the parent task's scientific-method/code/environment/budget preflight and native bundle verification; this review is not a blanket execution approval.

Initially considered 148669 (January13,3c273) for session diversity. It has complete legacy inputs and an independently bright source, but lacks the canonical APT-v2 source-bound product required by the accepted successor ingress. No established lossless adapter was found in the bounded review. Thus legacy inventory readiness does not imply successor readiness. Do not create or recalibrate its APT in this work. 152391 is also viable, but equally ready152430 offers a later acquisition and its own tune realization. This is preselection dependency resolution, not replacing a failed source check.

152430 is 1146+399, February19 2026 10:08UTC, an independent acquisition later than152389 in the same observing session. Own tune identity152430/0/1 (network-specific headers, not adjacency inference). The same source/session/calibration lineage does not establish statistically independent calibration errors.

## Bright-source expectation, existing independent evidence

`/Users/gwilson/work_toltec/local_data/citlali-validation/v2/science/pointings/reduced/redu00/152430/raw/ppt_commissioning_pointing_152430_citlali.ecsv` contains a2000 amplitude484.7236mJy/beam, amplitude error2.879235, peak/full-map-RMS23.36939 and fitSNR168.3515. These support bright-source expectation only, not successor flux truth, new error bars, or imported guard operands.

DCS header explicitly says Pointing and saved input/configuration is a pointing route. TelescopeBackend.ObsGoal contains stale `Oof`; retain this discrepancy in input provenance rather than letting that field silently redefine the mode. Source epoch2000.0, header RA3.084745672694985rad,Dec0.6977162348655682rad. Original telescope RA2000/Dec2000 streams agree with those values.

## Exact local dependencies

All original raw/tune/telescope paths below have prefix `/Users/gwilson/work_toltec/local_data/2025-C1-COM-01/data/`.

| Input | Filename | Bytes | SHA256 |
|---|---|---:|---|
| nw11raw | toltec11_152430_000_0002_2026_02_19_10_07_51.nc | 30825080 | 5536f00a12395909e9e1495e308fe04892411a5b44d73a97bac9d495c3bacde6 |
| nw12raw | toltec12_152430_000_0002_2026_02_19_10_07_51.nc | 31136056 | 1289ac9fd40fdc1579387b0ac718099564f7a3da9938928989c05d78a60bfc41 |
| nw11KMP | toltec11_152430_000_0001_2026_02_19_10_06_46_tune.txt | 99121 | 22022c5127646dbff33a12f3581748c24f90df0066759127a42a1b60e6877652 |
| nw12KMP | toltec12_152430_000_0001_2026_02_19_10_06_46_tune.txt | 102189 | 994f7597d66a4186fef9a25d3c206f826e4749ee90beb4f9b4d7ec5a90392bf8 |
| original telescope | tel_toltec_2026-02-19_152430_00_0002.nc | 1320968 | f6ac27248692af558a16af94f267a84b344e93f30d2248516d4388b9b25e5f19 |

Original HWPR `hwpr_152430_000_0002_2026_02_19_10_07_51.nc` is also present. The accepted successor POINT preparer consumes original telescope rather than recomputed telescope. It uses KMP text; associated processed and raw tune NetCDFs are present, but are not reasons to fetch or process a redundant tune history. Caller should bind only consumed products and record any auxiliary retained-but-not-consumed inputs explicitly.

Canonical manifest:
`/Users/gwilson/work_toltec/local_data/citlali-validation/v2/science/apts/v2/apt_152430_matched.apt-v2/manifest.ecsv`
SHA256 `dde3e24c5187ecd517c6006ed5ea3ee94aebbb6d60def7881f36774bc09f4acd`.
All10manifest role files exist and their actual transport SHA256s match. sources rows raw/KMP for nw11/12 match the four actual hashes/bytes above exactly. APT nw11 contains486 sequential channel rows, nw12 contains491; both are array2. Native `apt::verify_bundle_filesystem` remains required; matching file transport alone does not certify every semantic contract.

Saved input/configuration (one exact meta.name152430_0_2):
`/Users/gwilson/work_toltec/local_data/citlali-validation/v2/science/pointings/reduced/citlali_o152389_0_2_c8.yaml`
SHA256 `acf93c30b130abb00477162673fd6bded25e3645579fd7c669907c8cd6965893`.
It lists all11acquired networks0–5,7–9,11–12; a2000 is11/12. The basename152389 is a multi-observation config, not proof of a different input. Recorded remote paths have prefix `/work/toltec/citlali-validation/v2/science/`; those are historical provenance, not live Unity verification. No missing required input was found; proposed download bytes0.

## Acquisition/time coverage

nw11 Is/Qs7821×486; nw12 Is/Qs7820×491. Both sample frequency122.0703125Hz. RecvTime monotonic and finite; Ts sample counter column3 increments exactly1 throughout both. SampleType is0 throughout. Times in Unix seconds:
- nw11:1771495682.2328935 through1771495746.294367; adjacentRecvTime step0.0078051090–0.0086154938sec.
- nw12:1771495682.2379448 through1771495746.291233; adjacentRecvTime step0.0077800751–0.0086088181sec.
- original telescope3251records; BackendTime1771495681.9337304 through1771495746.9221332. Relevant pointing/time streams read finite and cover both detector ranges. AcuTime1771495681.9300342 through1771495746.9188173.

The complete saved11-network timing population was read; allRecvTime axes monotonic and allTs sample-counter increments1. Networks0,1,2,3,4,5,7,8,9,11,12 have native row counts7821,7820,7821,7821,7821,7821,7821,7820,7821,7821,7820 and channels630,494,510,533,532,474,420,460,488,486,491. Across networks firsttimes1771495682.2295074–1771495682.2379448; lasttimes1771495746.2855926–1771495746.2947364. Allcovered by original telescope. Other networks need not be reduced for this a2000-only pass; retain any timing-only population required by established processing segmentation.

This is complete against saved acquisition/reduction inventory; not a live remote archive attestation. No counter holes or endpoint loss found. Payload science quality still belongs to native ingress/VAL and may cause a processing stop; this review does not guarantee every detector usable. Missing6/10 are not candidate a2000 omissions.

## Generalization risks and required safeguards

- Preserve current RTC/CAL/AST/PTC scientific recipe, but bind this observation's inputs, segmentation and source protection. No152389failed-segment exceptions may carry over.
- Initialize new M0 noise/covariance, core exclusion, support and geometry exactly once, rather than importing152389numeric operands. Full-precision score thresholds alone transfer. New false-admission probability remains uncalibrated.
- Retained older fit supports selection only; BASE REF T10 and M0 own all authorized experiment shapes/measurement geometry.
- KMP/raw/APT identities and native sample validity must remain paired through injection; other-array raw timing need not imply other-array science processing.
- No final-ready claim until prospective runtime/disk ceiling and immutable shared parent lifecycle are reviewed. No152389timestream regeneration is needed.

Scientific/input disposition: pass with listed limitations. Architecture/input-boundary disposition: pass for reusing canonicalv2 and existing consumers; no legacy adapter added. Repository/evidence disposition: read-only, no inputs changed, review written outside local_data for incorporation into permanent task evidence.
