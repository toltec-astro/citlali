# Event-support successor validation 06

Date: 2026-08-12

## Trigger

The first successful broad-support fit-gate renderer averaged detectors at a
common timestamp. Its supplementary source-aligned renderer fixed that error
but still treated one contiguous 35-arcsec score-mask segment as one visual
crossing. ObsNum 150818 scan row 0, UID 1051 demonstrates the remaining
failure: that detector makes two geometrically distinct passages through the
source inside one broad segment. Combining the passages hides both the source
spike and the actual model failure.

This successor was selected from the authenticated 2026-08-11 crossing
analysis before inspecting a new real successor fit. It does not change raw
inputs, PTC preparation, telescope reconstruction, beam geometry, baseline,
source model, competing models, optimizer, bounds, or parameter conventions.

## Frozen event contract

- Define support at `tau=0` around the retained a1100 PPT center.
- A geometric event is one contiguous passage through the 0.5 elliptical-FWHM
  contour.
- Reject a passage whose half-power block touches a retained scan edge.
- Center a fixed +/-1.5 local elliptical-FWHM window on each event minimum.
- Intersect windows with the frozen score and validity masks.
- Union overlapping windows only within one detector-scan numerical objective
  mask, then require at least eight scored samples per detector-scan.
- Preserve every geometric event identity for the event catalog and detail
  PDF. A fitted timing, hysteresis, signal amplitude, S/N, or residual may not
  select event membership.

The frozen protocol is `crossing_support_protocol.json`. It binds the prior
22-observation crossing protocol SHA-256
`012fce31e106790fec3ab04baa1deff5c544219981b08a827b2899495267df75`
and its manifest digest
`3b383deaac7e3dd44359c5573369f57cebe40d182ce690eb6ced4ca23fc8e3a1`.
It also binds the immutable arithmetic core of the current timestream protocol
while permitting only campaign-specific scope, input-authority, corpus, and
campaign identity fields to change.

## Synthetic and regression validation

Six focused crossing-support tests pass. They establish half-open contiguous
event blocks, preserve two separate passages when their fit windows overlap,
prove event identity is invariant to detector signal, retain edge passages as
rejected evidence, and force both passages of the first multi-event
detector-scan into the review PDF. The complete related regression set passes:
39 tests covering timing/hysteresis recovery, wrap-safe interpolation,
checkpoint identity and restart, source-aligned rendering, and event support.

## ObsNum 150818 pilot

The independent pre-fit event census exactly reproduces the earlier
tau-zero/PPT-centered corpus result:

- 962 geometric events;
- 907 complete events and 55 edge-touching rejected events;
- 730 retained detector-scan groups;
- 546 retained detectors across all 12 scans;
- 24,298 scored sample values.

All four event-support fits completed in 37.2 seconds locally. Point results:

| Model | Objective | tau (ms) | h_az (arcsec) | h_el (arcsec) |
|---|---:|---:|---:|---:|
| constant | 600233.319311 | 0 | - | - |
| lag | 596372.486467 | +4.106037 | - | - |
| two-axis hysteresis | 597658.139303 | 0 | -0.228217 | -0.027569 |
| joint | 595866.168704 | +4.454494 | -0.076147 | +0.094734 |

The lag result agrees with the earlier direct tau-zero/PPT event-support fit
to numerical precision. That agreement is a validation result, not an event
selection or acceptance criterion.

Local ephemeral result identities:

- fit gate JSON: `f19bdb41bdda950f844f2d9148b8f39c0b5861697936f3f39bc44fd04c3f283f`;
- fit-gate manifest: `efbce14aae69bc4e87939e5baa033860a0048b31bd5176c064e09ea7761f1e2d`;
- review JSON: `eaefd356e3100d03657df94b97727a46088bd1d84198ba767819199b8b239b83`;
- review manifest: `1228e34594b3fb2fa1e501e6937b3fc719463c89bf4d39c327539a5306595a4c`.

## Complete visual inspection

All pages of the three final PDFs were rendered and inspected.

- `event_crossing_validation_o150818.pdf` has 16 pages, one event per page.
  Pages 1 and 2 separately show UID 1051 scan-row 0 events 0 and 1. The source
  spike is visible in each page's broader context while the fitted event model
  is correctly exposed as inadequate for those passages. Remaining pages span
  deterministic high-leverage, worst-residual, worst/best-correlation, and
  per-scan events. No detector or passage averaging is used.
- `event_source_aligned_stacks_o150818.pdf` has two pages. The all-event and
  four velocity-sign stacks visibly contain a compact source; the second page
  reports speed, angle, correlation, residual/leverage, and disposition
  censuses.
- `event_tau_profile_o150818.pdf` has one page with a smooth interior minimum
  near the lag point estimate.

No clipping, overlap, unreadable label, missing page, merged event, or checksum
failure was found. The three PDF digests are respectively
`20a36d0541c6f81b497fbd5e742392aa5c28f55347548d654a733d1dab2e3da3`,
`259513b83769f5398afe424a763fac69beae08b752bc74246293800ab70ecfc8`,
and `0abed2596970a091a627689eb0deeb6e97c0d89dc05616a366237a775189f355`.

## Campaign disposition and limits

The package now contains one owner-run Unity pilot, a prepared but gated
four-way array for the remaining 65 observations, and a checksum-authenticated
66-observation census. The remaining array must not be submitted before the
Unity pilot reproduces the event identity and the owner accepts the three
visual products.

This gate validates the baseline event algorithm and visual accounting. It
does not estimate uncertainty, establish a preferred physical model, identify
the upstream cause, support a universal correction, or authorize the later
held-out/bootstrap stages.

The repository also contained 36 pre-existing owner-owned untracked bundle
files at the start of this successor. They were not read, edited, staged, or
removed. Any cleanliness statement applies only to the scoped diagnostic
diff, never to the entire worktree.
