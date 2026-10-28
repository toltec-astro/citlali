# POINT 152418: saved-evidence diagnosis and decision proposal

28 October 2026. **Proposal only; no processing policy has changed.**

Subsequent owner decision: prioritize observation-local exclusion of compromised
detectors, with mandatory drop-count diagnostics. See
[the decision record](../doc/COMPROMISED_DETECTOR_POLICY_2026-10-28.md).
The recommendation below is retained as historical reasoning; short-stretch
assessment is not the selected next direction. The saved diagnosis is unchanged.

## Recommendation

Evaluate the existing short-stretch spectral-assessment method through an
explicit, bounded configurable-POINT option. Preserve the detector's samples,
event exclusions, filters, paired-coordinate validity, calibration, cleaning
ranks and map settings. Do not discard channel 92 or bypass its missing
assessment merely to obtain a map.

The method already exists and was owner-approved for a different, narrowly
scoped September trial. Extending its use to this pilot is a scientific-policy
decision, not a correction of a demonstrated implementation bug. The earlier
approval is precedent, not authorization for this observation.

Separately, investigate whether the existing whole-processing-segment exclusion
for detected level shifts sacrifices useful source signal. That is a larger
question than the immediate assessment failure and should not be bundled into
the first change.

## What was inspected

This is a read-only diagnosis of saved YAML receipts, byte masks and source.
Small counts and interval arithmetic were used to reconcile the saved facts.
No spectra, detector fits, filters, maps, reductions, or numerical experiments
were executed. No raw input, existing output, configuration, or application
source was modified. No Unity access occurred.

Pilot root:

`/Users/gwilson/work_toltec/local_data/2026-refactor/projects/COM01-local-trial-20261027`

Failed run: `pointings-configurable-check-152418/reduced/redu00`.
Evidence below it: `.point-work/upstream/products/network11`.

Executed source: `87e9217fc0016839d43684f539e93774199a32a1`.
Inspection checkout: `19d7f932799af88cf041101a6bebc6f5e0318557`, branch
`codex/configurable-point-20261028`; initially clean. The inspected RTC
execution, filter-support, spectral and completion implementation files are
unchanged between these revisions. The executed environment was the local Mac
candidate, not Unity/Spack qualification.

The engineering, successor and review governance documents were read for this
diagnosis, with the current status and original handoff. This report selects no
new scientific authority and makes no integration, implementation-review or
release claim. Existing scientific acceptance criteria remain unchanged.

Machine-readable extracts and input hashes are in
[POINT_152418_SAVED_EVIDENCE_2026-10-28.json](POINT_152418_SAVED_EVIDENCE_2026-10-28.json).

## Diagnosis

Ten of eleven networks completed their requested upstream processing. Network
11, on a2000, stopped at final RTC reassessment, before its CAL/PTC processing.
The application then withheld all-array MAP publication. The requested ALS
ranks and map pixel size did not cause this stop.

Network 11 has 486 detector columns and 972 x/r coordinate assessments:

| Saved outcome | Count | Meaning |
| --- | ---: | --- |
| Available assessments | 852 coordinates / 426 detectors | Assessment exists; this alone is not scientific qualification |
| Unavailable with no surviving assessment support | 118 coordinates / 59 detectors | Saved output masks also show no numerical x/r output for these detectors |
| Unavailable with nonempty support | 2 coordinates / 1 detector | Both coordinates of channel 92 fail the fixed-grid requirement |

The present completion exception allows certain fully empty detectors, with
additional evidence and donor-dependency checks. It explicitly excludes short
but nonempty support. Channel 92 is therefore a definite blocker. The census
does not prove all other completion checks would pass after addressing it.

### The important distinction: output support versus assessment support

Channel 92 has **3,289 native rows with numerical x/r output**, approximately
26.94 seconds. Its saved factor-two output has 1,592 admitted map-center
candidates. These are upstream candidates, not delivered map measurements.

Its spectral assessment is allowed to use only **1,741 native rows**, about
14.26 seconds, distributed over six separate stretches. Earlier descriptions
of 1,741 surviving samples should be read as *assessment-eligible samples*,
not all surviving filtered output. The distinction matters when deciding
whether to discard the detector.

The six assessment stretches contain 121, 416, 274, 344, 214 and 372 samples.
The longest is about 3.41 seconds. The selected assessment policy requires at
least one stretch of 488 samples, approximately four seconds. It therefore
returns `fixed_grid_unavailable` before accumulating any spectral windows.

### How the support became fragmented

The saved original spectra for both coordinates were available, with 7,878
eligible samples in a single physical acquisition run. The following chain
accounts for the later losses without reconstructing any numerical signal:

| Step | Saved evidence and effect |
| --- | --- |
| Two admitted level shifts | Events 535 and 537, near native rows 880 and 4004, trigger exclusion of processing segments 0 and 3: `[32,1220)` and `[3660,4880)`. Both coordinates are excluded under the paired policy. Total: 2,408 rows, about 19.73 seconds. |
| Three other excursions | Events 536, 538 and 539 lack isolated-event admission. Their unavailable-operation intervals are `[2498,2581)`, `[5530,5544)`, and `[7050,7107)`: 154 rows. Coincident events in other detectors are recorded; no isolated donor repair was admitted. |
| Processing coverage | 622 rows carry `processing_support_unavailable`. They account for the portions outside the selected processing support. This is not an acquisition gap. |
| Filter-input support | 4,694 rows remain. Under the existing speed-support policy, original measurements at speed-restricted centers can still supply the fixed filter. |
| Adaptive assessment support | An additional 170 otherwise supported rows fail the center-speed rules: 18 below the minimum and 152 above the output-sampling limit. These splits leave 4,524 assessment rows before the low-pass filter. |
| Finite filter footprint | The a2000 low-pass filter has 307 coefficients, requiring 153 samples of support on each side. Numerical filtering leaves 3,289 rows. Applying that footprint to the more fragmented assessment runs leaves the six stretches totaling 1,741 rows. |

The post-notch assessment is available. Its stored common-support intervals
exactly equal the retained runs in the input-cause mask. Trimming each such run
by 153 samples at both ends, discarding empty intervals, exactly reproduces the
saved post-low-pass assessment support. This directly links the final failure
to the interaction of event exclusions, speed-use restrictions and filter
footprint, rather than a guessed YAML or detector-identity problem.

The saved event classifications do not establish the physical cause of those
excursions. In particular, this run used no spatial source-protection mask.
We have not shown whether any excluded signal was astronomical, instrumental,
or mixed. The observed exclusions are not proof that the detector is garbage.

## What the assessment requirement protects

The assessment measures frequency-domain behavior of original and treated
signals on matched support. The fixed grid, minimum stretch duration, minimum
number of windows, validity checks and matched footprints prevent an absent or
incomparable measurement being presented as measured treatment performance.

The present default already pads eligible two-to-four-second stretches when
there is at least one full four-second stretch. Its additional prerequisite
rejects the entire spectrum when every surviving stretch is shorter than four
seconds. Thus the transition from one full stretch to none can stop the run
even when multiple shorter eligible stretches remain.

An available spectrum is not a certificate of acceptable residual lines or
source preservation. The saved decision explicitly reports unresolved
scientific qualification and an unselected residual-line acceptance policy.
Addressing measurement availability must not silently convert those limits
into scientific acceptance.

## Recovered prior work

[RUNTIME_REPAIR.md](../validation/rtc_point_transfer_2026-09-28/RUNTIME_REPAIR.md)
records the September 29 owner-approved option
`rtc-conditioned-padded-2s-fixed-4s-v1`. It permits conditioned and matched
assessment on separate eligible stretches of at least approximately two
seconds even when none reaches four seconds. It preserves:

- The original broad-spectrum selector and original samples.
- The four-second FFT grid, existing Hann normalization and centering rules.
- The two-second minimum duration and two-window minimum.
- Physical gaps, masks and identical before/after comparison footprints.
- Unavailable arithmetic, empty-detector restrictions and donor checks.

The implementation and focused tests already exist. They were inspected, not
rerun here. The current CLI selects the option only through a private profile
restricted to a different observation/source-probe campaign. That profile also
carried a PTC iteration-budget change. It must not be copied into this pilot or
made universally applicable as a shortcut.

For channel 92, four saved stretches meet the existing 244-sample minimum:
416, 274, 344 and 372 samples, totaling 1,406 samples (about 11.52 seconds).
That exceeds the existing two-window minimum by interval accounting. This is
evidence that the option is relevant, **not a newly computed spectrum or a
guarantee of successful completion**.

Zero padding supplies a common transform grid; it does not create missing
observations, independent information, or the resolving power of an actual
four-second stretch. Short-window power and line diagnostics retain their
profile-specific interpretation and cannot be promoted to calibrated noise
or general residual-line acceptance.

## Decision options

| Option | Benefit | Risk / cost | Recommendation |
| --- | --- | --- | --- |
| Keep the present requirement | No scientific-policy change | This observation remains blocked despite useful output elsewhere | Preserve as the recorded baseline |
| Explicit short-stretch assessment using the existing method | Retains current numerical samples and obtains an assessment where support permits | Less spectral information; requires truthful window-support reporting and bounded validation | **Preferred first proposal** |
| Exclude channel 92 and continue | Could preserve other detectors while retaining the strict assessment rule | Discards about 27 seconds of numerical output; can change donor and atmospheric-model dependencies; needs a new admission rule | Reserve for evidence that the channel is unusable, not solely this missing assessment |
| Relax event exclusions, speed rules, filter footprint or assessment duration | Might recover more data | Changes sample admission or signal processing; may admit artifacts or bias source response | Separate scientific investigation |
| Ignore the missing assessment or substitute a passing value | Produces apparent progress | Hides absent evidence and defeats the failure distinction | Reject |

## Proposed bounded next work, subject to owner decision

Extend only the existing conditioned/matched assessment choice to this local
POINT pilot. Keep the current default available and record the selected policy
explicitly in the effective configuration and final outcomes. No observation-
specific channel list, automatic fallback, automatic retry, or parameter sweep.
Do not import the old campaign's source probe, PTC iteration budget or feedback.

The implementation proposal should give the spectral owner the choice and let
the project adapter pass it through; it should not place scientific decisions
in launcher scripts. Unknown or incompatible requests must fail clearly.

Before an owner-run trial, verify that the change leaves original spectra,
sample masks, filtered values, event decisions, map-center candidates and all
unrelated processing settings unchanged. Verify that both matched sides use
the same policy and real sample footprints, and that too-short, insufficient,
nonfinite or inconsistent inputs still report unavailable. Reuse the existing
short-stretch tests and obtain a focused independent review of the exact
implementation. These are proposed gates, not checks claimed complete here.

Prepare one new 152418 reduction for Grant to launch after review, preserving
the failed run. Retain 152389 as a clean comparison; any rerun or additional
experiment must be explicitly prepared for Grant. Inspect all three arrays,
source positions and profiles, amplitude qualifications, residual structure,
coverage and every remaining failure. Mapmaking and CAL/PTC downstream of the
failed network remain untested by this diagnosis.

No new numerical scientific tolerance is proposed here. The existing
POINT-EXPECTED criteria remain in force, and flux-calibration qualification
remains open even if the trial completes.

## Follow-on question: avoiding unnecessary loss of detections

The two level shifts remove almost twenty seconds from this detector through
whole-segment exclusion. That is a real loss mechanism worth investigating,
but current evidence does not justify narrowing those exclusions. A separate
proposal should inspect the saved event/source associations and compare the
current rule with physically justified alternatives. Source-dependent cuts
must not be selected because they make the resulting map look better.

## Decision requested and remaining limits

Approve, reject or revise the bounded proposal to expose the **existing
short-stretch assessment policy for this local configurable-POINT trial**,
with all processing and exclusion policies held fixed. This report does not
implement that choice or authorize a reduction launch.

Remaining uncertainties: the short-stretch spectrum has not been calculated;
all remaining RTC completion checks have not been exercised under that option;
network 11's downstream CAL/PTC and the observation's maps do not exist;
physical causes and source consequences of the event exclusions remain
unresolved; calibrated flux and production suitability remain unqualified.

Grant alone launches reductions. No existing evidence or reference reduction
has been deleted or overwritten.
