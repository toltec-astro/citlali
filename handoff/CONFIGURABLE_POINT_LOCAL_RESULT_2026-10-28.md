# Configurable POINT: local implementation and handoff — 28 October 2026

The new POINT route now accepts an ordinary TolTECA-prepared configuration and
produces maps and qualified fit results on all three arrays. It uses the actual
project inputs and matched APTs, and passes per-array cleaning ranks, map pixels,
coverage and worker counts to the existing numerical components. It does not
substitute a fixed bright-source experiment for the requested settings.

**Grant launches every reduction from the shell. Codex prepares the work and
waits for Grant to report completion.** Codex incorrectly launched the two local
trials below before that boundary was reaffirmed. Both have finished; their
outputs are preserved. The implementation and source review do not confer
operational or production authorization.

## What is ready

The local trial remains:

`/Users/gwilson/work_toltec/local_data/2026-refactor/projects/COM01-local-trial-20261027`

The new primary POINT directory is `pointings-configurable-20261028`. The
independent check is `pointings-configurable-check-152418`. All original
preparation and reference reductions remain intact: 1,149 sealed preparation
files, 667 copied reference files, the corresponding 667 original reference
files, and 288 raw-data symlink targets passed the preservation check.

| Trial | Requested settings | Observed result |
| --- | --- | --- |
| POINT 152389 | All three arrays; 11 networks; observed-ALS ranks 5/5/5; 1 arcsec pixels; coverage 0.1; 6 workers; quiet mode | Completed in 111.83 seconds. All 66 processing segments converged. All three maps and fit diagnostics were published. No unexpected native error messages. |
| Independent POINT 152418 | All three arrays; 11 networks; ranks 3/4/6; 2 arcsec pixels; coverage 0.15; 3 workers; quiet mode | Stopped at the existing RTC availability guard on network 11. Other ten networks completed, with their requested array ranks. No MAP stage or discoverable final observation was published. |

The second case establishes that the observation and cleaning settings reach
execution; it does **not** establish independent-observation qualification or
real-data verification of the changed map pixels. Its failure remains an open
gate, not a result to conceal or retune away.

## Changes made

1. Added a versioned POINT configuration profile with an explicit migration
   from retired legacy controls. The received configuration and a separately
   named effective configuration are preserved. Unknown new controls and
   unsupported feedback requests fail clearly.
2. Connected project-relative KMP discovery, exact raw/telescope/APT bindings,
   actual network identities and observation selection to the existing native
   processing. Each observation's bindings are shared across arrays. Recovered
   the earlier fix that prevents sorting inputs from changing their YAML.
3. Published only complete all-array results in a new `reduNN` directory. Native
   FITS meanings, unavailable values, units and frames are checked. Required
   processing and output failures return nonzero status to TolTECA.
4. Added qualified forward fit inspection using the existing POINT-EXPECTED
   method. Corrected the application adapter for arcsecond map headers and a
   40-arcsecond fitting region at different pixel sizes. It avoids preparing
   unused feedback machinery.
5. Defaulted optional monitoring and intermediate retention off. The successful
   first trial removed 3,730,191,280 bytes of its generated binary transports
   after their consumers completed. Receipts and identities remain. No speed
   improvement relative to a controlled baseline is claimed.
6. Froze the Python/data/native runtime behind a launcher that survives TolProj
   executable copying and verifies its bound files. Existing TolTECA setup,
   configuration merge, APT validation, TolProj resource validation and
   node-local launcher were reused. No sibling repository was changed.

Independent review found and closed two additional compatibility gaps:
ordinary nonpositive MJD constant-offset sentinels now use the established
constant-offset meaning in the derived configuration; unsupported nonunit
flux-scale corrections are rejected rather than silently ignored. A native CLI
smoke check also corrected the spelling of its warning-level argument.

The local merge check exposed an existing deployment detail: the frozen Tollan
sorts configuration files by numeric prefix only. Two files numbered 99 must
not compete to select the executable. The pilot selects the source in the
71 runtime file, removes the inherited observation-file executable field, and
lets TolProj's generated 99 overlay select the launcher. The expert 99 file
contains scientific settings rather than another executable selection.

The Mac launch uses TolProj's existing job-start fallback because TolTECA clears
the subprocess environment and macOS lacks Linux `/proc` recovery. The actual
selected launcher hash matched the frozen runtime. The second trial explicitly
selected the prior immutable TolProj snapshot as its source. This local result
does not validate Slurm submission-environment recovery; that remains part of
future owner-run Unity execution. No Unity connection was made.

## Exact software and configuration identities

| Item | Identity |
| --- | --- |
| Reviewed implementation source | `87e9217fc0016839d43684f539e93774199a32a1` |
| Source tree | `d942fe4edcdcf4f8184ff7d9b278d022c19a8edb` |
| Recovered numerical base | `6e2a467b85f07f15b8fdfe947bed81df833d11bb` |
| Implementation branch | `codex/configurable-point-20261028` |
| Frozen runtime | Trial `software/configured-point-87e9217fc` |
| Runtime manifest SHA-256 | `44b98db5e36ecfdd0e78c4ab95fa22640b3108e84f08377e40a986f896347f5a` |
| Launcher SHA-256 | `ebff492fe20541e4784157382d3253f66b7602e9434d1aa71d89fd1ed3d2409f` |
| TolTECA preparation export | `188a38e21307295e880c4d96109abed5d02424df` |
| Tollan export | `1fad1b2b550ffc03470b1dacae70a7ef54b9e701` |
| Frozen TolProj preparation export | `d9225ffab364b9381af3984d5e362b15587feafa` |

The native executable was rebuilt with the matching source revision before
freezing. Its version also records the existing `kids 04088da-dirty` dependency
realization. This is supplemental AppleClang/Homebrew/cached-dependency Mac
evidence. It is not reproduction of the accepted Unity Spack V2 environment.
External Python package contents and shared libraries remain explicitly
unpinned. The runtime manifest records Python identity, package versions and
source roots; all 233 bundled file bindings passed independent verification.

Full configuration, executable, input, receipt and output hashes are in
[`LOCAL_EVIDENCE.json`](../validation/configured_point_2026-10-28/LOCAL_EVIDENCE.json).
The runtime's `RUN.json` records distinguish requested settings, effective native
inputs and actual results. The failed trial's input and partial native products
are retained for diagnosis.

## Products and acceptance

| Required or proposed product | Present status | Release limitation |
| --- | --- | --- |
| One map for each array | Present for 152389; withheld for failed 152418 | Six native planes; not the legacy analysis FITS layout |
| Fit diagnostic for each array | All three 152389 fits accepted | Conditional uncertainty and centroid are not yet conventional calibration/pointing products |
| Exact inputs/APT/config/executable and run outcomes | Recorded for both trials | Local environment only |
| Conventional calibration POINT table | Not issued | Need truthful amplitude-error and pointing-offset consumer semantics |
| Pointing-derived calibrated APT consumed by SCIENCE | Not demonstrated | Must trace the established calibration machinery into the exact science APT |
| All-array SCIENCE maps and genuine multi-observation coadd | Not demonstrated | First collaborator milestone remains incomplete |
| Existing downstream analysis | Not qualified | Old reader assumes different HDU positions and units |

Existing POINT-EXPECTED acceptance was used unchanged:

| Existing criterion | Meaning |
| --- | --- |
| Complete finite support in the 40 arcsec fit disk | At least 30 fit pixels and 10 central pixels |
| Converged finite Gaussian-plus-constant fit | One fit; positive amplitude |
| Centroid radius ≤20 arcsec | Source remains within the existing fit domain |
| Both fitted sigmas >1.5 arcsec | Neither lower bound active; existing solver width ceiling 20 arcsec |
| Identifiable amplitude with positive finite conditional variance | Unavailable is distinct from source rejection |
| Conditional amplitude S/N >10 | Existing diagnostic acceptance, not calibration qualification |

For 152389, amplitudes were 985.80, 669.57 and 698.73 mJy per nominal beam for
a1100/a1400/a2000. Conditional sigma_A values were 12.08, 9.32 and 9.09 in the
same units. These numbers are inspection evidence, not a new flux calibration.
Quiet and detailed saved-map fit calculations reproduced all three original
fit records exactly. Published maps were unchanged. These checks and the map
preview were completed before the owner-run correction; no further numerical
work follows automatically.

Grant's remaining scientific decisions concern the acceptable calibration-error
meaning, independent response/flux recovery checks, and SCIENCE/coadd acceptance.
No new thresholds have been invented to mark the present candidate successful.
Existing diagnostics do not themselves establish scientific improvement over
historical reductions.

## Independent check blocker

For 152418, network 11/channel 92 retains 1,741 conditioned samples in six
separate stretches. Its longest stretch is 416 samples; the existing four-second
spectral assessment requires 488 contiguous samples. Both x and r therefore
report `fixed_grid_unavailable`. The RTC reassessment records
`required-coordinate-outcome-unavailable` and correctly withholds completion.
The detector has nonempty support, so the existing narrowly scoped empty-detector
rule does not apply. This is upstream of CAL/PTC and unrelated to the requested
cleaning ranks or map pixels.

Next investigation: inspect that channel's already saved event/support decisions
against the existing RTC contract. Do not remove the guard, alter its duration,
exclude a detector or adjust a threshold merely to obtain a successful run.
Any numerical follow-up is prepared for Grant to trigger.

## Owner shell procedure

The ready primary script is:

```bash
cd /Users/gwilson/work_toltec/local_data/2026-refactor/projects/COM01-local-trial-20261027/pointings-configurable-20261028
bash 02_redu.sh
```

This is a command for **Grant to use when he chooses another run**, not a request
to repeat the already completed reduction. It verifies the prepared scope and
snapshot, then calls TolTECA; the adapter creates another fresh `reduNN`. The
existing `redu00` is preserved. The independent 152418 script is also prepared,
but its known RTC blocker has not been repaired, so no repeat is recommended.
Codex must wait for Grant's completion notice before inspecting a new run.

No push or canonical integration was performed. Source review passes with
recorded limitations: 168 focused tests reproduced independently; 1,332 enabled
native tests, 207 baseline-tool tests and the full configuration gate passed.
One pre-existing native lifecycle test remains disabled. Engineering completion
and scientific/operational release acceptance remain separate.

## Ordered remaining work

1. Diagnose the saved 152418/channel-92 support loss and disposition it under the
   accepted RTC contract. Prepare any justified follow-up for owner execution.
2. Complete the conventional POINT calibration-table contract and consumer
   verification, including units, identity, unavailable values and uncertainties.
3. Follow the resulting calibration through the established machinery to the
   exact APT consumed by SCIENCE.
4. Connect configurable SCIENCE and celestial coaddition on 152390 and 152392,
   retaining all three arrays and existing downstream consumers.
5. Execute the planned comparisons: historical reference with recorded
   calibration; new SCIENCE with that compatible calibration; and new SCIENCE
   with calibration from new POINT. Preserve the identified reference reduction
   `reference/v2-science/refactor/NGC4449/reduced/redu03`.
6. Prepare the exact Unity software/configuration identities and owner-run
   commands. Grant triggers builds/reductions and returns the evidence. Beammap
   production and OOF remain outside this release.
