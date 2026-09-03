# SCI-POINT Working-Wheel Adoption Register

Status: Stage A disposition record; candidate until owner approval

The purpose of this register is to make non-repetition auditable. An item may
not be freshly redesigned in Stage B unless the owner changes its disposition
or the author identifies a documented scientific contradiction.

| Recovered item | Stage A disposition | Consequence for Stage B |
| --- | --- | --- |
| Known bright pointing source observed near the map center | adopt | POINT is targeted inference, not blind source finding |
| One observation-local map per TolTEC array | adopt as baseline grouping | Preserve per-array scientific results even if aggregation is later admitted |
| Six-parameter elliptical Gaussian: amplitude, two-coordinate centroid, two widths, angle | owner-approved compatibility method under ODQ-004 | Use `POINT-FIT/ELLIPTICAL-GAUSSIAN-COMPATIBILITY@1`; do not invent another profile family in base v0.1 |
| Weighted fit over a bounded region with declared center, seed/search/fallback, and parameter bounds | owner-approved compatibility machinery under ODQ-005 | Preserve configurability; bind requested/effective/realized state and expose every fallback or sentinel meaning without copying source code |
| Per-array fit table with amplitude, centroid, widths, angle, and formal errors | owner-approved required fit-result and QC role under ODQ-008 | Preserve centroid as pointing measurement and fitted parameters as bounded telescope/observing-condition QC metrics; repair names/metadata only through explicit aliases and scientific meaning |
| `sig2noise = fitted amplitude / full-map RMS` | retain only as a legacy alias for the canonical `fitted_amplitude_over_full_map_rms` diagnostic | Never call it significance or empirical S/N; the value remains unavailable until `POINT-FULL-MAP-RMS-METHOD v0.1` is approved |
| `peak_over_full_map_rms` with the same quantity | hold as a recovered implementation label, not an admitted scientific alias | It may be admitted only if the approved compatibility method establishes that fitted amplitude is the relevant positive peak for the exact source model and parent route |
| `fit_sig2noise = amplitude / formal amplitude error` | adopt as formal-fit diagnostic | No Gaussian-significance or detection-probability claim |
| AltAz tangent-plane `x_t`, `y_t` offsets in arcsec for Pointing | recommend adopt as ordinary v0.1 frame | RA/Dec fit output remains outside base v0.1 unless separately approved |
| Implementation-labelled raw and filtered observation fits | abstract, do not equate to scientific routes | Bind the exact MAP/JINC/FLT parent method and, when applicable, complete FRUIT lineage; never infer either from directory names |
| Arithmetic mean of Pointing-table `x_t`,`y_t` rows in TolTECA | hold for ODQ-001 | Preserve as recovered operational behavior; do not silently ratify or replace |
| Downstream sign flip and telescope user/paddle-offset adjustment | keep downstream under approved ODQ-002 | POINT measurement is not a correction candidate or applied correction |
| TolProj pointing-source amplitude use for flux-scale child APTs | cite as downstream CAL/TolProj use | POINT supplies honest amplitude; CAL/TolProj owns photometric transfer |
| Shared Pointing/Beammap fitter implementation | reuse engineering wheel | Scientific contracts remain separate; SCI-BEAM keeps per-detector authority |
| Pointing and OOF grouped under historical `SCI-MODE-001` | supersede | SCI-POINT and SCI-OOF are separate packages |
| Generic `SCI-SRC-001` detection/catalog scope | defer | Preserve for later blank-field source package; exclude now |
| Existing validation and accepted reductions | evidence-only | Use later for conformity/validation, never as Stage B scientific authority |

## Change Rule

Any Stage B proposal that replaces an adopted or recommended compatibility
item must identify the exact item, scientific reason, changed estimand or
claim, compatibility consequence, and owner decision required. “Cleaner,”
“more modern,” or “easier to implement” is not sufficient scientific cause.
