# Historical control: internal recovery dossier

Manager/owner use only. Status: source recovery, not scientific authority or
implementation conformance assessment. Source:
`f70701ad488444f3e2528c6bbe3e798863c9e301`, tree
`2009a1397bd67d615a1d6e9a8419e18fc794a81e`.
[Exact copies and identities](SOURCE_IDENTITIES.json) bind every cited file.
Paths below are relative to `inputs/manager/historical/`.

## What the selected source establishes

| ID | Source and location | Recovered fact and limit |
| --- | --- | --- |
| H01 | `include/citlali/core/pipeline/reduction_iteration_loop.h`, `run_reduction_iterations`; `reduction_observation_pipeline.h` | Each loop invokes a full reduction iteration over the original observation objects. It does not feed a previous residual array directly into the next iteration. This recovers a control flow, not a proof that every upstream realization is identical. |
| H02 | `previous_fruit_loop_map_loading.h`, `load_previous_fruit_loop_maps_if_needed`; `fruit_loop_map_io.h` | The selected preceding map bundle is loaded. The save-all setting changes which reduction directory supplies it. The map's signal and available companions drive projection; the next completed selected map replaces it. |
| H03 | `include/citlali/core/engine/detail/pointing_run_impl.h` and `pointing_fruitloop_impl.h` | Original-observation RTC and pre-PTC masks precede removal. Residual atmosphere/line processing and PTC run before the residual weight/reset/noise-map pass; that pass then restores the map. Post-cleaning detector removal, optional post-addback weight recomputation, learning collection and final mapmaking follow. Only this named residual subset is bypassed. |
| H04 | `include/citlali/core/timestream/timestream.h:2522`, `TCProc::map_to_tod` | Removal and restoration call the same projection helper with opposite signs and the same loaded map/policy. Both consult current detector/sample flags. Residual processing can change those flags, so equal removal and rejoin support is not a recovered invariant. |
| H05 | `timestream.h:2384`, `sample_map_jinc` | Historical feedback interpolation is a signed kernel-weighted average normalized by the retained signed coefficient sum. Missing kernel state or a near-zero sum can fall back to bilinear sampling; empty/no-overlap paths return zero. This is not a proof of an adjoint, inverse, or equivalence to the frozen forward JINC estimator. |
| H06 | `timestream.h:2522`, selection branches in `map_to_tod` | Sign-dependent S/N-like, absolute-flux and adaptive gates are combined by OR. Adaptive spatial support constrains its own branch; it is not necessarily a universal support cut on the union. Missing RMS/flux/adaptive state can disable a gate or use a fallback lookup. Kernel propagation has separate enable/shape checks. These are material method choices, not universally legitimate unavailable-state rules. |
| H07 | `fruit_loop_iteration_policy.h`, `configure_fruit_loop_iteration_policy` | Interpolation follows mapmaker unless overridden, with a non-JINC fallback; centering, weighting-after-rejoin, JINC shape/support/phase settings and save policy affect the realized procedure. Equality to a later contract cannot be inferred from the shared name “JINC.” |
| H08 | `fruit_loop_iteration_state.h:23`, `fruit_loop_iteration_pending`; `reduction_iteration_setup.h`; `reduction_iteration_loop.h` | The loop checks `fruit_iter < max_iters` and an initialized-false convergence flag. The inspected orchestration supplies no scientific convergence decision. A count limit must not be relabeled convergence or terminal adequacy. |
| H09 | `fruit_loop_restart_lifecycle.h`, `initialize_fruit_loop_restart_if_requested` | The loader binds observation IDs, requested type, learning/PTC policy, completed/next iteration and carried learning/weight-validation state. It requires the next index below the cap. This is evidence of a continuation mechanism, not proof of complete causal state under the frozen core. |
| H10 | `config/tolteca/point/70_pipeline.yaml` and `90_user_overrides.yaml` | The template names pointing, JINC, no coaddition, `obsnum/raw` feedback, network PTC cleaning, residual weights, noise products, kernel and learning controls. FRUIT itself is disabled; optional post-map filtering is enabled separately. This template plus an empty override is not the ordered configuration of an executed FRUIT control. |
| H11 | `CMakeLists.txt`; recovered Gate-0 feasibility | The source build uses external dependencies, including a floating kidscpp `v1.x` reference. A source commit does not pin all compiler, library, runtime and executable bytes. No build or executable reconstruction was attempted. |

H01–H04 agree with the earlier
[recurrence recovery](inputs/manager/recovery/HISTORICAL_RECURRENCE_BASELINE.md).
Its study outcomes and tests remain historical evidence; they were not rerun
or promoted to scientific authority. H09's source term “restart” must be
translated using the current core: same-lineage continuation after absolute
index N begins at N+1; a scientific restart creates a new lineage.

## Concrete policy surface, not an executed control binding

These are literal entries in the historical pointing template, provided so
review does not hide significant choices behind “historical defaults.” The
actual ordered run configuration and overrides remain missing. None of these
values is proposed as a scientific default or admitted to the author channel.

| Historical field | Recovered template value | Why the method must bind it explicitly |
| --- | --- | --- |
| `fruit_loops.enabled`, `type`, `path` | `false`, `obsnum/raw`, `null` | The template does not itself request a FRUIT run or a map seed. |
| `max_iters`, `save_all_iters` | `10`, `true` | Count and retention are distinct from convergence and scientific completion. |
| `interp_mode_override`, `legacy_center` | `auto`, `false` | Projection and centering can change the numerical model. |
| `mode`, `sig2noise_limit` | `upper`, `100` | Sign and companion-dependent selection are scientific choices. |
| `array_flux_limit` | `[12, 18, 10]` in the configured signal scale | Exact array association, scale and OR-gate interaction need binding. |
| `adaptive_support_radius_arcsec`, `adaptive_support_radius_fwhm` | `18.0`, `1.5` | Adaptive branch support is not automatically total projection support. |
| `local_snr_floor`, `peak_fraction_limit` | `0.0`, `0.0` | Activation semantics and missing-state behavior need explicit rules. |
| `recompute_weights_after_addback` | `false` | Residual weights and post-rejoin weights define different descendants. |
| `learning.enabled`, `learn_iters`, `apply_start_iter` | `true`, `2`, `2` | Learning phases and absolute indexing are causal state choices. |
| `processed_time_chunk.clean.grouping`, ordinary PCA cuts | `[nw]`, `[5]` for each array | PTC detector grouping/mode count are separate from per-array map grouping and need upstream authority. |
| `processed_time_chunk.weighting.type` | `validated` | This implementation label does not identify a frozen JINC-permitted coefficient family. |
| `mapmaking.method`, `pixel_axes`, `pixel_size_arcsec` | `jinc`, `altaz`, `1.0` | Exact coordinate/frame/grid and numerical JINC authority are separate requirements. |
| `mapmaking.jinc_filter.r_max`, `subpixel_n` | `1.5`, `1` | Historical support/cache settings are not owner-authorized TolTEC parameters. |
| `mapmaking.jinc_filter.shape_params` | a1100 `[1.1,1.67,2.0]`; a1400 `[1.1,2.17,2.0]`; a2000 `[1.1,3.17,2.0]` | Each inherited shape value remains unauthorized numerically under frozen JINC. |
| `noise_maps.enabled`, `n_noise_maps`, RTC kernel | `true`, `5`, enabled Gaussian | Ensemble count and kernel presence supply neither NOI uncertainty nor a model/PSF truth role. |
| `post_processing.map_filtering.enabled` | `true` | A raw feedback branch may coexist with filtered outputs. Excluding filtered feedback does not prove filtering was absent from historical runtime. |

This is a selected consequential-field inventory, not the full ordered policy.
The exact template copy includes additional flagging, weighting, source-center,
randomization, fitting, output and learning controls. They must be dispositioned
if active in a later bound method; no omitted setting is declared irrelevant.

## Proposed control inventory, still unbound

A runnable control needs one immutable record binding the historical source,
actual executable and dependencies, ordered effective configuration, original
observation/calibration/geometry identities, initial map or unseeded state,
all carried learning/weight/flag state, stochastic controls, absolute iteration
extent, requested products and output destination. The template supplies none
of those realized identities by itself. No “use current defaults” placeholder
is acceptable.

Prefer an existing preserved executable/environment and exact run inputs.
If those do not exist, separately propose a pinned reconstruction, an artifact
anchor and its predeclared comparison tolerance. If neither can be established,
report exact executable control unavailable. Do not quietly substitute current
mainline or claim source similarity proves numerical identity.

## Scientific questions recovered from the code

- Is this internally constructed preceding-map model an adequate object for
  the named scientific use, with an explicit target relation and additive
  reference? Its acceptance grants feedback use, not sky truth.
- What exact model selection and map-to-sample projection are authorized,
  including sign, normalization, support, boundaries and missing-state behavior?
- Which upstream operations are rerun on each residual, which defining state
  is relearned, and which fixed-map application coordinates are reevaluated?
  Historical “coefficients” do not distinguish these roles.
- Can the rejoined signal and its coefficient/coordinate/QC ancestry enter the
  selected JINC route, and who supplies any selector-required companion?
- Which source, support, response, uncertainty and continuation claims can be
  made honestly, and which must remain unavailable?

These are questions for the scope and method owner. They are not instructions
to preserve a fallback, select a default, or change software. No comparison of
Citlali against a frozen contract, pass/fail audit, defect repair or new score
was performed.
