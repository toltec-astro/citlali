# SCI-CAL-001 AM 12.2 H2O-scale provenance-hypothesis report

## Status

This is diagnostic P1: a post-hoc candidate-input-recipe search. It is not historical custody proof, a holdout test, operator authorization, an operational-domain declaration, or observational photometric validation.

The legacy `am_q25/am_q50/am_q75/am_q95` targets are the generic unprefixed TolTECA registry family. The copied `annual`, `DJF`, `MAM`, `JJA`, and `SON` MERRA-2 profiles are separate explicitly named AM-12.2 families. A numerical match does not rename a copied profile as a registered generic q artifact.

## All-direct P1 rank-one hypotheses

The fitted-scale 0--500 GHz by 31-elevation AM grid was run directly for all 100 target/profile hypotheses. Transmission and Rayleigh-Jeans rankings are separate; no unregistered composite score or near-exact cutoff was invented. q95 has only the weaker nominal ratio-surface ranking because its registered raw grid is absent.

| Generic target | Direct ranking | Copied profile hypothesis | H2O scale | Direct RMS residual | Direct max absolute residual | Max correction error |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `am_q25` | transmission RMS | `LMT_MAM_5` | `1.81225445269332575e+00` | `5.11939193880871224e-03` | `2.26840000000000375e-02` | `7.79414740836802711e+00` |
| `am_q25` | Rayleigh-Jeans RMS | `LMT_DJF_5` | `3.01439309124786581e+00` | `7.77548133113115214e-01` | `3.59280000000001110e+00` | `8.34500816020430307e+02` |
| `am_q50` | transmission RMS | `LMT_MAM_25` | `9.15696647246186712e-01` | `3.23305754092318917e-03` | `1.95999000000000034e-02` | `9.98458439029974554e-01` |
| `am_q50` | Rayleigh-Jeans RMS | `LMT_DJF_25` | `2.02963214820032256e+00` | `6.04530350357074253e-01` | `2.22339999999999804e+00` | `7.36370582820754521e+02` |
| `am_q75` | transmission RMS | `LMT_DJF_50` | `1.88602893644962655e+00` | `1.56476455256103768e-03` | `1.03363000000000205e-02` | `4.83405416660586074e+00` |
| `am_q75` | Rayleigh-Jeans RMS | `LMT_DJF_75` | `1.01048455031671569e+00` | `4.92756783098706019e-01` | `1.84639999999998849e+00` | `1.41911867623572991e+01` |
| `am_q95` | nominal ratio-surface RMS | `LMT_DJF_25` | `6.88363302058917359e+00` | `5.41090729776348960e-03` | `1.88189832335683427e-02` | `1.19094929017647764e-02` |

For q25/q50/q75 the principal direct comparison is the complete 50001-by-31 legacy transmission and Rayleigh-Jeans grid. For q95, whose registered raw datafile ID 461 is absent, it is only the 93-point nominal-frequency elevation-ratio surface derived from repair-base degree-six literals; that q95 evidence is strictly weaker.

## Method boundary

Each immutable AMC file contains exactly one `Nscale troposphere h2o %9` statement. The scale was seeded only from direct AM scale 0 and the copied scale-1 T225 optical depth, checked against a direct scale-1 run, then located on the exact parsed-transmission plateau at 225 GHz and EL80. The canonical scale is the midpoint of the innermost observed plateau interval after a fixed 48 bisections. Every evaluation and bracket is preserved as a digest-bound external-cache trace.

Frozen P1 is fulfilled by the all-direct fitted-scale lane, not by a surrogate. The earlier all-profile affine LOS-tau/Trj construction is retained only as ancillary screening and is checked against every direct grid; it is not used for P1 completion or final ranking.

The 25 copied full products contain `12848` exact printed transmission zeros at opaque spectral samples. These are accepted only with finite nonnegative `atmTaun` and an absolute tau-to-transmission consistency difference no larger than `1.0e-06`. LOS tau, not `-log` of the rounded transmission field, is authoritative for construction and fractional-correction metrics.

The one-percent fields are provisional numerical diagnostics only. They do not establish 5--10% absolute flux accuracy or approximately 5% observation-to-observation repeatability, and they do not reduce common calibrator, Beammap-extinction, selector, aligned-elevation, timing, or airmass systematics.

No additional atmospheric profile, scale parameter, passband, frequency, elevation, or fitting degree was introduced. Numerical rank one can narrow a post-hoc candidate recipe but cannot establish generic-q custody, because profile selection and H2O-scale inference were performed after the legacy surfaces were known.

## Execution integrity and predecessor disposition

The canonical v3 cache binds every sidecar and scale trace to immutable execution-context SHA-256 `05148050e96e73577ec75be525b026b5bf37bbd2a8753f8e3702fc0b6dfb2bee`. One process held the whole-cache exclusive POSIX lock throughout execution and artifact construction; cache-only verification uses a shared lock. `LANG=C` and `LC_ALL=C` were pinned for AM subprocesses.

Across `13667` unique referenced v3 AM runs, `3875` returned status 1 with only the accepted unresolved-narrow-line warning structure. Those warnings, their counts, and normalized warning-bearing output identities remain explicit diagnostics; this report does not call the software execution clean or warning-free.

The interrupted external cache `sci_cal_001_h2o_scale_p1_20260801_root_v2` is noncanonical and excluded. It was stopped after cache-provenance review because it had neither a whole-cache cross-process lock nor immutable execution-context binding. Its retained partial inventory is 12,455 raw outputs, 12,455 execution sidecars, and 100 scale traces. It completed 1,764 general all-hypothesis plus 124 selected-rank-one direct fitted-scale grids: 1,888 of the expected 3,100 total. The targeted SIGINT also left three excluded status -2 failure sidecars and three empty outputs for `LMT_JJA_5/am_q25` at ZA 10, 50, and 54. The v2 cache is never used for v3 artifacts or rankings.

The first context-bound v3 development cache `sci_cal_001_h2o_scale_p1_context_v3_final_20260801_root` is also noncanonical and excluded. A pre-full-grid runtime review found that its digest inventory retained complete parsed arrays, projecting approximately 7.75 GB of avoidable retained memory. It was stopped during anchor inference after 1,811 matched raw-output/sidecar pairs and 16 traces; the targeted SIGINT left six empty outputs, three complete status -2 failure sidecars, and three empty atomic sidecar temporaries. The cache remains untouched and is never reused. This was a software-execution provenance correction, not a scientific-protocol change; the canonical process retains only frozen lightweight run identity, digest, and diagnostic records.
