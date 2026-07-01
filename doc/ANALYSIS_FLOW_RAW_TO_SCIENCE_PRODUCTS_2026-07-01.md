# Citlali Analysis Flow: Raw Data to Science Products

This note is a visual map of the current Citlali/TolTECA reduction flow as used
by the structural refactor validation work. It is intentionally higher level
than the C++ helper split: the goal is to show what happens scientifically and
operationally from raw observation inputs to the products we compare.

## End-to-End Flow

```mermaid
flowchart TD
    reduce_yaml["TolTECA reduction YAMLs<br/>70_reduce.yaml plus NN*.yaml overlays"]
    tolteca["tolteca reduce<br/>merge overlays and build low-level Citlali config"]
    lowlevel["Citlali low-level config<br/>citlali_o*_c*.yaml"]

    raw["Raw observation bundle<br/>KIDs timestreams + telescope data + APT/calibration tables"]
    runtime["Citlali runtime setup<br/>config validation, threads, logging, output dirs"]
    initial["Initial geometry pass<br/>read metadata, telescope pointing, map extents, coadd geometry"]

    iter["Reduction iteration loop<br/>fruit-loop / learning iteration policy"]
    obsloop["Observation loop<br/>one rawobs at a time"]
    preflight["Observation preflight<br/>calibration, sample rate, diagnostics, telescope alignment"]
    tod["TOD processing pipeline<br/>RTC/PTC cleaning, masking, filtering, map accumulation"]
    obsraw["Raw observation products<br/>maps, noise products, diagnostics, logs"]
    coadd["Coadd accumulation<br/>combine observations when coadd is enabled"]

    rawcoadd["Raw coadd products<br/>normalized maps, noise products, diagnostics"]
    wiener["Map filtering / Wiener filtering<br/>observation or coadd level"]
    filtered["Filtered products<br/>filtered maps, diagnostics, source finding/fits"]
    finalize["Finalize iteration<br/>learning summaries, indexes, logs"]
    products["Final delivered products<br/>FITS maps, ECSV point products, netCDF diagnostics, indexes"]

    reduce_yaml --> tolteca --> lowlevel --> runtime
    raw --> runtime --> initial --> iter
    iter --> obsloop --> preflight --> tod --> obsraw
    obsraw --> coadd
    coadd --> rawcoadd --> wiener --> filtered --> finalize
    tod -->|"coadd disabled"| wiener
    finalize -->|"next fruit-loop iteration"| iter
    finalize --> products
```

The fruit-loop back edge intentionally returns to the reduction iteration loop,
not to TolTECA config generation or the initial geometry pass. In the current
control flow, each fruit-loop iteration repeats the per-observation reduction
loop, including observation input prep, fruit-loop map loading, TOD processing,
observation output/coadd accumulation, and iteration finalization. The initial
map geometry pass is outside that loop and is reused.

## Product Families by Reduction Type

```mermaid
flowchart LR
    common["Common raw inputs<br/>KIDs files, telescope file, APT/calibration, config"]

    pointing["Pointing"]
    oof["OOF / focus holography"]
    beammap["Beammap"]
    science["Science"]

    pointing_products["Pointing products<br/>raw array maps, point-source fit table, diagnostics"]
    oof_products["OOF products<br/>PSF/beam-shape maps, focus/holography diagnostics"]
    beammap_products["Beammap products<br/>per-detector maps, detector flags, beam parameters"]
    science_products["Science products<br/>raw/filtered coadds, science maps, noise/diagnostic sidecars"]

    common --> pointing --> pointing_products
    common --> oof --> oof_products
    common --> beammap --> beammap_products
    common --> science --> science_products
```

## Runtime Control Flow

```mermaid
sequenceDiagram
    participant TolTECA
    participant CitlaliCLI as Citlali CLI
    participant Runtime
    participant Geometry
    participant Iteration
    participant Observation
    participant Outputs

    TolTECA->>CitlaliCLI: call citlali with low-level YAML
    CitlaliCLI->>Runtime: load config, validate keys, configure threads/logging
    Runtime->>Geometry: initial pass over inputs
    Geometry-->>Runtime: map extents, map coords, coadd geometry
    Runtime->>Iteration: run reduction iterations
    loop fruit-loop / learning iterations
        Iteration->>Observation: run each observation
        Observation->>Observation: preflight, load fruit-loop maps if needed
        Observation->>Observation: run TOD pipeline
        Observation->>Outputs: write raw observation output or accumulate coadd
        Iteration->>Outputs: write raw/filtered coadd output if enabled
        Outputs-->>Iteration: finalize learning/index files
    end
    Runtime-->>TolTECA: completed reduction directory
```

## Where the Current Refactor Boundaries Land

The current structural split is converging on these conceptual boundaries:

```mermaid
flowchart TD
    entry["observation_execution.h<br/>compatibility facade"]
    pipeline["reduction_pipeline.h<br/>top-level entrypoint"]
    iterloop["reduction_iteration_loop.h<br/>fruit-loop iteration loop"]
    iter["reduction_iteration.h<br/>one iteration"]
    obsloop["reduction_observation_loop.h<br/>loop over raw observations"]
    obs["reduction_observation.h<br/>one observation wrapper"]
    obsinputs["reduction_observation_inputs.h<br/>per-observation preflight"]
    obspipe["reduction_observation_pipeline.h<br/>fruit-loop map load + TOD run + output"]

    helpers["focused helper headers<br/>calibration, telescope, timing, output layout, coadd/obs outputs"]

    entry --> pipeline --> iterloop --> iter --> obsloop --> obs
    obs --> obsinputs
    obs --> obspipe
    obsinputs --> helpers
    obspipe --> helpers
    iter --> helpers
```

## Validation Hook Points

```mermaid
flowchart TD
    compile["Unity compile<br/>build/citlali"]
    point["Pointing validation<br/>obs 152389 deterministic seq/thread=1"]
    science["Science validation<br/>individual obs and coadd products"]
    manifest["Manifest comparison<br/>structured FITS/netCDF/ECSV summaries"]
    logs["Log scan<br/>fatal/error patterns and completion tail"]

    compile --> point --> manifest
    compile --> science --> manifest
    point --> logs
    science --> logs
```

The most useful validation checkpoints for the refactor remain:

- compile on Unity after each batch of structural commits
- deterministic pointing reduction against the protected OG baseline
- science individual-observation product comparison
- science filtered/coadded comparison when Wiener-filter runtime is practical
- log scan for fatal/error patterns plus product manifest parity
