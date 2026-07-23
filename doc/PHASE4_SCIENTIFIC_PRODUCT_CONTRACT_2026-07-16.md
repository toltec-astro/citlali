# Phase 4 Scientific Product Contract

## Purpose

Citlali validation now answers three different product questions separately:

1. Did the reduction finish cleanly and publish valid provenance?
2. Did it deliver the files and scientific structure requested by its effective
   low-level configuration?
3. Are the numerical contents acceptably equivalent to an accepted snapshot?

The reduction audit answers the first question,
`validate_product_contract.py` answers the second, and the established
mode-specific comparators answer the third. None substitutes for the others.

The machine-readable registry is `validation/product_contracts.json`. It is
versioned with the active validation profiles. An intentional product or schema
change creates a successor contract and profile rather than weakening an
accepted contract.

## Request-To-Delivery Rule

The generated `citlali_o*.yaml` file is the authority for output switches. A
configuration-controlled entry contains a machine-readable `required_when`
rule. When the rule evaluates true, the expected product must exist and satisfy
its structural checks. When it evaluates false, that product must be absent.

The active contracts currently evaluate explicit switches for:

- point filtered maps, fit tables, and filtered map diagnostics;
- point raw- and processed-timestream exports;
- science coadded and coadded-filtered products;
- Beammap split-by-flag maps; and
- Beammap source-crossing detector-TOD and associated diagnostics.

Successor Beammap contract `phase4.1-beammap-products-v2` also requires the
atomic reduction restart checkpoint exactly when fruit loops are enabled and
forbids it when they are disabled. Its inexpensive NetCDF check requires the
checkpoint schema identity, observation dimension, and core iteration,
configuration, observation, and learned-state count variables.

Products without an independent output switch are not labeled conditional just
because their writers are diagnostic in nature. Map histograms, map PSDs,
detector statistics, and core RTC/PTC diagnostics are required companions of
their parent output in the accepted profiles. Profiling and bounded learning
CSV records remain optional diagnostics because their absence does not make the
scientific reduction incomplete.

This design deliberately reads the merged low-level YAML that Citlali actually
received. The validation profile separately requires that YAML to match the
accepted profile, so TolTECA's ordered `NN*.yaml` overlays remain part of the
validated workflow.

## Contract Contents

Each family records:

- scientific identity;
- coordinate frame;
- axes and indexing convention;
- units policy;
- missing/non-finite policy; and
- failure policy.

The structural checker verifies the product inventory and selected inexpensive
format invariants without loading large FITS pixel arrays. Checks include FITS
extension names, WCS axes and units, NetCDF dimensions and variables, ECSV
columns and row counts, file non-emptiness, and complete one-time
classification of every FITS, NetCDF, ECSV, and CSV product.

The four active contracts cover the accepted snapshots completely:

| Mode | Snapshot | Classified products | Key map frame |
| --- | --- | ---: | --- |
| Point | `redu66` | 21/21 | AltAz offsets in arcsec |
| OOF | `redu02` | 31/31 | AltAz offsets in arcsec |
| Science | `redu31` | 28/28 | J2000 equatorial TAN in deg |
| Beammap | `redu06` | 13/13 | Per-detector AltAz offsets |

Only Stokes I is under the current validated contract. Polarimetry is not
declared unsupported forever, but enabled polarimetry remains outside the
validated capability set.

The Phase 4.1 Beammap smoke at `cfae989ce` intentionally adds one required
operational product. It passes `phase4.1-beammap-products-v2` with 14/14
products classified, including a
`citlali-reduction-restart-checkpoint-v1` checkpoint. This V2 contract is
candidate evidence for a successor profile; it does not replace or weaken the
active historical Beammap V1 profile.

## Failure Policy

Missing or malformed required products fail validation. Missing requested
configuration-controlled products also fail, as do products emitted when their
explicit switch is disabled. Citlali's production policy remains that required
or enabled output write failures fail the reduction. Optional diagnostics may
be absent, but they must not masquerade as complete requested output.

## Known Schema Debt

The contract records current behavior rather than assigning scientific meaning
that the files do not yet express:

- several diagnostic NetCDF variables lack complete units attributes;
- pointing and Beammap ECSV units live in table metadata rather than column
  unit fields; and
- several diagnostic NetCDF families lack standardized fill-value metadata.

Flags, finite-value conventions, existing metadata, and numerical comparison
remain authoritative for this validation epoch. Improving those schemas is a
future intentional product change with a successor contract, migration note,
and scientific-owner review.

## Usage

The normal entry point runs the contract automatically as one of four gates:

```bash
$HOME/tolteca/bin/python tools/baseline/validate_reduction.py \
  /path/to/candidate/reduNN \
  --profile phase4-science-152390-152392-v1
```

For focused diagnosis:

```bash
$HOME/tolteca/bin/python tools/baseline/validate_product_contract.py \
  /path/to/candidate/reduNN \
  --contract phase4-science-products-v1
```

The focused report names the merged low-level config, each product family,
whether a conditional family was requested, its matches, and structural
errors.
