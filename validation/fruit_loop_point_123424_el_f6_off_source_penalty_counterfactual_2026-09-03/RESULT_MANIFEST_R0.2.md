# FRUIT EL-F6 interpretation-result manifest r0.2

Generated after the scientific owner approved correction of the EL-F6
interpretation on 2026-09-03.

Revision r0.2 is a documentation and diagnostic-visualization revision. It
does not replace the registered test or the preserved r0.1 execution evidence.
No reduction was rerun, and the numerical result, intervention, metrics,
thresholds, and registered classification are unchanged.

## New repository artifacts

| Artifact | SHA-256 |
|---|---|
| `README.md` | `28aee5a221ef2fc6d36de42b1eb3a5ff83208e530fad2fc77b18d30af17aaf83` |
| `EXECUTION_RESULT_R0.2.md` | `954f72d2466868163339ac44f5284c6a8c3826a0a88c2ae5dd72dadf293a2c51` |
| `INTERPRETATION_METRICS_R0.2.json` | `92fbb766e5a1301245c86c229afd8188b002f76ab959723608f533a431749ee9` |
| `RESPONSE_DECOMPOSITION_R0.2.png` | `31735736da14d43115805bd59459d21e6bcf824f42d44c295b11e5570cf1f0b4` |
| scientific-owner interpretation direction | `d1b445f5ff8e8f34e55f120f675249979071392c27e0c36facd9bd89323a3287` |

The owner-direction artifact is
`doc/scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane/SCIENTIFIC_OWNER_EL_F6_INTERPRETATION_DIRECTION_2026-09-03.md`.

## Preserved r0.1 numerical evidence

| Artifact | SHA-256 |
|---|---|
| `EXECUTION_RESULT_R0.1.md` | `adfbb5ee0136f1ad92e3fcc780873930213c9f0c32e0150d337855a0d05b27c4` |
| `COUNTERFACTUAL_METRICS_R0.1.csv` | `0b1bdd1157387a0b4d37c4d58070d041ba61e792922af109f1309f21c53cdc45` |
| `COUNTERFACTUAL_RESULT_R0.1.json` | `64d585ab1d0249ddc76839c650dbb95625593a8e2c8369973055168214895b74` |

All other registered and execution artifacts retain the identities in
[`RESULT_MANIFEST_R0.1.md`](RESULT_MANIFEST_R0.1.md).

## External evidence used for the interpretation check

| Artifact | SHA-256 |
|---|---|
| EL-F5 control iteration-5 a1400 FITS | `0473c0073944bc3ccb3cdb5486f0a637c91ce0f57b31ea0c2ca336f27d179478` |
| EL-F5 off-source injected iteration-5 a1400 FITS | `8dca60369b279d4a54160420544c7d7f016b3b879cdd3e91045c7032cb3c2401` |
| EL-F6 no-UID-4460 iteration-5 a1400 FITS | `5d8594aa566d3bd30f00e4ca3beecef69e3c69f26503f57ce4f0c7834670b0cd` |
| EL-F5 off-source injected iteration-5 pointing-fit ECSV | `ccd6c96f5876aaf4e23751a024b748a75a1b3998e80b3b076fa546356ce75be1` |
| matched observation-123424 APT | `16389e5e58b76d39ef7fcedd3888c662e96db592bc3a8561530379e51c435626` |
| recomputed observation-123424 telescope product | `bfb95deda33e8f5a3e86a0990db2698bddb9ed7c39c6284733900d91fd99defe` |

The region measurements use finite pixels on the common frozen FITS grid.
The source and Neptune regions are circles of radius 20 arcsec at their
declared/fitted FITS-world positions. The off-source statistic uses the
40--120 arcsec injection-centered annulus after excluding a 25-arcsec circle
around the fitted Neptune position. The diagnostic image flips array columns
for display because the FITS `AZOFFSET` axis has `CDELT1=-1 arcsec`; source
calculations remain in FITS-world coordinates.
