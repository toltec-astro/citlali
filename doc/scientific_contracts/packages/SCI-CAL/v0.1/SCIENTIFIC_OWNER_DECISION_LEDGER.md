# SCI-CAL Scientific Owner Decision Ledger

Status: Q01--Q09 scientifically dispositioned for the SCI-CAL v0.1
r0.5/r0.4 revision; numerical products remain unavailable where stated

Date: 2026-08-20

State vocabulary: open, decided, deferred, superseded.

| Decision ID | Owning scientific authority | Status | Evidence or decision required | Blocked claim or product | Resolution authority | Resolution date | Affected documents |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SCI-CAL-OWNER-Q01 | Grant Wilson, ordinary-xs scientific owner | decided | Dimensionless KID `delta_f/f_res`; positive with optical power; no DC or baseline operation; operational Tune domain recorded | Physical input meaning resolved; wrong-Tune/high-noise states remain explicit validity/quality conditions | `SCIENTIFIC_OWNER_DECISIONS_R0.5.md` | 2026-08-20 | Science rationale; engineering contract; validation plan |
| SCI-CAL-OWNER-Q02 | Grant Wilson, SCI-CAL/PTC boundary owner | decided | CAL multiplicative operation precedes PTC; PTC owns DC/common-mode removal | Local and adjacent order resolved; realized downstream response remains producer-owned | `SCIENTIFIC_OWNER_DECISIONS_R0.5.md` | 2026-08-20 | Science rationale; engineering contract; PTC contract |
| SCI-CAL-OWNER-Q03 | SCI-BEAM authority adopted by Grant Wilson | decided | SCI-BEAM owns source model, atmosphere, fit, nominal beam, factor direction, and lineage | Source-factor meaning resolved by normative reference; present numerical uncertainty may remain unavailable | SCI-BEAM plus `SCIENTIFIC_OWNER_DECISIONS_R0.5.md` | 2026-08-20 | Science rationale; Beammap/source-APT contract; engineering contract |
| SCI-CAL-OWNER-Q04 | Grant Wilson; TolProj transformation boundary | decided | Closest accepted Beammap APT by default; optional scientist-directed per-array pointing-source photometric child rescale | Transfer policy resolved; universal accuracy and the future rescale-uncertainty mechanism remain unclaimed | `SCIENTIFIC_OWNER_DECISIONS_R0.5.md` | 2026-08-20 | Science rationale; TolProj contract; validation plan |
| SCI-CAL-OWNER-Q05 | Grant Wilson, photometric-convention owner | decided | 272/214/150 GHz centers; source-dependent spectrum; one array-average passband; downstream target color correction | Reference convention resolved; detector/network passband variation and its uncertainty remain unavailable | `SCIENTIFIC_OWNER_DECISIONS_R0.5.md` | 2026-08-20 | Science rationale; engineering contract; atmosphere contract |
| SCI-CAL-OWNER-Q06 | Grant Wilson plus recovered content-bound atmosphere authority | decided | LMT WVR in `tel*.nc`, time interpolation, exact AM12 fixed-DJF25 nodes/operator, broadband integration, support and digests | Numeric operator authority resolved; observational atmosphere truth and WVR uncertainty remain unvalidated/unavailable | atmosphere commit `7156881bd...` plus `SCIENTIFIC_OWNER_DECISIONS_R0.5.md` | 2026-08-20 | Science rationale; engineering contract; atmosphere authority; validation plan |
| SCI-CAL-OWNER-Q07 | Grant Wilson, SCI-CAL scientific owner | decided | Experience-based flexible limits; one CAL-owned observation class; no splitting; 0.025 excursion tolerance; owner changes policy | Policy rationale and ownership resolved; tolerance does not extend numeric operator support | `SCIENTIFIC_OWNER_DECISIONS_R0.5.md` | 2026-08-20 | Science rationale; engineering contract; segment-policy authority |
| SCI-CAL-OWNER-Q08 | Grant Wilson coordinating CAL/BEAM/TolProj uncertainty producers | decided | Noise downstream; systematic CAL error intended; current BEAM, WVR, and TolProj-rescale uncertainties unavailable; array/observation correlation scopes declared | Total calibrated uncertainty and total significance remain unavailable pending named producer mechanisms | `SCIENTIFIC_OWNER_DECISIONS_R0.5.md` | 2026-08-20 | Science rationale; engineering contract; MAP/NOI/BEAM/TolProj contracts |
| SCI-CAL-OWNER-Q09 | Grant Wilson, scientific acceptance authority | decided | Concrete Beammap closure and associated-pointing transfer workflow; per-array honest reporting; no arbitrary matrix/sample minimum | Validation procedure resolved; achieved acceptance remains unavailable until owner reviews executed evidence | `SCIENTIFIC_OWNER_DECISIONS_R0.5.md` | 2026-08-20 | Science rationale; engineering contract; validation plan; acceptance record |

The earlier SCI-CAL-OWNER-Q001 atmosphere-content question is superseded by
SCI-CAL-OWNER-Q06, which retains all of its required fields and adds an
explicit ownership and absence-classification decision.

This ledger separates a decided scientific policy from availability of a
numerical producer product or achieved validation result. Q08 and Q09 are
decided even though total uncertainty and achieved scientific acceptance are
not yet available.

Engineering conformance r0.4 was checked against this ledger on 2026-08-20.
It carries Q01--Q09 as decided authority with claim-specific unavailable
products and an explicit owner-acceptance gate for achieved performance.
