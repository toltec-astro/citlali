# a2000 OOF cold-M0 readiness — 2026-10-02

**Reuse the three historical a2000 maps; propose exactly three new successor maps and no R reductions.** The former unstarted a1100 proposal is superseded. POINT stays closed. Launch admission requires the exact source/binary gates, independent review and FREEZE.json described below; no observations are run by preparation.

## Frozen comparison

152385/0/1, 152386/0/1, 152387/0/1 separately; a2000 networks11/12 are the complete recorded band population. Each has977 raw channel identities (486+491),884 matched APT relations and853 finite positive flxscales (435+418). Those counts describe available calibration, not realized valid CAL exposure. Canonical matched APT-v2 parent148670, Tune152384/0/1, full-acquisition timing and historical recomputed TEL bytes are authenticated in the native metadata receipts. Maximum native cells7,704,622. Requested physical M2Z −3.566596717269,−1.566596717269,+0.433403282731mm is preserved; offsets −2/0/+2mm are relative to the middle, not best focus.

| Operand | Historical R reused | S cold M0 |
|---|---|---|
| Cleaning | Per-network standard PCA3 | Observed-entry ALS3, cap500, tolerance1e-5 |
| RTC/filter | Historical16Hz/32term, downsample2; exact effective config retained | Existing a2000 F2,235arcsec/s,80dB Kaiser; no new notch or source protection |
| Segmentation | Historical unforced timing | Existing explicit10s processing segments/context32; original clocks/gaps preserved |
| Calibration | Canonical148670 flxscale and extinction enabled | Same APT operands, native CAL/atmosphere; realized support measured after run |
| Map | 1″ AltAz JINC, validated weighting | Ordinary2″ FK5/equinox2000, current uniform coefficients, full SCI_SUPPORT |
| Feedback | Disabled | Zero; no POINT source gate |

R source c31a60a0, configuration, maps, provenance and actual inputs are bound by reference_audit.py. The historical directory relocation v2/oof/data → v1/oof/data is explicit; raw identities are authenticated against the canonical bundle by native preparation. Cleaning groups are networks; atmospheric/high-weight validation groups are arrays; map accumulation separates arrays. Other bands in the original job do not invalidate these a2000 maps. All recorded timing streams remain available to S. RTC despike/source protection is *not requested* in the realized R provenance even though enabled20″ protection appears inside its disabled despike configuration. PTC second-pass local flagging has20″ protection; weighting has30″ source exclusion. Do not describe those as an S gate. R/S is an operational comparison, not isolated PCA-versus-ALS or R truth.

## Operational consumer evidence

Recovered existing consumers: LMTOOF/python/OOFDriverFrontend.py and oof_ml_project/scripts/04_test_toltec.py → MultiBandDataLoader. Configuration is the installed a2000 LM defaults, corroborated by saved a2000 workflow examples, and the existing ML CLI default32×32 crop. A past152385 consumer command/config was not recovered; these rules are fixed before S results, not selected for agreement. Only the unchanged loader class is executed from the ML module; TensorFlow, inference and model generation are excluded.

| Consumer | Actual preprocessing | Samples per focus |
|---|---|---|
| LMTOOF | TolTECBeamMap3 weightcut0.5,100×100 pixel crop; existing option2 Gaussian *crop locator*; linear griddata to256² at4″; DataGrid ±12pixels inclusive | 600/625 finite and supported in all three;25 absent edge pixels |
| ML | Same reader option0,100² map-center crop; central32²; separate maximum normalization for each focus | 1024/1024 supported in all three |

Gaussian localization is existing crop preprocessing, never an OOF source acceptance gate or optical fit. The wrapper computes a cubic shift but does **not** assign it to the signal. LM uses no amplitude normalization/background subtraction: numeric mJy/beam survives, although its writer labels it `arbs`. ML divisors are126.99538,559.84589,214.69668mJy/beam; parent amplitudes and unnormalized windows remain saved. Weight threshold is0.5 times the existing sorted nonzero-weight statistic (~87.5th percentile); the wrapper zeros below-threshold samples. Separate masks confirm **zero finite-but-unsupported samples in these selected windows**. Parent support counts98806/99165/97276 are retained separately.

The current LM fixed-stride residual allocation is **not encountered as a defect for this invocation**:600+600+600 supports its3×600 allocation. The general unequal-count defect remains; the unchanged guard still rejects genuinely unequal post-preprocessing counts. No crop was invented to make counts equal. The256² FITS writer usesCRPIX128.5, so selected sample centers are−46..+50″; the current C++ reader reports−48..+48″ (2″ half-cell discrepancy). The grid transpose and actual C++ roundtrip are recorded. This is a retained optical-interface limitation, not a reason to prevent native M0 maps. No optical fitting, model inference or sibling modification is performed.

S stays in supported FK5. No established sample-level AltAz attachment is available here, so S is **not** relabeled or passed to these AltAz consumers. After completion, characterize S on its native tangent axes with the same angular spans (100″ LM fitting span /32″ ML span), preserving source-centered versus map-centered distinctions and all original maps. A16×16 selection is the32″ counterpart on2″ pixels; copy-and-crop32 S pixels would be64″ and is forbidden. Do not call a native-frame diagnostic an operational AltAz solver input. No paired R−S pixel residual, exact orientation agreement or optical result is admitted without physical frame alignment. Original/recomputed TEL hashes and differing RA/DEC/ActParAng remain explicit; unknown historical rationale alone does not demonstrate invalid native coordinates.

## Resources, checks and launch

Shared prior derived bytes20,087,839,565, including previous OOF preparation. Conservative active S bound5,042,205,088bytes; retained cap2GB per role;200MB metadata/report reserve. Three serial S roles reserve **29,330,044,653bytes peak**, below30GB; final bound26,287,839,565bytes. Separate2GB free reserve,2network workers,1numeric thread each. This is disk accounting, not a RAM prediction. Runtime/CPU and sampled process-tree peakRSS are recorded by the existing launcher. Scratch release only removes completed-role conditioned/centered duplicates after preservation hashing; maps, CAL/support, original PTC/filtered RTC/bases/receipts remain. No predecessor cleanup is needed. Stop if actual retention exceeds its bound.

`preflight.py` rebuilds the exact committed candidate, runs CTest, baseline/POINT/OOF/config/native-contract gates, three metadata preparations, five negative identity/policy probes, operational consumer diagnostics, reference authentication and launch-policy fixtures. `freeze.py` requires those receipts plus a passing exact-source independent review. `run_campaign.py --check-only` verifies the actual launch admission without creating a launch marker or scientific output. Frozen receipt locations are under:
`~/work_toltec/local_data/citlali-validation/successor-oof-a2000-m0-20261001/`.

See RESUME.md for the single owner-run foreground command. Any role failure stops the batch, preserves evidence and consumes that role's start. No retry or unused R slot becomes a replacement. No scientific decision is needed for this bounded native M0 run; AltAz attachment, optical conventions/half-cell issue, model/bandpass and inference are deferred. The quality conclusion remains **not established until S is run and assessed**.
