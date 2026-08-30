# SCI-FLT-FIXED v0.1 Normalization, Unit, Beam, And Low-Pass Table

Status: sanitized Stage A author candidate awaiting exact-byte owner approval

| Fact | Required decision/representation | Prohibited inference |
| --- | --- | --- |
| Kernel normalization | Exactly one declared convention: unit sum/DC gain, unit peak, unit angular integral, unit L2 norm, or another exact convention | No default from family/name |
| DC gain | Exact value on the declared complete-support domain | Unit sum does not establish point-source peak |
| Output units | Computed from parent units and operator coefficient units | Numerical unit preservation does not create calibration or a new beam convention |
| Parent nominal beam | Retained as originating response/calibration identity | Do not relabel `mJy` per filtered beam |
| Filter-composed response | Exact `L_Theta R_parent` on compatible full-footprint rows, or unavailable | Kernel is not automatically the complete source response/PSF |
| Peak response | Separately typed exact quantity or unavailable | No preserved peak from DC normalization |
| Signed integral | Separately typed with pixel-area convention | Not interchangeable with peak or effective solid angle |
| Effective beam solid angle | Externally authorized response-derived quantity or unavailable | No target beam from a Gaussian/kernel name |
| Integrated/extended-source fidelity | Requires external BEAM/CAL/response authority and exact normalization | No automatic flux fidelity |
| Low-pass claim | Qualified subtype only with complete domain/metric, DC gain, passband, transition, stopband/attenuation, phase, anisotropy, finite-grid/edge, kernel, normalization, and parameter provenance | “Smoothing,” a width, or a friendly name is insufficient |
| Missing low-pass facts | Operator may remain fixed convolution; low-pass claim unavailable | No partial low-pass label |
| Parameter source | Externally resolved, immutable, and independent of the parent/member application | No data-derived cutoff/kernel or automatic selection |

CAL retains absolute calibration, passband/color correction, and calibration
covariance. External BEAM/source authority owns the physical evidence for a
target-beam or source-response claim. SCI-FLT-FIXED owns only the exact local
transformation and its composed response state.
