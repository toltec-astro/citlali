# POINT 152389 JINC — current numerical binding

Owner responses after preparation supply the kernel values below and the exact
signal-accuracy requirement. This supersedes the missing-profile status in the
sealed preparation bundle; that historical evidence is unchanged. No JINC
observation execution or numerical certificate has yet occurred.

## Kernel selection

| Array | Exact angular scale s (radians) | a | b | c | rmax | nsub |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| a1100 | (1.1/1000)/45 | 1.1 | 1.67 | 2 | 3 | 16 |
| a1400 | (1.4/1000)/45 | 1.1 | 2.17 | 2 | 3 | 16 |
| a2000 | (2.0/1000)/45 | 1.1 | 3.17 | 2 | pending | pending |

Only the two blank a2000 fields still need owner input. The earlier question
whether a2000 also uses3 and16 remains pending; the numerical-requirement
message does not answer it. Scales are kernel scales, not calibration beam FWHM.
No physical-aperture interpretation is inferred from the scale formula.

## Owner signal-accuracy requirement

For every numerically accepted JINC signal pixel:

    abs(m_hat - m_star) <= 1e-6 * P_a + 1e-6 * abs(m_hat)

m_star is the exact-real-arithmetic selected discrete operator value. Both
signal terms and P_a are in mJy/nominal beam. P_a is the absolute original
REF T10 fitted Gaussian amplitude above its fitted constant, frozen before
JINC execution for all comparisons in that array. Neither central-pixel value
nor amplitude-plus-constant nor a new JINC fit replaces it.

| Array | Frozen P_a (mJy/nominal beam) | Absolute term 1e-6 P_a (mJy/nominal beam; rounded display) |
| --- | ---: | ---: |
| a1100 | 937.4216858608636 | 0.000937421685861 |
| a1400 | 664.7340776999814 | 0.000664734077700 |
| a2000 | 813.6176996671275 | 0.000813617699667 |

The machine-readable owner-numerical-binding.json records full saved values,
binary64 encodings, authenticated original measurement/map hashes, and agreement
with the C+D archived original reference. No new fit was performed.

This resolves the requested acceptance observable, units, per-pixel inequality,
reference amplitude, constants and frozen lifecycle. Implementation must now
establish a conservative error bound for that discrete target, covering kernel
and coordinate/phase implementation, accumulation, signed-denominator error and
final normalization. Near cancellation is not automatically valid merely because
C is nonzero. A pixel may be accepted only if its full bound meets the inequality;
absence of a certificate is not evidence of meeting it.

Intentional kernel/phase quantization is part of the selected discrete target;
scientific response and continuous-profile phase sensitivity remain separate.
Numerical mistakes in realizing that discrete target are covered by the bound.
The adequacy certificate and provenance mechanism remain engineering work, not
another missing owner choice of tolerance. No empirical oracle comparison alone
is asserted to prove an exact-real bound over every accepted observation pixel.

## Preserved authorities and stop

SCI-JINC v0.1/r0.3 frozen at a9f43877e01a661db13bd85b2e7f34ea5ac82fb7 remains
the analytic/discrete authority: exact first J1 root, accepted center/phase rules,
ceil square support and signed normalization. Uniform gamma1 JINC permission
remains granted by SCI-PTC_TO_SCI-JINC_BOUNDARY r0.4 (UFC-OD-009); it has no
inverse-variance meaning. Current owner order alone authorizes the diagnostic
rejoined-m9 parent adapter. Original15-state C+D baseline, saved means,
measurement method and all feedback exclusions remain unchanged.

Next: resolve a2000's two fields and implement/validate the certificate before
observation execution. No profile threshold is to be tuned to the observed
response, and no new JINC peak may relax the acceptance criterion.
