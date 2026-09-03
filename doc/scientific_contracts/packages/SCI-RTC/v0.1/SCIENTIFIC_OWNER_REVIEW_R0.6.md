# SCI-RTC v0.1 scientific-owner review r0.6

Status: binding bounded-correction authority, confirmed `2026-08-18`

Source SHA-256:
`2a4163d1ed0775e83ef981573d1a3a1f65fe2d89860bd92b0ad456e61fa8e266`

## Review disposition

The supplied review accepts the paired-coordinate, leakage-diagnostic, and
level-shift expansion in substance but blocks freezing r0.5. It identifies two
false owner attributions, one unresolved $r$-output boundary, and bounded
formal/scientific corrections. The owner subsequently confirmed the three
recommended architecture decisions below.

## Binding r0.6 decisions

1. Atmospheric templates are diagnostic evidence only in RTC v0.1. They may
   support leakage, Tune/readout-health, shift/artifact, selection, or review
   products but are not subtracted from science $x$. Numerical common-mode or
   atmospheric removal requires separate PTC or successor authority.
2. Level-shift learning uses the original aligned pair while isolated-spike
   candidates are excluded, masked, or robustly downweighted. Donor replacement
   begins only after shift boundaries are resolved, remains within stable
   segments, and never crosses an unresolved or accepted boundary.
3. Conditioned $x$ is the required RTC numerical output. The exact raw $r$
   parent and every causal diagnostic, selector, segmentation, and resolved-plan
   use remain in the atomic bundle. Any conditioned $r$ product requires a
   separate channel-specific operator, response, support, unit, validity,
   uncertainty, role, and provenance.

The r0.5 `RTC-SCI-D004` and `RTC-SCI-D005` approval attributions are withdrawn.
They are not silently reinterpreted.

## Required bounded corrections

- Use a general Tune-dependent $\mathcal T_{d,\zeta}(I,Q)$ mapping; affine or
  Jacobian forms are local-domain representations only.
- Reserve $\zeta$ for Tune/mapping identity and use $\epsilon$ for leakage,
  with a distinct residual symbol.
- Make donor replacement and the required numerical RTC operator explicitly
  $x$-only; use “raw aligned paired parent” for representative occurrence.
- Require role-specific plateau support, pre/post-shift leakage comparison or
  explicit unavailability, and a block on additive-only interpretation after a
  material response change.
- State atmospheric-estimator bias from noisy/shared coordinates and
  self-inclusion without selecting a numerical estimator.
- Add the missing scientific falsifiers for ideal/rotated mappings, estimator
  bias, source/drift discrimination, PSD/notch contamination, pre/post response,
  and role-specific short-plateau disposition.
- Replace “IQ-map parent” with the exact aligned paired `xs`/`rs` parent.

This review authorizes no implementation inspection, numerical default,
conformity claim, validation result, Tune-readiness claim, or production claim.
