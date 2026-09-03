# Schloerb JINC / OTF Mapmaking Memo — Reference Intake

Status: owner-designated authoritative LMT method reference; exact role and
author-packet admission awaiting final Stage A byte approval

Received: `2026-08-28`

## Original Source Record

| Field | Value |
| --- | --- |
| Repository path | `Schloerb_JINC_memo_v1.1.pdf` |
| Document title | *Spectral Line Data Reduction at the Large Millimeter Telescope* |
| Author | F. Peter Schloerb, University of Massachusetts Amherst |
| Document date | `2019-07-23` |
| Change history | Initial draft `2019-04-30`; expanded OTF mapping section and Appendix C `2019-07-23` |
| File metadata author | `F. Peter Schloerb` |
| File creation timestamp | `2019-09-16 16:03:10 EDT` |
| PDF properties | PDF 1.4; 42 letter-size pages; unencrypted; no forms or JavaScript |
| Exact byte size | `1563601` bytes |
| SHA-256 | `835fb02e842c9109c2c7ad3f03288882dfac283e63bfcd0f818c7d5379e7e5cd` |
| Supplied by | Grant Wilson, SCI-JINC scientific owner |
| Provenance statement | Supplied as an authoritative memo from an LMT scientist on JINC mapmaking, also called on-the-fly mapmaking |
| Distribution/access constraints | Not supplied. Preserve as repository-local scientific-contract source material until the owner states otherwise. |

The original PDF is immutable intake evidence. It must not be annotated,
normalized, overwritten or silently replaced. A replacement or new memo
revision receives a new source record and digest.

## Scientific Classification And Scope

Classification: **reusable scientific reference**, owner-designated as
authoritative for the generic LMT OTF/JINC analytic method and its physical
motivation.

Adoptable scientific content is limited to:

- the aperture/spatial-frequency motivation in Sections 5.1.2--5.1.4;
- the peak-normalized first-JINC construction in Equations 6--7;
- the practical two-JINC plus generalized-exponential filter family in
  Equation 9;
- meanings and ordering of `a`, `b`, `c`, and `RMAX` in that family;
- the dimensionless radial coordinate `r'=r/(lambda/D)` used by the memo;
- the second JINC's first zero at `RMAX`; and
- the qualitative tradeoff among sidelobe suppression, in-aperture spatial
  response and response outside the nominal aperture.

The memo is geared toward 3-mm LMT spectroscopic receivers and is **not**
authority for:

- TolTEC `a1100`, `a1400`, or `a2000` effective wavelengths, beam scales,
  array scales or WCS pixel scales;
- TolTEC-specific values or defaults for `a`, `b`, `c`, `r_max`, or
  `subpixel_n`;
- an optimum TolTEC parameter objective or three-band optimization result;
- TolTEC coefficient, covariance, response-companion, calibration, filtering,
  significance or product semantics;
- the memo's radial cutoff as a SCI-JINC support rule;
- pixel-area integration, subpixel phase, finite-map edges or atomic product
  lifecycle; or
- achieved TolTEC response, validation, performance, readiness or production
  behavior.

The memo's FCRAO nominal values (`a=1.1`, `b=4.75`, `c=2.0`, `RMAX=3`),
86-GHz/3.4-mm SEQUOIA simulation, figures, Monte Carlo outcomes and nearest-
neighbor comparisons are contextual/historical scientific evidence only.
They must not be transplanted into TolTEC authority.

## Sanitized Author Excerpt

`Schloerb_JINC_memo_v1.1_METHOD_EXCERPT_pp15-19.pdf` is a page-exact extraction
of original PDF pages 15--19, covering Sections 5.1.1--5.1.6 and the start of
5.2. Its SHA-256 is
`a065843b4b83c21aabb25233c588817e998773a5d6a7bd389874eab50c9a88e9`.

The excerpt was reopened, text-checked, rendered with Poppler and visually
compared page by page with the corresponding original pages. It changes no
page content. Its metadata identifies the source digest and derivative role.

The proposed implementation-blind packet admits the excerpt only when paired
with the exact package-level author reference cover. The full 42-page original
remains recovery provenance and is not an author input, preventing unrelated
instrument-specific reduction material and Appendix C parameter simulations
from entering Stage B.

## Disposition

- **Adopt** the generic analytic OTF/JINC method after collision-free notation
  normalization and application of later owner-approved JINC supersessions.
- **Cite** the exact method excerpt in the author packet after final owner byte
  approval.
- **Supersede** the memo's radial-cutoff support with square-cache support and
  its continuous/pixel-agnostic sampling with the approved point-phase rule.
- **Abstract** the spatial-response tradeoff as motivation for a future
  separately authorized TolTEC parameter study.
- **Defer** all TolTEC array-scale and parameter selection.
- **Exclude** unrelated spectral-line reduction content, 3-mm numerical
  examples, Appendix C simulation results and all implementation implications
  from Stage B authorship.
