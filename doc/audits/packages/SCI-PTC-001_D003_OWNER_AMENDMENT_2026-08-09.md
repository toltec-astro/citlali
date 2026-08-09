# SCI-PTC-001 D003 owner amendment — 2026-08-09

Record ID: `SCI-PTC-001-D003-OWNER-AMENDMENT-2026-08-09`

Status: owner-approved with amendment; documentation and coordination only

## Exact authority and preserved identities

The project owner approves this amendment for `SCI-PTC-001` at governing
application SHA `46ad23888a40f5102cdfd50c06e49a549bdf8a20`.

The completed audit remains immutable at final commit
`01ee247461d6c19bc4db81ccac4fec21af162c88`, parent/core
`66e8d6f98c3e22da74de4eea84e568a0b4cc6310`, and tree
`e6685c920ff37f1d4e51d27ecf23b73ac16087b5`. Its independent-core and final
report SHA-256 digests remain, respectively,
`82c0835f51ea9b1fa8a37489f289be89a8018a0b2700e84b1e25c2e4d2a013c2`
and `c46a15c142d0938baf9576d84a19332e0d46b34852b4d59c0029ba00ac62d7e6`.

This successor does not change the frozen audit artifacts, the coordinator
brief, the prior D001 authority, or
[`SCI-PTC-001_D002_OWNER_AMENDMENT_2026-08-08.md`](SCI-PTC-001_D002_OWNER_AMENDMENT_2026-08-08.md).

## Approved D003 response contract

1. The stored PTC kernel is the collaboration's
   `estimated_map_center_point_source_response` of the instrument after the
   declared RTC/PTC/analysis chain. It must bind the exact band,
   analysis/configuration, detector/mask/selection state, other recorded
   realization state, and the identity and status of every applicable
   upstream response. It is a computed/published estimate, not an unavailable
   response class. Its calibration, validation, and uncertainty domain must be
   stated honestly.
2. Response status distinguishes at least:

   - `computed_published`;
   - `not_computed_or_not_requested_for_this_product`;
   - `invalid`; and
   - `unavailable`.

   Absence from an ordinary product does not establish physical impossibility
   or scientific unknowability. `unavailable` is reserved for a response that
   cannot be supported for the declared product/domain with admitted
   authority or evidence; `invalid` identifies a response product that fails
   its own admission or validity contract.
3. Longer-wavelength and shorter-wavelength bands and distinct map/reduction
   modes are expected to have different transfer functions. No cross-band or
   cross-mode equality is implied or may be substituted for measurement.
4. The current map-center point-source estimate does not by itself establish
   off-center or spatially varying response, extended-source response,
   arbitrary morphology or amplitude response, or a universal cross-band or
   cross-mode transfer function. An optional measured response-family product
   may extend the existing estimate only under the separately scoped
   [`SCI-PTC-001_OPTIONAL_TRANSFER_CHARACTERIZATION_PLAN_2026-08-09.md`](SCI-PTC-001_OPTIONAL_TRANSFER_CHARACTERIZATION_PLAN_2026-08-09.md).
   It is not a replacement for or denial of the current kernel. Any broader
   measured claim is conditional on its exact declared band, mode,
   configuration, masks, detector selection, iteration/pass state, upstream
   response, sampled domain, and evidence.

## F005 disposition

`SCI-PTC-001-F005` remains a P1 contract gap until implemented and independently
re-audited. For ordinary PTC products, it may close without executing the
optional characterization if the product:

- publishes the stored kernel with the exact identity
  `estimated_map_center_point_source_response` and preserves the word
  `estimated`;
- binds that kernel to exact band, analysis/configuration,
  detector/mask/selection, parent, upstream, and realization state;
- states its domain and calibration, validation, and uncertainty status
  honestly;
- publishes an honest typed status for each response class promised by the
  product;
- uses `not_computed_or_not_requested_for_this_product` only for stronger or
  unmeasured extensions and `unavailable` only for genuinely unavailable
  classes;
- never promotes the map-center point-source estimate to an off-center,
  spatially varying, extended-source, arbitrary morphology/amplitude, or
  universal cross-band/cross-mode transfer claim.

A stronger measured transfer claim requires the optional evidence for its
exact declared validity domain. The optional study is not an ordinary PTC
repair or operation gate.

## Supersession and non-authorization

This record supersedes only the categorical coordination recommendation to
mark every stronger response class unavailable and the corresponding ordinary
F005 closure wording. The frozen audit and prior briefs remain immutable
historical evidence.

Package axes remain `proposed`, `nonconformant`, `in_progress`, and
`existing_use_only`, with verdict `amend`. D004--D006 remain unresolved. This
decision does not authorize PTC repair, the optional study, MAP or BEAM work,
Unity, reductions, external contact, re-audit, downstream launch, production
change, merge, push, or application/test/configuration edits.
