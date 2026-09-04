# OOF Attachment Envelope

Status: **future boundary envelope; no OOF implementation claim**

OOF analysis is not a product or edge in the accepted 17-product/32-edge
map-space graph.  This file records the minimum discipline for a future,
separately authorized attachment without adding a graph ID or claiming that a
current OOF route exists.

## Candidate upstream products

A future OOF boundary may consider only explicitly selected immutable
MSP-P004, MSP-P005, MSP-P007, MSP-P009, or MSP-P014 products after their
respective routes are conformant and after an OOF scientific owner identifies
the exact admissible role.  Shape similarity, FITS keywords, map names, or
legacy pointing tables are not sufficient.

## Minimum boundary record

The future record must bind exact parent/product/profile/source versions;
signal identity and units; WCS/frame; beam/response class; covariance and
formal-uncertainty state; support and missingness; processed-source-shape
versus intrinsic-optics meaning; per-array failure behavior; lifecycle; named
use; and owner action.  Any correction or telescope-state update must be a new
owner-authorized derived proposition, never an interpretation of displacement
alone.

## Required negative rules

- POINT displacement is not automatically a pointing correction.
- Processed fit width is not automatically an intrinsic beam or OOF state.
- Dynamic range, fit S/N, NOI standardized signal, and significance are not
  aliases.
- Missing response/covariance/formal-error information is not zero and cannot
  be rescued from field names or legacy validation.
- Partial per-array success is not whole-observation success.

## Present disposition

No OOF source, implementation, configuration, test, validation, performance,
readiness, production, or Unity conclusion is part of this packet.
