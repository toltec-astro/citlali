# SCI-FLT-MATCHED v0.1 r0.5 PDF Visual-QA and Metadata Report

Date: `2026-09-01`

Result: `PASS`

Scope: render, metadata, and layout quality only; not scientific approval,
implementation conformity, response/covariance fidelity, observational
validation, performance, readiness, production suitability, scientific
freeze, or Unity activity

## Artifacts inspected

| PDF | Pages | SHA-256 |
| --- | ---: | --- |
| `pdf/SCI-FLT-MATCHED-SCIENTIFIC-RATIONALE-v0.1.pdf` | 42 | `95007fb16de1eeb5a6efaa77e7af8b64981e1d5ff572e9c53e5254a0b7b81876` |
| `pdf/SCI-FLT-MATCHED-ENGINEERING-CONFORMANCE-v0.1.pdf` | 38 | `cee6476e664af1c47c89e06fc95b279f0cfeb6cea8e0553f0d9343047b338496` |

Both PDFs reopen with Poppler and `pypdf`. Every page is unrotated US Letter
(`612 x 792` points). The files are unencrypted and contain no forms or
JavaScript.

The PDF title metadata identifies the respective r0.5 Scientific Rationale and
Contract and Engineering Conformance Specification. Both subject fields state
that these are final targeted type, lifecycle, covariance-role, and
owner-disposition closure drafts. Provisional human-title rendering is
explicitly paired with `optimality_status=not_claimed`; it is not an owner
title disposition.

## Render review

All 80 pages were rendered at 120 dpi after the final PDF build. Eight contact
sheets cover every page. Full-size inspection included the final scientific
owner guide and the final engineering representation table.

The all-page review covered:

- title/status, metadata-facing text, contents, and final pages in both views;
- typed parent-fact, numerical-payload, construction, and exact application
  domains;
- exact sparse application, exact-zero activation, and `PRED-025`;
- signal, response, covariance, NOI, and FLT-to-FRUIT validity domains;
- LearnedCandidate through PublicationDecided and Published lifecycle states;
- pre-draw `h_pre` conditioning and fixed-template source authority;
- exact GLS reference variance versus operational-realized covariance roles;
- request-qualified NOI and FLT-to-FRUIT boundary states;
- all seven PA/SA/SP/CU/NU/RU/FH role profiles;
- the title and AO owner-disposition alternatives in both views; and
- requirements, predictions, edge cases, conformance tests, crosswalks, and
  owner/engineering authority distinctions.

Observed result: no clipping, overlap, black squares, broken glyphs or
equations, unreadable tables, missing headers or footers, accidental blank
pages, or malformed transitions. Narrow-table underfull-box warnings produce
ordinary justified spacing and no visible defect. Both logs contain zero
overfull boxes, undefined references or control sequences, missing characters,
or compile errors.
