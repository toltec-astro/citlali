# SCI-TEMPLATE_TO_SCI-FLT-MATCHED v0.1/r0.1

Status: exact template-producer boundary draft; producer package name remains
generic until separately authorized

Producer: authorized template-response producer

Consumer: `SCI-FLT-MATCHED`

The producer supplies one immutable scientific object giving expected
ordinary-MAP parent response per unit declared amplitude. It binds template
identity/generation, amplitude convention and units, source-domain reference,
parent response quantity, CAL/BEAM lineage, WCS/frame and output-anchor
relation, sampling, translation, phase, even/odd and tie rule, support,
validity, lifecycle, provenance, and supported query vocabulary.

The object may be exactly materialized, structured, or reconstructed by exact
lineage/query. The representation shall not alter its scientific identity or
template-amplitude estimand. `SCI-FLT-MATCHED` owns only the selected use in its
normalized operator and the resulting response; it does not infer a template,
rescale an ambiguous template, or learn one from target/candidate/NOI data.

Missing scale, phase, anchor relation, response meaning, lineage, validity, or
required query makes the dependent route unavailable. A point-source flux
meaning requires separate exact CAL/BEAM and parent/template authority.
