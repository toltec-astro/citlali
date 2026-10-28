# Independent review and repairs

First exact candidate:4f61e7649d6c3186e6ccf4f3721fe7ad9283c71b,
tree d2783bbb56aa7b7a55654fc5f2a5f4e69594590b,
parent6e2a467b85f07f15b8fdfe947bed81df833d11bb.
Fresh-context read-only reviewer:/root/configured_point_exact_review.
Independent reproduction:126 focused tests passed in10.72s.

Disposition:scientific/behavioral repair required; architecture passed within
scope; repository/evidence passed with supplemental-environment limitations.
Two concrete findings: ordinary zero/nonpositive MJD constant-offset sentinels
were not normalized for native CAL; nonunit flxscale_correction was accepted
but not consumed by native CAL. Repairs normalize the recognized sentinel only
in the derived copy and reject unsupported nonunit/invalid/duplicate scale
requests before processing. Genuine epoch pairs remain explicit. Unknown
calibration kinds cannot silently pass. Photometry is recorded as preserved
metadata, not a second flux calibration.

A subsequent non-processing native CLI smoke check found the quiet log option
must be spelled `warning`, not `warn`. The adapter/test now use the actual
native spelling; a bound empty-network request reaches native input validation
and fails for its expected reason without numerical processing.

The repaired source requires a new exact-SHA review. No first-candidate review
verdict is transferred to a later source implicitly.

## Repaired candidate review

Exact87e9217fc0016839d43684f539e93774199a32a1, tree
d942fe4edcdcf4f8184ff7d9b278d022c19a8edb, parent4f61e7649d6c3186e6ccf4f3721fe7ad9283c71b.
Same independent reviewer reproduced168passing tests in10.73s and confirmed
both findings closed; no new actionable findings. Scientific/behavioral and
architectural conformance pass within scope; repository/evidence passes with
recorded external-dependency and supplemental-environment limitations. Native
C++ unchanged by repair. Verdict:pass with recorded limitations. This review
was source acceptance only; it did not authorize a reduction launch.
