# RTC/PTC partial completion — owner binding, 2026-09-18

The owner directs completion with good detectors, retaining existing exclusions
of the 51 fully unavailable detectors. This supersedes the all-coordinate RTC
completion requirement only at the explicit development boundary, and the
PTC prohibition on pre-fit omission only for columns with zero CAL eligibility.
No detector is rescued, reclassified, or adaptively removed by fitting.

`rtc-available-detector-completion-2026-09-18-v1` permits an explicitly retained
complete development plan whose missing final-stage outcome belongs to a
detector with no numerical x or r output anywhere. Its original spectrum must
be available; its conditioned spectrum must report insufficient windows with
zero admitted support. An active detector must not depend on that detector as
a donor. At least one detector must have usable output and its required
outcomes must be available. Missing evidence, inconsistent input, intermediate
stage decisions and scientific qualification are not bypassed. The complete
RTC domain, exclusions, issues, original inputs and VAL facts remain retained.
This binds SCI-RTC r0.12 requirements 046–053/055–057 and existing SCI-VAL uses;
no frozen contract text or estimator is rewritten.

`ptc-cal-observed-entry-use-2026-09-18-v2` counts CAL eligibility per detector
and processing segment before either method fits. Zero-supported columns are
explicitly absent from that segment's fit. Both methods and every requested
rank use the same membership and mask. All requested identities and output
positions remain present; omitted columns have an explicit no-eligible-input
cause. Partially supported columns are not removed. Means, rank, covariance
initializer, overlap requirements, ALS convergence and masked Apply are
unchanged. Groups with fewer than two admitted detectors are unavailable;
other segments still publish. The frozen response operator gathers the same
columns and scatters onto the full requested output domain. This refines the
SCI-PTC r0.5 observed-entry binding's admission realization; it is not adaptive
membership or automatic rejection based on a fit.

Runtime Learn records membership and actual evidence; Consider freezes it and
the exact CAL/VAL identity; Apply publishes full-domain values and causes.
Engineering Learn/Consider/Apply separately governs development and review.

One Tier-2 spine: `codex/timestream-successor-partial-completion-001`, worktree
`/private/tmp/citlali-timestream-successor-cal-001`, verified canonical base
`cb5361f7425ba7d211e284ee57a9a3a67cf393b5`. Evidence is in
`/private/tmp/citlali-successor-partial-completion-20260918`. Verify focused and
regression gates, unchanged upstream real-network data, the ordinary default
RTC → CAL → PTC → VAL invocation, and independent fresh-context exact-SHA
three-axis review before canonical promotion. Owner performs pushes. Local
AppleClang/Homebrew verification does not extend prior Unity qualification.
The frozen legacy baseline, thresholds, filters, paired validity, source
protection and original-input replay are unchanged. MAP and FRUIT remain out
of scope. No new empty-detector recovery or scientific investigation is opened.
