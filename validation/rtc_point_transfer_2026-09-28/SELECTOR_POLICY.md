# Frozen portable rule: rtc-point-physical-run-quarantine-v1

Owner authority: OWNER_DIRECTIVE.md, sections 4–6. Freeze precedes all new
152420 reductions. This is a portable D88-inspired policy, not the historical
manual procedure or a production policy.

One proposal grid per physical run, anchored at its first native time, with
onsets every 0.125 seconds. Each proposal has a 3-second whole window
[onset−1, onset+2), before [−0.4,−0.2), event [0,+0.25), after [+0.5,+0.8).
All intervals are half open. At least 20 samples per flank/event and 95% of the
nominal full-window count are required. No window may cross a physical gap or
run edge. Numerical processing scans do not restart the grid. No refinement,
maximization over outcome-selected times, or fallback window is allowed.

The above fixed grid/flanks are the minimal new proposal formalization; the
old three examples used individually annotated flanks and downstream-selected
windows. The after flank verifies complete support but supplies no recovery
threshold: historical selection imposed none.

Original dimensionless x/r are used, with independent acquisition validity
unchanged. Require finite paired values, finite AST target-centered tangent
coordinates and available mapping throughout the whole window. Use a fixed
60-arcsecond guard about the commanded source position, available before RTC.
This explicitly replaces the historical fitted-M0-centroid/4-major-FWHM guard;
60 arcseconds is a conservative development guard, not borrowed target fit
geometry. It exceeds the existing 40-arcsecond feedback disk. No M0/T5 result
may enter membership. Missing mapping or ambiguous support fails eligibility.

For every eligible detector, median(event) minus median(before) in each
coordinate must have opposite raw signs. Its contrast must differ from the
median of other eligible same-network detector contrasts by strictly more
than 10 times 1.4826 MAD in each coordinate. Zero/nonfinite scale fails. Require
at least 100 other fully paired-valid, off-source peers with nonzero finite
whole-window variance in both coordinates. Absolute ordinary demeaned Pearson
correlation >=0.5 in both coordinates must occur in strictly fewer than 1% of
these peers. Both immediately adjacent readout channels must exist and meet
the same whole-window eligibility; each absolute coordinate correlation must
be strictly below 0.3. These are descriptive cuts, not Gaussian significance.

The first passing grid proposal for each detector/physical run selects that
entire physical run. Other runs remain unchanged. Save all population counts
and the first accepted event, including contrasts/cohort/correlation facts.
No hardcoded detector, observation, network, or event time enters this owner.
The CLI alone restricts activation to the authorized campaign domain.

B does not activate the selector in RTC. After B T5, the same compiled owner is
run on B's saved original-domain source-probe planes; this is saved-data
selection, not another reduction. Each Q role independently invokes that owner
inside its single RTC pass after its optional x-only injection, before normal
Learn/Consider/Apply. No frozen-list reuse for injected Q branches.

Historical validation: saved target/adjacent excerpts can check grid support,
source contact and waveform logic. Saved cohort summaries check historical
acceptance arithmetic only: the full historical peer waveforms were removed
by approved cleanup. Do not claim a complete new historical cohort replay or
tune the rule to recover D88. Synthetic full-cohort tests establish executable
behavior. Missing historical full-cohort evidence is an explicit limitation.
