# PTC rank response/noise comparison completed

Owned branch codex/ptc-rank-response-20260926, managed checkout
/Users/gwilson/.codex/worktrees/ptc-rank-response/citlali-refactor, permanent Git
/Users/gwilson/GitHub/citlali/.git. Base tag
`timestream-successor-bright-point-base-v1-20260926` remains at exact
5435bdbdb4e6708e884ae4cb2b0fd93e74102ce5. Science source is
**e3cbf1eeadda256e4d9f400a834f9f7d462c1ae2**, reviewed before execution; later commits
complete reporting, failure-status propagation, evidence and cleanup only.

Read [assessment](../validation/ptc_rank_response_2026-09-26/ASSESSMENT.md),
[matrix](../validation/ptc_rank_response_2026-09-26/compact/report/matrix.json),
[reproduction](../validation/ptc_rank_response_2026-09-26/REPRODUCTION.md), and
completion/review/cleanup records in that validation directory.

All 12 trajectories and 120 feedback applications completed without replacement.
Rank10 REF and all three injection regressions passed; REF numerical maximum
difference is zero. One shared original-input RTC/CAL parent; three rank-local
noise packages/endpoints, nine ideal controls. Ranks are learned fresh per native
network/segment on each residual. All 132 maps have exactly equal coverage/support;
all 1,572 residual/M0 segment receipts confirm configured rank, fresh learning and
unchanged eligibility. Original rank10 M0 has its separate upstream receipt checks.

Rank15 lowers compact response-normalized conditional uncertainty roughly 12%
(peak), 31% (template), 41% (aperture) versus rank10. Both networks support the
residual-power trend. Its broad shoulder aperture recovery worsens from 84.3% to
81.6%; coherent broad bias remains visible. Rank5 gives higher uncertainty but
slightly better broad response. Keep rank10 as reference; no promotion or general
optimum claim. The only recommended next action is a bounded independent-observation
rank15-versus-rank10 confirmation, not executed here.

Scientific source/configuration, compact maps/masks/noise operands/measurements,
matched-scale figures and native diagnostics are durable. Native detector-time
payloads are scratch and are enumerated for removal only after compact preservation.
See CLEANUP.json for actual completion. Saved reporting works without original
arrays and does not silently rerun science. RTC remains unchanged and deferred;
no pre-RTC transfer test, alternate array, seed, scope/method change or further
FRUIT development was launched. No Unity access, merge, push or production change.
