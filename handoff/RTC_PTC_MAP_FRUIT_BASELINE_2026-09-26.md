# RTC/PTC evaluation through the frozen bright-POINT endpoint

The closeout is complete at the local checkpoint
`timestream-successor-bright-point-base-v1-20260926` on
`codex/fruit-pipeline-freeze-20260926`. Start from that tag, not an old scratch tree.
The durable checkout is
`/Users/gwilson/.codex/worktrees/fruit-pipeline-freeze/citlali-refactor`;
permanent Git is `/Users/gwilson/GitHub/citlali/.git`. The verified starting
implementation was `ac1bf4eecc9973515195151bf865e25893e2e5d5`, whose tested science
ancestor was `ac2bee1febaab82b808f7f751374b797de033c6a`. The consolidated numerical
candidate is **b47c13e1f5b745b94880218e347f77b4142686ff**; later closeout commits contain
reporting, documentation and evidence only. Local integrated mainline
`e32c4584dec07617d6caa857d39bf13047b6c4b0` is an ancestor (171 commits behind the
starting implementation). It was not advanced; no remote freshness claim, merge,
push or production activation is implied.

Read, in order:

1. Repository AGENTS and effective governance pointers.
2. [One pipeline quickstart and ownership map](../doc/BRIGHT_POINT_PIPELINE.md).
3. [Frozen configuration](../data/development/bright_point_base_v1/reference.json).
4. [Regression/cleanup report](../validation/fruit_pipeline_freeze_2026-09-26/REPORT.md),
   including exact source/binary, original input bindings, strict comparator and
   isolation proof; then the top of current status/goals.

The executable command from recorded original inputs to M0 and T10 is, after the
quickstart's explicit environment/build and root setup:

```sh
$HOME/tolteca/bin/python "$POINT_REPO/tools/timestream_successor/point_pipeline.py" feedback \
  --config "$POINT_CONFIG" --input-root "$INPUT_ROOT" \
  --output "$POINT_OUTPUT" --binary "$POINT_BINARY"
```

`preflight`, `forward` and `report` are the other modes of this same entry point.
The successful cold-start run exercises the shared M0 implementation once; do not
repeat raw processing just to demonstrate `forward`. Reporting is read-only and
never launches missing science. Use a fresh output for an authorized new reduction.
The tested runtime was 117.213 s, native peak RSS 1.558 GB, peak recorded stage storage
0.814 GB against 3.323 GB ceiling. One M0/trajectory, 10 applications, one endpoint;
all remain exactly equal to the retained BASE reference, including uncertainties.
No second attempt, real injection trajectory, ensemble or parameter sweep ran.

Original 152430/0/2 data, associated 152430/0/1 tunes, canonical APT and telescope
remain available under the explicit input root, verified against the retained
manifest. No transfer is required. Compact results are 26.184 MB in Git; the existing
235 MB comparison is referenced without copying. New RTC/PTC numeric intermediates
were enumerated and removed. Neither an archived stream nor another observation's
result directory is an input to the consolidated route. A source rejection retains
the map and empty accepted output and stops dependent feedback; processing failure
is separately recorded. Native failure/validity does not become invented data.

## Boundaries for the next work order

Change an authorized RTC configuration through the effective YAML and explicit
array filter-plan owner. Change an authorized PTC configuration through the existing
native `ptc.method`/`ptc.rank` owner; reference.json currently selects observed-entry
ALS rank 10. Feedback reads current solver settings from the PTC receipt. The
shared runner remains sequencing, not the home of an RTC/PTC algorithm. Existing
per-network/per-processing-segment learning, physical gaps, paired x/r, calibration,
AST and VAL are preserved; resident parent data do not imply observation-wide PTC.

Every new observation/upstream configuration initializes its own M0 noise/guard
geometry; C0/BASE/K0 are fixed only within that trajectory. Fresh PTC Learn/Consider/
Apply occurs on each current `y - project(m_k)` residual. Restore the exact native
operand before mapping. Do not reuse old bases or a previous residual, reapply RTC,
union masks, feed back a fitted Gaussian, alter gain or map weights, or apply m10.

The downstream reference is ordinary MAP, BASE-POINT threshold 3, existing
POINT-EXPECTED seven-parameter W=I fit/guard, gain 1, replacement, 10 applications.
High coverage remains informational with unchanged numerical continuation; genuine
covariance/variance/projection and low-coverage/support checks remain. Nominal beams
are 5/7.5/10 arcsec, separate from measured PSFs. Bright POINT only; the cold proof
covers 152430/a2000, while the existing three array interfaces remain callable.
No all-mode or production qualification follows from this integration result.

Existing pre-PTC injection/projection controls and compact/shoulder definitions
remain in the native FruitPointModel/Residual path, `point_response_cd.py`, retained
comparison schemas and small contract tests. They bypass RTC/CAL. A pre-RTC source
injection requires a separate measured x/r units, response, calibration, native
occurrence/VAL and AST contract; it is not implemented in this closeout. Complete
upstream response and realized NOI remain unavailable where reported. Conditional
endpoint uncertainties are not a full adaptive uncertainty law, and the measured
shoulder-response limitation remains.

FRUIT method development, CORR/DIAG/JINC, CUT3, shoulder refinements, seeds and extra
cycles are parked. The proposed 5/10/15-mode comparison and upstream injection are
future owner-directed work. Do not start either merely because this handoff exists.
The historical September 9 FRUIT source recovery gap is documented in the preflight;
the current explicit owner freeze supplies this bounded route's authority.

## Ready-to-paste kickoff

> Start from local tag `timestream-successor-bright-point-base-v1-20260926` on
> `codex/fruit-pipeline-freeze-20260926`, in the durable Citlali checkout identified
> by `handoff/RTC_PTC_MAP_FRUIT_BASELINE_2026-09-26.md`. Read its short reading list
> and verify clean ancestry/source/build before acting. FRUIT is frozen at ordinary
> MAP + BASE-POINT; preserve its downstream code and reference settings. Recover the
> callable original-input -> RTC/CAL -> immutable PTC parent -> fresh PTC -> MAP/M0
> -> ten replacement-feedback applications boundaries. Use the existing owner
> configuration interfaces for the next explicitly authorized RTC/PTC work order.
> Do not launch a mode-count comparison, alter RTC settings, or implement a new
> injection boundary until that work order specifies the scientific question and
> budget. Use saved compact products for inspection without regenerating science.
