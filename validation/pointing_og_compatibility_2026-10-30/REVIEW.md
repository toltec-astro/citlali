# Independent exact-source review

Verdict: **pass with recorded limitations**, no findings or required repairs.
Reviewer: /root/og_pointing_review, fresh context, read-only.
Candidate28061144d2018d2f57cd6c47ee95990bae186c19;
tree5d78e1e78b6aecba8f9f0e48cc79655223ae0eb2;
parent7f960a377591a3ff41fb2c45647c6c7bd26cd68f.
Reviewed worktree clean on codex/configurable-point-20261028.
This closure records the verdict without changing reviewed scientific conclusions.

Scientific/behavioral: no findings against this audit. Original source supports
axes, units, single resolver negation, User/Paddle composition, corrected-elevation
rotation, no extra cosine on supplied corrections and equivalent saved-FITS
orientation. Original OG and frozen V2 identities are correctly distinguished.

Architecture/ownership: no findings. No application implementation/interface
changed. Proposed compatibility work preserves existing concrete owners; it
neither introduces a correction algorithm nor authorizes implementation silently.

Repository/evidence: no findings. Eight reviewed changed paths are audit evidence
and plan/status only. Exact ancestry/tree verified, all 18 source/binary hashes
and seven input hashes match, whitespace passed, checkout clean.

Independent in-memory Python reproduction:
- Actual pinned TolTECA resolver plus archived PPTs/telescope headers reproduces
  both saved SCIENCE bracket corrections and MJDs exactly.
- Configured POINT rejects single positive MJD and two offset values without
  epochs; accepts dated brackets and undated constants.
- Fractional-epoch same-time maxima reproduced:152390 az/el
  0.000965410564996/0.000773115746750 arcsec;152392
  0.000189330408023/0.000289073043869 arcsec. Endpoint evaluation correctly bounds
  absolute affine-interpolant differences on overlapping bracket support.
- Six native test sources and recorded passing outputs inspected and binary
  hashes verified; reviewer did not rerun binaries.

Read: AGENTS, three governance documents and effectiveness record, scientific
conventions, baseline/source authorities, audit package and plan/status diff.
Used local TolTECA Python -B. No reviewer file edits/builds/fits/reductions/Unity
or external messages. No numerical pipeline or source-fit execution by author;
six author-executed C++ checks are synthetic, not reductions.

Retained limitations: two-bracket arithmetic bounds only; integer-day tests do
not establish new CAL fractional-epoch parity; native/common-grid schedules differ;
conventional successor PPT publication/full correction fixture absent; conventions
do not prove equal fitted centroids, end-to-end compatibility or supported Spack/
Unity qualification. Audit acceptance is not integration, activation or production.
