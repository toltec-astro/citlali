# Preflight gates

134 focused native CTests pass (RTC, fixed11, pre-RTC mapping, PTC, FRUIT).
42 shared route/RTC/fixed11 Python tests pass.207 baseline tests and complete
config preflight pass. Initial fixture/build corrections are recorded separately;
no observational processing has started. Build uses the existing local
AppleClang/Homebrew cached dependencies (supplemental evidence).
The final native build will be refreshed after the exact source commit so its
embedded revision matches the independently reviewed candidate before execution.

## Executed candidate and reporting checks

The clean b3d997456 candidate was rebuilt and its embedded version verified;
final native fixtures passed before the campaign. All9 trajectories/45 applications
and2 captures completed with no replacement. All9 saved uncertainty operators/scales
reproduce exactly. The two-arm matched extraction has599,520 identical metric keys
and143,190 exact raw metrics. Final reporting/cleanup/member/diagnostic fixtures:
12 passed (REPORT_TESTS.txt). No further native source changes after execution.
