# EL-F12 scalar-dimension comparison repair r0.2

Date: 2026-09-05. Status: all H-uninjected compatibility checks pass on the
retained trajectory; no binary change or replay.

H completed all seven iterations successfully. Its original comparison
stopped because the NetCDF helper represents the newly registered H audit
string as a one-element string variable with its own length-one dimension,
`fruit_response_state_dim`. The original analyzer permitted the added
variable but mistakenly required identical dimension-name sets. No existing
scientific dimension differed. The original failed receipt is preserved.

The repair allows exactly that new dimension, requires length one and fixed
size, and requires that only the H audit string use it. Every pre-existing
dimension, variable, attribute and scientific value remains exactly compared.
The added policy entry must still be exactly `fruit_response_arm: H`.
Unrelated dimensions and scientific changes remain rejected by tests.

Rechecking the retained outputs passes all 84 science-plane comparisons
(seven iterations, three arrays, four planes), all existing checkpoint and
D19 state, exact ordered iteration-specific CSV records, and map diagnostics.
The 272 baseline/FRUIT Python tests and focused Ruff checks pass. Native
source, executable, all science inputs, method, regions, thresholds,
resource limits, and trajectory counts are unchanged. Three primary
trajectories and 21 passes have completed; the diagnostic replay allowance
remains unused.

Automatic approval review rejected a proposed general receipt-path override
because it could bypass prerequisites. That approach was discarded. The
controller instead recognizes only this exact recorded representation defect,
checks the immutable original receipt and bound retained products, and
reruns every original scientific prerequisite before dependent work. A
passed receipt alone cannot bypass these fresh checks; execution failures
and other scientific failures cannot enter this repair path.

All seven successful temporary occurrence spools were compressed and
verified losslessly under the original retention rule. The repair proof
binds every retained output and preserves the original failure. This is the
owner-authorized routine defect repair, not a gate, method, population,
input or execution-scope change. The next action is H injected after the
successor analyzer registration is frozen. No alternative has run.
