# Frozen review selection, before scientific execution

Per network: at most four evidence-domain decisions plus four original-domain
controls (32 total). Exact rules are in extract_step_campaign.py, committed
before launch. Event examples prioritize median, largest and r-only admitted
original offsets; fourth category varies prospectively by network (source,
repeated, unresolved, non-admitted), with identity fallback for absent classes.
Two candidate-free original x contrast maxima occupy separate time halves;
two time/identity coverage controls use 25%/75% fractions. Supported pre/post
values rank contrasts, not physical truth. Transients, source curvature and
common-mode changes can still win; original x/r/peer review must distinguish
these. Never equate candidate-free centers with no candidate anywhere in a
3-second excerpt. Selected JSON identities are immutable after first extraction.
Cleaned traces are not inputs to selection. Original plots will be inspected
before stage plots. Targeted windows do not estimate population completeness.

Original contrast/control eligibility requires all pre/post samples within one
physical native run. Event excerpts may cross boundaries, explicitly retained
and marked separately from processing-science/numerical segment boundaries.

Source-adjacent selection uses original AST distance to the predeclared
nominal center(0,0), never the cleaned M0 fit. Measured M0 geometry is used
only for later contextual source-crossing/exposure accounting. Population
figures use retained assigned times, drawing physical runs separately.
