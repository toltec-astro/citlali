# Optical response and noise weighting owner decision

Implementation follow-up: Grant subsequently requested the full suite and chose
“Use the resampled interval and report its limitations.” The explicit v2 candidate
is described in [the implementation record](../validation/weighted_point_2026-10-29/IMPLEMENTATION.md).
The historical v1/forthcoming statements below describe the original decision
point; v1 still rejects this control, while v2 implements it. Scientific outcomes
await Grant's shell trials. Conditional 99% intervals are not demonstrated 99%
coverage or an observation-wide false-rejection guarantee.

## Accepted default and configuration

On 2026-10-29 Grant accepted the saved-152418 atmospheric-response approach and
directed:

> Let's make that 10% cut the default but a free parameter in the configuration
> yaml. That one may need some tuning. Otherwise I'm very happy with this approach.

The default minimum relative optical response is **0.10**. It is a tunable
dimensionless fraction of the normal response within the same array; normal
response is normalized to one. This is an owner-selected default, not a threshold
statistically established by the single saved observation.

The configuration interface for the forthcoming optical-response gate is:

```yaml
successor:
  detector_response:
    min_relative_response: 0.10
```

Omitting the threshold selects 0.10 once that gate is implemented. A supplied
value overrides the default through the ordinary TolTECA merged configuration.
Use the same requested value on all three arrays, each with its own normal-response
reference. The effective configuration and compact detector diagnostic must record
the actual value. The pipeline must not retune it automatically to obtain a desired
detector count, map appearance or successful completion.

This key is the accepted interface to implement, **not an operational option in
the current `citlali-point-v1` runtime**. That runtime has no optical-response
consumer and continues to reject unsupported controls. Its runnable overlay and
the frozen 152418 trial are unchanged. Default parsing, effective-value recording,
the native decision input, exclusion and diagnostics must land together; accepting
and ignoring the key would misrepresent the reduction.

## Meaning preserved from the accepted approach

The cut concerns confidently weak response to shared optical variations before
the main cleaner. For a configured threshold `f`, the response uncertainty
interval must lie wholly inside `[-f, +f]`, with suitable shared-signal excitation
and consistency across reference constructions and temporal portions. A small
point estimate or low correlation alone is insufficient. Source protection,
reference and uncertainty adequacy remain required; the saved unmasked diagnostic
does not supply a production rejection list.

Strong reversed response is a separate sign/association/calibration issue.
Unresolved response remains explicit. Normal optical response with genuinely
small noise remains valuable; high noise calls for lower weight. The response
coefficient is not automatically applied as a new flux calibration.

The decision is local to the observation and is frozen before basis learning.
Excluded detectors must not influence cleaning or either map numerator or
normalization. Keep the supplied APT and past reductions intact. Required quiet
diagnostics distinguish prior unavailability, disturbance exclusions, weak response,
sign problems, unresolved response and final contribution without double counting.

Noise learning uses source-protected residuals in calibrated units. The endorsed
direction is inverse-variance cleaning and conservative map variance
`max(reference_variance, remaining_variance)`. FRUIT starts each pass from the
immutable post-RTC/CAL parent, subtracts the current sky model, relearns noise and
cleaning, then restores the exact subtracted source operand with consistent map
weights. Optical eligibility is not inferred anew from the shrinking FRUIT residual.

The accepted direction does not supply missing uncertainty mathematics, establish
a false-rejection rate, qualify source protection, or validate corrected-frame
FRUIT. Those implementation and conformance requirements remain visible; the 10%
default decision itself is settled and need not be requested again.

## Saved evidence and implementation boundary

The local assessment and combined proposal are preserved at:

`/Users/gwilson/.codex/visualizations/2026/10/23/01a1688f-ccdb-73b0-99e9-0ff65be99341/atmosphere-152418/ASSESSMENT_AND_PROPOSAL.md`

Its accompanying numerical findings, input identities and plot support the
decision. The clearest example, a1400 network9/channel431, has approximately
4.5% of typical response despite correlation0.977. The preliminary counts below
10% remain diagnostic candidates, not newly dropped detectors. Matched retained
CAL covers all a1100/a1400 networks and a2000 network12; network11 supplies only a
separately qualified older RTC proxy.

This record incorporates the owner decision into repository documentation. It
does not activate new scientific behavior or change the current uniform map
coefficient family. The implementation sequence remains weighted forward POINT
first, then the separately validated FRUIT connection and the collaborator
POINT/calibration/SCIENCE pilot. Grant alone launches reductions from the shell.

Preflight: Engineering Governance, Timestream Successor Governance and Review
and Conformance were read. This documentation-only addition uses clean local
base `f3d2e60a08e04e548dfd95ac509b10ed7a775066` on
`codex/configurable-point-20261028`; it creates no additional implementation lane.
Source, executable configurations, algorithms and frozen trials are unchanged.
Validation is documentation/link/YAML-example inspection and `git diff --check`;
no compilation, observational processing, Unity access or push is needed.
