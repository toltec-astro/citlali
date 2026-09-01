# SCI-FRUIT v0.1 — Support Comparison Rule

Status: **Stage A candidate rule; no support domain is selected**

For every spatial, morphological, flux, response, residual, or false-structure
metric, the protocol must report separately:

1. recovery on the prospectively declared common truth/comparison support;
2. candidate scientific-support and availability fraction;
3. historical-control scientific-support and availability fraction;
4. support gained and support lost, with geometry and weighting; and
5. failure and scientific-unavailability causes.

The target grid, WCS, response/kernel convention, edge domain, masks,
normalization, weighting measure, background treatment, missing/non-finite
policy, and any remapping are frozen before qualification. A paired metric is
valid only on its prospectively declared comparison support; reduced support
is not permission to hide hard regions.

A method cannot improve apparent accuracy by withholding difficult output
regions, changing support after outcomes, or treating unavailable values as
zero error. Support loss can independently fail a protected guardrail even
when common-support accuracy improves. Support gain is reported as a distinct
scientific endpoint and does not automatically count as paired improvement.
