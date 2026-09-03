# SCI-FRUIT post-EL-F7 direction

Date: 2026-09-03

Status: **scientific-owner direction to investigate the mechanism and prepare
the next bounded decision; not authorization to run EL-F8**

After EL-F7 completed, the scientific owner asked the manager to lead the next
step while raising this concern:

> I struggle to see how a single detector can make such a large imprint on a
> map without itself getting flagged or deweighted by the pipeline.

The owner was willing to continue after making that concern explicit. The
manager therefore performed a read-only check of the frozen implementation,
APT, checkpoint, learning table, logs, and EL-F5--F7 evidence before selecting
another experiment.

That check found that UID 4460 *was* explicitly flagged in the affected scan.
The surprising effect is produced after the learned record removes all of the
detector's samples before the next RTC and shared cleaning pass. The map
difference can consequently include changes in other a1400 detectors; it is
not the direct map contribution of UID 4460 alone.

This direction authorizes recording that explanation and preparing a narrow
owner-review packet that tests the placement of the exclusion. It does not
authorize implementation, a build, a replay, algorithm or configuration
changes, method selection, Gate D, qualification, Stage B, production use, or
Unity activity. Exact execution requires approval of the content-bound EL-F8
manifest.
