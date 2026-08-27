# Build Timing Evidence

Each campaign directory contains the JSON manifest and stage logs emitted by
`tools/build/measure_spack_build_times.py`. Accepted evidence must record an
empty source status, an exact source revision and Spack root DAG, a supported
compiler profile, and successful clean-configure, clean-build, no-op, and
incremental stages.

The harness uses a disposable build tree and removes it after the campaign.
Its default incremental case changes only the modification time of
`src/citlali/cli/main.cpp`, restores that time afterward, and verifies the file
content hash. Timing evidence does not modify source bytes.

Committed campaign directories are immutable observations. A later campaign
uses a new timestamped directory rather than replacing prior evidence.
