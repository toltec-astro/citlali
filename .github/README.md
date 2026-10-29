# Integration checks

`cmake-single-platform.yml` now checks pushes and pull requests targeting
`codex/refactor-mainline`, plus explicit dispatches. It runs synthetic Python
build/configuration/source-baseline tests, configured POINT synthetic pytest
tests, and checked-in configuration contracts. The explicit source-baseline
selector defers only `CanonicalAptProtocolV1Test` to the complete native suite. The
old Conan 1/v4.x job is retired; its history remains in Git. The separate
historical Doxygen publication workflow is unchanged.

`spack-native-checks.yml` is an owner-dispatched native job for the supported
`macos-llvm20` profile. Provision an isolated macOS runner with labels
`self-hosted`, `macOS`, and `citlali-spack-macos`, the current Actions runner,
`$HOME/tolteca/bin/python` and the engineering-check dependencies, the accepted
Spack dependencies/compiler, and the four read-only baseline YAML inputs named
by `tools/config/compact_compatibility_cases.yaml`. Set repository variables
`CITLALI_SPACK_ROOT` and `CITLALI_SPACK_ENVIRONMENT` to that runner's Spack root
and concrete environment containing `spack.yaml` and `spack.lock`. Its build-only
first-party sources must match `spack/upstream-revisions.json`; the job prepares
those into the checkout's ignored `build/spack-sources` directory.

The job requires a new build directory for each attempt. It builds the CLI and
`citlali_test_binaries`, runs CTest, verifies the executable revision against the source HEAD, runs the
complete baseline suite using `CITLALI_TEST_CLI`, and requires every
configuration baseline.
The truth-only, sanitizer, installed-package acceptance, and observational
suites remain explicit separate checks. No reduction, Unity access, Spack
package overwrite, or deployment is part of either workflow. Runner
provisioning and an actual dispatched pass are release prerequisites; this
workflow definition alone does not establish Spack qualification or Unity
qualification. Hosted source checks do not substitute for the full local
configuration compatibility gate.

Action interfaces were checked against their official documentation:
[checkout](https://github.com/actions/checkout),
[setup-python](https://github.com/actions/setup-python), and
[self-hosted runner labels](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/use-in-a-workflow).
