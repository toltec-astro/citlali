# Unity Sync And Build Workflow - 2026-06-29

This workflow is the rsync/ssh fallback for copying this local refactor
worktree to a separate Unity refactor checkout and building there. The normal
workflow is now the GitHub pull workflow in
`doc/UNITY_GIT_PULL_WORKFLOW_2026-06-30.md`.

The copied local Unity tree at `~/foo/citlali_dev/citlali` shows the gw_dev
comparison source path in its build cache:

```text
/home/toltec_umass_edu/work_toltec/citlali_dev/citlali
```

It also has an existing Unity comparison build directory:

```text
build_unity_release_native_lto
```

That build cache should remain untouched for `gw_dev` comparison. The refactor
source tree and build should live separately:

```text
/home/toltec_umass_edu/work_toltec/citlali_dev/citlali_refactor
/home/toltec_umass_edu/work_toltec/citlali_dev/citlali_refactor/build
```

## Recommended Path

Use `rsync` over SSH as the normal source-copy mechanism. This is safer than
manual `scp` for repeated iterations because it can dry-run and transfers only
changed files.

From `/Users/gwilson/GitHub/citlali-refactor`:

```bash
# Preview source copy. This contacts Unity but does not write files.
tools/unity/sync_to_unity.sh --dry-run

# Copy source/docs/tools only. Excludes .git, build dirs, reductions, coadds,
# Python caches, and local generated artifacts.
tools/unity/sync_to_unity.sh --live

# Print the build command without contacting Unity.
tools/unity/build_on_unity.sh

# Build on Unity when ready.
tools/unity/build_on_unity.sh --live
```

Defaults:

| Setting | Default |
| --- | --- |
| Host | `unity_toltec` |
| Protected comparison repo | `/home/toltec_umass_edu/work_toltec/citlali_dev/citlali` |
| Remote refactor repo | `/home/toltec_umass_edu/work_toltec/citlali_dev/citlali_refactor` |
| Refactor build dir | `build` |
| Configure preset fallback | `unity_release` |
| Build target | `citlali_cli` |
| Jobs | `8` |

Override example:

```bash
UNITY_HOST=unity \
UNITY_REPO=/home/toltec_umass_edu/work_toltec/citlali_dev/citlali_refactor \
tools/unity/sync_to_unity.sh --live

tools/unity/build_on_unity.sh --live --jobs 12
```

The sync and build helpers refuse to target the protected comparison repo unless
`ALLOW_BASELINE_REPO=true` is set.

## SCP Fallback

If direct `scp` is preferred, create a source-only tarball:

```bash
tools/unity/make_source_bundle.sh
scp /tmp/citlali-refactor-source.tar.gz unity_toltec:/tmp/
ssh unity_toltec 'mkdir -p /home/toltec_umass_edu/work_toltec/citlali_dev/citlali_refactor && cd /home/toltec_umass_edu/work_toltec/citlali_dev/citlali_refactor && tar -xzf /tmp/citlali-refactor-source.tar.gz'
```

Then run:

```bash
ssh unity_toltec 'cd /home/toltec_umass_edu/work_toltec/citlali_dev/citlali_refactor && cmake --build build --target citlali_cli -j 8'
```

The tarball excludes `.git`, build directories, reduction outputs, coadds,
Python caches, and common local generated files.

## Build Strategy

For the first validation:

1. Leave `/home/toltec_umass_edu/work_toltec/citlali_dev/citlali` untouched for
   gw_dev comparison.
2. Build target `citlali_cli`.
3. Use `/home/toltec_umass_edu/work_toltec/citlali_dev/citlali_refactor/build`
   for the refactor build.
4. Do not clean the comparison build directory.
5. If a clean configure is needed for the refactor build, use:

```bash
tools/unity/build_on_unity.sh --live --configure
```

The helper runs:

```bash
cmake -S . -B build --preset unity_release
cmake --build build --target citlali_cli -j 8
```

when `--configure` is requested, or when the refactor build directory lacks a
cache.

## Modernizing The Build

The current build can be modernized incrementally, but not before the first
baseline compiles:

- Keep out-of-source builds.
- Keep a named Unity preset.
- Prefer one stable release build dir for refactor validation and keep the
  gw_dev comparison build cache separate.
- Add a separate debug/sanitizer build dir later if needed.
- Avoid deleting or rebuilding fetched dependencies unless CMake cache state
  requires it.
- Once the refactor branch is pushed, prefer normal git fetch/checkout on Unity
  over source overlays for reviewed PR validation. Until then, `rsync` is the
  practical bridge for uncommitted local scaffolding.

## Safety Notes

- `sync_to_unity.sh` does not use `--delete` unless explicitly requested.
- The helpers default to `citlali_refactor/` and guard against accidentally
  targeting the protected gw_dev comparison repo.
- Use `--delete` only after reviewing a dry-run, because it can remove remote
  source files that are absent locally after filters.
- The sync excludes `build*`, `reduNN`, and `coadded` so reduction products and
  build products stay on their respective machines.
- The remote branch name on Unity may still show `gw_dev` after source overlay;
  use manifests and git SHA notes to record exactly what was tested.
