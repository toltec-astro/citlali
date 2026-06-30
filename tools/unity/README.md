# Unity Sync And Build Helpers

These helpers prepare the local refactor worktree for validation on Unity.
The preferred workflow is to push this branch to GitHub, pull it in a separate
Unity checkout, and build there.

Recommended local workflow from this repo:

```bash
git push -u origin codex/structural-refactor
```

Recommended one-time Unity setup:

```bash
cd "${HOME}/work_toltec/citlali_dev"
git clone git@github.com:toltec-astro/citlali.git citlali_refactor
cd citlali_refactor
git fetch origin codex/structural-refactor
git switch --track origin/codex/structural-refactor
echo 'source "${HOME}/work_toltec/citlali_dev/citlali_refactor/tools/unity/citlali_refactor_bashrc.sh"' >> ~/.bashrc
source ~/.bashrc
citlali-refactor-update
```

This keeps the existing gw_dev comparison checkout untouched:

- protected comparison repo: `/home/toltec_umass_edu/work_toltec/citlali_dev/citlali`
- refactor repo: `/home/toltec_umass_edu/work_toltec/citlali_dev/citlali_refactor`
- refactor build dir: `build`
- configure preset: none by default; mirrors the existing `citlali/build`
  workflow
- target: `citlali_cli`

The sourceable Unity helper is:

```bash
tools/unity/citlali_refactor_bashrc.sh
```

It defines:

```bash
citlali-refactor-update
citref-update
```

See `doc/UNITY_GIT_PULL_WORKFLOW_2026-06-30.md` for setup details.

## SSH/Rsync Fallback

The scripts below are retained for uncommitted local experiments or emergency
source overlays. They are not the normal validation path.

Defaults:

- remote host alias: `unity_toltec`
- protected comparison repo: `/home/toltec_umass_edu/work_toltec/citlali_dev/citlali`
- refactor repo: `/home/toltec_umass_edu/work_toltec/citlali_dev/citlali_refactor`
- refactor build dir: `build`
- configure preset: none by default

Override them with environment variables or command-line options:

```bash
UNITY_HOST=unity \
UNITY_REPO=/home/toltec_umass_edu/work_toltec/citlali_dev/citlali_refactor \
tools/unity/sync_to_unity.sh --live

tools/unity/build_on_unity.sh \
  --live \
  --build-dir build \
  --target citlali_cli \
  --jobs 12
```

Both helpers refuse to target the protected comparison repo unless
`ALLOW_BASELINE_REPO=true` is set.

## Why Rsync Instead Of Manual SCP

`rsync` over SSH is safer for repeated source overlays:

- it transfers only changed files
- it can dry-run before writing
- it keeps build directories, reduction outputs, and `.git` metadata out of
  the transfer
- it avoids accidentally copying local macOS build artifacts to Unity

If `scp` is preferred, use the bundle helper:

```bash
tools/unity/make_source_bundle.sh
scp /tmp/citlali-refactor-source.tar.gz unity_toltec:/tmp/
ssh unity_toltec 'mkdir -p /home/toltec_umass_edu/work_toltec/citlali_dev/citlali_refactor && cd /home/toltec_umass_edu/work_toltec/citlali_dev/citlali_refactor && tar -xzf /tmp/citlali-refactor-source.tar.gz'
```

## Build Strategy

The remote build helper builds under the refactor source tree:

```text
/home/toltec_umass_edu/work_toltec/citlali_dev/citlali_refactor/build
```

That leaves the gw_dev comparison executable and cache under
`/home/toltec_umass_edu/work_toltec/citlali_dev/citlali/` untouched.

If the build directory does not exist, or if `--configure` is passed, the ssh
fallback helper runs:

```bash
cmake -S . -B <build-dir> --preset unity_release
```

Then it runs:

```bash
cmake --build <build-dir> --target citlali_cli -j <jobs>
```

For the first morning validation, keep the gw_dev build directory intact and
use the default refactor build directory. If a comparison build needs a
different cache, create another directory under `citlali_refactor/`, not under
the protected `citlali/` tree.

The sourceable `citlali-refactor-update` helper does not use the `unity_release`
preset by default. It configures `citlali_refactor/build` directly, with
`CMAKE_BUILD_TYPE=Release`, `CITLALI_USE_WIENER_FILTER_OMP=ON`, and
`FETCHCONTENT_SOURCE_DIR_TULA` unset so CMake can populate `build/_deps/tula-src`
the same way the existing `citlali/build` cache does.

It also passes `CONAN_CMD` explicitly. By default it uses the Conan executable
recorded in the existing Unity `citlali/build` cache:

```text
/work/toltec/toltec_shared/toltec_astro/extern/pyenv/versions/conan1/bin/conan
```

Override with `CITLALI_CONAN_CMD=/path/to/conan` if Unity's shared Conan path
changes.
