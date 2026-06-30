# Unity Git Pull Workflow - 2026-06-30

This is the preferred workflow for validating the refactor branch on Unity.
Push from this machine to GitHub, then pull and build on Unity. Do not overlay
the refactor source into the existing `gw_dev` checkout.

## Local Machine

From `/Users/gwilson/GitHub/citlali-refactor`:

```bash
git status --short
git add doc handoff tools
git commit -m "Add structural refactor planning and Unity workflow tools"
git push -u origin codex/structural-refactor
```

Use a narrower `git add` if there are files you do not want in the first
refactor commit. Unity can only pull files that have been committed and pushed.

## One-Time Unity Setup

On Unity:

```bash
cd "${HOME}/work_toltec/citlali_dev"

# Keep this existing checkout untouched for gw_dev comparisons:
test -d citlali/.git

# Clone a separate refactor checkout.
git clone git@github.com:toltec-astro/citlali.git citlali_refactor
cd citlali_refactor
git fetch origin codex/structural-refactor
git switch --track origin/codex/structural-refactor
```

The refactor build must live under:

```text
${HOME}/work_toltec/citlali_dev/citlali_refactor/build
```

The existing comparison build stays under:

```text
${HOME}/work_toltec/citlali_dev/citlali/build_unity_release_native_lto
```

The Unity preset expects the sibling `tula` checkout used by the existing
`citlali` build:

```text
${HOME}/work_toltec/citlali_dev/tula
```

If it is missing, clone or restore it before configuring Citlali.

## Bashrc Integration

After the first clone, add this to Unity's `~/.bashrc`:

```bash
source "${HOME}/work_toltec/citlali_dev/citlali_refactor/tools/unity/citlali_refactor_bashrc.sh"
```

Then reload the shell:

```bash
source ~/.bashrc
```

The snippet defines:

```bash
citlali-refactor-update
citref-update
```

Both aliases pull `origin/codex/structural-refactor`, configure
`citlali_refactor/build`, build `citlali_cli`, and print the resulting Citlali
version when available.

## Normal Unity Update

After pushing new commits from this machine, run on Unity:

```bash
citlali-refactor-update
```

Equivalent explicit command:

```bash
cd "${HOME}/work_toltec/citlali_dev/citlali_refactor"
git fetch origin codex/structural-refactor
git switch codex/structural-refactor
git pull --ff-only origin codex/structural-refactor
cmake -S . -B build --preset unity_release \
  -DFETCHCONTENT_SOURCE_DIR_TULA="${HOME}/work_toltec/citlali_dev/tula"
cmake --build build --target citlali_cli -j 15
./build/bin/citlali --version
```

## Useful Overrides

Set these in the Unity shell before calling `citlali-refactor-update` if needed:

```bash
export CITLALI_REFACTOR_BRANCH=codex/structural-refactor
export CITLALI_REFACTOR_JOBS=15
export CITLALI_REFACTOR_PRESET=unity_release
export CITLALI_REFACTOR_TARGET=citlali_cli
export CITLALI_TULA_DIR="${HOME}/work_toltec/citlali_dev/tula"
```

The helper refuses to use the protected baseline checkout
`${HOME}/work_toltec/citlali_dev/citlali` unless explicitly overridden. That
guard exists to preserve the `gw_dev` executable for output and performance
comparisons.

## Relationship To Rsync Tools

`tools/unity/sync_to_unity.sh` remains available as a fallback for uncommitted
local experiments, but the normal path should be:

```text
local commit -> git push -> Unity git pull -> Unity build
```

This gives every Unity build a real Git commit SHA and avoids ambiguous source
overlays.
