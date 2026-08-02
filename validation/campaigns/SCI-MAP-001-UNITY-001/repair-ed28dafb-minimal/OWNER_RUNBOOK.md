# Grant's minimal Unity runbook

These are human commands for Grant to run after confirming the candidate,
operational paths, TolProj site configuration, and canonical inputs. They are
not authorization for Codex or for unattended execution. Every remote command
uses `unity_toltec`.

## 1. Transfer the package

### Local

```sh
LOCAL_PACKAGE=/Users/gwilson/.codex/worktrees/aa31/citlali-refactor/validation/campaigns/SCI-MAP-001-UNITY-001/repair-ed28dafb-minimal
UNITY_RUN_ROOT="$HOME/c2025t/2026-ENG-citlali-MAP"
UNITY_PACKAGE="$UNITY_RUN_ROOT/repair-ed28dafb-minimal"
rsync -a --checksum --protect-args "$LOCAL_PACKAGE/" \
  "unity_toltec:~/c2025t/2026-ENG-citlali-MAP/repair-ed28dafb-minimal/"
```

### Unity terminal

Paste these into the already logged-in Unity terminal (and, when appropriate,
the allocated compute-node shell):

```sh
UNITY_RUN_ROOT="$HOME/c2025t/2026-ENG-citlali-MAP"
UNITY_PACKAGE="$UNITY_RUN_ROOT/repair-ed28dafb-minimal"
cd "$UNITY_PACKAGE"
sha256sum -c SHA256SUMS
```

## 2. Create the two TolProj workspaces

The agreed Unity root is `$HOME/c2025t/2026-ENG-citlali-MAP`. Set the three
remaining operational values in Unity. Do not reuse an old reduction or copy
an old reduction tree into it.

### Local

No additional local command is needed after the transfer.

### Unity terminal

```sh
TOLPROJ='<approved TolProj executable on Unity>'
TOLPROJ_SITE_CONFIG='<approved TolProj site configuration>'
GRANT_USER='<Grant Unity/TolProj user>'

set -euo pipefail
point="$UNITY_RUN_ROOT/SCI-MAP-001-POINT-SOURCE"
science="$UNITY_RUN_ROOT/SCI-MAP-001-SCIENCE-SOURCE"
test ! -e "$point"; test ! -e "$science"
"$TOLPROJ" init-test "$UNITY_PACKAGE/tolproj-point-source.json" --root "$UNITY_RUN_ROOT" --user "$GRANT_USER" --config "$TOLPROJ_SITE_CONFIG"
"$TOLPROJ" init-test "$UNITY_PACKAGE/tolproj-science-source.json" --root "$UNITY_RUN_ROOT" --user "$GRANT_USER" --config "$TOLPROJ_SITE_CONFIG"
test -f "$point/project.yaml"; test -f "$science/project.yaml"
```

Stage only the canonical raw files and matched APT/PPT inputs whose observation
identities appear in `campaign.json`: Point APT 152389; Science APTs 152390 and
152392; Science pointing support 152389, 152391, and 152393. Stop if a required
input is absent or ambiguous. Do not copy an old reduction, and do not change
the selected observations or groups.

## 3. Set up and run the selected reductions

On Unity, use the approved TolProj workflow to prepare the pointing-support
reduction before Science. Use `--refactor`, the explicit source/group below,
and only the seven case objects in `campaign.json`. Each case records its exact
mode, observations, arrays, coadd/products setting, coverage cut, and thread
count. If the accepted configuration cannot be materialized with those values,
stop rather than hand-editing a different scientific configuration.

### Local

No local command is needed; keep the Unity compute-node terminal active.

### Unity terminal

```sh
POINT_PROJECT="$UNITY_RUN_ROOT/SCI-MAP-001-POINT-SOURCE"
SCIENCE_PROJECT="$UNITY_RUN_ROOT/SCI-MAP-001-SCIENCE-SOURCE"

"$TOLPROJ" setup-pointing-reductions "$POINT_PROJECT" --config "$TOLPROJ_SITE_CONFIG" \
  --refactor --source 1146+399 --pointings-dir pointings --apt-dir apts --cpus 1
"$TOLPROJ" setup-pointing-reductions "$SCIENCE_PROJECT" --config "$TOLPROJ_SITE_CONFIG" \
  --refactor --source 1146+399 --pointings-dir pointings --apt-dir apts --cpus 1

# After the selected Science pointing-support reduction finishes as reduNN:
"$TOLPROJ" setup-science-reductions "$SCIENCE_PROJECT" --config "$TOLPROJ_SITE_CONFIG" \
  --refactor --user "$GRANT_USER" --pointing-reduction reduNN --apt-product matched

# For each prepared P-SEQ, P-OMP, S-C-SEQ, S-C-OMP, S-E-SEQ, S-E-OMP, and S-X-SEQ directory:
"$TOLPROJ" validate-reduction '<case reduction directory>'
"$TOLPROJ" submit-reduction '<case reduction directory>'
```

For the auxiliary CAP-POINT/CAP-SCIENCE primitive captures only, apply
`processed-time-chunk-full-overlay.yaml`: `enabled=true`, `mode=full`, and
`indices=all`. Do not apply that overlay to the seven acceptance cases. TolProj
freezes the selected executable at submission; use one ordinary candidate
binary for every selected reduction.

## 4. Collect ordinary artifacts

After all seven selected reductions finish, set `RESULTS_DIR` to a directory
under `UNITY_RUN_ROOT` containing those seven completed reduction directories,
named by case ID.

### Unity terminal

```sh
tar -C "$UNITY_RUN_ROOT" -czf "$UNITY_RUN_ROOT/SCI-MAP-001-UNITY-001-return.tar.gz" \
  SCI-MAP-001-POINT-SOURCE/project.yaml SCI-MAP-001-SCIENCE-SOURCE/project.yaml \
  "$(basename "$RESULTS_DIR")"
```

### Local

```sh
LOCAL_RETURN_DIR='<local destination for returned MAP artifacts>'
rsync -a --checksum --protect-args \
  "unity_toltec:~/c2025t/2026-ENG-citlali-MAP/SCI-MAP-001-UNITY-001-return.tar.gz" \
  "$LOCAL_RETURN_DIR/"
```

Give the archive path and SHA-256 value to the coordinator for interpretation.
This package does not interpret results or close any finding.
