# MAP-UNITY-ED1 independent read-only stop review — 2026-08-02

Review mode: independent read-only repository and governing-artifact review

Reviewer identity: delegated Codex subagent `/root/compact_design`

Edits by reviewer: none

Unity access by reviewer: none

## Conclusion

The stop condition is confirmed. Existing fixed seven-case products do not
expose the complete primitive authority needed for the nine compact groups and
deterministic traces. A full processed-TOD adapter would require config and
product changes not authorized by the current handoff.

## Findings checked

1. Point PTC output is enabled but fixed to mini mode. Mini mode stores signal
   as float32 and omits kernel and detector-pointing geometry.
2. Science PTC output is disabled and its configured selection is mini and
   selected-scan rather than full/all.
3. The generic timestream contract and PTC diagnostics do not require or expose
   the complete weights, kernel, pointing, network/APT flags, sample flags,
   scan layout, and realization information needed by ED1.
4. A hypothetical full/all PTC product is technically positioned immediately
   before map population and contains the relevant signal, flags, kernel,
   detector geometry, APT columns, weights, and scan layout.
5. Enabling that product changes frozen configuration/products and adds an
   unapproved product-producing capture execution. The handoff explicitly
   stops on those changes.
6. The output cadence needs an explicit effective-rate binding because the TOD
   `SAMPRATE` metadata uses native `telescope.fsmp` while mapmaking uses
   `telescope.d_fsmp`.

## Recommendation

Return a bounded decision brief. If the owner wishes to resume this campaign,
obtain explicit authorization for a segregated full/all PTC capture (including
resource and cleanup bounds), an application-owned compact stream hook and new
candidate, or a separately governed instrumented validation executable. Do not
construct the missing authority from final FITS products.
