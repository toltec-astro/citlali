# SCI-ALIGN-001 D005 sky-domain coordinator review — 2026-08-02

Status: evidence verified and integrated; no new owner decision required;
bounded phase-one authorization remains in force

Package: `SCI-ALIGN-001`

## Identity and verification

The dedicated ALIGN task returned the additive, evidence-only sky-domain
commit `bfffe0e60fa8ce05a75ae34b89383bceaadb8fc2` on
`codex/repair-sci-align-001`. Its parent is exact D005 evidence commit
`5a0d64b8f1b9b246b1b5d575c548269823203d22`; the governing application remains
`9aae0e669384c5c0c0dda93debc194d6b8dac787`. The worktree is clean. The commit
adds one diagnostic generator and one validation directory; it changes no
application source, test, configuration, MAP, AST, TolProj, or sibling-
repository path.

The package is
`validation/sci_align_001_align_p0_d005_sky_domain_2026-08-02`. All 15 entries
in `SHA256SUMS` verify. Bound identities are:

- `SHA256SUMS`:
  `ef0dcd523dcc6c46259866e3016a53cb3cde50862b3e103a0a60e9cc0bf049ee`;
- `REPORT.md`:
  `c0a3a2daee0a7e4894a17416f8a71c2398ca41c147149ebbb1a6eccc35c1ca93`;
- `owner_decision_brief.json`:
  `764558bb76b30c55803081ff2ccfcd87e5b32774067345238bf19b7c2b9b485e`;
- `preregistration_protocol.json`:
  `a237697bf7f240f84b1fe02a496790f7b29ef48569f5e40008c52ad1cfda35be`;
- `timestamp_semantics_summary.json`:
  `199c1c0428be31177564d66cf4d35156140e3ac96a469571fd147ecfd18ac529`;
  and
- `trajectory_observation_summary.csv`:
  `81dfc808ff9df4c53de52c58eeba8a940610538f79d26224eedd4072af267738`.

Full raw-file identities are inherited from the verified frozen D005 manifest;
the additive generator independently digests the variables it actually reads.
This review preserves that distinction rather than claiming a second full-file
custody chain.

The task intentionally bound the earlier immutable coordination snapshot
`6785152c2a2d4113c9ba89073de00cb454aa70c4` and therefore reported phase one as
unauthorized. That statement was correct for its frozen input. It does not
supersede the later owner authorization at content commit
`99886afb84c0ef582cee070ec156342a2f0b3327`, identity-bound by coordination
commit `395b51e4e10c5c077c3a3a9d6183323fa383f608`.

## Result and scientific interpretation

The diagnostic evaluated 4,645,586 detector rows across six canonical native-
1x Pointing/Beammap observations and 66 interfaces. Every one of the 4,645,476
governing-supported ordinary rows retained exact slot identity, assigned time,
and AltAz tangent coordinate. Assigned-time and along/cross-scan displacement
are exactly zero; the full-slot reassignment rate is zero. The 110 union-edge
rows have no governing paired baseline and remain explicitly unavailable, not
zero.

The trajectory-derived half/full-slot displacements are useful sensitivity
descriptions, not tolerances. The Pointing 152389 `34.062668 us` half-cell
margin remains solely an engineering unique-slot margin. It is not a physical
timestamp error or sky-placement accuracy claim.

No selected producer authority identifies detector timestamps as integration
start, end, centroid, capture, packet formation, or another physical event.
The 8.192-ms cadence relation does not prove a contiguous exposure window.
Absolute assigned-time-to-integration-centroid error and absolute physical sky
placement therefore remain unavailable even though governing-to-candidate
differential placement is exactly unchanged.

## SKY-Q1--Q5 disposition under existing owner authority

The addendum refines the validation framing but creates no new scientific
choice that must block phase one.

1. `SKY-Q1-PHYSICAL-TIME` is resolved by the approved D001 compatibility
   boundary and the owner's proportionality judgment. Phase one may preserve
   relative existing-use timing while absolute physical timestamp/sky
   correctness remains explicitly unavailable. A versioned producer ICD is a
   future amendment trigger, not a prerequisite for this repair.

2. `SKY-Q2-ANGULAR-GATE` is resolved by the D005 changed-product rule. Require
   exact zero ordinary assigned-time/coordinate displacement and the existing
   exact Pointing/Beammap policy for unaffected behavior. Any nonzero
   scientific change returns to the owner; no angular allowance is inferred
   from cadence, residuals, or this cohort.

3. `SKY-Q3-MAP-SENTINEL` is bounded as validation, not implementation.
   Naturally produced exact whole-application Pointing/Beammap products under
   the existing 1-arcsec fixed-WCS/fixed-JINC configuration may serve as
   downstream sentinels in the later control/successor campaign. Do not add a
   special map requirement or campaign. The 1/2-arcsec calculations remain
   descriptive sensitivity evidence; do not create a second map reduction,
   modify gridding, or broaden the repair into SCI-MAP. Zero input-coordinate
   change already supplies the phase-one local coordinate result.

4. `SKY-Q4-MISSING-RATES` is resolved by the D005 sequencing decision. Exercise
   0.5x/1x/2x/4x algebra synthetically, use native 1x observational evidence in
   phase one, and leave native 0.5x/2x/4x sky evidence pending without
   resampled substitution or production expansion.

5. `SKY-Q5-HOLD` is resolved by `D005-Q1-HOLD`. Preserve the named existing-use
   compatibility adapter without selecting physical bit or transition-side
   semantics. The owner's telescope-engineer request follows the future
   explicit-amendment path and is nonblocking.

## Authorization and next gate

The bounded phase-one application implementation and local-validation
authorization remains unchanged. This review adds the sky-domain package as
preregistered evidence and clarifies its gates; it does not widen repair scope.

The dedicated ALIGN task may now continue on the exact clean descendant
`bfffe0e60fa8ce05a75ae34b89383bceaadb8fc2`, implementing only the approved
handoff and decisions. It must stop after local gates with the exact candidate
SHA, evidence digest, exact-difference disposition, runtime disposition, and
any owner-return item.

This review does not authorize Unity contact or execution, repair acceptance,
re-audit launch/execution/completion, merge, rebase, push, new production
profiles, physical timestamp or `Hold` semantics, nonzero scientific
tolerances, non-1x production, polarization/HWPR science, MAP implementation,
or production expansion.
