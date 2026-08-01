# SCI-CAL-001 APT identity decision amendment — 2026-08-01

Status: approved project-owner clarification; supersedes the identity-only
clauses of `CAL-D002`, the `SCI-CAL-001-F004` repair handoff, and the
corresponding `SCI-AST-001-D001` interpretation

Affected findings: `SCI-CAL-001-F004`, `SCI-AST-001-F013`

## Owner clarification

The project owner believes the existing association by row is correct, while
recognizing that its evidence is weaker than the end-to-end astrometric
evidence. There is no known tone-frequency-to-design-detector mapping that is
100 percent correct. Prior TolTECA/TolProj/TolAPT work improved diagnostics but
did not produce a dramatically better independent identity provider and is
currently paused.

This clarification rejects two over-broad requirements in the earlier
coordinator interpretation:

- a table row can be an admitted observation-local acquisition binding when
  the producer's ordered-row contract is proven at the consumer boundary; and
- a supposedly universal stable UID or exact design match is not required for
  ordinary AST/CAL use of measured Beammap quantities.

It does not make unvalidated row order scientific identity, claim that a tone
match is exact, or close either affected finding.

## Read-only supporting evidence

The following local evidence was reviewed without modifying its repositories.

- `tolteca-tone-match-lab` was clean at commit
  `2c66784868f6aab83f23f254dd53c962aa6ecf98`. Its
  `docs/data_contract.md` (SHA-256
  `ab7fe938c4e992a00a5f57e6fe886ca02957b35b6401d5a027362d686cba2ab2`)
  reports that, for all 344 tested observation/network pairs, raw, processed,
  text, and raw Beammap APT row counts agree; raw Beammap `kids_tone` is the
  zero-based Tune row; and Beammap `fp`, `fr`, and `Qr` equal the reduced Tune
  values. Across 165,783 rows the recorded raw, processed, Tune, and raw
  Beammap frequency relations are exact. This is a same-observation
  acquisition association, not a cross-Tune detector identity.
- The same contract reports that identical parent/child tone-bank order does
  not establish detector identity across Tunes. In 926 tested cases with
  endpoint `fp` disagreement above 200 kHz, the same-`kids_tone` identity was
  wrong in every case.
- The production-algorithm `fp`-to-`fp` benchmark,
  `docs/tolproj_production_fp_fp_benchmark_2026-07-22.md` (SHA-256
  `93ad3cccc7d522076d932ca49ae0805ab663508bd2a13af4c0b312fa31b9f482`)
  found 1,265/1,325 correct, or 95.47 percent precision at 98.59 percent
  coverage, on four same-VNA positive-control directions. It is not a global
  matcher accuracy or the ordinary operational frequency contract. The
  full-reference-age study,
  `docs/tolproj_full_reference_age_benchmark_2026-07-22.md` (SHA-256
  `db8cde67d5eeabea08d6f863b08b5e59d7084ce134b0cddfa4e1f477926160c4`)
  reports different results for broader cohorts. The residual-addressability
  study, `docs/residual_addressability_study_v1_2026-07-28.md` (SHA-256
  `736af7caf817c876565d00a8d016681540cbc95ec83fdca1713ab5aac4a2057c`)
  records outcome B: no additional identity provider was promoted because an
  adequate independent physical anchor was unavailable.
- TolProj was clean at commit
  `74395c824860ca41410dde5cf2e0272e5535fc19`. Its matched-APT writer
  `tolproj/legacy_scripts/make_matched_apt.py` (SHA-256
  `d18325e34f444b26a9e1fee2220f7e44167444e83a4c99a62816a75f7d661d9d`)
  preserves the target observation's row spine, records the selected source
  Beammap row separately, transfers source metadata, and sorts the result back
  into target-row order. Commit
  `2ad6821aed442eaafc7667dcca6bed6d7a290230` specifically corrected confusion
  between matcher row indices and `kids_tone` values.

These observations are supporting local evidence, not immutable audit truth.
They must be reproduced or replaced by digest-bound fixtures on the selected
repair base before a finding can close.

## Approved layered identity contract

### 1. Observation-local acquisition identity

The exact acquisition key is the observation/Tune identity, network or
interface identity, and network-local `kids_tone` or equivalent row slot. A
global TOD column is only a dense locator in that exact assembled observation.

`verified_row_order` is an allowed compatibility binding. Admission must prove
the raw-file and interface order, network headers, tone counts and order,
per-network APT counts and order, uniqueness of the acquisition key, and exact
observation and artifact provenance. Reordered rows without an explicit
mapping fail closed. An explicit keyed mapping may replace this mode.

### 2. Measured Beammap APT identity and association

A measured Beammap row is identified inside one immutable artifact by its APT
digest/version, Beammap observation identity, network, and measured local row
or UID. Fitted `x_t`, `y_t`, `responsivity`, `flxscale`, `sens`, and beam
quantities attach to that identity.

For the Beammap's own raw data, the acquisition-to-measured binding may be the
verified same-observation row/tone relation. For another target observation,
the target-tone-to-source-Beammap-row association is a separate matcher edge.
A matched APT may preserve the target acquisition row spine exactly while the
source-row assignment remains imperfect. Its provenance must therefore retain
the target and source keys, matcher and version, frequency fields and
transform, selected source row, match/abstention state, and any valid quality
or uncertainty evidence. Aggregate benchmark precision is not a per-row
probability.

### 3. Design identity

A design key is a design-catalog digest/version plus design ID. The
measured-Beammap-to-design edge is a separate TolAPT match result. It must
retain method/version, candidate and cost information where supplied, the
defined meaning of any confidence field, and matched, unmatched, or ambiguous
state. A local UID, versioned common UID, heuristic confidence, or aggregate
accuracy must not be relabeled as certain physical identity.

Ordinary AST/CAL use of measured Beammap quantities does not require a design
match. A consumer of design-derived metadata must declare its own admission
rule and treat an inadmissible or unavailable design match as unavailable or
fail closed. The accuracy and uncertainty policy for cross-observation
source-row matching remains distinct from design identity and from exact
target-row admission.

## Finding and repair consequences

`SCI-CAL-001-F004` remains open. Its closure question is whether each raw TOD
column is admitted to the correct target-row record and whether every applied
Beammap quantity retains its source-row association and validity. A verified
row-order mode or explicit keyed mode may close the acquisition-binding part;
perfect design matching may not be demanded or claimed.

`SCI-AST-001-F013` remains open. Its closure question is the same binding plus
focal-plane basis/version, `x_t`/`y_t` size and finiteness, and coordinate
consumer validity. Design identity is required only if design-derived geometry
is selected.

Required falsification tests include network/file reorder, APT-row reorder,
missing/extra/duplicate tones, tone-frequency disagreement, missing network or
subset, duplicate acquisition keys, and forced unmatched/ambiguous matcher
states. Row-mode reordering without a mapping must be rejected; explicit-key
mode must be permutation invariant. Changing or omitting a design ID must not
move coordinates or alter calibration when only admitted measured Beammap
fields are consumed.

This amendment changes the repair closure contract, not the audit findings,
implementation verdicts, production restrictions, repair authorization,
Unity-evidence state, or re-audit state.
