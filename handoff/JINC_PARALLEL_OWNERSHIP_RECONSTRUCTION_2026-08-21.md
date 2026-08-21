# JINC Parallel Ownership Reconstruction — 2026-08-21

## Verdict

The independently re-audited JINC parallel ownership repair
`e6c8d126157674a9990abc8d1e96ce2dd69f9374` has been reconstructed against the
Unity-tested convergence implementation and passes every required local gate.
It is admitted as fail-closed contract hardening for existing detector-grouped
JINC use. It does not broaden JINC algorithms, grouping, production scope, or
the targeted `redu04` science claim.

## Identity and ancestry

- convergence branch: `codex/converge-apt-align-jinc`;
- reconstruction parent: `c67f12120ca29f0c2d603fd551146635ef7b3782`;
- Unity-tested implementation ancestor:
  `e77460cffad49387795009539d6abc7e370e8b58`;
- historical repair authority:
  `e6c8d126157674a9990abc8d1e96ce2dd69f9374`;
- independent re-audit:
  `f541d81a266fce0f7baed58e9ec275dadba260ee`;
- accepted Unity evidence:
  `handoff/JINC_WORKING_SUPPORT_UNITY_VALIDATION_2026-08-21.md`.

The historical source commit was not cherry-picked. Its behavioral contract
and focused test matrix were applied to current state so the accepted
working-support repair and newer JINC product lifecycle remain authoritative.

## Implementation scope

The application/test reconstruction is exactly three paths:

1. `include/citlali/core/mapmaking/jinc_mm.h`;
2. `tests/test_jinc_parallel_ownership.cpp`;
3. `tests/CMakeLists.txt`.

Before contribution-diagnostic allocation or `grppi::map`, the parallel entry
now requires:

- one map-index entry per detector;
- a valid signal destination for every detector;
- unique detector ownership of every selected map slot;
- correct cardinality and shape for signal, grid-weight, realized-weight,
  optional coverage, optional kernel, and selected noise destinations; and
- correct cardinality and shape for the current JINC absolute-denominator and
  contributor-count destinations.

The last item is the only adaptation beyond the historical repair: these
conditioning planes are written by the current implementation but did not
exist in the old candidate. Formal-support state is not written by population
and remains governed by finalization.

Invalid input is rejected before diagnostic allocation, observable accumulator
mutation, output side effects, or parallel work. Valid-path arithmetic,
accumulation order, active-map handling, cache geometry, support policy, and
noise behavior are unchanged. The worker suffix beginning at `grppi::map` has
SHA-256
`5d3030566e7616139c73061f8f7556078a4e2e5b9be504577fc2fa6466309ccf`
in both the reconstruction parent and working candidate.

## Validation

The reconstruction was configured in a fresh disconnected build at
`/private/tmp/citlali-converge-apt-align-jinc/build`, using the accepted local
dependency-source set, AppleClang 21.0.0, Release mode, and OpenMP 5.1.

| Gate | Result |
| --- | --- |
| Ownership executable, `OMP_NUM_THREADS=1` | 6/6 passed |
| Ownership executable, `OMP_NUM_THREADS=2` | 6/6 passed |
| Ownership executable, `OMP_NUM_THREADS=4` | 6/6 passed |
| Ownership executable, `OMP_NUM_THREADS=8` | 6/6 passed |
| Focused current JINC CTest selection | 28/28 passed |
| CLI build | passed |
| Complete CTest | 732/732 runnable passed; one established disabled test not run |
| Baseline-tool unit suite | 203/203 passed |
| Required config preflight | 127/127 unit tests; all four mode kits; 8/8 compact cases, zero skips; 100% surface coverage; all audits passed |
| Validation ledger | valid, 60 records |
| Intended-science-change ledger | valid, 3 changes and 5 integration commits |
| Diff hygiene | `git diff --check` passed |

The six focused cases cover eager duplicate rejection, cardinality and index
rejection, all current destination failures without mutation, exact serial
equivalence, exact sequential/OpenMP repeated results, and permitted unused
map slots. The invalid-state snapshots now include the current JINC
conditioning planes and formal-support plane as well as the historical map,
noise, and contribution-diagnostic planes.

## Unity and promotion boundary

No Unity rerun is required by the convergence audit's routing rule. The
accepted `redu04` Beammap used detector-grouped JINC mapmaking; the independent
ownership disposition already establishes unique per-detector destinations
and sequential scan invocation for this existing use; and the valid worker is
byte-identical. The reconstruction can only reject an input that violated that
governed contract before any output exists. It cannot change valid `redu04`
arithmetic or products.

The targeted observation-148670 Unity evidence at `e77460cff` therefore
remains applicable. This does not fill in the four deliberately unavailable
large FITS products, create a complete accepted-run record, authorize general
JINC production, or authorize a push or merge.

## Next integration action

The next admitted convergence task is a separate compact-v2 native-consumer
reconstruction plan derived from the intended behavior of `fd3627fc7` and
`9d9d55a54`. Their canonical APT v1 implementation must not be copied or
cherry-picked. Implementation requires a clean compact-v2 design and an
independent review before it can join application ancestry. The optional PTC
metadata lane remains separate.
