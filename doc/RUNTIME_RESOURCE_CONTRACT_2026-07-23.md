# Runtime CPU Resource Contract

## Purpose

Unity jobs can wait one or two hours for a Slurm allocation. A deterministic
CPU mismatch should therefore be rejected before submission when possible, but
it should not waste an allocation if it reaches Citlali at runtime.

This contract coordinates TolPROJ's generated `02_redu.sh` scripts with
Citlali's typed runtime configuration without making either project depend on
the other's implementation.

## Requested, Available, Effective, And Realized

- **Requested threads** are the final
  `reduce.steps.0.config.low_level.runtime.n_threads` value after numbered YAML
  precedence.
- **Allocated CPUs** are TolPROJ's generated Slurm
  `#SBATCH --cpus-per-task` value.
- **Available threads** are discovered by Citlali from
  `SLURM_CPUS_PER_TASK`, process CPU affinity, and hardware concurrency.
  Citlali uses the most restrictive reliable scheduler/affinity limit.
- **Effective threads** are the requested count capped to the discovered
  available count.
- **Realized threads** are the values actually installed in OpenMP, Eigen, and
  FFTW.

Requesting fewer threads than the allocation is valid. Requesting more threads
than the allocation is a pre-submit error in TolPROJ and a recoverable runtime
adjustment in Citlali.

## TolPROJ Responsibilities

For refactor mode, setup commands use the preserved
`71_MODE_runtime.yaml` thread count for the generated Slurm allocation. An
explicit `--cpus N` updates that one runtime value without reformatting or
discarding operator comments, and writes the same count to `02_redu.sh`.
Legacy setup retains its historical defaults.

Before submission:

```bash
tolproj validate-reduction .
tolproj submit-reduction .
```

`submit-reduction` validates the final numbered-config thread request against
the generated script and calls `sbatch` only when the request fits. Direct
`sbatch 02_redu.sh` remains supported, with Citlali providing the runtime
safety net.

## Citlali Responsibilities

At the CLI boundary, Citlali discovers the resources visible to the process
and resolves the typed runtime plan before configuring thread libraries.

When the requested count exceeds availability, Citlali:

1. emits one warning containing requested, available, source, and effective
   counts;
2. uses the capped count for OpenMP and non-Wiener FFTW planning;
3. retains Eigen's established single-thread policy and Wiener's established
   single-thread FFTW planning policy;
4. continues the reduction; and
5. records requested, discovered, effective, adjustment, and realized state in
   `citlali-runtime-provenance-v2`.

Malformed or non-positive configured thread counts remain configuration
errors. Resource discovery failure does not invent a cap; Citlali preserves
the validated request and records availability as unavailable.

## Validation And Closeout

Local gates cover config precedence, setup synchronization, pre-submit
rejection, submission gating, resource selection, capping, warning count,
thread-library plans, and versioned provenance.

The 2026-07-23 local implementation checkpoint passes the Citlali CLI build,
all 500 CTests, all 119 baseline-tool tests, all 118 config-preflight tests and
the complete four-mode preflight, plus all 147 TolPROJ tests and Ruff checks.

Retained-debt item D16 closes only after Unity confirms:

1. a normal generated reduction with matching CPU/thread values; and
2. an intentionally mismatched direct submission that logs the cap, completes
   successfully, and records the expected V2 provenance.
