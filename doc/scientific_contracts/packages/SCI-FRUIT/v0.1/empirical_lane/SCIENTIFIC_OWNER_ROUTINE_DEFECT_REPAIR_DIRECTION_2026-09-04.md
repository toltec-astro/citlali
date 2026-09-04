# Scientific owner direction — routine defect repairs

Date: `2026-09-04`

The scientific owner stated:

> In the future, do not stop for my approval following simple bugs. Just fix
> the bug and push on. I only need to be involved in significant decision
> points.

For this program, a narrow implementation defect discovered during an already
authorized analysis or validation run may therefore be repaired, tested,
documented, and followed by continuation of that run without a new owner
decision when all of the following remain unchanged:

- the scientific method and interpretation scope;
- every registered scientific gate, numerical bound, region, and trigger;
- the registered external inputs and retained products;
- the authorized external reduction or Citlali replay count; and
- the algorithm and production configuration under study.

Examples include parser type handling, plotting or output-format defects, and
serialization plumbing that do not alter scientific content. Owner review is
still required whenever a proposed repair would change a scientific gate,
method, interpretation, external input, reduction or replay count, algorithm,
configuration, or another material scope decision. A repaired local analysis
or validation may be rerun as needed to complete the already authorized work.
