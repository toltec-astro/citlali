# Independent exact-SHA review: pass with recorded limitations

Reviewer /root/blank_fit_review began from fresh context and operated read-only.
This record transcribes the review disposition. Reviewed candidate:
1761e943ff26ceb9998f3c2687e7a37bddc6808a; tree:
af640ed86bdaf50767697f0b7600f9ed26659947; base:
2aae3c5608e6fff7ce0e55337942a5ea25c51998. Worktree was clean at review.

Scientific/behavioral: pass. Independently checked all18 archived outcome
matches, nine failures at3000 evaluations, three probes still failing at6000,
identical probe starts/settings except budget, and finite parameters/residuals
throughout. Recomputed the three objective changes and full archived census.
The tiny deterministic sample is not representative of all failures; it does
not identify a universal conditioning discriminator. A successful blank control
has a more ill-conditioned normalized Jacobian than the sampled failed fits.

Architecture/ownership: pass. The existing fitter owns calculations; diagnostic
instrumentation preserves its objective and restores the intercepted function
in a finally block. No Engine, application algorithm, source-presence decision,
threshold or feedback changes. The three probes are diagnostic only.

Repository/evidence hygiene: pass. Three changed paths at the reviewed source
match scope. Verified13 consumed-parent digests, predecessor manifest identity,
diagnostic manifest and final report manifest. Initial diagnosis executed at
970a86dbcb1ad8a86b6025c91cfcecbf0f15b986; subsequent source adds reporting only.
The reviewer inspected the legible trace figure and numerical report. The
requested “function-evaluation budget” wording correction was applied; it
avoids confusing nfev with iterations or finite-difference residual calls.

Report SHA256:
3a5a47af4e959063452d0dc0c8a2abf43cfb4f966210b7865934c51a1d829b71.
Figure SHA256:
2b1fa7d664007f92c4864177dcb79e3e3596d781552ab8ee78ef8fe2d2196edd.
The corrected report's figure is byte-identical to the inspected first render.

No unresolved findings. Reviewer ran no additional fits or noise realizations.
Ten focused correlation-admission tests pass;18 exact replays and3 probes are
the direct diagnostic checks. No native/build/config code changed, so those
broader gates are inherited evidence, not newly rerun claims. Existing local
Python environment only; no Unity or Spack application validation claim.

No universal failure cause, source-absence rule, calibrated threshold or
production qualification is established. The predecessor's six admission
gates remain unavailable. Final completion records the separately verified
closure seal. No integration, merge, push, cleanup or production activation.
