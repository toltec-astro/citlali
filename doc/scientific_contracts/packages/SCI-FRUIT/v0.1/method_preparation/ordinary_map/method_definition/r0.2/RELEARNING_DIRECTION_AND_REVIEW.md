# Residual-relearning direction and review disposition

Owner: Grant Wilson. Date: 2026-09-08.
Decision: `SCI-FRUIT-OD-RESIDUAL-RELEARNING-DIRECTION-2026-09-08`.
Status: residual-relearning direction and targeted paper successor authorized.

The owner supplied the [ChatGPT discussion](inputs/manager/RESIDUAL_RELEARNING_DISCUSSION_2026-09-08.txt)
for review. The manager recommended a targeted successor adopting residual
relearning as the proposed schedule while preserving the frozen generic core
and execution gates. The owner replied: **“sounds good”**.

This assent authorizes that revision and resolves the intended learning
schedule for the paper method: estimate PTC state anew from each iteration's
model-subtracted residual under the same declared recipe; hold the resolved
state fixed for that iteration's application. It does not approve the finished
method packet, PTC/MAP amendments, selector, gridding coefficients, support
value, rank value, observation, executable, experiment or author dispatch.
The earlier [scope approval](OWNER_SCOPE_APPROVAL_2026-09-08.md) remains in force.
The already accepted learning direction is not presented for a second decision.

## Scientific reason and limits

Astronomical signal can influence both the signal removed by a cleaner and
the subspace estimated during learning. A fixed-bootstrap cleaner addresses
model protection but cannot revise its estimated subspace. Learning from a
model-subtracted residual addresses that additional mechanism. An imperfect
model can also subtract nuisance structure from learning and protect it at
rejoin; neither schedule guarantees better recovery.

The manager's review checked the discussion's published support. Downes et al.
found that injected sources changed PCA eigenvectors even when the removed
mode count did not change. That supports the mechanism, not superiority of
this FRUIT candidate. [Downes et al. (2012)](https://academic.oup.com/mnras/article/423/1/529/1747027)
Chapin et al. describes improved subsequent common-mode estimates after
astronomical subtraction, and map/common-mode degeneracies. This is a related
principle in a different estimator. [Chapin et al. (2013), §§3.2.1 and 4.1](https://arxiv.org/pdf/1301.3652)
These links and the discussion are manager review context, not new independent
author inputs or a substitute for frozen PTC science.

The discussion's simple covariance and linear-cleaner equations are conditional
illustrations. This method retains the exact masked/group-local PTC recipe,
its per-pass centering, affine reference loss and joint model/parent dependence.
Relearning is one method with new realized states. Changing the learning rule
would change its identity. No adaptive rank, support-changing refinement,
noise-derived gridding weight, source threshold or required subspace change
follows from this direction.

## Targeted changes from method-definition r0.1

| Item | r0.1 | Current r0.2 disposition |
| --- | --- | --- |
| Method identity | `ordinary-map-fixed-bootstrap@r0.1` | `ordinary-map-residual-relearning@r0.2`; a different proposed FRUIT method, not a state update to the old one. |
| Learning parent | Bootstrap CAL only | CAL at k=0; fresh FRUIT residual at every k>=1, reconstructed from that iteration's calibrated parent and applied model. |
| Learned state | Bootstrap centering/subspace carried unchanged | Same centering/fit rules evaluated anew per pass; current Theta_k fixed during its own Apply step. |
| BC-IN | Proposed residual application permission | Proposed residual learning, resolution and application permission; exact upstream amendment still pending. |
| Response/state retention | Bootstrap fit plus later fixed-state applications | Per-pass learning influence/Theta_k and full-procedure dependence, with separate conditional fixed-state queries. |
| Other rules | Proposed model/projection, unit gridding family, fixed populations, extent and failure | Retained as proposals with their existing gates; not approved by the learning-direction assent. |

The predecessor packet at `b3a3af55ecfaa3fcba6be3ab196acdf24e5ff151` and its
archive remain byte-exact, SHA-256
`4ac6897806c45c395f90825a91d14b2e475b57b68339210b1fb7bcfb39d8c56e`.
Its fixed-bootstrap recommendation is superseded for this intended first
method, while remaining a meaningful alternative. No alternative study is
launched. Historical JINC control and every frozen source/product are preserved.
