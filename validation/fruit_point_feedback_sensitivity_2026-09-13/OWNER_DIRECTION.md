**Run one small, paired feedback-sensitivity experiment using the solutions we already saved.** I would not change the solver, introduce a regularizer, or launch another full campaign yet.

The question should be:

> **Do the differences between the nominal and tighter starlet solutions actually change what FRUIT learns and returns—or are we rejecting them mainly because of an overly sensitive image-level statistic?**

That is the missing piece needed to choose the next method change intelligently.

## Why this comes next

I should qualify something from my previous answer: **poorly constrained reconstruction is a plausible explanation, not an established diagnosis.** The reported 67% change in image peak does not, by itself, prove that we need regularization. Nor does it establish that the change damages pointing or amplitude recovery.

There are several possibilities. The difference might be concentrated in a few pixels that matter little after projection. It might be coherent source structure that materially changes the next PCA calculation. Or the nominal stopping tolerance might simply leave substantial progress unfinished in an otherwise adequately constrained problem.

The gradient test cannot distinguish those possibilities. It measures numerical stationarity, not errors in astronomical measurements or sensitivity of the feedback loop. SciPy’s documented projected-gradient stopping condition likewise does not provide such a scientific error bound. 

**We should measure the consequence before choosing the cure.**

## The experiment I would authorize

### 1. First locate the instability in the saved solutions

For each saved nominal/tighter pair, form

\[
\Delta m = m_{\mathrm{tight}}-m_{\mathrm{nominal}}.
\]

Use the available pairs to answer three closely related questions:

- **Where does the difference live?** In the source core, broad wings, isolated pixels, or poorly supported regions? Report the locations of both image maxima—not just their percentage difference.
- **Which spatial scales change?** Is this mainly fine-scale structure, or does the coherent, beam-scale source change?
- **Does the change improve the actual objective appreciably?** Does the reconstruction move substantially while its predicted selected wavelet coefficients barely change, or does the tighter solution genuinely fit those coefficients materially better?

That last comparison is a direct way to investigate my underconstraint hypothesis. A large image change with little change in the quantities the objective constrains would support it. It would not, on its own, prove an exact mathematical null space.

This part requires **no new optimization and no PTC relearning**.

### 2. Project the differences through the actual feedback path

For the same pairs, calculate

\[
\Delta d=\Pi(m_{\mathrm{tight}})-\Pi(m_{\mathrm{nominal}}),
\]

where \(\Pi\) is the **existing map-to-timestream projector**, with the actual sampling, validity, and calibration conventions.

Do not add an extra smoothing operation to make the difference look smaller.

This tells us whether the instability survives into the quantity that is actually subtracted before PTC learning. Examine its concentration in source-crossing samples and its coherence across detector groups, rather than reducing everything to one whole-observation RMS.

In particular, **“smaller than the noise in an individual sample” should not be our clearance criterion**. The experiment needs to test the effect on the correlated-mode calculation and the recovered source, not assume that a sample-level comparison settles it.

### 3. Do a paired, one-step PTC replay on three selected states

I would choose:

1. One ordinary compact-source case.
2. The compact/mildly degraded case with the largest relevant reconstruction change.
3. One real-data state with both saved solutions available.

For each state, start from **the same parent timestream and the same preceding FRUIT state**. Make two diagnostic branches: one uses the saved nominal model; the other uses its tighter counterpart.

In each branch, subtract its model, **relearn PTC normally**, clean, and make the next reconstructed-total map using the normal model-restoration path. Do not freeze the PCA basis: its response to the different subtraction is precisely what we need to test.

That is **six one-step branch evaluations**, not another full trajectory campaign. The unqualified models remain diagnostic inputs; this does not admit them into production or overwrite the previous failed qualification.

## What I would measure

The primary outputs should be **the changes in the next map’s fitted peak and centroid**, plus truth-scored errors for the injected cases. Neither solution should be treated as astronomical truth merely because it was optimized more tightly.

I would also compare the removed correlated-mode **subspaces** and the remaining source morphology. Individual eigenvectors are a less useful comparison because their signs—and their orientations within nearly degenerate subspaces—can differ without representing a substantive change.

Keep the already proposed **0.5% peak and 0.1″ centroid numerical allocations** visible as reference scales, alongside the full **5% and 1″ operational budgets**. But do not retrospectively replace the original image-stability gate and declare the candidate passed.

The purpose is to establish the size and mechanism of the effect. **A one-step result cannot establish stability of a complete FRUIT trajectory.**

## How this determines the next move

**If the image differences have little effect on the projected feedback or next POINT measurements**, then our image-peak stability test may be stricter than the intended use requires. The next step would be a prospectively defined, feedback-relevant stability criterion, followed by the bounded operational comparison. There would be no demonstrated reason to redesign the estimator solely to make every pixel settle.

**If the differences materially change PTC and the next source measurement, while barely changing the constrained wavelet evidence**, then we have a much stronger case that the reconstruction admits consequential freedom. That would justify **one explicitly constrained or regularized starlet candidate**, targeted at the identified unstable structure—not another adjustment to `maxiter`.

**If the tighter solution materially improves the fit to the constrained evidence**, then the immediate problem may instead be the numerical criterion or conditioning. That would warrant a numerical remedy rather than an unmotivated scientific prior.

In every outcome, the evaluator improvements should remain, and the candidate still needs to demonstrate a useful scientific advantage—not merely become easier to declare converged.

## My recommendation

**Spend the next increment identifying the effect of the instability, rather than trying another remedy.** The existing saved solutions give us a controlled perturbation experiment without inventing a new candidate.

I would give Codex this direction:

> Preserve the evaluator repair and the failed stopping qualification. Using saved nominal/tighter solutions, locate the reconstruction differences, assess how strongly the frozen objective constrains them, and project them through the existing feedback path. Then perform paired, single-step PTC relearning on three selected saved states. Report the effect on the next reconstructed-total map’s peak, centroid, and source fidelity. Do not alter the estimator or acceptance rules. Recommend a formulation change only if the diagnostic identifies consequential freedom in the feedback reconstruction.

That is the shortest useful route from **“the images differ”** to **“we know what to change, and why.”**