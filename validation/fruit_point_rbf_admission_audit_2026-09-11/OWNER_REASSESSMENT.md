**On reassessment, I still agree with rejecting the two tested RBF candidates—but I am less convinced by the diagnosis I gave earlier.** The results establish that those particular **estimator–admission combinations** did not work adequately. They do not establish that dense RBFs are fundamentally the wrong representation, or that simply using fewer, broader RBFs would solve the problem.

**Wavelets are a worthwhile next candidate**, particularly because they let us treat structure at different spatial scales differently. But I would first make a small diagnostic pass over the existing results so that we do not reproduce the same admission failure with a different basis.

I am taking the numerical outcomes from the Codex report quoted here. I recovered the original experiment directive, but not the complete final report and result tables, so some of the causal interpretation below remains provisional.

## 1. What the RBF experiment actually established

There are several distinct questions here, and the experiment answered them with different levels of confidence.

| Question | What the reported evidence supports |
|---|---|
| Can the RBF basis express the relevant PSFs? | **Yes, in the noiseless representation tests.** |
| Can the weakly regularized estimator extract a usable model from these noisy maps? | **Not with the tested admission procedure.** Synthetic-source models were rejected. |
| Does stronger regularization solve that problem without unacceptable bias? | **Not at the tested strength.** It improved admission in some compact cases but introduced substantial noiseless fitted-model error. |
| Does either complete candidate deliver acceptable POINT recovery? | **No.** Reported compact amplitudes remained 12–23% low. |
| Have we meaningfully tested recovery of detectable comatic structure? | **Not convincingly across arrays.** The report explicitly says detectability was not established. |

The distinction between the first and third rows matters. The large noiseless errors after increasing the penalty were **errors introduced by the regularized estimator**, not an inability of the underlying basis to represent the PSF. My earlier explanation blurred that distinction.

The runtime result also provides no reason to favor these candidates: the reported revised-RBF time of 7.84 seconds versus 6.56 seconds for the matched Gaussian control is about a **20% overhead**, without a demonstrated recovery benefit. That is a result for the reported harness and timing convention, not a production-pipeline benchmark.

And “378 passes without trajectory failures” is an implementation result, not 378 independent scientific tests: it is two configurations applied to nine cases, three arms, and seven successive passes.

### The coma test missed an important part of its intended purpose

The original directive specifically called for **at least one regime where the source is detectable but important structure is faint**. That is the regime needed to ask whether spatial coherence helps recover shoulders and tails. 

A source whose *entire presence* is uncertain asks a different question: can the procedure detect an unknown extended object at all?

Both questions are legitimate. But failing the latter does not cleanly answer the former.

Consequently, I would retain the negative result as:

> These candidates failed on the chosen comatic injections, but the experiment did not isolate their ability to recover faint structure around a reliably detected non-Gaussian source.

That is a limitation of the experiment—not an excuse to adopt the methods despite their compact-source failures.

## 2. The admission mechanism deserves more scrutiny than I initially gave it

This may be the most important reassessment.

### A source can be detectable before its detailed morphology is reproducible

The reported RBF gate required both held-out scores to exceed 5 and agreement between the fitted fields to exceed a cosine of 0.8.

Those conditions do not ask exactly the same thing.

A detection score can ask:

> Is there evidence for astronomical signal in this region?

Agreement between two flexible reconstructions asks something closer to:

> Can both subsets already estimate substantially the same spatial distribution?

The second can be considerably harder. A broad source might be detected collectively while its individual shoulders, secondary lobes, and tail remain poorly determined. Rejecting the *entire source* because those uncertain details disagree may defeat the purpose of coherent inference.

This does **not** mean that the gate should simply be relaxed until something passes. It means that its detection power should be checked separately from its ability to reject noise.

Also, I need to correct my earlier description of the splits: **pixel-parity subsets are not automatically independent data sets.** They can be useful for checking interpolation or local predictive consistency, but independence depends on the map-noise covariance and shared upstream processing. Spatially correlated data can give misleading validation results when split without accounting for those dependencies. 

The converse is important too: disagreement does not uniquely prove overfitting. It can reflect limited signal-to-noise, uneven coverage, interpolation behavior, or genuine estimator variance. “The flexible fits followed noise” is a plausible diagnosis, but not the only one established by a low cosine.

### There is a possible FRUIT bootstrap problem

Under the accepted recurrence, each pass returns to the same parent data, subtracts the current sky model, relearns PTC, cleans, restores the model, and reconstructs a total map. The next feedback model is a complete replacement inferred from that map. 

That gives the following potential failure loop:

\[
\text{source attenuated by first-pass cleaning}
\;\rightarrow\;
\text{feedback rejected}
\;\rightarrow\;
\text{same unprotected source on the next pass}.
\]

If the input, cleaning, inference, and decision are deterministic, and the admitted model remains zero, **repeating the pass does not create additional evidence**.

This is not a reason to feed back unsubstantiated structure. It is a reason to ask whether the initial admission rule can recognize the *attenuated* signal that it actually receives.

There is another bookkeeping point worth checking in the existing records: **when did the two compact cases first receive an admitted model, and when was that model first applied?** In the visible code, `next_model` is distinct from `applied_model`. Admission after the seventh map cannot improve that seventh map; it would affect the eighth pass. Late admission would still be an operational failure within the seven-pass budget, but it would be a different failure from “an applied RBF model gave poor recovery.”

### Stronger smoothing may have addressed the wrong aspect of agreement

Increasing the penalty can make two reconstructions agree by suppressing their differences. That does not necessarily make either reconstruction more accurate.

The reported noiseless fitted-model errors show that the revision paid a substantial fidelity cost. So I would not interpret the improved cosine as evidence that the underlying inference problem was nearly solved.

My revised diagnosis is:

> The weak candidate did not produce sufficiently admissible feedback; the strong candidate traded away substantial fidelity without delivering acceptable output recovery. We have not yet separated inadequate detectability, estimator variance, and overly demanding whole-model admission.

## 3. Why wavelets are a sensible alternative

The attraction is **not** that wavelets magically have fewer coefficients. An undecimated wavelet representation is redundant and can have more coefficients than image pixels. Its value is that it organizes the image into localized structures at different scales, making scale-dependent inference possible. 

For a first test, I would favor an **undecimated starlet transform**, rather than start by comparing a collection of wavelet families.

A useful way to think about it is to construct successively smoothed versions of an image:

\[
c_0=M,\qquad c_j=h_j*c_{j-1},
\]

and define the detail at each scale by

\[
w_j=c_{j-1}-c_j.
\]

The image is reconstructed as

\[
M=c_J+\sum_{j=1}^{J}w_j.
\]

Thus the transform separates fine detail, intermediate-scale structure, and a broad component without initially discarding any of them. Starlet implementations use this multiscale decomposition and admit fast algorithms whose work grows approximately linearly with the number of pixels times the number of scales. 

For our problem, the conceptual advantage is straightforward:

**A compact core need not be smoothed as strongly as uncertain fine structure in a broad wing, and a broad shoulder need not clear a pixel-scale detection threshold.**

A shoulder that is weak in every individual pixel may have a useful collective signature at an intermediate scale. Meanwhile, a genuinely narrow core can retain its fine-scale coefficients. That is a more direct way to express the intended prior than requiring all neighboring RBF amplitudes to vary smoothly with one penalty strength.

This remains a hypothesis for the TolTEC maps. Atmospheric residuals can also be spatially coherent and occupy those same scales.

### “Isotropic” does not force a symmetric PSF

The starlet’s local filtering is approximately isotropic, but the coefficients vary independently with position. An asymmetric arrangement of coefficients can describe an asymmetric image. It does not force the reconstructed source to be circular.

I would start there rather than immediately introduce directional transforms. Whether a more directional representation is needed should be decided from a specific failure to recover the coma, not from the fact that coma is asymmetric.

Using an undecimated transform also avoids the downsampling that makes critically sampled wavelet methods sensitive to grid alignment. It does **not** eliminate the need for subpixel-translation, boundary, and masking tests. 

### This also changes my earlier “fewer, broader RBFs” suggestion

That suggestion was too glib.

With nonnegative Gaussian components, simply making every component broader makes it harder—or impossible—to represent a narrower core. Removing centers can introduce translation dependence and remove flexibility needed for a faint tail.

The better principle is:

> **Restrict unsupported complexity without removing the spatial scales that the telescope can genuinely produce.**

Wavelets offer one natural way to attempt that. A carefully designed multiscale RBF model could offer another, but I would not launch both redesigns simultaneously.

## 4. The wavelet method must not become “threshold, reconstruct, hope”

There are several ways a superficially reasonable wavelet implementation could reproduce our current problems.

### Separate deciding where structure exists from estimating its amplitude

A standard soft threshold is

\[
\widehat w=\operatorname{sign}(w)\max(|w|-\tau,0).
\]

By construction, it reduces the magnitude even of coefficients that survive. That can be useful for denoising, but it is a warning sign when the primary concern is already **under-recovered source amplitude**. Hard thresholding avoids that particular subtraction, but still introduces selection effects and can discard real faint structure. 

My preferred first candidate would therefore be closer to:

> Use multiscale evidence to select supported structure, then perform a bounded reconstruction that estimates its amplitudes without retaining the original threshold shrinkage.

That is often described as reconstruction on a *multiresolution support*: support here identifies locations **and scales**, not just a spatial aperture.

This is not an exotic invention. Astronomical wavelet methods have long distinguished coefficient selection from iterative reconstruction, and the reconstruction of modified undecimated coefficients requires care because the representation is redundant. 

Such a reconstruction is **not automatically unbiased**. Support selection, imperfect noise modeling, and regularization can still bias it. The point is to avoid building a known amplitude-shrinkage mechanism into the candidate unnecessarily.

Wavelet-based radio imaging methods such as MORESANE provide a useful precedent for separating multiscale analysis from image synthesis when recovering mixed compact and diffuse emission. I would borrow that conceptual distinction, not transplant its interferometric deconvolution algorithm into FRUIT. 

### Noise must be assessed at the scale being tested

For a linear wavelet operator \(W_j\) and map-noise covariance \(C\), ordinary covariance propagation gives

\[
C_j=W_j C W_j^{T}.
\]

So the uncertainty of a broad-scale coefficient is not determined merely by the single-pixel RMS. Shared noise and coverage matter.

For this bounded experiment, I would reuse the existing nuisance realizations to characterize coefficient scatter by scale and relevant coverage region. That need not reopen detector weighting or become a full covariance-estimation project. It does mean that an empirical threshold should remain labeled empirical—not treated as a calibrated false-alarm probability. The original RBF directive explicitly required that distinction. 

Nor should the admission rule demand a bright central seed or connected morphology. A comatic or defocused source must be allowed to supply its evidence through distributed structure.

### Positivity belongs to the reconstructed source, not all wavelet coefficients

For this positive-calibrator experiment, requiring the inferred source image to be nonnegative can be sensible. Requiring every wavelet coefficient to be positive is not: the coefficients are differences between resolutions and can legitimately be negative.

This follows directly from the decomposition. Clipping negative coefficients would change the representation of a perfectly positive source.

### The coarsest component cannot simply be labeled “background”

The broad component \(c_J\) contains large-scale image information. It can contain both nuisance background and real source brightness. The wavelet decomposition itself does not distinguish them. 

Dropping that component indiscriminately risks discarding the broad wings we are trying to recover. Keeping it indiscriminately risks feeding atmosphere or baseline structure back as sky.

I would retain explicit nuisance-background treatment and a declared source domain, with the same background-only check. If the footprint cannot distinguish a very broad source component from a nuisance term, that is an identifiability limitation to report—not something wavelets remove.

Finally, the method must estimate **the PSF the telescope produced**, not deconvolve it toward an ideal point source. Wavelet processing belongs only in the map-to-feedback estimator; the ordinary total map remains a separate product. That preserves the existing experimental boundary. 

## 5. The next experiment should answer a more specific question

I would not authorize a large wavelet survey or reopen the RBF penalty search. I would recommend one new, bounded comparison with a small preliminary diagnostic.

### First: audit the existing admission failure

Using the retained products, establish per array:

- Which gate failed, and at which pass?
- When was feedback first admitted, and when was it actually applied?
- Was the comatic source detectable in the first-pass map under a diagnostic that knew its morphology?

That last diagnostic can use the injected truth **for evaluation only**. It should test the source response after the processing it actually experienced, against the relevant nuisance background—not simply divide the unprocessed truth norm by a pixel RMS. A known-template, noise-aware detector provides a favorable benchmark for detectability; it is not an implementable unknown-morphology method or permission to use truth in feedback. Matched-filter detection in correlated noise explicitly depends on both the template and the noise model. 

This audit distinguishes “the candidate could not recognize available evidence” from “the test did not supply enough evidence for the intended question.”

### Second: screen one wavelet estimator before running complete trajectories

Use the saved first-pass maps where possible. Define one starlet configuration, one noise-based multiscale selection rule, and one bounded amplitude-reconstruction procedure.

Check the **actual estimator**, not just transform–inverse-transform identity, on the compact PSF, comatic PSF, null, and background cases. Include the existing translation and coverage checks. A transform can reproduce a noiseless image exactly while its selection-and-reconstruction policy still destroys scientifically important structure.

For a new clearly detectable coma case, freeze its normalization before comparing methods. Equal total brightness alone is not sufficient to make compact and spread-out sources equally detectable. Keep the original faint cases as challenges; do not replace or relabel their negative results.

### Third: run a small matched FRUIT comparison

Only a candidate that passes that preliminary screen should consume another full set of trajectories.

Preserve rank-5 PTC relearning, the accepted recurrence, current mapmaking and coefficients, flags, geometry, array independence, and the seven-pass operational budget. Compare against the matched pixelwise and Gaussian controls; use the frozen RBF products wherever the inputs genuinely match. Do not change admission policy for several methods and change representation at the same time without making that a separately identified comparison.

The main decision should be whether the wavelet candidate improves **output-map brightness and non-Gaussian structure** while retaining acceptable pointing, compact-source recovery, null behavior, and runtime—not whether its feedback model looks cleaner. Those were already the intended acceptance principles. 

I would leave 129081 reserved for new comparisons until something passes this discovery screen.

## Bottom line

**The tested RBF implementations should remain rejected. The broader RBF idea remains unresolved. Wavelets are my preferred next new representation to test—but with an explicit separation between multiscale evidence, amplitude estimation, and feedback admission.**

The scientific question I would now put at the center is:

> **Can we admit the spatial scales supported by the data without demanding that the entire faint PSF already be reproducible—and then recover their brightness without smoothing or thresholding it away?**

That is more precise than “try a different basis,” and it directly addresses both the reported RBF failure and the reason your wavelet suggestion is promising.