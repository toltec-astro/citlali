# Synthetic gate failure 01 and pre-real-data refreeze

No real PTC timestream fit had been run when the first synthetic suite stopped.

The exact scalar-lag estimator recovered injected zero, negative, and positive
lags, including a lag combined with a free static source offset.  Two separate
synthetic failures remained:

1. The pure-hysteresis and joint optimizations could remain in a local basin
   when initialized only at zero direction-sign displacement.  The synthetic
   trajectory also coupled its two scan phases and provided unnecessarily weak
   direction-sign leverage at the source crossing.
2. The first derivative template was evaluated after the no-lag source fit but
   was not projected against the free source-position and detector-amplitude
   nuisance directions.  Those nuisances absorbed the injected first-order
   shift.

The repair is numerical and diagnostic, not result-driven:

- use independent deterministic scan phases in the realistic synthetic
  trajectory;
- add a frozen symmetric set of direction-sign optimizer starts; and
- solve the derivative cross-check after projecting the timing template
  against global source-position and detector-scan amplitude templates.

The lag bounds, real observation selection, common support, source window,
baseline treatment, beam treatment, weights, bootstrap design, and acceptance
criteria are unchanged.  All synthetic tests must pass after this refreeze
before the anchor observation may be prepared or fit.
