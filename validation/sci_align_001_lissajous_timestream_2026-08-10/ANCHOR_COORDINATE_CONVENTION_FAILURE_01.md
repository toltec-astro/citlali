# Anchor coordinate-convention gate failure 01

The revision-3 exact anchor fit passed its numerical and zero-lag coordinate
gates. It then exposed a comparison-convention failure that is algebraic and
does not depend on accepting either fitted value.

The exact estimator defines positive tau by evaluating the complete detector
coordinate at `t + tau`. For locally linear motion `x(t + tau) = x(t) + v*tau`.
If the source crossing occurs when `x(t + tau) = x0`, a map constructed at the
recorded coordinate places that signal at `x(t) = x0 - v*tau`. Consequently,
the committed map regression

```
centroid = intercept + tau_map_slope * velocity
```

has `tau_map_slope = -tau_coordinate_shift`. Directly subtracting that raw
coefficient from the exact-shift tau would compare opposite conventions.

The repair does not alter either estimator. It defines the authenticated map
comparison value as

```
tau_map_coordinate_shift = -tau_map_slope
delta_tau = tau_timestream - tau_map_coordinate_shift
```

and applies the same sign transform to every paired bootstrap realization.
A synthetic linear-motion test fixes this convention. All synthetic tests and
a successor implementation/protocol freeze are required before any second
real observation is examined.
