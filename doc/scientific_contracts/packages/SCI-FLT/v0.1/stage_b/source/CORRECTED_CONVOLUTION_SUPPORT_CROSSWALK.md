# SCI-FLT-FIXED v0.1 Corrected Convolution Equation and Support Crosswalk

Record identity: `SCI-FLT-FIXED-CONVOLUTION-SUPPORT-CROSSWALK v0.1/draft-r0.4`

Status: implementation-blind Stage B closure artifact; scientific-owner review required

Scientific owner: Grant Wilson

Stage B date: `2026-08-31`

Normative authority remains the shared normative core. For the ordinary
method, exactly

```text
(L_Theta m)_p = sum over r in K_nonzero of k_Theta(r) m_(p-r)
              = sum over r in K_req of k_Theta(r) m_(p-r).
```

The support crosswalk is:

```text
Set                 Scientific role
K_geom_science      representation-invariant geometric description
K_store             scientifically nonauthoritative serialization footprint
K_nonzero           canonical exact-nonzero arithmetic offsets
K_req                ordinary required dependency set, equal to K_nonzero
```

An exact-zero coefficient creates no arithmetic term, payload dependency,
influence, covariance contribution, or row exclusion. Its parent payload is
not evaluated or dereferenced. Dense, sparse, cropped, and zero-padded
encodings of one canonical kernel therefore leave the scientific operator and
row admission unchanged.

Identity retains `K_req = {0}`. The exact zero operator is an empty arithmetic
sum on its explicitly inherited admitted parent-support row domain; it is not
an ordinary nonzero convolution with empty output support.
