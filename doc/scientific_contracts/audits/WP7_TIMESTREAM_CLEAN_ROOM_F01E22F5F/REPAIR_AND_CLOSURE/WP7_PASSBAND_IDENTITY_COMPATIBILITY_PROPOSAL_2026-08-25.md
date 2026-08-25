# WP-7 Passband-Identity Compatibility Proposal

Status: **approved binding scientific-owner authority**
Scientific owner: Grant Wilson
Prepared: `2026-08-25`

## 1. Proposed bound scope

This proposal would apply only to the following already approved exact
objects:

| Object | SHA-256 |
| --- | --- |
| Atmosphere machine contract | `7a064ff768a3de4f427f1338d94ef6cb9026d248f3c3c816fc3dfc96d156e36a` |
| Atmosphere node table | `fd688a4cd3f46585b08631bc63a562aed482feb9b24ec9ee0071b70db7eb8a5f` |
| TolTECA-v1 passband set | `5e6f38f14bcae93a29ffe8362c52b15209f51aee4e48373b23aaa5ec2f8a6433` |

It would resolve only the passband-identifier spelling transition inside those
bound atmosphere objects. It would change no numerical node value, passband
byte, operator, interpolation rule, support domain, quality regime, or
scientific claim.

## 2. Proposed canonical identities and exact aliases

The proposed canonical passband member identifiers are:

| Canonical identifier | Array | Exact passband member | Member SHA-256 |
| --- | --- | --- | --- |
| `tolteca_v1_a1100` | `a1100` | `data/a1100_passband.ecsv` | `13b8fd009bb8d7c375d3c46d21e26d0a779f7f00a949a2a5ccd619d1fe56fd72` |
| `tolteca_v1_a1400` | `a1400` | `data/a1400_passband.ecsv` | `a7b671d9f659cbc98dad99d3015ce81a3d7a3486c702819d9b3305703e7c682e` |
| `tolteca_v1_a2000` | `a2000` | `data/a2000_passband.ecsv` | `77e4b33c7bbc2c345ef94d41480d5fee5cb096d789f4fe78e1b4f80a37e0d6ff` |

Within the exact bound atmosphere node table and machine contract only, the
following legacy spellings would be exact aliases for those canonical
identities:

| Legacy spelling | Canonical identifier |
| --- | --- |
| `toltec_v1_a1100` | `tolteca_v1_a1100` |
| `toltec_v1_a1400` | `tolteca_v1_a1400` |
| `toltec_v1_a2000` | `tolteca_v1_a2000` |

No other spelling, prefix, suffix, array inference, or similarly named object
would be an alias under this decision.

## 3. Proposed canonicalization and join rules

For a row in the bound atmosphere node table:

1. Preserve the original `passband_id` as source provenance.
2. Replace only an exact legacy spelling from the table above with its exact
   canonical identifier when constructing the logical node identity.
3. Require the row's `array` to equal the array bound to that canonical
   identifier. A mismatch fails closed.
4. Bind the canonical identifier to the exact passband member digest above and
   to the exact TolTECA-v1 passband-set digest. A missing or different member
   fails closed.

The proposed canonical logical row identity is

```text
(operator_id,
 reference_profile_id,
 anchor_id,
 elevation_deg,
 canonical_passband_id,
 array,
 reference_spectral_index_alpha)
```

The proposed canonical interpolation-lane identity across opacity anchors is

```text
(operator_id,
 reference_profile_id,
 canonical_passband_id,
 array,
 reference_spectral_index_alpha)
```

`tau225` and `elevation_deg` remain the interpolation coordinates governed by
the existing operator contract. Canonicalization would not permit joining
different arrays, spectral indices, profiles, operator identities, or
passband-member digests.

## 4. Proposed precedence and limitations

For the exact objects in Section 1, approval would supersede only the absence
of an alias and canonical join rule for the six listed passband spellings. The
canonical `tolteca_v1_*` identifiers would govern downstream identity and
product provenance; the original source spelling would remain available for
replay and traceability.

Approval would not establish observational validation, interpolation-fidelity
evidence, implementation conformity, production approval, or permission to
substitute another TolTECA commit or passband set.
