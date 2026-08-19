# Canonical compact APT v2

Status: owner-directed repair candidate; all APT-dependent validation remains
suspended until the gates in this document pass.

Durable decision: [ADR 0012](adr/0012-canonical-apt-v2-compact-normalization.md)

Machine authority:

- [`../validation/product_contracts.json`](../validation/product_contracts.json)
- `citlali::pipeline::canonical_apt_v2`
- the public `--canonical-apt-contract-v2` Citlali protocol

This contract supersedes new issuance and ordinary admission of the verbose
canonical APT v1 products. Historical v1 bytes remain evidence. They may be
read only by an explicit migration or comparison operation; they are never an
automatic fallback and a migrated product is never relabelled as a fresh
Beammap baseline.

## Scientist-readable contract

An APT is still an ordinary set of transparent ECSV tables. The primary
`apt` table has one row per detector used by Citlali and contains the actual
typed scientific values, units, masks, and missing values consumed by the
pipeline. A matched observation has exactly one compact relation record per
target detector. That record names the target occurrence and local row and
either one selected baseline occurrence/local row, or the explicit state
`unmatched` or `ambiguous`. It never uses table position as identity.

Provenance is normalized:

- one relation row per target detector;
- one rule row per governed field;
- one source row per raw or KMP input;
- one exception row only for a genuine departure from a field rule, an
  ambiguity candidate, or an exceptional seed disposition.

An ordinary matched value therefore does not carry a per-cell transformation
object. A matched baseline-governed field is copied from the selected seed;
the same field is typed missing for an unmatched or ambiguous target. A
target-governed field is retained from its exact source. Those rules are
declared once. The output is deterministically reconstructable from the
verified baseline snapshot, target values, relation, rules, and exceptions.

The only observation-specific KMP facts admitted by v2 are:

| Canonical field | Exact source column | Type | Unit | Presence | Authority |
| --- | --- | --- | --- | --- | --- |
| `kids_fr` | `fr` | finite binary64 | Hz | required | `kids:model-params-v1` |
| `kids_f_out` | `f_out` | finite binary64 | Hz | required | `kids:model-params-v1` |
| `kids_Qr` | `Qr` | finite binary64 | N/A | required | `kids:model-params-v1` |
| `kids_flag` | `flag` | signed int64 | N/A | optional for the complete target | `kids:fit-report-v1` |

Other diagnostics may exist in a selected KMP file. The complete file bytes
remain bound by SHA-256 and byte count, but an unregistered diagnostic is not
typed, interpreted, copied, assigned units, used for matching, used as
identity, or placed in an APT. Naming one for any such use fails closed.

Citlali owns the APT schemas, encoding, identities, validation, issuance, and
publication transaction. TolAPT owns the observation tone-matching science
and returns immutable occurrence-scoped match facts. TolProj selects inputs
and orchestrates the public TolAPT and Citlali interfaces; it does not write
APT bytes or reproduce either authority's algorithms. A consumer admits a
v2 product only after the Citlali guardian verifies its root receipt and every
component. TolTECA remains outside this contract and must not trim, synthesize,
or re-key a canonical v2 APT.

The scientific matching behavior is preserved for non-ties: the existing
network shift, median-Qr source, 200 kHz gate, good-seed-first then bad-seed
pass, and missing-network behavior remain unchanged. Internal array indices
may be used while computing, but every returned endpoint is translated to an
occurrence plus artifact-local key. Multiple exactly equal correlation maxima
are a typed matching failure. An exact equal-separation or exact contention
tie is recorded as `ambiguous` with its actual candidate references and no
selected seed. No epsilon, near-tie threshold, stable-key tie winner, or new
assignment objective is introduced.

## Logical records and identity

The baseline and matched products have different product kinds but share the
same component grammar. A product occurrence is opaque. `uid`, target UID,
relation UID, source UID, network, channel, row rank, and KMP row index are
artifact-local facts only. A detector endpoint is always the tuple
`(parent semantic identity, parent occurrence, local key)`. No persistent
detector namespace is created.

The four orderings remain nonidentity facts:

- source rank records how target rows appeared in the selected KMP sources;
- application rank records matcher input/application order;
- presentation rank records final APT row order;
- baseline presentation remains the immutable baseline table order.

Each rank set is a complete duplicate-free permutation of its own rows. A
physical row reorder cannot change logical identity; canonical serialization
uses the declared presentation order for the scientific table and local-key
order for the normalized provenance tables.

For a matched product, every target has exactly one disposition record:

- `matched`: exactly one selected seed reference and no ambiguity candidates;
- `unmatched`: no selected seed and no fabricated endpoint;
- `ambiguous`: no selected seed and two or more exact candidate exception
  records.

Selected seed keys are unique for the current observation-tone profile. The
complete baseline seed partition is derived from the verified baseline row
set: a selected seed is matched and every other seed is unused. A nondefault
seed-disposition reason requires an exception record. This is complete without
duplicating one ordinary disposition record per seed.

## Physical bundle

All data components are ECSV 1.0 with `schema: astropy-2.0`, UTF-8, LF-only
line endings, one final LF, comma delimiter, the canonical v2 metadata order,
and strict canonical reserialization. JSON is used only by the machine
request/response protocols and is never an APT component.

A baseline bundle contains component roles `apt`, `fields`, and `sources`.
A matched bundle contains `apt`, `relation`, `fields`, `sources`, and
`exceptions`, plus a flattened byte-exact snapshot of the verified baseline
bundle: `baseline-apt`, `baseline-fields`, `baseline-sources`,
`baseline-manifest`, and `baseline-receipt`. The baseline snapshot makes the
matched product portable and permits independent relation and copied-value
verification after relocation.

Every component except the root manifest has a basename of the form
`sha256-<64 lowercase hex>.<role>.ecsv`; the baseline receipt uses
`sha256-<64 lowercase hex>.baseline-receipt.txt`. The hex is the SHA-256 of
the exact component bytes. Components are regular non-symlink files in one
bundle directory. Absolute paths, subdirectories, `.`/`..`, alternate
spellings, duplicate roles, duplicate basenames, hard-link aliases between
roles, and a digest/name disagreement are rejected.

The fixed root names are `manifest.ecsv` and `manifest.ecsv.sha256`. The root
manifest does not enumerate itself, so no identity cycle exists. Its receipt
binds the exact root-manifest bytes, root envelope identity, and byte count.
Moving the complete directory does not change any component or product
identity.

### `manifest.ecsv`

One row per component, sorted by role, with columns:

1. `role` (`string`, nonnull)
2. `relative_path` (`string`, nonnull)
3. `schema` (`string`, nonnull)
4. `semantic_sha256` (`string`, nonnull)
5. `envelope_sha256` (`string`, nonnull)
6. `transport_sha256` (`string`, nonnull)
7. `byte_count` (`uint64`, nonnull)
8. `row_count` (`uint64`, nonnull)

Normative metadata fixes the v2 contract/profile, product kind, opaque
occurrence, event reference, producer, software revision, configuration
identity, observation tuple, component count, semantic/envelope scopes, and
the root semantic/envelope digests. Transport hash/count are never embedded
in the bytes they hash.

### `apt` and `baseline-apt`

The first five scientific columns retain the canonical APT meanings:
`uid:int64`, `tone_freq:float64[Hz]`, `array:int64`, `nw:int64`, and
`kids_tone:int64`. Closed registered scientific fields follow in UTF-8 byte
order. Baseline columns retain their exact v1 types, units, masks, values, and
authorities. Matched rows retain exact target structural/KMP values and exact
selected-seed values; baseline-governed cells are null only for unmatched or
ambiguous targets when the v2 rule permits it. There is exactly one output row
per target and its `uid` is newly output-artifact-local.

### `fields.ecsv`

One row per governed field, sorted by name:

`field_uid:int64, name:string, datatype:string, unit:string, nullable:bool,
authority:string, authority_reference:string?, identity_role:string,
rule:string, source_field:string?, missing_policy:string, description:string`.

The closed ordinary rules are `preserve-structural`, `preserve-target`,
`copy-seed-or-null`, `derive-declared`, and `override-declared`. The final two
require a contract-authorized field and exact source/operation evidence; no
generic mutation authority exists. `identity_role` is `nonidentity` except
for the artifact-local `uid` row-key declaration. The registry contains no
unknown or caller-declared field.

### `sources.ecsv`

One row per immutable input source, sorted by source UID:

`source_uid:int64, role:string, content_sha256:string, byte_count:uint64,
obsnum:int64, subobsnum:int64, scannum:int64, nw:int64, interface:string,
channel_count:uint64`.

Roles are closed by product kind. Source paths and diagnostic locators are
runtime handles, never content, identity, or provenance authority. Validators
read the named runtime input once and compare its exact hash/count and header
facts before issuance.

### `relation.ecsv`

Exactly one row per target, sorted by relation UID:

`relation_uid:int64, output_uid:int64, target_occurrence:string,
target_uid:int64, target_input_uid:int64, raw_source_uid:int64,
kmp_source_uid:int64, kmp_row_index:int64, source_rank:int64,
application_rank:int64, presentation_rank:int64, disposition:string,
seed_occurrence:string?, seed_uid:int64?, pair_uid:int64?,
separation_hz:float64?, is_good_match:bool?, network_evidence_uid:int64,
reason:string`.

The nullable seed/pair/separation/quality cells are present exactly for
`matched`; they are null for `unmatched` and `ambiguous`. Target, source, and
rank fields are never null. Normative metadata carries one closed typed
network-evidence record per target network and binds matcher occurrence,
implementation, configuration, method, backend, baseline parent identity,
target issuance context, and target-manifest identity. Each network record has
exactly one status: `matched-capable`, `missing-baseline-network`, or
`no-good-seed`. A `matched-capable` record carries finite exact binary64
shift, gate, and realized Qr values; the other two states carry canonical null
for all three rather than fabricated numbers. Network evidence is sorted by
network and is not repeated per detector.

### `exceptions.ecsv`

The table is present even when empty. Its closed columns are:

`exception_uid:int64, kind:string, target_uid:int64?, field_name:string?,
candidate_seed_occurrence:string?, candidate_seed_uid:int64?,
separation_hz:float64?, is_good_match:bool?, operation:string?,
before_datatype:string?, before_value:string?, after_datatype:string?,
after_value:string?, authority_reference:string?, reason:string`.

`kind` is `field-deviation`, `ambiguity-candidate`, or
`seed-disposition`. Per-kind nullability is exact. Typed scalar spellings use
the common rules below. Ordinary target preservation, selected-seed copying,
and unmatched nulls are never exceptions.

## Canonical scalar and digest rules

Strings are canonical single-line UTF-8. NUL, DEL, C0 other than TAB, C1,
surrogates, Unicode noncharacters, U+2028, and U+2029 are rejected. Strings
are not normalized or trimmed. YAML double quoting and CSV quoting use the
same closed escapes as canonical APT v1.

Signed and unsigned integers use canonical ASCII decimal with no leading
plus, leading zero, or negative zero and must fit their declared width.
Binary64 cells use the v1 max-digits decimal spelling for ECSV; semantic
preimages use the exact lowercase 16-hex IEEE-754 bit token for finite values
and the single canonical `nan` token for every admitted NaN representation.
`-0` and denormals are preserved. Each field's nonfinite policy governs NaN;
infinities are rejected for the current catalogs. Booleans are lowercase `true` or
`false`. Null is only an unquoted empty CSV cell and is legal only where the
schema permits it.

Each component has a labelled type-length semantic preimage over validated
typed content and an envelope preimage over semantic identity plus occurrence,
event, producer, software, configuration, parent, and observation facts.
Collection identity is independent of physical input order. The root semantic
preimage frames the validated component descriptors in role order; the root
envelope adds root issuance facts. Every SHA spelling is `sha256:` plus 64
lowercase hex digits.

Parsing is strict and bounded by the receipt/manifest counts. A validator
decodes each component once, validates the model, recomputes identities,
reserializes byte-for-byte, and retains that parsed object for cross-component
checks. It does not repeatedly parse components or construct a detector by
field provenance cross product.

## Publication and admission

Publication uses one sibling staging directory on the destination filesystem.
Citlali serializes all components, rereads and validates each once, assigns
content-addressed names, writes and validates the root manifest, and prepares
the complete response before publication. It then publishes components and
the manifest with no-replace semantics. The root receipt is linked into place
last and is the sole success transition. Any pre-receipt failure removes only
owned staged/final aliases; pre-existing or raced paths are never overwritten.
After the receipt appears, no fallible artifact mutation occurs.

Admission reads the small root receipt first, verifies the root manifest, then
follows only its safe relative component names. Missing, extra, duplicated,
swapped, stale, altered, noncanonical, aliased, or wrong-parent components
fail before a scientific table is exposed. A partial directory without the
root receipt is incomplete, not a product. Ordinary guardians admit only
fresh v2 product kinds. Bare ECSV, v1, migration-marked v2, and synthesized
fallback APTs are rejected unless the caller explicitly enters the bounded
migration/validation operation.

The transaction is receipt-atomic but not promised fsync/crash durable before
receipt publication. A process or host failure may leave incomplete files
without a receipt; it cannot leave a falsely complete bundle. A broken stdout
after receipt publication can yield a false-negative acknowledgement; the
caller recovers by invoking `validate` on the requested destination. The
strict local protocol has no owner-specified absolute stdin byte quota.

### Current Citlali-only checkpoint

The public `--canonical-apt-contract-v2` boundary currently implements exactly
two read-only operations:

- `validate-bundle-v2` verifies either fresh v2 bundle and returns its complete
  typed descriptor; and
- `describe-baseline-v2` performs the same verification but requires a fresh
  Beammap baseline bundle.

Both requests have the exact payload
`{"root_manifest":"/absolute/path/to/manifest.ecsv"}`.
`canonicalize-target-v2`, `issue-observation-apt-v2`, and
`migrate-v1-to-v2` are reserved operation names but fail closed before doing
work until the separately owned TolAPT/TolProj compact-v2 boundary is landed.
The legacy v1 public issue operation likewise fails before payload or source
processing. Read-only v1 describe/validate remain available solely for
historical comparison and deliberate migration development.

The Beammap producer now writes only a fresh baseline directory named
`<historical-apt-stem>.apt-v2/`, containing the compact components,
`manifest.ecsv`, and `manifest.ecsv.sha256`. It does not publish a new v1 APT.
The ordinary Citlali APT consumer admits only a fresh matched-v2 root manifest
whose raw-source bytes exactly match the current observation; therefore a
baseline root cannot be mistaken for a matched input.

### Owner-run 148670 Beammap baseline canary (not executed by Codex)

This canary tests only the Citlali-owned baseline producer. It does not resume
pointing, matching, consumer acceptance, or APT-dependent scientific
validation. From the local checkout, the owner publishes the reviewed commit:

```bash
git push -u origin codex/repair-apt-prod-003-compact-v2
```

On Unity, using the required SSH alias and the existing isolated refactor
checkout:

```bash
ssh unity_toltec
cd /home/toltec_umass_edu/work_toltec/citlali_dev/citlali_refactor
git fetch origin codex/repair-apt-prod-003-compact-v2
git switch codex/repair-apt-prod-003-compact-v2 || \
  git switch --track origin/codex/repair-apt-prod-003-compact-v2
git pull --ff-only
source tools/unity/citlali_refactor_bashrc.sh
citlali-refactor-update
```

The owner then reruns the established raw-only ObsNum 148670 Beammap project
without adding an APT input and without changing its scientific configuration.
After that command completes, set `REDU_DIR` to the new reduction root and
verify the sole generated v2 baseline with the same binary:

```bash
test "$(find "${REDU_DIR}" -type f -path '*.apt-v2/manifest.ecsv' | wc -l)" -eq 1
APT_V2_MANIFEST="$(find "${REDU_DIR}" -type f -path '*.apt-v2/manifest.ecsv')"
printf '%s\n' "{\"protocol\":\"citlali-canonical-apt-protocol-v2\",\"request_id\":\"beammap-148670-baseline-canary\",\"operation\":\"describe-baseline-v2\",\"payload\":{\"root_manifest\":\"${APT_V2_MANIFEST}\"}}" \
  | /home/toltec_umass_edu/work_toltec/citlali_dev/citlali_refactor/build/bin/citlali \
      --canonical-apt-contract-v2
```

The response must be one JSON line with `status:"ok"`, product kind
`baseline`, profile `citlali-beammap-baseline-apt-v2`, parser count `5`, and a
complete byte count below 20 MiB. The bundle must contain only `apt`, `fields`,
and `sources` components plus its root manifest/receipt. Retain the command,
JSON response, root identity, component inventory, byte count, reduction log,
and exact commit as canary evidence. Do not run the 148669 match or pointing
canary until the other package repairs are available.

## Migration and equivalence

`migrate-v1` is explicit, never automatic, and always writes a
`migration-only` v2 product. It first verifies the complete v1 bytes and
receipt. It validates every historical transformation, folds all ordinary
cases into the field registry, and writes exceptions only for actual
deviations/candidates. A v1 baseline lacks source content hashes; baseline
migration therefore also requires the original raw source bytes. Missing
source evidence is an error, never an invented digest.

The external ObsNum 148669 to baseline 148670 case is the acceptance oracle:
5,234 targets, 5,054 matched, 180 unmatched, zero new ambiguities, identical
occurrence-scoped detector endpoints, authorities, field types/units/masks,
binary64/integer values, dispositions, and pair set. The v2 ordinary field
exception count is zero. The complete uncompressed portable product must be
below 20 MiB, with a target near or below 10 MiB. Two fresh processes with the
same fully specified issuance facts must produce byte-identical components.
Relocation must preserve every identity. The 238 MiB v1 file remains external
evidence and is never committed as a fixture.

## Activation and limits

The registry entries remain unactivated. This repair does not activate a
profile, accepted run, ingestion campaign, CAL/ALIGN workflow, production
baseline, or downstream analysis. Flux calibration and hero overlay currently
have no authorized v2 field-rule issuance and must fail closed rather than
mutate a canonical v2 APT. Any near-tie ambiguity tolerance, new matching
objective, additional KMP field, or generic mutation rule requires separate
scientific/product authority.
