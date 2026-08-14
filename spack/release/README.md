# Citlali Release Bundle Contract

This directory defines what Citlali requires before a Spack build composition
may be called a release. It does not publish a release and does not replace the
deployment owner's activation or installation tooling.

## Development Versus Release

The checked development environments under `spack/environments/` use Spack
`develop` entries that point at local source checkouts. Their concrete locks
therefore contain absolute `dev_path` values. Those locks are useful evidence
for an observed build, but they are not portable release inputs.

`development-candidate.json` records the exact development graph and accepted
evidence available today. It intentionally fails the strict release gate. This
keeps completed work machine-readable without overstating release readiness.

A release bundle must contain:

- separate exact source and recipe revisions, with HTTPS archive checksums, for
  Citlali, Tula CMake, Tula, and Kidscpp;
- an accepted audit proving that every recipe revision resolves its declared
  source revision;
- one source-based `spack.yaml` and concrete `spack.lock` for each supported
  platform profile;
- no `develop` entries or absolute `dev_path` source bindings;
- the exact Spack version and revision used to concretize the profiles;
- accepted build, test, provenance, and reduction evidence;
- a source-build path even when signed build caches are provided.

Locks are portable within their declared platform/compiler profile, not across
unlike operating systems and compilers. macOS LLVM 20 and Unity GCC 13 thus
have separate locks while sharing one release manifest and source set.

Source and recipe revisions are deliberately separate. A repository first
freezes the source revision to be built, then publishes a later recipe revision
that names that immutable source. Treating them as one commit would make a
self-hosted recipe depend on source identity that could not exist until after
the recipe commit was created.

## Bundle Layout

The release producer should publish the manifest adjacent to the artifacts it
describes. A release archive cannot contain a manifest that records that same
archive's checksum without creating a circular identity.

```text
citlali-<release-id>/
  manifest.json
  sources/
    <repository>-source-<commit>.tar.gz
    <repository>-recipes-<commit>.tar.gz
  profiles/
    macos-llvm20/
      spack.yaml
      spack.lock
    unity-gcc13/
      spack.yaml
      spack.lock
```

The manifest schema is `release-manifest.schema.json`. The repository validator
hashes every bundled source archive and profile file, requires each source URL
to identify its declared commit, rejects paths that escape the bundle root,
checks the declared root DAG in each lock, and rejects any nested `dev_path`
value. The manifest also checksum-binds the recipe/source audit and requires it
to cover every repository. Validation therefore does not depend on network
access.

Before generating either profile lock, audit the decentralized recipes:

```console
$HOME/tolteca/bin/python tools/build/audit_release_recipe_sources.py
```

The 2026-08-14 development candidate is intentionally blocked: Citlali's
recipe has no immutable source, all seven Tula CMake package recipes identify
an older commit, and the Tula and Kidscpp recipes also identify older commits.
The checked report is
`validation/release_recipe_source_audit_2026-08-14.json`. Do not generate a
release profile or lock until a later recipe revision makes this audit pass.

Validate the current development candidate with:

```console
$HOME/tolteca/bin/python tools/build/validate_release_manifest.py
```

Validate a completed bundle with the strict gate:

```console
$HOME/tolteca/bin/python tools/build/validate_release_manifest.py \
  /path/to/citlali-<release-id>/manifest.json \
  --repository-root /path/to/citlali-<release-id> \
  --require-release
```

## Build Cache Policy

Source builds are the default and remain available for independent
reconstruction. A build cache is optional. If a release declares one, the
manifest must list HTTPS mirrors and trusted signing-key fingerprints. An
unsigned cache is never an accepted release input.

## Ownership Boundary

Citlali owns this schema, validator, profile requirements, and acceptance
evidence. Each source repository owns its source publication and decentralized
package recipes. The deployment system owns installation locations,
activation, and propagation of the selected profile and lock identity into
TolTECA-launched reductions. This repository does not modify or vendor
`tolteca_deploy`.
