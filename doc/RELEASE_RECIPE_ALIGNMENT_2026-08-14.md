# Release Recipe Alignment, 2026-08-14

## Scope

This checkpoint corrects release packaging metadata only. It does not change
Citlali science behavior, dependency variants, package dependencies, CMake
logic, runtime code, or `tolteca_deploy`. Each repository continues to own its
canonical Spack recipes; Citlali records the accepted composition without
vendoring private copies of dependency recipes.

The initial audit is retained at
`validation/release_recipe_source_audit_2026-08-14.json`. It found that none of
the ten canonical recipes resolved the exact source revisions exercised by the
accepted development builds. The corrected audit is
`validation/release_recipe_source_audit_accepted_2026-08-14.json`.

## Accepted Source And Recipe Identities

| Owner | Accepted source revision | Proposed recipe revision |
| --- | --- | --- |
| Citlali | `b8e80fb1562b0ab9974a4c5fb183682ab2d351cc` | `0dc5db2ccde818838d30b192eb43bb0d43b18309` |
| Tula CMake | `6433cdabe7010d0af2d0ba69da7af27510391b80` | `d3c0ad6e646d2a37c9f66b48a42d99a3059d2338` |
| Tula | `aa16c853c6b596c04ccdc90dc3acc4ce2006d947` | `6de0a44a79d035e3c851e5f2a34e02d79f3745f4` |
| Kidscpp | `498ece1113001ae2d42d96d9fc29152aea3eaaef` | `4850383da810b6daf721e4fd8b35f2545e275f04` |

The source revision is the code identity already tested. The recipe revision
is necessarily later: it contains metadata that names the source revision.
Conflating these identities either creates a self-reference or causes the
recipe to retrieve code other than the tested code.

## Minimal Changes

| Owner and recipes | Files | Previous source identity | Accepted source identity |
| --- | --- | --- | --- |
| Citlali: `citlali@4.0.0` | `spack/spack_repo/toltec/citlali/packages/citlali/package.py` | Unbound; no Git source and no commit | `b8e80fb1562b0ab9974a4c5fb183682ab2d351cc` |
| Tula CMake: `tula-cmake`, `tula-logging`, `tula-yaml-cpp`, `tula-eigen3`, `tula-ccfits`, `tula-netcdf-cxx4`, `tula-perflibs` | The seven corresponding `spack_repo/toltec/tula_cmake/packages/*/package.py` files | `a7d411dc9f342014586785a2c985b0fd16888f13` | `6433cdabe7010d0af2d0ba69da7af27510391b80` |
| Tula: `tula@3.1.0` | `spack_repo/toltec/tula/packages/tula/package.py` | `212717a2844fe1da7c4248dfefdead2ff21e80be` | `aa16c853c6b596c04ccdc90dc3acc4ce2006d947` |
| Kidscpp: `kidscpp@3.1.0` | `spack_repo/toltec/kidscpp/packages/kidscpp/package.py` | `06b3130ba7f6f96b509011c5855f635f76a25087` | `498ece1113001ae2d42d96d9fc29152aea3eaaef` |

The patch adds two lines and removes one in Citlali, then changes one commit
literal in each of the other nine recipes. There is no broader architectural
consequence beyond making the existing decentralized ownership model
effective and auditable.

## Audit Result

The manifest-bound audit reads each recipe from its declared recipe revision,
not from an uncommitted working tree. It confirms:

- 10 canonical recipes inspected;
- 10 recipes resolve the accepted repository and source commit;
- zero missing, unbound, or stale recipes;
- source and recipe revisions remain distinct for all four repositories.

This closes the recipe/source identity gate locally. It does not close archive,
publication, release-profile, portable-lock, deployment, or four-mode science
acceptance gates. No portable release lock may be generated until the proposed
recipe revisions are reviewed and published.

## Development And Release Workflow

Normal development does not need to change. Spack `develop:` overrides may
continue to select sibling working checkouts for rapid iteration. Those paths
remain development evidence and are not release inputs.

For a release candidate, each repository owner must:

1. freeze and publish the accepted source revision;
2. publish a later recipe revision that retrieves that exact source;
3. avoid silently repointing a package version after it has been released;
4. provide immutable source and recipe archives with checksums;
5. allow Citlali's composition audit to verify the complete set before locks
   are generated.

The current package-version labels are not tags in the inspected dependency
repositories. The proposed dependency bindings are therefore suitable for
review as a one-time candidate correction, provided those labels have not
already been distributed as immutable releases.

Citlali is different: the repository already has a `v4.0.0` tag, and it does
not identify the accepted refactor source. The proposed Citlali recipe proves
the required commit binding, but it must not be published as the final
`citlali@4.0.0` identity. The owner must assign the accepted source a new
release or immutable snapshot version before publication. The same rule
applies to any dependency label that maintainers confirm has already been
published: add a new version rather than repoint an existing release.
