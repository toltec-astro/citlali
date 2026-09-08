# Extracted-tree links and external delivery artifacts - r0.4

The outer `SCI-FRUIT-v0.1-stage-b-r0.4-owner-review.tar.gz` and `SCI-FRUIT-v0.1-stage-b-r0.4-owner-review.tar.gz.sha256` are separately verified external sibling artifacts. They are excluded from their own archive and are not in-tree Markdown links. The [manifest](ARTIFACT_MANIFEST.md) and [manifest sidecar](ARTIFACT_MANIFEST.sha256) are members.

Every in-tree local Markdown link resolves inside the extracted delivery without inserting the outer archive or sidecar. Historical archive references resolve to actual opaque input files within the delivered tree. Packaging checks exact names, sizes and bytes, rejecting symlinks, hard links, absolute paths and traversal members. Task caches, renders and intermediates are excluded. This accepted delivery distinction is unchanged from r0.3.
