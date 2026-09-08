# Extracted-tree links and external delivery artifacts - r0.3

The final `SCI-FRUIT-v0.1-stage-b-r0.3-owner-review.tar.gz` and its `.tar.gz.sha256` sidecar are external sibling delivery artifacts. They are not members of their own archive and are not in-tree Markdown links. The sidecar is shipped beside the final archive. The [artifact manifest](ARTIFACT_MANIFEST.md) and [manifest sidecar](ARTIFACT_MANIFEST.sha256) are archive members.

Every in-tree local Markdown link resolves inside the extracted delivery. The outer archive and digest sidecar are separately verified external siblings. The extraction check does not copy the outer files into the extracted r0.3 directory. Historical r0.2 links are different: that complete input subtree actually contains its own historical archive/sidecar files, so its preserved links resolve in-tree without rewriting history.

Packaging verifies exact member names, regular-file type, declared sizes and bytes; no symlink, hard link, absolute path or traversal member is permitted. Every manifest/sidecar hash reproduces. No historical archive is unpacked for science. Build caches, rendered images and other QA intermediates are excluded from delivery.
