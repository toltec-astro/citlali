# Structural Refactor Tools

This directory contains static-analysis helpers for the Citlali structural
refactor. They do not build or run Citlali.

`refactor_inventory.py` scans the repository and writes a JSON and/or Markdown
inventory covering:

- direct `std::exit` calls by subsystem
- large public headers and simple non-template member definitions in headers
- source files listed as commented-out CMake entries
- simple config-key references by owning source file

Example:

```bash
$HOME/tolteca/bin/python tools/refactor/refactor_inventory.py \
  --repo . \
  --markdown-out /tmp/citlali_refactor_inventory.md \
  --json-out /tmp/citlali_refactor_inventory.json
```

The results are advisory. Template code, macros, dynamic config accesses, and
multi-line C++ expressions require manual review before any refactor PR.
