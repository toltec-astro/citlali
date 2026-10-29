"""Write compact evidence, retaining failures rather than only successful cases."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

source, destination = map(Path, sys.argv[1:])
data = json.loads(source.read_text())
summary = {}
for name in dict.fromkeys(c["case"] for c in data["seed_sweep"]):
    cases = [c for c in data["seed_sweep"] if c["case"] == name]
    summary[name] = {
        "realizations": len(cases),
        "eligible": sum(c["decision"]["eligible"] for c in cases),
        "old_six_interval_eligible": sum(c["old_six_interval_admission"] for c in cases),
        "temporal_states": dict(Counter(c["decision"]["temporal"] for c in cases)),
    }
result = {k: data[k] for k in data if k != "seed_sweep"}
result.update({
    "scientific_disposition": "revise_temporal_veto_before_activation",
    "seed_sweep_summary": summary,
    "stationary_counterexamples": [c for c in data["seed_sweep"]
                                   if c["case"] == "healthy" and not c["decision"]["eligible"]],
    "seed_scope": "0..15, same noise realization reused across related physical cases; not independent replications of a false-rejection rate",
    "full_result_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "environment": {"python": platform.python_version(), "numpy": np.__version__,
                    "platform": platform.platform()},
})
destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
