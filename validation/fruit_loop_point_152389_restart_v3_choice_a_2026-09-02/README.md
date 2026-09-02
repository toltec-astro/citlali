# FRUIT checkpoint-v3 Choice-A validation

Status: **PASS; D19 closure evidence; development-only**

Checkpoint v3 exactly reproduced uninterrupted FRUIT iterations 5, 6, and 7
after restarting the point-152389 injected-source trajectory at the completed
iteration-4 boundary. All 27 required `signal_I`, `kernel_I`, and `weight_I`
planes are bitwise equal, and all checkpoint variables are bitwise equal at
all three post-restart boundaries.

See [`TEST_DEFINITION.md`](TEST_DEFINITION.md) for the frozen setup, pass
conditions, result, and scientific limits. Exact map comparisons are in
[`restart_replay_comparison.csv`](restart_replay_comparison.csv), and
[`restart_replay_manifest.json`](restart_replay_manifest.json) records the
complete hashed evidence set and all three checkpoint comparisons.
