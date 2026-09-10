# Authorized routine implementation repair

Attempt 01 stopped during its first resource check, after the 15 deterministic
checks passed and before input hashing/scientific access or any stochastic
trial. macOS sandbox restrictions deny psutil's host-wide process enumeration.
The failure, source hashes and all products remain in `attempt_01` unchanged.
The exact previous runner source is retained in `repairs/attempt_01/run_screen.py`.

The replacement runner removes process enumeration. This harness contains no
child-process launches; a Python audit hook now rejects subprocess, fork,
posix_spawn and system calls explicitly. It remains one process, so ru_maxrss
supplies aggregate process-tree peak memory. All elapsed-time, memory and output
bounds remain enforced. The full approved campaign restarts as attempt 02 with
the same scientific source laws, seeds, settings, populations, input identities,
metrics and scope. No new owner decision is required under the approved routine
repair provision. Attempt 01 has no scientific results to replace.

Attempt 02 stopped even earlier: the new process guard correctly rejected
`platform.platform()` attempting to launch `uname -p` while assembling startup
metadata. No input file or stochastic trial was opened. Its startup failure
is explicitly reconstructed after the exception in `attempt_02/FAILURE.json`;
no nonexistent pre-run record is asserted. The exact source is retained in
`repairs/attempt_02/run_screen.py`. Use direct `os.uname()` system metadata
instead, with the guard unchanged. Attempt 03 repeats the same approved
campaign. Neither repair changes a scientific choice or resource bound.
