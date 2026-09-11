"""Replay three fixed injection backgrounds, preserving the sealed control."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import time


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("executable", type=Path)
    p.add_argument("control_root", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    binary, control, out = (x.resolve() for x in (args.executable, args.control_root, args.output))
    if out == control or control in out.parents:
        p.error("output must be outside the immutable control root")
    out.mkdir(parents=True, exist_ok=False)
    inputs = ["synthetic", str(control / "campaign-01/152385-04/injection-background.txt"),
              str(control / "campaign-01/152430-08/injection-background.txt")]

    def run(i):
        expected = control / f"injection-final-{i:02d}.jsonl"
        with expected.open() as stream:
            identity = json.loads(next(stream))["background_identity"]
        if i and sha(Path(inputs[i])) != identity.split(":sha256:")[-1]:
            raise ValueError("fixed real-background digest mismatch")
        records = []
        for mode in ("legacy", "diagnostic"):
            command = [str(binary), inputs[i], identity] + (["--loss-diagnosis"] if mode == "diagnostic" else [])
            start = time.perf_counter()
            with (out / f"{i:02d}-{mode}.out").open("w") as stdout, (out / f"{i:02d}-{mode}.err").open("w") as stderr:
                result = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
            record = dict(index=i, mode=mode, command=command, exit_code=result.returncode,
                          elapsed_seconds=time.perf_counter() - start)
            if mode == "legacy" and result.returncode == 0:
                record["byte_equal_to_control"] = (out / f"{i:02d}-{mode}.out").read_bytes() == expected.read_bytes()
            records.append(record)
        return records

    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(run, range(3)))
    passed = all(r["exit_code"] == 0 and r.get("byte_equal_to_control", True) for group in records for r in group)
    result = dict(status="PASS" if passed else "FAIL", executable=str(binary), executable_sha256=sha(binary),
                  control_root=str(control), invocations=records, full_corpus_rerun=False)
    (out / "invocations.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
