"""Lifetime of the native child that owns one ordinary weighted POINT observation."""

import hashlib
import os
from pathlib import Path
import re
import subprocess

import yaml


class PointObservationSession:
    """One sequential request stream, one private acknowledgement pipe and log."""

    def __init__(self, work):
        self.work = Path(work)
        self.process = None
        self.replies = None
        self.log_path = self.work / "point-observation.log"
        self.log_position = 0

    def __enter__(self):
        return self

    def _start(self, binary, descriptor, env, verbose):
        if descriptor != self.work / "health/development.yaml":
            raise ValueError("POINT observation must begin with health")
        launch = self.work / "point-observation.yaml"
        settings = yaml.safe_load(descriptor.read_text())
        settings["terminal"] = "point-observation"
        with launch.open("x") as stream:
            yaml.safe_dump(settings, stream, sort_keys=False)
        read_fd, write_fd = os.pipe()
        self.replies = os.fdopen(read_fd, "rb")
        try:
            with self.log_path.open("xb") as log:
                self.process = subprocess.Popen(
                    [str(binary), "-l", "debug" if verbose else "warning", str(launch)],
                    stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                    pass_fds=(write_fd,), env=dict(env, CITLALI_POINT_REPLY_FD=str(write_fd)),
                )
        finally:
            os.close(write_fd)

    def _new_log_text(self):
        with self.log_path.open("rb") as stream:
            stream.seek(self.log_position)
            text = stream.read().decode(errors="replace")
            self.log_position = stream.tell()
        return text

    def _abort(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()

    def run(self, binary, descriptor, log, env, verbose):
        if self.process is None:
            self._start(binary, descriptor, env, verbose)
        digest = hashlib.sha256(descriptor.read_bytes()).hexdigest().encode() + b"\n"
        try:
            self.process.stdin.write(digest)
            self.process.stdin.flush()
            reply = self.replies.readline(len(digest) + 1)
        except BrokenPipeError:
            reply = b""
        if reply != digest or hashlib.sha256(descriptor.read_bytes()).hexdigest().encode() != digest[:-1]:
            self._abort()
            log.write(self._new_log_text())
            log.write("POINT observation ended without a bound success acknowledgement\n")
            return self.process.returncode or 1
        log.write(self._new_log_text())
        return 0

    def __exit__(self, exc_type, exc, traceback):
        try:
            if self.process is not None and exc_type is None:
                self.process.stdin.write(b"finish\n")
                self.process.stdin.flush()
                self.process.stdin.close()
                code = self.process.wait()
                tail = self._new_log_text()
                if code or re.search(r"\[(?:error|critical)\]", tail, re.I):
                    raise RuntimeError(f"POINT observation finish failed ({code}); see {self.log_path}")
        finally:
            self._abort()
            if self.process is not None and not self.process.stdin.closed:
                self.process.stdin.close()
            if self.replies is not None:
                self.replies.close()
