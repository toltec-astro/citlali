"""Lifetime of the native child that owns one ordinary weighted POINT observation."""

import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys

import yaml


def native_command(binary, descriptor, verbose):
    """Info is the most detailed level available in the Release executable."""
    return [str(binary), "-l", "info" if verbose else "warning", str(descriptor)]


class PointObservationSession:
    """One sequential request stream, one private acknowledgement pipe and log."""

    finish_timeout = (
        30  # Only teardown is bounded; the final scientific pass has already acknowledged.
    )

    def __init__(self, work, *, start_at_cal=False):
        self.work = Path(work)
        self.start_at_cal = start_at_cal
        self.final_binding = None
        self.process = None
        self.replies = None
        self.log_path = self.work / "point-observation.log"
        self.log_position = 0

    def __enter__(self):
        return self

    def _start(self, binary, descriptor, env, verbose):
        first = "upstream" if self.start_at_cal else "health"
        if descriptor != self.work / first / "development.yaml":
            raise ValueError(f"POINT observation must begin with {first}")
        launch = self.work / "point-observation.yaml"
        settings = yaml.safe_load(descriptor.read_text())
        settings["terminal"] = "point-cal-observation" if self.start_at_cal else "point-observation"
        with launch.open("x") as stream:
            yaml.safe_dump(settings, stream, sort_keys=False)
        read_fd, write_fd = os.pipe()
        self.replies = os.fdopen(read_fd, "rb")
        try:
            with self.log_path.open("xb") as log:
                self.process = subprocess.Popen(
                    native_command(binary, launch, verbose),
                    stdin=subprocess.PIPE,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    pass_fds=(write_fd,),
                    env=dict(env, CITLALI_POINT_REPLY_FD=str(write_fd)),
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

    def _request(self, message):
        try:
            self.process.stdin.write(message)
            self.process.stdin.flush()
            # Success is a 64-character digest and a newline.
            return self.replies.readline(66)
        except BrokenPipeError:
            return b""

    def run(self, binary, descriptor, log, env, verbose):
        if self.process is None:
            self._start(binary, descriptor, env, verbose)
        digest = hashlib.sha256(descriptor.read_bytes()).hexdigest().encode() + b"\n"
        if (
            self._request(digest) != digest
            or hashlib.sha256(descriptor.read_bytes()).hexdigest().encode() != digest[:-1]
        ):
            self._abort()
            log.write(self._new_log_text())
            log.write("POINT observation ended without a bound success acknowledgement\n")
            return self.process.returncode or 1
        log.write(self._new_log_text())
        return 0

    def _finish(self):
        if self.start_at_cal:
            # Mutable original sources still need verification after final fits.
            # This is scientific work, outside the post-acknowledgement timeout.
            final_digest = self._request(b"verify-originals\n")
            if not re.fullmatch(b"[0-9a-f]{64}\n", final_digest):
                raise RuntimeError(
                    f"POINT observation final input verification failed; see {self.log_path}"
                )
        try:
            self.process.stdin.write(b"finish\n")
            self.process.stdin.flush()
            self.process.stdin.close()
            code = self.process.wait(timeout=self.finish_timeout)
        except BrokenPipeError as error:
            raise RuntimeError(
                f"POINT observation closed during finish; see {self.log_path}"
            ) from error
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                f"POINT observation finish timed out; see {self.log_path}"
            ) from error
        tail = self._new_log_text()
        if code or re.search(r"\[(?:error|critical)\]", tail, re.I):
            raise RuntimeError(f"POINT observation finish failed ({code}); see {self.log_path}")
        if self.start_at_cal:
            receipt = self.work / "final-originals.yaml"
            sha = hashlib.sha256(receipt.read_bytes()).hexdigest()
            if sha.encode() != final_digest[:-1]:
                raise RuntimeError(
                    f"POINT final verification receipt differs from native acknowledgement: {receipt}"
                )
            self.final_binding = dict(path=str(receipt), sha256=sha)

    def _dispose(self):
        try:
            self._abort()
        finally:
            try:
                if self.process is not None and not self.process.stdin.closed:
                    self.process.stdin.close()
            except BrokenPipeError:
                pass  # Failed peer may already have closed its read end.
            finally:
                if self.replies is not None:
                    self.replies.close()

    def __exit__(self, exc_type, exc, traceback):
        try:
            if self.process is not None and exc_type is None:
                self._finish()
        finally:
            primary = exc if exc is not None else sys.exc_info()[1]
            try:
                self._dispose()
            except Exception as cleanup:
                if primary is None:
                    raise
                if hasattr(primary, "add_note"):
                    primary.add_note(f"POINT child cleanup also failed: {cleanup}")
