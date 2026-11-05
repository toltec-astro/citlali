"""Owner-run process lifecycle checks using a private fake native peer."""
import sys
import textwrap

import pytest

import configured_point_cli as cli
from configured_point_session import PointObservationSession


@pytest.fixture
def peer(tmp_path):
    binary = tmp_path / "native"
    binary.write_text("#!" + sys.executable + "\n" + textwrap.dedent("""
        from pathlib import Path
        import os
        import sys
        import time
        import yaml
        config = yaml.safe_load(Path(sys.argv[-1]).read_text())
        assert config["terminal"] == "point-observation"
        fd = int(os.environ["CITLALI_POINT_REPLY_FD"])
        if os.environ.get("POINT_FAKE") == "die-before-read":
            sys.exit(7)
        for ordinal, line in enumerate(sys.stdin.buffer):
            if line == b"finish\\n":
                if os.environ.get("POINT_FAKE") == "stall-finish":
                    time.sleep(60)
                sys.exit(4 if os.environ.get("POINT_FAKE") == "finish-error" else 0)
            print(f"[warning] pass {ordinal} pid {os.getpid()}", flush=True)
            if os.environ.get("POINT_FAKE") == "no-ack":
                sys.exit(7)
            os.write(fd, b"x" * 64 + b"\\n" if os.environ.get("POINT_FAKE") == "wrong-ack" else line)
    """))
    binary.chmod(0o755)
    work = tmp_path / ".point-work"
    work.mkdir()
    return binary, work


def call(peer, session, name):
    binary, work = peer
    request = work / (name + "-request.json")
    request.write_text("{}\n")
    return cli.native_call(binary, request, work/name, terminal="map", workers=3, session=session)


def test_one_process_per_observation_and_separate_stage_logs(peer):
    _, work = peer
    with PointObservationSession(work) as session:
        call(peer, session, "health")
        child = session.process
        call(peer, session, "weighted-00")
        assert session.process is child and child.poll() is None
    assert child.returncode == 0
    first = (work/"health/native.log").read_text()
    second = (work/"weighted-00/native.log").read_text()
    assert f"pass 0 pid {child.pid}" in first and "pass 1" not in first
    assert f"pass 1 pid {child.pid}" in second and "pass 0" not in second


@pytest.mark.parametrize("failure", ["no-ack", "wrong-ack"])
def test_missing_or_wrong_ack_fails_and_reaps(peer, monkeypatch, failure):
    _, work = peer
    monkeypatch.setenv("POINT_FAKE", failure)
    with pytest.raises(RuntimeError, match="map failed"):
        with PointObservationSession(work) as session:
            call(peer, session, "health")
    assert session.process.poll() is not None
    assert (work/"health/stage.json").is_file()


def test_feedback_failure_reaps_child(peer):
    _, work = peer
    with pytest.raises(ValueError, match="feedback failed"):
        with PointObservationSession(work) as session:
            call(peer, session, "health")
            raise ValueError("feedback failed")
    assert session.process.poll() is not None


def test_finish_failure_propagates_before_publication(peer, monkeypatch):
    _, work = peer
    monkeypatch.setenv("POINT_FAKE", "finish-error")
    with pytest.raises(RuntimeError, match="finish failed"):
        with PointObservationSession(work) as session:
            call(peer, session, "health")
    assert session.process.returncode == 4


def test_stalled_finish_is_bounded_and_reaps_child(peer, monkeypatch):
    _, work = peer
    monkeypatch.setenv("POINT_FAKE", "stall-finish")
    monkeypatch.setattr(PointObservationSession, "finish_timeout", 0.2)
    with pytest.raises(RuntimeError, match="finish timed out"):
        with PointObservationSession(work) as session:
            call(peer, session, "health")
    assert session.process.poll() is not None and session.replies.closed


def test_peer_dead_before_first_write_preserves_stage_failure(peer, monkeypatch):
    _, work = peer
    monkeypatch.setenv("POINT_FAKE", "die-before-read")
    start = PointObservationSession._start
    def wait_for_dead_peer(session, *args):
        start(session, *args)
        assert session.process.wait(timeout=10) == 7
    monkeypatch.setattr(PointObservationSession, "_start", wait_for_dead_peer)
    with pytest.raises(RuntimeError, match="map failed"):
        with PointObservationSession(work) as session:
            call(peer, session, "health")
    assert session.replies.closed
