"""Synthetic byte-restoration tests, without invoking a reduction executable."""

from pathlib import Path
from unittest.mock import patch

import pytest

from configured_point_cli import bind
from point_recovery_restore import materialize


@pytest.mark.parametrize("tamper", [False, True])
def test_exact_saved_bytes_and_original_preservation(tmp_path, tamper):
    original = tmp_path / "archive"
    original.mkdir()
    (original / "receipt.yaml").write_text("original identity")
    data = tmp_path / "data"
    data.write_bytes(b"original CAL bytes")
    recipe = dict(
        native=dict(path="unused"),
        request=dict(path="unused"),
        workers=3,
        networks=[
            dict(
                network=0,
                original_root=str(original),
                metadata=["receipt.yaml"],
                payloads={"native-time.f64": bind(data)["sha256"]},
            )
        ],
    )

    def fake_native(binary, request, stage, **kwargs):
        assert kwargs["terminal"] == "cal" and kwargs["workers"] == 3
        target = stage / "products/network0"
        target.mkdir(parents=True)
        (target / "native-time.f64").write_bytes(
            b"changed" if tamper else data.read_bytes()
        )
        return dict(fixture=True)

    destination = tmp_path / "run"
    destination.mkdir()
    with patch("point_recovery_restore.native_call", fake_native):
        if tamper:
            with pytest.raises(ValueError, match="differs"):
                materialize(recipe, destination, 100)
            assert not (destination / "RESTORATION.json").exists()
        else:
            roots = materialize(recipe, destination, 100)
            target = Path(roots[str(original)])
            assert (target / "native-time.f64").read_bytes() == data.read_bytes()
            assert (target / "receipt.yaml").read_bytes() == (
                original / "receipt.yaml"
            ).read_bytes()
    assert list(original.iterdir()) == [original / "receipt.yaml"]
