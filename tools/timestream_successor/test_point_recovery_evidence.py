"""Frozen-map accounting tests use only locally constructed synthetic planes."""

import json

import numpy as np
from astropy.io import fits

from configured_point_cli import bind
from point_recovery_evidence import publish_seed
from point_recovery_products import ARRAYS


def test_seed_is_separate_signed_operand_and_does_not_relabel_base(tmp_path):
    y, x = np.indices((41, 41))
    valid = np.hypot(x - 20, y - 20) <= 20
    model = np.where(valid, 0.0, np.nan)
    model[20, 20] = 10.0
    mask = np.zeros_like(valid)
    mask[20, 20] = True
    path = tmp_path / "base.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(model, header=fits.Header({"EXTNAME": "CANDIDATE"})),
            fits.ImageHDU(mask.astype("u1"), name="ADMITTED"),
            fits.ImageHDU(valid.astype("u1"), name="VALID"),
        ]
    ).writeto(path, checksum=True)
    original = bind(path)
    signal = np.zeros((41, 41))
    signal[20, 19] = 3.0
    raster = tmp_path / "raster.json"
    raster.write_text(json.dumps(dict(values=signal.ravel().tolist())))
    models = {a: dict(candidate=original) for a in ARRAYS}
    output = tmp_path / "seed"
    record = publish_seed(models, {a: bind(raster) for a in ARRAYS}, output)
    assert bind(path) == original
    assert json.loads(open(record["path"]).read())["applied_only_to_cycle"] == 1
    with fits.open(output / "a1100.fits") as h:
        assert h["CANDIDATE"].data[20, 19] == 3
        assert h["CANDIDATE"].data[20, 20] == 10
        assert h["ADMITTED"].data.sum() == 2
        assert np.isnan(h["CANDIDATE"].data[~valid]).all()
