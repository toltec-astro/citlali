#!/usr/bin/env python3
"""Analyze the copied AM 12.2 LMT atmosphere suite for SCI-CAL-001.

This follow-up is an evidence and stress-test generator, not a regeneration of
the historical Citlali q-model lineage.  It verifies the copied AMC inputs and
NPZ products, compares every same-percentile annual/seasonal family with the
recovered legacy grids and repair-base literals, and tests the legacy
piecewise-linear and PCHIP LOS-tau surfaces without extrapolation.  Dataverse
uploader logs are deliberately never read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy.interpolate import PchipInterpolator


REPAIR_BASE_SHA = "9aae0e669384c5c0c0dda93debc194d6b8dac787"
REPAIR_LINE_EVIDENCE_HEAD = "ae99be1cef8c390d0e7490835ffca1f31da7ebc0"
DEFAULT_AM_ROOT = Path("/Users/gwilson/work_toltec/local_data/AM/Big_Atmosphere")
DEFAULT_LEGACY_SOURCE_DIR = Path(
    "/Users/gwilson/GitHub/toltec_beammap/src/toltec_sensitivity"
)
DEFAULT_TOLTECA_ROOT = Path("/Users/gwilson/GitHub/tolteca")
TOLTECA_REGISTRY_COMMIT = "25ccce10bfb50145424c88257a584ab92486ddf1"
TOLTECA_REGISTRY_OBJECT = "tolteca/simu/lmt/__init__.py"
TOLTECA_REGISTRY_OBJECT_SHA256 = (
    "5f117c3e5644faf3141ff647ec256f0f0404b9d0ebc1b16218222ee5daed8b72"
)
CALIBRATE_REL = Path("include/citlali/core/timestream/rtc/calibrate.h")
SELECTOR_REL = Path("include/citlali/core/timestream/extinction_model_selection.h")
PHASE0_SCRIPT_REL = Path(
    "validation/sci_cal_001_phase0_2026-07-31/generate_q_model_continuity.py"
)
FROZEN_REPAIR_INPUTS = {
    CALIBRATE_REL: "d70a55278227b43cdd7de19bc67e4ddb332524d40e1455c5fa7a80ae5e2d11ee",
    SELECTOR_REL: "45cf86bbb2318c22514411f6d2a0e0371e22e9e355e61b293d93c628d9f3469d",
    PHASE0_SCRIPT_REL: (
        "a46211c007bdc1fa11d1408c6db4c4a68264ca00cd383806fd421ba978fffe78"
    ),
}

EXPECTED_COPIED_SHA256 = {
    "LMT_DJF_25.npz": "b43982166d4755fca5ae2454c151c1cae55637dbb6b1b0a8e9dd3dd5555ef717",
    "LMT_DJF_5.npz": "214d9fa975c73afa01a4e1b5c5f068245779989578acd8574831b7fe2b6ed6cc",
    "LMT_DJF_50.npz": "57e6eebebe40f0bd1fb9b3a2411a73a1bd4cc0f4762a316af1c23758060a02ec",
    "LMT_DJF_75.npz": "67f7af72dd94a0c26cdf9759d75345c1d5e53b756418937b9471137c6f10f552",
    "LMT_DJF_95.npz": "3dd961143e31a8db8182c35dd55472ad9ec943a711f652f6d55d485ee5ddb42d",
    "LMT_JJA_25.npz": "a8131ade7fdc5d0fa0a036a74f5c911d31023be33e228d29e1b968575baf6ffc",
    "LMT_JJA_5.npz": "72e151b610c350ff8790ff1827d35822d6b1985081a2aec19e5d7925bf4c417a",
    "LMT_JJA_50.npz": "1ecdb57022d15882ecacfe2f9038c9d27a2cbcf379fd45104c18ffa01cd60650",
    "LMT_JJA_75.npz": "3f3003075f449cc739b579d683c4ff5f087e5982112a250fd53b29c2c0099911",
    "LMT_JJA_95.npz": "0e57576febe7f0736a2015bb8e8e85766a1489db81aa09210b16c2a355bc3fa2",
    "LMT_MAM_25.npz": "9851853d67e8f4b8eb257aec82e9d758d6fba0daaf2348b44bd95a1cd7714195",
    "LMT_MAM_5.npz": "7ddefd6056c932c2233b52f5d0ddec5a7ee6bd4292b1c301477684677d659518",
    "LMT_MAM_50.npz": "a6d02472c4b095caf47de7e1685885b82233564b3530764f7f9d43dc380c9720",
    "LMT_MAM_75.npz": "93e411f37396b0be93f641b6cd9d364c0b2f9e887f29db3d203f999fb34cdc3c",
    "LMT_MAM_95.npz": "6979af52ce7f6491d67537811322e333232c1d68be26d5a6a544c98ee287fc7b",
    "LMT_SON_25.npz": "f155f769519302cde304fc214376eb2181cfcd11c33865d74830150e7404c06a",
    "LMT_SON_5.npz": "bdab16e63b5dbd2a2993bd29ed906d1bda91e971bc548fd20044719a375bf880",
    "LMT_SON_50.npz": "bb56847337c77d0a79052995c544e29ac76c962088771d58994b0b24303e8a68",
    "LMT_SON_75.npz": "b87cf2bef8233fc320dd3117ef2df78648da2d3ede2c40cf9b830c9d8121b500",
    "LMT_SON_95.npz": "86839682b473783004aa1f9c1eb1572afe526aa8ed5985f5a7f33c5053065622",
    "LMT_annual_25.npz": "4dc19a1d4a6da41b8cae51bcdc910fd65f8783cb744d9ea0ae6a8108456abd25",
    "LMT_annual_5.npz": "3d166853b2be791389a1a4209a4378ef4db1c5f9c84e9610eac248e44ab0ed52",
    "LMT_annual_50.npz": "fc3b1c8ecb3cb966f994539be38c127b35084c6c8153309221508fae81645168",
    "LMT_annual_75.npz": "8ce0cf1a4a839f9a2cdf1699d74d7d29edf8ca63398a127bde48cd95c1f4d0e5",
    "LMT_annual_95.npz": "387df474b96a6d59ab7ea5be685498462f8b9ced225b27ccec288162b856e545",
}
EXPECTED_AMC_SHA256 = {
    "LMT_DJF_25.amc": "aeeeeb48bef422f2d9392b5d7a3d62ab1887fd9e7c10322d5246d914841ba866",
    "LMT_DJF_5.amc": "fcb3b70f44cad98cf0586fede9dcd3b2e35f3cb45023d0485c782c108b25b474",
    "LMT_DJF_50.amc": "d7c256d04d922beb51c9f8ab715e5be1a962252580eff2d08ba1be4d206eb5b0",
    "LMT_DJF_75.amc": "b63503c7f4170404d18f3797735b64fb947ce73eed35f0315155d0a29d499721",
    "LMT_DJF_95.amc": "b87b918b302425ef3d85aeedc285863a987579923289a37b97c6de5c935175e6",
    "LMT_JJA_25.amc": "13ea1837e3f2afeb605d8f8e8329472032f27c7a9d526d1b381bd7e75830e9b6",
    "LMT_JJA_5.amc": "f5a3a92f41803247da504271eafe8a62af6df51e7a3ec8740c5e89c2b97409fc",
    "LMT_JJA_50.amc": "00b86aa0f2331f6a138c8efcc89ed5a4d918baef948f0da55feb114b0df2eb76",
    "LMT_JJA_75.amc": "1b54ac2d0d5c7cd8f0805d1117d44ad1ff938ccb74433a93e2a20cbc77b3fc95",
    "LMT_JJA_95.amc": "54a4345d487babbffcc9b36b9ccbaec2904b58d35458f6476a382da8d70cf437",
    "LMT_MAM_25.amc": "82ac1e2a49a528244c1571daadcc8d42bd6d13c0ba8a7b5d2f81d10ebc13caee",
    "LMT_MAM_5.amc": "ecdf228e34ca8f4b0f5930865179fd8afe7c8e602b1863d7ee8ac4352c65351f",
    "LMT_MAM_50.amc": "2f282452f3932024be26b4579ed765996c08ff1e74cafb7d2750396d234fa6ac",
    "LMT_MAM_75.amc": "937ecf9b3725b03a2745a61546f3659a9506bbaead72afbb41141b51e88a630e",
    "LMT_MAM_95.amc": "dc7a3acc1fbc5ce92ef98dd5d00f45db997c6f2680c3c88e485f94bdbac398b2",
    "LMT_SON_25.amc": "e9b06bae87e742801a751270aadd7939f94b39bfb04c7196ecf12c7586cce627",
    "LMT_SON_5.amc": "79a50d0275d026886feefb83e68e88b84801c5e3efa0066edbfc925e2b134926",
    "LMT_SON_50.amc": "b47140d5680449c83327ec8ffaa3a36d2472f7d7d042119d95b84689f06b42b2",
    "LMT_SON_75.amc": "c2d6a7b6aee60639168dfcb03d85dc07b85ed23ea7ea2b8e202033d16c14a770",
    "LMT_SON_95.amc": "a4348f003b44205c9c4f367da42ea9a5962689cdfd6f1c12580c28c853526984",
    "LMT_annual_25.amc": "a9524553a5808a549eb18046a9ed6f8bd67ca1e29ccd1c91e05b351b64ea23e6",
    "LMT_annual_5.amc": "f58921d3cb222965df86b05f89cbf716f92f8193465d18f0106bf09b52fd718d",
    "LMT_annual_50.amc": "ee3946b48db6049b26231ff22d456c8fc2f2dc96ecabd1a861c4d8002c81c3c3",
    "LMT_annual_75.amc": "8e7a250764c8583ef23f9ca140248e62670e3e3d9b709baf005b31c24dc52387",
    "LMT_annual_95.amc": "687218c4633e03f61e179cd41314ca720572eabd6015404fd7a8149e2280b1e5",
}
EXPECTED_AMC_SHA256SUM_RECORDS_SHA256 = (
    "d3e4d9e1c095ffafb77b22a7d72a988335f36e476e240aadc27b8c23ef0f3bde"
)
EXPECTED_AMC_NUL_RECORDS_SHA256 = (
    "b7dd766852b4f422bdc861337e04d8f0184732045ea1a06a962560e86d2ce87c"
)
EXPECTED_AMC_TOTAL_BYTES = 121_065
EXPECTED_COPIED_MANIFEST_SHA256 = (
    "18dfd96f4438151197d3b6be5201476f7a71710363d81ec49c801101fa12b3ac"
)
EXPECTED_RAW_OUTPUT_MANIFEST_SHA256 = (
    "b9bcdb36952444f4db33549fa621318c5f757dbe36c4b6a11addceb46ec95053"
)
EXPECTED_RAW_OUTPUT_TOTAL_BYTES = 2_983_517_161
EXPECTED_AM_IDENTITY = "am version 12.2 (build date Aug 26 2022 19:20:13)"
EXPECTED_RAW_ROW_COUNT = 50001
DEVIATION_LOG_NAME = "FOLLOWUP_STUDY_DEVIATION_LOG.md"
EXPECTED_DEVIATION_LOG_SHA256 = (
    "b537960e9ab164353a2516f43572bb4e3dbe587e31a3ab922578b823738620e7"
)
EXPECTED_DEVIATION_LOG_BYTES = 2405
EXPECTED_FROZEN_PROTOCOLS = {
    "FOLLOWUP_STUDY_PREREGISTRATION.md": {
        "bytes": 8528,
        "sha256": ("65935dbc906317e984cf2ae8b35c5868a3f216eca2ec6290f2887976892d8457"),
    },
    "FOLLOWUP_STUDY_PROTOCOL_ADDENDUM.md": {
        "bytes": 5236,
        "sha256": ("0d47c11479a1ba0176babd3ea285e2871edbb1341406b6b044cbc53114c51a1d"),
    },
}
EXPECTED_LEGACY = {
    25: {
        "filename": "amLMT25.npz",
        "sha256": "6ddffcd2c68bbc0f6d8f6470eba0d1aa81457dcc2f348fd2d7e44c9dfe48c87b",
        "md5": "008d7fa69aff187a9edf419f3d961b4c",
    },
    50: {
        "filename": "amLMT50.npz",
        "sha256": "1fe6dd2ab7a4d65f445e20c5a8f438eb42884836e7932d86f80c30e235710f81",
        "md5": "6ec393672be8af4dfa06a3f4cf9aa32e",
    },
    75: {
        "filename": "amLMT75.npz",
        "sha256": "adbb8eb974c4e2744c3efb0f627708565f954c4029d9345e4f434699e8843f8e",
        "md5": "d6cf4bb27008179ec491864388deac58",
    },
}
EXPECTED_Q95_MD5 = "0ca7b331823237767d26016d19bffb3d"
EXPECTED_SEASONAL_REGISTRY = {
    "am_djf_q05": {"id": "463", "md5": "91545dca93d0e9300718b049893b8eea"},
    "am_djf_q25": {"id": "466", "md5": "3abe83329e39baa734b62f0e87db5a9c"},
    "am_djf_q50": {"id": "465", "md5": "004cb342896210fd23d81b329d0246f0"},
    "am_djf_q75": {"id": "462", "md5": "e2478719dd67fdbe6ea0d1fb753ab267"},
    "am_djf_q95": {"id": "464", "md5": "dc8e9e15e5df3238d9e5ecdb39e17dd4"},
    "am_jja_q05": {"id": "474", "md5": "5eae399cba2948630164230c461e24e6"},
    "am_jja_q25": {"id": "483", "md5": "0e677b7ce7f52718584c25c0fbd801c1"},
    "am_jja_q50": {"id": "478", "md5": "c866c558d1c6c1d20e323927dde1df6c"},
    "am_jja_q75": {"id": "471", "md5": "e0139e6760b2e45f2768bea572b3c081"},
    "am_jja_q95": {"id": "468", "md5": "91473b3fbb2fd50fa8ae8afb76fea8c0"},
    "am_mam_q05": {"id": "480", "md5": "b0347815c968aea4873fdff8d1d0258e"},
    "am_mam_q25": {"id": "482", "md5": "074009197665f6c642ca8a9659f6f650"},
    "am_mam_q50": {"id": "469", "md5": "1c0b684bb540ddc13bef67e27a8c15bc"},
    "am_mam_q75": {"id": "473", "md5": "72d39cbcd474622f2ac3874efe0b882b"},
    "am_mam_q95": {"id": "467", "md5": "d24a7e89e50033de830fe9fbb4627bbf"},
    "am_son_q05": {"id": "472", "md5": "e587b429e123adde3c77f524d59e71e2"},
    "am_son_q25": {"id": "475", "md5": "6c4384ff61fa00efc59e624946f4e6b6"},
    "am_son_q50": {"id": "485", "md5": "5d8b89c054e8b9dd6279cad6d1f33854"},
    "am_son_q75": {"id": "481", "md5": "d047c7f6bebfc01242ee9a7898af3165"},
    "am_son_q95": {"id": "477", "md5": "072078db4d8ad0fc2f56e51c39c42034"},
}
EXPECTED_GENERIC_REGISTRY = {
    "am_q25": {"id": "454", "md5": "008d7fa69aff187a9edf419f3d961b4c"},
    "am_q50": {"id": "455", "md5": "6ec393672be8af4dfa06a3f4cf9aa32e"},
    "am_q75": {"id": "456", "md5": "d6cf4bb27008179ec491864388deac58"},
    "am_q95": {"id": "461", "md5": "0ca7b331823237767d26016d19bffb3d"},
}

MODELS = ("am_q0", "am_q25", "am_q50", "am_q75", "am_q95")
PROFILE_FAMILIES = ("annual", "DJF", "MAM", "JJA", "SON")
BANDS = ("a1100", "a1400", "a2000")
BAND_FREQUENCIES_GHZ = {
    "a1100": 272.73,
    "a1400": 214.29,
    "a2000": 150.00,
}
REFERENCE_FREQUENCY_GHZ = 225.00
CANDIDATES = (
    "piecewise_linear_los_tau_v0",
    "pchip_los_tau_v0",
)
EXPECTED_ELEVATIONS_DEG = np.arange(10.0, 82.0, 2.0, dtype=np.float64)
STRESS_ELEVATIONS_DEG = np.arange(20.0, 82.0, 2.0, dtype=np.float64)
EXPECTED_FREQUENCY_GHZ = np.arange(50001, dtype=np.float64) / 100.0

INVENTORY_NAME = "copied_am_product_inventory.csv"
RAW_OUTPUT_INVENTORY_NAME = "copied_am_raw_output_inventory.csv"
COMPARISON_NAME = "copied_am_legacy_comparison.csv"
ANNUAL_COEFFICIENT_NAME = "copied_am_annual_fit_coefficients.csv"
STRESS_ROWS_NAME = "copied_am_operator_stress_rows.csv"
STRESS_METRICS_NAME = "copied_am_operator_stress_metrics.csv"
MANIFEST_NAME = "copied_am_manifest.json"
REPORT_NAME = "COPIED_AM_FOLLOWUP_REPORT.md"


def digest_path(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    return digest_path(path, "sha256")


def f64(value: float) -> str:
    return format(float(value), ".17e")


def render_csv(rows: list[dict[str, Any]]) -> bytes:
    if not rows:
        raise RuntimeError("cannot render an empty CSV")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=list(rows[0]),
        lineterminator="\n",
        quoting=csv.QUOTE_MINIMAL,
    )
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def render_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def load_phase0_module(repo_root: Path):
    for relative, expected in FROZEN_REPAIR_INPUTS.items():
        path = repo_root / relative
        actual = sha256_path(path)
        if actual != expected:
            raise RuntimeError(
                f"frozen repair input mismatch for {relative}: {actual} != {expected}"
            )
    path = repo_root / PHASE0_SCRIPT_REL
    spec = importlib.util.spec_from_file_location("sci_cal_001_copied_am", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import phase-0 parser: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_tolteca_registry(tolteca_root: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(tolteca_root),
            "show",
            f"{TOLTECA_REGISTRY_COMMIT}:{TOLTECA_REGISTRY_OBJECT}",
        ],
        check=True,
        stdout=subprocess.PIPE,
    )
    if sha256_bytes(result.stdout) != TOLTECA_REGISTRY_OBJECT_SHA256:
        raise RuntimeError("TolTECA seasonal-registry source-object digest mismatch")
    source = result.stdout.decode("utf-8")
    pattern = re.compile(
        r"'(am_(?:q(?:25|50|75|95)|(?:djf|jja|mam|son)_q(?:05|25|50|75|95)))'"
        r"\s*:\s*\{\s*'id'\s*:\s*'(\d+)'\s*,\s*'md5'\s*:\s*'([0-9a-f]{32})'",
        re.MULTILINE,
    )
    parsed = {
        name: {"id": datafile_id, "md5": md5}
        for name, datafile_id, md5 in pattern.findall(source)
    }
    expected = {**EXPECTED_GENERIC_REGISTRY, **EXPECTED_SEASONAL_REGISTRY}
    if parsed != expected:
        raise RuntimeError(
            "TolTECA registry entries differ from frozen identities: "
            f"parsed={parsed}, expected={expected}"
        )
    if len({item["id"] for item in EXPECTED_SEASONAL_REGISTRY.values()}) != 20:
        raise RuntimeError("seasonal registry datafile IDs are not unique")
    if len({item["md5"] for item in EXPECTED_SEASONAL_REGISTRY.values()}) != 20:
        raise RuntimeError("seasonal registry MD5 values are not unique")
    return {
        "seasonal": EXPECTED_SEASONAL_REGISTRY,
        "generic": EXPECTED_GENERIC_REGISTRY,
        "commit": TOLTECA_REGISTRY_COMMIT,
        "object": TOLTECA_REGISTRY_OBJECT,
        "object_sha256": TOLTECA_REGISTRY_OBJECT_SHA256,
    }


def modified_secant_airmass(
    elevation_rad: np.ndarray | float, pi_value: float, correction: float
) -> np.ndarray:
    values = np.asarray(elevation_rad, dtype=np.float64)
    cosine_zenith = np.cos(pi_value / 2.0 - values)
    secant_zenith = 1.0 / cosine_zenith
    return secant_zenith * (1.0 - correction * (np.square(secant_zenith) - 1.0))


def source_order_polynomial(
    coefficient_literals: tuple[str, ...], elevation_rad: np.ndarray
) -> np.ndarray:
    coefficients = tuple(float(value) for value in coefficient_literals)
    output = []
    for elevation in np.asarray(elevation_rad, dtype=np.float64):
        terms = [
            coefficients[index] * math.pow(float(elevation), 6 - index)
            for index in range(7)
        ]
        result = terms[0]
        for term in terms[1:]:
            result += term
        output.append(result)
    return np.asarray(output, dtype=np.float64)


def frequency_index(frequency_ghz: float) -> int:
    index = round(frequency_ghz * 100.0)
    if EXPECTED_FREQUENCY_GHZ[index] != frequency_ghz:
        raise RuntimeError(f"frequency is not on copied grid: {frequency_ghz}")
    return index


def parse_profile_name(filename: str) -> tuple[str, int]:
    parts = Path(filename).stem.split("_")
    if len(parts) != 3 or parts[0] != "LMT":
        raise RuntimeError(f"unexpected copied profile filename: {filename}")
    season = parts[1]
    percentile = int(parts[2])
    if season not in {"DJF", "JJA", "MAM", "SON", "annual"}:
        raise RuntimeError(f"unexpected copied season: {filename}")
    if percentile not in {5, 25, 50, 75, 95}:
        raise RuntimeError(f"unexpected copied percentile: {filename}")
    return season, percentile


def validate_deviation_log(repo_root: Path) -> dict[str, Any]:
    package_dir = repo_root / "validation/sci_cal_001_atmosphere_operator_2026-08-01"
    frozen_protocols = []
    for filename, expected in EXPECTED_FROZEN_PROTOCOLS.items():
        protocol_path = package_dir / filename
        actual = {
            "filename": filename,
            "bytes": protocol_path.stat().st_size,
            "sha256": sha256_path(protocol_path),
        }
        if (
            actual["bytes"] != expected["bytes"]
            or actual["sha256"] != expected["sha256"]
        ):
            raise RuntimeError(f"frozen follow-up protocol changed: {actual}")
        frozen_protocols.append(actual)

    path = package_dir / DEVIATION_LOG_NAME
    if path.stat().st_size != EXPECTED_DEVIATION_LOG_BYTES:
        raise RuntimeError(f"protocol-deviation byte mismatch: {path}")
    digest = sha256_path(path)
    if digest != EXPECTED_DEVIATION_LOG_SHA256:
        raise RuntimeError(f"protocol-deviation digest mismatch: {path}")
    return {
        "filename": DEVIATION_LOG_NAME,
        "bytes": path.stat().st_size,
        "sha256": digest,
        "status": "clarification_only_no_candidate_or_numeric_reinterpretation",
        "stopped_study_c_identities": [
            "piecewise_linear_los_tau_v1",
            "pchip_los_tau_v1",
        ],
        "diagnostic_c1_evaluated_identities": list(CANDIDATES),
        "frozen_protocol_records": frozen_protocols,
    }


def inventory_amc_inputs(am_root: Path) -> dict[str, Any]:
    input_root = am_root / "LMT_am_inputs"
    release_root = am_root.parent / "am-12.2/cookbook/sites/LMT"
    actual_names = sorted(path.name for path in input_root.glob("LMT_*.amc"))
    release_names = sorted(path.name for path in release_root.glob("LMT_*.amc"))
    expected_names = sorted(EXPECTED_AMC_SHA256)
    if actual_names != expected_names or release_names != expected_names:
        raise RuntimeError(
            "copied AMC inventory differs from frozen 25-profile set: "
            f"inputs={actual_names}, release={release_names}, expected={expected_names}"
        )

    files: list[dict[str, Any]] = []
    sha256sum_records = bytearray()
    nul_records = hashlib.sha256()
    total_bytes = 0
    for filename in sorted(expected_names, key=lambda item: item.encode("utf-8")):
        input_path = input_root / filename
        release_path = release_root / filename
        digest = sha256_path(input_path)
        release_digest = sha256_path(release_path)
        expected = EXPECTED_AMC_SHA256[filename]
        if digest != expected or release_digest != expected:
            raise RuntimeError(
                f"copied AMC SHA-256 mismatch for {filename}: "
                f"input={digest}, release={release_digest}, expected={expected}"
            )
        size = input_path.stat().st_size
        if release_path.stat().st_size != size:
            raise RuntimeError(f"copied/release AMC byte mismatch for {filename}")
        total_bytes += size
        sha256sum_records.extend(
            f"{digest}  cookbook/sites/LMT/{filename}\n".encode("utf-8")
        )
        nul_records.update(filename.encode("utf-8"))
        nul_records.update(b"\0")
        nul_records.update(bytes.fromhex(digest))
        nul_records.update(b"\0")
        files.append(
            {
                "filename": filename,
                "path_relative_to_am_root": f"LMT_am_inputs/{filename}",
                "release_path_relative_to_am_source_root": (
                    f"cookbook/sites/LMT/{filename}"
                ),
                "bytes": size,
                "sha256": digest,
                "release_copy_exact": True,
            }
        )

    sha256sum_digest = sha256_bytes(bytes(sha256sum_records))
    nul_digest = nul_records.hexdigest()
    if sha256sum_digest != EXPECTED_AMC_SHA256SUM_RECORDS_SHA256:
        raise RuntimeError("copied AMC sha256sum-record aggregate mismatch")
    if nul_digest != EXPECTED_AMC_NUL_RECORDS_SHA256:
        raise RuntimeError("copied AMC NUL-record aggregate mismatch")
    if total_bytes != EXPECTED_AMC_TOTAL_BYTES:
        raise RuntimeError("copied AMC total-byte mismatch")
    return {
        "file_count": len(files),
        "total_bytes": total_bytes,
        "canonical_sha256sum_records": {
            "algorithm": (
                "UTF-8 concatenation in AMC-basename bytewise sort order of "
                "sha256<TWO_SPACES>cookbook/sites/LMT/basename<LF>"
            ),
            "sha256": sha256sum_digest,
        },
        "canonical_nul_records": {
            "algorithm": "sha256(basename UTF-8 NUL raw 32-byte SHA-256 NUL)",
            "sha256": nul_digest,
        },
        "files": files,
    }


def build_legacy_state(repo_root: Path) -> dict[str, Any]:
    phase0 = load_phase0_module(repo_root)
    source_model = phase0.parse_source(repo_root)
    _, thresholds_by_name = phase0.build_rows(source_model)
    thresholds_by_name["am_q0"] = 0.0
    thresholds = np.asarray(
        [thresholds_by_name[model] for model in MODELS], dtype=np.float64
    )
    thresholds[0] = 0.0
    pi_value = float(source_model.pi_literal)
    degree_divisor = float(source_model.degree_divisor_literal)
    correction = float(source_model.airmass_correction_literal)
    elevation_rad = STRESS_ELEVATIONS_DEG * pi_value / degree_divisor
    airmass = modified_secant_airmass(elevation_rad, pi_value, correction)
    anchor_los_tau: dict[str, np.ndarray] = {}
    anchor_ratio: dict[str, np.ndarray] = {}
    for band in BANDS:
        los_rows = [np.zeros_like(elevation_rad)]
        ratio_rows = [np.ones_like(elevation_rad)]
        for model in MODELS[1:]:
            ratio = source_order_polynomial(
                source_model.coefficients[model][band], elevation_rad
            )
            transmission = ratio * np.exp(-airmass * thresholds_by_name[model])
            if not (
                np.all(np.isfinite(transmission))
                and np.all(transmission > 0.0)
                and np.all(transmission <= 1.0)
            ):
                raise RuntimeError(f"invalid legacy surface for {model}/{band}")
            ratio_rows.append(ratio)
            los_rows.append(-np.log(transmission))
        anchor_los_tau[band] = np.stack(los_rows)
        anchor_ratio[band] = np.stack(ratio_rows)
    return {
        "source_model": source_model,
        "thresholds": thresholds,
        "thresholds_by_name": thresholds_by_name,
        "pi_value": pi_value,
        "degree_divisor": degree_divisor,
        "correction": correction,
        "elevation_rad": elevation_rad,
        "airmass": airmass,
        "anchor_los_tau": anchor_los_tau,
        "anchor_ratio": anchor_ratio,
    }


def validate_legacy_sources(source_dir: Path) -> None:
    for metadata in EXPECTED_LEGACY.values():
        path = source_dir / metadata["filename"]
        if sha256_path(path) != metadata["sha256"]:
            raise RuntimeError(f"legacy SHA-256 mismatch: {path}")
        if digest_path(path, "md5") != metadata["md5"]:
            raise RuntimeError(f"legacy MD5 mismatch: {path}")


def profile_support_interval(tau225: float, thresholds: np.ndarray) -> str:
    if tau225 < thresholds[0] or tau225 > thresholds[-1]:
        return "outside_legacy_q0_q95_support_excluded_no_extrapolation"
    if tau225 <= thresholds[1]:
        return "q0_q25"
    upper = int(np.searchsorted(thresholds, tau225, side="left"))
    upper = min(max(upper, 2), len(thresholds) - 1)
    return f"q{MODELS[upper - 1][4:]}_q{MODELS[upper][4:]}"


def load_copied_suite(
    am_root: Path, legacy: dict[str, Any], registry: dict[str, Any]
) -> dict[str, Any]:
    amc_inputs = inventory_amc_inputs(am_root)
    amc_by_stem = {Path(item["filename"]).stem: item for item in amc_inputs["files"]}
    source_dir = am_root / "LMT_am_npz"
    actual_names = sorted(path.name for path in source_dir.glob("*.npz"))
    expected_names = sorted(EXPECTED_COPIED_SHA256)
    if actual_names != expected_names:
        raise RuntimeError(
            "copied NPZ inventory differs from frozen 25-product suite: "
            f"actual={actual_names}, expected={expected_names}"
        )

    stress_indices = np.flatnonzero(
        np.isin(EXPECTED_ELEVATIONS_DEG, STRESS_ELEVATIONS_DEG)
    )
    elevation_80_index = int(np.flatnonzero(EXPECTED_ELEVATIONS_DEG == 80.0)[0])
    reference_index = frequency_index(REFERENCE_FREQUENCY_GHZ)
    airmass_80 = float(
        modified_secant_airmass(
            80.0 * legacy["pi_value"] / legacy["degree_divisor"],
            legacy["pi_value"],
            legacy["correction"],
        )
    )
    profiles: dict[str, dict[str, Any]] = {}
    manifest_lines: list[str] = []
    inventory_rows: list[dict[str, str]] = []

    for filename in actual_names:
        path = source_dir / filename
        sha256 = sha256_path(path)
        if sha256 != EXPECTED_COPIED_SHA256[filename]:
            raise RuntimeError(f"copied suite SHA-256 mismatch: {path}")
        md5 = digest_path(path, "md5")
        size = path.stat().st_size
        manifest_lines.append(f"{filename}\t{size}\t{sha256}\t{md5}\n")
        season, percentile = parse_profile_name(filename)
        registry_name = ""
        registry_id = ""
        registry_md5 = ""
        if season == "annual":
            registry_status = "no_matching_generic_registry_identity"
            generic_relation = "no_matching_generic_registry_identity"
        else:
            registry_name = f"am_{season.lower()}_q{percentile:02d}"
            registry_entry = registry["seasonal"].get(registry_name)
            if registry_entry is None:
                raise RuntimeError(
                    f"missing seasonal registry identity: {registry_name}"
                )
            registry_id = registry_entry["id"]
            registry_md5 = registry_entry["md5"]
            if md5 != registry_md5:
                raise RuntimeError(
                    f"seasonal registry MD5 mismatch for {filename}: "
                    f"{md5} != {registry_md5}"
                )
            registry_status = "exact_seasonal_registry_md5_match"
            generic_relation = "seasonal_registry_identity_generic_q_artifacts_separate"
        generic_md5s = {item["md5"] for item in registry["generic"].values()}
        if md5 in generic_md5s:
            raise RuntimeError(
                f"copied product unexpectedly matches separate generic registry: {filename}"
            )

        with np.load(path, allow_pickle=False) as archive:
            if archive.files != ["el", "atmFreq", "atmTRJ", "atmTtx", "atmTaun"]:
                raise RuntimeError(f"unexpected NPZ members: {path}: {archive.files}")
            elevation = archive["el"]
            frequency = archive["atmFreq"]
            trj = archive["atmTRJ"]
            transmission = archive["atmTtx"]
            direct_tau = archive["atmTaun"]
            expected_shape = (50001, 36)
            if elevation.shape != (36,) or any(
                values.shape != expected_shape
                for values in (frequency, trj, transmission, direct_tau)
            ):
                raise RuntimeError(f"unexpected array shape: {path}")
            if not np.array_equal(elevation, EXPECTED_ELEVATIONS_DEG):
                raise RuntimeError(f"unexpected elevation grid: {path}")
            if not (
                np.array_equal(frequency[:, 0], EXPECTED_FREQUENCY_GHZ)
                and np.all(frequency == frequency[:, [0]])
            ):
                raise RuntimeError(f"unexpected frequency grid: {path}")
            if not all(
                bool(np.all(np.isfinite(values)))
                for values in (trj, transmission, direct_tau)
            ):
                raise RuntimeError(f"non-finite copied data: {path}")
            if not (
                np.all(transmission >= 0.0)
                and np.all(transmission <= 1.0)
                and np.all(direct_tau >= 0.0)
            ):
                raise RuntimeError(f"invalid copied transmission/tau domain: {path}")

            tx225_80 = float(transmission[reference_index, elevation_80_index])
            if not 0.0 < tx225_80 <= 1.0:
                raise RuntimeError(f"invalid copied T225 at 80 degrees: {path}")
            tau225 = -math.log(tx225_80) / airmass_80
            positive = transmission > 0.0
            tau_tx_consistency = float(
                np.max(np.abs(direct_tau[positive] + np.log(transmission[positive])))
            )
            band_tau = {
                band: direct_tau[frequency_index(frequency_ghz), stress_indices].copy()
                for band, frequency_ghz in BAND_FREQUENCIES_GHZ.items()
            }
            band_tx = {
                band: transmission[
                    frequency_index(frequency_ghz), stress_indices
                ].copy()
                for band, frequency_ghz in BAND_FREQUENCIES_GHZ.items()
            }
            tx225 = transmission[reference_index, stress_indices].copy()

        support = profile_support_interval(tau225, legacy["thresholds"])
        eligible = not support.startswith("outside_")
        profile_id = Path(filename).stem
        amc_input = amc_by_stem.get(profile_id)
        if amc_input is None:
            raise RuntimeError(f"missing one-to-one AMC input for {profile_id}")
        profiles[profile_id] = {
            "profile_id": profile_id,
            "filename": filename,
            "path": path,
            "season": season,
            "percentile": percentile,
            "sha256": sha256,
            "md5": md5,
            "registry_name": registry_name,
            "registry_id": registry_id,
            "registry_md5": registry_md5,
            "registry_status": registry_status,
            "generic_registry_relation": generic_relation,
            "size": size,
            "tx225_80": tx225_80,
            "tau225": tau225,
            "support": support,
            "eligible": eligible,
            "band_tau": band_tau,
            "band_tx": band_tx,
            "tx225": tx225,
            "amc_input": amc_input,
        }
        inventory_rows.append(
            {
                "profile_id": profile_id,
                "season": season,
                "percentile": str(percentile),
                "npz_path_relative_to_am_root": f"LMT_am_npz/{filename}",
                "bytes": str(size),
                "sha256": sha256,
                "md5": md5,
                "amc_filename": amc_input["filename"],
                "amc_path_relative_to_am_root": amc_input["path_relative_to_am_root"],
                "amc_bytes": str(amc_input["bytes"]),
                "amc_sha256": amc_input["sha256"],
                "amc_release_copy_exact": str(amc_input["release_copy_exact"]).lower(),
                "tolteca_registry_name": registry_name,
                "tolteca_datafile_id": registry_id,
                "tolteca_registry_md5": registry_md5,
                "tolteca_registry_provenance_commit": registry["commit"],
                "tolteca_registry_status": registry_status,
                "generic_q_registry_relation": generic_relation,
                "npz_members": "el;atmFreq;atmTRJ;atmTtx;atmTaun",
                "elevation_grid_deg": "10:80:2",
                "frequency_grid_ghz": "0:500:0.01",
                "spectral_array_shape": "50001x36",
                "zero_transmission_count": str(
                    int(np.size(transmission) - np.count_nonzero(transmission))
                ),
                "max_abs_direct_tau_plus_log_positive_tx": f64(tau_tx_consistency),
                "tx225_at_80deg": f64(tx225_80),
                "modified_secant_tau225_coordinate": f64(tau225),
                "legacy_support_interval": support,
                "stress_eligible_no_extrapolation": str(eligible).lower(),
                "scientific_identity": "copied_am_12_2_distinct_product_identity",
                "historical_generic_generator_association": "not_established",
            }
        )

    copied_manifest_digest = sha256_bytes("".join(manifest_lines).encode("utf-8"))
    if copied_manifest_digest != EXPECTED_COPIED_MANIFEST_SHA256:
        raise RuntimeError(
            "copied canonical manifest mismatch: "
            f"{copied_manifest_digest} != {EXPECTED_COPIED_MANIFEST_SHA256}"
        )
    return {
        "profiles": profiles,
        "inventory_rows": inventory_rows,
        "manifest_sha256": copied_manifest_digest,
        "manifest_definition": "sorted basename\\tbytes\\tsha256\\tmd5\\n",
        "total_bytes": sum(profile["size"] for profile in profiles.values()),
        "source_dir": source_dir,
        "amc_inputs": amc_inputs,
        "airmass_80": airmass_80,
    }


def build_raw_output_inventory(am_root: Path, copied: dict[str, Any]) -> dict[str, Any]:
    output_dir = am_root / "LMT_am_outputs"
    expected: dict[str, tuple[dict[str, Any], int, int]] = {}
    for profile_id in sorted(copied["profiles"]):
        profile = copied["profiles"][profile_id]
        for zenith_angle in range(10, 81, 2):
            filename = f"{profile_id}_{zenith_angle}.dat"
            expected[filename] = (profile, zenith_angle, 90 - zenith_angle)
    actual_names = sorted(path.name for path in output_dir.glob("*.dat"))
    expected_names = sorted(expected)
    if actual_names != expected_names:
        raise RuntimeError(
            "copied raw-output inventory differs from expected 900-file matrix: "
            f"actual_count={len(actual_names)}, expected_count={len(expected_names)}"
        )

    rows: list[dict[str, str]] = []
    manifest_records: list[tuple[str, int, str]] = []
    for profile_id in sorted(copied["profiles"]):
        profile = copied["profiles"][profile_id]
        with np.load(profile["path"], allow_pickle=False) as archive:
            elevation_grid = archive["el"]
            npz_frequency = archive["atmFreq"]
            npz_tau = archive["atmTaun"]
            npz_transmission = archive["atmTtx"]
            npz_trj = archive["atmTRJ"]
            for zenith_angle in range(10, 81, 2):
                filename = f"{profile_id}_{zenith_angle}.dat"
                path = output_dir / filename
                _, expected_za, elevation = expected[filename]
                if expected_za != zenith_angle:
                    raise RuntimeError(f"internal ZA mapping error for {filename}")
                elevation_matches = np.flatnonzero(elevation_grid == elevation)
                if elevation_matches.size != 1:
                    raise RuntimeError(
                        f"NPZ elevation column not unique for {filename}"
                    )
                column = int(elevation_matches[0])

                content = path.read_bytes()
                sha256 = sha256_bytes(content)
                size = len(content)
                relative_path = f"LMT_am_outputs/{filename}"
                manifest_records.append((relative_path, size, sha256))
                try:
                    lines = content.decode("ascii").splitlines()
                except UnicodeDecodeError as error:
                    raise RuntimeError(f"non-ASCII raw AM output: {path}") from error

                identity_lines = [
                    line[2:] for line in lines if line.startswith("# am version ")
                ]
                if identity_lines != [EXPECTED_AM_IDENTITY]:
                    raise RuntimeError(
                        f"unexpected AM identity in {path}: {identity_lines}"
                    )
                output_declarations = [
                    " ".join(line.split())
                    for line in lines
                    if line.startswith("output ")
                ]
                expected_declaration = "output f GHz tau neper tx none Trj K Tb K"
                if output_declarations != [expected_declaration]:
                    raise RuntimeError(
                        f"unexpected AM output columns in {path}: {output_declarations}"
                    )

                try:
                    numeric_start = next(
                        index
                        for index, line in enumerate(lines)
                        if line and line[0].isdigit()
                    )
                except StopIteration as error:
                    raise RuntimeError(f"no numeric grid in {path}") from error
                numeric_end = numeric_start
                while (
                    numeric_end < len(lines)
                    and lines[numeric_end]
                    and lines[numeric_end][0].isdigit()
                ):
                    numeric_end += 1
                numeric_lines = lines[numeric_start:numeric_end]
                row_count = len(numeric_lines)
                if row_count != EXPECTED_RAW_ROW_COUNT:
                    raise RuntimeError(
                        f"unexpected numeric row count in {path}: {row_count}"
                    )
                values = np.fromstring("\n".join(numeric_lines), sep=" ")
                if values.size != row_count * 5:
                    raise RuntimeError(
                        f"numeric AM output is not five columns in {path}: "
                        f"values={values.size}, rows={row_count}"
                    )
                grid = values.reshape(row_count, 5)

                frequency_equal = bool(
                    np.array_equal(grid[:, 0], npz_frequency[:, column])
                )
                tau_equal = bool(np.array_equal(grid[:, 1], npz_tau[:, column]))
                transmission_equal = bool(
                    np.array_equal(grid[:, 2], npz_transmission[:, column])
                )
                trj_equal = bool(np.array_equal(grid[:, 3], npz_trj[:, column]))
                tb_present = grid.shape[1] == 5
                tb_finite = bool(np.all(np.isfinite(grid[:, 4])))
                if not all(
                    (
                        frequency_equal,
                        tau_equal,
                        transmission_equal,
                        trj_equal,
                        tb_present,
                        tb_finite,
                    )
                ):
                    raise RuntimeError(
                        "raw DAT/NPZ validation failed for "
                        f"{path}: f={frequency_equal}, tau={tau_equal}, "
                        f"tx={transmission_equal}, Trj={trj_equal}, "
                        f"Tb_present={tb_present}, Tb_finite={tb_finite}"
                    )

                footer = lines[numeric_end:]
                warning_matches = [
                    re.search(r"for which this occurred\.\s+Count:\s+(\d+)$", line)
                    for line in footer
                ]
                warning_counts = [
                    int(match.group(1))
                    for match in warning_matches
                    if match is not None
                ]
                warning_header_present = any(
                    line.startswith("! Warning: Encountered in-band lines narrower")
                    for line in footer
                )
                if not warning_header_present or len(warning_counts) != 1:
                    raise RuntimeError(
                        f"unexpected unresolved-line warning footer in {path}"
                    )
                exit_matches = [
                    re.search(r"Exited with exit code (\d+)$", line)
                    for line in footer
                    if line.startswith("srun: error:")
                ]
                exit_codes = [
                    int(match.group(1)) for match in exit_matches if match is not None
                ]
                if exit_codes != [1]:
                    raise RuntimeError(f"unexpected Slurm return footer in {path}")
                retry_flag = any(
                    line.startswith("srun: Job ")
                    and "step creation temporarily disabled, retrying" in line
                    for line in lines[:numeric_start]
                )
                footer_status = "complete_numeric_grid_then_slurm_exit_code_1"
                rows.append(
                    {
                        "relative_path": relative_path,
                        "bytes": str(size),
                        "sha256": sha256,
                        "profile_id": profile_id,
                        "season": profile["season"],
                        "percentile": str(profile["percentile"]),
                        "zenith_angle_deg": str(zenith_angle),
                        "elevation_deg": str(elevation),
                        "numeric_row_count": str(row_count),
                        "am_identity": EXPECTED_AM_IDENTITY,
                        "return_footer_status": footer_status,
                        "footer_exit_code": "1",
                        "unresolved_line_warning_count": str(warning_counts[0]),
                        "slurm_retry_flag": str(retry_flag).lower(),
                        "frequency_exact_npz": str(frequency_equal).lower(),
                        "direct_tau_exact_npz": str(tau_equal).lower(),
                        "transmission_exact_npz": str(transmission_equal).lower(),
                        "trj_exact_npz": str(trj_equal).lower(),
                        "tb_column_present": str(tb_present).lower(),
                        "tb_all_finite": str(tb_finite).lower(),
                        "tb_npz_comparison_status": (
                            "not_applicable_npz_omits_tb_no_comparison_invented"
                        ),
                        "validation_status": (
                            "complete_grid_four_retained_fields_exact_npz"
                        ),
                    }
                )

    rows.sort(key=lambda row: row["relative_path"].encode("utf-8"))
    manifest_records.sort(key=lambda item: item[0].encode("utf-8"))
    manifest_text = "".join(
        f"{relative_path}\t{size}\t{sha256}\n"
        for relative_path, size, sha256 in manifest_records
    )
    manifest_sha256 = sha256_bytes(manifest_text.encode("utf-8"))
    total_bytes = sum(size for _, size, _ in manifest_records)
    if manifest_sha256 != EXPECTED_RAW_OUTPUT_MANIFEST_SHA256:
        raise RuntimeError(
            "raw-output canonical manifest mismatch: "
            f"{manifest_sha256} != {EXPECTED_RAW_OUTPUT_MANIFEST_SHA256}"
        )
    if total_bytes != EXPECTED_RAW_OUTPUT_TOTAL_BYTES:
        raise RuntimeError(
            f"raw-output total-byte mismatch: {total_bytes} != "
            f"{EXPECTED_RAW_OUTPUT_TOTAL_BYTES}"
        )
    warning_values = sorted({int(row["unresolved_line_warning_count"]) for row in rows})
    return {
        "rows": rows,
        "file_count": len(rows),
        "total_bytes": total_bytes,
        "manifest_definition": (
            "UTF-8 concatenation in relative-path bytewise sort order of "
            "relative_path<TAB>bytes<TAB>sha256<LF>"
        ),
        "manifest_sha256": manifest_sha256,
        "warning_count_distribution": {
            str(value): sum(
                int(row["unresolved_line_warning_count"]) == value for row in rows
            )
            for value in warning_values
        },
        "slurm_retry_file_count": sum(
            row["slurm_retry_flag"] == "true" for row in rows
        ),
        "return_footer_status_distribution": {
            status: sum(row["return_footer_status"] == status for row in rows)
            for status in sorted({row["return_footer_status"] for row in rows})
        },
        "all_four_retained_fields_exact_npz": all(
            all(
                row[field] == "true"
                for field in (
                    "frequency_exact_npz",
                    "direct_tau_exact_npz",
                    "transmission_exact_npz",
                    "trj_exact_npz",
                )
            )
            for row in rows
        ),
        "all_tb_present_and_finite": all(
            row["tb_column_present"] == "true" and row["tb_all_finite"] == "true"
            for row in rows
        ),
    }


def interpolate_legacy(
    candidate: str,
    tau225: float,
    thresholds: np.ndarray,
    anchor_los_tau: np.ndarray,
) -> np.ndarray:
    if tau225 < thresholds[0] or tau225 > thresholds[-1]:
        raise ValueError("no extrapolation outside legacy q0--q95 support")
    if tau225 <= thresholds[1]:
        return anchor_los_tau[1] * (tau225 / thresholds[1])
    if candidate == "piecewise_linear_los_tau_v0":
        upper = int(np.searchsorted(thresholds, tau225, side="right"))
        upper = min(upper, len(thresholds) - 1)
        lower = upper - 1
        fraction = (tau225 - thresholds[lower]) / (
            thresholds[upper] - thresholds[lower]
        )
        return (1.0 - fraction) * anchor_los_tau[lower] + fraction * anchor_los_tau[
            upper
        ]
    if candidate == "pchip_los_tau_v0":
        interpolator = PchipInterpolator(
            thresholds[1:], anchor_los_tau[1:], axis=0, extrapolate=False
        )
        result = np.asarray(interpolator(tau225), dtype=np.float64)
        if not np.all(np.isfinite(result)):
            raise RuntimeError("PCHIP returned non-finite in-support values")
        return result
    raise ValueError(f"unknown candidate: {candidate}")


def build_all_family_legacy_comparison_rows(
    copied: dict[str, Any],
    legacy: dict[str, Any],
    legacy_source_dir: Path,
) -> list[dict[str, str]]:
    model_indices = {25: 1, 50: 2, 75: 3, 95: 4}
    copied_elevation_indices = np.flatnonzero(
        np.isin(EXPECTED_ELEVATIONS_DEG, STRESS_ELEVATIONS_DEG)
    )
    raw_grids: dict[int, dict[str, np.ndarray]] = {}
    for percentile, metadata in EXPECTED_LEGACY.items():
        path = legacy_source_dir / metadata["filename"]
        with np.load(path, allow_pickle=False) as archive:
            if archive.files != ["el", "atmFreq", "atmTRJ", "atmTtx"]:
                raise RuntimeError(f"unexpected legacy members: {path}")
            raw_grids[percentile] = {
                name: np.asarray(archive[name]).copy()
                for name in ("atmFreq", "atmTRJ", "atmTtx")
            }

    rows: list[dict[str, str]] = []
    for family in PROFILE_FAMILIES:
        for percentile in (25, 50, 75, 95):
            profile = copied["profiles"][f"LMT_{family}_{percentile}"]
            raw_status = "missing_expected_q95_not_substituted"
            legacy_sha = ""
            legacy_md5 = EXPECTED_Q95_MD5
            common_frequency = "not_evaluable"
            common_count = ""
            raw_equal = "not_evaluable_missing_q95"
            max_ttx = ""
            rms_ttx = ""
            max_trj = ""
            rms_trj = ""
            if percentile in raw_grids:
                metadata = EXPECTED_LEGACY[percentile]
                with np.load(profile["path"], allow_pickle=False) as archive:
                    copied_frequency = archive["atmFreq"][:, copied_elevation_indices]
                    copied_ttx = archive["atmTtx"][:, copied_elevation_indices]
                    copied_trj = archive["atmTRJ"][:, copied_elevation_indices]
                frequency_equal = bool(
                    np.array_equal(raw_grids[percentile]["atmFreq"], copied_frequency)
                )
                ttx_difference = copied_ttx - raw_grids[percentile]["atmTtx"]
                trj_difference = copied_trj - raw_grids[percentile]["atmTRJ"]
                max_ttx_value = float(np.max(np.abs(ttx_difference)))
                rms_ttx_value = float(np.sqrt(np.mean(np.square(ttx_difference))))
                max_trj_value = float(np.max(np.abs(trj_difference)))
                rms_trj_value = float(np.sqrt(np.mean(np.square(trj_difference))))
                raw_status = "recovered_legacy_raw_grid_compared"
                legacy_sha = metadata["sha256"]
                legacy_md5 = metadata["md5"]
                common_frequency = str(frequency_equal).lower()
                common_count = str(ttx_difference.size)
                raw_equal = str(
                    frequency_equal and max_ttx_value == 0.0 and max_trj_value == 0.0
                ).lower()
                max_ttx = f64(max_ttx_value)
                rms_ttx = f64(rms_ttx_value)
                max_trj = f64(max_trj_value)
                rms_trj = f64(rms_trj_value)

            model_name = f"am_q{percentile}"
            model_index = model_indices[percentile]
            for band in BANDS:
                legacy_los_tau = legacy["anchor_los_tau"][band][model_index]
                copied_los_tau = profile["band_tau"][band]
                signed_error = np.expm1(legacy_los_tau - copied_los_tau)
                absolute_error = np.abs(signed_error)
                max_index = int(np.argmax(absolute_error))
                copied_ratio = profile["band_tx"][band] / profile["tx225"]
                legacy_ratio = legacy["anchor_ratio"][band][model_index]
                ratio_fit_tx = legacy_ratio * profile["tx225"]
                ratio_error = np.abs(
                    np.expm1(-np.log(ratio_fit_tx) + np.log(profile["band_tx"][band]))
                )
                copied_fit = np.polyfit(legacy["elevation_rad"], copied_ratio, 6)
                source_coefficients = np.asarray(
                    [
                        float(value)
                        for value in legacy["source_model"].coefficients[model_name][
                            band
                        ]
                    ],
                    dtype=np.float64,
                )
                rows.append(
                    {
                        "legacy_model": model_name,
                        "copied_profile_family": family,
                        "copied_profile_id": profile["profile_id"],
                        "band": band,
                        "frequency_ghz": f64(BAND_FREQUENCIES_GHZ[band]),
                        "copied_sha256": profile["sha256"],
                        "copied_md5": profile["md5"],
                        "legacy_raw_status": raw_status,
                        "legacy_raw_sha256": legacy_sha,
                        "legacy_raw_or_expected_md5": legacy_md5,
                        "common_grid_frequency_exact": common_frequency,
                        "common_grid_sample_count": common_count,
                        "legacy_raw_common_grid_content_equal": raw_equal,
                        "max_abs_common_grid_transmission_difference": max_ttx,
                        "rms_common_grid_transmission_difference": rms_ttx,
                        "max_abs_common_grid_trj_difference_k": max_trj,
                        "rms_common_grid_trj_difference_k": rms_trj,
                        "legacy_selector_tau225": f64(
                            legacy["thresholds_by_name"][model_name]
                        ),
                        "copied_modified_secant_tau225": f64(profile["tau225"]),
                        "signed_tau225_coordinate_difference": f64(
                            profile["tau225"] - legacy["thresholds_by_name"][model_name]
                        ),
                        "copied_tx225_at_80deg": f64(profile["tx225_80"]),
                        "legacy_tx225_at_80deg_literal": legacy[
                            "source_model"
                        ].transmissions[model_name],
                        "max_abs_copied_fit_coefficient_difference": f64(
                            float(np.max(np.abs(copied_fit - source_coefficients)))
                        ),
                        "max_abs_ratio_only_fractional_correction_error": f64(
                            float(np.max(ratio_error))
                        ),
                        "max_abs_full_legacy_anchor_fractional_correction_error": f64(
                            float(np.max(absolute_error))
                        ),
                        "p95_abs_full_legacy_anchor_fractional_correction_error": f64(
                            float(np.quantile(absolute_error, 0.95, method="linear"))
                        ),
                        "worst_full_legacy_anchor_elevation_deg": f64(
                            STRESS_ELEVATIONS_DEG[max_index]
                        ),
                        "family_ranking_claim": "none_metric_values_only",
                        "legacy_identity_match": "false",
                        "disposition": (
                            "identity_diagnostic_only_copied_am12_2_not_substitute"
                        ),
                    }
                )
    return rows


def build_annual_fit_coefficient_rows(
    copied: dict[str, Any], legacy: dict[str, Any]
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for percentile in (25, 50, 75, 95):
        profile = copied["profiles"][f"LMT_annual_{percentile}"]
        model_name = f"am_q{percentile}"
        for band in BANDS:
            copied_ratio = profile["band_tx"][band] / profile["tx225"]
            coefficients = np.polyfit(legacy["elevation_rad"], copied_ratio, 6)
            rounded = np.round(coefficients, 8)
            source_literals = legacy["source_model"].coefficients[model_name][band]
            for coefficient_index, values in enumerate(
                zip(coefficients, rounded, source_literals, strict=True)
            ):
                value, rounded_value, source_literal = values
                source_8_decimal = f"{float(source_literal):.8f}"
                rounded_8_decimal = f"{float(rounded_value):.8f}"
                rows.append(
                    {
                        "copied_profile_id": profile["profile_id"],
                        "copied_profile_sha256": profile["sha256"],
                        "legacy_model_comparison_identity": model_name,
                        "band": band,
                        "frequency_ghz": f64(BAND_FREQUENCIES_GHZ[band]),
                        "fit_elevation_grid_deg": "20:80:2",
                        "fit_elevation_coordinate": "repair_base_radians",
                        "fit_degree": "6",
                        "coefficient_index_descending": str(coefficient_index),
                        "polynomial_power": str(6 - coefficient_index),
                        "unrounded_binary64_coefficient": f64(value),
                        "rounded_8_decimal_coefficient": rounded_8_decimal,
                        "legacy_source_8_decimal_coefficient": source_8_decimal,
                        "rounded_8_decimal_matches_legacy_source": str(
                            rounded_8_decimal == source_8_decimal
                        ).lower(),
                        "signed_rounded_coefficient_difference": f64(
                            float(rounded_value) - float(source_literal)
                        ),
                        "identity_disposition": (
                            "annual_am12_2_fit_diagnostic_not_generic_q_substitute"
                        ),
                    }
                )
    return rows


def build_stress_rows(
    copied: dict[str, Any], legacy: dict[str, Any]
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for profile_id in sorted(copied["profiles"]):
        profile = copied["profiles"][profile_id]
        if not profile["eligible"]:
            continue
        for candidate in CANDIDATES:
            for band in BANDS:
                candidate_tau = interpolate_legacy(
                    candidate,
                    profile["tau225"],
                    legacy["thresholds"],
                    legacy["anchor_los_tau"][band],
                )
                truth_tau = profile["band_tau"][band]
                signed_error = np.expm1(candidate_tau - truth_tau)
                for elevation_index, elevation in enumerate(STRESS_ELEVATIONS_DEG):
                    rows.append(
                        {
                            "profile_id": profile_id,
                            "season": profile["season"],
                            "percentile": str(profile["percentile"]),
                            "modified_secant_tau225_coordinate": f64(profile["tau225"]),
                            "legacy_support_interval": profile["support"],
                            "candidate": candidate,
                            "band": band,
                            "frequency_ghz": f64(BAND_FREQUENCIES_GHZ[band]),
                            "elevation_deg": f64(elevation),
                            "copied_direct_los_tau": f64(truth_tau[elevation_index]),
                            "legacy_candidate_los_tau": f64(
                                candidate_tau[elevation_index]
                            ),
                            "signed_fractional_correction_error": f64(
                                signed_error[elevation_index]
                            ),
                            "absolute_fractional_correction_error": f64(
                                abs(signed_error[elevation_index])
                            ),
                            "truth_kind": (
                                "copied_am12_2_direct_atmTaun_post_discovery_stress"
                            ),
                            "evaluation_status": (
                                "in_support_no_extrapolation_not_operator_authorization"
                            ),
                        }
                    )
    return rows


def build_stress_metrics(
    stress_rows: list[dict[str, str]], copied: dict[str, Any]
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    eligible_profiles = {
        profile["profile_id"]
        for profile in copied["profiles"].values()
        if profile["eligible"]
    }
    excluded_profiles = set(copied["profiles"]) - eligible_profiles
    prediction = {
        (row["profile_id"], row["band"], row["elevation_deg"], row["candidate"]): float(
            row["legacy_candidate_los_tau"]
        )
        for row in stress_rows
    }
    for candidate in CANDIDATES:
        for band in BANDS:
            selected = [
                row
                for row in stress_rows
                if row["candidate"] == candidate and row["band"] == band
            ]
            signed = np.asarray(
                [float(row["signed_fractional_correction_error"]) for row in selected]
            )
            absolute = np.abs(signed)
            max_index = int(np.argmax(absolute))
            worst = selected[max_index]
            disagreement = []
            other_candidate = next(item for item in CANDIDATES if item != candidate)
            for row in selected:
                key = (
                    row["profile_id"],
                    band,
                    row["elevation_deg"],
                    other_candidate,
                )
                difference = abs(
                    float(row["legacy_candidate_los_tau"]) - prediction[key]
                )
                disagreement.append(math.expm1(difference))
            rows.append(
                {
                    "candidate": candidate,
                    "band": band,
                    "frequency_ghz": f64(BAND_FREQUENCIES_GHZ[band]),
                    "evaluated_profile_count": str(len(eligible_profiles)),
                    "excluded_profile_count_no_extrapolation": str(
                        len(excluded_profiles)
                    ),
                    "evaluated_row_count": str(len(selected)),
                    "max_abs_fractional_correction_error": f64(float(np.max(absolute))),
                    "p95_abs_fractional_correction_error": f64(
                        float(np.quantile(absolute, 0.95, method="linear"))
                    ),
                    "median_abs_fractional_correction_error": f64(
                        float(np.quantile(absolute, 0.5, method="linear"))
                    ),
                    "rms_fractional_correction_error": f64(
                        float(np.sqrt(np.mean(np.square(signed))))
                    ),
                    "min_signed_fractional_correction_error": f64(
                        float(np.min(signed))
                    ),
                    "max_signed_fractional_correction_error": f64(
                        float(np.max(signed))
                    ),
                    "worst_profile_id": worst["profile_id"],
                    "worst_modified_secant_tau225": worst[
                        "modified_secant_tau225_coordinate"
                    ],
                    "worst_elevation_deg": worst["elevation_deg"],
                    "max_symmetric_pl_pchip_fractional_correction_difference": f64(
                        max(disagreement)
                    ),
                    "passes_post_discovery_am12_2_stress_1pct": str(
                        bool(np.max(absolute) <= 0.01)
                    ).lower(),
                    "provisional_gate_scope": (
                        "numerical_stress_only_not_physical_photometric_accuracy"
                    ),
                    "operator_authorization": "none",
                }
            )
    return rows


def build_report(
    copied: dict[str, Any],
    raw_outputs: dict[str, Any],
    comparison_rows: list[dict[str, str]],
    annual_coefficient_rows: list[dict[str, str]],
    metric_rows: list[dict[str, str]],
    registry: dict[str, Any],
    deviation: dict[str, Any],
) -> bytes:
    eligible = [
        profile for profile in copied["profiles"].values() if profile["eligible"]
    ]
    excluded = [
        profile for profile in copied["profiles"].values() if not profile["eligible"]
    ]
    worst_comparison = max(
        comparison_rows,
        key=lambda row: float(
            row["max_abs_full_legacy_anchor_fractional_correction_error"]
        ),
    )
    lines = [
        "# SCI-CAL-001 copied AM follow-up report",
        "",
        "## Status",
        "",
        "This deterministic follow-up inventories the newly copied AM 12.2 AMC inputs and NPZ suite and performs a post-discovery, non-blinded numerical stress comparison. It does **not** establish identity with the historical Citlali atmosphere generator, authorize an operator, or declare an operational opacity/elevation domain.",
        "",
        "The analysis never reads the nearby Dataverse uploader logs. Those logs are excluded because they are not scientific model inputs and may contain credentials.",
        "",
        "## Copied products",
        "",
        f"The 25 AMC inputs total `{copied['amc_inputs']['total_bytes']}` bytes. Their frozen sha256sum-record aggregate is `{copied['amc_inputs']['canonical_sha256sum_records']['sha256']}` and their independent basename/NUL/raw-digest aggregate is `{copied['amc_inputs']['canonical_nul_records']['sha256']}`. Every staged input exactly matches its AM 12.2 cookbook copy; per-file byte counts and SHA-256 identities are machine-readable in the manifest and product inventory.",
        "",
        f"The frozen suite contains `{len(copied['profiles'])}` NPZ products totaling `{copied['total_bytes']}` bytes. Its canonical manifest SHA-256 is `{copied['manifest_sha256']}`. Every product has elevations 10--80 degrees in 2-degree steps and spectra from 0--500 GHz in 0.01-GHz steps, with direct `atmTaun` retained.",
        "",
        f"All 20 DJF/JJA/MAM/SON products exactly match their explicit TolTECA seasonal datafile ID and MD5 identities at registry commit `{registry['commit']}`. The five annual products have no matching generic registry identity. The generic `am_q25`, `am_q50`, `am_q75`, and `am_q95` registry artifacts are separate products and are not aliases for any copied seasonal or annual file.",
        "",
        "## Direct AM output validation",
        "",
        f"All `{raw_outputs['file_count']}` copied raw DAT outputs parse as `{EXPECTED_RAW_ROW_COUNT}` five-column numeric rows and identify `{EXPECTED_AM_IDENTITY}`. Their canonical aggregate SHA-256 is `{raw_outputs['manifest_sha256']}`, computed as {raw_outputs['manifest_definition']}.",
        "",
        "For every DAT file, zenith angle is mapped to NPZ elevation by `elevation = 90 deg - ZA`. Frequency, direct tau, transmission, and RJ temperature match the corresponding NPZ column exactly for every row. The fifth brightness-temperature column is present and finite in all files; the NPZ omits Tb, so no Tb equality comparison is claimed or invented.",
        "",
        "Every complete grid is followed by an unresolved-line warning and a Slurm exit-code-1 footer. The unresolved-line count distribution is `"
        + json.dumps(raw_outputs["warning_count_distribution"], sort_keys=True)
        + "`; `"
        + str(raw_outputs["slurm_retry_file_count"])
        + "` files also contain a Slurm step-creation retry notice. These historical nonzero return footers are retained as provenance and are not reclassified as clean successful runs.",
        "",
        f"The modified-secant T225-at-80 coordinate places `{len(eligible)}` profiles inside the exact legacy q0--q95 diagnostic support. `{len(excluded)}` profiles are excluded without extrapolation: "
        + ", ".join(
            f"`{profile['profile_id']}`"
            for profile in sorted(excluded, key=lambda item: item["profile_id"])
        )
        + ".",
        "",
        "## Legacy identity comparison",
        "",
        "Annual, DJF, MAM, JJA, and SON q25/q50/q75 products were each compared with the recovered same-percentile legacy grid over all 50,001 frequencies and 31 common elevations. None is content-identical. The table reports maximum and RMS transmission and Rayleigh-Jeans differences without assigning a best or closest family. The expected historical q95 MD5 remains `0ca7b331823237767d26016d19bffb3d`; no q95 common-grid comparison is invented and none of the copied products is substituted for it.",
        "",
        f"`{ANNUAL_COEFFICIENT_NAME}` records all {len(annual_coefficient_rows)} annual q25/q50/q75/q95 degree-six coefficient values (four profiles by three bands by seven descending powers), including unrounded binary64 values, explicit eight-decimal values, and comparison with the repair-base literals. This is a deterministic copied-family fit diagnostic, not generic-q identity evidence.",
        "",
        "Across all same-percentile copied families, the largest exact repair-base anchor correction difference is `"
        + f"{100.0 * float(worst_comparison['max_abs_full_legacy_anchor_fractional_correction_error']):.6f}%`"
        + " for `"
        + worst_comparison["copied_profile_id"]
        + "/"
        + worst_comparison["band"]
        + "`. This is an identity diagnostic, not an interpolation result.",
        "",
        f"Protocol identity is resolved by `{deviation['filename']}` (SHA-256 `{deviation['sha256']}`). Annual-anchor Study C `v1` was stopped; diagnostic C1 evaluates only the already defined legacy-anchor `v0` candidates. The clarification does not authorize or reinterpret a successor.",
        "",
        "## In-support operator stress",
        "",
        "Truth is the copied direct line-of-sight `atmTaun` at monochromatic 272.73, 214.29, and 150 GHz. The interpolation coordinate is zenith tau225 derived from copied T225 at 80 degrees using the repair-base modified secant. Both candidates use the fixed linear q0--q25 LOS-tau segment; above q25 they use either piecewise linear or PCHIP interpolation through the exact repair-base fitted surfaces.",
        "",
        "| Candidate | Band | Maximum correction error | P95 correction error | Worst profile/elevation | PL--PCHIP maximum difference | Provisional 1% stress result |",
        "| --- | --- | ---: | ---: | --- | ---: | --- |",
    ]
    for row in metric_rows:
        lines.append(
            f"| `{row['candidate']}` | `{row['band']}` | "
            f"`{100.0 * float(row['max_abs_fractional_correction_error']):.6f}%` | "
            f"`{100.0 * float(row['p95_abs_fractional_correction_error']):.6f}%` | "
            f"`{row['worst_profile_id']}` / `{float(row['worst_elevation_deg']):.1f} deg` | "
            f"`{100.0 * float(row['max_symmetric_pl_pchip_fractional_correction_difference']):.6f}%` | "
            f"`{row['passes_post_discovery_am12_2_stress_1pct']}` |"
        )
    lines.extend(
        [
            "",
            "These results are useful provisional representation stress evidence only. The profiles and candidates were inspected before this analysis, the convention is monochromatic rather than band integrated, and the copied products are distinct from the generic-q products while their historical generic-generator association is not established. C1 spans the legacy q0--q95 diagnostic range and is not the selected q95-excluding AM 12.2 successor study. Passing one percent here is not per-sample physical photometric accuracy and does not address the separate 5--10% absolute or approximately 5% repeatability observational gates.",
            "",
            "## Disposition",
            "",
            "The owner has selected evaluation of a separately versioned AM 12.2 successor, with q95 conditions retained as historical/diagnostic evidence only. Retain piecewise-linear LOS tau as the baseline and PCHIP as the challenger for that declared study. Do not authorize either candidate or an operational domain from this follow-up. The successor profile rule, spectral convention, preregistered independent runs, exact domain endpoints, warning policy, and the SCI-ALIGN-001 sample-identity eligibility dependency remain separate gates.",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def build_manifest(
    repo_root: Path,
    am_root: Path,
    legacy_source_dir: Path,
    tolteca_root: Path,
    copied: dict[str, Any],
    raw_outputs: dict[str, Any],
    registry: dict[str, Any],
    comparison_rows: list[dict[str, str]],
    annual_coefficient_rows: list[dict[str, str]],
    stress_rows: list[dict[str, str]],
    metric_rows: list[dict[str, str]],
    artifact_bytes: dict[str, bytes],
    deviation: dict[str, Any],
) -> bytes:
    profiles = copied["profiles"]
    manifest = {
        "schema_version": "sci-cal-001-copied-am-followup-manifest-v2",
        "identity": {
            "package": "SCI-CAL-001",
            "repair_base_sha": REPAIR_BASE_SHA,
            "repair_line_evidence_head": REPAIR_LINE_EVIDENCE_HEAD,
            "evidence_status": "post_discovery_non_blinded_stress",
            "copied_suite_identity": "am_12_2_distinct_registered_product_family",
            "historical_generic_generator_association": "not_established",
            "owner_direction": "versioned_am12_successor_evaluation_only",
            "adoption_status": "evaluation_only_not_adopted",
            "q95_operational_disposition": "historical_diagnostic_only",
            "successor_study_status": "pending_results",
            "study_artifact_binding_status": "unbound_pending_study_results",
            "operator_authorization": "none",
            "operational_domain_authorization": "none",
        },
        "input_roots": {
            "repo_root": str(repo_root),
            "copied_am_root": str(am_root),
            "legacy_source_dir": str(legacy_source_dir),
            "tolteca_root": str(tolteca_root),
        },
        "protocol_deviation": deviation,
        "amc_inputs": copied["amc_inputs"],
        "copied_suite": {
            "product_count": len(profiles),
            "total_bytes": copied["total_bytes"],
            "canonical_manifest_definition": copied["manifest_definition"],
            "canonical_manifest_sha256": copied["manifest_sha256"],
            "products": [
                {
                    "filename": profile["filename"],
                    "bytes": profile["size"],
                    "sha256": profile["sha256"],
                    "md5": profile["md5"],
                    "amc_input": {
                        "filename": profile["amc_input"]["filename"],
                        "bytes": profile["amc_input"]["bytes"],
                        "sha256": profile["amc_input"]["sha256"],
                        "release_copy_exact": profile["amc_input"][
                            "release_copy_exact"
                        ],
                    },
                    "tolteca_registry_identity": (
                        {
                            "name": profile["registry_name"],
                            "datafile_id": profile["registry_id"],
                            "md5": profile["registry_md5"],
                            "status": profile["registry_status"],
                        }
                        if profile["registry_name"]
                        else {
                            "name": None,
                            "datafile_id": None,
                            "md5": None,
                            "status": profile["registry_status"],
                        }
                    ),
                    "generic_q_registry_relation": profile["generic_registry_relation"],
                    "modified_secant_tau225_coordinate": f64(profile["tau225"]),
                    "legacy_support": profile["support"],
                }
                for profile in sorted(
                    profiles.values(), key=lambda item: item["filename"]
                )
            ],
        },
        "tolteca_registry_provenance": {
            "commit": registry["commit"],
            "object": registry["object"],
            "object_sha256": registry["object_sha256"],
            "seasonal_identity_count": len(registry["seasonal"]),
            "seasonal_identities": [
                {"name": name, **identity}
                for name, identity in sorted(registry["seasonal"].items())
            ],
            "seasonal_verification": (
                "all_20_copied_seasonal_md5_values_exactly_match_registry"
            ),
            "annual_product_identity": "no_matching_generic_registry_identity",
            "generic_q_artifacts": [
                {
                    "name": name,
                    **identity,
                    "relation_to_copied_suite": "separate_registry_artifact",
                }
                for name, identity in sorted(registry["generic"].items())
            ],
        },
        "copied_raw_outputs": {
            "file_count": raw_outputs["file_count"],
            "total_bytes": raw_outputs["total_bytes"],
            "canonical_manifest_algorithm": raw_outputs["manifest_definition"],
            "canonical_manifest_sha256": raw_outputs["manifest_sha256"],
            "expected_rows_per_file": EXPECTED_RAW_ROW_COUNT,
            "am_identity": EXPECTED_AM_IDENTITY,
            "za_to_elevation_mapping": "elevation_deg=90-zenith_angle_deg",
            "retained_npz_fields": ["f", "tau", "tx", "Trj"],
            "all_four_retained_fields_exact_npz": raw_outputs[
                "all_four_retained_fields_exact_npz"
            ],
            "tb_status": {
                "present_and_finite_in_all_dat_files": raw_outputs[
                    "all_tb_present_and_finite"
                ],
                "npz_comparison": "not_applicable_npz_omits_tb",
            },
            "unresolved_line_warning_count_distribution": raw_outputs[
                "warning_count_distribution"
            ],
            "slurm_retry_file_count": raw_outputs["slurm_retry_file_count"],
            "return_footer_status_distribution": raw_outputs[
                "return_footer_status_distribution"
            ],
            "historical_return_disposition": (
                "complete_numeric_outputs_with_nonzero_footer_not_clean_run_proof"
            ),
        },
        "legacy_inputs": {
            "repair_files": [
                {
                    "path": str(relative),
                    "sha256": digest,
                }
                for relative, digest in sorted(
                    FROZEN_REPAIR_INPUTS.items(), key=lambda item: str(item[0])
                )
            ],
            "raw_q25_q50_q75": [
                {"model": f"am_q{percentile}", **metadata}
                for percentile, metadata in sorted(EXPECTED_LEGACY.items())
            ],
            "q95_expected_md5_not_present_not_substituted": EXPECTED_Q95_MD5,
        },
        "scientific_conventions": {
            "truth": "copied_direct_atmTaun_line_of_sight_neper",
            "opacity_coordinate": (
                "-log(copied_T225_at_elevation_80deg)/repair_base_modified_secant_airmass_80deg"
            ),
            "spectral_convention": {
                "kind": "monochromatic_legacy_parity",
                "reference_ghz": f64(REFERENCE_FREQUENCY_GHZ),
                "band_frequencies_ghz": {
                    band: f64(frequency)
                    for band, frequency in BAND_FREQUENCIES_GHZ.items()
                },
                "bandpass_integration": "not_performed",
            },
            "stress_elevation_grid_deg": "20:80:2",
            "interpolation_space": "line_of_sight_optical_depth",
            "outside_support_policy": "exclude_without_extrapolation",
            "fractional_correction_error": (
                "expm1(candidate_los_tau-copied_direct_los_tau)"
            ),
        },
        "candidates": {
            "piecewise_linear_los_tau_v0": (
                "piecewise affine through exact legacy LOS-tau anchors"
            ),
            "pchip_los_tau_v0": (
                "fixed linear q0-q25 segment then PCHIP through q25-q95 anchors"
            ),
        },
        "stress_scope": {
            "eligible_profile_count": sum(
                bool(profile["eligible"]) for profile in profiles.values()
            ),
            "excluded_profile_count": sum(
                not bool(profile["eligible"]) for profile in profiles.values()
            ),
            "excluded_profiles": [
                profile["profile_id"]
                for profile in sorted(
                    profiles.values(), key=lambda item: item["profile_id"]
                )
                if not profile["eligible"]
            ],
            "stress_row_count": len(stress_rows),
            "metric_row_count": len(metric_rows),
            "comparison_row_count": len(comparison_rows),
            "comparison_cartesian_contract": {
                "profile_families": list(PROFILE_FAMILIES),
                "percentiles": [25, 50, 75, 95],
                "bands": list(BANDS),
                "family_ranking_claim": "none_metric_values_only",
            },
            "annual_fit_coefficient_row_count": len(annual_coefficient_rows),
            "annual_fit_coefficient_contract": (
                "four annual percentiles by three bands by seven descending powers; "
                "numpy.polyfit degree six over elevation 20:80:2 in repair-base radians; "
                "explicit eight-decimal rounding"
            ),
            "provisional_threshold_fraction": f64(0.01),
            "qualification": (
                "post_discovery_non_blinded_numerical_stress_not_photometric_accuracy"
            ),
        },
        "excluded_inputs": [
            {
                "pattern": "LMT_am_npz/DVUploaderLog*.log",
                "read_by_generator": False,
                "reason": (
                    "not_scientific_input_may_contain_credentials_and_not_upload_proof"
                ),
            }
        ],
        "numeric_environment": {
            "python": ".".join(str(value) for value in sys.version_info[:3]),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "float_radix": sys.float_info.radix,
            "float_mantissa_bits": sys.float_info.mant_dig,
        },
        "generated_artifacts": [
            {
                "filename": filename,
                "bytes": len(content),
                "sha256": sha256_bytes(content),
            }
            for filename, content in sorted(artifact_bytes.items())
        ],
        "limitations": [
            "The copied AM 12.2 products are distinct from the generic-q products; historical generic-generator association is not established.",
            "The stress was designed after the copied products and candidates were inspected.",
            "No profile outside the legacy q0-q95 tau225 support is evaluated.",
            "The test is monochromatic and does not select a band-integrated convention.",
            "A provisional one-percent numerical stress result is not per-sample physical photometric accuracy.",
            "No operator or operational opacity/elevation domain is authorized.",
        ],
    }
    return render_json(manifest)


def expected_artifacts(
    repo_root: Path,
    am_root: Path,
    legacy_source_dir: Path,
    tolteca_root: Path,
) -> dict[str, bytes]:
    validate_legacy_sources(legacy_source_dir)
    registry = load_tolteca_registry(tolteca_root)
    legacy = build_legacy_state(repo_root)
    deviation = validate_deviation_log(repo_root)
    copied = load_copied_suite(am_root, legacy, registry)
    raw_outputs = build_raw_output_inventory(am_root, copied)
    comparison_rows = build_all_family_legacy_comparison_rows(
        copied, legacy, legacy_source_dir
    )
    annual_coefficient_rows = build_annual_fit_coefficient_rows(copied, legacy)
    stress_rows = build_stress_rows(copied, legacy)
    metric_rows = build_stress_metrics(stress_rows, copied)
    artifacts = {
        INVENTORY_NAME: render_csv(copied["inventory_rows"]),
        RAW_OUTPUT_INVENTORY_NAME: render_csv(raw_outputs["rows"]),
        COMPARISON_NAME: render_csv(comparison_rows),
        ANNUAL_COEFFICIENT_NAME: render_csv(annual_coefficient_rows),
        STRESS_ROWS_NAME: render_csv(stress_rows),
        STRESS_METRICS_NAME: render_csv(metric_rows),
        REPORT_NAME: build_report(
            copied,
            raw_outputs,
            comparison_rows,
            annual_coefficient_rows,
            metric_rows,
            registry,
            deviation,
        ),
    }
    artifacts[MANIFEST_NAME] = build_manifest(
        repo_root,
        am_root,
        legacy_source_dir,
        tolteca_root,
        copied,
        raw_outputs,
        registry,
        comparison_rows,
        annual_coefficient_rows,
        stress_rows,
        metric_rows,
        artifacts,
        deviation,
    )
    return artifacts


def main() -> int:
    script_path = Path(__file__).resolve()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=script_path.parents[2],
        help="Citlali repair-line repository root",
    )
    parser.add_argument(
        "--am-root",
        type=Path,
        default=DEFAULT_AM_ROOT,
        help="read-only copied AM Big_Atmosphere root",
    )
    parser.add_argument(
        "--legacy-source-dir",
        type=Path,
        default=DEFAULT_LEGACY_SOURCE_DIR,
        help="read-only directory containing legacy q25/q50/q75 NPZ files",
    )
    parser.add_argument(
        "--tolteca-root",
        type=Path,
        default=DEFAULT_TOLTECA_ROOT,
        help="read-only TolTECA repository containing the frozen registry commit",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_path.parent,
        help="directory for generated follow-up artifacts",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify generated artifacts instead of rewriting them",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    am_root = args.am_root.resolve()
    legacy_source_dir = args.legacy_source_dir.resolve()
    tolteca_root = args.tolteca_root.resolve()
    output_dir = args.output_dir.resolve()
    artifacts = expected_artifacts(repo_root, am_root, legacy_source_dir, tolteca_root)
    if not args.check:
        output_dir.mkdir(parents=True, exist_ok=True)
    failed = False
    for filename, expected in artifacts.items():
        path = output_dir / filename
        if args.check:
            if not path.exists() or path.read_bytes() != expected:
                print(f"stale or missing copied-AM artifact: {path}", file=sys.stderr)
                failed = True
        else:
            path.write_bytes(expected)
            print(f"wrote {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
