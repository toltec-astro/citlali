import base64
import contextlib
import copy
import hashlib
import io
import json
import math
import struct
import subprocess
import tempfile
import unittest
import zlib
from pathlib import Path

from tools.baseline import validate_product_contract as product_contract


class ValidateProductContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.reduction = self.root / "redu00"
        self.reduction.mkdir()

    def tearDown(self) -> None:
        self.directory.cleanup()

    def family(self) -> dict[str, object]:
        return {
            "scientific_identity": "test product",
            "coordinate_frame": "not_applicable",
            "axes": [],
            "units_policy": "dimensionless",
            "indexing_policy": "not_applicable",
            "missing_value_policy": "none",
            "failure_policy": "fatal when required",
        }

    def registry(self, entries: list[dict[str, object]]) -> dict[str, object]:
        return {
            "schema_version": product_contract.SCHEMA_VERSION,
            "families": {"test": self.family()},
            "contracts": [
                {
                    "contract_id": "contract",
                    "profile_id": "profile",
                    "mode": "point",
                    "arrays": ["a1100", "a1400", "a2000"],
                    "minimum_observations": 0,
                    "entries": entries,
                }
            ],
        }

    def entry(self, **overrides: object) -> dict[str, object]:
        value: dict[str, object] = {
            "entry_id": "test-entry",
            "family_id": "test",
            "scope": "reduction",
            "classification": "required",
            "condition": "always",
            "pattern": "product.csv",
            "checks": {"nonempty": True},
        }
        value.update(overrides)
        return value

    def write_registry(self, value: dict[str, object]) -> Path:
        path = self.root / "contracts.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def validate(
        self,
        entries: list[dict[str, object]],
        config: dict[str, object] | None = None,
    ) -> dict[str, object]:
        registry = self.registry(entries)
        path = self.write_registry(registry)
        loaded = product_contract.load_registry(path)
        contract = product_contract.contract_by_id(loaded, "contract")
        return product_contract.validate_reduction(
            loaded, contract, self.reduction, config or {}
        )

    def test_accepts_classified_required_product(self) -> None:
        (self.reduction / "product.csv").write_text("value\n1\n", encoding="utf-8")

        result = self.validate([self.entry()])

        self.assertTrue(result["passed"])
        self.assertEqual(result["classified_product_count"], 1)
        self.assertFalse(result["errors"])

    def test_rejects_missing_required_product(self) -> None:
        result = self.validate([self.entry()])

        self.assertFalse(result["passed"])
        self.assertTrue(any("requires at least 1" in error for error in result["errors"]))

    def test_allows_absent_optional_product(self) -> None:
        result = self.validate(
            [
                self.entry(
                    classification="optional_diagnostic",
                )
            ]
        )

        self.assertTrue(result["passed"])

    def test_requires_configured_product_when_enabled(self) -> None:
        entry = self.entry(
            classification="config_conditional",
            condition="feature.enabled=true",
            required_when={"path": "feature.enabled", "equals": True},
        )

        result = self.validate([entry], {"feature": {"enabled": True}})

        self.assertFalse(result["passed"])
        self.assertTrue(
            any("requires at least 1" in error for error in result["errors"])
        )

    def test_rejects_configured_product_when_disabled(self) -> None:
        (self.reduction / "product.csv").write_text(
            "value\n1\n", encoding="utf-8"
        )
        entry = self.entry(
            classification="config_conditional",
            condition="feature.enabled=true",
            required_when={"path": "feature.enabled", "equals": True},
        )

        result = self.validate([entry], {"feature": {"enabled": False}})

        self.assertFalse(result["passed"])
        self.assertTrue(
            any("allows at most 0" in error for error in result["errors"])
        )

    def test_rejects_conditional_entry_without_machine_rule(self) -> None:
        registry = self.registry(
            [
                self.entry(
                    classification="config_conditional",
                    condition="feature.enabled=true",
                )
            ]
        )

        with self.assertRaisesRegex(product_contract.ContractError, "required_when"):
            product_contract.load_registry(self.write_registry(registry))

    def test_rejects_missing_config_condition_path(self) -> None:
        entry = self.entry(
            classification="config_conditional",
            condition="feature.enabled=true",
            required_when={"path": "feature.enabled", "equals": True},
        )

        with self.assertRaisesRegex(product_contract.ContractError, "does not contain"):
            self.validate([entry], {})

    def test_rejects_unclassified_product(self) -> None:
        (self.reduction / "product.csv").write_text("value\n1\n", encoding="utf-8")
        (self.reduction / "extra.nc").write_bytes(b"not netcdf")

        result = self.validate([self.entry()])

        self.assertFalse(result["passed"])
        self.assertEqual(result["unclassified_products"], ["extra.nc"])

    def test_rejects_incomplete_family_semantics(self) -> None:
        registry = self.registry([self.entry()])
        del registry["families"]["test"]["units_policy"]  # type: ignore[index]
        path = self.write_registry(registry)

        with self.assertRaisesRegex(product_contract.ContractError, "units_policy"):
            product_contract.load_registry(path)

    def test_materializes_successor_contract_without_mutating_predecessor(self) -> None:
        registry = self.registry([self.entry()])
        registry["contracts"].append(  # type: ignore[union-attr]
            {
                "contract_id": "successor",
                "profile_id": "successor-profile",
                "extends_contract_id": "contract",
                "entry_overrides": {
                    "test-entry": {"pattern": "successor.csv"}
                },
            }
        )

        loaded = product_contract.load_registry(self.write_registry(registry))
        predecessor = product_contract.contract_by_id(loaded, "contract")
        successor = product_contract.contract_by_id(loaded, "successor")

        self.assertEqual(predecessor["entries"][0]["pattern"], "product.csv")
        self.assertEqual(successor["entries"][0]["pattern"], "successor.csv")
        self.assertEqual(successor["mode"], predecessor["mode"])

    def test_science_map_schema_rejects_alias_semantic_drift(self) -> None:
        checked = (
            Path(__file__).resolve().parents[2]
            / "validation/product_contracts.json"
        )
        registry = copy.deepcopy(json.loads(checked.read_text(encoding="utf-8")))
        registry["science_map_contracts"]["sci-map-001-f010-v1"]["aliases"][
            "coverage_bool_I"
        ]["validity_authority"] = True

        with self.assertRaisesRegex(product_contract.ContractError, "does not freeze"):
            product_contract.load_registry(self.write_registry(registry))

    def test_science_map_schema_rejects_alias_type_drift(self) -> None:
        checked = (
            Path(__file__).resolve().parents[2]
            / "validation/product_contracts.json"
        )
        registry = copy.deepcopy(json.loads(checked.read_text(encoding="utf-8")))
        registry["science_map_contracts"]["sci-map-001-f010-v1"]["aliases"][
            "coverage_I"
        ]["unit"] = "s"

        with self.assertRaisesRegex(product_contract.ContractError, "does not freeze"):
            product_contract.load_registry(self.write_registry(registry))

    def test_science_map_schema_requires_nonarray_absence_and_parallel_policy(
        self,
    ) -> None:
        checked = (
            Path(__file__).resolve().parents[2]
            / "validation/product_contracts.json"
        )
        registry = copy.deepcopy(json.loads(checked.read_text(encoding="utf-8")))
        science = registry["science_map_contracts"]["sci-map-001-f010-v1"]
        del science["non_array_grouping_absence_policy"]

        with self.assertRaisesRegex(
            product_contract.ContractError,
            "non_array_grouping_absence_policy",
        ):
            product_contract.load_registry(self.write_registry(registry))

        registry = copy.deepcopy(json.loads(checked.read_text(encoding="utf-8")))
        registry["science_map_contracts"]["sci-map-001-f010-v1"][
            "parallel_equivalence_policy"
        ]["identity"] = "unbounded"
        with self.assertRaisesRegex(
            product_contract.ContractError,
            "parallel_equivalence_policy.identity",
        ):
            product_contract.load_registry(self.write_registry(registry))

    def test_jinc_contract_freezes_signed_conditioning_and_product_truth(self) -> None:
        checked = (
            Path(__file__).resolve().parents[2]
            / "validation/product_contracts.json"
        )
        registry = product_contract.load_registry(checked)
        jinc = registry["jinc_map_contracts"]["sci-map-002-jinc-v1"]

        self.assertEqual(
            jinc["estimator"], "signed-N-over-C-formal-C2-over-Q-v1"
        )
        self.assertIn("no radial predicate", jinc["support_geometry"])
        self.assertIn("point sampling", jinc["phase_policy"])
        self.assertEqual(
            jinc["conditioning_policy"]["summation_method"],
            "naive-binary64-two-level-2gamma-n-sumabs-v1",
        )
        self.assertIn(
            "authoritative formal-support",
            jinc["products"]["coverage_bool_I"],
        )
        self.assertIn("in seconds", jinc["products"]["coverage_I"])
        self.assertIn("K/C", jinc["products"]["kernel_I"])
        self.assertEqual(
            jinc["preserved_contracts"], ["SCI-MAP-001", "SCI-NOI-002"]
        )

    @unittest.skipIf(product_contract.fits is None, "astropy is unavailable")
    def test_checks_fits_structure_without_reading_pixels(self) -> None:
        from astropy.io import fits

        header = fits.Header()
        header["BUNIT"] = "mJy/beam"
        header["CTYPE1"] = "AZOFFSET"
        header["CTYPE2"] = "ELOFFSET"
        header["CUNIT1"] = "arcsec"
        header["CUNIT2"] = "arcsec"
        hdus = fits.HDUList(
            [
                fits.PrimaryHDU(header=fits.Header({"BUNIT": "mJy/beam"})),
                fits.ImageHDU(data=[[1.0]], header=header, name="signal_I"),
            ]
        )
        hdus.writeto(self.reduction / "product.fits")
        result = self.validate(
            [
                self.entry(
                    pattern="product.fits",
                    checks={
                        "min_hdus": 2,
                        "required_extnames": ["signal_I"],
                        "primary_bunit": "mJy/beam",
                        "axis_types": ["AZOFFSET", "ELOFFSET"],
                        "axis_units": ["arcsec", "arcsec"],
                    },
                )
            ]
        )

        self.assertTrue(result["passed"], result["errors"])

    @unittest.skipIf(product_contract.fits is None, "astropy is unavailable")
    def test_detector_contract_rejects_new_significance_product_family(
        self,
    ) -> None:
        import numpy as np
        from astropy.io import fits

        checked = (
            Path(__file__).resolve().parents[2]
            / "validation/product_contracts.json"
        )
        registry = product_contract.load_registry(checked)
        checks = registry["checks"]["sci_map_detector_group_v1"]
        header = fits.Header(
            {
                "CTYPE1": "AZOFFSET",
                "CTYPE2": "ELOFFSET",
                "CTYPE3": "FREQ",
                "CTYPE4": "STOKES",
                "CUNIT1": "arcsec",
                "CUNIT2": "arcsec",
                "CUNIT3": "Hz",
                "CUNIT4": "",
            }
        )
        weight_header = header.copy()
        weight_header.update(
            {
                "ESTTYPE": "nonprecision_normalization_coefficient",
                "TYPE": "nonprecision_normalization_coefficient",
                "PRECSTAT": "not_established",
                "COVSTAT": "unavailable",
                "CALTYPE": "formal",
            }
        )
        data = np.ones((1, 1, 1, 1), dtype=np.float64)
        fits.HDUList(
            [
                fits.PrimaryHDU(
                    header=fits.Header({"BUNIT": "mJy/beam"})
                ),
                fits.ImageHDU(
                    data=data, header=header, name="signal_det_0_I"
                ),
                fits.ImageHDU(
                    data=data, header=weight_header, name="weight_det_0_I"
                ),
                fits.ImageHDU(
                    data=data, header=header, name="kernel_det_0_I"
                ),
                fits.ImageHDU(
                    data=data,
                    header=header,
                    name="formal_standardized_signal_det_0_I",
                ),
            ]
        ).writeto(self.reduction / "detector.fits")

        errors = product_contract.validate_fits(
            self.reduction / "detector.fits", checks
        )

        self.assertTrue(
            any(
                "forbidden FITS extension prefixes" in error
                and "formal_standardized_signal_" in error
                for error in errors
            ),
            errors,
        )

    @unittest.skipIf(product_contract.fits is None, "astropy is unavailable")
    def test_checks_typed_masks_aliases_headers_shape_and_wcs(self) -> None:
        import numpy as np
        from astropy.io import fits

        def header(estimator: str) -> fits.Header:
            return fits.Header(
                {
                    "BUNIT": "1",
                    "ESTTYPE": estimator,
                    "CTYPE1": "AZOFFSET",
                    "CTYPE2": "ELOFFSET",
                    "CUNIT1": "arcsec",
                    "CUNIT2": "arcsec",
                    "CRPIX1": 1.0,
                    "CRPIX2": 1.0,
                    "CRVAL1": 0.0,
                    "CRVAL2": 0.0,
                    "CDELT1": -1.0,
                    "CDELT2": 1.0,
                    "EQUINOX": 2000.0,
                }
            )

        retained = np.array([[1.25, 0.0]], dtype=np.float64)
        policy = np.array([[1, 0]], dtype=np.uint8)
        hdus = fits.HDUList(
            [
                fits.PrimaryHDU(),
                fits.ImageHDU(
                    data=np.array([[2, 0]], dtype=np.int64),
                    header=header("count"),
                    name="geometric_hits_I",
                ),
                fits.ImageHDU(
                    data=retained,
                    header=header("retained_exposure"),
                    name="retained_exposure_I",
                ),
                fits.ImageHDU(
                    data=retained.copy(),
                    header=header("retained_exposure_alias"),
                    name="coverage_I",
                ),
                fits.ImageHDU(
                    data=policy,
                    header=header("science_policy_support"),
                    name="science_policy_support_I",
                ),
                fits.ImageHDU(
                    data=policy.copy(),
                    header=header("deprecated_policy_alias"),
                    name="coverage_bool_I",
                ),
                fits.ImageHDU(
                    data=policy.copy(),
                    header=header("science_validity"),
                    name="science_valid_I",
                ),
            ]
        )
        hdus.writeto(self.reduction / "product.fits")
        names = [
            "geometric_hits_I",
            "retained_exposure_I",
            "coverage_I",
            "science_policy_support_I",
            "coverage_bool_I",
            "science_valid_I",
        ]
        result = self.validate(
            [
                self.entry(
                    pattern="product.fits",
                    checks={
                        "required_extnames": names,
                        "required_ext_bitpix": {
                            "geometric_hits_I": 64,
                            "retained_exposure_I": -64,
                            "science_policy_support_I": 8,
                            "science_valid_I": 8,
                        },
                        "required_ext_dtypes": {
                            "geometric_hits_I": "int64",
                            "retained_exposure_I": "float64",
                            "science_policy_support_I": "uint8",
                            "science_valid_I": "uint8",
                        },
                        "binary_extnames": [
                            "science_policy_support_I",
                            "coverage_bool_I",
                            "science_valid_I",
                        ],
                        "exact_aliases": {
                            "retained_exposure_I": "coverage_I",
                            "science_policy_support_I": "coverage_bool_I",
                        },
                        "required_ext_headers": {
                            "science_valid_I": {
                                "BUNIT": "1",
                                "ESTTYPE": "science_validity",
                            }
                        },
                        "same_shape_extnames": names,
                        "same_wcs_extnames": names,
                    },
                )
            ]
        )

        self.assertTrue(result["passed"], result["errors"])

    @unittest.skipIf(product_contract.fits is None, "astropy is unavailable")
    def test_checks_filtered_raw_parent_header_carriage(self) -> None:
        import numpy as np
        from astropy.io import fits

        digest = "sha256:" + "0123456789abcdef" * 4

        def raw_header() -> fits.Header:
            return fits.Header(
                {"RAWSTATE": "immutable_input", "RAWPDGST": digest}
            )

        fits.HDUList(
            [
                fits.PrimaryHDU(),
                fits.ImageHDU(
                    data=np.array([[1.0]], dtype=np.float64),
                    header=raw_header(),
                    name="signal_I",
                ),
                fits.ImageHDU(
                    data=np.array([[2.0]], dtype=np.float64),
                    header=raw_header(),
                    name="weight_I",
                ),
                fits.ImageHDU(
                    data=np.array([[1]], dtype=np.uint8),
                    header=raw_header(),
                    name="science_valid_I",
                ),
            ]
        ).writeto(self.reduction / "product.fits")
        checks = {
            "required_extnames": [
                "signal_I",
                "weight_I",
                "science_valid_I",
            ],
            "required_ext_headers": {
                name: {"RAWSTATE": "immutable_input"}
                for name in ["signal_I", "weight_I", "science_valid_I"]
            },
            "required_ext_headers_present": {
                name: ["RAWPDGST"]
                for name in ["signal_I", "weight_I", "science_valid_I"]
            },
            "same_ext_header_values": {
                "RAWPDGST": ["signal_I", "weight_I", "science_valid_I"]
            },
        }

        result = self.validate(
            [self.entry(pattern="product.fits", checks=checks)]
        )

        self.assertTrue(result["passed"], result["errors"])

        with fits.open(self.reduction / "product.fits", mode="update") as hdus:
            hdus["weight_I"].header["RAWPDGST"] = "sha256:drifted"
        result = self.validate(
            [self.entry(pattern="product.fits", checks=checks)]
        )
        self.assertFalse(result["passed"])
        self.assertTrue(
            any(
                "cross-HDU RAWPDGST values differ" in error
                for error in result["errors"]
            )
        )

        with fits.open(self.reduction / "product.fits", mode="update") as hdus:
            del hdus["science_valid_I"].header["RAWPDGST"]
        result = self.validate(
            [self.entry(pattern="product.fits", checks=checks)]
        )
        self.assertFalse(result["passed"])
        self.assertTrue(
            any(
                "missing required FITS headers" in error
                for error in result["errors"]
            )
        )

    @unittest.skipIf(product_contract.fits is None, "astropy is unavailable")
    def test_rejects_nonbinary_alias_and_cross_hdu_wcs_drift(self) -> None:
        import numpy as np
        from astropy.io import fits

        canonical_header = fits.Header(
            {"CTYPE1": "AZOFFSET", "CRPIX1": 1.0, "CDELT1": 1.0}
        )
        drifted_header = canonical_header.copy()
        drifted_header["CRPIX1"] = 2.0
        fits.HDUList(
            [
                fits.PrimaryHDU(),
                fits.ImageHDU(
                    data=np.array([[1]], dtype=np.uint8),
                    header=canonical_header,
                    name="science_policy_support_I",
                ),
                fits.ImageHDU(
                    data=np.array([[2]], dtype=np.uint8),
                    header=drifted_header,
                    name="coverage_bool_I",
                ),
            ]
        ).writeto(self.reduction / "product.fits")
        result = self.validate(
            [
                self.entry(
                    pattern="product.fits",
                    checks={
                        "required_extnames": [
                            "science_policy_support_I",
                            "coverage_bool_I",
                        ],
                        "binary_extnames": ["coverage_bool_I"],
                        "exact_aliases": {
                            "science_policy_support_I": "coverage_bool_I"
                        },
                        "same_wcs_extnames": [
                            "science_policy_support_I",
                            "coverage_bool_I",
                        ],
                    },
                )
            ]
        )

        self.assertFalse(result["passed"])
        joined = "\n".join(result["errors"])
        self.assertIn("not a binary 0/1 mask", joined)
        self.assertIn("not bitwise equal", joined)
        self.assertIn("cross-HDU WCS cards differ", joined)

    @unittest.skipIf(product_contract.fits is None, "astropy is unavailable")
    def test_rejects_missing_cross_hdu_wcs_inventory(self) -> None:
        from astropy.io import fits

        fits.HDUList(
            [
                fits.PrimaryHDU(),
                fits.ImageHDU(data=[[1.0]], name="signal_I"),
                fits.ImageHDU(data=[[1.0]], name="weight_I"),
            ]
        ).writeto(self.reduction / "product.fits")
        result = self.validate(
            [
                self.entry(
                    pattern="product.fits",
                    checks={
                        "required_extnames": ["signal_I", "weight_I"],
                        "same_wcs_extnames": ["signal_I", "weight_I"],
                    },
                )
            ]
        )

        self.assertFalse(result["passed"])
        self.assertTrue(
            any("no WCS-card inventory" in error for error in result["errors"])
        )

    @unittest.skipIf(product_contract.fits is None, "astropy is unavailable")
    def test_selects_method_specific_fits_checks_from_config(self) -> None:
        from astropy.io import fits

        fits.HDUList(
            [fits.PrimaryHDU(), fits.ImageHDU(data=[[1.0]], name="signal_I")]
        ).writeto(self.reduction / "product.fits")
        entry = self.entry(
            pattern="product.fits",
            checks_by_config=[
                {
                    "when": {"path": "mapmaking.method", "equals": "naive"},
                    "checks": {"required_extnames": ["science_valid_I"]},
                },
                {
                    "when": {"path": "mapmaking.method", "equals": "jinc"},
                    "checks": {"forbidden_extnames": ["science_valid_I"]},
                },
            ],
            require_matching_config_check=True,
        )

        naive = self.validate([entry], {"mapmaking": {"method": "naive"}})
        jinc = self.validate([entry], {"mapmaking": {"method": "jinc"}})

        self.assertFalse(naive["passed"])
        self.assertTrue(jinc["passed"], jinc["errors"])

    @unittest.skipIf(product_contract.netCDF4 is None, "netCDF4 is unavailable")
    def test_checks_netcdf_identity_dimensions_and_variables(self) -> None:
        import netCDF4

        path = self.reduction / "product.nc"
        with netCDF4.Dataset(path, "w") as dataset:
            dim = dataset.createDimension("n_dets", 2)
            dataset.createVariable("signal", "f8", (dim.name,))
            schema_dim = dataset.createDimension("schema", 1)
            schema = dataset.createVariable(
                "schema_version", str, (schema_dim.name,)
            )
            schema[0] = "test-schema-v1"
        result = self.validate(
            [
                self.entry(
                    pattern="product.nc",
                    checks={
                        "required_dimensions": ["n_dets"],
                        "positive_dimensions": ["n_dets"],
                        "required_variables": ["signal", "schema_version"],
                        "scalar_equals": {
                            "schema_version": "test-schema-v1",
                        },
                    },
                )
            ]
        )

        self.assertTrue(result["passed"], result["errors"])

    @unittest.skipIf(product_contract.netCDF4 is None, "netCDF4 is unavailable")
    def test_rejects_wrong_netcdf_scalar_identity(self) -> None:
        import netCDF4

        path = self.reduction / "product.nc"
        with netCDF4.Dataset(path, "w") as dataset:
            dim = dataset.createDimension("schema", 1)
            schema = dataset.createVariable("schema_version", str, (dim.name,))
            schema[0] = "wrong-schema"

        result = self.validate(
            [
                self.entry(
                    pattern="product.nc",
                    checks={
                        "scalar_equals": {
                            "schema_version": "test-schema-v1",
                        },
                    },
                )
            ]
        )

        self.assertFalse(result["passed"])
        self.assertTrue(
            any("expected 'test-schema-v1'" in error for error in result["errors"])
        )

    @unittest.skipIf(product_contract.netCDF4 is None, "netCDF4 is unavailable")
    def test_checks_schema_variant_required_variables(self) -> None:
        import netCDF4

        path = self.reduction / "product.nc"
        with netCDF4.Dataset(path, "w") as dataset:
            dim = dataset.createDimension("schema", 1)
            schema = dataset.createVariable(
                "schema_version", str, (dim.name,)
            )
            schema[0] = "test-schema-v2"

        checks = {
            "scalar_one_of": {
                "schema_version": [
                    "test-schema-v1",
                    "test-schema-v2",
                ],
            },
            "required_variables_by_scalar": {
                "schema_version": {
                    "test-schema-v2": ["v2_state"],
                },
            },
        }
        result = self.validate(
            [self.entry(pattern="product.nc", checks=checks)]
        )
        self.assertFalse(result["passed"])
        self.assertTrue(
            any("requires variables ['v2_state']" in error
                for error in result["errors"])
        )

        with netCDF4.Dataset(path, "a") as dataset:
            dataset.createVariable("v2_state", "i4")
        result = self.validate(
            [self.entry(pattern="product.nc", checks=checks)]
        )
        self.assertTrue(result["passed"], result["errors"])


class CanonicalAptArtifactContractTest(unittest.TestCase):
    CONTRACT_ID = "apt-prod-001-canonical-baseline-apt-v1"
    SEMANTIC = (
        "sha256:a7911ac3b08ffdb9f3c6aaab36c33bb5abb47fac2bb729c2d09d79e68228f6db"
    )
    ENVELOPE = (
        "sha256:cb1e83e3f1a236f51ae80d8ab3f4f79106f3dde20153699513d15c883673b67b"
    )
    TRANSPORT = (
        "sha256:4adc27eac0f9934b885b916a9d2b537ea70f37d6a14fe6e1db891c6c51dee9be"
    )
    FIXTURE_ZLIB_BASE64 = (
        "eNrtHMuO47jx3l9B9CA3qkcPv/uUDDZYLDDJIYNdYA8RKImymZZFDUXZ7Q7yUfmRfFOKesuW3TLtnm2g7QE0slSsKtabZLk/oT/99OUfvyLrwbz7hAzDgGtAJJG7hC7UExSTNV2g+4wF9/AdNW/RPYvlZFQ+pakvWCIZj+EFfSa+RDGPY7okkm0oIkKyEB4aEfdJhATfoie6e0Qx3VCBEipSlkoaS0AkqS+5QCyAr0zu7ttcSB5TNxT0+wEvYcRJzU0WMwnPfn7pY06QLRKUBDyTSOEzFL6Mxv4OESkF8zJJgTEu+1kgQpDdcFH4BMTA1Jy/8ejbT19QPh7ROFsXRJQoegnF2+FUYE7GmsQspCnIncotF09KwB2ETyxIXTXj4XhfqOCGR1IaICU2f0VApZHCjLZMrlhcEeuKyA23q/UgFRHhp9TvIx0yKYEseWHrTK5Afn/97eevPWRcKsRbkEKAl4suwXUyiNJzeorKOomYzAK6j3rwRIah7+M/XkZ0EAlBghM0BIfB8ATVCLskBs/kNJkcF4D6VEgCJtq1Zu+aRhbJQmYHNuZd28balHpUBIHCVbawpnLFg3PCTMQ8UWqlNoECC/I53P/bxNZ/urR4DMF3SV0mqRhOyqNkvSYJqoaDfpDCkBPvEAggekiXRnRziTXkWIqZKVTHLC9cDp9CE/GXgmfJIyLg91xABAZeBE15tMntL1A5rDc4hywGTSWCceEG9unpdWnHlAgVpVMeSiNHgL4SCDqQJzyWogDyIFEiTb9nABgcpZpGIFsWB/R5+Lx7aCs0j8enGZEzxOqRoMnfaugj8iOukseGRBlNe2xQQdlnUGByy1KK/lLaIMgoUloriJEo4lugtibpEzKfw3CP1HMKbjIsAK5/2X1Whv75SLCNsufSAVJljjXimhrUOcOnpYoidathiAA8nA4AAwpdUskZppDwiAj2UjirrpMBUMLjlG3KF0NdrD3uESWrXZoXYH30O/RSGqdnWQd6Ruk/zYdxHxMKGZM17w0RtrRjDkZ8zozK7AFjIQBA1YpqDE1pTCHsE0ndOGEnQ0I5hwaqW7qvEyZycVUIUUkcBkDp5/Mslh3Cz668dv3FwzClB1TcPBNcSKvMJkPIvUFhWVDqyfuKHlTYF9JTNfrJae2upquqjukncm1VvULt+uXZUUXtrqeow0lBvUYWOXi9bHRJIt2NVTxFkGhWdE3cMu+oIo7JCHKgUQ8wYICxse7LAYngIYtoC7Ks3/JlXcRg/dsdEDIaBa6gSyhExO4ohRzMqMDa4wVZs3jpQnnFA7hpYYiIR6OIBoaSmRHReClXrZEpzAySgO+mPk/oUcoVmJGuiD2e9CLI36hgW9yQ6dyyiO945iwMA28eOv6EEOI5E99xPG9MPG80DYlve97Unvt2YM6D6ZxOZrY9CyeBVxGgkPAj4O0VDiuwQw4bBHsc+p5FZw51QovYziQcW4TOzGAGLIajELg3J6ETBNQ2rbEzmc/HlhNYY382cyZTx5tMaw69HWQAKUgMGVDIV/hUwEYNfMgt9/1MCFXgq9qh+SKhhvzMEwI19CfTbCa3gRwOlhPSakz+pADft7r2QJ7JJIORPLfTPvtsWXOQwXqwmVKteyhpt1DdAv0NK31jNCGB7cxmMzIyQaSm7QdhMDZ9c0JHczIezb0gnBHbrHBAQReyZVasZDoTKd4UM8k3t9o6zact2Zq6mfQB2DbtiWHODMv5ZtkLZ7QYT36v2fRVBcZCMFLAKemzrDw7n9y/oHZ2WVAgGRtfLOPL378ajaTUPDPhQ5IvotEvWaKWXuh//20guJdSUSyTjnFlLWxgzHmw7PHvzUCfcwEeq2oI5cT5cj6S5KVWU4N40UdtgawxiHve8Jp5nddm8wYMEcDL7xAN3Wr3qsFsVHtL7YEIQSVDBfhqsR0YwWrDvG+9Lreo3LxOWSC7D930JLrpCXQ1x3ncA/MI3DwQph22D3fBis+xdFF8epJG8YmzCCKn8g0pMtp6URe0qpwrPeO+733XlgvHWVR+BqmvZc05QWXtwE2+C0liQ/InGrcBXs0Ote8W0tnDP3Cjr1ZRuTaoAsTecqFX7E1JcBP9maJvih5NBVS7pGdIvlplfyiptzeCtUWtZegfWtwX23ezi36GyOvNzY8k88ODggtErmXpH1LsvQcnepL3bkXMWZLfP0i6ROq3GkZX8peG+L5TuAMdtLZUOxr42+c/94s/JFF6ffnXG7ZG6wzwhEYEVYvMs9QBq1S1mQ1oe/Z70CVnkZrqOTy4fK/aOe0dGro4xzVePanVE//+se67TcdLysHqDqz1x8an08fXehqozrovt/oTcs9isiEsh3td9PVTozlb++MMX/t8X1MdPc0AZziFjnL0naI5+7+6W5yZJzQ6IS7Xz37bxJu60Btr6e1T+dCGEU291N0l7zV9l/0lf2AKH9xSo68C+wPq4Ew/0Og70tVHu0npjCRy0LD0Q8LUsCXHD1gEHuvH0tNC3bx1K7D65X1+w5qeIurutpsi+hWh1dGnp4vktug4qYpLWh71NHLYH/nGq4631M7bJ4kz20L1dNL0kGok724/6YfK4HvtsprC7/bWvts1+DvZM+9pJ9aTe0/v8eAssTfi422XD+641tNN3Z79ng+R3sVO7Yk2dG3RtxuhbwoYtlV+dR3czlH1rf/So9TODxtuHvBaiXrkBxx6st/dQr9mE8EVJH+L/HqR/5oquAV+bdu/NO7vbnH/3Lh/keUrKbk+jaJ8m+Z7xkGnarHTOTgBftVPs0qwEiiT4cxImepIMHJ+C/gAmF+rbhGAxOpB8dMzpSJAw5OdYT+Y93cZC3D9V1lw/sdNcLzF9d8awUU/Pm7a8jFZJ7jsXMZ5IwSu+ztx0fmGmwY43O3Fwp3eH9y0ouBwibvn8Lj/2BerM5v8AhDlaQuOuI/BVHCyxO1dIqx2JXC9q4A761wMpQ2uS3xcFpq4LHgwOACu0wAuoxEu3eJubppTaz63x6PpyJzPLWyPzfKDLTzFJjYf7PGgC8DnNxYGO8WG1X5iVbfNxRqK+W5kY6tmS3Gk0OUoBlywXdyU/5ntJ3Z121zsoZjvzLaoin92juH1i4md4qb8T4lqVD1xqqfNxRmK+e7/R0EVug=="
    )

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.registry_path = (
            Path(__file__).resolve().parents[2]
            / "validation/product_contracts.json"
        )
        self.apt_path = self.root / "beammap_apt.ecsv"
        self.apt_bytes = zlib.decompress(
            base64.b64decode(self.FIXTURE_ZLIB_BASE64)
        )
        self.receipt_path = Path(str(self.apt_path) + ".sha256")

    def tearDown(self) -> None:
        self.directory.cleanup()

    def receipt_bytes(self) -> bytes:
        return (
            "citlali-canonical-apt-publication-receipt-v1\n"
            "scope=citlali-canonical-apt-byte-transport-sha256-v1\n"
            f"envelope_sha256={self.ENVELOPE}\n"
            f"byte_sha256={self.TRANSPORT}\n"
            "byte_count=18759\n"
        ).encode("ascii")

    @staticmethod
    def bits(value: float) -> str:
        return struct.pack(">d", value).hex()

    def observation_fixture(
        self, *, include_target_kids_flag: bool = True
    ) -> dict[str, object]:
        registry = product_contract.load_registry(self.registry_path)
        contracts = {
            contract_id: product_contract.artifact_contract_by_id(
                registry, contract_id
            )
            for contract_id in (self.BASELINE_ID, *self.SUCCESSOR_IDS)
        }
        baseline_contract = contracts[self.BASELINE_ID]
        baseline_document, _ = product_contract._parse_canonical_apt_v1_bytes(
            self.apt_bytes, baseline_contract
        )
        # The retained APT-PROD-001 byte fixture deliberately pins canonical
        # NaN.  APT-PROD-002's cross-language fixture starts from the ordinary
        # C++ make_document() value so the exact immutable baseline bytes are
        # independently reconstructed on both sides.
        baseline_document["rows"][0]["fields"]["final_prior_d2"] = 0.25
        seed_kids_flag = next(
            field
            for field in baseline_contract["optional_extensions"]
            if field["name"] == "kids_flag"
        )
        baseline_document["registered_fields"].append(
            copy.deepcopy(seed_kids_flag)
        )
        baseline_document["registered_fields"].sort(
            key=lambda field: field["name"]
        )
        for row, value in zip(
            baseline_document["rows"], [91, -5, 77], strict=True
        ):
            row["fields"]["kids_flag"] = value
        baseline_bytes = product_contract._serialize_canonical_apt_document(
            baseline_document, baseline_contract
        )
        baseline_digests = product_contract._canonical_apt_digests(
            baseline_document, baseline_contract
        )
        baseline_transport = (
            "sha256:" + hashlib.sha256(baseline_bytes).hexdigest()
        )
        baseline_receipt = (
            baseline_contract["receipt_schema"]
            + "\nscope="
            + baseline_contract["byte_transport_scope"]
            + "\nenvelope_sha256="
            + baseline_digests["envelope_sha256"]
            + "\nbyte_sha256="
            + baseline_transport
            + "\nbyte_count="
            + str(len(baseline_bytes))
            + "\n"
        ).encode("ascii")
        descriptor = product_contract.verified_baseline_descriptor_from_bytes(
            baseline_bytes, baseline_receipt, baseline_contract
        )

        def source(
            source_key: int,
            role: str,
            digest_digit: str,
            header: dict[str, str],
            network: int,
            channel_count: int,
        ) -> dict[str, object]:
            return {
                "source_key": str(source_key),
                "role": role,
                "diagnostic_locator": f"fixture/café/toltec{network}",
                "content_sha256": "sha256:" + digest_digit * 64,
                "byte_count": str(1000 + source_key),
                "header_observation": copy.deepcopy(header),
                "network": str(network),
                "interface": f"toltec{network}",
                "channel_count": str(channel_count),
            }

        observation = {
            "observation": "152390",
            "subobservation": "0",
            "scan": "1",
        }
        kmp_observation = {
            "observation": "152300",
            "subobservation": "0",
            "scan": "0",
        }
        target = {
            "schema_version": "citlali-observation-target-manifest-v1",
            "contract_authority": "citlali",
            "observation_value_issuer": "tolproj",
            "envelope": {
                "occurrence": "occurrence:tolproj/target#opaque-A",
                "event_reference": "event:tolproj/observation-target#A",
                "software_revision": "tolproj-clean-fixture-revision",
                "configuration_reference": "tolproj-config:sha256:fixture",
                "event_time_utc": "2026-08-14T01:02:03Z",
            },
            "observation": observation,
            "inputs": [
                {
                    "input_key": "70",
                    "network": "7",
                    "interface": "toltec7",
                    "channel_count": "1",
                    "raw_source": source(
                        700, "raw", "7", observation, 7, 1
                    ),
                    "kmp_source": source(
                        701, "kmp", "8", kmp_observation, 7, 1
                    ),
                },
                {
                    "input_key": "10",
                    "network": "0",
                    "interface": "toltec0",
                    "channel_count": "2",
                    "raw_source": source(
                        100, "raw", "1", observation, 0, 2
                    ),
                    "kmp_source": source(
                        101, "kmp", "2", kmp_observation, 0, 2
                    ),
                },
            ],
            "registered_fields": product_contract.canonical_target_fields_v1(
                include_kids_flag=include_target_kids_flag
            ),
            "rows": [],
            "target_source_sequence": ["11", "701", "5"],
            "target_application_sequence": ["5", "11", "701"],
        }
        target_row_values = (
            (
                "701", "70", "701", "0", self.bits(2.5e9),
                self.bits(2.5001e9), "1", "7", "0", self.bits(42000.0),
                "-7",
            ),
            (
                "11", "10", "101", "1", self.bits(-0.0),
                self.bits(-0.0), "0", "0", "1", self.bits(-3.0), "3",
            ),
            (
                "5", "10", "101", "0", "0000000000000001",
                self.bits(1.2501e9), "0", "0", "0",
                "0000000000000001", "42",
            ),
        )
        for (
            row_key,
            input_key,
            kmp_source_key,
            kmp_row_index,
            kids_fr,
            kids_f_out,
            array,
            network,
            channel,
            kids_qr,
            kids_flag,
        ) in target_row_values:
            fields: dict[str, object] = {
                "kids_fr": kids_fr,
                "kids_f_out": kids_f_out,
                "kids_Qr": kids_qr,
            }
            if include_target_kids_flag:
                fields["kids_flag"] = kids_flag
            target["rows"].append(
                {
                    "row_key": row_key,
                    "input_key": input_key,
                    "kmp_source_key": kmp_source_key,
                    "kmp_row_index": kmp_row_index,
                    "matching_frequency_hz": kids_fr,
                    "output_tone_frequency_hz": kids_f_out,
                    "array": array,
                    "network": network,
                    "channel": channel,
                    "fields": fields,
                }
            )
        target_contract = contracts[self.SUCCESSOR_IDS[0]]
        target_identity = product_contract.observation_target_identity(
            target, target_contract
        )
        seed_identity = product_contract._baseline_artifact_identity(descriptor)

        def row_reference(
            identity: dict[str, object], local_key: str
        ) -> dict[str, object]:
            return product_contract._row_reference(identity, local_key)

        relation = {
            "schema_version": "citlali-apt-match-dispositions-v1",
            "contract_authority": "citlali",
            "observation_value_issuer": "tolproj",
            "mapping_domain": "tolproj-observation-tone-to-beammap-seed-v1",
            "envelope": {
                "occurrence": "occurrence:tolproj/relation#opaque-B",
                "event_reference": "event:tolproj/matcher-run#B",
                "software_revision": "tolproj-clean-fixture-revision",
                "configuration_reference": "tolproj-match-config:sha256:fixture",
                "event_time_utc": "2026-08-14T01:03:04Z",
            },
            "baseline_parent": product_contract.verified_baseline_reference(
                descriptor
            ),
            "target_parent": target_identity,
            "matcher": {
                "matcher_run_occurrence": "occurrence:tolproj/matcher-policy-run#opaque",
                "implementation_revision": "tolproj-clean-fixture-revision",
                "configuration_reference": "tolproj-match-config:sha256:fixture",
                "target_frequency_field": "kids_fr",
                "target_quality_factor_field": "kids_Qr",
                "method": "astropy",
                "backend": "join-distance-v1",
            },
            "network_evidence": [
                {
                    "network": "7",
                    "frequency_shift_hz": self.bits(-0.0),
                    "gate_hz": self.bits(200000.0),
                    "quality_factor": self.bits(42000.0),
                    "quality_factor_field": "kids_Qr",
                    "quality_factor_authority_reference": "kids:model-params-v1",
                },
                {
                    "network": "0",
                    "frequency_shift_hz": "0000000000000001",
                    "gate_hz": self.bits(200000.0),
                    "quality_factor": self.bits(20000.0),
                    "quality_factor_field": "kids_Qr",
                    "quality_factor_authority_reference": "kids:model-params-v1",
                },
            ],
            "pairs": [
                {
                    "pair_key": "902",
                    "target": row_reference(target_identity, "11"),
                    "seed": row_reference(seed_identity, "42"),
                    "separation_hz": self.bits(3.0),
                    "is_good_match": True,
                },
                {
                    "pair_key": "900",
                    "target": row_reference(target_identity, "5"),
                    "seed": row_reference(seed_identity, "0"),
                    "separation_hz": self.bits(-0.0),
                    "is_good_match": True,
                },
                {
                    "pair_key": "901",
                    "target": row_reference(target_identity, "5"),
                    "seed": row_reference(seed_identity, "42"),
                    "separation_hz": "0000000000000001",
                    "is_good_match": False,
                },
            ],
            "target_dispositions": [
                {
                    "disposition_key": "1000",
                    "endpoint": row_reference(target_identity, "701"),
                    "state": "unmatched",
                    "pair_keys": [],
                    "reason": "no selected seed endpoint",
                },
                {
                    "disposition_key": "1002",
                    "endpoint": row_reference(target_identity, "5"),
                    "state": "matched",
                    "pair_keys": ["900", "901"],
                    "reason": "two realized candidate endpoints retained",
                },
                {
                    "disposition_key": "1001",
                    "endpoint": row_reference(target_identity, "11"),
                    "state": "matched",
                    "pair_keys": ["902"],
                    "reason": "one realized endpoint",
                },
            ],
            "seed_dispositions": [
                {
                    "disposition_key": "2000",
                    "endpoint": row_reference(
                        seed_identity, str(product_contract.CANONICAL_APT_UID_MAX)
                    ),
                    "state": "unused",
                    "pair_keys": [],
                    "reason": "seed not used",
                },
                {
                    "disposition_key": "2002",
                    "endpoint": row_reference(seed_identity, "0"),
                    "state": "matched",
                    "pair_keys": ["900"],
                    "reason": "one target endpoint",
                },
                {
                    "disposition_key": "2001",
                    "endpoint": row_reference(seed_identity, "42"),
                    "state": "matched",
                    "pair_keys": ["901", "902"],
                    "reason": "two target endpoints",
                },
            ],
            "seed_source_sequence": [
                "42", str(product_contract.CANONICAL_APT_UID_MAX), "0"
            ],
        }
        relation_contract = contracts[self.SUCCESSOR_IDS[1]]
        relation_identity = product_contract.match_dispositions_identity(
            relation, relation_contract, descriptor, target, target_contract
        )
        output_contract = contracts[self.SUCCESSOR_IDS[2]]
        output_fields = product_contract.canonical_output_field_contracts_v1(
            descriptor, target, target_contract
        )
        target_by_key = {row["row_key"]: row for row in target["rows"]}
        baseline_by_key = {row["uid"]: row for row in descriptor["rows"]}
        baseline_fields = {
            field["name"]: field for field in descriptor["registered_fields"]
        }

        def output_row(
            uid: str,
            target_key: str,
            pair_keys: list[str],
            source_pair_key: str | None,
            seed_key: str | None,
        ) -> dict[str, object]:
            target_row = target_by_key[target_key]
            target_reference = row_reference(target_identity, target_key)
            fields: dict[str, object] = {}
            transformations: list[dict[str, object]] = []
            for field_contract in output_fields:
                field = field_contract["field"]
                name = field["name"]
                if field_contract["authorized_operation"] == "preserve-target":
                    value = target_row["fields"][name]
                    fields[name] = value
                    transformations.append(
                        {
                            "field_name": name,
                            "operation": "preserve-target",
                            "before": value,
                            "after": value,
                            "value_source": "target-row",
                            "source_pair_key": None,
                            "source_row": target_reference,
                            "authority_reference": field["authority_reference"],
                            "provenance_reference": (
                                f"target-kmp-source:{target_row['kmp_source_key']}:"
                                f"row:{target_row['kmp_row_index']}:"
                                f"column:{field['source_column']}"
                            ),
                        }
                    )
                elif seed_key is not None:
                    value = baseline_by_key[seed_key]["fields"][name]["value"]
                    fields[name] = value
                    transformations.append(
                        {
                            "field_name": name,
                            "operation": "copy-baseline-when-matched-null-when-unmatched",
                            "before": None,
                            "after": value,
                            "value_source": "baseline-seed-row",
                            "source_pair_key": source_pair_key,
                            "source_row": row_reference(seed_identity, seed_key),
                            "authority_reference": baseline_fields[name][
                                "authority_reference"
                            ],
                            "provenance_reference": f"relation-pair:{source_pair_key}",
                        }
                    )
                else:
                    fields[name] = None
                    transformations.append(
                        {
                            "field_name": name,
                            "operation": "copy-baseline-when-matched-null-when-unmatched",
                            "before": None,
                            "after": None,
                            "value_source": "canonical-null",
                            "source_pair_key": None,
                            "source_row": None,
                            "authority_reference": "citlali:typed-missing-unmatched-v1",
                            "provenance_reference": "target-unmatched:no-fabricated-seed",
                        }
                    )
            return {
                "uid": uid,
                "target": target_reference,
                "target_input_key": target_row["input_key"],
                "tone_frequency_hz": target_row["output_tone_frequency_hz"],
                "array": target_row["array"],
                "network": target_row["network"],
                "channel": target_row["channel"],
                "relation_pair_keys": pair_keys,
                "fields": fields,
                "transformations": transformations,
            }

        output = {
            "schema_version": "citlali-observation-matched-apt-v1",
            "contract_authority": "citlali",
            "observation_value_issuer": "tolproj",
            "transformation_registry": "citlali-observation-apt-field-transformations-v1",
            "envelope": {
                "occurrence": "occurrence:tolproj/matched-output#opaque-C",
                "event_reference": "event:tolproj/observation-output#C",
                "software_revision": "tolproj-clean-fixture-revision",
                "configuration_reference": "tolproj-output-config:sha256:fixture",
                "event_time_utc": "2026-08-14T01:04:05Z",
            },
            "baseline_parent": product_contract.verified_baseline_reference(
                descriptor
            ),
            "target_parent": target_identity,
            "relation_parent": relation_identity,
            "registered_fields": output_fields,
            "rows": [
                output_row("888", "701", [], None, None),
                output_row("4", "5", ["900", "901"], "900", "0"),
                output_row(
                    str(product_contract.CANONICAL_APT_UID_MAX - 1),
                    "11", ["902"], "902", "42",
                ),
            ],
            "output_presentation_sequence": [
                str(product_contract.CANONICAL_APT_UID_MAX - 1), "888", "4"
            ],
        }
        return {
            "contracts": contracts,
            "baseline_bytes": baseline_bytes,
            "baseline_receipt": baseline_receipt,
            "descriptor": descriptor,
            "target": target,
            "relation": relation,
            "output": output,
        }

    def write_valid_pair(self) -> None:
        self.apt_path.write_bytes(self.apt_bytes)
        self.receipt_path.write_bytes(self.receipt_bytes())

    def contract(self) -> dict[str, object]:
        registry = product_contract.load_registry(self.registry_path)
        return product_contract.artifact_contract_by_id(
            registry, self.CONTRACT_ID
        )

    def receipt_for(
        self, artifact_bytes: bytes, envelope_sha256: str
    ) -> bytes:
        byte_sha256 = "sha256:" + hashlib.sha256(artifact_bytes).hexdigest()
        return (
            "citlali-canonical-apt-publication-receipt-v1\n"
            "scope=citlali-canonical-apt-byte-transport-sha256-v1\n"
            f"envelope_sha256={envelope_sha256}\n"
            f"byte_sha256={byte_sha256}\n"
            f"byte_count={len(artifact_bytes)}\n"
        ).encode("ascii")

    def write_pair(self, artifact_bytes: bytes, envelope_sha256: str) -> None:
        self.apt_path.write_bytes(artifact_bytes)
        self.receipt_path.write_bytes(
            self.receipt_for(artifact_bytes, envelope_sha256)
        )

    def document(self) -> tuple[dict[str, object], dict[str, object]]:
        contract = self.contract()
        document, _digests = product_contract._parse_canonical_apt_v1_bytes(
            self.apt_bytes, contract
        )
        return copy.deepcopy(document), contract

    def bytes_for_document(
        self, document: dict[str, object], contract: dict[str, object]
    ) -> tuple[bytes, dict[str, str]]:
        digests = product_contract._canonical_apt_digests(document, contract)
        artifact_bytes = product_contract._serialize_canonical_apt_document(
            document, contract, digests
        )
        return artifact_bytes, digests

    def validate_document(
        self, document: dict[str, object], contract: dict[str, object]
    ) -> dict[str, object]:
        artifact_bytes, digests = self.bytes_for_document(document, contract)
        self.write_pair(artifact_bytes, digests["envelope_sha256"])
        return product_contract.validate_canonical_apt_v1_artifact(
            self.apt_path, contract
        )

    def test_checked_contract_is_exact_unactivated_and_unrouted(self) -> None:
        registry = product_contract.load_registry(self.registry_path)
        artifact = product_contract.artifact_contract_by_id(
            registry, self.CONTRACT_ID
        )
        self.assertEqual(artifact["activation_state"], "unactivated")
        self.assertEqual(len(artifact["required_fields"]), 27)
        self.assertEqual(len(artifact["optional_extensions"]), 20)
        self.assertEqual(
            artifact["optional_extensions"][6]["name"], "kids_flag"
        )
        self.assertFalse(
            any(
                entry.get("artifact_contract_id") == self.CONTRACT_ID
                for contract in registry["contracts"]
                for entry in contract["entries"]
            )
        )

    def test_cross_language_vectors_and_valid_artifact_pair(self) -> None:
        self.assertEqual(len(self.apt_bytes), 18759)
        self.assertEqual(
            "sha256:" + hashlib.sha256(self.apt_bytes).hexdigest(),
            self.TRANSPORT,
        )
        self.assertEqual(
            hashlib.sha256(
                product_contract.canonical_frame(
                    "uid", "int64", "9007199254740991"
                )
            ).hexdigest(),
            "5e86d924a3acd47ae21e8fcb5c21bb40da9f37a9a16755dcdb6cf112c166250b",
        )
        self.write_valid_pair()
        registry = product_contract.load_registry(self.registry_path)
        artifact = product_contract.artifact_contract_by_id(
            registry, self.CONTRACT_ID
        )
        result = product_contract.validate_canonical_apt_v1_artifact(
            self.apt_path, artifact
        )
        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["semantic_sha256"], self.SEMANTIC)
        self.assertEqual(result["envelope_sha256"], self.ENVELOPE)
        self.assertEqual(result["byte_sha256"], self.TRANSPORT)
        self.assertEqual(result["byte_count"], 18759)

        float_frames = b"".join(
            [
                product_contract.canonical_frame(
                    "one", "float64-ieee754", "3ff0000000000000"
                ),
                product_contract.canonical_frame(
                    "negative_zero", "float64-ieee754", "8000000000000000"
                ),
                product_contract.canonical_frame(
                    "denorm_min", "float64-ieee754", "0000000000000001"
                ),
                product_contract.canonical_frame(
                    "positive_infinity", "float64-ieee754", "+inf"
                ),
                product_contract.canonical_frame(
                    "quiet_nan", "float64-ieee754", "nan"
                ),
            ]
        )
        self.assertEqual(
            hashlib.sha256(float_frames).hexdigest(),
            "4a566f76572a46c00bd06d035851e1cd80dbfbe640f90d1643bfc38197732ded",
        )
        null_frame = product_contract.canonical_frame(
            "missing", "null-int64", "null"
        )
        self.assertEqual(
            hashlib.sha256(null_frame).hexdigest(),
            "667dbdc49e83c7d94a4d5ea215328fbd76440a4a80e89cef2b984b7c19c0c872",
        )
        self.assertIn(b"V2:\xce\xb1;", product_contract.canonical_frame(
            "source", "utf8", "α"
        ))
        self.assertEqual(
            hashlib.sha256(b"abc").hexdigest(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        )
        boundary_tokens = [
            "-0",
            "4.9406564584124654e-324",
            "2.2250738585072014e-308",
            "1.7976931348623157e+308",
            "1.0000000000000002",
            "0.0001",
            "1.0000000000000001e-05",
            "10000000000000000",
            "1e+17",
        ]
        for token in boundary_tokens:
            with self.subTest(token=token):
                value = product_contract._parse_float64(token, "vector")
                self.assertEqual(product_contract._format_float64(value), token)

    def test_missing_and_tampered_receipt_fail_closed(self) -> None:
        self.apt_path.write_bytes(self.apt_bytes)
        registry = product_contract.load_registry(self.registry_path)
        artifact = product_contract.artifact_contract_by_id(
            registry, self.CONTRACT_ID
        )
        result = product_contract.validate_canonical_apt_v1_artifact(
            self.apt_path, artifact
        )
        self.assertFalse(result["passed"])
        malformed_receipts = [
            self.receipt_bytes().replace(
                b"citlali-canonical-apt-publication-receipt-v1",
                b"wrong-receipt-v1",
            ),
            self.receipt_bytes().replace(
                b"scope=citlali-canonical-apt-byte-transport-sha256-v1",
                b"scope=wrong",
            ),
            self.receipt_bytes().replace(self.ENVELOPE.encode(), b"sha256:" + b"0" * 64),
            self.receipt_bytes().replace(self.TRANSPORT.encode(), b"sha256:" + b"0" * 64),
            self.receipt_bytes() + b"extra=true\n",
            self.receipt_bytes()[:-1],
            self.receipt_bytes().replace(b"\n", b"\r\n"),
        ]
        for receipt in malformed_receipts:
            with self.subTest(receipt=receipt[:48]):
                self.receipt_path.write_bytes(receipt)
                result = product_contract.validate_canonical_apt_v1_artifact(
                    self.apt_path, artifact
                )
                self.assertFalse(result["passed"])
        self.receipt_path.write_bytes(self.receipt_bytes())
        self.apt_path.write_bytes(self.apt_bytes.replace(b"Jupiter", b"Saturn", 1))
        result = product_contract.validate_canonical_apt_v1_artifact(
            self.apt_path, artifact
        )
        self.assertFalse(result["passed"])
        self.receipt_path.write_bytes(
            self.receipt_bytes().replace(b"byte_count=18759", b"byte_count=1")
        )
        result = product_contract.validate_canonical_apt_v1_artifact(
            self.apt_path, artifact
        )
        self.assertFalse(result["passed"])

    def test_registry_catalog_drift_duplicate_json_and_legacy_isolation(self) -> None:
        registry = json.loads(self.registry_path.read_text(encoding="utf-8"))
        registry["artifact_contracts"][self.CONTRACT_ID]["required_fields"][0][
            "description"
        ] = "drifted"
        drifted = self.root / "drifted.json"
        drifted.write_text(json.dumps(registry), encoding="utf-8")
        with self.assertRaisesRegex(product_contract.ContractError, "catalog drift"):
            product_contract.load_registry(drifted)

        duplicate = self.root / "duplicate.json"
        duplicate.write_text(
            '{"schema_version":"citlali-product-contract-registry-v2",'
            '"schema_version":"citlali-product-contract-registry-v2",'
            '"families":{},"contracts":[]}',
            encoding="utf-8",
        )
        with self.assertRaisesRegex(product_contract.ContractError, "duplicate JSON"):
            product_contract.load_registry(duplicate)

        nonfinite = self.root / "nonfinite.json"
        nonfinite.write_text(
            '{"schema_version":"citlali-product-contract-registry-v2",'
            '"families":{},"contracts":[],"bad":NaN}',
            encoding="utf-8",
        )
        with self.assertRaisesRegex(product_contract.ContractError, "non-finite JSON"):
            product_contract.load_registry(nonfinite)

        self.apt_path.write_bytes(self.apt_bytes)
        errors = product_contract.validate_ecsv(
            self.apt_path, {"required_columns": ["uid"], "min_rows": 1}, []
        )
        self.assertFalse(errors)
        self.assertFalse(self.receipt_path.exists())

    def test_wire_text_metadata_and_embedded_digest_tampering_fail_closed(self) -> None:
        contract = self.contract()

        def invalid(artifact_bytes: bytes) -> dict[str, object]:
            self.write_pair(artifact_bytes, self.ENVELOPE)
            return product_contract.validate_canonical_apt_v1_artifact(
                self.apt_path, contract
            )

        self.assertFalse(invalid(self.apt_bytes.replace(b"\n", b"\r\n"))["passed"])
        self.assertFalse(invalid(self.apt_bytes[:-1])["passed"])
        self.assertFalse(
            invalid(self.apt_bytes.replace("α".encode(), b"\xff", 1))["passed"]
        )
        for noncharacter in ("\ufdd0", "\U0001fffe", "\U0010ffff"):
            with self.subTest(noncharacter=hex(ord(noncharacter))):
                self.assertFalse(
                    invalid(
                        self.apt_bytes.replace(
                            "Jupiter α".encode(),
                            ("Jupiter " + noncharacter).encode(),
                        )
                    )["passed"]
                )
        duplicated_meta = self.apt_bytes.replace(
            b'#     profile: "citlali-beammap-baseline-apt-v1"\n',
            b'#     profile: "citlali-beammap-baseline-apt-v1"\n'
            b'#     profile: "citlali-beammap-baseline-apt-v1"\n',
            1,
        )
        self.assertFalse(invalid(duplicated_meta)["passed"])
        quoted_uid = self.apt_bytes.replace(
            b"9007199254740991,2500000000,", b'"9007199254740991",2500000000,', 1
        )
        self.assertFalse(invalid(quoted_uid)["passed"])
        missing_uid = self.apt_bytes.replace(
            b"9007199254740991,2500000000,", b",2500000000,", 1
        )
        self.assertFalse(invalid(missing_uid)["passed"])
        fractional_uid = self.apt_bytes.replace(
            b"42,1250000000,", b"42.5,1250000000,", 1
        )
        self.assertFalse(invalid(fractional_uid)["passed"])
        stale_semantic = self.apt_bytes.replace(
            self.SEMANTIC.encode(),
            ("sha256:0" + self.SEMANTIC[8:]).encode(),
            1,
        )
        self.assertFalse(invalid(stale_semantic)["passed"])
        stale_envelope = self.apt_bytes.replace(
            self.ENVELOPE.encode(),
            ("sha256:0" + self.ENVELOPE[8:]).encode(),
            1,
        )
        self.assertFalse(invalid(stale_envelope)["passed"])

    def test_uid_raw_relation_and_array_counterexamples(self) -> None:
        document, contract = self.document()
        rows = document["rows"]
        self.assertEqual([row["uid"] for row in rows], [9007199254740991, 42, 0])
        self.assertEqual(
            {(row["nw"], row["kids_tone"]) for row in rows},
            {(7, 0), (0, 1), (0, 0)},
        )

        duplicate_uid = copy.deepcopy(document)
        duplicate_uid["rows"][0]["uid"] = 42
        self.assertFalse(self.validate_document(duplicate_uid, contract)["passed"])

        out_of_range_uid = copy.deepcopy(document)
        out_of_range_uid["rows"][0]["uid"] = 9007199254740992
        self.assertFalse(
            self.validate_document(out_of_range_uid, contract)["passed"]
        )

        duplicate_relation = copy.deepcopy(document)
        duplicate_relation["rows"][1]["kids_tone"] = 0
        self.assertFalse(
            self.validate_document(duplicate_relation, contract)["passed"]
        )

        wrong_count = copy.deepcopy(document)
        wrong_count["raw_inputs"][0]["channel_count"] = 3
        self.assertFalse(self.validate_document(wrong_count, contract)["passed"])

        duplicate_input = copy.deepcopy(document)
        duplicate_input["raw_inputs"].append(
            copy.deepcopy(duplicate_input["raw_inputs"][0])
        )
        self.assertFalse(
            self.validate_document(duplicate_input, contract)["passed"]
        )

        wrong_interface = copy.deepcopy(document)
        wrong_interface["raw_inputs"][0]["interface"] = "toltec00"
        self.assertFalse(
            self.validate_document(wrong_interface, contract)["passed"]
        )

        wrong_array = copy.deepcopy(document)
        wrong_array["rows"][0]["array"] = 0
        self.assertFalse(self.validate_document(wrong_array, contract)["passed"])

    def test_field_catalog_nonfinite_and_flag_domains(self) -> None:
        document, contract = self.document()
        missing = copy.deepcopy(document)
        missing["registered_fields"] = [
            field for field in missing["registered_fields"] if field["name"] != "amp"
        ]
        for row in missing["rows"]:
            del row["fields"]["amp"]
        self.assertFalse(self.validate_document(missing, contract)["passed"])

        rogue = copy.deepcopy(document)
        field = copy.deepcopy(contract["optional_extensions"][0])
        field["name"] = "runtime_surprise"
        rogue["registered_fields"].append(field)
        for row in rogue["rows"]:
            row["fields"]["runtime_surprise"] = 1.0
        self.assertFalse(self.validate_document(rogue, contract)["passed"])

        protected = copy.deepcopy(document)
        field = copy.deepcopy(contract["optional_extensions"][0])
        field["name"] = "occurrence"
        protected["registered_fields"].append(field)
        for row in protected["rows"]:
            row["fields"]["occurrence"] = 1.0
        self.assertFalse(
            self.validate_document(protected, contract)["passed"]
        )

        infinity = copy.deepcopy(document)
        infinity["rows"][0]["fields"]["final_prior_d2"] = float("inf")
        self.assertFalse(self.validate_document(infinity, contract)["passed"])

        with_kids_flag = copy.deepcopy(document)
        kids_flag = next(
            field
            for field in contract["optional_extensions"]
            if field["name"] == "kids_flag"
        )
        with_kids_flag["registered_fields"].append(copy.deepcopy(kids_flag))
        for row, value in zip(with_kids_flag["rows"], [3, -7, 42], strict=True):
            row["fields"]["kids_flag"] = value
        result = self.validate_document(with_kids_flag, contract)
        self.assertTrue(result["passed"], result["errors"])

        bad_final_flag = copy.deepcopy(with_kids_flag)
        bad_final_flag["rows"][0]["fields"]["flag"] = 2
        self.assertFalse(
            self.validate_document(bad_final_flag, contract)["passed"]
        )

    def test_row_order_and_occurrence_have_separate_identity_scopes(self) -> None:
        document, contract = self.document()
        original_bytes, original_digests = self.bytes_for_document(document, contract)

        reordered = copy.deepcopy(document)
        reordered["rows"] = list(reversed(reordered["rows"]))
        reordered_bytes, reordered_digests = self.bytes_for_document(
            reordered, contract
        )
        self.assertEqual(
            reordered_digests["semantic_sha256"],
            original_digests["semantic_sha256"],
        )
        self.assertEqual(
            reordered_digests["envelope_sha256"],
            original_digests["envelope_sha256"],
        )
        self.assertNotEqual(
            hashlib.sha256(reordered_bytes).digest(),
            hashlib.sha256(original_bytes).digest(),
        )
        self.write_pair(reordered_bytes, reordered_digests["envelope_sha256"])
        result = product_contract.validate_canonical_apt_v1_artifact(
            self.apt_path, contract
        )
        self.assertTrue(result["passed"], result["errors"])

        event = copy.deepcopy(document)
        event["envelope"]["event_reference"] = "event:test/distinct/B"
        _event_bytes, event_digests = self.bytes_for_document(event, contract)
        self.assertEqual(
            event_digests["semantic_sha256"],
            original_digests["semantic_sha256"],
        )
        self.assertNotEqual(
            event_digests["envelope_sha256"],
            original_digests["envelope_sha256"],
        )

        occurrence = copy.deepcopy(document)
        occurrence["envelope"]["occurrence"] = "opaque:event/B"
        occurrence_bytes, occurrence_digests = self.bytes_for_document(
            occurrence, contract
        )
        self.assertEqual(
            occurrence_digests["semantic_sha256"],
            original_digests["semantic_sha256"],
        )
        self.assertNotEqual(
            occurrence_digests["envelope_sha256"],
            original_digests["envelope_sha256"],
        )
        self.write_pair(
            occurrence_bytes, occurrence_digests["envelope_sha256"]
        )
        result = product_contract.validate_canonical_apt_v1_artifact(
            self.apt_path, contract
        )
        self.assertTrue(result["passed"], result["errors"])

    def test_artifact_cli_is_unactivated_stdout_only(self) -> None:
        self.write_valid_pair()
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = product_contract.main(
                [
                    str(self.apt_path),
                    "--artifact-contract",
                    self.CONTRACT_ID,
                    "--registry",
                    str(self.registry_path),
                ]
            )
        self.assertEqual(status, 0, stderr.getvalue())
        self.assertIn("VALID / conformant", stdout.getvalue())
        self.assertIn("unactivated / deferred", stdout.getvalue())

        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = product_contract.main(
                [
                    str(self.apt_path),
                    "--artifact-contract",
                    self.CONTRACT_ID,
                    "--json-out",
                    str(self.apt_path),
                    "--registry",
                    str(self.registry_path),
                ]
            )
        self.assertEqual(status, 2)
        self.assertEqual(self.apt_path.read_bytes(), self.apt_bytes)


class CanonicalAptProtocolV1Test(unittest.TestCase):
    bits = staticmethod(CanonicalAptArtifactContractTest.bits)
    BASELINE_ID = "apt-prod-001-canonical-baseline-apt-v1"
    SEMANTIC = CanonicalAptArtifactContractTest.SEMANTIC
    ENVELOPE = CanonicalAptArtifactContractTest.ENVELOPE
    TRANSPORT = CanonicalAptArtifactContractTest.TRANSPORT
    SUCCESSOR_IDS = (
        "apt-prod-002-observation-target-manifest-v1",
        "apt-prod-002-match-dispositions-v1",
        "apt-prod-002-observation-matched-apt-v1",
    )

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.registry_path = (
            Path(__file__).resolve().parents[2]
            / "validation/product_contracts.json"
        )
        self.apt_bytes = zlib.decompress(
            base64.b64decode(
                CanonicalAptArtifactContractTest.FIXTURE_ZLIB_BASE64
            )
        )
        self.receipt_bytes = (
            "citlali-canonical-apt-publication-receipt-v1\n"
            "scope=citlali-canonical-apt-byte-transport-sha256-v1\n"
            "envelope_sha256="
            + CanonicalAptArtifactContractTest.ENVELOPE
            + "\nbyte_sha256="
            + CanonicalAptArtifactContractTest.TRANSPORT
            + "\nbyte_count=18759\n"
        ).encode("ascii")

    def tearDown(self) -> None:
        self.directory.cleanup()

    def protocol_cli(self) -> Path:
        path = Path(__file__).resolve().parents[2] / "build/bin/citlali"
        self.assertTrue(
            path.is_file(),
            "Phase-B protocol tests require the focused citlali_cli build",
        )
        return path

    def run_protocol(self, request: str | dict[str, object]) -> tuple[
        subprocess.CompletedProcess[str], dict[str, object]
    ]:
        if isinstance(request, dict):
            request = json.dumps(request, separators=(",", ":")) + "\n"
        completed = subprocess.run(
            [str(self.protocol_cli()), "--canonical-apt-contract-v1"],
            input=request,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.stderr, "")
        self.assertEqual(len(completed.stdout.splitlines()), 1)
        return completed, json.loads(completed.stdout)

    def contracts(self) -> dict[str, object]:
        registry = product_contract.load_registry(self.registry_path)
        return {
            contract_id: product_contract.artifact_contract_by_id(
                registry, contract_id
            )
            for contract_id in (self.BASELINE_ID, *self.SUCCESSOR_IDS)
        }

    def expected_baseline_descriptor(self) -> dict[str, object]:
        return product_contract.verified_baseline_descriptor_from_bytes(
            self.apt_bytes,
            self.receipt_bytes,
            self.contracts()[self.BASELINE_ID],
        )

    def write_protocol_baseline(self) -> Path:
        path = self.root / "baseline.ecsv"
        path.write_bytes(self.apt_bytes)
        Path(str(path) + ".sha256").write_bytes(self.receipt_bytes)
        return path

    def describe_protocol_baseline(
        self, baseline_path: Path
    ) -> dict[str, object]:
        completed, response = self.run_protocol(
            {
                "protocol": "citlali-canonical-apt-protocol-v1",
                "request_id": "python-bootstrap-describe",
                "operation": "describe-baseline-v1",
                "payload": {"baseline_ecsv": str(baseline_path)},
            }
        )
        self.assertEqual(completed.returncode, 0, response)
        return response["result"]

    def protocol_issue_request(
        self,
    ) -> tuple[dict[str, object], dict[str, object], list[Path]]:
        baseline_path = self.write_protocol_baseline()
        described = self.describe_protocol_baseline(baseline_path)
        descriptor = described["baseline"]
        baseline_occurrence = descriptor["occurrence"]
        seed_uids = [row["uid"] for row in descriptor["rows"]]
        self.assertEqual(set(seed_uids), {"0", "42", "9007199254740991"})

        observation = {
            "observation": "152390", "subobservation": "0", "scan": "1"
        }
        source_paths: list[Path] = []

        def source(source_key: str, role: str) -> dict[str, object]:
            source_bytes = f"bound-{role}-network-0\n".encode("ascii")
            source_path = self.root / f"source-{role}.bin"
            source_path.write_bytes(source_bytes)
            source_paths.append(source_path)
            return {
                "source_key": source_key,
                "network": "0",
                "interface": "toltec0",
                "channel_count": "1",
                "diagnostic_locator": str(source_path),
                "content_sha256": (
                    "sha256:" + hashlib.sha256(source_bytes).hexdigest()
                ),
                "byte_count": str(len(source_bytes)),
                "header_observation": copy.deepcopy(observation),
            }

        target_occurrence = "occurrence:tolproj/target#opaque-A"
        relation_occurrence = "occurrence:tolproj/relation#opaque-B"
        target = {
            "envelope": {
                "occurrence": target_occurrence,
                "event_reference": "event:tolproj/target#opaque-A",
                "software_revision": "tolproj:test-revision",
                "configuration_reference": "tolproj:test-config",
                "event_time_utc": "2026-08-14T12:00:00Z",
            },
            "observation": observation,
            "inputs": [{
                "input_key": "10",
                "network": "0",
                "interface": "toltec0",
                "channel_count": "1",
                "raw_source": source("100", "raw"),
                "kmp_source": source("101", "kmp"),
            }],
            "rows": [{
                "row_key": "5",
                "input_key": "10",
                "kmp_source_key": "101",
                "kmp_row_index": "0",
                "array": "0",
                "network": "0",
                "channel": "0",
                "fields": {
                    "kids_fr": self.bits(1.25e9),
                    "kids_f_out": self.bits(1.2501e9),
                    "kids_Qr": self.bits(20000.0),
                    "kids_flag": "42",
                },
            }],
            "target_source_sequence": ["5"],
            "target_application_sequence": ["5"],
        }
        scoped_target = {"occurrence": target_occurrence, "local_key": "5"}

        def scoped_seed(uid: str) -> dict[str, str]:
            return {"occurrence": baseline_occurrence, "local_key": uid}

        pairs = [
            {
                "pair_key": "900",
                "target": copy.deepcopy(scoped_target),
                "seed": scoped_seed("0"),
                "separation_hz": self.bits(1250.0),
                "is_good_match": True,
            },
            {
                "pair_key": "901",
                "target": copy.deepcopy(scoped_target),
                "seed": scoped_seed("42"),
                "separation_hz": self.bits(2500.0),
                "is_good_match": True,
            },
        ]
        seed_pair_keys = {"0": ["900"], "42": ["901"]}
        relation = {
            "envelope": {
                "occurrence": relation_occurrence,
                "event_reference": "event:tolproj/relation#opaque-B",
                "software_revision": "tolproj:test-revision",
                "configuration_reference": "tolproj:test-config",
                "event_time_utc": "2026-08-14T12:01:00Z",
            },
            "matcher": {
                "matcher_run_occurrence": "occurrence:tolproj/matcher#A",
                "implementation_revision": "tolproj-matcher:test-revision",
                "configuration_reference": "tolproj-matcher:test-config",
                "method": "issuer-selected-test-method",
                "backend": "issuer-selected-test-backend",
            },
            "network_evidence": [{
                "network": "0",
                "frequency_shift_hz": self.bits(0.0),
                "gate_hz": self.bits(1.0e6),
                "quality_factor": self.bits(20000.0),
            }],
            "pairs": pairs,
            "target_dispositions": [{
                "disposition_key": "1000",
                "endpoint": copy.deepcopy(scoped_target),
                "state": "matched",
                "pair_keys": ["900", "901"],
                "reason": "issuer-declared complete candidate set",
            }],
            "seed_dispositions": [
                {
                    "disposition_key": str(2000 + index),
                    "endpoint": scoped_seed(uid),
                    "state": "matched" if uid in seed_pair_keys else "unused",
                    "pair_keys": seed_pair_keys.get(uid, []),
                    "reason": (
                        "issuer-declared selected candidate"
                        if uid in seed_pair_keys else "issuer-declared unused"
                    ),
                }
                for index, uid in enumerate(sorted(seed_uids, key=int))
            ],
            "seed_source_sequence": seed_uids,
        }
        selections = [{
            "target": copy.deepcopy(scoped_target),
            "default_source_pair": {
                "occurrence": relation_occurrence,
                "local_key": "900",
            },
            "field_overrides": [{
                "field_name": "amp",
                "source_pair": {
                    "occurrence": relation_occurrence,
                    "local_key": "901",
                },
            }],
        }]
        output_path = self.root / "issued.apt.ecsv"
        request = {
            "protocol": "citlali-canonical-apt-protocol-v1",
            "request_id": "python-direct-issue",
            "operation": "issue-observation-apt-v1",
            "payload": {
                "baseline_ecsv": str(baseline_path),
                "expected_baseline": copy.deepcopy(
                    described["baseline_reference"]
                ),
                "target": target,
                "relation": relation,
                "field_source_selections": selections,
                "publication": {
                    "output_ecsv": str(output_path),
                    "configuration_reference": "python-direct-config:opaque",
                    "event_time_utc": "2026-08-14T12:34:56Z",
                },
            },
        }
        verifier = {
            "contracts": self.contracts(),
            "descriptor": self.expected_baseline_descriptor(),
            "described": described,
        }
        return verifier, request, source_paths

    def test_versioned_protocol_describes_complete_verified_baseline(self) -> None:
        baseline_path = self.write_protocol_baseline()
        descriptor = self.expected_baseline_descriptor()
        completed, response = self.run_protocol(
            {
                "protocol": "citlali-canonical-apt-protocol-v1",
                "request_id": "python-direct-describe",
                "operation": "describe-baseline-v1",
                "payload": {"baseline_ecsv": str(baseline_path)},
            }
        )
        self.assertEqual(completed.returncode, 0)
        self.assertEqual(
            set(response),
            {"protocol", "request_id", "status", "operation", "result"},
        )
        self.assertEqual(response["protocol"],
                         "citlali-canonical-apt-protocol-v1")
        self.assertEqual(response["request_id"], "python-direct-describe")
        self.assertEqual(response["status"], "ok")
        self.assertEqual(response["operation"], "describe-baseline-v1")
        self.assertEqual(
            set(response["result"]),
            {"baseline", "baseline_reference", "baseline_ecsv", "receipt"},
        )
        self.assertEqual(response["result"]["baseline"], descriptor)
        self.assertEqual(
            response["result"]["baseline_reference"],
            product_contract.verified_baseline_reference(descriptor),
        )

    def test_versioned_protocol_strict_json_and_receipts_fail_closed(self) -> None:
        baseline_path = self.write_protocol_baseline()
        descriptor = self.expected_baseline_descriptor()
        protocol = "citlali-canonical-apt-protocol-v1"
        good = {
            "protocol": protocol,
            "request_id": "strict-case",
            "operation": "describe-baseline-v1",
            "payload": {"baseline_ecsv": str(baseline_path)},
        }
        compact = json.dumps(good, separators=(",", ":"))
        duplicate = (
            '{"protocol":"' + protocol
            + '","request_id":"a","request_id":"b",'
            '"operation":"describe-baseline-v1","payload":'
            '{"baseline_ecsv":"' + str(baseline_path) + '"}}\n'
        )
        unknown = copy.deepcopy(good)
        unknown["payload"]["unknown"] = True
        wrong_numeric = copy.deepcopy(good)
        wrong_numeric["payload"]["baseline_ecsv"] = 1
        negative_zero = copy.deepcopy(good)
        negative_zero["payload"]["expected_baseline"] = (
            product_contract.verified_baseline_reference(descriptor)
        )
        negative_zero["payload"]["expected_baseline"]["byte_count"] = "-0"
        requests = (
            duplicate,
            json.dumps(unknown, separators=(",", ":")) + "\n",
            json.dumps(wrong_numeric, separators=(",", ":")) + "\n",
            json.dumps(negative_zero, separators=(",", ":")) + "\n",
            compact + "{}\n",
            compact.replace(
                json.dumps(good["payload"], separators=(",", ":")),
                "NaN",
            ) + "\n",
            compact + "\n{}\n",
        )
        for request in requests:
            with self.subTest(request=request[:40]):
                completed, response = self.run_protocol(request)
                self.assertEqual(completed.returncode, 2)
                self.assertEqual(
                    set(response),
                    {"protocol", "request_id", "status", "error"},
                )
                self.assertEqual(response["status"], "error")
                self.assertEqual(response["error"]["category"], "protocol")

        missing_path = self.root / "missing-receipt.ecsv"
        missing_path.write_bytes(self.apt_bytes)
        missing = copy.deepcopy(good)
        missing["request_id"] = "missing-receipt"
        missing["payload"]["baseline_ecsv"] = str(missing_path)
        completed, response = self.run_protocol(missing)
        self.assertEqual(completed.returncode, 1)
        self.assertEqual(response["error"]["category"], "contract")

        tampered_path = self.root / "tampered-receipt.ecsv"
        tampered_path.write_bytes(self.apt_bytes)
        receipt = self.receipt_bytes.decode("ascii")
        Path(str(tampered_path) + ".sha256").write_text(
            receipt.replace(
                f"byte_count={len(self.apt_bytes)}",
                f"byte_count={len(self.apt_bytes) - 1}",
            ),
            encoding="ascii",
        )
        tampered = copy.deepcopy(good)
        tampered["request_id"] = "tampered-receipt"
        tampered["payload"]["baseline_ecsv"] = str(tampered_path)
        completed, response = self.run_protocol(tampered)
        self.assertEqual(completed.returncode, 1)
        self.assertEqual(response["error"]["category"], "contract")

    def test_versioned_protocol_rejects_new_v1_issuance(self) -> None:
        _verifier, request, _source_paths = self.protocol_issue_request()
        baseline_path = self.root / "baseline.ecsv"
        baseline_receipt_path = Path(str(baseline_path) + ".sha256")
        baseline_before = baseline_path.read_bytes()
        baseline_receipt_before = baseline_receipt_path.read_bytes()

        def assert_baseline_unchanged() -> None:
            self.assertEqual(baseline_path.read_bytes(), baseline_before)
            self.assertEqual(
                baseline_receipt_path.read_bytes(), baseline_receipt_before
            )

        completed, issued = self.run_protocol(request)
        self.assertEqual(completed.returncode, 1)
        self.assertEqual(issued["status"], "error")
        self.assertEqual(issued["error"]["category"], "contract")
        self.assertIn("v1 issuance is disabled", issued["error"]["message"])
        output_path = Path(request["payload"]["publication"]["output_ecsv"])
        self.assertFalse(output_path.exists())
        self.assertFalse(Path(str(output_path) + ".sha256").exists())
        assert_baseline_unchanged()

    def test_v1_issuance_rejects_before_private_structure_or_source_use(
        self,
    ) -> None:
        _verifier, request, _source_paths = self.protocol_issue_request()
        baseline_path = self.root / "baseline.ecsv"
        baseline_receipt_path = Path(str(baseline_path) + ".sha256")
        baseline_before = baseline_path.read_bytes()
        baseline_receipt_before = baseline_receipt_path.read_bytes()

        self.assertNotIn("schema_version", request["payload"]["target"])
        self.assertNotIn("registered_fields", request["payload"]["target"])
        self.assertNotIn("target_parent", request["payload"]["relation"])
        self.assertNotIn("baseline_parent", request["payload"]["relation"])
        self.assertEqual(
            set(request["payload"]["relation"]["pairs"][0]["target"]),
            {"occurrence", "local_key"},
        )

        request["payload"]["target"]["schema_version"] = "caller-forbidden"
        request["payload"]["target"]["inputs"][0]["raw_source"][
            "diagnostic_locator"
        ] = str(self.root / "does-not-exist")
        completed, rejected = self.run_protocol(request)
        self.assertEqual(completed.returncode, 1)
        self.assertEqual(rejected["error"]["category"], "contract")
        self.assertIn("v1 issuance is disabled", rejected["error"]["message"])
        output_path = Path(request["payload"]["publication"]["output_ecsv"])
        self.assertFalse(output_path.exists())
        self.assertFalse(Path(str(output_path) + ".sha256").exists())
        self.assertEqual(baseline_path.read_bytes(), baseline_before)
        self.assertEqual(
            baseline_receipt_path.read_bytes(), baseline_receipt_before
        )

class CanonicalAptObservationContractTest(unittest.TestCase):
    bits = staticmethod(CanonicalAptArtifactContractTest.bits)
    observation_fixture = CanonicalAptArtifactContractTest.observation_fixture
    BASELINE_ID = "apt-prod-001-canonical-baseline-apt-v1"
    SEMANTIC = CanonicalAptArtifactContractTest.SEMANTIC
    ENVELOPE = CanonicalAptArtifactContractTest.ENVELOPE
    TRANSPORT = CanonicalAptArtifactContractTest.TRANSPORT
    SUCCESSOR_IDS = (
        "apt-prod-002-observation-target-manifest-v1",
        "apt-prod-002-match-dispositions-v1",
        "apt-prod-002-observation-matched-apt-v1",
    )

    def setUp(self) -> None:
        self.registry_path = (
            Path(__file__).resolve().parents[2]
            / "validation/product_contracts.json"
        )
        self.apt_bytes = zlib.decompress(
            base64.b64decode(
                CanonicalAptArtifactContractTest.FIXTURE_ZLIB_BASE64
            )
        )
        self.receipt_bytes = (
            "citlali-canonical-apt-publication-receipt-v1\n"
            "scope=citlali-canonical-apt-byte-transport-sha256-v1\n"
            "envelope_sha256="
            + CanonicalAptArtifactContractTest.ENVELOPE
            + "\nbyte_sha256="
            + CanonicalAptArtifactContractTest.TRANSPORT
            + "\nbyte_count=18759\n"
        ).encode("ascii")

    def test_successor_contracts_are_exact_unactivated_and_unrouted(self) -> None:
        registry = product_contract.load_registry(self.registry_path)
        self.assertEqual(
            product_contract._canonical_json_sha256(
                registry["artifact_contracts"][self.BASELINE_ID]
            ),
            product_contract.CANONICAL_APT_ARTIFACT_CONTRACT_SHA256,
        )
        for contract_id in self.SUCCESSOR_IDS:
            with self.subTest(contract_id=contract_id):
                contract = product_contract.artifact_contract_by_id(
                    registry, contract_id
                )
                self.assertEqual(contract["activation_state"], "unactivated")
                self.assertEqual(contract["contract_authority"], "citlali")
                self.assertEqual(
                    contract["observation_value_issuer"], "tolproj"
                )
        target_contract = registry["artifact_contracts"][self.SUCCESSOR_IDS[0]]
        relation_contract = registry["artifact_contracts"][self.SUCCESSOR_IDS[1]]
        for logical in (target_contract, relation_contract):
            self.assertEqual(
                logical["persistence_state"], "embedded-logical-record-v1"
            )
            self.assertNotIn("artifact_suffix", logical)
            self.assertNotIn("byte_transport_scope", logical)
            self.assertNotIn("receipt_schema", logical)
        output_contract = registry["artifact_contracts"][self.SUCCESSOR_IDS[2]]
        self.assertEqual(
            output_contract["persistence_state"], "persisted-final-artifact-v1"
        )
        self.assertEqual(output_contract["artifact_suffix"], ".apt.ecsv")
        self.assertEqual(
            output_contract["embedded_logical_records"],
            list(self.SUCCESSOR_IDS[:2]),
        )
        envelope_authorities = {
            member["name"]: member["authority"]
            for member in output_contract["record_schemas"]
            ["issuance_envelope"]["members"]
        }
        self.assertEqual(
            envelope_authorities,
            {
                "occurrence": "citlali-canonical-issuer",
                "event_reference": "citlali-canonical-issuer",
                "software_revision": "citlali-build-authority",
                "configuration_reference": "tolproj-request",
                "event_time_utc": "tolproj-request",
            },
        )
        routed = json.dumps(
            {
                "families": registry["families"],
                "checks": registry["checks"],
                "contracts": registry["contracts"],
            },
            sort_keys=True,
        )
        for contract_id in self.SUCCESSOR_IDS:
            self.assertNotIn(contract_id, routed)

    def test_successor_cli_dispatch_is_explicitly_unactivated(self) -> None:
        missing_target = Path("/apt-prod-002-phase-a-must-not-be-read.ecsv")
        for contract_id in self.SUCCESSOR_IDS:
            stdout = io.StringIO()
            stderr = io.StringIO()
            with self.subTest(contract_id=contract_id), \
                    contextlib.redirect_stdout(stdout), \
                    contextlib.redirect_stderr(stderr):
                status = product_contract.main(
                    [
                        str(missing_target),
                        "--artifact-contract",
                        contract_id,
                        "--registry",
                        str(self.registry_path),
                    ]
                )
            self.assertEqual(status, 2)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn(
                "target/relation are embedded logical records",
                stderr.getvalue(),
            )

    def test_verified_descriptor_is_reconstructed_from_exact_bytes_and_receipt(
        self,
    ) -> None:
        registry = product_contract.load_registry(self.registry_path)
        baseline = product_contract.artifact_contract_by_id(
            registry, self.BASELINE_ID
        )
        descriptor = product_contract.verified_baseline_descriptor_from_bytes(
            self.apt_bytes, self.receipt_bytes, baseline
        )
        self.assertEqual(
            descriptor["schema_version"],
            "citlali-verified-beammap-baseline-descriptor-v1",
        )
        self.assertEqual(descriptor["contract_authority"], "citlali")
        self.assertEqual(descriptor["semantic_sha256"], self.SEMANTIC)
        self.assertEqual(descriptor["envelope_sha256"], self.ENVELOPE)
        self.assertEqual(descriptor["byte_sha256"], self.TRANSPORT)
        self.assertEqual(descriptor["byte_count"], "18759")
        self.assertEqual(len(descriptor["rows"]), 3)
        self.assertEqual(len(descriptor["raw_manifest"]), 2)
        self.assertEqual(
            set(descriptor["wire_presentation_sequence"]),
            {"0", "42", "9007199254740991"},
        )
        self.assertEqual(
            descriptor["receipt_sha256"],
            "sha256:" + hashlib.sha256(self.receipt_bytes).hexdigest(),
        )

        for artifact_bytes, receipt_bytes in (
            (self.apt_bytes.replace(b"Jupiter", b"Saturn", 1), self.receipt_bytes),
            (self.apt_bytes, self.receipt_bytes.replace(b"18759", b"1")),
        ):
            with self.subTest():
                with self.assertRaises(product_contract.ContractError):
                    product_contract.verified_baseline_descriptor_from_bytes(
                        artifact_bytes, receipt_bytes, baseline
                    )

    def test_successor_scalar_encoding_vectors_are_exact(self) -> None:
        frames = b"".join(
            [
                product_contract.canonical_observation_scalar_frame(
                    "i64-min", "int64", "-9223372036854775808"
                ),
                product_contract.canonical_observation_scalar_frame(
                    "u64-max", "uint64", "18446744073709551615"
                ),
                product_contract.canonical_observation_scalar_frame(
                    "negative-zero", "float64-ieee754", "8000000000000000"
                ),
                product_contract.canonical_observation_scalar_frame(
                    "denorm-min", "float64-ieee754", "0000000000000001"
                ),
                product_contract.canonical_observation_scalar_frame(
                    "canonical-nan", "float64-ieee754", "7ff8000000000000"
                ),
                product_contract.canonical_observation_scalar_frame(
                    "missing-float", "null-float64", "null"
                ),
                product_contract.canonical_observation_scalar_frame(
                    "text", "utf8", "Jupiter α"
                ),
            ]
        )
        self.assertEqual(
            hashlib.sha256(frames).hexdigest(),
            "a97e7c29a17da562d44108968d120c393428577f9f218154bc5147e8f32029ec",
        )
        invalid = [
            ("int64", "+1"),
            ("uint64", "-0"),
            ("float64-ieee754", "7FF8000000000000"),
            ("float64-ieee754", "7ff0000000000001"),
            ("null-float64", "NULL"),
            ("null-garbage", "null"),
            ("utf8", "bad\ntext"),
        ]
        for datatype, value in invalid:
            with self.subTest(datatype=datatype, value=value):
                with self.assertRaises(product_contract.ContractError):
                    product_contract.canonical_observation_scalar_frame(
                        "bad", datatype, value
                    )

    def test_restrictive_kmp_catalog_requires_complete_successor_api(self) -> None:
        registry = json.loads(self.registry_path.read_text(encoding="utf-8"))
        target = registry["artifact_contracts"][self.SUCCESSOR_IDS[0]]
        field_registry = target["field_authorization_registry"]
        self.assertEqual(
            [field["name"] for field in field_registry["registered_fields"]],
            ["kids_Qr", "kids_f_out", "kids_flag", "kids_fr"],
        )
        self.assertEqual(
            field_registry["required_field_names"],
            ["kids_Qr", "kids_f_out", "kids_fr"],
        )
        self.assertEqual(field_registry["optional_field_names"], ["kids_flag"])
        self.assertEqual(
            field_registry["authorized_use_roles"],
            product_contract.OBSERVATION_KMP_AUTHORIZED_USE_ROLES_V1,
        )
        observation_members = {
            member["name"]: member
            for member in target["record_schemas"]["observation_identity"][
                "members"
            ]
        }
        self.assertEqual(
            {member["authority"] for member in observation_members.values()},
            {"source-header"},
        )
        target_members = {
            member["name"]: member
            for member in target["record_schemas"]["target_manifest"][
                "members"
            ]
        }
        self.assertEqual(target_members["observation"]["authority"], "raw-header")
        output = registry["artifact_contracts"][self.SUCCESSOR_IDS[2]]
        transformation_members = {
            member["name"]: member
            for member in output["record_schemas"]["field_transformation"][
                "members"
            ]
        }
        self.assertEqual(
            transformation_members["source_pair_key"]["cardinality"],
            "baseline-matched-exactly-one-target-preserve-and-unmatched-zero",
        )
        self.assertEqual(
            transformation_members["authority_reference"]["authority"],
            "target-or-verified-baseline-field-catalog-or-citlali-typed-missing",
        )
        self.assertTrue(
            callable(product_contract.validate_observation_matched_apt_v1)
        )
        self.assertTrue(
            callable(product_contract.canonical_observation_matched_apt_preimage)
        )

    def test_full_successor_bundle_vectors_match_cpp(self) -> None:
        fixture = self.observation_fixture()
        contracts = fixture["contracts"]
        target_contract = contracts[self.SUCCESSOR_IDS[0]]
        relation_contract = contracts[self.SUCCESSOR_IDS[1]]
        output_contract = contracts[self.SUCCESSOR_IDS[2]]
        self.assertEqual(
            {
                key: fixture["descriptor"][key]
                for key in (
                    "semantic_sha256",
                    "envelope_sha256",
                    "byte_sha256",
                    "byte_count",
                    "receipt_sha256",
                    "receipt_byte_count",
                )
            },
            {
                "semantic_sha256": "sha256:8ac14aca51f660b015e6427483e05968d1443a33d812a28ef46ed027261f0a37",
                "envelope_sha256": "sha256:f44e40ae8604b85ea82f783212eb785561fbbd6b478ab1311f406bc63a1d2838",
                "byte_sha256": "sha256:b4cfecf45c611ba6378bd7b88d78978b8004aa0ee8db499367499c75db05f34b",
                "byte_count": "19327",
                "receipt_sha256": "sha256:536f689f3325e5a1d298a69bba277ef686971c97152034a1bb1bc861d1acbe30",
                "receipt_byte_count": "287",
            },
        )
        self.assertEqual(
            product_contract.baseline_descriptor_sha256(fixture["descriptor"]),
            "sha256:b801161d65dfea02b3c579ac5766154900b82e92de6945537caae2691d2707af",
        )
        self.assertEqual(
            product_contract.observation_target_digests(
                fixture["target"], target_contract
            ),
            {
                "semantic_sha256": "sha256:8ad86d382b31eed82deab3118bbd5efe1fc5ce41389eac561ad2aef7e24cb30b",
                "envelope_sha256": "sha256:3dca742ac86f93666762e33557ab91b4d061be178b936d17d379077233bd6fc5",
            },
        )
        self.assertEqual(
            product_contract.match_dispositions_digests(
                fixture["relation"], relation_contract,
                fixture["descriptor"], fixture["target"], target_contract,
            ),
            {
                "semantic_sha256": "sha256:7555c3f35ef57db23d32ef833d635c06cd06690ca1543cb635272328f29c93a4",
                "envelope_sha256": "sha256:25cd94197f41b3ec6132adfde525eeb1baae9b2dbeae7d47af38966817f5e8dc",
            },
        )
        self.assertEqual(
            product_contract.observation_matched_apt_digests(
                fixture["output"], output_contract, fixture["descriptor"],
                fixture["target"], target_contract, fixture["relation"],
                relation_contract,
            ),
            {
                "semantic_sha256": "sha256:cac3fabbb34907013b7558c5db855c3c861e370bb05ff0ff15051dd9f4e44dba",
                "envelope_sha256": "sha256:96fe37adc1b743dbcd7d907bb0f63b4859ff44102cc1be8556914f7978212dce",
            },
        )
        for contract_id in self.SUCCESSOR_IDS:
            self.assertEqual(
                product_contract._canonical_json_sha256(
                    contracts[contract_id]
                ),
                product_contract.OBSERVATION_ARTIFACT_CONTRACT_SHA256[
                    contract_id
                ],
            )

    def test_descriptor_revalidation_rejects_every_typed_mutation(self) -> None:
        descriptor = self.observation_fixture()["descriptor"]
        mutations = []
        changed = copy.deepcopy(descriptor)
        changed["rows"][0]["fields"]["amp"]["value"] = self.bits(123.0)
        mutations.append(changed)
        changed = copy.deepcopy(descriptor)
        changed["raw_manifest"][0]["channel_count"] = "3"
        mutations.append(changed)
        changed = copy.deepcopy(descriptor)
        changed["scientific_context"]["source_name"] = "Saturn β"
        mutations.append(changed)
        changed = copy.deepcopy(descriptor)
        changed["wire_presentation_sequence"].reverse()
        mutations.append(changed)
        changed = copy.deepcopy(descriptor)
        changed["envelope"]["event_reference"] = "event:forged"
        mutations.append(changed)
        changed = copy.deepcopy(descriptor)
        changed["occurrence"] = "occurrence:forged"
        mutations.append(changed)
        for changed in mutations:
            with self.subTest():
                with self.assertRaises(product_contract.ContractError):
                    product_contract.baseline_descriptor_sha256(changed)
        with self.assertRaises(product_contract.ContractError):
            product_contract.baseline_descriptor_sha256(dict(descriptor))

    def test_target_source_field_and_order_counterexamples(self) -> None:
        fixture = self.observation_fixture()
        target = fixture["target"]
        contract = fixture["contracts"][self.SUCCESSOR_IDS[0]]
        original = product_contract.observation_target_digests(target, contract)

        reordered = copy.deepcopy(target)
        reordered["inputs"].reverse()
        reordered["rows"].reverse()
        reordered["registered_fields"].reverse()
        self.assertEqual(
            product_contract.observation_target_digests(reordered, contract),
            original,
        )
        locator = copy.deepcopy(target)
        locator["inputs"][0]["kmp_source"]["diagnostic_locator"] = "other/path"
        locator_digests = product_contract.observation_target_digests(
            locator, contract
        )
        self.assertEqual(locator_digests["semantic_sha256"], original["semantic_sha256"])
        self.assertNotEqual(locator_digests["envelope_sha256"], original["envelope_sha256"])
        changed_source = copy.deepcopy(target)
        changed_source["inputs"][0]["kmp_source"]["content_sha256"] = (
            "sha256:" + "9" * 64
        )
        changed_source_digests = product_contract.observation_target_digests(
            changed_source, contract
        )
        self.assertNotEqual(
            changed_source_digests["semantic_sha256"],
            original["semantic_sha256"],
        )
        occurrence = copy.deepcopy(target)
        occurrence["envelope"]["occurrence"] = "occurrence:target/other"
        occurrence_digests = product_contract.observation_target_digests(
            occurrence, contract
        )
        self.assertEqual(
            occurrence_digests["semantic_sha256"], original["semantic_sha256"]
        )
        self.assertNotEqual(
            occurrence_digests["envelope_sha256"], original["envelope_sha256"]
        )
        sequence = copy.deepcopy(target)
        sequence["target_application_sequence"][0:2] = reversed(
            sequence["target_application_sequence"][0:2]
        )
        self.assertNotEqual(
            product_contract.observation_target_digests(sequence, contract)[
                "semantic_sha256"
            ],
            original["semantic_sha256"],
        )

        mutations = []
        changed = copy.deepcopy(target)
        changed["rows"][0]["kmp_source_key"] = "101"
        mutations.append(changed)
        changed = copy.deepcopy(target)
        changed["rows"][0]["kmp_row_index"] = "1"
        mutations.append(changed)
        changed = copy.deepcopy(target)
        changed["rows"][0]["matching_frequency_hz"] = self.bits(1.0)
        mutations.append(changed)
        changed = copy.deepcopy(target)
        changed["rows"][0]["fields"]["kids_Qr"] = "7ff0000000000000"
        mutations.append(changed)
        changed = copy.deepcopy(target)
        changed["rows"][0]["fields"]["rogue"] = self.bits(1.0)
        mutations.append(changed)
        changed = copy.deepcopy(target)
        changed["rows"].pop()
        mutations.append(changed)
        changed = copy.deepcopy(target)
        changed["rows"][1]["row_key"] = changed["rows"][0]["row_key"]
        mutations.append(changed)
        changed = copy.deepcopy(target)
        changed["rows"][0]["row_key"] = str(
            product_contract.CANONICAL_APT_UID_MAX + 1
        )
        mutations.append(changed)
        changed = copy.deepcopy(target)
        changed["rows"][1]["channel"] = changed["rows"][2]["channel"]
        changed["rows"][1]["kmp_row_index"] = changed["rows"][2][
            "kmp_row_index"
        ]
        mutations.append(changed)
        changed = copy.deepcopy(target)
        changed["registered_fields"][0]["source_column"] = "f_out"
        mutations.append(changed)
        for member, value in (
            ("datatype", "int64"),
            ("unit", "N/A"),
            ("nullable", True),
            ("authority", "tolproj"),
            ("authority_reference", "caller:self"),
        ):
            changed = copy.deepcopy(target)
            changed["registered_fields"][0][member] = value
            mutations.append(changed)
        changed = copy.deepcopy(target)
        changed["registered_fields"][0]["identity_role"] = "identity"
        mutations.append(changed)
        changed = copy.deepcopy(target)
        changed["inputs"][0]["kmp_source"]["channel_count"] = "2"
        mutations.append(changed)
        changed = copy.deepcopy(target)
        changed["inputs"][0]["raw_source"]["header_observation"]["scan"] = "2"
        mutations.append(changed)
        changed = copy.deepcopy(target)
        changed["target_source_sequence"][0] = changed[
            "target_source_sequence"
        ][1]
        mutations.append(changed)
        changed = copy.deepcopy(target)
        changed["target_application_sequence"].pop()
        mutations.append(changed)
        changed = copy.deepcopy(target)
        changed["rows"][0]["fields"]["kids_flag"] = "9223372036854775808"
        mutations.append(changed)
        for changed in mutations:
            with self.subTest():
                with self.assertRaises(product_contract.ContractError):
                    product_contract.validate_observation_target_manifest_v1(
                        changed, contract
                    )
        kmp_fallback = copy.deepcopy(target)
        kmp_fallback["inputs"][0]["kmp_source"]["header_observation"] = {
            "observation": "1", "subobservation": "2", "scan": "3"
        }
        product_contract.validate_observation_target_manifest_v1(
            kmp_fallback, contract
        )
        self.assertEqual(
            [row["fields"]["kids_flag"] for row in target["rows"]],
            ["-7", "3", "42"],
        )

    def test_kmp_source_boundary_is_closed_without_diagnostic_bag(self) -> None:
        available = [
            "fr", "f_out", "Qr", "flag", "diagnostic_chi2", "kids_flag"
        ]
        requested = {
            "identity": [],
            "matching": ["kids_fr", "kids_Qr"],
            "application": ["kids_f_out"],
            "transformation": [],
            "output": ["kids_fr", "kids_f_out", "kids_Qr", "kids_flag"],
            "authority": ["kids_fr", "kids_f_out", "kids_Qr", "kids_flag"],
        }
        self.assertEqual(
            product_contract.validate_kmp_source_column_boundary_v1(
                available, requested
            ),
            ("kids_Qr", "kids_f_out", "kids_flag", "kids_fr"),
        )
        no_flag = copy.deepcopy(requested)
        no_flag["output"].remove("kids_flag")
        no_flag["authority"].remove("kids_flag")
        self.assertEqual(
            product_contract.validate_kmp_source_column_boundary_v1(
                ["fr", "f_out", "Qr", "unrelated", "kids_flag"], no_flag
            ),
            ("kids_Qr", "kids_f_out", "kids_fr"),
        )
        bad_requests = []
        changed = copy.deepcopy(requested)
        changed["identity"] = ["diagnostic_chi2"]
        bad_requests.append(changed)
        changed = copy.deepcopy(requested)
        changed["matching"] = ["kids_f_out"]
        bad_requests.append(changed)
        changed = copy.deepcopy(requested)
        changed["application"] = ["diagnostic_chi2"]
        bad_requests.append(changed)
        changed = copy.deepcopy(requested)
        changed["transformation"] = ["kids_fr"]
        bad_requests.append(changed)
        changed = copy.deepcopy(requested)
        changed["output"] = ["diagnostic_chi2"]
        bad_requests.append(changed)
        changed = copy.deepcopy(requested)
        changed["authority"] = ["diagnostic_chi2"]
        bad_requests.append(changed)
        for changed in bad_requests:
            with self.subTest():
                with self.assertRaises(product_contract.ContractError):
                    product_contract.validate_kmp_source_column_boundary_v1(
                        available, changed
                    )
        with self.assertRaises(product_contract.ContractError):
            product_contract.validate_kmp_source_column_boundary_v1(
                ["kids_fr", "kids_f_out", "kids_Qr"], requested
            )

    def test_relation_coverage_cardinality_and_evidence_counterexamples(self) -> None:
        fixture = self.observation_fixture()
        contracts = fixture["contracts"]
        target_contract = contracts[self.SUCCESSOR_IDS[0]]
        relation_contract = contracts[self.SUCCESSOR_IDS[1]]
        args = (
            relation_contract, fixture["descriptor"], fixture["target"],
            target_contract,
        )
        original = product_contract.match_dispositions_digests(
            fixture["relation"], *args
        )
        reordered = copy.deepcopy(fixture["relation"])
        for name in (
            "network_evidence", "pairs", "target_dispositions",
            "seed_dispositions",
        ):
            reordered[name].reverse()
        self.assertEqual(
            product_contract.match_dispositions_digests(reordered, *args),
            original,
        )
        self.assertEqual(
            next(
                item for item in fixture["relation"]["target_dispositions"]
                if item["endpoint"]["local_key"] == "5"
            )["pair_keys"],
            ["900", "901"],
        )
        self.assertEqual(
            next(
                item for item in fixture["relation"]["seed_dispositions"]
                if item["endpoint"]["local_key"] == "42"
            )["pair_keys"],
            ["901", "902"],
        )
        mutations = []
        changed = copy.deepcopy(fixture["relation"])
        changed["target_dispositions"].pop()
        mutations.append(changed)
        changed = copy.deepcopy(fixture["relation"])
        changed["seed_dispositions"][0]["disposition_key"] = "900"
        mutations.append(changed)
        changed = copy.deepcopy(fixture["relation"])
        changed["target_dispositions"][1]["pair_keys"] = ["900"]
        mutations.append(changed)
        changed = copy.deepcopy(fixture["relation"])
        changed["pairs"][0]["target"] = changed["pairs"][1]["target"]
        mutations.append(changed)
        changed = copy.deepcopy(fixture["relation"])
        changed["network_evidence"][0]["quality_factor"] = "7ff8000000000000"
        mutations.append(changed)
        changed = copy.deepcopy(fixture["relation"])
        changed["network_evidence"][0]["quality_factor_authority_reference"] = "caller:self"
        mutations.append(changed)
        changed = copy.deepcopy(fixture["relation"])
        changed["matcher"]["target_frequency_field"] = "kids_f_out"
        mutations.append(changed)
        changed = copy.deepcopy(fixture["relation"])
        changed["envelope"]["occurrence"] = fixture["target"]["envelope"]["occurrence"]
        mutations.append(changed)
        changed = copy.deepcopy(fixture["relation"])
        changed["seed_source_sequence"][0] = changed[
            "seed_source_sequence"
        ][1]
        mutations.append(changed)
        for changed in mutations:
            with self.subTest():
                with self.assertRaises(product_contract.ContractError):
                    product_contract.validate_match_dispositions_v1(
                        changed, *args
                    )

    def test_output_union_collision_and_transformation_counterexamples(self) -> None:
        fixture = self.observation_fixture()
        contracts = fixture["contracts"]
        args = (
            contracts[self.SUCCESSOR_IDS[2]], fixture["descriptor"],
            fixture["target"], contracts[self.SUCCESSOR_IDS[0]],
            fixture["relation"], contracts[self.SUCCESSOR_IDS[1]],
        )
        original = product_contract.observation_matched_apt_digests(
            fixture["output"], *args
        )
        reordered = copy.deepcopy(fixture["output"])
        reordered["rows"].reverse()
        reordered["registered_fields"].reverse()
        for row in reordered["rows"]:
            row["transformations"].reverse()
        self.assertEqual(
            product_contract.observation_matched_apt_digests(reordered, *args),
            original,
        )
        field_names = {
            item["field"]["name"]
            for item in fixture["output"]["registered_fields"]
        }
        self.assertIn("kids_flag", field_names)
        self.assertEqual(
            [row["fields"]["kids_flag"] for row in fixture["output"]["rows"]],
            ["-7", "42", "3"],
        )
        self.assertNotEqual(
            fixture["output"]["rows"][1]["fields"]["kids_flag"], "91"
        )
        optional = self.observation_fixture(include_target_kids_flag=False)
        self.assertNotIn(
            "kids_flag",
            {
                item["field"]["name"]
                for item in optional["output"]["registered_fields"]
            },
        )

        def transformation(row: dict[str, object], name: str) -> dict[str, object]:
            return next(
                item for item in row["transformations"]
                if item["field_name"] == name
            )

        mutations = []
        changed = copy.deepcopy(fixture["output"])
        changed["rows"].pop()
        mutations.append(changed)
        changed = copy.deepcopy(fixture["output"])
        changed["rows"][0]["tone_frequency_hz"] = self.bits(1.0)
        mutations.append(changed)
        changed = copy.deepcopy(fixture["output"])
        changed["rows"][1]["relation_pair_keys"] = ["900"]
        mutations.append(changed)
        changed = copy.deepcopy(fixture["output"])
        transformation(changed["rows"][1], "kids_Qr")["operation"] = "issuer-declared"
        mutations.append(changed)
        changed = copy.deepcopy(fixture["output"])
        change = transformation(changed["rows"][1], "kids_flag")
        changed["rows"][1]["fields"]["kids_flag"] = "91"
        change["after"] = "91"
        mutations.append(changed)
        changed = copy.deepcopy(fixture["output"])
        transformation(changed["rows"][1], "amp")["source_pair_key"] = "901"
        mutations.append(changed)
        changed = copy.deepcopy(fixture["output"])
        change = transformation(changed["rows"][0], "amp")
        change["after"] = self.bits(1.0)
        changed["rows"][0]["fields"]["amp"] = self.bits(1.0)
        mutations.append(changed)
        changed = copy.deepcopy(fixture["output"])
        transformation(changed["rows"][1], "kids_fr")[
            "provenance_reference"
        ] = "target-kmp-source:101:row:0:column:rogue"
        mutations.append(changed)
        changed = copy.deepcopy(fixture["output"])
        changed["envelope"]["occurrence"] = fixture["relation"]["envelope"]["occurrence"]
        mutations.append(changed)
        changed = copy.deepcopy(fixture["output"])
        changed["rows"][1]["uid"] = changed["rows"][0]["uid"]
        mutations.append(changed)
        changed = copy.deepcopy(fixture["output"])
        changed["output_presentation_sequence"].pop()
        mutations.append(changed)
        for changed in mutations:
            with self.subTest():
                with self.assertRaises(product_contract.ContractError):
                    product_contract.validate_observation_matched_apt_v1(
                        changed, *args
                    )

    def test_final_matched_ecsv_transport_matches_cpp_and_revalidates(self) -> None:
        fixture = self.observation_fixture()
        contracts = fixture["contracts"]
        args = (
            fixture["output"], contracts[self.SUCCESSOR_IDS[2]],
            fixture["descriptor"], fixture["target"],
            contracts[self.SUCCESSOR_IDS[0]], fixture["relation"],
            contracts[self.SUCCESSOR_IDS[1]],
        )
        serialized = (
            product_contract.serialize_observation_matched_apt_ecsv_v1(*args)
        )
        self.assertEqual(
            serialized["semantic_sha256"],
            "sha256:cac3fabbb34907013b7558c5db855c3c861e370bb05ff0ff15051dd9f4e44dba",
        )
        self.assertEqual(
            serialized["envelope_sha256"],
            "sha256:96fe37adc1b743dbcd7d907bb0f63b4859ff44102cc1be8556914f7978212dce",
        )
        self.assertEqual(
            serialized["byte_sha256"],
            "sha256:a4016feb82b2d7b007ea6ae3dbbfbbf18022f25f467e50bb0fd324552bff6ded",
        )
        self.assertEqual(serialized["byte_count"], 125302)
        self.assertEqual(
            serialized["receipt_sha256"],
            "sha256:fa48cca9fc8218712ac0be2e3e86bd9ed2dbd3877d2af436677c880c9a90e1e8",
        )
        self.assertEqual(serialized["receipt_byte_count"], 298)
        reparsed = product_contract.validate_observation_matched_apt_ecsv_bytes_v1(
            serialized["bytes"], serialized["receipt_bytes"], *args
        )
        self.assertEqual(reparsed["bytes"], serialized["bytes"])

        reordered = self.observation_fixture()
        reordered["target"]["inputs"].reverse()
        reordered["target"]["registered_fields"].reverse()
        reordered["target"]["rows"].reverse()
        reordered["relation"]["network_evidence"].reverse()
        reordered["relation"]["pairs"].reverse()
        reordered["relation"]["target_dispositions"].reverse()
        reordered["relation"]["seed_dispositions"].reverse()
        reordered["output"]["registered_fields"].reverse()
        reordered["output"]["rows"].reverse()
        for row in reordered["output"]["rows"]:
            row["transformations"].reverse()
        reordered_args = (
            reordered["output"],
            reordered["contracts"][self.SUCCESSOR_IDS[2]],
            reordered["descriptor"], reordered["target"],
            reordered["contracts"][self.SUCCESSOR_IDS[0]],
            reordered["relation"],
            reordered["contracts"][self.SUCCESSOR_IDS[1]],
        )
        self.assertEqual(
            product_contract.serialize_observation_matched_apt_ecsv_v1(
                *reordered_args
            )["bytes"],
            serialized["bytes"],
        )

        tampered = serialized["bytes"].replace(
            b"fixture/caf\xc3\xa9/toltec7", b"fixture/caf\xc3\xa9/toltecX", 1
        )
        self.assertNotEqual(tampered, serialized["bytes"])
        with self.assertRaisesRegex(
            product_contract.ContractError, "tampered, stale, reordered"
        ):
            product_contract.validate_observation_matched_apt_ecsv_bytes_v1(
                tampered, serialized["receipt_bytes"], *args
            )
        bad_receipt = serialized["receipt_bytes"].replace(
            b"byte_count=125302", b"byte_count=125303", 1
        )
        with self.assertRaisesRegex(
            product_contract.ContractError, "receipt is tampered"
        ):
            product_contract.validate_observation_matched_apt_ecsv_bytes_v1(
                serialized["bytes"], bad_receipt, *args
            )

    def test_record_schema_references_and_catalogs_fail_closed(self) -> None:
        registry = json.loads(self.registry_path.read_text(encoding="utf-8"))
        target_id = self.SUCCESSOR_IDS[0]
        target = copy.deepcopy(registry["artifact_contracts"][target_id])
        target["record_schemas"]["target_manifest"]["members"][5][
            "datatype"
        ] = "list:missing_record"
        old = product_contract.OBSERVATION_ARTIFACT_CONTRACT_SHA256[target_id]
        try:
            product_contract.OBSERVATION_ARTIFACT_CONTRACT_SHA256[target_id] = (
                product_contract._canonical_json_sha256(target)
            )
            with self.assertRaisesRegex(
                product_contract.ContractError, "unresolved record"
            ):
                product_contract._validate_observation_artifact_contract(
                    target_id, target
                )
        finally:
            product_contract.OBSERVATION_ARTIFACT_CONTRACT_SHA256[target_id] = old
        target = copy.deepcopy(registry["artifact_contracts"][target_id])
        target["field_authorization_registry"]["registered_fields"][0][
            "authority"
        ] = "caller-self-authorized"
        try:
            product_contract.OBSERVATION_ARTIFACT_CONTRACT_SHA256[target_id] = (
                product_contract._canonical_json_sha256(target)
            )
            with self.assertRaisesRegex(
                product_contract.ContractError, "closed v1 KMP catalog"
            ):
                product_contract._validate_observation_artifact_contract(
                    target_id, target
                )
        finally:
                product_contract.OBSERVATION_ARTIFACT_CONTRACT_SHA256[target_id] = old


class CanonicalAptCompactV2ContractTest(unittest.TestCase):
    BASELINE_ID = "apt-prod-003-canonical-baseline-bundle-v2"
    MATCHED_ID = "apt-prod-003-observation-matched-bundle-v2"

    def setUp(self) -> None:
        self.registry_path = (
            Path(__file__).resolve().parents[2]
            / "validation/product_contracts.json"
        )

    def test_compact_contract_is_exact_unactivated_and_v1_is_bounded(self) -> None:
        registry = product_contract.load_registry(self.registry_path)
        self.assertEqual(
            product_contract._canonical_json_sha256(
                registry["apt_contract_lifecycle"]
            ),
            product_contract.COMPACT_APT_LIFECYCLE_SHA256,
        )
        self.assertEqual(
            registry["apt_contract_lifecycle"]["new_v1_issuance"],
            "forbidden",
        )
        self.assertEqual(
            registry["apt_contract_lifecycle"]["v1_guardian_default"],
            "reject",
        )
        self.assertEqual(
            product_contract._canonical_json_sha256(
                registry["apt_compact_v2_shared_contract"]
            ),
            product_contract.COMPACT_APT_SHARED_CONTRACT_SHA256,
        )
        shared = registry["apt_compact_v2_shared_contract"]
        self.assertEqual(
            shared["relation_metadata"]["network_evidence_statuses"],
            ["matched-capable", "missing-baseline-network", "no-good-seed"],
        )
        self.assertEqual(
            shared["target_logical_record"]["kids_flag_presence"],
            "artifact-optional-all-rows-or-none",
        )
        for contract_id in (self.BASELINE_ID, self.MATCHED_ID):
            with self.subTest(contract_id=contract_id):
                contract = product_contract.artifact_contract_by_id(
                    registry, contract_id
                )
                self.assertEqual(contract["activation_state"], "unactivated")
                self.assertEqual(contract["contract_authority"], "citlali")
                self.assertEqual(
                    product_contract._canonical_json_sha256(contract),
                    product_contract.COMPACT_APT_CONTRACT_SHA256[contract_id],
                )
        matched = registry["artifact_contracts"][self.MATCHED_ID]
        self.assertEqual(
            [field["name"] for field in matched["kmp_field_registry"]],
            ["kids_fr", "kids_f_out", "kids_Qr", "kids_flag"],
        )
        self.assertEqual(matched["hard_size_limit_bytes"], 20 * 1024 * 1024)
        routed = json.dumps(
            {
                "families": registry["families"],
                "checks": registry["checks"],
                "contracts": registry["contracts"],
            },
            sort_keys=True,
        )
        self.assertNotIn(self.BASELINE_ID, routed)
        self.assertNotIn(self.MATCHED_ID, routed)

    def test_compact_contract_drift_and_route_injection_fail_closed(self) -> None:
        registry = json.loads(self.registry_path.read_text(encoding="utf-8"))
        cases = []
        lifecycle = copy.deepcopy(registry)
        lifecycle["apt_contract_lifecycle"]["new_v1_issuance"] = "allowed"
        cases.append(("lifecycle", lifecycle, "lifecycle drift"))
        shared = copy.deepcopy(registry)
        shared["apt_compact_v2_shared_contract"]["relation_columns"].pop()
        cases.append(("shared", shared, "shared contract drift"))
        network_status = copy.deepcopy(registry)
        network_status["apt_compact_v2_shared_contract"][
            "relation_metadata"
        ]["network_evidence_statuses"].append("fabricated")
        cases.append(("network-status", network_status, "shared contract drift"))
        target = copy.deepcopy(registry)
        target["apt_compact_v2_shared_contract"]["target_logical_record"][
            "tone_policy"
        ] = "derive from row order"
        cases.append(("target-logical", target, "shared contract drift"))
        kmp = copy.deepcopy(registry)
        kmp["artifact_contracts"][self.MATCHED_ID]["kmp_field_registry"][0][
            "name"
        ] = "kids_unknown"
        cases.append(("kmp", kmp, "compact v2 contract drift"))
        role = copy.deepcopy(registry)
        role["artifact_contracts"][self.MATCHED_ID]["required_roles"].pop()
        cases.append(("role", role, "compact v2 contract drift"))
        routed = copy.deepcopy(registry)
        routed["contracts"][0]["description"] = self.MATCHED_ID
        cases.append(("route", routed, "referenced by a reduction"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for label, value, message in cases:
                with self.subTest(label=label):
                    path = root / f"{label}.json"
                    path.write_text(json.dumps(value), encoding="utf-8")
                    with self.assertRaisesRegex(
                        product_contract.ContractError, message
                    ):
                        product_contract.load_registry(path)

    def test_compact_component_lexical_vector_roundtrips_exactly(self) -> None:
        document = {
            "schema": "citlali-canonical-apt-v2-lexical-vector-v1",
            "role": "vector",
            "product_kind": "observation-matched",
            "issuance": {
                "occurrence": "urn:citlali:occurrence:vector",
                "event_reference": "urn:citlali:event:vector",
                "producer": "citlali",
                "software_revision": "fixture-revision",
                "configuration_reference": "sha256:" + "11" * 32,
                "event_time_utc": "2026-08-19T12:34:56.123Z",
            },
            "observation": {
                "obsnum": 148669,
                "subobsnum": 0,
                "scannum": 2,
            },
            "metadata": {"alpha": "α", "empty": ""},
            "columns": [
                {"name": "i", "datatype": "int64", "unit": "N/A", "nullable": False},
                {"name": "u", "datatype": "uint64", "unit": "byte", "nullable": False},
                {"name": "f", "datatype": "float64", "unit": "Hz", "nullable": False},
                {"name": "b", "datatype": "bool", "unit": "N/A", "nullable": False},
                {"name": "s", "datatype": "string", "unit": "N/A", "nullable": False},
                {"name": "n", "datatype": "float64", "unit": "N/A", "nullable": True},
            ],
            "rows": [
                [-(2**63), 2**64 - 1, -0.0, True, "café", None],
                [
                    2**63 - 1,
                    0,
                    struct.unpack(">d", (1).to_bytes(8, "big"))[0],
                    False,
                    "alpha",
                    float("nan"),
                ],
            ],
        }
        serialized = product_contract._serialize_compact_v2_component(document)
        self.assertEqual(
            serialized["semantic_sha256"],
            "sha256:439f6582412e428f98db3094d1e4dd6dcc0e7afea15727171299b40afae48db4",
        )
        self.assertEqual(
            serialized["envelope_sha256"],
            "sha256:2c53758ea4674368420f2643a8c9c46a2e969beea1d5e1f52ae6878e37206876",
        )
        self.assertEqual(
            serialized["transport_sha256"],
            "sha256:60dccb8e8c859fce779dcddfd46a9c022e7fb3a76bba5bde67a946336bcce8e8",
        )
        self.assertEqual(serialized["byte_count"], 1729)
        parsed = product_contract._parse_compact_v2_component_bytes(
            serialized["bytes"]
        )
        self.assertEqual(parsed["bytes"], serialized["bytes"])
        self.assertEqual(parsed["document"]["rows"][0][2], -0.0)
        self.assertLess(
            math.copysign(1.0, parsed["document"]["rows"][0][2]), 0.0
        )
        self.assertEqual(
            struct.pack(">d", parsed["document"]["rows"][1][2]).hex(),
            "0000000000000001",
        )
        self.assertTrue(math.isnan(parsed["document"]["rows"][1][5]))
        for tampered in (
            serialized["bytes"].replace(b'"vector"', b'"vectors"', 1),
            serialized["bytes"].replace(b',-0,true,', b',0,true,', 1),
            serialized["bytes"].replace(b'"alpha"', b'alpha', 1),
            serialized["bytes"].replace(b'"alpha"', b'""', 1),
            serialized["bytes"].replace(b"\n", b"\r\n", 1),
        ):
            with self.subTest(tampered=tampered[:32]):
                with self.assertRaises(product_contract.ContractError):
                    product_contract._parse_compact_v2_component_bytes(tampered)

    def test_compact_target_identity_vector_and_closed_relations(self) -> None:
        digest = "sha256:" + "11" * 32
        target = {
            "issuance": {
                "occurrence": "o",
                "event_reference": "e",
                "producer": "p",
                "software_revision": "s",
                "configuration_reference": digest,
                "event_time_utc": "2026-08-19T00:00:00.000Z",
            },
            "observation": {"obsnum": 1, "subobsnum": 0, "scannum": 0},
            "sources": [
                {
                    "source_uid": 0,
                    "role": "raw",
                    "content_sha256": digest,
                    "byte_count": 1,
                    "header_observation": {
                        "obsnum": 1,
                        "subobsnum": 0,
                        "scannum": 0,
                    },
                    "network": 0,
                    "interface": "toltec0",
                    "channel_count": 1,
                },
                {
                    "source_uid": 1,
                    "role": "kmp",
                    "content_sha256": digest,
                    "byte_count": 1,
                    "header_observation": {
                        "obsnum": 2,
                        "subobsnum": 0,
                        "scannum": 1,
                    },
                    "network": 0,
                    "interface": "toltec0",
                    "channel_count": 1,
                },
            ],
            "rows": [{
                "uid": 0,
                "input_uid": 0,
                "raw_source_uid": 0,
                "kmp_source_uid": 1,
                "kmp_row_index": 0,
                "source_rank": 0,
                "application_rank": 0,
                "tone_frequency_hz": 2.0,
                "array": 0,
                "network": 0,
                "channel": 0,
                "fields": {
                    "kids_fr": 1.0,
                    "kids_f_out": 2.0,
                    "kids_Qr": 3.0,
                },
            }],
        }
        self.assertEqual(
            product_contract._compact_v2_target_identity(target),
            {
                "schema": "citlali-observation-target-manifest-v2",
                "occurrence": "o",
                "semantic_sha256": (
                    "sha256:c031af9cad8683860f023e2bfb3de1fb601cdac3ad29c86e"
                    "95f7717def27df56"
                ),
                "envelope_sha256": (
                    "sha256:4445ecf32b33652ad50d926ebcdc8d301daf87d216746150e"
                    "db303cd7c416023"
                ),
            },
        )
        for label, mutator in (
            (
                "foreign-raw-observation",
                lambda value: value["sources"][0]["header_observation"].update(
                    obsnum=2
                ),
            ),
            (
                "KMP-row-position",
                lambda value: value["rows"][0].update(kmp_row_index=1),
            ),
            (
                "tone-alias",
                lambda value: value["rows"][0].update(tone_frequency_hz=4.0),
            ),
        ):
            changed = copy.deepcopy(target)
            mutator(changed)
            with self.subTest(label=label):
                with self.assertRaises(product_contract.ContractError):
                    product_contract._compact_v2_target_identity(changed)


if __name__ == "__main__":
    unittest.main()
