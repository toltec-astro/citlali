import base64
import contextlib
import copy
import hashlib
import io
import json
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
        "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"
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


if __name__ == "__main__":
    unittest.main()
