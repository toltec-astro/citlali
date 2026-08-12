import copy
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from tools.baseline import validate_product_contract as product_contract
from tools.baseline.test_audit_reduction_run import (
    valid_raw_v3_document,
    valid_raw_v4_document,
    write_raw_v4_observation,
)


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

    def test_checked_in_contracts_split_historical_and_current_admission(
        self,
    ) -> None:
        repository = Path(__file__).resolve().parents[2]
        registry = product_contract.load_registry(
            repository / "validation/product_contracts.json"
        )
        fixture = repository / (
            "tools/baseline/examples/sci_cal_001_selected_calibration_apt.ecsv"
        )
        self.assertTrue(fixture.is_file())
        for contract in registry["contracts"]:
            selected_entries = [
                entry for entry in contract["entries"]
                if entry["pattern"] == "{obs}/selected_calibration_apt.ecsv"
            ]
            provenance_entries = [
                entry for entry in contract["entries"]
                if entry["pattern"] == "{obs}/raw_timestream_provenance.yaml"
            ]
            if contract["contract_id"].startswith("sci-cal-001-current-"):
                self.assertEqual(
                    contract["admission_scope"],
                    "current_production_candidate",
                )
                self.assertEqual(len(selected_entries), 1)
                self.assertEqual(
                    selected_entries[0]["classification"],
                    "optional_diagnostic",
                )
                self.assertEqual(len(provenance_entries), 1)
                self.assertEqual(
                    provenance_entries[0]["checks"],
                    {
                        "schema_version": (
                            "citlali-raw-timestream-provenance-v4"
                        ),
                        "current_calibration_package": True,
                    },
                )
            else:
                self.assertFalse(selected_entries, contract["contract_id"])
                self.assertFalse(provenance_entries, contract["contract_id"])

    def test_current_contract_yaml_check_rejects_legacy_and_accepts_exact_v4(
        self,
    ) -> None:
        checks = {
            "schema_version": "citlali-raw-timestream-provenance-v4",
            "current_calibration_package": True,
        }
        observation = self.reduction / "000042"
        observation.mkdir()
        legacy_path = observation / "raw_timestream_provenance.yaml"
        legacy_path.write_text(
            yaml.safe_dump(valid_raw_v3_document(), sort_keys=False),
            encoding="utf-8",
        )
        self.assertTrue(any(
            "expected 'citlali-raw-timestream-provenance-v4'" in error
            for error in product_contract.validate_file(
                legacy_path, checks, ["a1100", "a1400", "a2000"]
            )
        ))

        legacy_path.unlink()
        observation.rmdir()
        current_path = write_raw_v4_observation(
            self.reduction, "000042", valid_raw_v4_document("000042")
        )
        self.assertEqual(
            product_contract.validate_file(
                current_path, checks, ["a1100", "a1400", "a2000"]
            ),
            [],
        )

    def test_current_contract_yaml_check_owns_uncalibrated_membership(self) -> None:
        checks = {
            "schema_version": "citlali-raw-timestream-provenance-v4",
            "current_calibration_package": True,
        }
        document = valid_raw_v4_document("000042")
        document["effective"]["config"]["flux_calibration"]["enabled"] = False
        document["calibration_lineage"] = {"available": False}
        for section in (document["observation"]["value"], document["realized"]):
            section["calibration_identity"] = {"available": False}
            section["calibration_package_identity"] = {"available": False}
        path = write_raw_v4_observation(
            self.reduction, "000042", document, write_member=False
        )
        self.assertEqual(
            product_contract.validate_file(
                path, checks, ["a1100", "a1400", "a2000"]
            ),
            [],
        )
        (path.parent / "selected_calibration_apt.ecsv").write_bytes(
            (
                Path(__file__).parent
                / "examples/sci_cal_001_selected_calibration_apt.ecsv"
            ).read_bytes()
        )
        self.assertIn(
            "uncalibrated v4 unexpectedly publishes a selected APT member",
            product_contract.validate_file(
                path, checks, ["a1100", "a1400", "a2000"]
            ),
        )

    def test_selected_calibration_contract_accepts_single_and_multi_layouts(
        self,
    ) -> None:
        fixture = Path(__file__).parent / (
            "examples/sci_cal_001_selected_calibration_apt.ecsv"
        )
        entry = self.entry(
            scope="per_observation",
            classification="config_conditional",
            condition="effective calibration",
            required_when={"path": "calibration.enabled", "equals": True},
            pattern="{obs}/selected_calibration_apt.ecsv",
        )
        for observation_count in (1, 2):
            with self.subTest(observation_count=observation_count):
                for child in list(self.reduction.iterdir()):
                    if child.is_dir():
                        for member in child.iterdir():
                            member.unlink()
                        child.rmdir()
                for index in range(observation_count):
                    observation = self.reduction / f"{42 + index:06d}"
                    observation.mkdir()
                    (observation / "selected_calibration_apt.ecsv").write_bytes(
                        fixture.read_bytes()
                    )

                result = self.validate(
                    [entry], {"calibration": {"enabled": True}}
                )

                self.assertTrue(result["passed"], result["errors"])
                self.assertEqual(
                    result["classified_product_count"], observation_count
                )

    def test_selected_calibration_contract_rejects_partial_publication(
        self,
    ) -> None:
        fixture = Path(__file__).parent / (
            "examples/sci_cal_001_selected_calibration_apt.ecsv"
        )
        for obsnum in ("000042", "000043"):
            (self.reduction / obsnum).mkdir()
        (self.reduction / "000042/selected_calibration_apt.ecsv").write_bytes(
            fixture.read_bytes()
        )
        entry = self.entry(
            scope="per_observation",
            classification="config_conditional",
            condition="effective calibration",
            required_when={"path": "calibration.enabled", "equals": True},
            pattern="{obs}/selected_calibration_apt.ecsv",
        )

        result = self.validate(
            [entry], {"calibration": {"enabled": True}}
        )

        self.assertFalse(result["passed"])
        self.assertTrue(any(
            "000043/selected_calibration_apt.ecsv" in error
            and "requires at least 1" in error
            for error in result["errors"]
        ))

    def test_selected_calibration_contract_accepts_uncalibrated_absence_only(
        self,
    ) -> None:
        observation = self.reduction / "000042"
        observation.mkdir()
        entry = self.entry(
            scope="per_observation",
            classification="config_conditional",
            condition="effective calibration",
            required_when={"path": "calibration.enabled", "equals": True},
            pattern="{obs}/selected_calibration_apt.ecsv",
        )

        result = self.validate(
            [entry], {"calibration": {"enabled": False}}
        )
        self.assertTrue(result["passed"], result["errors"])

        (observation / "selected_calibration_apt.ecsv").write_bytes(
            (
                Path(__file__).parent
                / "examples/sci_cal_001_selected_calibration_apt.ecsv"
            ).read_bytes()
        )
        result = self.validate(
            [entry], {"calibration": {"enabled": False}}
        )
        self.assertFalse(result["passed"])
        self.assertTrue(any(
            "allows at most 0" in error for error in result["errors"]
        ))

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


if __name__ == "__main__":
    unittest.main()
