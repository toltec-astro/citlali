import json
import tempfile
import unittest
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
