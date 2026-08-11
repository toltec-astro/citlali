#include <citlali/core/engine/calib.h>
#include <citlali/core/utils/sha256.h>

#include <gtest/gtest.h>

#include <netcdf>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

namespace {

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        const auto nonce =
            std::chrono::steady_clock::now().time_since_epoch().count();
        path = std::filesystem::temp_directory_path() /
               ("citlali-calib-apt-binding-" + std::to_string(nonce));
        std::filesystem::create_directories(path);
    }

    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }

    std::filesystem::path path;
};

struct AptRow {
    double uid;
    int network;
    double tone_frequency_hz;
    double det_id = std::numeric_limits<double>::quiet_NaN();
    double det_id_right = std::numeric_limits<double>::quiet_NaN();
    double meas_idx = std::numeric_limits<double>::quiet_NaN();
    double design_idx = std::numeric_limits<double>::quiet_NaN();
    double match_id = std::numeric_limits<double>::quiet_NaN();
    std::string det_id_text;
    std::string measured_id;
    std::string matched_design_id;
    std::string match_status;
};

constexpr const char *design_input_sha256 =
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
constexpr const char *measured_input_sha256 =
    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

std::string valid_tolapt_inputs_yaml() {
    return
        "inputs:\n"
        "  design_apt:\n"
        "    path: /tolapt-inputs/design.ecsv\n"
        "    sha256: " + std::string{design_input_sha256} + "\n"
        "    bytes: 1234\n"
        "    mtime_utc: 2026-08-01T12:34:56Z\n"
        "  measured_apt:\n"
        "    path: /tolapt-inputs/measured.ecsv\n"
        "    sha256: " + std::string{measured_input_sha256} + "\n"
        "    bytes: 5678\n"
        "    mtime_utc: 2026-08-02T01:02:03Z\n";
}

void write_raw_network_file(const std::filesystem::path &path, int nw,
                            const std::vector<double> &absolute_tones_hz,
                            bool include_identity = true) {
    netCDF::NcFile file(path.string(), netCDF::NcFile::replace);
    auto roach_index =
        file.addVar("Header.Toltec.RoachIndex", netCDF::ncInt);
    roach_index.putVar(&nw);
    if (!include_identity) {
        return;
    }
    const auto samples = file.addDim("sample", 2);
    const auto detectors = file.addDim("detector", absolute_tones_hz.size());
    const auto sweeps = file.addDim("sweep", 1);
    file.addVar("Data.Toltec.Is", netCDF::ncDouble, {samples, detectors});
    const double lo_hz = 1.0e9;
    auto lo = file.addVar("Header.Toltec.LoCenterFreq", netCDF::ncDouble);
    lo.putVar(&lo_hz);
    auto tones = file.addVar(
        "Header.Toltec.ToneFreq", netCDF::ncDouble, {sweeps, detectors});
    std::vector<double> offsets;
    offsets.reserve(absolute_tones_hz.size());
    for (const double frequency : absolute_tones_hz) {
        offsets.push_back(frequency - lo_hz);
    }
    tones.putVar(offsets.data());
}

std::filesystem::path write_apt(const std::filesystem::path &base_path,
                                const engine::Calib &calib,
                                const std::vector<AptRow> &rows,
                                YAML::Node meta = {}) {
    auto headers = calib.apt_header_keys;
    const auto append_optional = [&](const std::string &name,
                                     auto value_for_row) {
        if (std::any_of(rows.begin(), rows.end(), [&](const auto &row) {
                return std::isfinite(value_for_row(row));
            })) {
            headers.push_back(name);
        }
    };
    append_optional("det_id", [](const auto &row) { return row.det_id; });
    append_optional("det_id_right",
                    [](const auto &row) { return row.det_id_right; });
    append_optional("meas_idx",
                    [](const auto &row) { return row.meas_idx; });
    append_optional("design_idx",
                    [](const auto &row) { return row.design_idx; });
    append_optional("match_id",
                    [](const auto &row) { return row.match_id; });
    Eigen::MatrixXd table = Eigen::MatrixXd::Ones(
        static_cast<Eigen::Index>(rows.size()), headers.size());
    auto column_index = [&](const std::string &name) {
        const auto it = std::find(
            headers.begin(), headers.end(), name);
        return static_cast<Eigen::Index>(
            std::distance(headers.begin(), it));
    };
    for (Eigen::Index row = 0; row < table.rows(); ++row) {
        table(row, column_index("uid")) = rows[row].uid;
        table(row, column_index("nw")) = rows[row].network;
        table(row, column_index("tone_freq")) = rows[row].tone_frequency_hz;
        table(row, column_index("array")) = 0.0;
        table(row, column_index("fg")) = 0.0;
        table(row, column_index("flag")) = 0.0;
        table(row, column_index("a_fwhm")) = 10.0;
        table(row, column_index("b_fwhm")) = 10.0;
        table(row, column_index("flxscale")) = 1.0;
        table(row, column_index("responsivity")) = 1.0;
        table(row, column_index("sens")) = 1.0;
        if (std::find(headers.begin(), headers.end(), "det_id") !=
            headers.end()) {
            table(row, column_index("det_id")) = rows[row].det_id;
        }
        if (std::find(headers.begin(), headers.end(), "det_id_right") !=
            headers.end()) {
            table(row, column_index("det_id_right")) =
                rows[row].det_id_right;
        }
        if (std::find(headers.begin(), headers.end(), "meas_idx") !=
            headers.end()) {
            table(row, column_index("meas_idx")) = rows[row].meas_idx;
        }
        if (std::find(headers.begin(), headers.end(), "design_idx") !=
            headers.end()) {
            table(row, column_index("design_idx")) = rows[row].design_idx;
        }
        if (std::find(headers.begin(), headers.end(), "match_id") !=
            headers.end()) {
            table(row, column_index("match_id")) = rows[row].match_id;
        }
    }
    meta["Radesys"] = "altaz";
    to_ecsv_from_matrix(base_path.string(), table, headers, meta);
    return std::filesystem::path(base_path.string() + ".ecsv");
}

std::filesystem::path write_mixed_apt(
    const std::filesystem::path &path, const engine::Calib &calib,
    const std::vector<AptRow> &rows) {
    auto headers = calib.apt_header_keys;
    const auto any_finite = [&](auto value_for_row) {
        return std::any_of(rows.begin(), rows.end(), [&](const auto &row) {
            return std::isfinite(value_for_row(row));
        });
    };
    const auto any_text = [&](auto value_for_row) {
        return std::any_of(rows.begin(), rows.end(), [&](const auto &row) {
            return !value_for_row(row).empty();
        });
    };
    const auto append_numeric = [&](const std::string &name,
                                    auto value_for_row) {
        if (any_finite(value_for_row)) {
            headers.push_back(name);
        }
    };
    append_numeric("det_id", [](const auto &row) { return row.det_id; });
    append_numeric(
        "det_id_right", [](const auto &row) { return row.det_id_right; });
    append_numeric("meas_idx", [](const auto &row) { return row.meas_idx; });
    append_numeric(
        "design_idx", [](const auto &row) { return row.design_idx; });
    append_numeric("match_id", [](const auto &row) { return row.match_id; });
    const auto append_string = [&](const std::string &name,
                                   auto value_for_row) {
        if (any_text(value_for_row)) {
            headers.push_back(name);
        }
    };
    if (!any_finite([](const auto &row) { return row.det_id; })) {
        append_string(
            "det_id", [](const auto &row) { return row.det_id_text; });
    }
    append_string(
        "measured_id", [](const auto &row) { return row.measured_id; });
    append_string("matched_design_id",
                  [](const auto &row) { return row.matched_design_id; });
    append_string(
        "match_status", [](const auto &row) { return row.match_status; });

    std::ofstream output(path);
    output << "# %ECSV 1.0\n# ---\n# datatype:\n";
    for (const auto &name : headers) {
        const bool is_string =
            name == "measured_id" || name == "matched_design_id" ||
            name == "match_status" ||
            (name == "det_id" &&
             !any_finite([](const auto &row) { return row.det_id; }));
        output << "# - {name: " << name << ", datatype: "
               << (is_string ? "string" : "float64") << "}\n";
    }
    output << "# meta: !!omap\n# - {Radesys: altaz}\n"
           << "# schema: astropy-2.0\n";
    for (std::size_t column = 0; column < headers.size(); ++column) {
        if (column != 0) {
            output << ' ';
        }
        output << headers[column];
    }
    output << '\n' << std::setprecision(17);

    const auto numeric_value = [](const AptRow &row,
                                  const std::string &name) {
        if (name == "uid") return row.uid;
        if (name == "tone_freq") return row.tone_frequency_hz;
        if (name == "nw") return static_cast<double>(row.network);
        if (name == "array" || name == "fg" || name == "flag") return 0.0;
        if (name == "a_fwhm" || name == "b_fwhm") return 10.0;
        if (name == "det_id") return row.det_id;
        if (name == "det_id_right") return row.det_id_right;
        if (name == "meas_idx") return row.meas_idx;
        if (name == "design_idx") return row.design_idx;
        if (name == "match_id") return row.match_id;
        return 1.0;
    };
    const auto text_value = [](const AptRow &row,
                               const std::string &name) {
        if (name == "det_id") return row.det_id_text;
        if (name == "measured_id") return row.measured_id;
        if (name == "matched_design_id") return row.matched_design_id;
        if (name == "match_status") return row.match_status;
        return std::string{};
    };
    for (const auto &row : rows) {
        for (std::size_t column = 0; column < headers.size(); ++column) {
            if (column != 0) {
                output << ' ';
            }
            const auto &name = headers[column];
            const auto value = text_value(row, name);
            if (!value.empty()) {
                output << value;
            }
            else {
                output << numeric_value(row, name);
            }
        }
        output << '\n';
    }
    return path;
}

std::vector<AptRow> canonical_rows() {
    return {{10.0, 0, 1.1e9}, {11.0, 0, 1.2e9},
            {20.0, 1, 1.3e9}, {21.0, 1, 1.4e9}};
}

void write_canonical_raw(const TemporaryDirectory &temp,
                         std::vector<std::string> &files,
                         std::vector<std::string> &interfaces,
                         bool reverse_networks = false) {
    const auto raw0 = temp.path / "toltec0.nc";
    const auto raw1 = temp.path / "toltec1.nc";
    write_raw_network_file(raw0, 0, {1.1e9, 1.2e9});
    write_raw_network_file(raw1, 1, {1.3e9, 1.4e9});
    files = reverse_networks
        ? std::vector<std::string>{raw1.string(), raw0.string()}
        : std::vector<std::string>{raw0.string(), raw1.string()};
    interfaces = reverse_networks
        ? std::vector<std::string>{"toltec1", "toltec0"}
        : std::vector<std::string>{"toltec0", "toltec1"};
}

TEST(calib_apt_binding, explicit_join_is_invariant_to_apt_row_permutation) {
    TemporaryDirectory temp;
    engine::Calib canonical;
    engine::Calib permuted;
    auto rows = canonical_rows();
    const auto apt0 = write_apt(temp.path / "apt0", canonical, rows);
    std::reverse(rows.begin(), rows.end());
    const auto apt1 = write_apt(temp.path / "apt1", permuted, rows);
    std::vector<std::string> files;
    std::vector<std::string> interfaces;
    write_canonical_raw(temp, files, interfaces);

    ASSERT_NO_THROW(canonical.get_apt(apt0.string(), files, interfaces));
    ASSERT_NO_THROW(permuted.get_apt(apt1.string(), files, interfaces));
    EXPECT_TRUE(canonical.apt.at("uid").isApprox(permuted.apt.at("uid"), 0.0));
    EXPECT_TRUE(canonical.apt.at("kids_tone").isApprox(
        permuted.apt.at("kids_tone"), 0.0));
    EXPECT_EQ(canonical.apt_acquisition_binding.raw_observation_identity,
              permuted.apt_acquisition_binding.raw_observation_identity);
    EXPECT_NE(canonical.apt_acquisition_binding.artifact_sha256,
              permuted.apt_acquisition_binding.artifact_sha256);
    EXPECT_NE(canonical.apt_acquisition_binding.binding_sha256,
              permuted.apt_acquisition_binding.binding_sha256);
    EXPECT_EQ(canonical.apt_acquisition_binding.mode,
              "explicit_network_local_tone_frequency_join_v1");
    EXPECT_TRUE(canonical.apt_acquisition_binding.valid);
}

TEST(calib_apt_binding, network_file_reorder_preserves_keyed_rows) {
    TemporaryDirectory temp;
    engine::Calib calib;
    const auto apt = write_apt(temp.path / "apt", calib, canonical_rows());
    std::vector<std::string> files;
    std::vector<std::string> interfaces;
    write_canonical_raw(temp, files, interfaces, true);

    ASSERT_NO_THROW(calib.get_apt(apt.string(), files, interfaces));
    EXPECT_EQ(calib.apt.at("uid")(0), 20.0);
    EXPECT_EQ(calib.apt.at("uid")(2), 10.0);
}

TEST(calib_apt_lineage,
     preserves_legacy_observation_source_row_identity_and_validity) {
    TemporaryDirectory temp;
    engine::Calib calib;
    auto rows = canonical_rows();
    rows[0].det_id = 1000.0;
    rows[0].det_id_right = 2000.0;
    rows[0].meas_idx = 30.0;
    rows[0].design_idx = 40.0;
    rows[1].det_id = 1001.0;
    rows[1].det_id_right = -1.0;
    rows[1].meas_idx = 31.0;
    rows[1].design_idx = 41.0;
    rows[2].det_id = 1002.0;
    rows[2].det_id_right = 2002.0;
    rows[3].det_id = 1003.0;
    rows[3].det_id_right = 2003.0;
    YAML::Node meta;
    meta["obsnum"] = 134723;
    meta["Header ObsNum"] = 134723;
    meta["obsnum_matched"] = 137389;
    meta["source"] = "Neptune";
    const auto apt = write_apt(temp.path / "legacy", calib, rows, meta);
    std::vector<std::string> files;
    std::vector<std::string> interfaces;
    write_canonical_raw(temp, files, interfaces);

    ASSERT_NO_THROW(calib.get_apt(apt.string(), files, interfaces));
    ASSERT_TRUE(calib.apt_lineage.available);
    ASSERT_TRUE(calib.apt_lineage.valid);
    EXPECT_TRUE(calib.apt_lineage.legacy_metadata_available);
    EXPECT_FALSE(calib.apt_lineage.modern_tolapt_manifest_available);
    EXPECT_TRUE(calib.apt_lineage.modern_tolapt_design_input.path.empty());
    EXPECT_TRUE(calib.apt_lineage.modern_tolapt_measured_input.path.empty());
    EXPECT_EQ(calib.apt_lineage.observation_identity, "134723");
    EXPECT_EQ(calib.apt_lineage.matched_observation_identity, "137389");
    EXPECT_EQ(calib.apt_lineage.selected_source, "Neptune");
    EXPECT_EQ(calib.apt_lineage.ordered_rows.size(), rows.size());
    EXPECT_EQ(calib.apt_lineage.ordered_rows[0].det_id, "1000");
    EXPECT_TRUE(calib.apt_lineage.ordered_rows[0].measured_id.empty());
    EXPECT_TRUE(calib.apt_lineage.ordered_rows[0].matched_design_id.empty());
    EXPECT_TRUE(calib.apt_lineage.ordered_rows[0].match_status.empty());
    EXPECT_TRUE(calib.apt_lineage.ordered_rows[0].eligible);
    EXPECT_FALSE(calib.apt_lineage.ordered_rows[1].eligible);
    EXPECT_NE(calib.apt_lineage.ordered_rows[0].stable_association, "");
    EXPECT_NE(calib.apt_lineage.row_association_sha256, "");
    EXPECT_NE(calib.apt_acquisition_binding.binding_sha256, "");
    ASSERT_EQ(calib.apt_acquisition_binding.raw_artifacts.size(),
              files.size());
    EXPECT_EQ(calib.apt_acquisition_binding.raw_artifacts[0].path,
              files[0]);
    EXPECT_EQ(calib.apt_acquisition_binding.raw_artifacts[0].sha256,
              citlali::utils::sha256_file(files[0]));
    EXPECT_EQ(calib.apt_acquisition_binding.raw_artifacts[0].interface,
              interfaces[0]);
    EXPECT_EQ(calib.apt_acquisition_binding.raw_artifacts[0].network, 0);
    EXPECT_FALSE(calib.apt_acquisition_binding.raw_artifacts[0]
                     .absolute_tone_frequency_hz.empty());
}

TEST(calib_apt_lineage,
     consumes_only_unique_contract_associated_tolapt_manifest) {
    TemporaryDirectory temp;
    const auto run_root = temp.path / "tolapt-run";
    const auto tables = run_root / "tables";
    std::filesystem::create_directories(tables);
    engine::Calib calib;
    auto rows = canonical_rows();
    rows[0].match_id = 10.0;
    rows[1].match_id = 13.0;
    rows[2].match_id = 11.0;
    rows[3].match_id = 12.0;
    for (std::size_t index = 0; index < rows.size(); ++index) {
        rows[index].det_id_right =
            index == 1 ? -1.0 : 2000.0 + static_cast<double>(index);
        rows[index].det_id_text = "measured-det-" + std::to_string(index);
        rows[index].measured_id = "measured-row-" + std::to_string(index);
        rows[index].matched_design_id =
            "design-row-" + std::to_string(index);
        rows[index].match_status = "matched";
    }
    const auto apt = write_mixed_apt(
        tables / "measured.enriched.ecsv", calib, rows);
    {
        std::ofstream manifest(run_root / "manifest.yaml");
        manifest << "contract_version: tolapt.run.v1\n"
                 << "run_id: fixture-run\n"
                 << valid_tolapt_inputs_yaml()
                 << "outputs:\n"
                 << "  measured_enriched: tables/measured.enriched.ecsv\n";
    }
    std::vector<std::string> files;
    std::vector<std::string> interfaces;
    write_canonical_raw(temp, files, interfaces);

    ASSERT_NO_THROW(calib.get_apt(apt.string(), files, interfaces));
    EXPECT_TRUE(calib.apt_lineage.modern_tolapt_manifest_available);
    EXPECT_EQ(calib.apt_lineage.modern_tolapt_contract_version,
              "tolapt.run.v1");
    EXPECT_EQ(calib.apt_lineage.modern_tolapt_run_id, "fixture-run");
    EXPECT_EQ(calib.apt_lineage.modern_tolapt_output_key,
              "measured_enriched");
    EXPECT_EQ(calib.apt_lineage.modern_tolapt_output_path,
              "tables/measured.enriched.ecsv");
    EXPECT_EQ(calib.apt_lineage.modern_tolapt_design_input.path,
              "/tolapt-inputs/design.ecsv");
    EXPECT_EQ(calib.apt_lineage.modern_tolapt_design_input.sha256,
              design_input_sha256);
    EXPECT_EQ(calib.apt_lineage.modern_tolapt_design_input.bytes, 1234U);
    EXPECT_EQ(calib.apt_lineage.modern_tolapt_design_input.mtime_utc,
              "2026-08-01T12:34:56Z");
    EXPECT_EQ(calib.apt_lineage.modern_tolapt_measured_input.path,
              "/tolapt-inputs/measured.ecsv");
    EXPECT_EQ(calib.apt_lineage.modern_tolapt_measured_input.sha256,
              measured_input_sha256);
    EXPECT_EQ(calib.apt_lineage.modern_tolapt_measured_input.bytes, 5678U);
    EXPECT_EQ(calib.apt_lineage.modern_tolapt_measured_input.mtime_utc,
              "2026-08-02T01:02:03Z");
    EXPECT_NE(calib.apt_lineage.modern_tolapt_manifest_sha256, "");
    EXPECT_NE(calib.apt_lineage.modern_tolapt_association_sha256, "");
    EXPECT_EQ(calib.apt_lineage.selected_apt_sha256,
              calib.apt_acquisition_binding.artifact_sha256);
    EXPECT_EQ(calib.apt_lineage.ordered_rows[0].modern_match_id, "10");
    EXPECT_EQ(calib.apt_lineage.ordered_rows[0].det_id,
              "measured-det-0");
    EXPECT_EQ(calib.apt_lineage.ordered_rows[0].measured_id,
              "measured-row-0");
    EXPECT_EQ(calib.apt_lineage.ordered_rows[0].matched_design_id,
              "design-row-0");
    EXPECT_EQ(calib.apt_lineage.ordered_rows[0].match_status, "matched");
    const auto &retained = calib.apt_lineage.ordered_rows[0].retained_fields;
    const auto retained_field = [&](const std::string &name) {
        return std::find_if(retained.begin(), retained.end(),
                            [&](const auto &field) {
                                return field.name == name;
                            });
    };
    const auto design_field = retained_field("matched_design_id");
    ASSERT_NE(design_field, retained.end());
    EXPECT_EQ(design_field->ecsv_datatype, "string");
    EXPECT_EQ(design_field->value, "design-row-0");
    const auto match_id_field = retained_field("match_id");
    ASSERT_NE(match_id_field, retained.end());
    EXPECT_EQ(match_id_field->ecsv_datatype, "float64");
    EXPECT_EQ(match_id_field->value, "10");
    EXPECT_NE(calib.apt_lineage.ordered_rows[0].stable_association.find(
                  "matched_design_id"),
              std::string::npos);
    EXPECT_NE(calib.apt_lineage.ordered_rows[0].stable_association.find(
                  "string"),
              std::string::npos);
    EXPECT_TRUE(calib.apt_lineage.ordered_rows[0].eligible);
    EXPECT_FALSE(calib.apt_lineage.ordered_rows[1].eligible);
}

TEST(calib_apt_lineage,
     rejects_malformed_conflicting_and_ambiguous_modern_lineage) {
    TemporaryDirectory temp;
    const auto run_root = temp.path / "tolapt-run";
    const auto tables = run_root / "tables";
    std::filesystem::create_directories(tables);
    engine::Calib fixture;
    auto rows = canonical_rows();
    for (std::size_t index = 0; index < rows.size(); ++index) {
        rows[index].match_id = static_cast<double>(index);
        rows[index].measured_id = "measured-row-" + std::to_string(index);
        rows[index].matched_design_id =
            "design-row-" + std::to_string(index);
        rows[index].match_status = "matched";
    }
    const auto apt = write_mixed_apt(
        tables / "measured.enriched.ecsv", fixture, rows);
    std::vector<std::string> files;
    std::vector<std::string> interfaces;
    write_canonical_raw(temp, files, interfaces);

    {
        std::ofstream manifest(run_root / "manifest.yaml");
        manifest << "contract_version: tolapt.run.v1\n"
                 << "run_id: malformed-input-run\n"
                 << "inputs:\n"
                 << "  design_apt:\n"
                 << "    path: /tolapt-inputs/design.ecsv\n"
                 << "    sha256: " << design_input_sha256 << "\n"
                 << "    bytes: 1234\n"
                 << "    mtime_utc: 2026-08-01T12:34:56Z\n"
                 << "  measured_apt:\n"
                 << "    path: /tolapt-inputs/measured.ecsv\n"
                 << "    sha256: " << measured_input_sha256 << "\n"
                 << "    bytes: 5678\n"
                 << "outputs:\n"
                 << "  measured_enriched: tables/measured.enriched.ecsv\n";
    }
    engine::Calib malformed;
    EXPECT_THROW(malformed.get_apt(apt.string(), files, interfaces),
                 std::runtime_error);

    {
        std::ofstream manifest(run_root / "manifest.yaml");
        manifest << "contract_version: tolapt.run.v1\n"
                 << "run_id: conflicting-input-run\n"
                 << "inputs:\n"
                 << "  design_apt:\n"
                 << "    path: /tolapt-inputs/design.ecsv\n"
                 << "    sha256: " << design_input_sha256 << "\n"
                 << "    sha256: " << measured_input_sha256 << "\n"
                 << "    bytes: 1234\n"
                 << "    mtime_utc: 2026-08-01T12:34:56Z\n"
                 << "  measured_apt:\n"
                 << "    path: /tolapt-inputs/measured.ecsv\n"
                 << "    sha256: " << measured_input_sha256 << "\n"
                 << "    bytes: 5678\n"
                 << "    mtime_utc: 2026-08-02T01:02:03Z\n"
                 << "outputs:\n"
                 << "  measured_enriched: tables/measured.enriched.ecsv\n";
    }
    engine::Calib conflicting_input;
    EXPECT_THROW(conflicting_input.get_apt(
                     apt.string(), files, interfaces),
                 std::runtime_error);

    {
        std::ofstream manifest(run_root / "manifest.yaml");
        manifest << "contract_version: tolapt.run.v1\n"
                 << "run_id: ambiguous-run\n"
                 << valid_tolapt_inputs_yaml()
                 << "outputs:\n"
                 << "  measured_enriched: tables/measured.enriched.ecsv\n"
                 << "  duplicate: tables/measured.enriched.ecsv\n";
    }
    engine::Calib ambiguous;
    EXPECT_THROW(ambiguous.get_apt(apt.string(), files, interfaces),
                 std::runtime_error);

    rows[0].match_id = -1.0;
    write_mixed_apt(tables / "measured.enriched.ecsv", fixture, rows);
    {
        std::ofstream manifest(run_root / "manifest.yaml");
        manifest << "contract_version: tolapt.run.v1\n"
                 << "run_id: conflicting-row-run\n"
                 << valid_tolapt_inputs_yaml()
                 << "outputs:\n"
                 << "  measured_enriched: tables/measured.enriched.ecsv\n";
    }
    engine::Calib conflicting_row;
    EXPECT_THROW(conflicting_row.get_apt(
                     apt.string(), files, interfaces),
                 std::runtime_error);
}

TEST(calib_apt_binding, rejects_missing_extra_duplicate_and_mismatched_keys) {
    for (const auto &rows : std::vector<std::vector<AptRow>>{
             {{10.0, 0, 1.1e9}, {20.0, 1, 1.3e9}, {21.0, 1, 1.4e9}},
             {{10.0, 0, 1.1e9}, {11.0, 0, 1.2e9}, {12.0, 0, 1.25e9},
              {20.0, 1, 1.3e9}, {21.0, 1, 1.4e9}},
             {{10.0, 0, 1.1e9}, {11.0, 0, 1.2e9},
              {20.0, 1, 1.3e9}, {21.0, 1, 1.4e9},
              {30.0, 2, 1.5e9}},
             {{10.0, 0, 1.1e9}, {11.0, 0, 1.1e9},
              {20.0, 1, 1.3e9}, {21.0, 1, 1.4e9}},
             {{10.0, 0, 1.1e9}, {11.0, 0, 1.25e9},
              {20.0, 1, 1.3e9}, {21.0, 1, 1.4e9}}}) {
        TemporaryDirectory temp;
        engine::Calib calib;
        const auto apt = write_apt(temp.path / "apt", calib, rows);
        std::vector<std::string> files;
        std::vector<std::string> interfaces;
        write_canonical_raw(temp, files, interfaces);
        EXPECT_THROW(calib.get_apt(apt.string(), files, interfaces),
                     std::runtime_error);
    }
}

TEST(calib_apt_binding, rejects_unavailable_or_ambiguous_raw_identity) {
    TemporaryDirectory temp;
    engine::Calib calib;
    const auto apt = write_apt(temp.path / "apt", calib, canonical_rows());
    const auto unavailable = temp.path / "unavailable.nc";
    write_raw_network_file(unavailable, 0, {}, false);
    std::vector<std::string> unavailable_files{unavailable.string()};
    std::vector<std::string> unavailable_interfaces{"toltec0"};
    EXPECT_THROW(calib.get_apt(apt.string(), unavailable_files,
                               unavailable_interfaces),
                 std::runtime_error);

    const auto duplicate = temp.path / "duplicate.nc";
    write_raw_network_file(duplicate, 0, {1.1e9, 1.1e9});
    std::vector<std::string> duplicate_files{duplicate.string()};
    EXPECT_THROW(calib.get_apt(apt.string(), duplicate_files,
                               unavailable_interfaces),
                 std::runtime_error);
}

TEST(calib_apt_binding, rejects_interface_roach_identity_conflict) {
    TemporaryDirectory temp;
    engine::Calib calib;
    const auto apt = write_apt(temp.path / "apt", calib, canonical_rows());
    const auto raw = temp.path / "toltec0.nc";
    write_raw_network_file(raw, 0, {1.1e9, 1.2e9});
    std::vector<std::string> files{raw.string()};
    std::vector<std::string> interfaces{"toltec1"};
    EXPECT_THROW(calib.get_apt(apt.string(), files, interfaces),
                 std::runtime_error);
    interfaces = {"toltec0suffix"};
    EXPECT_THROW(calib.get_apt(apt.string(), files, interfaces),
                 std::runtime_error);
}

TEST(calib_unit_policy, production_path_accepts_only_mjy_per_beam) {
    TemporaryDirectory temp;
    engine::Calib calib;
    const auto apt = write_apt(temp.path / "apt", calib, canonical_rows());
    std::vector<std::string> files;
    std::vector<std::string> interfaces;
    write_canonical_raw(temp, files, interfaces);
    ASSERT_NO_THROW(calib.get_apt(apt.string(), files, interfaces));

    ASSERT_NO_THROW(calib.calc_flux_calibration("mJy/beam", 1.0));
    EXPECT_TRUE(calib.flux_conversion_factor.isOnes());
    for (const std::string unit : {"MJy/sr", "uK", "Jy/pixel", "Jy/beam"}) {
        EXPECT_THROW(calib.calc_flux_calibration(unit, 1.0),
                     std::runtime_error);
    }
}

}  // namespace
