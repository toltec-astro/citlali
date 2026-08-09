#include <citlali/core/engine/calib.h>

#include <gtest/gtest.h>

#include <netcdf>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iterator>
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
};

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
                                const std::vector<AptRow> &rows) {
    Eigen::MatrixXd table = Eigen::MatrixXd::Ones(
        static_cast<Eigen::Index>(rows.size()), calib.apt_header_keys.size());
    auto column_index = [&](const std::string &name) {
        const auto it = std::find(
            calib.apt_header_keys.begin(), calib.apt_header_keys.end(), name);
        return static_cast<Eigen::Index>(
            std::distance(calib.apt_header_keys.begin(), it));
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
    }
    YAML::Node meta;
    meta["Radesys"] = "altaz";
    auto headers = calib.apt_header_keys;
    to_ecsv_from_matrix(base_path.string(), table, headers, meta);
    return std::filesystem::path(base_path.string() + ".ecsv");
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
