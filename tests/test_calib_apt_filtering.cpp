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
               ("citlali-calib-apt-filtering-" + std::to_string(nonce));
        std::filesystem::create_directories(path);
    }

    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }

    std::filesystem::path path;
};

void write_raw_network_file(const std::filesystem::path &path, int nw) {
    netCDF::NcFile file(path.string(), netCDF::NcFile::replace);
    auto roach_index =
        file.addVar("Header.Toltec.RoachIndex", netCDF::ncInt);
    roach_index.putVar(&nw);
}

std::filesystem::path write_apt(
    const std::filesystem::path &base_path,
    const engine::Calib &calib) {
    constexpr Eigen::Index n_rows = 3;
    Eigen::MatrixXd table =
        Eigen::MatrixXd::Ones(n_rows, calib.apt_header_keys.size());

    auto column_index = [&](const std::string &name) {
        const auto it = std::find(
            calib.apt_header_keys.begin(), calib.apt_header_keys.end(), name);
        return static_cast<Eigen::Index>(
            std::distance(calib.apt_header_keys.begin(), it));
    };

    table.col(column_index("uid")) << 0.0, 1.0, 2.0;
    table.col(column_index("nw")) << 0.0, 0.0, 6.0;
    table.col(column_index("array")) << 0.0, 0.0, 1.0;
    table.col(column_index("flag")) << 0.0, 0.0, 1.0;
    table.col(column_index("a_fwhm")).setConstant(10.0);
    table.col(column_index("b_fwhm")).setConstant(10.0);

    YAML::Node meta;
    meta["Radesys"] = "altaz";
    auto headers = calib.apt_header_keys;
    to_ecsv_from_matrix(base_path.string(), table, headers, meta);
    return std::filesystem::path(base_path.string() + ".ecsv");
}

TEST(calib_apt_filtering,
     ignores_fully_flagged_network_absent_from_raw_observation) {
    TemporaryDirectory temp;
    engine::Calib calib;
    const auto apt_path = write_apt(temp.path / "apt", calib);
    const auto raw_path = temp.path / "toltec0.nc";
    write_raw_network_file(raw_path, 0);

    std::vector<std::string> raw_filenames{raw_path.string()};
    std::vector<std::string> interfaces{"toltec0"};

    EXPECT_NO_THROW(
        calib.get_apt(apt_path.string(), raw_filenames, interfaces));
    EXPECT_EQ(calib.n_dets, 2);
    EXPECT_EQ(calib.n_nws, 1);
    EXPECT_EQ(calib.nws.size(), 1);
    EXPECT_EQ(calib.nws(0), 0);
}

TEST(calib_apt_filtering,
     still_rejects_fully_flagged_network_present_in_raw_observation) {
    TemporaryDirectory temp;
    engine::Calib calib;
    const auto apt_path = write_apt(temp.path / "apt", calib);
    const auto raw_path = temp.path / "toltec6.nc";
    write_raw_network_file(raw_path, 6);

    std::vector<std::string> raw_filenames{raw_path.string()};
    std::vector<std::string> interfaces{"toltec6"};

    EXPECT_THROW(
        calib.get_apt(apt_path.string(), raw_filenames, interfaces),
        std::runtime_error);
}

}  // namespace
