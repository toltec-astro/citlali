#include <citlali/core/pipeline/ptcdiag_netcdf.h>

#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <vector>

namespace {

struct PtcTodSchemaCalib {
    Eigen::Index n_dets = 2;
    Eigen::Index n_nws = 1;
    Eigen::VectorXi nws = Eigen::VectorXi::Zero(1);
};

struct PtcTodSchemaProc {
    struct {
        bool enabled = true;
    } second_pass_local;

    struct {
        struct {
            bool enabled = false;
        } corr_grouping;
        struct {
            bool enabled = false;
        } adaptive_selector;
        std::vector<std::string> grouping;
    } cleaner;

    struct {
        bool enabled = false;
    } weight_corr_penalty;

    struct {
        bool enabled = false;
    } busy_row_suppression;
};

TEST(ptc_tod_schema, includes_all_second_pass_summary_fields) {
    const auto path =
        std::filesystem::path(::testing::TempDir()) /
        "citlali_ptc_tod_schema_test.nc";
    std::filesystem::remove(path);

    {
        netCDF::NcFile file(path.string(), netCDF::NcFile::replace);
        const auto n_scans = file.addDim("n_scans", 1);
        const auto n_pts = file.addDim("n_pts", 3);
        const auto n_dets = file.addDim("n_dets", 2);
        const std::vector<netCDF::NcDim> signal_dims = {n_pts, n_dets};
        const std::vector<std::size_t> chunk_sizes = {1, 2};

        citlali::pipeline::add_ptcdiag_tod_optional_diag(
            file, PtcTodSchemaCalib{}, PtcTodSchemaProc{}, signal_dims,
            netCDF::NcVar::nc_CHUNKED, chunk_sizes, n_scans, n_dets, 1,
            citlali::pipeline::ptcdiag_fill_int(),
            citlali::pipeline::ptcdiag_fill_double());

        for (const std::string &name : {
                 "ptc_second_pass_n_rejected_clusters",
                 "ptc_second_pass_n_rejected_events",
                 "ptc_second_pass_n_source_protected_clusters",
                 "ptc_second_pass_n_source_protected_events"}) {
            EXPECT_FALSE(file.getVar(name).isNull()) << name;
        }
    }

    std::filesystem::remove(path);
}

}  // namespace
