#include <citlali/core/engine/detail/beammap_ptc_product_output_helpers.h>
#include <citlali/core/pipeline/output_netcdf_metadata.h>
#include <citlali/core/pipeline/ptcdiag_netcdf.h>
#include <citlali/core/pipeline/tod_output_append_bounds.h>

#include <gtest/gtest.h>

#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace {

struct PtcTodSchemaCalib {
    Eigen::Index n_dets = 2;
    Eigen::Index n_nws = 1;
    Eigen::VectorXi nws = Eigen::VectorXi::Zero(1);
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
        citlali::config::ProcessedTimeChunkConfig config;
        config.flagging.second_pass_local.enabled = true;

        citlali::pipeline::add_ptcdiag_tod_optional_diag(
            file, PtcTodSchemaCalib{}, config, signal_dims,
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

TEST(ptc_tod_schema, iteration_field_exists_before_final_header_and_updates) {
    const auto path =
        std::filesystem::path(::testing::TempDir()) /
        "citlali_ptc_tod_iteration_metadata_test.nc";
    std::filesystem::remove(path);

    {
        netCDF::NcFile file(path.string(), netCDF::NcFile::replace);
        citlali::pipeline::add_tod_fruit_loop_iteration_var(file, 0);
    }

    const std::map<std::string, std::string> tod_filenames{
        {"ptc", path.string()}};
    EXPECT_NO_THROW(
        beammap_ptc_product_output_helpers::update_ptc_tod_fruitloops_iter(
            tod_filenames, 4, nullptr));

    {
        netCDF::NcFile file(path.string(), netCDF::NcFile::write);
        citlali::pipeline::add_tod_auxiliary_metadata_vars(
            file, 122.0703125, "apt_test.ecsv", 4);
    }

    {
        netCDF::NcFile file(path.string(), netCDF::NcFile::read);
        int fruit_loop_iter = -1;
        file.getVar("FRUITLOOPS_ITER").getVar(&fruit_loop_iter);
        EXPECT_EQ(fruit_loop_iter, 4);
        EXPECT_FALSE(file.getVar("SAMPRATE").isNull());
        EXPECT_FALSE(file.getVar("APT").isNull());
    }

    std::filesystem::remove(path);
}

TEST(tod_output_scan_metadata, follows_variable_length_append_extents) {
    std::size_t existing_samples = 0;
    for (const auto expected : std::vector<std::array<std::size_t, 2>>{
             {0, 605}, {606, 1380}, {1381, 2131}, {2132, 2898}}) {
        const auto appended_samples = expected[1] - expected[0] + 1;
        const auto actual = citlali::pipeline::tod_output_append_bounds(
            existing_samples, appended_samples);
        EXPECT_EQ(actual, expected);
        existing_samples = actual[1] + 1;
    }
    EXPECT_THROW(
        citlali::pipeline::tod_output_append_bounds(existing_samples, 0),
        std::invalid_argument);
}

}  // namespace
