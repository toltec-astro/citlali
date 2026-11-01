#include <citlali/core/engine/engine.h>
#include <citlali/core/engine/detail/beammap_ptc_product_output_helpers.h>
#include <citlali/core/pipeline/output_netcdf_metadata.h>
#include <citlali/core/pipeline/ptcdiag_netcdf.h>

#include <gtest/gtest.h>
#include <spdlog/sinks/null_sink.h>

#include <array>
#include <chrono>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace {

struct MetadataDirectory {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("citlali-ptc-metadata-" + std::to_string(
            std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    MetadataDirectory() { std::filesystem::create_directories(path / "raw"); }
    ~MetadataDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

void ensure_metadata_logger() {
    if (!spdlog::get("citlali_logger")) {
        spdlog::register_logger(std::make_shared<spdlog::logger>(
            "citlali_logger", std::make_shared<spdlog::sinks::null_sink_mt>()));
    }
}

TEST(ptc_metadata, production_schema_allows_iteration_update_before_final_header) {
    ensure_metadata_logger();
    for (const auto mode : {citlali::config::TodStreamOutputMode::full,
                            citlali::config::TodStreamOutputMode::mini}) {
        MetadataDirectory directory;
        Engine engine;
        engine.logger = spdlog::get("citlali_logger");
        engine.typed_config.runtime.reduction_type = citlali::config::ReductionType::beammap;
        engine.runtime_config_provenance = citlali::config::make_runtime_config_provenance(
            engine.typed_config.runtime, false);
        engine.typed_config.timestream.output.processed_time_chunk.mode = mode;
        engine.output_paths.obsnum_dir_name = directory.path.string() + "/";
        engine.observation_identity.obsnum = "123456";
        engine.telescope.sim_obs = false;
        engine.telescope.pixel_axes = "altaz";
        engine.telescope.tel_header["Header.Source.Ra"] = Eigen::VectorXd::Zero(1);
        engine.telescope.tel_header["Header.Source.Dec"] = Eigen::VectorXd::Zero(1);
        engine.telescope.scan_indices.resize(4, 1);
        engine.telescope.scan_indices.col(0) << 0, 2, 0, 2;
        engine.calib.n_dets = 2;
        engine.calib.n_nws = 1;
        engine.calib.nws = Eigen::VectorXI::Zero(1);
        engine.calib.run_hwpr = false;
        engine.rtcproc.run_polarization = false;
        engine.ptcproc.cleaner.n_calc = 1;
        engine.tod_outputs.n_ptc_output_scans = 1;
        engine.iteration.fruit_iter = 2;
        engine.omb.sig_unit = "mJy/beam";

        // Invoke the production initial-schema entry point, not a test-created
        // iteration variable. Beammap updates it before the final header pass.
        ASSERT_NO_THROW(engine.create_tod_files<engine_utils::toltecIO::ptc_timestream>());
        const auto filename = engine.output_paths.tod_filename.at("ptc");
        {
            netCDF::NcFile file(filename, netCDF::NcFile::read);
            const auto iteration = file.getVar("FRUITLOOPS_ITER");
            ASSERT_FALSE(iteration.isNull());
            EXPECT_EQ(iteration.getType().getId(), NC_INT);
            ASSERT_EQ(iteration.getDims().size(), 1U);
            EXPECT_EQ(iteration.getDim(0).getSize(), 1U);
            int value = -1;
            iteration.getVar(&value);
            EXPECT_EQ(value, 2);
            EXPECT_TRUE(file.getVar("SAMPRATE").isNull());
        }
        ASSERT_NO_THROW(beammap_ptc_product_output_helpers::update_ptc_tod_fruitloops_iter(
            engine.output_paths.tod_filename, 4, nullptr));
        {
            netCDF::NcFile file(filename, netCDF::NcFile::write);
            int value = -1;
            file.getVar("FRUITLOOPS_ITER").getVar(&value);
            EXPECT_EQ(value, 4);
            citlali::pipeline::add_tod_auxiliary_metadata_vars(file, 122.0, "fixture.ecsv", 5);
            file.getVar("FRUITLOOPS_ITER").getVar(&value);
            EXPECT_EQ(value, 5);
            EXPECT_EQ(file.getVars("FRUITLOOPS_ITER").size(), 1U);
        }
    }
}

TEST(ptc_metadata, missing_iteration_schema_still_fails_closed) {
    MetadataDirectory directory;
    const auto path = directory.path / "missing.nc";
    { netCDF::NcFile file(path.string(), netCDF::NcFile::replace); }
    const std::map<std::string, std::string> paths{{"ptc", path.string()}};
    EXPECT_THROW(beammap_ptc_product_output_helpers::update_ptc_tod_fruitloops_iter(
        paths, 4, nullptr), std::runtime_error);
    netCDF::NcFile file(path.string(), netCDF::NcFile::read);
    EXPECT_TRUE(file.getVar("FRUITLOOPS_ITER").isNull());
}

struct AppendCalibration {
    bool run_hwpr = false;
    std::map<std::string, Eigen::VectorXd> apt;
};

template <timestream::TCDataKind kind = timestream::TCDataKind::PTC>
void check_append(const std::array<int, 3> &lengths, bool mini, bool outer,
                  const std::array<std::array<int, 2>, 3> &bounds) {
    ensure_metadata_logger();
    MetadataDirectory directory;
    const auto path = directory.path / "append.nc";
    std::vector<double> expected_signal;
    std::vector<int> expected_flags;
    {
        netCDF::NcFile file(path.string(), netCDF::NcFile::replace);
        const auto dims = citlali::pipeline::add_tod_file_dims(file, 3, 4, 2);
        citlali::pipeline::add_tod_scan_index_placeholders(
            file, dims.raw_scans, dims.scans, dims.n_scans, 3, 4, outer,
            citlali::pipeline::tod_output_fill_int());
        citlali::pipeline::add_source_radec_vars(file, 0.0, 0.0);
        // This fixture needs no detector coordinates; retain both actual signal
        // representations so the production appender writes real NetCDF samples.
        file.addVar("signal", mini ? netCDF::NcType(netCDF::ncFloat) : netCDF::NcType(netCDF::ncDouble), dims.signal);
        file.addVar("flags", mini ? netCDF::NcType(netCDF::ncByte) : netCDF::NcType(netCDF::ncDouble), dims.signal);
        file.addVar("TelTime", netCDF::ncDouble, dims.n_pts);
        timestream::TCProc processor;
        AppendCalibration calibration;
        std::map<std::string, Eigen::VectorXd> offsets;
        std::string axes = "altaz";
        const std::array<int, 3> original_scan_ids{2, 7, 11};
        int appended = 0;
        for (int row = 0; row < 3; ++row) {
            timestream::TCData<kind, Eigen::MatrixXd> scan;
            scan.index.data = original_scan_ids[row];
            scan.scans.data.resize(lengths[row], 2);
            scan.flags.data.resize(lengths[row], 2);
            scan.tel_data.data["TelTime"].resize(lengths[row]);
            scan.scan_indices.data.resize(4);
            scan.scan_indices.data << 101, 100 + lengths[row] - 2, 100, 100 + lengths[row] - 1;
            for (int sample = 0; sample < lengths[row]; ++sample) {
                scan.tel_data.data["TelTime"](sample) = appended + sample;
                for (int detector = 0; detector < 2; ++detector) {
                    const double value = 100 * row + 10 * sample + detector;
                    const bool flag = (sample + detector) % 2;
                    scan.scans.data(sample, detector) = value;
                    scan.flags.data(sample, detector) = flag;
                    expected_signal.push_back(value);
                    expected_flags.push_back(flag);
                }
            }
            processor.append_base_to_netcdf(file, scan, "detector", axes,
                offsets, calibration, false, row, outer, mini);
            appended += lengths[row];
        }
    }
    // Reopen the persisted product and compare independent expected positions,
    // complete sample/flag payload and original (selected) scan identities.
    netCDF::NcFile file(path.string(), netCDF::NcFile::read);
    const int total = lengths[0] + lengths[1] + lengths[2];
    ASSERT_EQ(file.getDim("n_pts").getSize(), static_cast<std::size_t>(total));
    std::vector<double> signal(total * 2);
    std::vector<int> flags(total * 2);
    std::vector<double> time(total);
    file.getVar("signal").getVar(signal.data());
    file.getVar("flags").getVar(flags.data());
    file.getVar("TelTime").getVar(time.data());
    EXPECT_EQ(signal, expected_signal);
    EXPECT_EQ(flags, expected_flags);
    for (int i = 0; i < total; ++i) EXPECT_EQ(time[i], i);
    std::array<int, 3> ids{};
    file.getVar("output_scan_index").getVar(ids.data());
    EXPECT_EQ(ids, (std::array<int, 3>{3, 8, 12}));
    int outer_start = 0;
    for (int row = 0; row < 3; ++row) {
        std::array<double, 2> actual{};
        std::array<int, 4> raw{};
        file.getVar("scan_indices").getVar({static_cast<std::size_t>(row), 0}, {1, 2}, actual.data());
        file.getVar("raw_scan_indices").getVar({static_cast<std::size_t>(row), 0}, {1, 4}, raw.data());
        EXPECT_EQ(actual[0], bounds[row][0]) << "row " << row;
        EXPECT_EQ(actual[1], bounds[row][1]) << "row " << row;
        EXPECT_EQ(raw[0], bounds[row][0]);
        EXPECT_EQ(raw[1], bounds[row][1]);
        EXPECT_EQ(raw[2], outer ? outer_start : bounds[row][0]);
        EXPECT_EQ(raw[3], outer ? outer_start + lengths[row] - 1 : bounds[row][1]);
        outer_start += lengths[row];
    }
}

TEST(ptc_metadata, variable_length_full_and_mini_appends_cover_persisted_samples) {
    for (bool mini : {false, true})
        check_append({3, 5, 2}, mini, false, {{{0, 2}, {3, 7}, {8, 9}}});
}
TEST(ptc_metadata, equal_length_appends_preserve_existing_bounds) {
    for (bool mini : {false, true})
        check_append({4, 4, 4}, mini, false, {{{0, 3}, {4, 7}, {8, 11}}});
}
TEST(ptc_metadata, outer_context_keeps_inner_and_outer_bounds_distinct) {
    for (bool mini : {false, true})
        check_append({4, 6, 3}, mini, true, {{{1, 2}, {5, 8}, {11, 11}}});
}

TEST(ptc_metadata, raw_outer_context_preserves_existing_bounds) {
    for (bool mini : {false, true})
        check_append<timestream::TCDataKind::RTC>(
            {4, 6, 3}, mini, true, {{{1, 2}, {5, 8}, {11, 11}}});
}

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

}  // namespace
