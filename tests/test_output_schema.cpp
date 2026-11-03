// Include Pointing first so its declarations cannot rely on a caller's includes.
#include <citlali/core/engine/pointing.h>
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

template <mapmaking::MapType map_type>
void check_pointing_fit_header(Pointing &pointing, const Eigen::MatrixXf &table,
                               const std::filesystem::path &base) {
    constexpr bool observation =
        map_type == mapmaking::RawObs || map_type == mapmaking::FilteredObs;
    Eigen::MatrixXd signal(2, 3);
    signal << 1.0, -2.0, 3.0, 4.0, 5.0, 6.0;
    {
        fitsIO<file_type_enum::write_fits, CCfits::ExtHDU *> output(base.string());
        for (Eigen::Index row = 0; row < 2; ++row) {
            output.add_hdu("signal_" + std::to_string(row), signal);
            // Exercise the production header writer, with output scope carried
            // by the same MapType used by Pointing::output and its map writer.
            pointing.add_pointing_fit_header_keys<map_type>(
                *output.hdus.back(), table, row);
        }
    }
    fitsIO<file_type_enum::read_fits, CCfits::ExtHDU *> input(base.string() + ".fits");
    for (Eigen::Index row = 0; row < 2; ++row) {
        SCOPED_TRACE(row);
        const auto name = "signal_" + std::to_string(row);
        const auto actual = input.get_hdu(name);
        EXPECT_TRUE(actual.isApprox(signal, 0.0));
        auto &hdu = input.pfits->extension(name);
        int enabled = -1, valid = -1;
        ASSERT_NO_THROW(hdu.readKey("POINTING.fit_enabled", enabled));
        ASSERT_NO_THROW(hdu.readKey("POINTING.fit_valid", valid));
        EXPECT_EQ(enabled, observation ? 1 : 0);
        if constexpr (observation) {
            EXPECT_EQ(valid, pointing.fit_valid(row));
        } else {
            EXPECT_EQ(valid, 0);
        }
        for (Eigen::Index column = 0; column < pointing.ppt_header.size(); ++column) {
            const auto key = "POINTING." + pointing.ppt_header[column];
            double value = -1.0;
            if constexpr (observation) {
                ASSERT_NO_THROW(hdu.readKey(key, value)) << key;
                EXPECT_DOUBLE_EQ(value, static_cast<double>(table(row, column))) << key;
            } else {
                if (column == 0) {
                    ASSERT_NO_THROW(hdu.readKey(key, value));
                    EXPECT_DOUBLE_EQ(value, pointing.map_indices.maps_to_arrays(row));
                } else {
                    EXPECT_THROW(hdu.readKey(key, value), CCfits::HDU::NoSuchKeyword) << key;
                }
            }
        }
    }
}

template <mapmaking::MapType map_type>
void check_pointing_output_scope() {
    ensure_metadata_logger();
    MetadataDirectory directory;
    Pointing pointing;
    pointing.logger = spdlog::get("citlali_logger");
    pointing.typed_config.pointing.fit_gaussian = true;
    pointing.map_indices.maps_to_arrays.resize(2);
    pointing.map_indices.maps_to_arrays << 0, 2;
    pointing.fit_valid.resize(2);
    pointing.fit_valid << 1, 0;
    Eigen::MatrixXf observation_table(2, pointing.ppt_header.size());
    for (Eigen::Index row = 0; row < observation_table.rows(); ++row) {
        for (Eigen::Index column = 0; column < observation_table.cols(); ++column) {
            observation_table(row, column) = static_cast<float>(100 * (row + 1) + column);
        }
        observation_table(row, 0) = pointing.map_indices.maps_to_arrays(row);
    }
    // Distinct observation measurements and validity must remain unchanged on
    // observation outputs, and must never be borrowed for either coadd scope.
    check_pointing_fit_header<map_type>(pointing, observation_table,
                                        directory.path / "with-observation-state");
    EXPECT_EQ(pointing.fit_valid(0), 1);
    EXPECT_EQ(pointing.fit_valid(1), 0);
    if constexpr (map_type == mapmaking::RawCoadd || map_type == mapmaking::FilteredCoadd) {
        // Coadds must also work with no observation fit storage at all. This
        // catches any attempt to read an empty table or stale fit_valid row.
        pointing.fit_valid.resize(0);
        check_pointing_fit_header<map_type>(pointing, Eigen::MatrixXf{},
                                            directory.path / "without-observation-state");
    }
}

TEST(pointing_fit_metadata, raw_observation_preserves_measurements_and_validity) {
    check_pointing_output_scope<mapmaking::RawObs>();
}

TEST(pointing_fit_metadata, filtered_observation_preserves_measurements_and_validity) {
    check_pointing_output_scope<mapmaking::FilteredObs>();
}

TEST(pointing_fit_metadata, raw_coadd_omits_measurements_and_marks_fit_invalid) {
    check_pointing_output_scope<mapmaking::RawCoadd>();
}

TEST(pointing_fit_metadata, filtered_coadd_omits_measurements_and_marks_fit_invalid) {
    check_pointing_output_scope<mapmaking::FilteredCoadd>();
}

// Supply the observation fields consumed by the actual final-header entry point.
// The empty array list intentionally avoids unrelated beam geometry/calibration.
void prepare_metadata_header(Engine &engine) {
    engine.typed_config.timestream.raw_time_chunk.flux_calibration_enabled = false;
    engine.typed_config.timestream.raw_time_chunk.extinction_correction_enabled = false;
    engine.typed_config.timestream.processed_time_chunk.clean.enabled = false;
    engine.typed_config.beammap.reference.subtract_reference_detector = false;
    engine.telescope.source_name = "fixture";
    engine.telescope.project_id = "metadata-regression";
    engine.telescope.obs_goal = "metadata";
    engine.telescope.fsmp = 122.0;
    for (const auto key : {"Header.M2.XReq", "Header.M2.YReq", "Header.M2.ZReq"}) {
        engine.telescope.tel_header[key] = Eigen::VectorXd::Zero(1);
    }
    for (const auto key : {"TelElAct", "TelAzAct", "ActParAng"}) {
        engine.telescope.tel_data[key] = Eigen::VectorXd::Zero(1);
    }
    engine.calib.arrays.resize(0);
    engine.observation_dates.date_obs.push_back("2026-01-13");
}

TEST(ptc_metadata, production_beammap_header_preserves_completed_iteration) {
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
        prepare_metadata_header(engine);

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
        // The outer counter must stay distinct; finalization must not copy it
        // over the Beammap writer's completed fitting iteration.
        auto *buffer = &engine.omb;
        ASSERT_NO_THROW(engine.add_tod_header(buffer));
        EXPECT_EQ(engine.iteration.fruit_iter, 2);
        {
            netCDF::NcFile file(filename, netCDF::NcFile::read);
            const auto iteration = file.getVar("FRUITLOOPS_ITER");
            EXPECT_EQ(iteration.getType().getId(), NC_INT);
            ASSERT_EQ(iteration.getDimCount(), 1);
            EXPECT_EQ(iteration.getDim(0).getSize(), 1U);
            int value = -1;
            iteration.getVar(&value);
            EXPECT_EQ(value, 4);
            EXPECT_EQ(file.getVars("FRUITLOOPS_ITER").size(), 1U);
            double sample_rate = 0;
            file.getVar("SAMPRATE").getVar(&sample_rate);
            EXPECT_EQ(sample_rate, 122.0);
        }
    }
}

TEST(ptc_metadata, production_other_stream_headers_use_outer_iteration) {
    ensure_metadata_logger();
    for (const auto reduction : {citlali::config::ReductionType::pointing,
                                 citlali::config::ReductionType::beammap}) {
        for (const auto mode : {citlali::config::TodStreamOutputMode::full,
                                citlali::config::TodStreamOutputMode::mini}) {
            MetadataDirectory directory;
            Engine engine;
            engine.logger = spdlog::get("citlali_logger");
            engine.typed_config.runtime.reduction_type = reduction;
            engine.runtime_config_provenance = citlali::config::make_runtime_config_provenance(
                engine.typed_config.runtime, false);
            engine.typed_config.timestream.output.processed_time_chunk.mode = mode;
            engine.typed_config.timestream.output.raw_time_chunk.mode = mode;
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
            engine.tod_outputs.n_rtc_output_scans = 1;
            engine.tod_outputs.n_ptc_output_scans = 1;
            engine.iteration.fruit_iter = 2;
            engine.omb.sig_unit = "mJy/beam";
            prepare_metadata_header(engine);
            ASSERT_NO_THROW(engine.create_tod_files<engine_utils::toltecIO::rtc_timestream>());
            if (reduction == citlali::config::ReductionType::pointing) {
                ASSERT_NO_THROW(engine.create_tod_files<engine_utils::toltecIO::ptc_timestream>());
            }
            engine.iteration.fruit_iter = 7;
            auto *buffer = &engine.omb;
            ASSERT_NO_THROW(engine.add_tod_header(buffer));
            for (const auto &[stream, filename] : engine.output_paths.tod_filename) {
                SCOPED_TRACE(stream);
                netCDF::NcFile file(filename, netCDF::NcFile::read);
                int value = -1;
                file.getVar("FRUITLOOPS_ITER").getVar(&value);
                EXPECT_EQ(value, 7);
            }
        }
    }
}

TEST(ptc_metadata, beammap_header_rejects_missing_or_malformed_iteration) {
    for (int shape = 0; shape < 5; ++shape) {
        SCOPED_TRACE(shape);
        MetadataDirectory directory;
        netCDF::NcFile file((directory.path / "bad-header.nc").string(), netCDF::NcFile::replace);
        if (shape == 1) {
            file.addVar("FRUITLOOPS_ITER", netCDF::ncDouble, file.addDim("iter", 1));
        } else if (shape == 2) {
            file.addVar("FRUITLOOPS_ITER", netCDF::ncInt, file.addDim("iter", 2));
        } else if (shape == 3) {
            file.addVar("FRUITLOOPS_ITER", netCDF::ncInt); // Scalar is not int32[1].
        } else if (shape == 4) {
            const auto dim = file.addDim("iter", 1);
            file.addVar("FRUITLOOPS_ITER", netCDF::ncInt,
                        std::vector<netCDF::NcDim>{dim, dim});
        }
        EXPECT_THROW(citlali::pipeline::add_tod_auxiliary_metadata_vars(
            file, 122.0, "fixture.ecsv", 7,
            citlali::pipeline::TodIterationOwner::beammap_ptc), std::runtime_error);
        EXPECT_TRUE(file.getVar("SAMPRATE").isNull());
        if (shape == 0) EXPECT_TRUE(file.getVar("FRUITLOOPS_ITER").isNull());
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
