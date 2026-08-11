#include <citlali/core/cli/exception_reporting.h>
#include <citlali/core/pipeline/timestream_output_context.h>
#include <citlali/core/utils/netcdf_io.h>

#include <gtest/gtest.h>

#include <netcdf>

#include <array>
#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace {

void create_row_file(const std::filesystem::path &path) {
    netCDF::NcFile file(path.string(), netCDF::NcFile::replace);
    const auto rows = file.addDim("rows", 3);
    const auto value = file.addVar("value", netCDF::ncInt, rows);
    const std::array<int, 3> initial{-1, -1, -1};
    value.putVar(initial.data());
}

void write_row(const std::filesystem::path &path, std::size_t row, int value) {
    netCDF::NcFile file(path.string(), netCDF::NcFile::write);
    file.getVar("value").putVar(
        std::vector<std::size_t>{row}, std::vector<std::size_t>{1}, &value);
}

std::array<int, 3> read_rows(const std::filesystem::path &path) {
    netCDF::NcFile file(path.string(), netCDF::NcFile::read);
    std::array<int, 3> values{};
    file.getVar("value").getVar(values.data());
    return values;
}

TEST(ordered_writer, serializes_out_of_order_workers) {
    citlali::pipeline::OrderedWriter writer;
    std::mutex values_mutex;
    std::vector<int> values;

    auto second = std::async(std::launch::async, [&] {
        writer.write_when_ready(1, [&] {
            std::lock_guard<std::mutex> lock(values_mutex);
            values.push_back(1);
        });
    });
    auto first = std::async(std::launch::async, [&] {
        writer.write_when_ready(0, [&] {
            std::lock_guard<std::mutex> lock(values_mutex);
            values.push_back(0);
        });
    });

    first.get();
    second.get();
    EXPECT_EQ(values, (std::vector<int>{0, 1}));
}

TEST(ordered_writer, failure_cancels_and_wakes_waiters) {
    citlali::pipeline::OrderedWriter writer;
    std::atomic<bool> later_write_ran{false};
    std::promise<void> waiter_started;
    auto waiter_ready = waiter_started.get_future();

    auto later = std::async(std::launch::async, [&] {
        waiter_started.set_value();
        writer.write_when_ready(1, [&] { later_write_ran = true; });
    });
    waiter_ready.get();

    EXPECT_THROW(
        writer.write_when_ready(
            0, [] { throw std::runtime_error("injected write failure"); }),
        std::runtime_error);
    EXPECT_THROW(later.get(), std::runtime_error);
    EXPECT_FALSE(later_write_ran.load());

    citlali::pipeline::OrderedWriter next_run_writer;
    bool next_run_write_ran = false;
    EXPECT_NO_THROW(next_run_writer.write_when_ready(
        0, [&] { next_run_write_ran = true; }));
    EXPECT_TRUE(next_run_write_ran);
}

TEST(ordered_writer, required_output_failure_cancels_other_streams) {
    citlali::pipeline::TimestreamOutputFlags flags;
    flags.write_rtc = true;
    flags.write_ptc = true;
    const auto writers =
        citlali::pipeline::make_timestream_output_writers(flags);
    std::atomic<bool> ptc_write_ran{false};
    std::promise<void> waiter_started;
    auto waiter_ready = waiter_started.get_future();

    auto waiting_ptc = std::async(std::launch::async, [&] {
        waiter_started.set_value();
        writers.write_when_ready(
            writers.ptc, 1, [&] { ptc_write_ran = true; });
    });
    waiter_ready.get();

    EXPECT_NO_THROW(
        writers.write_when_ready(
            writers.rtc, 0,
            [] { throw std::runtime_error("injected RTC write failure"); }));
    EXPECT_NO_THROW(waiting_ptc.get());
    EXPECT_FALSE(ptc_write_ran.load());
    EXPECT_THROW(writers.rethrow_if_failed(), std::runtime_error);
}

TEST(ordered_writer, verifies_required_output_cardinality) {
    citlali::pipeline::TimestreamOutputFlags flags;
    flags.write_rtc = true;
    flags.write_ptcdiag = true;
    const auto writers =
        citlali::pipeline::make_timestream_output_writers(flags);

    EXPECT_TRUE(writers.write_when_ready(writers.rtc, 0, [] {}));
    EXPECT_TRUE(writers.write_when_ready(writers.rtc, 1, [] {}));
    EXPECT_TRUE(writers.write_when_ready(writers.ptcdiag, 0, [] {}));

    const citlali::pipeline::TimestreamOutputExpectations exact{2, 0, 0, 1};
    EXPECT_NO_THROW(writers.verify_complete(exact));

    const citlali::pipeline::TimestreamOutputExpectations missing{3, 0, 0, 1};
    EXPECT_THROW(writers.verify_complete(missing), std::runtime_error);
}

TEST(ordered_writer, derives_standard_and_beammap_output_expectations) {
    struct FakeEngine {
        struct {
            Eigen::MatrixXi scan_indices;
        } telescope;
        struct {
            Eigen::Index n_rtc_output_scans = 0;
            Eigen::Index n_ptc_output_scans = 0;
        } tod_outputs;
    } engine;
    engine.telescope.scan_indices.resize(2, 5);
    engine.tod_outputs.n_rtc_output_scans = 2;
    engine.tod_outputs.n_ptc_output_scans = 3;

    citlali::pipeline::TimestreamOutputFlags flags;
    flags.write_rtc = true;
    flags.write_ptc = true;
    flags.write_rtcdiag = true;
    flags.write_ptcdiag = true;
    const auto standard =
        citlali::pipeline::standard_timestream_output_expectations(
            engine, flags);
    EXPECT_EQ(standard.rtc, 2);
    EXPECT_EQ(standard.ptc, 3);
    EXPECT_EQ(standard.rtcdiag, 5);
    EXPECT_EQ(standard.ptcdiag, 5);

    const auto beammap =
        citlali::pipeline::beammap_timestream_output_expectations(
            engine, flags);
    EXPECT_EQ(beammap.rtc, 2);
    EXPECT_EQ(beammap.ptc, 0);
    EXPECT_EQ(beammap.rtcdiag, 5);
    EXPECT_EQ(beammap.ptcdiag, 0);
}

TEST(ordered_writer, netcdf_failure_leaves_diagnosed_partial_product_and_next_run_recovers) {
    const auto path = std::filesystem::path(::testing::TempDir()) /
                      "citlali_ordered_writer_failure.nc";
    std::filesystem::remove(path);
    create_row_file(path);

    citlali::pipeline::TimestreamOutputFlags flags;
    flags.write_rtc = true;
    const auto writers =
        citlali::pipeline::make_timestream_output_writers(flags);

    std::atomic<bool> later_write_ran{false};
    std::promise<void> later_started;
    std::promise<void> failing_started;
    auto later_ready = later_started.get_future();
    auto failing_ready = failing_started.get_future();

    auto later = std::async(std::launch::async, [&] {
        later_started.set_value();
        return writers.write_when_ready(writers.rtc, 2, [&] {
            later_write_ran = true;
            write_row(path, 2, 30);
        });
    });
    later_ready.get();

    auto failing = std::async(std::launch::async, [&] {
        failing_started.set_value();
        return writers.write_when_ready(
            writers.rtc, 1,
            [&] { write_row(path, 3, 20); });  // Fixed dimension is [0, 3).
    });
    failing_ready.get();

    auto first = std::async(std::launch::async, [&] {
        return writers.write_when_ready(
            writers.rtc, 0, [&] { write_row(path, 0, 10); });
    });

    EXPECT_TRUE(first.get());
    EXPECT_FALSE(failing.get());
    EXPECT_FALSE(later.get());
    EXPECT_FALSE(later_write_ran.load());
    EXPECT_TRUE(writers.failed());
    EXPECT_ANY_THROW(writers.rethrow_if_failed());
    EXPECT_EQ(
        citlali::cli::run_with_exception_reporting([&] {
            writers.rethrow_if_failed();
            return EXIT_SUCCESS;
        }),
        EXIT_FAILURE);
    EXPECT_EQ(read_rows(path), (std::array<int, 3>{10, -1, -1}));

    create_row_file(path);
    const auto next_run_writers =
        citlali::pipeline::make_timestream_output_writers(flags);
    for (Eigen::Index row = 0; row < 3; ++row) {
        EXPECT_TRUE(next_run_writers.write_when_ready(
            next_run_writers.rtc, row,
            [&, row] { write_row(path, static_cast<std::size_t>(row),
                                 static_cast<int>((row + 1) * 10)); }));
    }
    EXPECT_FALSE(next_run_writers.failed());
    EXPECT_NO_THROW(next_run_writers.rethrow_if_failed());
    EXPECT_EQ(read_rows(path), (std::array<int, 3>{10, 20, 30}));

    std::filesystem::remove(path);
}

TEST(ordered_writer,
     required_netcdf_publication_preserves_prior_generation_until_commit) {
    const auto final_path = std::filesystem::path(::testing::TempDir()) /
                            "citlali_atomic_ordered_writer.nc";
    std::filesystem::remove(final_path);
    write_netcdf_atomic(final_path.string(), [](netCDF::NcFile &file) {
        const int value = 7;
        file.addVar("generation", netCDF::ncInt).putVar(&value);
    });

    for (const auto stage : {
             NetcdfAtomicFailureStage::create,
             NetcdfAtomicFailureStage::write,
             NetcdfAtomicFailureStage::sync,
             NetcdfAtomicFailureStage::close,
             NetcdfAtomicFailureStage::publish}) {
        EXPECT_THROW(
            write_netcdf_atomic(
                final_path.string(),
                [](netCDF::NcFile &file) {
                    const int value = 99;
                    file.addVar("generation", netCDF::ncInt).putVar(&value);
                },
                stage),
            DataIOError);
        netCDF::NcFile file(final_path.string(), netCDF::NcFile::read);
        int value = 0;
        file.getVar("generation").getVar(&value);
        EXPECT_EQ(value, 7);
        for (const auto &entry :
             std::filesystem::directory_iterator(final_path.parent_path())) {
            EXPECT_NE(entry.path().filename().string().rfind(
                          final_path.filename().string() +
                              netcdf_atomic_staging_marker,
                          0),
                      0u);
        }
    }

    EXPECT_THROW(
        write_netcdf_staging(final_path.string(), [](netCDF::NcFile &file) {
            const int value = 8;
            file.addVar("generation", netCDF::ncInt).putVar(&value);
            throw std::runtime_error("injected pre-publication failure");
        }),
        std::runtime_error);
    {
        netCDF::NcFile file(final_path.string(), netCDF::NcFile::read);
        int value = 0;
        file.getVar("generation").getVar(&value);
        EXPECT_EQ(value, 7);
    }

    const auto staging = write_netcdf_staging(
        final_path.string(), [](netCDF::NcFile &file) {
            const int value = 8;
            file.addVar("generation", netCDF::ncInt).putVar(&value);
        });
    EXPECT_TRUE(is_netcdf_atomic_staging_path(staging));
    EXPECT_EQ(netcdf_atomic_final_path_from_staging(staging),
              final_path.string());
    EXPECT_EQ(publish_netcdf_atomic_staging(staging), final_path.string());
    EXPECT_FALSE(std::filesystem::exists(staging));
    {
        netCDF::NcFile file(final_path.string(), netCDF::NcFile::read);
        int value = 0;
        file.getVar("generation").getVar(&value);
        EXPECT_EQ(value, 8);
    }
    cleanup_netcdf_atomic_staging(final_path.string());
    EXPECT_TRUE(std::filesystem::exists(final_path));
    std::filesystem::remove(final_path);
}

}  // namespace
