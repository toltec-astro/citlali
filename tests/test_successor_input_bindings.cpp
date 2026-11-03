#include "successor_input_bindings.h"
#include <citlali/core/utils/sha256.h>
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>
#include <array>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
namespace inputs = citlali::cli::input_bindings;
namespace profiles = citlali::cli::experiment_profiles;
namespace fs = std::filesystem;

void write(const fs::path &path, const std::string &text) {
    std::ofstream output(path);
    output << text;
    ASSERT_TRUE(output.good());
}

TEST(SuccessorInputBindings, NetcdfFirstValuePreservesScalarAndSingletonHeaders) {
    inputs::TemporaryDirectory scratch;
    const auto path = scratch.path() / "headers.nc";
    {
        netCDF::NcFile file(path.string(), netCDF::NcFile::replace);
        const int integer = 42;
        const double real = 2000.0;
        file.addVar("scalar_int", netCDF::ncInt).putVar(&integer);
        file.addVar("scalar_double", netCDF::ncDouble).putVar(&real);
        file.addVar("singleton", netCDF::ncDouble, file.addDim("one", 1))
            .putVar(&real);
    }
    netCDF::NcFile file(path.string(), netCDF::NcFile::read);
    EXPECT_EQ(inputs::read_netcdf_scalar<int>(file, "scalar_int"), 42);
    EXPECT_DOUBLE_EQ(inputs::read_netcdf_scalar<double>(file, "scalar_double"), 2000.0);
    EXPECT_DOUBLE_EQ(inputs::read_netcdf_scalar<double>(file, "singleton"), 2000.0);
}

TEST(SuccessorInputBindings, NetcdfFirstValueIgnoresLaterHeaderValues) {
    inputs::TemporaryDirectory scratch;
    const auto path = scratch.path() / "repeated-headers.nc";
    {
        netCDF::NcFile file(path.string(), netCDF::NcFile::replace);
        const auto three = file.addDim("three", 3);
        const auto two = file.addDim("two", 2);
        const std::array<double, 3> repeated{2000.0, 2000.0, 2000.0};
        const std::array<double, 3> differing{122.0, 244.0, 366.0};
        const std::array<int, 3> integers{19, 20, 21};
        const std::array<int, 6> matrix{7, 8, 9, 10, 11, 12};
        file.addVar("repeated", netCDF::ncDouble, three).putVar(repeated.data());
        file.addVar("differing", netCDF::ncDouble, three).putVar(differing.data());
        file.addVar("integers", netCDF::ncInt, three).putVar(integers.data());
        file.addVar("matrix", netCDF::ncInt, std::vector<netCDF::NcDim>{two, three})
            .putVar(matrix.data());
    }
    netCDF::NcFile file(path.string(), netCDF::NcFile::read);
    EXPECT_DOUBLE_EQ(inputs::read_netcdf_scalar<double>(file, "repeated"), 2000.0);
    EXPECT_DOUBLE_EQ(inputs::read_netcdf_scalar<double>(file, "differing"), 122.0);
    EXPECT_EQ(inputs::read_netcdf_scalar<int>(file, "integers"), 19);
    EXPECT_EQ(inputs::read_netcdf_scalar<int>(file, "matrix"), 7);
}

TEST(SuccessorInputBindings, NetcdfFirstValueRejectsEmptyAndMissingHeaders) {
    inputs::TemporaryDirectory scratch;
    const auto path = scratch.path() / "empty-headers.nc";
    {
        netCDF::NcFile file(path.string(), netCDF::NcFile::replace, netCDF::NcFile::nc4);
        const auto empty = file.addDim("empty");
        const auto two = file.addDim("two", 2);
        file.addVar("empty_vector", netCDF::ncDouble, empty);
        file.addVar("empty_matrix", netCDF::ncInt, std::vector<netCDF::NcDim>{two, empty});
    }
    netCDF::NcFile file(path.string(), netCDF::NcFile::read);
    const auto reject = [&](const std::string &name, const std::string &reason) {
        try {
            (void)inputs::read_netcdf_scalar<double>(file, name);
            FAIL() << "expected rejection for " << name;
        } catch (const std::runtime_error &error) {
            EXPECT_EQ(error.what(), reason);
        }
    };
    reject("empty_vector", "empty NetCDF scalar empty_vector");
    reject("empty_matrix", "empty NetCDF scalar empty_matrix");
    reject("missing", "missing NetCDF scalar missing");
}

TEST(SuccessorInputBindings, TimestampSliceUsesFirstRepeatedHeaderValues) {
    inputs::TemporaryDirectory scratch;
    const auto path = scratch.path() / "timing.nc";
    const std::array<int, 18> timestamps{
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
    {
        netCDF::NcFile file(path.string(), netCDF::NcFile::replace);
        const auto three = file.addDim("three", 3);
        const auto six = file.addDim("six", 6);
        const std::array<double, 3> fpga{256000000.0, 256000000.0, 256000000.0};
        const std::array<double, 3> sample{122.0, 999.0, 1000.0};
        const std::array<int, 3> accumulation{524288, 1, 2};
        file.addVar("Header.Toltec.FpgaFreq", netCDF::ncDouble, three).putVar(fpga.data());
        file.addVar("Header.Toltec.SampleFreq", netCDF::ncDouble, three).putVar(sample.data());
        file.addVar("Header.Toltec.AccumLen", netCDF::ncInt, three).putVar(accumulation.data());
        file.addVar("Data.Toltec.Ts", netCDF::ncInt, std::vector<netCDF::NcDim>{three, six})
            .putVar(timestamps.data());
    }
    double fpga = 0.0, sample = 0.0;
    std::int64_t accumulation = 0;
    const auto slice = inputs::read_timestamp_slice(path, 1, 2, fpga, sample, accumulation);
    EXPECT_DOUBLE_EQ(fpga, 256000000.0);
    EXPECT_DOUBLE_EQ(sample, 122.0);
    EXPECT_EQ(accumulation, 524288);
    ASSERT_EQ(slice.rows(), 2);
    ASSERT_EQ(slice.cols(), 6);
    for (Eigen::Index row = 0; row < slice.rows(); ++row) {
        for (Eigen::Index column = 0; column < slice.cols(); ++column) {
            EXPECT_EQ(slice(row, column), timestamps[(row + 1) * 6 + column]);
        }
    }
}

TEST(SuccessorInputBindings, RuntimeConfigKeepsExactInterfaceOffsets) {
    inputs::TemporaryDirectory scratch;
    const auto path = scratch.path() / "config.yaml";
    write(path, "interface_sync_offset: [{toltec12: -0.25}, {toltec0: 0.125}]\n"
                "kids: {fitter: {modelspec: gainlintrend}}\n");
    const auto config = inputs::load_runtime_config(path);
    EXPECT_EQ(config.kids_model, "gainlintrend");
    EXPECT_EQ(config.interface_offsets_sec[12], -0.25);
    EXPECT_EQ(config.interface_offsets_sec[0], 0.125);
    EXPECT_TRUE(config.interface_offset_present[12]);
    EXPECT_FALSE(config.interface_offset_present[1]);
    for (const std::string offsets : {
        "[{toltec0: 0}, {toltec0: 0}]", "[{toltec13: 0}]",
        "[{toltec1x: 0}]", "[{toltec0: .nan}]"}) {
        write(path, "interface_sync_offset: " + offsets + "\n"
                    "kids: {fitter: {modelspec: gainlintrend}}\n");
        EXPECT_THROW(inputs::load_runtime_config(path), std::runtime_error);
    }
}

TEST(SuccessorInputBindings, RawBindingRejectsChangedBytes) {
    inputs::TemporaryDirectory scratch;
    const auto path = scratch.path() / "raw.nc";
    write(path, "synthetic raw bytes");
    citlali::pipeline::CanonicalAptRawSourceBindingV2 binding;
    binding.byte_count = fs::file_size(path);
    binding.content_sha256 = "sha256:" + citlali::utils::sha256_file(path);
    EXPECT_NO_THROW(inputs::verify_raw_file(path, binding));
    write(path, "different raw bytes");
    EXPECT_THROW(inputs::verify_raw_file(path, binding), std::runtime_error);
}

TEST(SuccessorInputBindings, TelescopeProfilesRetainHistoricalScopeGuards) {
    const fs::path absent = "not-an-observation.nc";
    using citlali::pipeline::NativeObservationScope;
    const auto reject = [&](NativeObservationScope scope,
                            inputs::TelescopeBindingProfile profile,
                            const std::string &reason) {
        try {
            (void)inputs::load_telescope(absent, scope, profile);
            FAIL() << "expected telescope scope rejection";
        } catch (const std::runtime_error &error) {
            EXPECT_EQ(error.what(), reason);
        }
    };
    reject(NativeObservationScope{152390, 0, 2}, {.bounded_repeat_census = true},
           "repeat telescope binding is outside the fixed census scope");
    reject(NativeObservationScope{152390, 0, 2}, {.bounded_point_map = true},
           "configured telescope binding outside selected scope");
    reject(NativeObservationScope{152389, 0, 2}, {.configured_digest = "digest"},
           "configured telescope requires POINT route");
    reject(NativeObservationScope{152385, 0, 2},
           {.bounded_point_map = true, .configured_digest = "digest", .bounded_oof = true},
           "configured telescope binding outside selected scope");
    EXPECT_EQ(profiles::IdentityRouteExperimentProfile::telescope_record_count, 62109);
}

TEST(SuccessorInputBindings, MappingDefaultRemainsExplicitHistoricalIdentity) {
    inputs::NetworkInput input;
    input.source.network = 11;
    input.source.interface_name = "toltec11";
    input.source.content_sha256 = "raw-digest";
    input.tune_sha256 = "tune-digest";
    input.accumulation_length = 10;
    input.tune_accumulation_length = 20;
    input.fpga_frequency_hz = 1000;
    inputs::RuntimeConfig config;
    config.kids_model = "gainlintrend";
    const auto mapping = inputs::mapping_identity(input, config);
    EXPECT_EQ(mapping->applicability_domain_id, "observation=152390:network=11");
    EXPECT_EQ(mapping->producer_instance_id, "raw-digest:network=11:tune=sha256:tune-digest");
    EXPECT_EQ(mapping->native_time_unit_id, "second");
    EXPECT_EQ(mapping->x.unit_or_scale_id, "dimensionless");
}

TEST(SuccessorInputBindings, InvalidMeasuredMembersStayAvailableAndDistinct) {
    citlali::pipeline::NativePairedReadoutMatrix values(1, 3);
    values << 2.0, 3.0, std::numeric_limits<double>::quiet_NaN();
    const auto states = inputs::member_states(values, {true, false, true});
    ASSERT_EQ(states.size(), 3);
    EXPECT_TRUE(states[0].valid());
    EXPECT_TRUE(states[1].payload_available());
    EXPECT_FALSE(states[1].producer_valid());
    EXPECT_TRUE(states[1].finite_payload());
    EXPECT_TRUE(states[2].payload_available());
    EXPECT_FALSE(states[2].finite_payload());
    EXPECT_THROW(inputs::member_states(values, {true}), std::runtime_error);
}

TEST(SuccessorInputBindings, TuneNormalizationPreservesOriginalAndOwnsScratch) {
    fs::path owned;
    {
        inputs::TemporaryDirectory scratch;
        owned = scratch.path();
        inputs::NetworkInput input;
        input.source.network = 3;
        input.tune_accumulation_length = 42;
        input.tune_path = owned / "original.ecsv";
        const std::string original = "# header\n#   - Header.Toltec.AccumLen: 42\n1 2\n";
        write(input.tune_path, original);
        const auto digest = citlali::utils::sha256_file(input.tune_path);
        const auto normalized = inputs::normalized_tune_report(input, owned);
        EXPECT_TRUE(fs::is_regular_file(normalized));
        EXPECT_EQ(citlali::utils::sha256_file(input.tune_path), digest);
        std::ifstream stream(normalized);
        std::string text((std::istreambuf_iterator<char>(stream)), {});
        EXPECT_EQ(text, "# header\n#   - accumlen: 42\n1 2\n");
    }
    EXPECT_FALSE(fs::exists(owned));
}

TEST(SuccessorExperimentProfiles, SavedMaskedCovarianceScopeIsNotBroadened) {
    for (const int observation : {152389, 152390, 152430, 152431}) {
        for (int network = 0; network < 13; ++network) {
            YAML::Node input;
            input["observation"] = observation;
            input["network"] = network;
            EXPECT_EQ(profiles::masked_covariance_count_152430(input),
                      observation == 152430 && (network == 11 || network == 12));
        }
    }
    auto foreign = YAML::Load("{observation: 152390}");
    EXPECT_FALSE(profiles::masked_covariance_count_152430(foreign));
    auto incomplete = YAML::Load("{observation: 152430}");
    EXPECT_THROW(profiles::masked_covariance_count_152430(incomplete), YAML::Exception);
}
} // namespace
