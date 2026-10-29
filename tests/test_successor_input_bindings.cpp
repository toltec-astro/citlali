#include "successor_input_bindings.h"
#include <citlali/core/utils/sha256.h>
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>
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
