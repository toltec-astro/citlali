#include <citlali/core/pipeline/config_source_manifest.h>
#include <citlali/core/pipeline/kids_external_config.h>
#include <citlali/core/pipeline/kids_external_provenance.h>
#include <citlali/core/pipeline/output_config_copy.h>
#include <citlali/core/utils/sha256.h>

#include <gtest/gtest.h>
#include <tula/config/yamlconfig.h>
#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {

struct NullLogger {
    template <class... Args>
    void debug(Args &&...) const {}
};

tula::config::YamlConfig kids_test_config(bool extra_output = true) {
    return tula::config::YamlConfig::from_str(
        "kids:\n"
        "  fitter:\n"
        "    modelspec: gainlintrend\n"
        "    weight_window:\n"
        "      type: lorentz\n"
        "      fwhm_Hz: 15000.0\n"
        "  solver:\n"
        "    fitreportdir: /tmp/fits\n"
        "    parallel_policy: seq\n"
        "    extra_output: " +
        std::string{extra_output ? "true\n" : "false\n"});
}

TEST(kids_external_config, records_requested_and_effective_identity) {
    auto config = kids_test_config();
    const auto plan = citlali::pipeline::make_kids_external_config_plan(
        config, citlali::config::TodType::qs, "toltec.1", "04088da");

    EXPECT_TRUE(plan.initialized);
    EXPECT_EQ(plan.data_schema, "toltec.1");
    EXPECT_EQ(plan.dependency_version, "04088da");
    EXPECT_EQ(plan.selected_tod_type, citlali::config::TodType::qs);
    EXPECT_EQ(plan.requested.values.fitter.modelspec, "gainlintrend");
    EXPECT_DOUBLE_EQ(plan.requested.values.fitter.weight_window_fwhm_hz,
                     15000.0);
    EXPECT_TRUE(plan.requested.solver_extra_output_present);
    EXPECT_TRUE(plan.requested.values.solver.extra_output);
    EXPECT_FALSE(plan.effective.values.solver.extra_output);
    EXPECT_TRUE(plan.effective.solver_extra_output_forced_disabled);
    EXPECT_NO_THROW(
        citlali::pipeline::require_valid_kids_external_config_plan(plan));
}

TEST(kids_external_config, declares_all_four_tod_types_supported) {
    EXPECT_EQ(citlali::pipeline::supported_kids_tod_types.size(), 4U);
    for (const auto type : citlali::pipeline::supported_kids_tod_types) {
        EXPECT_TRUE(citlali::pipeline::is_supported_kids_tod_type(type));
    }
}

TEST(kids_external_config, serializes_external_contract) {
    auto config = kids_test_config(false);
    const auto plan = citlali::pipeline::make_kids_external_config_plan(
        config, citlali::config::TodType::xs, "toltec.1", "04088da");
    const auto node = citlali::pipeline::kids_external_provenance_node(plan);

    EXPECT_EQ(node["schema_version"].as<std::string>(),
              "citlali-kids-external-provenance-v1");
    EXPECT_EQ(node["authority"].as<std::string>(), "kidscpp");
    EXPECT_EQ(node["config_schema"].as<std::string>(),
              "citlali-kidscpp-bridge-v1");
    EXPECT_EQ(node["data_schema"].as<std::string>(), "toltec.1");
    EXPECT_EQ(node["dependency"]["version"].as<std::string>(), "04088da");
    ASSERT_EQ(node["supported_tod_types"].size(), 4U);
    EXPECT_EQ(node["selected_tod_type"].as<std::string>(), "xs");
    EXPECT_FALSE(node["effective"]["values"]["solver"]["extra_output"]
                     .as<bool>());
}

TEST(sha256, matches_standard_vectors) {
    EXPECT_EQ(citlali::utils::sha256(""),
              "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    EXPECT_EQ(citlali::utils::sha256("abc"),
              "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    EXPECT_EQ(
        citlali::utils::sha256(
            "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
}

TEST(config_source_manifest, preserves_order_hashes_and_colliding_basenames) {
    namespace fs = std::filesystem;
    const auto root = fs::path(::testing::TempDir()) /
                      "citlali_external_config_provenance";
    fs::remove_all(root);
    fs::create_directories(root / "first");
    fs::create_directories(root / "second");
    fs::create_directories(root / "reduced");
    const auto first = root / "first" / "config.yaml";
    const auto second = root / "second" / "config.yaml";
    {
        std::ofstream(first) << "value: 1\n";
        std::ofstream(second) << "value: 2\n";
    }
    const std::vector<std::string> sources{first.string(), second.string()};
    auto logger = std::make_shared<NullLogger>();
    citlali::pipeline::copy_config_files_to_reduction_dir(
        sources, (root / "reduced").string(), logger);
    citlali::pipeline::write_config_source_manifest(
        root / "reduced", sources, "value: 2\n");

    EXPECT_TRUE(fs::exists(root / "reduced" / "source_000_config.yaml"));
    EXPECT_TRUE(fs::exists(root / "reduced" / "source_001_config.yaml"));
    const auto manifest = YAML::LoadFile(
        (root / "reduced" / "config_source_manifest.yaml").string());
    EXPECT_EQ(manifest["schema_version"].as<std::string>(),
              "citlali-config-source-manifest-v1");
    ASSERT_EQ(manifest["sources"].size(), 2U);
    EXPECT_EQ(manifest["sources"][0]["precedence"].as<std::size_t>(), 0U);
    EXPECT_EQ(manifest["sources"][1]["precedence"].as<std::size_t>(), 1U);
    EXPECT_NE(manifest["sources"][0]["sha256"].as<std::string>(),
              manifest["sources"][1]["sha256"].as<std::string>());
    EXPECT_EQ(manifest["merged"]["sha256"].as<std::string>(),
              citlali::utils::sha256_file(
                  root / "reduced" / "citlali_merged_config.yaml"));
    fs::remove_all(root);
}

}  // namespace
