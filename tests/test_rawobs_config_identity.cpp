#include <citlali/core/engine/io.h>
#include <citlali/core/pipeline/config_source_manifest.h>

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

const std::vector<std::string> source_interfaces{
    "hwpr", "lmt", "toltec0", "toltec1", "toltec2", "toltec3",
    "toltec4", "toltec5", "toltec7", "toltec8", "toltec9",
    "toltec11", "toltec12"};
const std::vector<std::string> runtime_interfaces{
    "toltec0", "toltec1", "toltec2", "toltec3", "toltec4", "toltec5",
    "toltec7", "toltec8", "toltec9", "toltec11", "toltec12", "lmt", "hwpr"};

tula::config::YamlConfig make_config(
    const std::vector<std::string> &interfaces = source_interfaces) {
    std::string yaml = "inputs:\n";
    for (const auto *name : {"152390_0_2", "152392_0_2"}) {
        yaml += "  - meta: {name: '" + std::string{name} + "'}\n";
        yaml += "    data_items:\n";
        for (const auto &interface : interfaces) {
            yaml += "      - meta: {interface: " + interface + "}\n";
            yaml += "        filepath: /inputs/" + std::string{name} + "/" +
                    interface + ".nc\n";
        }
        yaml += "    cal_items:\n";
        yaml += "      - type: array_prop_table\n";
        yaml += "        meta: {interface: apt}\n";
        yaml += "        filepath: /apts/" + std::string{name} + ".ecsv\n";
        yaml += "      - type: astrometry\n";
        yaml += "        pointing_offsets:\n";
        yaml += "          - {axes_name: az, value_arcsec: [1.25]}\n";
        yaml += "          - {axes_name: alt, value_arcsec: [-2.5]}\n";
    }
    return tula::config::YamlConfig::from_str(yaml);
}

void expect_item_config_identity(const RawObs &raw,
                                const std::vector<std::string> &expected) {
    ASSERT_EQ(raw.n_data_items(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const auto &item = raw.data_items()[i];
        SCOPED_TRACE(raw.name() + ":" + expected[i]);
        EXPECT_EQ(item.interface(), expected[i]);
        EXPECT_EQ(item.filepath(),
                  "/inputs/" + raw.name() + "/" + expected[i] + ".nc");
        EXPECT_EQ(item.config().get_str(std::tuple{"meta", "interface"}),
                  item.interface());
        EXPECT_EQ(item.config().get_filepath("filepath"), item.filepath());
    }
    EXPECT_EQ(raw.teldata().interface(), "lmt");
    EXPECT_EQ(raw.array_prop_table().filepath(),
              "/apts/" + raw.name() + ".ecsv");
}

TEST(rawobs_config_identity, preserves_request_and_item_configs_after_ordering) {
    auto config = make_config();
    const auto before = config.to_str();
    const SeqIOCoordinator coordinator(config);
    ASSERT_EQ(coordinator.n_inputs(), 2U);
    EXPECT_EQ(config.to_str(), before);
    EXPECT_EQ(coordinator.config().to_str(), before);
    for (const auto &raw : coordinator.inputs()) {
        expect_item_config_identity(raw, runtime_interfaces);
        ASSERT_EQ(raw.kidsdata().size(), 11U);
        ASSERT_TRUE(raw.hwpdata().has_value());
        EXPECT_EQ(raw.hwpdata()->interface(), "hwpr");
    }
}

TEST(rawobs_config_identity, retains_numeric_network_order_without_optional_hwpr) {
    auto config = make_config({"toltec12", "lmt", "toltec2", "toltec0"});
    const auto before = config.to_str();
    const SeqIOCoordinator coordinator(config);
    EXPECT_EQ(config.to_str(), before);
    for (const auto &raw : coordinator.inputs()) {
        expect_item_config_identity(raw, {"toltec0", "toltec2", "toltec12", "lmt"});
        EXPECT_EQ(raw.kidsdata().size(), 3U);
        EXPECT_FALSE(raw.hwpdata().has_value());
    }
}

TEST(rawobs_config_identity, merged_snapshot_retains_input_and_calibration_associations) {
    namespace fs = std::filesystem;
    const auto root = fs::path(::testing::TempDir()) /
                      "citlali_rawobs_config_identity";
    fs::remove_all(root);
    fs::create_directories(root / "reduced");
    auto config = make_config();
    const auto before = config.to_str();
    const auto source = root / "source.yaml";
    std::ofstream(source) << before;
    const SeqIOCoordinator coordinator(config);
    const std::vector<std::string> sources{source.string()};
    citlali::pipeline::copy_config_files_to_reduction_dir(
        sources, (root / "reduced").string(), spdlog::default_logger());
    citlali::pipeline::write_config_source_manifest(
        root / "reduced", sources, config.to_str());
    const auto snapshot = YAML::LoadFile(
        (root / "reduced" / "citlali_merged_config.yaml").string());
    EXPECT_EQ(YAML::Dump(snapshot), before);
    const auto original = YAML::Load(before);
    ASSERT_EQ(snapshot["inputs"].size(), 2U);
    for (std::size_t i = 0; i < 2; ++i) {
        const auto saved = snapshot["inputs"][i];
        EXPECT_EQ(saved["meta"]["name"].as<std::string>(),
                  coordinator.inputs()[i].name());
        ASSERT_EQ(saved["data_items"].size(), source_interfaces.size());
        for (std::size_t j = 0; j < source_interfaces.size(); ++j) {
            EXPECT_EQ(saved["data_items"][j]["meta"]["interface"].as<std::string>(),
                      source_interfaces[j]);
            EXPECT_EQ(saved["data_items"][j]["filepath"].as<std::string>(),
                      original["inputs"][i]["data_items"][j]["filepath"].as<std::string>());
        }
        EXPECT_EQ(YAML::Dump(saved["cal_items"]),
                  YAML::Dump(original["inputs"][i]["cal_items"]));
    }
    fs::remove_all(root);
}

} // namespace
