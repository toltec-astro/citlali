#include <citlali/core/pipeline/iteration_lifecycle.h>
#include <citlali/core/pipeline/product_index_file.h>
#include <citlali/core/pipeline/raw_timestream_execution_plan.h>

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>

namespace {

std::string read_text(const std::filesystem::path &path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to read test publication");
    }
    return {std::istreambuf_iterator<char>{input}, {}};
}

struct PublicationTestLogger {
    template <class... Args>
    void info(const char *, Args &&...) const {}
};

struct PublicationTestEngine {
    citlali::pipeline::RawTimestreamExecutionPlan raw_timestream_plan;
    struct {
        std::string redu_dir_name = "unused";
    } output_paths;
};

struct PublicationTestTodProc {
    PublicationTestEngine state;
    int index_write_count = 0;

    PublicationTestEngine &engine() { return state; }
    void make_index_file(const std::string &) { ++index_write_count; }
};

TEST(SciAlignNativePublication,
     IterationIndexIsDeferredUntilNativeCanonicalPublication) {
    PublicationTestTodProc todproc;
    citlali::pipeline::RawTimestreamObservationState observation;
    observation.native_consumer_route =
        citlali::pipeline::NativeConsumerRoute::native_required;
    todproc.state.raw_timestream_plan.observation = observation;
    const auto logger = std::make_shared<PublicationTestLogger>();

    citlali::pipeline::make_reduction_iteration_index_file(todproc, logger);
    EXPECT_EQ(todproc.index_write_count, 0);

    todproc.state.raw_timestream_plan.observation->native_consumer_route =
        citlali::pipeline::NativeConsumerRoute::legacy_inactive;
    citlali::pipeline::make_reduction_iteration_index_file(todproc, logger);
    EXPECT_EQ(todproc.index_write_count, 1);
}

TEST(SciAlignNativePublication,
     RequiredProductsGateAtomicDeterministicIndexReplacement) {
    const auto root = std::filesystem::temp_directory_path() /
        "citlali_sci_align_native_publication_v2";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root / "nested");
    const auto required = root / "native_cohort_product_provenance.yaml";
    {
        std::ofstream output(required);
        ASSERT_TRUE(output.good());
        output << "complete: true\n";
    }
    {
        std::ofstream output(root / "nested" / "product.fits");
        ASSERT_TRUE(output.good());
        output << "fixture\n";
    }
    {
        std::ofstream output(root / "citlali.log.gz");
        ASSERT_TRUE(output.good());
        output << "still open\n";
    }

    citlali::pipeline::write_final_product_index_file(root, {required});
    const auto root_index = root / "index.yaml";
    const auto child_index = root / "nested" / "index.yaml";
    const auto first_root = read_text(root_index);
    const auto first_child = read_text(child_index);
    EXPECT_FALSE(first_root.empty());
    EXPECT_FALSE(first_child.empty());

    citlali::pipeline::write_final_product_index_file(root, {required});
    EXPECT_EQ(read_text(root_index), first_root);
    EXPECT_EQ(read_text(child_index), first_child);
    EXPECT_EQ(first_root.find("- index.yaml"), std::string::npos);
    EXPECT_EQ(first_child.find("- index.yaml"), std::string::npos);
    const auto root_node = YAML::Load(first_root);
    EXPECT_EQ(root_node["schema_version"].as<std::string>(),
              "citlali-product-index-v2");
    ASSERT_TRUE(root_node["products"].IsSequence());
    bool required_is_bound = false;
    bool live_log_is_unbound = false;
    for (const auto &product : root_node["products"]) {
        if (product["name"].as<std::string>() == required.filename()) {
            required_is_bound = true;
            EXPECT_EQ(product["kind"].as<std::string>(), "file");
            EXPECT_FALSE(product["sha256"].as<std::string>().empty());
            EXPECT_EQ(product["size_bytes"].as<std::uintmax_t>(),
                      std::filesystem::file_size(required));
        }
        if (product["name"].as<std::string>() == "citlali.log.gz") {
            live_log_is_unbound = true;
            EXPECT_EQ(product["kind"].as<std::string>(),
                      "operational_mutable_file");
            EXPECT_EQ(product["checksum_policy"].as<std::string>(),
                      "excluded_live_at_canonical_publication");
            EXPECT_FALSE(product["sha256"]);
        }
    }
    EXPECT_TRUE(required_is_bound);
    EXPECT_TRUE(live_log_is_unbound);

    std::filesystem::remove(required);
    EXPECT_THROW(
        citlali::pipeline::write_final_product_index_file(root, {required}),
        std::logic_error);
    EXPECT_EQ(read_text(root_index), first_root);
    EXPECT_EQ(read_text(child_index), first_child);
    std::filesystem::remove_all(root);
}

TEST(SciAlignNativePublication,
     ProductIndexTreatsCompactAptV2BundleAsOpaque) {
    const auto root = std::filesystem::temp_directory_path() /
        "citlali_sci_align_opaque_apt_v2";
    std::filesystem::remove_all(root);
    const auto bundle = root / "baseline.apt-v2";
    std::filesystem::create_directories(bundle);
    const auto required = root / "required.yaml";
    {
        std::ofstream output(required);
        ASSERT_TRUE(output.good());
        output << "complete: true\n";
    }
    for (const auto *name : {"manifest.ecsv", "manifest.ecsv.sha256",
                             "sha256-fixture.apt.ecsv"}) {
        std::ofstream output(bundle / name);
        ASSERT_TRUE(output.good());
        output << "fixture\n";
    }
    const auto members_before =
        citlali::pipeline::sorted_directory_entries(bundle);

    citlali::pipeline::write_final_product_index_file(root, {required});

    EXPECT_TRUE(std::filesystem::is_regular_file(root / "index.yaml"));
    EXPECT_FALSE(std::filesystem::exists(bundle / "index.yaml"));
    EXPECT_EQ(citlali::pipeline::sorted_directory_entries(bundle),
              members_before);
    EXPECT_NE(read_text(root / "index.yaml").find("- baseline.apt-v2"),
              std::string::npos);
    const auto index = YAML::LoadFile((root / "index.yaml").string());
    bool opaque_is_bound = false;
    for (const auto &product : index["products"]) {
        if (product["name"].as<std::string>() == "baseline.apt-v2") {
            opaque_is_bound = true;
            EXPECT_EQ(product["kind"].as<std::string>(),
                      "opaque_product_bundle");
            EXPECT_FALSE(
                product["root_manifest_sha256"].as<std::string>().empty());
            EXPECT_FALSE(
                product["root_receipt_sha256"].as<std::string>().empty());
        }
    }
    EXPECT_TRUE(opaque_is_bound);
    std::filesystem::remove_all(root);
}

}  // namespace
