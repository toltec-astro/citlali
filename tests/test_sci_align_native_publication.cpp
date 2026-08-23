#include <citlali/core/pipeline/product_index_file.h>

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

namespace {

std::string read_text(const std::filesystem::path &path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to read test publication");
    }
    return {std::istreambuf_iterator<char>{input}, {}};
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
    std::filesystem::remove_all(root);
}

}  // namespace
