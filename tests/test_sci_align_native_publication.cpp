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

}  // namespace
