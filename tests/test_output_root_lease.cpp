#include <citlali/core/error/error.h>
#include <citlali/core/pipeline/output_root_lease.h>

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <string>

const char *output_root_lease_header_lock_name();

namespace {

class TemporaryOutputRoots {
public:
    TemporaryOutputRoots() {
        const auto suffix = std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count());
        base_ = std::filesystem::temp_directory_path() /
                ("citlali-output-root-lease-" + suffix);
    }

    ~TemporaryOutputRoots() {
        std::error_code ignored;
        std::filesystem::remove_all(base_, ignored);
    }

    std::filesystem::path root(const std::string &name) const {
        return base_ / name;
    }

private:
    std::filesystem::path base_;
};

}  // namespace

TEST(output_root_lease, public_header_links_across_translation_units) {
    EXPECT_STREQ(output_root_lease_header_lock_name(),
                 citlali::pipeline::output_root_lock_filename);
}

TEST(output_root_lease, rejects_a_concurrent_owner) {
    TemporaryOutputRoots roots;
    const auto output_root = roots.root("shared");
    citlali::pipeline::OutputRootLease first(output_root);

    try {
        citlali::pipeline::OutputRootLease second(output_root);
        FAIL() << "expected a concurrent output-root lease to fail";
    }
    catch (const citlali::error::Error &error) {
        EXPECT_EQ(error.code(), citlali::error::Code::output);
        EXPECT_NE(std::string{error.what()}.find("already in use"),
                  std::string::npos);
    }
}

TEST(output_root_lease, releases_ownership_at_scope_exit) {
    TemporaryOutputRoots roots;
    const auto output_root = roots.root("sequential");
    {
        citlali::pipeline::OutputRootLease first(output_root);
    }

    EXPECT_NO_THROW(citlali::pipeline::OutputRootLease second(output_root));
}

TEST(output_root_lease, permits_independent_output_roots) {
    TemporaryOutputRoots roots;
    citlali::pipeline::OutputRootLease first(roots.root("first"));
    EXPECT_NO_THROW(
        citlali::pipeline::OutputRootLease second(roots.root("second")));
}
