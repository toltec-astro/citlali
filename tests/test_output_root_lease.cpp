#include <citlali/core/error/error.h>
#include <citlali/core/pipeline/output_root_lease.h>

#include <gtest/gtest.h>
#include <citlali/core/pipeline/staged_output_directory.h>
#include <atomic>
#include <array>
#include <fstream>
#include <future>
#include <thread>

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


TEST(staged_output_directory, two_contenders_publish_exactly_one_result) {
    TemporaryOutputRoots roots;
    const auto output = roots.root("contested");
    std::promise<void> start;
    const auto ready = start.get_future().share();
    std::atomic<int> attempted{0}, won{0};
    std::array<std::string, 2> errors;
    std::array<std::thread, 2> contenders;
    for (int i = 0; i < 2; ++i) {
        contenders[i] = std::thread([&, i] {
            ready.wait();
            try {
                citlali::pipeline::StagedOutputDirectory attempt(output);
                ++attempted;
                while (attempted.load() != 2) std::this_thread::yield();
                std::ofstream(attempt.staging() / "owner") << i;
                attempt.publish();
                ++won;
            } catch (const std::exception &error) {
                errors[i] = error.what();
                ++attempted;
            }
        });
    }
    start.set_value();
    for (auto &thread : contenders) thread.join();
    ASSERT_EQ(won.load(), 1);
    ASSERT_EQ(errors[0].empty() + errors[1].empty(), 1);
    const auto &failure = errors[0].empty() ? errors[1] : errors[0];
    EXPECT_NE(failure.find("pending output already claimed"), std::string::npos);
    int owner = -1;
    std::ifstream(output / "owner") >> owner;
    EXPECT_EQ(owner, errors[0].empty() ? 0 : 1);
    EXPECT_FALSE(std::filesystem::exists(output.string() + ".pending"));
}

TEST(staged_output_directory, preserves_failed_attempt_and_rejects_reuse) {
    TemporaryOutputRoots roots;
    const auto output = roots.root("failed");
    {
        citlali::pipeline::StagedOutputDirectory attempt(output);
        std::ofstream(attempt.staging() / "failure") << "diagnostic";
    }
    EXPECT_FALSE(std::filesystem::exists(output));
    EXPECT_TRUE(std::filesystem::is_regular_file(output.string() + ".pending/failure"));
    EXPECT_THROW(citlali::pipeline::StagedOutputDirectory second(output), std::exception);
}

TEST(staged_output_directory, never_replaces_an_intervening_destination) {
    TemporaryOutputRoots roots;
    for (const bool populated : {false, true}) {
        const auto output = roots.root(populated ? "populated" : "empty");
        citlali::pipeline::StagedOutputDirectory attempt(output);
        std::ofstream(attempt.staging() / "owned") << "ours";
        std::filesystem::create_directory(output);
        if (populated) std::ofstream(output / "foreign") << "theirs";
        EXPECT_THROW(attempt.publish(), std::filesystem::filesystem_error);
        EXPECT_TRUE(std::filesystem::is_directory(output));
        EXPECT_FALSE(std::filesystem::exists(output / "owned"));
        EXPECT_TRUE(std::filesystem::is_regular_file(attempt.staging() / "owned"));
        if (populated) EXPECT_TRUE(std::filesystem::is_regular_file(output / "foreign"));
    }
}

TEST(staged_output_directory, rejects_existing_output_and_dangling_alias) {
    TemporaryOutputRoots roots;
    const auto output = roots.root("existing");
    std::filesystem::create_directories(output);
    EXPECT_THROW(citlali::pipeline::StagedOutputDirectory attempt(output), std::exception);
    const auto alias = roots.root("alias");
    std::filesystem::create_directory_symlink(roots.root("absent"), alias);
    EXPECT_THROW(citlali::pipeline::StagedOutputDirectory attempt(alias), std::exception);
    EXPECT_TRUE(std::filesystem::is_symlink(alias));
    EXPECT_FALSE(std::filesystem::exists(alias.string() + ".pending"));
}
