#include "../src/citlali/cli/native_rtc_output_layout.h"
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

TEST(native_rtc_output_layout, prepares_all_network_and_terminal_paths_at_setup) {
    TemporaryOutputRoots roots;
    using namespace citlali::cli;
    NativeRtcOutputLayout layout(roots.root("observation") / "", {0, 7, 12}, true, false,
                                 DevelopmentTerminal::ptc);
    for (std::size_t i = 0; i < 3; ++i) {
        const auto &network = layout.network(i);
        ASSERT_EQ(network.treatments.size(), 1);
        const auto &arm = network.treatment(true);
        EXPECT_EQ(arm.root, network.root / "donor-continuity");
        EXPECT_TRUE(std::filesystem::is_directory(arm.cal));
        EXPECT_TRUE(std::filesystem::is_directory(arm.ptc));
        EXPECT_FALSE(std::filesystem::exists(network.root / "performance.jsonl"));
    }
}

TEST(native_rtc_output_layout, respects_terminal_and_comparison_layout) {
    TemporaryOutputRoots roots;
    using namespace citlali::cli;
    for (auto terminal : {DevelopmentTerminal::rtc_only, DevelopmentTerminal::cal,
                          DevelopmentTerminal::prepare}) {
        NativeRtcOutputLayout layout(roots.root(std::to_string(int(terminal))), {}, true,
                                     false, terminal);
        const auto &arm = layout.network(0).treatment(true);
        EXPECT_EQ(std::filesystem::exists(arm.root), terminal != DevelopmentTerminal::prepare);
        EXPECT_EQ(std::filesystem::exists(arm.cal), terminal == DevelopmentTerminal::cal);
        EXPECT_FALSE(std::filesystem::exists(arm.ptc));
    }
    NativeRtcOutputLayout comparison(roots.root("comparison"), {}, true, true,
                                     DevelopmentTerminal::rtc_only);
    EXPECT_TRUE(std::filesystem::is_directory(comparison.network(0).treatment(false).root));
    EXPECT_TRUE(std::filesystem::is_directory(comparison.network(0).treatment(true).root));
    NativeRtcOutputLayout manual(roots.root("manual"), {}, false, true,
                                 DevelopmentTerminal::rtc_only);
    EXPECT_EQ(manual.network(0).treatment(true).root, manual.root());
}

TEST(native_rtc_output_layout, rejects_existing_entries_and_preserves_failed_attempt) {
    TemporaryOutputRoots roots;
    using namespace citlali::cli;
    const auto path = roots.root("failed");
    {
        NativeRtcOutputLayout layout(path, {}, true, false, DevelopmentTerminal::ptc);
        std::ofstream(path / "performance.jsonl") << "original";
    }
    EXPECT_THROW((NativeRtcOutputLayout(path, {}, true, false, DevelopmentTerminal::ptc)),
                 std::exception);
    std::string original;
    std::ifstream(path / "performance.jsonl") >> original;
    EXPECT_EQ(original, "original");
    const auto file = roots.root("file");
    std::ofstream(file) << "original";
    const auto alias = roots.root("alias");
    std::filesystem::create_directory_symlink(roots.root("missing"), alias);
    for (const auto &entry : {file, alias})
        EXPECT_THROW((NativeRtcOutputLayout(entry, {}, true, false, DevelopmentTerminal::rtc_only)),
                     std::exception);
    EXPECT_TRUE(std::filesystem::is_symlink(alias));
}

TEST(native_rtc_output_layout, simultaneous_contenders_cannot_share_output) {
    TemporaryOutputRoots roots;
    const auto path = roots.root("contested");
    std::promise<void> start;
    const auto ready = start.get_future().share();
    std::atomic<int> won{0}, rejected{0};
    std::array<std::thread, 2> contenders;
    for (int i = 0; i < 2; ++i) contenders[i] = std::thread([&, i] {
        ready.wait();
        try {
            citlali::cli::NativeRtcOutputLayout layout(path, {0, 7}, true, false,
                                                      citlali::cli::DevelopmentTerminal::ptc);
            std::ofstream(layout.root() / "owner") << i;
            ++won;
        } catch (const std::exception &) { ++rejected; }
    });
    start.set_value();
    for (auto &thread : contenders) thread.join();
    EXPECT_EQ(won.load(), 1);
    EXPECT_EQ(rejected.load(), 1);
    EXPECT_TRUE(std::filesystem::is_regular_file(path / "owner"));
}

TEST(native_rtc_output_layout, symlink_parent_dotdot_claims_physical_destination) {
    for (const bool trailing_separator : {false, true}) {
        SCOPED_TRACE(trailing_separator);
        TemporaryOutputRoots roots;
        const auto work = roots.root("work");
        const auto storage = roots.root("storage");
        std::filesystem::create_directories(work);
        std::filesystem::create_directories(storage / "child");
        std::filesystem::create_directory_symlink(storage / "child", work / "link");
        auto requested = work / "link" / ".." / "run";
        if (trailing_separator) requested /= "";
        citlali::cli::NativeRtcOutputLayout layout(requested, {0, 7}, true, false,
                                                  citlali::cli::DevelopmentTerminal::ptc);
        ASSERT_TRUE(std::filesystem::is_directory(storage / "run"));
        EXPECT_TRUE(std::filesystem::equivalent(layout.root(), storage / "run"));
        for (int network : {0, 7}) {
            const auto arm = storage / "run" / ("network" + std::to_string(network)) /
                             "donor-continuity";
            EXPECT_TRUE(std::filesystem::is_directory(arm / "cal"));
            EXPECT_TRUE(std::filesystem::is_directory(arm / "ptc"));
        }
        EXPECT_FALSE(std::filesystem::exists(work / "run"));
    }
}

TEST(native_rtc_output_layout, symlink_parent_dotdot_preserves_existing_destination) {
    for (const bool trailing_separator : {false, true}) {
        SCOPED_TRACE(trailing_separator);
        TemporaryOutputRoots roots;
        const auto work = roots.root("work");
        const auto storage = roots.root("storage");
        std::filesystem::create_directories(work);
        std::filesystem::create_directories(storage / "child");
        std::filesystem::create_directories(storage / "run");
        std::filesystem::create_directory_symlink(storage / "child", work / "link");
        const auto sentinel = storage / "run" / "performance.jsonl";
        std::ofstream(sentinel) << "original";
        auto requested = work / "link" / ".." / "run";
        if (trailing_separator) requested /= "";
        EXPECT_THROW((citlali::cli::NativeRtcOutputLayout(requested, {}, true, false,
                      citlali::cli::DevelopmentTerminal::ptc)), std::runtime_error);
        std::string original;
        std::ifstream(sentinel) >> original;
        EXPECT_EQ(original, "original");
        EXPECT_FALSE(std::filesystem::exists(storage / "run" / "donor-continuity"));
        EXPECT_FALSE(std::filesystem::exists(work / "run"));
    }
}

TEST(native_rtc_output_layout, invalid_inventory_fails_before_claim) {
    TemporaryOutputRoots roots;
    const auto path = roots.root("invalid");
    EXPECT_THROW((citlali::cli::NativeRtcOutputLayout(path, {7, 7}, true, false,
                  citlali::cli::DevelopmentTerminal::ptc)), std::invalid_argument);
    EXPECT_FALSE(std::filesystem::exists(path));
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
