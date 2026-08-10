#include <gtest/gtest.h>

#include <citlali/core/provenance/deployment_identity.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>

namespace {

using citlali::provenance::DeploymentBinding;
using citlali::provenance::DeploymentIdentity;

class deployment_identity : public ::testing::Test {
  protected:
    void SetUp() override {
        root_ =
            std::filesystem::temp_directory_path() /
            ("citlali-deployment-" +
             std::to_string(
                 std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(root_);
    }

    void TearDown() override { std::filesystem::remove_all(root_); }

    void write_lock(const std::string &hash, int roots = 1) const {
        std::ofstream output(root_ / "spack.lock");
        output << "{\"roots\":[";
        for (int index = 0; index < roots; ++index) {
            if (index != 0) {
                output << ',';
            }
            output << "{\"hash\":\"" << hash << "\",\"spec\":\"citlali\"}";
        }
        output << "]}";
    }

    std::filesystem::path root_;
};

TEST_F(deployment_identity, accepts_unmanaged_process) {
    const auto identity = citlali::provenance::deployment_identity_from_values(
        std::nullopt, std::nullopt, std::nullopt);
    EXPECT_FALSE(identity.managed());
    EXPECT_EQ(
        citlali::provenance::require_deployment_matches_build(identity, ""),
        DeploymentBinding::unmanaged);
}

TEST_F(deployment_identity, rejects_partial_managed_identity) {
    EXPECT_THROW(
        citlali::provenance::deployment_identity_from_values(
            std::string{"release/unity"}, std::nullopt, root_.string()),
        std::runtime_error);
}

TEST_F(deployment_identity, rejects_invalid_lock_digest) {
    EXPECT_THROW(citlali::provenance::deployment_identity_from_values(
                     std::string{"release/unity"}, std::string{"not-a-digest"},
                     root_.string()),
                 std::runtime_error);
}

TEST_F(deployment_identity, accepts_matching_managed_dag) {
    const std::string hash(32, 'a');
    write_lock(hash);
    const DeploymentIdentity identity{"release/unity", std::string(64, 'b'),
                                      root_};
    EXPECT_EQ(
        citlali::provenance::require_deployment_matches_build(identity, hash),
        DeploymentBinding::dag_match);
}

TEST_F(deployment_identity, rejects_mismatched_managed_dag) {
    write_lock(std::string(32, 'a'));
    const DeploymentIdentity identity{"release/unity", std::string(64, 'b'),
                                      root_};
    EXPECT_THROW(citlali::provenance::require_deployment_matches_build(
                     identity, std::string(32, 'c')),
                 std::runtime_error);
}

TEST_F(deployment_identity, rejects_multiple_lock_roots) {
    const std::string hash(32, 'a');
    write_lock(hash, 2);
    const DeploymentIdentity identity{"release/unity", std::string(64, 'b'),
                                      root_};
    EXPECT_THROW(
        citlali::provenance::require_deployment_matches_build(identity, hash),
        std::runtime_error);
}

} // namespace
