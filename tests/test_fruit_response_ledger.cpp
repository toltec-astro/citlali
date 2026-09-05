#include <citlali/core/fruit/response_ledger.h>
#include <gtest/gtest.h>
#include <netcdf>
#include <chrono>
#include <limits>

namespace {
using namespace citlali::fruit;

class FruitResponseLedgerTest : public ::testing::Test {
protected:
    std::filesystem::path directory;
    void SetUp() override {
        directory = std::filesystem::temp_directory_path() /
            ("citlali-fruit-ledger-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        ASSERT_TRUE(std::filesystem::create_directory(directory));
    }
    void TearDown() override { std::filesystem::remove_all(directory); }
    auto ledger() {
        ResponseOccurrenceLedger::KernelBank kernels{{0, {Eigen::MatrixXd::Ones(1, 1)}}};
        return std::make_unique<ResponseOccurrenceLedger>(directory / "occurrences.bin", "synthetic", 0, 1, 1,
                                                         std::vector<int>{0}, kernels, kernels);
    }
    static auto plane(double value) { return std::vector<Eigen::MatrixXd>{Eigen::MatrixXd::Constant(1, 1, value)}; }
    static ResponseCandidate candidate(int uid) {
        ResponseCandidate value;
        value.key = {"synthetic", 0, uid, 0};
        return value;
    }
    static auto state() {
        ResponseInterventionState value;
        value.configure("Half");
        value.begin(0, false);
        return value;
    }
};

TEST_F(FruitResponseLedgerTest, JointDeletionSubtractsContributionsBeforeNormalization) {
    auto value = ledger();
    value->begin_scan(0);
    for (int uid : {100, 101, 102}) value->register_detector(uid, 0);
    value->append(0, 100, 0, 0, 0, 0, 0, 20, 1, 1, 20, 1);
    value->append(0, 101, 0, 0, 0, 0, 0, 10, 1, 1, 10, 1);
    value->append(0, 102, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1);
    value->capture_totals(plane(30), plane(3), plane(3));
    auto decision = state();
    value->finish(decision, {candidate(101), candidate(100)}, plane(10), plane(3), plane(3), 0.5);
    EXPECT_DOUBLE_EQ(value->deletions().at(candidate(100).key).difference(0), -5);
    EXPECT_DOUBLE_EQ(value->deletions().at(candidate(101).key).difference(0), 0);
    EXPECT_DOUBLE_EQ(value->joint_deletions().at(0).difference(0), -10);
    EXPECT_EQ(value->totals().at(0).unique(0), 3);
    EXPECT_EQ(value->joint_deletions().at(0).target.unique(0), 2);
    EXPECT_EQ(decision.assignments().size(), 2U);
    value->write_receipt(directory / "receipt.nc", decision, 1e-5, "mJy/beam");
    netCDF::NcFile receipt((directory / "receipt.nc").string(), netCDF::NcFile::read);
    std::string stored;
    receipt.getAtt("state").getValues(stored);
    EXPECT_EQ(stored, decision.serialize());
    EXPECT_EQ(value->occurrence_count(), 3U);
    EXPECT_EQ(value->spool_bytes(), 4 * ResponseOccurrenceLedger::record_bytes);
}

TEST_F(FruitResponseLedgerTest, LostSupportSelectsEvenWithUnavailableResponse) {
    auto value = ledger();
    value->begin_scan(0);
    value->register_detector(100, 0);
    value->append(0, 100, 0, 0, 0, 0, 0, 3, 1, 1, 3, 1);
    value->capture_totals(plane(3), plane(1), plane(1));
    auto decision = state();
    value->finish(decision, {candidate(100)}, plane(3), plane(1), plane(1), 0.5);
    const auto &score = decision.census().front().response;
    EXPECT_FALSE(score.available);
    EXPECT_TRUE(score.support_risk);
    EXPECT_EQ(score.lost, 1);
    EXPECT_TRUE(decision.census().front().selected);
}

TEST_F(FruitResponseLedgerTest, ZeroContributionKeyIsAValidNoOpportunity) {
    auto value = ledger();
    value->begin_scan(0);
    value->register_detector(101, 0);
    value->append(0, 101, 0, 0, 0, 0, 0, 3, 1, 1, 3, 1);
    value->capture_totals(plane(3), plane(1), plane(1));
    auto decision = state();
    value->finish(decision, {candidate(100)}, plane(3), plane(1), plane(1), 0.5);
    EXPECT_EQ(decision.census().front().response.ratio, 0);
    EXPECT_TRUE(decision.census().front().response.available);
    EXPECT_FALSE(decision.census().front().selected);
}

TEST_F(FruitResponseLedgerTest, MissingNonfiniteAndDuplicateOccurrenceStateFailClosed) {
    auto value = ledger();
    value->begin_scan(0);
    EXPECT_THROW(value->begin_scan(0), std::runtime_error); // duplicate science/noise pass
    value->register_detector(100, 0);
    EXPECT_THROW(value->append(0, 100, 0, 0, 0, 0, 0,
                 std::numeric_limits<double>::quiet_NaN(), 1, 1, 3, 1), std::runtime_error);
    value->append(0, 100, 0, 0, 0, 0, 0, 3, 1, 1, 3, 1);
    value->append(0, 100, 0, 0, 0, 0, 0, 3, 1, 1, 3, 1);
    auto decision = state();
    EXPECT_THROW(value->finish(decision, {}, plane(3), plane(2), plane(2), 0.5), std::runtime_error);
    value->capture_totals(plane(6), plane(2), plane(2));
    EXPECT_THROW(value->finish(decision, {}, plane(3), plane(2), plane(2), 0.5), std::runtime_error);
    EXPECT_FALSE(decision.completed());
}

TEST_F(FruitResponseLedgerTest, BrokenAccountingCannotPublishASelection) {
    auto value = ledger();
    value->begin_scan(0);
    value->register_detector(100, 0);
    value->append(0, 100, 0, 0, 0, 0, 0, 3, 1, 1, 3, 1);
    value->capture_totals(plane(4), plane(1), plane(1));
    auto decision = state();
    EXPECT_THROW(value->finish(decision, {candidate(100)}, plane(4), plane(1), plane(1), 0.5), std::runtime_error);
    EXPECT_FALSE(decision.completed());
    EXPECT_THROW(value->write_receipt(directory / "receipt.nc", decision, 1e-5, "mJy/beam"), std::runtime_error);
    EXPECT_FALSE(std::filesystem::exists(directory / "receipt.nc"));
}
}  // namespace
