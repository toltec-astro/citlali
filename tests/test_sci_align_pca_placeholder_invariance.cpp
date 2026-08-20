#include <citlali/core/pipeline/timestream_coincidence_cohort.h>

#include <citlali/core/config/mapmaking_config.h>
#include <citlali/core/timestream/ptc/clean.h>

#include <gtest/gtest.h>
#include <spdlog/sinks/null_sink.h>

#include <Eigen/Core>

#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

using citlali::pipeline::PcaCompatibilityHazard;
using citlali::pipeline::PcaCompatibilityInputs;
using citlali::pipeline::CoincidenceAbsenceReason;
using citlali::pipeline::CoincidenceCohortBuilder;
using citlali::pipeline::FinitePcaPlaceholder;
using citlali::pipeline::NativeOperationIdentity;
using citlali::pipeline::NativeSampleIdentity;
using citlali::pipeline::NativeSampleLedger;
using citlali::pipeline::classify_pca_compatibility;
using citlali::pipeline::make_pca_rectangular_working_set;
using citlali::pipeline::require_pca_compatibility;

std::shared_ptr<spdlog::logger> ensure_sci_align_logger() {
    auto logger = spdlog::get("citlali_logger");
    if (logger == nullptr) {
        auto sink = std::make_shared<spdlog::sinks::null_sink_mt>();
        logger =
            std::make_shared<spdlog::logger>("citlali_logger", sink);
        spdlog::register_logger(logger);
    }
    return logger;
}

struct OrdinaryPcaResult {
    Eigen::VectorXd eigenvalues;
    Eigen::MatrixXd eigenvectors;
    Eigen::MatrixXd cleaned;
};

OrdinaryPcaResult run_existing_ordinary_pca(
    const Eigen::MatrixXd &values,
    const Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> &flags,
    const std::string &grouping) {
    timestream::Cleaner cleaner;
    cleaner.logger = ensure_sci_align_logger();
    cleaner.stddev_limit = 0.0;
    cleaner.n_calc = 0;
    cleaner.standard_pca.enabled = true;
    cleaner.null_model.enabled = false;
    cleaner.marchenko_pastur.enabled = false;
    cleaner.adaptive_selector.enabled = false;

    Eigen::VectorXi apt_flags = Eigen::VectorXi::Zero(values.cols());
    constexpr Eigen::Index cut = 1;
    auto [eigenvalues, eigenvectors] =
        cleaner.calc_eig_values<timestream::Cleaner::SpectraBackend>(
            values, flags, apt_flags, cut);
    Eigen::MatrixXd cleaned = values;
    cleaner.remove_eig_values<timestream::Cleaner::SpectraBackend>(
        values, flags, eigenvalues, eigenvectors, cleaned, cut, -1,
        grouping, -1, 0);
    return {std::move(eigenvalues), std::move(eigenvectors),
            std::move(cleaned)};
}

struct PlaceholderFixture {
    Eigen::MatrixXd low_placeholder;
    Eigen::MatrixXd high_placeholder;
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> flags;
};

PlaceholderFixture make_placeholder_fixture() {
    constexpr Eigen::Index rows = 220;
    constexpr Eigen::Index columns = 6;
    PlaceholderFixture fixture{
        Eigen::MatrixXd{rows, columns},
        Eigen::MatrixXd{rows, columns},
        Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>{rows, columns}};
    fixture.flags.setConstant(false);
    for (Eigen::Index row = 0; row < rows; ++row) {
        for (Eigen::Index column = 0; column < columns; ++column) {
            const double value =
                std::sin(0.05 * static_cast<double>(row)) +
                0.02 * static_cast<double>(column + 1) *
                    std::cos(0.031 * static_cast<double>(row)) +
                0.001 * static_cast<double>(column * row) /
                    static_cast<double>(rows);
            fixture.low_placeholder(row, column) = value;
            fixture.high_placeholder(row, column) = value;
        }
    }
    for (Eigen::Index column = 0; column < columns; ++column) {
        const Eigen::Index flagged_row = 3 + column * 17;
        fixture.flags(flagged_row, column) = true;
        fixture.low_placeholder(flagged_row, column) = 17.0 + column;
        fixture.high_placeholder(flagged_row, column) = 911000.0 + column;
    }
    return fixture;
}

TEST(sci_align_pca_placeholder,
     ordinary_pca_all_group_is_finite_placeholder_invariant) {
    const auto fixture = make_placeholder_fixture();
    ASSERT_TRUE(timestream::Cleaner::is_supported_clean_group("all"));
    const auto low = run_existing_ordinary_pca(
        fixture.low_placeholder, fixture.flags, "all");
    const auto high = run_existing_ordinary_pca(
        fixture.high_placeholder, fixture.flags, "all");

    EXPECT_TRUE(low.eigenvalues.isApprox(high.eigenvalues, 1.0e-12));
    const auto low_projector =
        low.eigenvectors.leftCols(1) *
        low.eigenvectors.leftCols(1).transpose();
    const auto high_projector =
        high.eigenvectors.leftCols(1) *
        high.eigenvectors.leftCols(1).transpose();
    EXPECT_TRUE(low_projector.isApprox(high_projector, 1.0e-12));

    for (Eigen::Index row = 0; row < fixture.flags.rows(); ++row) {
        for (Eigen::Index column = 0;
             column < fixture.flags.cols(); ++column) {
            if (fixture.flags(row, column)) {
                EXPECT_DOUBLE_EQ(low.cleaned(row, column),
                                 fixture.low_placeholder(row, column));
                EXPECT_DOUBLE_EQ(high.cleaned(row, column),
                                 fixture.high_placeholder(row, column));
            }
            else {
                EXPECT_NEAR(low.cleaned(row, column),
                            high.cleaned(row, column), 1.0e-12);
            }
        }
    }
}

TEST(sci_align_pca_placeholder,
     corr_nw_group_construction_is_placeholder_invariant) {
    const auto fixture = make_placeholder_fixture();
    timestream::Cleaner cleaner;
    cleaner.logger = ensure_sci_align_logger();
    cleaner.corr_grouping.enabled = true;
    cleaner.corr_grouping.metric = "abs";
    cleaner.corr_grouping.corr_min = 0.6;
    cleaner.corr_grouping.min_overlap = 200;
    cleaner.corr_grouping.min_good_frac = 0.7;
    cleaner.corr_grouping.min_group_size = 6;
    cleaner.corr_grouping.max_samples = 20000;
    cleaner.corr_grouping.clean_residual = true;
    const Eigen::VectorXi apt_flags =
        Eigen::VectorXi::Zero(fixture.flags.cols());

    const auto low = cleaner.get_corr_groups(
        fixture.low_placeholder, fixture.flags, apt_flags);
    const auto high = cleaner.get_corr_groups(
        fixture.high_placeholder, fixture.flags, apt_flags);

    ASSERT_EQ(low.groups, high.groups);
    ASSERT_EQ(low.groups.size(), 1U);
    EXPECT_EQ(low.groups.front(),
              (std::vector<Eigen::Index>{0, 1, 2, 3, 4, 5}));

    const auto low_pca = run_existing_ordinary_pca(
        fixture.low_placeholder, fixture.flags, "corr_nw");
    const auto high_pca = run_existing_ordinary_pca(
        fixture.high_placeholder, fixture.flags, "corr_nw");
    EXPECT_TRUE(
        low_pca.eigenvalues.isApprox(high_pca.eigenvalues, 1.0e-12));
    const auto low_projector =
        low_pca.eigenvectors.leftCols(1) *
        low_pca.eigenvectors.leftCols(1).transpose();
    const auto high_projector =
        high_pca.eigenvectors.leftCols(1) *
        high_pca.eigenvectors.leftCols(1).transpose();
    EXPECT_TRUE(low_projector.isApprox(high_projector, 1.0e-12));
}

TEST(sci_align_pca_placeholder,
     optional_modes_with_exclusions_are_classified_fail_closed) {
    const NativeSampleIdentity valid_identity{0, 0, 30.0};
    NativeSampleLedger<double> valid_ledger{{{valid_identity, 1.0}}};
    CoincidenceCohortBuilder valid_builder{
        NativeOperationIdentity{20, 0}, {0}, 1};
    valid_builder.assign_mapped_valid(0, 0, valid_identity, 0);
    auto valid_cohort = std::move(valid_builder).finish();
    const auto valid_working = make_pca_rectangular_working_set(
        valid_ledger, valid_cohort,
        FinitePcaPlaceholder::checked(123.0));

    NativeSampleLedger<double> excluded_ledger{
        std::vector<NativeSampleLedger<double>::Seed>{}};
    CoincidenceCohortBuilder excluded_builder{
        NativeOperationIdentity{21, 0}, {0}, 1};
    excluded_builder.assign_absent(
        0, 0, CoincidenceAbsenceReason::no_candidate);
    auto excluded_cohort = std::move(excluded_builder).finish();
    const auto excluded_working = make_pca_rectangular_working_set(
        excluded_ledger, excluded_cohort,
        FinitePcaPlaceholder::checked(123.0));

    PcaCompatibilityInputs ordinary;
    EXPECT_TRUE(classify_pca_compatibility(
                    excluded_working, ordinary)
                    .compatible());

    PcaCompatibilityInputs unbanded_mp = ordinary;
    unbanded_mp.marchenko_pastur_active_for_operation = true;
    EXPECT_TRUE(classify_pca_compatibility(
                    excluded_working, unbanded_mp)
                    .compatible());

    PcaCompatibilityInputs null_only = ordinary;
    null_only.null_model_active_for_operation = true;
    const auto null_classification =
        classify_pca_compatibility(excluded_working, null_only);
    EXPECT_FALSE(null_classification.compatible());
    EXPECT_TRUE(null_classification.has(PcaCompatibilityHazard::null_model));
    EXPECT_THROW(
        require_pca_compatibility(null_classification), std::logic_error);

    PcaCompatibilityInputs adaptive_only = ordinary;
    adaptive_only.adaptive_selector_active_for_operation = true;
    const auto adaptive_classification =
        classify_pca_compatibility(excluded_working, adaptive_only);
    EXPECT_FALSE(adaptive_classification.compatible());
    EXPECT_TRUE(adaptive_classification.has(
        PcaCompatibilityHazard::adaptive_selector));
    EXPECT_THROW(
        require_pca_compatibility(adaptive_classification),
        std::logic_error);

    PcaCompatibilityInputs banded_mp_only = unbanded_mp;
    banded_mp_only.marchenko_pastur_band_requested = true;
    const auto banded_mp_classification =
        classify_pca_compatibility(excluded_working, banded_mp_only);
    EXPECT_FALSE(banded_mp_classification.compatible());
    EXPECT_TRUE(banded_mp_classification.has(
        PcaCompatibilityHazard::band_limited_marchenko_pastur));
    EXPECT_THROW(
        require_pca_compatibility(banded_mp_classification),
        std::logic_error);

    PcaCompatibilityInputs incompatible = ordinary;
    incompatible.null_model_active_for_operation = true;
    incompatible.adaptive_selector_active_for_operation = true;
    incompatible.marchenko_pastur_active_for_operation = true;
    incompatible.marchenko_pastur_band_requested = true;
    const auto classification =
        classify_pca_compatibility(excluded_working, incompatible);
    EXPECT_FALSE(classification.compatible());
    EXPECT_TRUE(classification.has(PcaCompatibilityHazard::null_model));
    EXPECT_TRUE(
        classification.has(PcaCompatibilityHazard::adaptive_selector));
    EXPECT_TRUE(classification.has(
        PcaCompatibilityHazard::band_limited_marchenko_pastur));
    EXPECT_THROW(
        require_pca_compatibility(classification), std::logic_error);

    EXPECT_NO_THROW(require_pca_compatibility(
        classify_pca_compatibility(valid_working, incompatible)));
}

}  // namespace
