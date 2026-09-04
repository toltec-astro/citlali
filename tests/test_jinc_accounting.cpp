#include <citlali/core/mapmaking/jinc_accounting.h>
#include <citlali/core/mapmaking/map.h>
#include <citlali/core/pipeline/jinc_accounting_output.h>

#include <gtest/gtest.h>
#include <netcdf>

#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace {

mapmaking::JincAccountingState make_state(Eigen::Index rows = 2,
                                          Eigen::Index cols = 2) {
    mapmaking::JincAccountingState state;
    state.configure("a1400", 1, 4460, 5, 1, "123424", 5, 2, 3.0,
                    {1.1, 2.17, 2.0}, rows, cols);
    return state;
}

TEST(jinc_accounting, disabled_state_allocates_nothing) {
    mapmaking::JincAccountingState state;

    EXPECT_FALSE(state.enabled());
    EXPECT_EQ(state.target_n.size(), 0);
    EXPECT_TRUE(state.target_samples.empty());
}

TEST(jinc_accounting, records_exact_signed_total_and_target_construction) {
    auto state = make_state();
    std::map<std::string, Eigen::VectorXd> apt;
    apt["array"].resize(3);
    apt["array"] << 1.0, 1.0, 0.0;
    apt["uid"].resize(3);
    apt["uid"] << 4460.0, 4461.0, 1000.0;
    state.prepare_uid_inventory(apt);

    state.record_contribution(0, 0, 6.0, 3.0, 4.5, 4460, true);
    state.record_contribution(0, 0, -8.0, -2.0, 2.0, 4461, false);
    state.record_contribution(0, 0, 2.0, 1.0, 0.5, 4460, true);

    EXPECT_DOUBLE_EQ(state.target_n(0, 0), 8.0);
    EXPECT_DOUBLE_EQ(state.target_c(0, 0), 4.0);
    EXPECT_DOUBLE_EQ(state.target_q(0, 0), 5.0);
    EXPECT_DOUBLE_EQ(state.total_abs_n(0, 0), 16.0);
    EXPECT_DOUBLE_EQ(state.total_abs_c(0, 0), 6.0);
    EXPECT_DOUBLE_EQ(state.target_abs_n(0, 0), 8.0);
    EXPECT_DOUBLE_EQ(state.target_abs_c(0, 0), 4.0);
    EXPECT_EQ(state.total_occurrence_count(0, 0), 3);
    EXPECT_EQ(state.target_occurrence_count(0, 0), 2);
    EXPECT_EQ(state.total_unique_detector_count(0, 0), 2);
    EXPECT_EQ(state.target_unique_detector_count(0, 0), 1);
}

TEST(jinc_accounting, snapshots_actual_accumulators_and_sample_reasons) {
    auto state = make_state();
    Eigen::MatrixXd n = Eigen::MatrixXd::Constant(2, 2, 7.0);
    Eigen::MatrixXd c = Eigen::MatrixXd::Constant(2, 2, -3.0);
    Eigen::MatrixXd q = Eigen::MatrixXd::Constant(2, 2, 11.0);
    state.capture_totals(n, c, q);
    n.setZero();
    c.setZero();
    q.setZero();

    mapmaking::JincAccountingSample sample;
    sample.scan_index = 5;
    sample.sample_index = 17;
    sample.reason = "final_flagged";
    state.record_sample(sample);

    EXPECT_DOUBLE_EQ(state.total_n(1, 1), 7.0);
    EXPECT_DOUBLE_EQ(state.total_c(1, 1), -3.0);
    EXPECT_DOUBLE_EQ(state.total_q(1, 1), 11.0);
    ASSERT_EQ(state.target_samples.size(), 1U);
    EXPECT_EQ(state.target_samples.front().reason, "final_flagged");
    EXPECT_TRUE(state.is_target(4460, 5, 1, 1));
    EXPECT_FALSE(state.is_target(4461, 5, 1, 1));
    EXPECT_FALSE(state.is_target(4460, 4, 1, 1));
    EXPECT_EQ(mapmaking::jinc_accounting_admission_reason(
                  true, true, true, true),
              "final_flagged");
    EXPECT_EQ(mapmaking::jinc_accounting_admission_reason(
                  false, false, true, true),
              "nonfinite_signal");
    EXPECT_EQ(mapmaking::jinc_accounting_admission_reason(
                  false, true, false, true),
              "analysis_coefficient_unavailable");
    EXPECT_EQ(mapmaking::jinc_accounting_admission_reason(
                  false, true, true, false),
              "center_outside_map");
    EXPECT_EQ(mapmaking::jinc_accounting_admission_reason(
                  false, true, true, true),
              "admitted");
}

TEST(jinc_accounting, normalization_receipt_is_observational_only) {
    mapmaking::MapBuffer control{"control"};
    control.n_rows = 1;
    control.n_cols = 4;
    control.parallel_policy = "seq";
    control.cov_cut = 0.0;
    control.signal = {Eigen::MatrixXd(1, 4)};
    control.grid_weight = {Eigen::MatrixXd(1, 4)};
    control.weight = {Eigen::MatrixXd::Constant(1, 4, 4.0)};
    control.signal[0] << 2.0, 2.0, 2.0, 2.0;
    control.grid_weight[0] << 2.0, -2.0, 0.0, 1e-9;
    auto diagnostic = control;
    diagnostic.jinc_accounting.configure(
        "a1400", 1, 4460, 5, 0, "123424", 5, 1, 3.0,
        {1.1, 2.17, 2.0}, 1, 4);

    control.normalize_maps();
    diagnostic.normalize_maps();

    ASSERT_EQ(control.signal.size(), diagnostic.signal.size());
    ASSERT_EQ(control.weight.size(), diagnostic.weight.size());
    for (Eigen::Index i = 0; i < control.signal[0].size(); ++i) {
        EXPECT_DOUBLE_EQ(control.signal[0].data()[i],
                         diagnostic.signal[0].data()[i]);
        EXPECT_DOUBLE_EQ(control.weight[0].data()[i],
                         diagnostic.weight[0].data()[i]);
    }
    EXPECT_DOUBLE_EQ(diagnostic.signal[0](0, 0), 1.0);
    EXPECT_DOUBLE_EQ(diagnostic.signal[0](0, 1), -1.0);
    EXPECT_DOUBLE_EQ(diagnostic.signal[0](0, 2), 0.0);
    EXPECT_DOUBLE_EQ(diagnostic.signal[0](0, 3), 0.0);
    EXPECT_TRUE(diagnostic.jinc_accounting.totals_captured);
    EXPECT_TRUE(diagnostic.jinc_accounting.normalization_captured);
}

TEST(jinc_accounting, required_sample_output_failure_propagates) {
    const auto missing_parent =
        std::filesystem::path(testing::TempDir()) /
        "jinc-accounting-missing-parent" / "samples.ecsv";
    std::filesystem::remove_all(missing_parent.parent_path());

    EXPECT_THROW(
        citlali::pipeline::write_jinc_accounting_sample_table_atomic(
            missing_parent.string(), {}),
        citlali::error::Error);
}

TEST(jinc_accounting, writes_complete_required_receipt) {
    const auto output_dir =
        std::filesystem::path(testing::TempDir()) / "jinc-accounting-receipt";
    std::filesystem::remove_all(output_dir);
    std::filesystem::create_directories(output_dir);
    mapmaking::MapBuffer buffer{"omb"};
    buffer.n_rows = 2;
    buffer.n_cols = 2;
    buffer.pixel_size_rad = 1e-5;
    buffer.cov_cut = 0.0;
    buffer.sig_unit = "mJy/beam";
    buffer.wcs.cdelt = {-1e-5F, 1e-5F};
    buffer.wcs.crpix = {1.5F, 1.5F};
    buffer.wcs.crval = {0.0F, 0.0F};
    buffer.rows_tan_vec = Eigen::VectorXd::LinSpaced(2, -0.5, 0.5);
    buffer.cols_tan_vec = Eigen::VectorXd::LinSpaced(2, -0.5, 0.5);
    buffer.weight = {Eigen::MatrixXd::Ones(2, 2),
                     Eigen::MatrixXd::Ones(2, 2)};
    buffer.weight_formal = buffer.weight;
    buffer.noise_weight_scale = Eigen::VectorXd::Ones(2);
    buffer.jinc_accounting = make_state();
    std::map<std::string, Eigen::VectorXd> apt;
    apt["array"] = Eigen::VectorXd::Constant(1, 1.0);
    apt["uid"] = Eigen::VectorXd::Constant(1, 4460.0);
    buffer.jinc_accounting.prepare_uid_inventory(apt);
    buffer.jinc_accounting.capture_totals(
        Eigen::MatrixXd::Ones(2, 2), Eigen::MatrixXd::Ones(2, 2),
        Eigen::MatrixXd::Ones(2, 2));
    buffer.jinc_accounting.capture_normalization(
        buffer.weight_formal[0], Eigen::ArrayXXd::Ones(2, 2), 0.0);
    mapmaking::JincAccountingSample sample;
    sample.reason = "admitted";
    sample.admitted = 1;
    buffer.jinc_accounting.record_sample(sample);

    const auto mapdiag_base = output_dir / "observation_mapdiag";
    citlali::pipeline::write_jinc_accounting_receipt(
        buffer, mapdiag_base.string());

    const auto netcdf_path = output_dir / "observation_jinc_accounting.nc";
    const auto sample_path =
        output_dir / "observation_jinc_accounting_target_samples.ecsv";
    ASSERT_TRUE(std::filesystem::exists(netcdf_path));
    ASSERT_TRUE(std::filesystem::exists(sample_path));
    netCDF::NcFile file(netcdf_path.string(), netCDF::NcFile::read);
    EXPECT_FALSE(file.getVar("total_N").isNull());
    EXPECT_FALSE(file.getVar("target_C").isNull());
    EXPECT_FALSE(file.getVar("normalization_support").isNull());
    EXPECT_FALSE(file.getVar("science_policy_support").isNull());
    EXPECT_FALSE(file.getVar("schema_identity").isNull());
}

}  // namespace
