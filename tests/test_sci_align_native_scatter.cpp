#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace {

std::string source_text(const std::filesystem::path &relative) {
    const auto root = std::filesystem::path{__FILE__}.parent_path().parent_path();
    std::ifstream input{root / relative};
    if (!input) {
        throw std::runtime_error("unable to open source-architecture guard input");
    }
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

void expect_absent(const std::string &text, const std::string &token) {
    EXPECT_EQ(text.find(token), std::string::npos) << token;
}

TEST(sci_align_native_scatter,
     measured_stage_has_no_per_sample_update_or_revision_store) {
    const auto text = source_text(
        "include/citlali/core/pipeline/timestream_measured_scan.h");
    expect_absent(text, "class Update");
    expect_absent(text, "std::map<std::size_t, double>");
    expect_absent(text, "std::vector<TimestreamNativeRevision>");
    EXPECT_NE(text.find("commit_operation"), std::string::npos);
}

TEST(sci_align_native_scatter,
     ptc_stage_has_no_detector_sample_cell_narrative) {
    const auto text = source_text(
        "include/citlali/core/pipeline/timestream_ptc_cohort_adapter.h");
    expect_absent(text, "NativePtcCellBinding");
    expect_absent(text, "detector_samples");
    expect_absent(text, "std::vector<NativeMeasuredDetectorLedger::Update>");
    EXPECT_NE(text.find("NativePtcRowBinding"), std::string::npos);
}

TEST(sci_align_native_scatter,
     science_projection_has_no_duplicate_projection_cell_inventory) {
    const auto text = source_text(
        "include/citlali/core/pipeline/timestream_native_science_projection.h");
    expect_absent(text, "NativeScienceProjectionCell");
    expect_absent(text, "cells_");
    EXPECT_NE(text.find("Eigen::MatrixXd values_"), std::string::npos);
}

TEST(sci_align_native_scatter,
     superseded_generic_sample_ledger_and_cohort_are_unavailable) {
    const auto samples = source_text(
        "include/citlali/core/pipeline/timestream_native_sample.h");
    const auto cohort = source_text(
        "include/citlali/core/pipeline/timestream_coincidence_cohort.h");
    expect_absent(samples, "NativeSampleLedger");
    expect_absent(cohort, "CoincidenceCohortBuilder");
    expect_absent(cohort, "PcaRectangularWorkingSet");
}

}  // namespace
