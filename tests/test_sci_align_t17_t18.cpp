#include <citlali/core/config/reduction_config.h>
#include <citlali/core/pipeline/output_path_state.h>
#include <citlali/core/pipeline/sci_align_contract.h>
#include <citlali/core/pipeline/sci_align_field_registry.h>
#include <citlali/core/pipeline/timestream_alignment_helpers.h>
#include <citlali/core/pipeline/timestream_alignment_state.h>
#include <citlali/core/pipeline/timestream_output_provenance.h>
#include <citlali/core/pipeline/tod_output_state.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <future>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace align = citlali::pipeline::sci_align;

Eigen::VectorXd vector(std::initializer_list<double> values) {
    Eigen::VectorXd result(static_cast<Eigen::Index>(values.size()));
    Eigen::Index index = 0;
    for (const double value : values) {
        result[index++] = value;
    }
    return result;
}

struct OperatorCase {
    std::string identity;
    Eigen::VectorXd source_time;
    Eigen::VectorXd source_value;
    double target_time = 0.0;
    align::FieldContract contract;
};

// This contract-level operator supplies compact generative coverage for field
// policy. It is not the production telescope-alignment execution path; that
// path is exercised independently below through
// interpolate_telescope_data_to_common_time().
using ContractReferenceOperator = decltype(&align::align_field_at);
constexpr ContractReferenceOperator compiled_contract_reference_operator =
    &align::align_field_at;

align::AlignedValue evaluate_case(const OperatorCase &item,
                                  ContractReferenceOperator align_operator) {
    return align_operator(item.source_time, item.source_value,
                          item.target_time, item.contract,
                          align::DetailLevel::expanded);
}

void expect_identical(const align::AlignedValue &expected,
                      const align::AlignedValue &actual) {
    if (std::isnan(expected.value)) {
        EXPECT_TRUE(std::isnan(actual.value));
    } else {
        EXPECT_DOUBLE_EQ(actual.value, expected.value);
    }
    EXPECT_EQ(actual.quality.origin, expected.quality.origin);
    EXPECT_EQ(actual.quality.validity, expected.quality.validity);
    EXPECT_EQ(actual.quality.method, expected.quality.method);
    EXPECT_EQ(actual.quality.reason, expected.quality.reason);
    ASSERT_EQ(actual.quality.expanded_sources.size(),
              expected.quality.expanded_sources.size());
    for (std::size_t source = 0;
         source < expected.quality.expanded_sources.size(); ++source) {
        EXPECT_EQ(actual.quality.expanded_sources[source].source_row,
                  expected.quality.expanded_sources[source].source_row);
        EXPECT_DOUBLE_EQ(actual.quality.expanded_sources[source].weight,
                         expected.quality.expanded_sources[source].weight);
    }
}

TEST(sci_align_T17,
     contract_reference_operator_is_exact_across_async_reverse_order) {
    const align::FieldContract scalar{
        align::FieldTopology::continuous_scalar, 2.0, 0.0, std::nullopt};
    const align::FieldContract circular{
        align::FieldTopology::circular, 2.0, 360.0, std::nullopt};
    const align::FieldContract step{
        align::FieldTopology::declared_half_open_step, 2.0, 0.0, 3.0};
    const align::FieldContract exact_only{
        align::FieldTopology::exact_only, 0.0, 0.0, std::nullopt};
    const align::FieldContract short_support{
        align::FieldTopology::continuous_scalar, 0.5, 0.0, std::nullopt};

    const std::vector<OperatorCase> corpus{
        {"scalar_linear", vector({0.0, 1.0, 2.0}),
         vector({4.0, 6.0, 8.0}), 0.25, scalar},
        {"scalar_exact", vector({0.0, 1.0, 2.0}),
         vector({4.0, 6.0, 8.0}), 1.0, scalar},
        {"scalar_outside", vector({0.0, 1.0, 2.0}),
         vector({4.0, 6.0, 8.0}), -0.25, scalar},
        {"circular_shortest_arc", vector({0.0, 1.0, 2.0}),
         vector({350.0, 10.0, 30.0}), 0.5, circular},
        {"half_open_step", vector({0.0, 1.0, 2.0}),
         vector({7.0, 9.0, 11.0}), 1.5, step},
        {"exact_only_rejects_interpolation", vector({0.0, 1.0, 2.0}),
         vector({7.0, 9.0, 11.0}), 1.5, exact_only},
        {"nonfinite_source", vector({0.0, 1.0, 2.0}),
         vector({4.0, std::numeric_limits<double>::quiet_NaN(), 8.0}),
         0.5, scalar},
        {"support_span_exceeded", vector({0.0, 1.0, 2.0}),
         vector({4.0, 6.0, 8.0}), 0.5, short_support},
    };

    std::vector<align::AlignedValue> sequential;
    sequential.reserve(corpus.size());
    for (const auto &item : corpus) {
        sequential.push_back(
            evaluate_case(item, compiled_contract_reference_operator));
    }

    std::vector<std::future<std::pair<std::size_t, align::AlignedValue>>>
        futures;
    futures.reserve(corpus.size());
    for (std::size_t reverse = corpus.size(); reverse > 0; --reverse) {
        const std::size_t index = reverse - 1;
        futures.push_back(std::async(
            std::launch::async, [&corpus, index] {
                return std::pair<std::size_t, align::AlignedValue>{
                    index,
                    evaluate_case(corpus[index],
                                  compiled_contract_reference_operator)};
            }));
    }

    std::vector<align::AlignedValue> asynchronous(corpus.size());
    for (auto &future : futures) {
        auto [index, result] = future.get();
        asynchronous[index] = std::move(result);
    }

    ASSERT_EQ(asynchronous.size(), sequential.size());
    for (std::size_t index = 0; index < corpus.size(); ++index) {
        SCOPED_TRACE(corpus[index].identity);
        expect_identical(sequential[index], asynchronous[index]);
    }
}

using TelescopeMap = std::map<std::string, Eigen::VectorXd>;

struct ProductionTelescopeCase {
    std::string identity;
    TelescopeMap native_tel_data;
    Eigen::VectorXd common_time;
    bool skip_tel_utc_during_loop = false;
};

struct ProductionTelescopeResult {
    TelescopeMap aligned_tel_data;
    citlali::pipeline::AlignmentTelescopeSummary summary;
};

Eigen::VectorXd native_time_axis(Eigen::Index row_count, double first_sec,
                                 double cadence_sec) {
    Eigen::VectorXd result(row_count);
    for (Eigen::Index row = 0; row < row_count; ++row) {
        result(row) = first_sec + cadence_sec * static_cast<double>(row);
    }
    return result;
}

Eigen::VectorXd exact_and_midpoint_targets(const Eigen::VectorXd &native_time) {
    Eigen::VectorXd result(2 * native_time.size() - 1);
    for (Eigen::Index row = 0; row < native_time.size(); ++row) {
        result(2 * row) = native_time(row);
        if (row + 1 < native_time.size()) {
            result(2 * row + 1) =
                0.5 * (native_time(row) + native_time(row + 1));
        }
    }
    return result;
}

Eigen::VectorXd continuous_native_field(Eigen::Index row_count,
                                        double field_offset) {
    Eigen::VectorXd result(row_count);
    for (Eigen::Index row = 0; row < row_count; ++row) {
        const double index = static_cast<double>(row);
        result(row) = field_offset + 2.5e-6 * index +
                      7.0e-10 * index * index;
    }
    return result;
}

Eigen::VectorXd circular_native_field(Eigen::Index row_count,
                                      double field_offset) {
    constexpr double two_pi =
        6.283185307179586476925286766559005768;
    Eigen::VectorXd result(row_count);
    for (Eigen::Index row = 0; row < row_count; ++row) {
        const double unwrapped =
            6.18 + field_offset + 0.013 * static_cast<double>(row);
        double wrapped = std::fmod(unwrapped, two_pi);
        if (wrapped < 0.0) {
            wrapped += two_pi;
        }
        result(row) = wrapped;
    }
    return result;
}

TelescopeMap real_shaped_native_telescope_map(
    const Eigen::VectorXd &native_time, double field_offset,
    bool include_pps_time) {
    TelescopeMap result;
    std::size_t circular_index = 0;
    std::size_t continuous_index = 0;
    for (const auto &entry : align::active_field_registry) {
        const std::string name{entry.canonical_name};
        switch (entry.permitted_operator) {
            case align::FieldOperator::native_coordinate:
                result[name] = native_time;
                break;
            case align::FieldOperator::bracketed_linear:
                result[name] = continuous_native_field(
                    native_time.size(),
                    field_offset +
                        0.01 * static_cast<double>(continuous_index++));
                break;
            case align::FieldOperator::bracketed_shortest_arc:
                result[name] = circular_native_field(
                    native_time.size(),
                    field_offset +
                        0.02 * static_cast<double>(circular_index++));
                break;
            case align::FieldOperator::legacy_whole_word_linear_any_nonzero: {
                Eigen::VectorXd raw_words(native_time.size());
                for (Eigen::Index row = 0; row < native_time.size(); ++row) {
                    raw_words(row) = (row / 17) % 3 == 1 ? 8.0 : 0.0;
                }
                result[name] = std::move(raw_words);
                break;
            }
            case align::FieldOperator::exact_diagnostic:
                if (name == "TelUTC" || include_pps_time) {
                    result[name] = native_time;
                }
                break;
        }
    }
    return result;
}

ProductionTelescopeResult run_production_telescope_case(
    const ProductionTelescopeCase &item) {
    ProductionTelescopeResult result{item.native_tel_data, {}};
    citlali::pipeline::TimestreamAlignmentState alignment_state;
    citlali::pipeline::interpolate_telescope_data_to_common_time(
        result.aligned_tel_data, item.common_time,
        item.skip_tel_utc_during_loop, &alignment_state);
    result.summary = std::move(alignment_state.telescope);
    return result;
}

void expect_same_double_bits(double expected, double actual) {
    std::uint64_t expected_bits = 0;
    std::uint64_t actual_bits = 0;
    static_assert(sizeof(expected_bits) == sizeof(expected));
    std::memcpy(&expected_bits, &expected, sizeof(expected));
    std::memcpy(&actual_bits, &actual, sizeof(actual));
    EXPECT_EQ(actual_bits, expected_bits);
}

void expect_identical_telescope_map(const TelescopeMap &expected,
                                    const TelescopeMap &actual) {
    ASSERT_EQ(actual.size(), expected.size());
    auto expected_it = expected.begin();
    auto actual_it = actual.begin();
    for (; expected_it != expected.end(); ++expected_it, ++actual_it) {
        ASSERT_NE(actual_it, actual.end());
        EXPECT_EQ(actual_it->first, expected_it->first);
        const auto &expected_values = expected_it->second;
        const auto &actual_values = actual_it->second;
        ASSERT_EQ(actual_values.size(), expected_values.size());
        EXPECT_EQ(std::memcmp(actual_values.data(), expected_values.data(),
                              static_cast<std::size_t>(expected_values.size()) *
                                  sizeof(double)),
                  0);
    }
}

void expect_identical_telescope_summary(
    const citlali::pipeline::AlignmentTelescopeSummary &expected,
    const citlali::pipeline::AlignmentTelescopeSummary &actual) {
    EXPECT_EQ(actual.initialized, expected.initialized);
    EXPECT_EQ(actual.interface_id, expected.interface_id);
    EXPECT_EQ(actual.coordinate_identity, expected.coordinate_identity);
    EXPECT_EQ(actual.unit, expected.unit);
    EXPECT_EQ(actual.epoch_event_precision_authority,
              expected.epoch_event_precision_authority);
    EXPECT_EQ(actual.support_rule, expected.support_rule);
    EXPECT_EQ(actual.native_row_count, expected.native_row_count);
    expect_same_double_bits(expected.native_first_coordinate_sec,
                            actual.native_first_coordinate_sec);
    expect_same_double_bits(expected.native_last_coordinate_sec,
                            actual.native_last_coordinate_sec);
    EXPECT_EQ(actual.exact_target_count, expected.exact_target_count);
    EXPECT_EQ(actual.interpolated_target_count,
              expected.interpolated_target_count);
    expect_same_double_bits(expected.minimum_used_bracket_span_sec,
                            actual.minimum_used_bracket_span_sec);
    expect_same_double_bits(expected.maximum_used_bracket_span_sec,
                            actual.maximum_used_bracket_span_sec);
    EXPECT_EQ(actual.native_tel_utc_available,
              expected.native_tel_utc_available);
    EXPECT_EQ(actual.native_pps_time_available,
              expected.native_pps_time_available);
}

void expect_exact_native_coincidences_preserved(
    const ProductionTelescopeCase &item,
    const ProductionTelescopeResult &result) {
    for (const auto &[name, aligned_values] : result.aligned_tel_data) {
        const auto native_it = item.native_tel_data.find(name);
        if (native_it == item.native_tel_data.end()) {
            continue;
        }
        const auto &native_values = native_it->second;
        ASSERT_EQ(aligned_values.size(), item.common_time.size());
        for (Eigen::Index native_row = 0; native_row < native_values.size();
             ++native_row) {
            SCOPED_TRACE(name + " native row " +
                         std::to_string(native_row));
            expect_same_double_bits(native_values(native_row),
                                    aligned_values(2 * native_row));
        }
    }
}

TEST(sci_align_T17,
     production_telescope_alignment_is_bit_exact_across_async_reverse_order) {
    const Eigen::VectorXd pointing_time =
        native_time_axis(257, 1706000000.0, 0.01);
    const Eigen::VectorXd beammap_time =
        native_time_axis(193, 1706001000.0, 0.02);
    const std::vector<ProductionTelescopeCase> corpus{
        {"pointing_shaped",
         real_shaped_native_telescope_map(pointing_time, 0.0, true),
         exact_and_midpoint_targets(pointing_time), false},
        {"beammap_shaped",
         real_shaped_native_telescope_map(beammap_time, 0.07, false),
         exact_and_midpoint_targets(beammap_time), true},
    };

    std::vector<ProductionTelescopeResult> sequential;
    sequential.reserve(corpus.size());
    for (const auto &item : corpus) {
        sequential.push_back(run_production_telescope_case(item));
    }

    std::vector<
        std::future<std::pair<std::size_t, ProductionTelescopeResult>>>
        futures;
    futures.reserve(corpus.size());
    for (std::size_t reverse = corpus.size(); reverse > 0; --reverse) {
        const std::size_t index = reverse - 1;
        futures.push_back(std::async(
            std::launch::async, [&corpus, index] {
                return std::pair<std::size_t, ProductionTelescopeResult>{
                    index, run_production_telescope_case(corpus[index])};
            }));
    }

    std::vector<ProductionTelescopeResult> asynchronous(corpus.size());
    for (auto &future : futures) {
        auto [index, result] = future.get();
        asynchronous[index] = std::move(result);
    }

    for (std::size_t index = 0; index < corpus.size(); ++index) {
        SCOPED_TRACE(corpus[index].identity);
        expect_identical_telescope_map(
            sequential[index].aligned_tel_data,
            asynchronous[index].aligned_tel_data);
        expect_identical_telescope_summary(sequential[index].summary,
                                           asynchronous[index].summary);
        EXPECT_EQ(sequential[index].summary.exact_target_count,
                  static_cast<std::uint64_t>(
                      corpus[index].native_tel_data.at("TelTime").size()));
        EXPECT_EQ(sequential[index].summary.interpolated_target_count,
                  static_cast<std::uint64_t>(
                      corpus[index].native_tel_data.at("TelTime").size() -
                      1));
        expect_exact_native_coincidences_preserved(corpus[index],
                                                   sequential[index]);
    }
}

struct ScalarSeries {
    std::vector<double> values;

    double operator()(Eigen::Index index) const {
        return values.at(static_cast<std::size_t>(index));
    }

    Eigen::Index size() const {
        return static_cast<Eigen::Index>(values.size());
    }
};

struct TelescopeState {
    std::map<std::string, ScalarSeries> tel_data;
    Eigen::MatrixXI scan_indices;
};

struct ProvenanceEngine {
    citlali::config::ReductionConfig typed_config;
    citlali::pipeline::TodOutputState tod_outputs;
    citlali::pipeline::OutputPathState output_paths;
    TelescopeState telescope;
    citlali::pipeline::TimestreamAlignmentState alignment;
};

citlali::pipeline::TimestreamAlignmentState valid_compact_alignment() {
    citlali::pipeline::TimestreamAlignmentState state;
    state.grid.initialized = true;
    state.grid.phase_sec = 10.0;
    state.grid.cadence_sec = 0.5;
    state.grid.exclusive_half_cell_sec = 0.25;
    state.grid.first_global_slot = 0;
    state.grid.last_global_slot = 0;
    state.common_time = vector({10.0});
    state.governing_compatibility_axis =
        citlali::pipeline::make_governing_gap_compatibility_axis(
            state.grid, 10.0);
    citlali::pipeline::install_governing_compatibility_assigned_times(state);

    Eigen::VectorXi mask(1);
    mask << 1;
    state.masks.push_back(mask);
    state.interfaces.push_back(
        {"toltec0", 0, 1, 1, 0.0, 0.0, 0.0, 0, 0, 0, 0});

    state.telescope.initialized = true;
    state.telescope.native_row_count = 2;
    state.telescope.native_first_coordinate_sec = 9.5;
    state.telescope.native_last_coordinate_sec = 10.5;
    state.telescope.exact_target_count = 1;
    state.telescope.interpolated_target_count = 0;
    state.telescope.minimum_used_bracket_span_sec = 0.0;
    state.telescope.maximum_used_bracket_span_sec = 0.0;
    state.hwpr =
        citlali::pipeline::bounded_nonpolarimetric_hwpr_summary(false);

    state.support.nominal_slot_count = 1;
    state.support.acquired_original_count = 1;
    state.support.timing_coordinate_valid_original_count = 1;
    state.support.synthesized_count = 0;
    state.support.unavailable_count = 0;
    state.support.guarded_original_count = 0;
    state.support.gap_policy_eligible_original_count = 1;
    state.support.nominal_span_sec = 0.5;
    state.support.acquired_original_cadence_weighted_support_sec = 0.5;
    state.field_registry_version = "sci-align-active-field-registry-v2";
    return state;
}

ProvenanceEngine valid_provenance_engine() {
    ProvenanceEngine engine;
    auto &output = engine.typed_config.timestream.output;
    output.type = citlali::config::TodOutputType::both;
    output.raw_time_chunk.enabled = true;
    output.processed_time_chunk.enabled = true;
    engine.telescope.scan_indices.resize(1, 1);
    engine.tod_outputs.rtc_scan_to_output_scan.resize(1);
    engine.tod_outputs.rtc_scan_to_output_scan << 0;
    engine.tod_outputs.ptc_scan_to_output_scan.resize(1);
    engine.tod_outputs.ptc_scan_to_output_scan << 0;
    engine.tod_outputs.n_rtc_output_scans = 1;
    engine.tod_outputs.n_ptc_output_scans = 1;
    engine.alignment = valid_compact_alignment();
    return engine;
}

TEST(sci_align_T18,
     incomplete_required_compact_provenance_fails_before_any_output) {
    auto engine = valid_provenance_engine();
    engine.alignment.field_registry_version.clear();

    const auto output_dir = std::filesystem::path(::testing::TempDir()) /
                            "sci_align_t18_incomplete";
    std::filesystem::remove_all(output_dir);
    std::filesystem::create_directories(output_dir);
    engine.output_paths.obsnum_dir_name = output_dir.string();
    const auto output_path =
        citlali::pipeline::timestream_output_provenance_path(output_dir);

    EXPECT_THROW(
        citlali::pipeline::write_timestream_output_provenance_file(engine),
        std::logic_error);
    EXPECT_FALSE(std::filesystem::exists(output_path));
    EXPECT_FALSE(std::filesystem::exists(output_path.string() + ".tmp"));
    std::filesystem::remove_all(output_dir);
}

TEST(sci_align_T18, required_atomic_write_failure_propagates_without_output) {
    auto engine = valid_provenance_engine();
    const auto missing_dir = std::filesystem::path(::testing::TempDir()) /
                             "sci_align_t18_missing" / "nested";
    std::filesystem::remove_all(missing_dir.parent_path());
    engine.output_paths.obsnum_dir_name = missing_dir.string();
    const auto output_path =
        citlali::pipeline::timestream_output_provenance_path(missing_dir);

    EXPECT_THROW(
        citlali::pipeline::write_timestream_output_provenance_file(engine),
        std::ios_base::failure);
    EXPECT_FALSE(std::filesystem::exists(output_path));
    EXPECT_FALSE(std::filesystem::exists(output_path.string() + ".tmp"));
}

}  // namespace
