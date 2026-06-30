#pragma once

#include <citlali/core/config/config_error.h>
#include <citlali/core/config/enum_parser.h>

#include <array>
#include <optional>
#include <string>
#include <string_view>

namespace citlali::config {

enum class ReductionType {
    science,
    pointing,
    beammap
};

enum class ParallelPolicy {
    seq,
    omp
};

inline constexpr std::array<EnumName<ReductionType>, 3> reduction_type_names{{
    {ReductionType::science, "science"},
    {ReductionType::pointing, "pointing"},
    {ReductionType::beammap, "beammap"},
}};

inline constexpr std::array<EnumName<ParallelPolicy>, 2> parallel_policy_names{{
    {ParallelPolicy::seq, "seq"},
    {ParallelPolicy::omp, "omp"},
}};

inline std::optional<ReductionType> parse_reduction_type(std::string_view value) {
    return parse_enum(value, reduction_type_names);
}

inline std::optional<ParallelPolicy> parse_parallel_policy(std::string_view value) {
    return parse_enum(value, parallel_policy_names);
}

inline std::string_view to_string(ReductionType value) {
    return enum_name(value, reduction_type_names);
}

inline std::string_view to_string(ParallelPolicy value) {
    return enum_name(value, parallel_policy_names);
}

struct RuntimeConfig {
    bool verbose = true;
    bool interp_over_gaps = true;
    int n_threads = 1;
    std::string output_dir = "/path/to/redu/directory/";
    ParallelPolicy parallel_policy = ParallelPolicy::seq;
    ReductionType reduction_type = ReductionType::science;
    bool use_subdir = true;
};

inline void validate(const RuntimeConfig &config, ValidationReport &report) {
    check_minimum(config.n_threads, 1, {"runtime", "n_threads"}, report);
    if (!config.interp_over_gaps) {
        report.add_error({"runtime", "interp_over_gaps"},
                         "false is not supported by the current pipeline");
    }
    if (config.output_dir.empty()) {
        report.add_error({"runtime", "output_dir"}, "must not be empty");
    }
}

}  // namespace citlali::config
