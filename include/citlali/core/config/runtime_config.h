#pragma once

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

inline bool is_reduction_type(std::string_view value, ReductionType type) {
    return value == to_string(type);
}

inline bool is_reduction_type(ReductionType value, ReductionType type) {
    return value == type;
}

inline bool is_science_reduction_type(std::string_view value) {
    return is_reduction_type(value, ReductionType::science);
}

inline bool is_science_reduction_type(ReductionType value) {
    return is_reduction_type(value, ReductionType::science);
}

inline bool is_pointing_reduction_type(std::string_view value) {
    return is_reduction_type(value, ReductionType::pointing);
}

inline bool is_pointing_reduction_type(ReductionType value) {
    return is_reduction_type(value, ReductionType::pointing);
}

inline bool is_beammap_reduction_type(std::string_view value) {
    return is_reduction_type(value, ReductionType::beammap);
}

inline bool is_beammap_reduction_type(ReductionType value) {
    return is_reduction_type(value, ReductionType::beammap);
}

struct RuntimeConfig {
    bool verbose = true;
    bool interp_over_gaps = true;
    // Explicit diagnostic-only admission policy. It does not alter raw files
    // or detector timestamps; it only limits the in-memory common lattice to
    // rows bracketed by native telescope support.
    bool crop_detector_to_telescope_support = false;
    int n_threads = 1;
    std::string output_dir = "/path/to/redu/directory/";
    ParallelPolicy parallel_policy = ParallelPolicy::seq;
    ReductionType reduction_type = ReductionType::science;
    bool use_subdir = true;
};

inline bool reduction_subdirs_active(const RuntimeConfig &config) {
    return config.use_subdir;
}

inline bool timing_gap_interpolation_active(const RuntimeConfig &config) {
    return config.interp_over_gaps;
}

}  // namespace citlali::config
