#pragma once

#include <citlali/core/config/timestream_config.h>

#include <algorithm>
#include <cctype>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace citlali::pipeline {

inline std::string normalize_processed_clean_group(std::string group) {
    std::transform(
        group.begin(), group.end(), group.begin(), [](unsigned char value) {
            return static_cast<char>(std::tolower(value));
        });
    if (group == "network") {
        return "nw";
    }
    return group;
}

inline bool is_supported_processed_clean_group(std::string_view group) {
    return group == "all" || group == "array" || group == "nw" ||
           group == "detector" || group == "fg" || group == "corr_nw";
}

struct ProcessedCleanGroupingResolution {
    std::vector<std::string> effective;
    std::vector<std::string> unsupported;
    std::vector<std::string> duplicates;
    int aliases_normalized = 0;
};

struct ProcessedCleanerModeResolution {
    citlali::config::ProcessedTimeChunkCleanerMode effective =
        citlali::config::ProcessedTimeChunkCleanerMode::none;
    int enabled_mode_count = 0;
};

inline ProcessedCleanerModeResolution resolve_processed_cleaner_mode(
    const citlali::config::ProcessedTimeChunkCleanConfig &requested) {
    ProcessedCleanerModeResolution resolution;
    const auto select = [&](bool enabled, auto mode) {
        if (!enabled) {
            return;
        }
        ++resolution.enabled_mode_count;
        if (resolution.effective ==
            citlali::config::ProcessedTimeChunkCleanerMode::none) {
            resolution.effective = mode;
        }
    };
    if (!requested.enabled) {
        return resolution;
    }
    select(requested.standard_pca.enabled,
           citlali::config::ProcessedTimeChunkCleanerMode::standard_pca);
    select(requested.null_model.enabled,
           citlali::config::ProcessedTimeChunkCleanerMode::null_model);
    select(
        requested.marchenko_pastur.enabled,
        citlali::config::ProcessedTimeChunkCleanerMode::marchenko_pastur);
    select(
        requested.adaptive_selector.enabled,
        citlali::config::ProcessedTimeChunkCleanerMode::adaptive_selector);
    return resolution;
}

inline ProcessedCleanGroupingResolution resolve_processed_clean_grouping(
    const std::vector<std::string> &requested) {
    ProcessedCleanGroupingResolution resolution;
    std::unordered_set<std::string> seen;
    for (const auto &raw_group : requested) {
        auto group = normalize_processed_clean_group(raw_group);
        if (group != raw_group) {
            ++resolution.aliases_normalized;
        }
        if (!is_supported_processed_clean_group(group)) {
            resolution.unsupported.push_back(raw_group);
            continue;
        }
        if (!seen.insert(group).second) {
            resolution.duplicates.push_back(raw_group);
            continue;
        }
        resolution.effective.push_back(std::move(group));
    }
    return resolution;
}

}  // namespace citlali::pipeline
