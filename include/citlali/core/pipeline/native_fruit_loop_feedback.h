#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

namespace citlali::pipeline {

inline constexpr const char *native_fruit_loop_feedback_authority_v3 =
    "citlali.fruit_loops.native_projection_v1";

// Bounded realization census for one scan. The projected detector-sample
// model remains runtime state and is regenerated from the previous map.
struct NativeFruitLoopFeedbackSummaryV3 {
    bool enabled = false;
    bool source_model_available = false;
    bool noise_map_pass_applied = false;
    bool keep_source_subtracted_weights = false;
    int iteration = 0;
    std::size_t model_map_count = 0;
    std::size_t subtraction_sample_count = 0;
    std::size_t addback_sample_count = 0;
    std::string interpolation_mode;
    std::string support_authority;

    void validate() const {
        if (!enabled) {
            if (source_model_available || noise_map_pass_applied ||
                keep_source_subtracted_weights || iteration != 0 ||
                model_map_count != 0 || subtraction_sample_count != 0 ||
                addback_sample_count != 0 ||
                !interpolation_mode.empty() ||
                !support_authority.empty()) {
                throw std::logic_error(
                    "disabled native fruit-loop feedback carries realized state");
            }
            return;
        }
        if (iteration < 0 || interpolation_mode.empty() ||
            support_authority != native_fruit_loop_feedback_authority_v3 ||
            (!source_model_available &&
             (noise_map_pass_applied ||
              keep_source_subtracted_weights || model_map_count != 0 ||
              subtraction_sample_count != 0 || addback_sample_count != 0)) ||
            (source_model_available && model_map_count == 0) ||
            addback_sample_count > subtraction_sample_count) {
            throw std::logic_error(
                "native fruit-loop feedback summary is incomplete");
        }
    }
};

}  // namespace citlali::pipeline
