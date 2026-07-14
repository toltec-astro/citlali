#pragma once

#include <citlali/core/config/beammap_config.h>
#include <citlali/core/config/config_value.h>

#include <algorithm>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace citlali::pipeline {

struct BeammapRequestPresence {
    bool max_d2_iter0 = false;
    bool max_d2_after_iter0 = false;
    bool score_lambda_iter0 = false;
    bool score_lambda_after_iter0 = false;
    bool split_flag_values = false;
};

struct BeammapEffectiveResolutionRecord {
    BeammapRequestPresence explicit_request;
    bool mapmaking_enabled = false;
    int requested_max_iterations = 1;
    int effective_max_iterations = 1;
    bool max_iterations_forced_without_mapmaking = false;
    int requested_locator_iter = 0;
    int effective_locator_iter = 0;
    bool locator_iter_forced_zero = false;
    int requested_measurement_start_iter = 1;
    int effective_measurement_start_iter = 1;
    bool measurement_start_iter_adjusted = false;
    bool legacy_phase_behavior = false;
    bool measurement_pass_available = false;
    bool convergence_check_available = false;
    bool convergence_active = false;
    bool prior_path_available = false;
    bool priors_disabled_by_missing_path = false;
    bool max_d2_iter0_inherited = false;
    bool max_d2_after_iter0_inherited = false;
    bool score_lambda_iter0_inherited = false;
    bool score_lambda_after_iter0_inherited = false;
    bool split_flag_values_defaulted = false;
    bool split_flag_values_sorted = false;
    bool split_flag_values_deduplicated = false;
    std::size_t requested_split_flag_count = 0;
    std::size_t effective_split_flag_count = 0;
};

enum class BeammapIterationPhase {
    legacy,
    locator,
    pre_measurement,
    measurement_start,
    measurement,
};

inline const char *beammap_iteration_phase_name(
    BeammapIterationPhase phase) {
    switch (phase) {
        case BeammapIterationPhase::legacy:
            return "legacy";
        case BeammapIterationPhase::locator:
            return "locator";
        case BeammapIterationPhase::pre_measurement:
            return "pre_measurement";
        case BeammapIterationPhase::measurement_start:
            return "measurement_start";
        case BeammapIterationPhase::measurement:
            return "measurement";
    }
    throw std::logic_error("unknown beammap iteration phase");
}

enum class BeammapTerminationReason {
    none,
    maximum_iterations,
    all_maps_converged,
};

inline const char *beammap_termination_reason_name(
    BeammapTerminationReason reason) {
    switch (reason) {
        case BeammapTerminationReason::none:
            return "none";
        case BeammapTerminationReason::maximum_iterations:
            return "maximum_iterations";
        case BeammapTerminationReason::all_maps_converged:
            return "all_maps_converged";
    }
    throw std::logic_error("unknown beammap termination reason");
}

struct BeammapIterationState {
    std::size_t iteration_index = 0;
    BeammapIterationPhase phase = BeammapIterationPhase::legacy;
    std::size_t active_map_count = 0;
    std::size_t mapmaking_pass_count = 0;
    std::optional<bool> source_aware_rtc_rerun;
    bool fitting_completed = false;
    std::size_t newly_converged_map_count = 0;
    std::size_t total_converged_map_count = 0;
    BeammapTerminationReason termination_reason =
        BeammapTerminationReason::none;
    bool completed = false;
};

struct BeammapDetectorTodRealizedState {
    bool required = false;
    std::size_t completed_write_count = 0;
    std::optional<std::size_t> output_iteration;
    std::optional<std::size_t> detector_count;
    std::optional<std::size_t> slot_count;
    std::optional<std::size_t> maximum_sample_count;
};

struct BeammapObservationState {
    std::size_t observation_index = 0;
    std::string obsnum;
    std::size_t detector_count = 0;
    std::size_t map_count = 0;
    std::size_t scan_count = 0;
    std::vector<BeammapIterationState> iterations;
    std::optional<std::size_t> terminal_iteration;
    BeammapTerminationReason termination_reason =
        BeammapTerminationReason::none;
    BeammapDetectorTodRealizedState detector_tod;
    bool outputs_completed = false;
};

struct BeammapRealizedState {
    bool reduction_completed = false;
    bool beammap_executed = false;
    std::optional<std::size_t> completed_observation_count;
    std::size_t completed_iteration_count = 0;
    bool outputs_completed = false;
};

inline std::vector<int> resolve_beammap_split_flag_values(
    const std::vector<int> &requested,
    BeammapEffectiveResolutionRecord &resolution) {
    resolution.requested_split_flag_count = requested.size();
    if (requested.empty()) {
        resolution.split_flag_values_defaulted = true;
        resolution.effective_split_flag_count = 2;
        return {0, 1};
    }

    auto effective = requested;
    resolution.split_flag_values_sorted =
        !std::is_sorted(effective.begin(), effective.end());
    std::sort(effective.begin(), effective.end());
    const auto unique_end = std::unique(effective.begin(), effective.end());
    resolution.split_flag_values_deduplicated =
        unique_end != effective.end();
    effective.erase(unique_end, effective.end());
    resolution.effective_split_flag_count = effective.size();
    return effective;
}

class BeammapExecutionPlan {
public:
    [[nodiscard]] bool initialized() const { return initialized_; }

    [[nodiscard]] const citlali::config::BeammapConfig &requested() const {
        return requested_;
    }

    [[nodiscard]] const citlali::config::BeammapConfig &effective() const {
        return effective_;
    }

    [[nodiscard]] const BeammapEffectiveResolutionRecord &resolution() const {
        return resolution_;
    }

    [[nodiscard]] const std::vector<BeammapObservationState> &observations()
        const {
        return observations_;
    }

    [[nodiscard]] const BeammapRealizedState &realized() const {
        return realized_;
    }

    void reset_from_request(
        const citlali::config::BeammapConfig &request,
        const BeammapRequestPresence &presence,
        bool mapmaking_enabled) {
        requested_ = request;
        effective_ = request;
        resolution_ = {};
        resolution_.explicit_request = presence;
        resolution_.mapmaking_enabled = mapmaking_enabled;
        resolution_.requested_max_iterations =
            request.iteration.max_iterations;
        if (!mapmaking_enabled) {
            effective_.iteration.max_iterations = 1;
            resolution_.max_iterations_forced_without_mapmaking =
                request.iteration.max_iterations != 1;
        }
        resolution_.effective_max_iterations =
            effective_.iteration.max_iterations;

        resolution_.requested_locator_iter =
            request.phase_strategy.locator_iter;
        if (effective_.phase_strategy.locator_iter != 0) {
            effective_.phase_strategy.locator_iter = 0;
            resolution_.locator_iter_forced_zero = true;
        }
        resolution_.effective_locator_iter =
            effective_.phase_strategy.locator_iter;
        resolution_.requested_measurement_start_iter =
            request.phase_strategy.measurement_start_iter;
        if (effective_.phase_strategy.measurement_start_iter <=
            effective_.phase_strategy.locator_iter) {
            effective_.phase_strategy.measurement_start_iter =
                effective_.phase_strategy.locator_iter + 1;
            resolution_.measurement_start_iter_adjusted = true;
        }
        resolution_.effective_measurement_start_iter =
            effective_.phase_strategy.measurement_start_iter;
        resolution_.legacy_phase_behavior =
            !effective_.phase_strategy.enabled;
        const int first_measurement_iteration =
            effective_.phase_strategy.enabled
                ? effective_.phase_strategy.measurement_start_iter
                : 1;
        resolution_.measurement_pass_available =
            effective_.iteration.max_iterations > first_measurement_iteration;
        resolution_.convergence_check_available =
            effective_.iteration.max_iterations >
            first_measurement_iteration + 1;
        resolution_.convergence_active =
            mapmaking_enabled && effective_.iteration.tolerance > 0.0 &&
            resolution_.convergence_check_available;

        resolution_.max_d2_iter0_inherited = !presence.max_d2_iter0;
        if (resolution_.max_d2_iter0_inherited) {
            effective_.priors.max_d2_iter0 = effective_.priors.max_d2;
        }
        resolution_.max_d2_after_iter0_inherited =
            !presence.max_d2_after_iter0;
        if (resolution_.max_d2_after_iter0_inherited) {
            effective_.priors.max_d2_after_iter0 = effective_.priors.max_d2;
        }
        resolution_.score_lambda_iter0_inherited =
            !presence.score_lambda_iter0;
        if (resolution_.score_lambda_iter0_inherited) {
            effective_.priors.score_lambda_iter0 =
                effective_.priors.score_lambda;
        }
        resolution_.score_lambda_after_iter0_inherited =
            !presence.score_lambda_after_iter0;
        if (resolution_.score_lambda_after_iter0_inherited) {
            effective_.priors.score_lambda_after_iter0 =
                effective_.priors.score_lambda;
        }
        resolution_.prior_path_available =
            citlali::config::has_config_value(effective_.priors.filepath);
        resolution_.priors_disabled_by_missing_path =
            effective_.priors.enabled && !resolution_.prior_path_available;
        if (resolution_.priors_disabled_by_missing_path) {
            effective_.priors.enabled = false;
        }

        effective_.split_fits_by_flag.flag_values =
            resolve_beammap_split_flag_values(
                presence.split_flag_values
                    ? request.split_fits_by_flag.flag_values
                    : std::vector<int>{},
                resolution_);
        observations_.clear();
        realized_ = {};
        initialized_ = true;
    }

    void begin_iteration() {
        require_initialized();
        observations_.clear();
        realized_ = {};
        realized_.completed_observation_count = std::size_t{0};
    }

    BeammapObservationState &begin_observation(
        std::size_t observation_index, std::string obsnum,
        std::size_t detector_count, std::size_t map_count,
        std::size_t scan_count) {
        require_active_iteration();
        if (!resolution_.mapmaking_enabled) {
            throw std::logic_error(
                "beammap observation requires enabled mapmaking");
        }
        if (obsnum.empty() || detector_count == 0 || map_count == 0 ||
            scan_count == 0) {
            throw std::logic_error(
                "beammap observation identity and counts must be positive");
        }
        if (!observations_.empty() &&
            !observations_.back().outputs_completed) {
            throw std::logic_error(
                "previous beammap observation is incomplete");
        }
        observations_.push_back(BeammapObservationState{
            observation_index, std::move(obsnum), detector_count,
            map_count, scan_count});
        observations_.back().detector_tod.required =
            effective_.detector_tod_output.enabled;
        return observations_.back();
    }

    BeammapIterationState &begin_internal_iteration(
        std::size_t iteration_index, BeammapIterationPhase phase,
        std::size_t active_map_count) {
        auto &observation = current_observation();
        if (observation.outputs_completed ||
            observation.terminal_iteration.has_value()) {
            throw std::logic_error(
                "cannot begin iteration for completed beammap observation");
        }
        if (!observation.iterations.empty() &&
            !observation.iterations.back().completed) {
            throw std::logic_error(
                "previous beammap iteration is incomplete");
        }
        if (iteration_index != observation.iterations.size()) {
            throw std::logic_error(
                "beammap iteration indices must be contiguous from zero");
        }
        if (active_map_count == 0 ||
            active_map_count > observation.map_count) {
            throw std::logic_error(
                "beammap active map count is inconsistent");
        }
        observation.iterations.push_back(BeammapIterationState{
            iteration_index, phase, active_map_count});
        return observation.iterations.back();
    }

    void record_source_aware_rtc_rerun(bool rerun) {
        auto &iteration = current_internal_iteration();
        if (iteration.source_aware_rtc_rerun.has_value()) {
            throw std::logic_error(
                "beammap source-aware RTC decision already recorded");
        }
        iteration.source_aware_rtc_rerun = rerun;
    }

    void record_mapmaking_pass_completed() {
        auto &iteration = current_internal_iteration();
        ++iteration.mapmaking_pass_count;
    }

    void record_fitting_completed() {
        auto &iteration = current_internal_iteration();
        if (iteration.fitting_completed) {
            throw std::logic_error(
                "beammap fitting completion already recorded");
        }
        if (iteration.mapmaking_pass_count == 0) {
            throw std::logic_error(
                "beammap fitting completed before mapmaking");
        }
        iteration.fitting_completed = true;
    }

    void record_detector_tod_written(
        std::size_t output_iteration, std::size_t detector_count,
        std::size_t slot_count, std::size_t maximum_sample_count) {
        const auto &iteration = current_internal_iteration();
        auto &observation = current_observation();
        auto &detector_tod = observation.detector_tod;
        if (!detector_tod.required) {
            throw std::logic_error(
                "unexpected beammap detector TOD write");
        }
        if (detector_tod.completed_write_count != 0) {
            throw std::logic_error(
                "beammap detector TOD was written more than once");
        }
        if (output_iteration != iteration.iteration_index ||
            detector_count != observation.detector_count ||
            slot_count == 0 || maximum_sample_count == 0) {
            throw std::logic_error(
                "beammap detector TOD output shape is inconsistent");
        }
        detector_tod.completed_write_count = 1;
        detector_tod.output_iteration = output_iteration;
        detector_tod.detector_count = detector_count;
        detector_tod.slot_count = slot_count;
        detector_tod.maximum_sample_count = maximum_sample_count;
    }

    void complete_internal_iteration(
        std::size_t total_converged_map_count,
        BeammapTerminationReason termination_reason) {
        auto &observation = current_observation();
        auto &iteration = current_internal_iteration();
        if (!iteration.source_aware_rtc_rerun.has_value() ||
            iteration.mapmaking_pass_count == 0 ||
            !iteration.fitting_completed) {
            throw std::logic_error(
                "beammap iteration lifecycle is incomplete");
        }
        if (total_converged_map_count > observation.map_count) {
            throw std::logic_error(
                "beammap converged map count exceeds map count");
        }
        const std::size_t previous_converged =
            observation.iterations.size() > 1
                ? observation.iterations[
                      observation.iterations.size() - 2]
                      .total_converged_map_count
                : 0;
        if (total_converged_map_count < previous_converged) {
            throw std::logic_error(
                "beammap converged map count decreased");
        }
        const auto completed_count = iteration.iteration_index + 1;
        const auto maximum_iterations = static_cast<std::size_t>(
            effective_.iteration.max_iterations);
        if (termination_reason ==
                BeammapTerminationReason::maximum_iterations &&
            completed_count != maximum_iterations) {
            throw std::logic_error(
                "beammap maximum-iteration termination is inconsistent");
        }
        if (termination_reason ==
                BeammapTerminationReason::all_maps_converged &&
            total_converged_map_count != observation.map_count) {
            throw std::logic_error(
                "beammap convergence termination is incomplete");
        }
        if (termination_reason == BeammapTerminationReason::none &&
            (completed_count >= maximum_iterations ||
             total_converged_map_count == observation.map_count)) {
            throw std::logic_error(
                "non-terminal beammap iteration has terminal state");
        }
        iteration.newly_converged_map_count =
            total_converged_map_count - previous_converged;
        iteration.total_converged_map_count =
            total_converged_map_count;
        iteration.termination_reason = termination_reason;
        iteration.completed = true;
    }

    void complete_observation() {
        auto &observation = current_observation();
        if (observation.outputs_completed) {
            throw std::logic_error(
                "beammap observation outputs already completed");
        }
        if (observation.iterations.empty() ||
            !observation.iterations.back().completed ||
            observation.iterations.back().termination_reason ==
                BeammapTerminationReason::none) {
            throw std::logic_error(
                "beammap observation has no completed terminal iteration");
        }
        const auto &detector_tod = observation.detector_tod;
        const std::size_t expected_detector_tod_writes =
            detector_tod.required ? 1 : 0;
        if (detector_tod.completed_write_count !=
            expected_detector_tod_writes) {
            throw std::logic_error(
                "beammap detector TOD write cardinality is incomplete");
        }
        observation.terminal_iteration =
            observation.iterations.back().iteration_index;
        observation.termination_reason =
            observation.iterations.back().termination_reason;
        observation.outputs_completed = true;
        ++*realized_.completed_observation_count;
        realized_.completed_iteration_count +=
            observation.iterations.size();
    }

    void complete_reduction(bool beammap_executed) {
        require_active_iteration();
        if (realized_.reduction_completed) {
            throw std::logic_error(
                "beammap reduction is already completed");
        }
        if (beammap_executed != resolution_.mapmaking_enabled) {
            throw std::logic_error(
                "beammap execution state differs from effective policy");
        }
        const auto completed_observations = static_cast<std::size_t>(
            std::count_if(
                observations_.begin(), observations_.end(),
                [](const auto &observation) {
                    return observation.outputs_completed;
                }));
        std::size_t completed_iterations = 0;
        for (const auto &observation : observations_) {
            completed_iterations += static_cast<std::size_t>(
                std::count_if(
                    observation.iterations.begin(),
                    observation.iterations.end(),
                    [](const auto &iteration) {
                        return iteration.completed;
                    }));
        }
        if (*realized_.completed_observation_count !=
                completed_observations ||
            completed_observations != observations_.size() ||
            realized_.completed_iteration_count != completed_iterations) {
            throw std::logic_error(
                "beammap lifecycle cardinality is incomplete");
        }
        if (beammap_executed && observations_.empty()) {
            throw std::logic_error(
                "beammap completed without observations");
        }
        if (!beammap_executed && !observations_.empty()) {
            throw std::logic_error(
                "disabled beammap recorded observations");
        }
        realized_.beammap_executed = beammap_executed;
        realized_.outputs_completed = true;
        realized_.reduction_completed = true;
    }

private:
    void require_initialized() const {
        if (!initialized_) {
            throw std::logic_error("beammap plan is not initialized");
        }
    }

    void require_active_iteration() const {
        require_initialized();
        if (!realized_.completed_observation_count.has_value()) {
            throw std::logic_error(
                "beammap reduction iteration was not initialized");
        }
    }

    BeammapObservationState &current_observation() {
        require_active_iteration();
        if (observations_.empty()) {
            throw std::logic_error(
                "beammap observation has not begun");
        }
        return observations_.back();
    }

    BeammapIterationState &current_internal_iteration() {
        auto &observation = current_observation();
        if (observation.iterations.empty()) {
            throw std::logic_error(
                "beammap internal iteration has not begun");
        }
        auto &iteration = observation.iterations.back();
        if (iteration.completed) {
            throw std::logic_error(
                "beammap internal iteration is already completed");
        }
        return iteration;
    }

    bool initialized_ = false;
    citlali::config::BeammapConfig requested_;
    citlali::config::BeammapConfig effective_;
    BeammapEffectiveResolutionRecord resolution_;
    std::vector<BeammapObservationState> observations_;
    BeammapRealizedState realized_;
};

inline void install_beammap_effective_compatibility_config(
    const BeammapExecutionPlan &plan,
    citlali::config::BeammapConfig &target) {
    if (!plan.initialized()) {
        throw std::logic_error(
            "cannot install an uninitialized beammap execution plan");
    }
    target = plan.effective();
}

template <class Logger>
void log_beammap_effective_resolution(
    const BeammapExecutionPlan &plan, const Logger &logger) {
    if (!plan.initialized()) {
        throw std::logic_error(
            "cannot log an uninitialized beammap execution plan");
    }
    const auto &requested = plan.requested();
    const auto &effective = plan.effective();
    const auto &resolution = plan.resolution();
    if (resolution.locator_iter_forced_zero) {
        logger->warn(
            "beammap.phase_strategy.locator_iter={} requested, but the locator pass must be iter 0; using 0",
            resolution.requested_locator_iter);
    }
    if (resolution.measurement_start_iter_adjusted) {
        logger->warn(
            "beammap.phase_strategy.measurement_start_iter={} must be after locator_iter={}; using {}",
            resolution.requested_measurement_start_iter,
            effective.phase_strategy.locator_iter,
            resolution.effective_measurement_start_iter);
    }
    if (resolution.requested_max_iterations <=
        resolution.effective_measurement_start_iter) {
        logger->warn(
            "beammap.iter_max={} will not run a measurement pass with measurement_start_iter={}",
            resolution.requested_max_iterations,
            resolution.effective_measurement_start_iter);
    }
    if (resolution.explicit_request.split_flag_values &&
        requested.split_fits_by_flag.flag_values.empty()) {
        logger->warn(
            "beammap.split_fits_by_flag.flag_values is empty; using defaults [0, 1]");
    }
    if (resolution.priors_disabled_by_missing_path) {
        logger->warn(
            "beammap.priors.enabled=true but beammap.priors.filepath is null; disabling priors");
    }
}

}  // namespace citlali::pipeline
