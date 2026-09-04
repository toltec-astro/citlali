#pragma once

#include <citlali/core/config/mapmaking_config.h>
#include <citlali/core/pipeline/output_policy.h>

#include <stdexcept>
#include <type_traits>
#include <utility>

namespace citlali::pipeline {

template <class Engine>
bool should_allocate_observation_map_buffers(const Engine &engine) {
    return mapmaking_outputs_enabled(engine);
}

template <class Engine>
void configure_observation_pixel_contribution_targets(Engine &engine) {
    engine.configure_map_pixel_contribution_targets(engine.omb, "raw_obs");
}

template <class Engine, class = void>
struct has_observation_jinc_accounting_boundary : std::false_type {};

template <class Engine>
struct has_observation_jinc_accounting_boundary<
    Engine,
    std::void_t<
        decltype(std::declval<Engine &>().omb.jinc_accounting),
        decltype(std::declval<Engine &>().omb.signal),
        decltype(std::declval<Engine &>().toltec_io.array_name_map),
        decltype(std::declval<Engine &>().map_indices.maps_to_arrays),
        decltype(std::declval<Engine &>().jinc_mm.shape_params),
        decltype(std::declval<Engine &>().observation_identity.obsnum),
        decltype(std::declval<Engine &>().iteration.fruit_iter)>>
    : std::true_type {};

template <class Engine>
void configure_observation_jinc_accounting(Engine &engine) {
    if constexpr (!has_observation_jinc_accounting_boundary<Engine>::value) {
        return;
    }
    else {
        auto &state = engine.omb.jinc_accounting;
        state.clear();
        const auto &config = mapmaking_config(engine);
        const auto &request = config.jinc_accounting;
        if (!request.enabled) {
            return;
        }
        if (!citlali::config::is_jinc_map_method(config.method) ||
            !citlali::config::is_array_map_grouping(config.grouping) ||
            polarimetry_config(engine).enabled) {
            throw std::runtime_error(
                "JINC accounting requires non-polarized raw-observation JINC array maps");
        }

        int array_id = -1;
        for (const auto &[candidate_id, candidate_name] :
             engine.toltec_io.array_name_map) {
            if (candidate_name == request.array) {
                if (array_id >= 0) {
                    throw std::runtime_error(
                        "JINC accounting selected an ambiguous array name");
                }
                array_id = static_cast<int>(candidate_id);
            }
        }
        Eigen::Index map_index = -1;
        int matching_slots = 0;
        for (Eigen::Index slot = 0;
             slot < engine.map_indices.maps_to_arrays.size(); ++slot) {
            if (engine.map_indices.maps_to_arrays(slot) == array_id) {
                map_index = slot;
                matching_slots++;
            }
        }
        if (array_id < 0 || matching_slots != 1 || map_index < 0 ||
            map_index >= static_cast<Eigen::Index>(engine.omb.signal.size())) {
            throw std::runtime_error(
                "JINC accounting target array does not resolve to exactly one raw-observation map");
        }
        const auto shape_it = engine.jinc_mm.shape_params.find(array_id);
        if (shape_it == engine.jinc_mm.shape_params.end() ||
            shape_it->second.size() != 3) {
            throw std::runtime_error(
                "JINC accounting cannot identify the selected kernel shape");
        }
        std::vector<double> shape(
            shape_it->second.data(),
            shape_it->second.data() + shape_it->second.size());
        state.configure(
            request.array, array_id, request.uid, request.scan_index,
            map_index, engine.observation_identity.obsnum,
            engine.iteration.fruit_iter, engine.jinc_mm.subpixel_n,
            engine.jinc_mm.r_max, std::move(shape), engine.omb.n_rows,
            engine.omb.n_cols);
    }
}

template <class Engine>
bool should_allocate_observation_noise_maps(const Engine &engine) {
    // Noise realizations are observation-owned until the whole normalized map
    // bundle passes SCI-MAP-001 coadd admission. This keeps signal, kernel, and
    // realizations on one accepted map-operator boundary.
    return noise_maps_enabled(engine);
}

}  // namespace citlali::pipeline
