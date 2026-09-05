#pragma once

#include <citlali/core/config/mapmaking_config.h>
#include <citlali/core/fruit/response_ledger.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

namespace citlali::pipeline {

template <class Engine>
void configure_observation_fruit_response(Engine &engine) {
    if constexpr (requires { engine.learning.fruit_response; engine.omb.fruit_response_ledger; }) {
        auto &buffer = engine.omb;
        buffer.fruit_response_ledger.reset();
        buffer.fruit_response_state = nullptr;
        auto &state = engine.learning.fruit_response;
        if (!state.enabled()) return;
        const auto &config = mapmaking_config(engine);
        if (engine.observation_identity.obsnum != "123424" || engine.iteration.fruit_iter != state.iteration() ||
            !citlali::config::is_jinc_map_method(config.method) || buffer.sig_unit != "mJy/beam" ||
            engine.jinc_mm.run_polarization || engine.jinc_mm.parallel_policy != "seq" ||
            buffer.signal.size() != 3 || engine.map_indices.maps_to_arrays.size() != 3)
            throw std::runtime_error("EL-F12 requires its registered one-observation, all-array serial JINC scope");
        citlali::fruit::ResponseOccurrenceLedger::KernelBank kernels, squares;
        std::vector<int> arrays;
        for (Eigen::Index slot = 0; slot < engine.map_indices.maps_to_arrays.size(); ++slot) {
            const int array = engine.map_indices.maps_to_arrays(slot);
            arrays.push_back(array);
            if (engine.jinc_mm.subpixel_n > 1) {
                kernels[array] = engine.jinc_mm.jinc_weights_mat_subpix.at(array);
                squares[array] = engine.jinc_mm.jinc_weights_sq_mat_subpix.at(array);
            } else {
                kernels[array] = {engine.jinc_mm.jinc_weights_mat.at(array)};
                squares[array] = {engine.jinc_mm.jinc_weights_sq_mat.at(array)};
            }
        }
        if (std::set<int>(arrays.begin(), arrays.end()) != std::set<int>{0, 1, 2})
            throw std::runtime_error("EL-F12 all-array map identity mismatch");
        const auto path = std::filesystem::path(engine.output_paths.redu_dir_name) /
            ("fruit_response_123424_iter" + std::to_string(state.iteration()) + ".bin");
        buffer.fruit_response_ledger = std::make_shared<citlali::fruit::ResponseOccurrenceLedger>(
            path, engine.observation_identity.obsnum, state.iteration(), buffer.n_rows, buffer.n_cols,
            std::move(arrays), std::move(kernels), std::move(squares));
        buffer.fruit_response_state = &state;
    }
}

template <class Learning>
std::vector<citlali::fruit::ResponseCandidate> fruit_response_census(
    const Learning &learning, const std::string &observation, int iteration) {
    using namespace citlali::fruit;
    const auto records = learning.effective_detector_penalty_records();
    std::map<ResponseKey, ResponseCandidate> keyed;
    for (const auto &record : records) {
        if (record.obsnum != observation || !record.scan_local ||
            record.reason != "map_pixel_outlier_detector_dominance" || record.producer.rfind("mapdiag:", 0) != 0)
            continue;
        if (!std::isfinite(record.factor) || record.factor < 0.0 || record.iter < 0 || record.iter > iteration)
            throw std::runtime_error("EL-F12 malformed map-dominance proposal");
        if (record.factor != 0.0) continue;
        ResponseKey key{record.obsnum, record.array, record.uid, record.scan};
        key.validate();
        ResponseCandidate candidate;
        candidate.key = key;
        if (iteration == ResponseInterventionState::horizon) candidate.disposition = "horizon";
        else if (learning.fruit_response.assigned(key)) candidate.disposition = "already_assigned";
        else if (learning.fruit_response_entry_hard_keys.count(key)) candidate.disposition = "entry_hard_exclusion";
        else if (record.iter != iteration) candidate.disposition = "not_new";
        else {
            for (const auto &other : records) {
                if (other.obsnum != observation || !other.scan_local || other.scan != record.scan ||
                    !std::isfinite(other.factor) || other.factor > 0.0 || other.iter > iteration)
                    continue;
                if (other.reason == "map_pixel_outlier_detector_dominance" && other.producer.rfind("mapdiag:", 0) == 0)
                    continue;
                if (other.uid == record.uid || (other.uid < 0 && other.nw >= 0 && other.nw == record.nw)) {
                    candidate.disposition = "independent_exclusion";
                    break;
                }
            }
        }
        auto [it, inserted] = keyed.emplace(key, candidate);
        if (!inserted && it->second.disposition != candidate.disposition)
            throw std::runtime_error("EL-F12 duplicate proposal has inconsistent causal eligibility");
    }
    std::vector<ResponseCandidate> result;
    for (auto &[key, candidate] : keyed) { (void) key; result.push_back(std::move(candidate)); }
    return result;
}

template <class Engine, class MapBuffer>
void finalize_observation_fruit_response(Engine &engine, MapBuffer &buffer, const std::string &filename) {
    if (!engine.learning.fruit_response.enabled()) return;
    if (!buffer.fruit_response_ledger || buffer.fruit_response_state != &engine.learning.fruit_response)
        throw std::runtime_error("EL-F12 missing observation occurrence ledger");
    auto census = fruit_response_census(engine.learning, engine.observation_identity.obsnum, engine.iteration.fruit_iter);
    buffer.fruit_response_ledger->finish(engine.learning.fruit_response, std::move(census),
        buffer.signal, buffer.weight, buffer.weight_formal, buffer.cov_cut);
    buffer.fruit_response_ledger->write_receipt(filename + "_fruit_response.nc",
        engine.learning.fruit_response, buffer.pixel_size_rad, buffer.sig_unit);
}

}  // namespace citlali::pipeline
