#pragma once

#include <cstdlib>
#include <string>
#include <vector>

namespace citlali::pipeline {

inline std::vector<std::string> allowed_map_regimes() {
    return {"source_dominant", "source_faint", "blank_field", "unknown"};
}

inline bool map_grouping_disallows_polarization(
    bool run_polarization, const std::string &redu_type,
    const std::string &map_grouping) {
    return run_polarization &&
           ((redu_type == "beammap" && map_grouping == "auto") ||
            map_grouping == "detector");
}

template <class Logger>
void enforce_map_grouping_polarization_policy(
    bool run_polarization, const std::string &redu_type,
    const std::string &map_grouping, const Logger &logger) {
    if (!map_grouping_disallows_polarization(
            run_polarization, redu_type, map_grouping)) {
        return;
    }
    logger->error(
        "Detector grouping reductions do not currently support polarimetry mode");
    std::exit(EXIT_FAILURE);
}

template <class OutputMapBlock, class CoaddMapBlock>
void mirror_noise_map_settings_to_coadd(const OutputMapBlock &omb,
                                        CoaddMapBlock &cmb) {
    cmb.n_noise = omb.n_noise;
    cmb.randomize_dets = omb.randomize_dets;
}

template <class OutputMapBlock, class CoaddMapBlock, class NoiseConfig>
void disable_noise_map_settings(OutputMapBlock &omb, CoaddMapBlock &cmb,
                                NoiseConfig &typed_noise_config) {
    omb.n_noise = 0;
    cmb.n_noise = 0;
    typed_noise_config.n_noise_maps = 0;
}

template <class NaiveMapmaker, class JincMapmaker>
void set_mapmaker_polarization(bool run_polarization,
                               NaiveMapmaker &naive_mm,
                               JincMapmaker &jinc_mm) {
    naive_mm.run_polarization = run_polarization;
    jinc_mm.run_polarization = run_polarization;
}

}  // namespace citlali::pipeline
