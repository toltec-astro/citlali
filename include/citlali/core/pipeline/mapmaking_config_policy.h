#pragma once

namespace citlali::pipeline {

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

}  // namespace citlali::pipeline
