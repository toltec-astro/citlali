#pragma once

namespace citlali::pipeline {

template <class KidsProc, class RawObs, class Logger>
auto load_rawobs_kids_meta(KidsProc &kidsproc, const RawObs &rawobs,
                           const Logger &logger) {
    logger->debug("getting rawobs kids meta info");
    return kidsproc.get_rawobs_meta(rawobs);
}

template <class KidsDataProc, class Config>
auto make_kids_data_proc(Config &config) {
    return KidsDataProc::from_config(config.get_config("kids"));
}

template <class KidsDataProc, class Config>
auto make_reduction_observation_kids_proc(Config &config) {
    return make_kids_data_proc<KidsDataProc>(config);
}

template <class Engine, class RawObsKidsMeta, class Logger>
void update_sample_rate_from_rawobs_meta(Engine &engine,
                                         const RawObsKidsMeta &rawobs_kids_meta,
                                         const Logger &logger) {
    logger->debug("getting sample rate");
    engine.telescope.fsmp =
        rawobs_kids_meta.back().template get_typed<double>("fsmp");
}

}  // namespace citlali::pipeline
