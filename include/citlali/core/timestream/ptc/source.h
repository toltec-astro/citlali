# pragma once

#include <citlali/core/utils/beam.h>

// add or subtract gaussian source
enum SourceType {
    Add = 1,
    Subtract = -1,
};

// Source
template <typename TCDataType>
class Source : public PipelineComponent<TCDataType> {
public:
    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    double fwhm_limit_factor_ = 3.0;
    int add_subtract_factor;
    Instrument& toltec;
    Telescope& telescope;
    DataMapsContainer& obs_maps;

    Beam beam;

    template <typename ConfigType>
    Source(SourceType source_type, Instrument& toltec_ref, Telescope& telescope_ref, DataMapsContainer& obs_maps_ref, ConfigType& config)
        : toltec(toltec_ref), telescope(telescope_ref), obs_maps(obs_maps_ref) {

        add_subtract_factor = static_cast<int>(source_type);
    }

    void init() {}
    void process(TCDataType& tcdata) override {
        logger->info("source processing");

        int n_dets = tcdata.n_dets();
        int n_pts = tcdata.n_pts();

        for (int det = 0; det < n_dets; ++det) {
            if (toltec.apt["flag"].data(det)) continue;

            // keys of current detector
            int array = toltec.apt["array"].data(det);
            int group = toltec.apt[obs_maps.map_grouping].data(det);
            auto params = obs_maps.fits[array][group].params.i;

            auto xy = telescope.calc_pointing(params(1), params(2), tcdata.tel_data);

            Eigen::VectorXd beam_params(6);
            for (int i = 0; i < n_pts; ++i) {
                // use maximum of fwhms due to atmospheric cleaning along scan direction
                double fwhm = std::max(params(3), params(4));

                beam_params << fwhm * ASEC_TO_RAD * FWHM_TO_STD, fwhm * ASEC_TO_RAD * FWHM_TO_STD,
                    params(5), params(1) * ASEC_TO_RAD, params(2) * ASEC_TO_RAD, params(0);

                double beam_value = beam.calculate_gaussian(xy.first(i), xy.second(i), beam_params, fwhm_limit_factor_ * fwhm);
                tcdata.signal(i, det) += add_subtract_factor * beam_value;
            }
        }
    }
};
