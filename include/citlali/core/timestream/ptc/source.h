# pragma once

#include <citlali/core/utils/beam.h>

using namespace citlali::config::options;

// add or subtract gaussian source
enum class SourceMode {
    Add = 1,
    Subtract = -1,
};

// Source
template <typename TCDataType>
class Source : public PipelineComponent<TCDataType> {
public:
    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    double fwhm_limit_factor = 3.0;
    int sign;
    Instrument& toltec;
    Telescope& telescope;
    ObsMaps<>& obs_maps;

    Beam beam;

    template <typename ConfigType>
    Source(SourceMode source_mode, Instrument& toltec_, Telescope& tel_, ObsMaps<>& om_, ConfigType& config)
        : toltec(toltec_), telescope(tel_), obs_maps(om_) {

        sign = static_cast<int>(source_mode);
    }

    void init() override {}
    void process(TCDataType& tcdata) override {
        logger->info("source processing");

        if (obs_maps.params.size() > 0) {
            auto [n_pts, n_dets] = tcdata.dims();

            for (int det = 0; det < n_dets; ++det) {
                if (toltec.apt["flag"].data(det)) continue;

                // keys of current detector
                MapKey key(toltec.apt["array"].data(det), toltec.apt[map_grouping].data(det), "I");
                // model params for map
                auto params = obs_maps.params.col(obs_maps.signal_lookup[key]);

                auto xy = calc_pointing(params(1), params(2), tcdata.tel_data, telescope.pixel_axes);

                Eigen::VectorXd beam_params(6);
                for (int i = 0; i < n_pts; ++i) {
                    // use maximum of fwhms due to atmospheric cleaning along scan direction which squishes sources
                    double fwhm = std::max(params(3), params(4));

                    beam_params << fwhm * ASEC_TO_RAD * FWHM_TO_STD, fwhm * ASEC_TO_RAD * FWHM_TO_STD,
                        params(5), params(1) * ASEC_TO_RAD, params(2) * ASEC_TO_RAD, params(0);

                    double beam_value = beam.calculate_gaussian(xy.first(i), xy.second(i), beam_params, fwhm_limit_factor * fwhm);
                    tcdata.signal(i, det) += sign * beam_value;
                }
            }
        }
    }
};
