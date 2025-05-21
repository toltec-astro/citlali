# pragma once

#include <citlali/core/utils/fft.h>

// MapPsd
template <typename MapType>
class MapPsd : public PipelineComponent<MapType> {
public:
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    Instrument& toltec;
    Telescope& telescope;

    template <typename ConfigType>
    MapPsd(Instrument& toltec_, Telescope& telescope_, ConfigType& config)
        : toltec(toltec_), telescope(telescope_) {}

    void init() {}
    void process(MapType& maps) override {
        logger->info("psd processing");

        double dx = 1. / (pix_size_radians * maps.wcs.naxis[0]);
        double dy = 1. / (pix_size_radians * maps.wcs.naxis[1]);

        for (int i = 0; i < maps.signal.size(); ++i) {
            auto [radial_psd, radial_freqs, psd, freqs] = citlali::utils::fft::calc_psd_2d(maps.signal[i].data, dy, dx);

            maps.radial_psd.push_back(radial_psd);
            maps.radial_freqs.push_back(radial_freqs);
            maps.psd.push_back(psd);
            maps.freqs.push_back(freqs);
        }
    }
};
