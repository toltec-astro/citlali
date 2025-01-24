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
    MapPsd(InstrumentContainer& toltec_ref, Telescope& telescope_ref, ConfigType& config)
        : toltec(toltec_ref), telescope(telescope_ref) {}

    void init() {}
    void process(MapType& maps) override {
        logger->info("psd processing");

        double dx = 1. / (maps.pix_size_radians * maps.n_cols);
        double dy = 1. / (maps.pix_size_radians * maps.n_rows);

        for (int i = 0; i < maps.signal_map.size(); ++i) {
            auto [radial_psd, radial_freqs, psd, freqs] = citlali::utils::fft::calc_psd_2d(maps.signal_map[i], dy, dx);
        }
    }
};
