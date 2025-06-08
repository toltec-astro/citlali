#pragma once

// MapHist
template <typename MapType>
class MapHist : public PipelineComponent<MapType> {
public:
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    Instrument& toltec;
    Telescope& telescope;

    template <typename ConfigType>
    MapHist(Instrument& toltec_, Telescope& telescope_, ConfigType& config)
        : toltec(toltec_), telescope(telescope_) {}

    void init() override {}
    void process(MapType& maps) override {
        logger->info("hist processing");

        for (int i = 0; i < maps.signal.size(); ++i) {
            if constexpr (is_std_vector<decltype(maps.signal[i])>::value) {
                for (int j = 0; j < maps.signal[i].size(); ++j) {

                }
            }
        }
    }
};
