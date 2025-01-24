#pragma once

// MapOutput
template <typename MapType>
class MapOutput : public PipelineComponent<MapType> {
public:
    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    Instrument& toltec;
    Telescope& telescope;

    template <typename ConfigType>
    MapOutput(Instrument& toltec_ref, Telescope& telescope_ref, ConfigType& config)
        : toltec(toltec_ref), telescope(telescope_ref) {}

    void init() {}
    void process(MapType& maps) override {
        logger->info("map output processing");
    }
};
