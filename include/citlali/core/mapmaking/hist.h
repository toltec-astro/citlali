#pragma once

// MapHist
template <typename MapType>
class MapHist : public PipelineComponent<MapType> {
public:
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    Instrument& toltec;
    Telescope& telescope;

    template <typename ConfigType>
    MapHist(Instrument& toltec_ref, Telescope& telescope_ref, ConfigType& config)
        : toltec(toltec_ref), telescope(telescope_ref) {}

    void init() {}
    void process(MapType& maps) override {}
};
