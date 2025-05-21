# pragma once

// Normalize
template <typename MapType>
class Normalize : public PipelineComponent<MapType> {
public:
    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    Instrument& toltec;
    Telescope& telescope;

    template <typename ConfigType>
    Normalize(Instrument& toltec_, Telescope& telescope_, const ConfigType& config)
        : toltec(toltec_), telescope(telescope_) {}

    void init() {}
    void process(MapType& maps) override {
        logger->info("map normalize processing");

        bool normalize_kernel = !maps.kernel.empty();

        // loop through maps and normalize
        for (int i = 0; i < maps.signal.size(); ++i) {
            maps.signal[i].data =
                (maps.weight[i].data.array() > 0).select(maps.signal[i].data.array() / maps.weight[i].data.array(), 0).matrix();
        }

        if (normalize_kernel) {
            // loop through maps and normalize
            for (const auto& [key, i] : maps.kernel_lookup) {
                const auto &weight = maps.weight[maps.weight_lookup.at(key)].data.array();
                const auto &kernel = maps.kernel[i].data.array();

                maps.kernel[i].data = (weight > 0)
                                          .select(kernel / weight, 0)
                                          .matrix();
            }
        }
    }
};
