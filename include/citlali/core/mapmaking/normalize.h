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
    Normalize(Instrument& toltec_ref, Telescope& telescope_ref, const ConfigType& config)
        : toltec(toltec_ref), telescope(telescope_ref) {}

    void init() {}
    void process(MapType& maps) override {
        logger->info("map normalize processing");

        bool normalize_kernel = !maps.kernel_map.empty();

        // loop through maps and normalize
        for (int i = 0; i < maps.signal_map.size(); ++i) {
            maps.signal_map[i] =
                (maps.weight_map[i].array() > 0).select(maps.signal_map[i].array() / maps.weight_map[i].array(), 0).matrix();
        }

        if (normalize_kernel) {
            for (auto& [key, lower_keys] : maps.maps) {
                for (auto& [lower_key, lower_key_map] : lower_keys) {
                    lower_key_map.kernel.i =
                        (lower_key_map.weight.i.array() > 0).select(lower_key_map.kernel.i.array() / lower_key_map.weight.i.array(), 0).matrix();
                }
            }
        }
    }
};
