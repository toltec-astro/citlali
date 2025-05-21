#pragma once

// Coadd
template <typename MapType>
class Coadd : public PipelineComponent<MapType> {
public:
    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    Instrument& toltec;
    Telescope& telescope;
    ObsMaps<>& coadd_maps;

    template <typename ConfigType>
    Coadd(ObsMaps<>& coadd_maps_, Instrument& toltec_, Telescope& telescope_, ConfigType& config)
        : coadd_maps(coadd_maps_), toltec(toltec_), telescope(telescope_) {}

    void init() override {}
    void process(MapType& maps) override {
        logger->info("coadd processing");

        const bool coadd_kernel = !maps.kernel.empty();
        const bool coadd_coverage = !maps.coverage.empty();

        const int n_rows = maps.wcs.naxis[1];
        const int n_cols = maps.wcs.naxis[0];

        const int start_row = (coadd_maps.wcs.naxis[1] - n_rows) / 2;
        const int start_col = (coadd_maps.wcs.naxis[0] - n_cols) / 2;

        // loop through maps and fit
        for (int i = 0; i < maps.signal.size(); ++i) {
            coadd_maps.weight[i].data
                .block(start_row, start_col, n_rows, n_cols)
                .noalias() += maps.weight[i].data;

            coadd_maps.signal[i].data
                .block(start_row, start_col, n_rows, n_cols)
                .noalias() += (maps.signal[i].data.array() * maps.weight[i].data.array()).matrix();
        }

        if (coadd_kernel) {
            for (const auto& [key, i] : maps.kernel_lookup) {
                const auto &weight = maps.weight[maps.weight_lookup.at(key)].data.array();
                coadd_maps.kernel[i].data
                    .block(start_row, start_col, n_rows, n_cols)
                    .noalias() += (maps.kernel[i].data.array() * weight).matrix();
            }
        }

        if (coadd_coverage) {
            for (int i = 0; i < maps.coverage.size(); ++i) {
                coadd_maps.coverage[i].data
                    .block(start_row, start_col, n_rows, n_cols)
                    .noalias() += maps.coverage[i].data;
            }
        }
    }
};
