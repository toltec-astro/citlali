#pragma once

// Coadd
template <typename MapType>
class Coadd : public PipelineComponent<MapType> {
public:
    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    template <typename ConfigType>
    Coadd(MapContainer& coadd_maps_ref, Instrument& toltec_ref, Telescope& telescope_ref)
        : coadd_maps(coadd_maps_ref), toltec(toltec_ref), telescope(telescope_ref) {}

    void init() {}
    void process(MapType& maps) override {
        logger->info("coadd processing");

        const bool coadd_kernel = !maps.kernel.empty();
        const bool coadd_coverage = !maps.coverage.empty();

        const int n_rows = maps.n_rows;
        const int n_cols = maps.n_cols;

        const int start_row = (coadd_maps.n_rows - n_rows) / 2;
        const int start_col = (coadd_maps.n_cols - n_cols) / 2;

        // loop through maps and fit
        for (int i = 0; i < maps.signal.size(); ++i) {
            coadd_maps.weight[i].get()
                .block(start_row, start_col, n_rows, n_cols)
                .noalias() += maps.weight[i].get();

            coadd_maps.signal[i].get()
                .block(start_row, start_col, n_rows, n_cols)
                .noalias() += (maps.signal[i].get().array() * maps.weight[i].get().array()).matrix();
        }

        if (coadd_kernel) {
            for (int i = 0; i < maps.kernel.size(); ++i) {
                coadd_maps.kernel[i].get()
                    .block(start_row, start_col, n_rows, n_cols)
                    .noalias() += (maps.kernel[i].get().array() * maps.weight[i].get().array()).matrix();
            }
        }

        if (coadd_coverage) {
            for (int i = 0; i < maps.coverage.size(); ++i) {
                coadd_maps.coverage[i].get()
                    .block(start_row, start_col, n_rows, n_cols)
                    .noalias() += maps.coverage[i].get();
            }
        }
    }

private:
    Instrument& toltec;
    Telescope& telescope;
    MapContainer& coadd_maps;
};
