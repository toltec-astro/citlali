#pragma once

#include <citlali/core/utils/utils.h>

// Filter
template <typename TCDataType>
class Filter : public PipelineComponent<TCDataType> {
public:
    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    Instrument& toltec;
    Telescope& telescope;

    int filter_order;
    double gibbs_factor;
    double low_cutoff_Hz;
    double high_cutoff_Hz;
    Eigen::VectorXd filter;

    template <typename ConfigType>
    Filter(Instrument& toltec_ref, Telescope& telescope_ref, ConfigType& config)
        : toltec(toltec_ref), telescope(telescope_ref) {

        config.get(gibbs_factor, std::tuple{"timestream","raw_time_chunk","filter","a_gibbs"});
        config.get(low_cutoff_Hz, std::tuple{"timestream","raw_time_chunk","filter","freq_low_Hz"});
        config.get(high_cutoff_Hz, std::tuple{"timestream","raw_time_chunk","filter","freq_high_Hz"});
        config.get(filter_order, std::tuple{"timestream","raw_time_chunk","filter","n_terms"});
    }

    void init() override {
        // uses toltec.data_fs_hz
        filter = create_kaiser_filter(toltec.data_fs_hz, filter_order, gibbs_factor,
                                        low_cutoff_Hz, high_cutoff_Hz);
    }

    // process method that integrates the filter stage into a data processing pipeline
    void process(TCDataType& tcdata) override {
        logger->info("filter processing");

        // start index of inner scans
        int start = filter_order;
        // end index of inner scans
        int size = tcdata.chunk_indices(1) - tcdata.chunk_indices(0) + 1;

        convolve_filter(tcdata.signal, filter);
        if (tcdata.kernel.size() > 0) {
            convolve_filter(tcdata.kernel, filter);
        }
        tcdata.shrink(start, size);
    }
};
