# pragma once

// Despike
template <typename TCDataType>
class Despike : public PipelineComponent<TCDataType> {
public:
    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    double min_spike_sigma;
    double time_constant_sec;
    int stats_window_size, window_size;
    bool run_filter;

    Instrument& toltec;
    Telescope& telescope;

    template <typename ConfigType>
    Despike(Instrument& toltec_ref, Telescope& telescope_ref, ConfigType& config)
        : toltec(toltec_ref), telescope(telescope_ref) {

        config.get(min_spike_sigma, std::tuple{"timestream","raw_time_chunk","despike","min_spike_sigma"});
        config.get(window_size, std::tuple{"timestream","raw_time_chunk","despike","time_constant_sec"});
        config.get(stats_window_size, std::tuple{"timestream","raw_time_chunk","despike","time_constant_sec"});
        config.get(run_filter, std::tuple{"timestream","raw_time_chunk","filter","enabled"});

        if (run_filter) {
            // filter order
            int filter_order;
            config.get(filter_order, std::tuple{"timestream","raw_time_chunk","filter","n_terms"});

            // use filter size if larger than despike window
            if (window_size < filter_order) {
                window_size = filter_order;
            }
        }
    }

    void init() {}

    void process(TCDataType& tcdata) override {
        logger->info("despike processing");
        despike(tcdata);
    }

    void despike(TCDataType& tcdata) {
        int n_dets = tcdata.n_dets();

        tcdata.n_spikes.resize(n_dets);

        for (int det = 0; det < n_dets; ++det) {
            if (tcdata.apt_flag(det)) {
                tcdata.n_spikes(det) = 0;
                continue;
            }

            auto data = tcdata.signal.col(det);
            auto flags = tcdata.flag.col(det);

            tcdata.n_spikes(det) = despike_with_moving_window(data, flags);
        }
    }

    template <typename DerivedA, typename DerivedB>
    std::pair<typename DerivedA::Scalar, typename DerivedA::Scalar>
    calculate_window_stats_exclude_flags(const Eigen::DenseBase<DerivedA>& data,
                                         const Eigen::DenseBase<DerivedB>& flags,
                                         int start, int end) {
        using Scalar = typename DerivedA::Scalar;

        // Extract the window from the data and corresponding flags
        auto window = data.derived().segment(start, end - start);
        auto flag_segment = flags.derived().segment(start, end - start);

        // Filter out flagged points
        Eigen::Array<Scalar, Eigen::Dynamic, 1> unflagged_data =
            window.array() * (flag_segment.array() == 0).template cast<Scalar>();

        // Count non-flagged data points
        Scalar count = (flag_segment.array() == 0).template cast<Scalar>().sum();

        // Calculate mean and standard deviation on unflagged data
        Scalar mean = unflagged_data.sum() / count;
        Scalar stddev = std::sqrt((unflagged_data - mean).square().sum() / count);

        return {mean, stddev};
    }

    // despike function with spike count output
    template <typename DerivedA, typename DerivedB>
    int despike_with_moving_window(Eigen::DenseBase<DerivedA>& data, Eigen::DenseBase<DerivedB> &flags_expr) {
        bool spikes_found;
        int spike_count = 0;
        Eigen::Matrix<double, Eigen::Dynamic, 1> flags = flags_expr.derived().template cast<double>();

        do {
            spikes_found = false;

            // iterate through the data
            for (int i = 0; i < data.size(); ++i) {
                // skip calculation if the current point is already flagged as a spike
                if (flags(i) == 1) {
                    continue;
                }

                // define window boundaries
                int start = (i >= stats_window_size / 2) ? i - stats_window_size / 2 : 0;
                int end = std::min(i + stats_window_size / 2 + 1, static_cast<int>(data.size()));

                // calculate local mean and standard deviation excluding flagged points
                auto [mean, stddev] = calculate_window_stats_exclude_flags(data, flags, start, end);

                // flag the spike if the value deviates significantly from the local mean
                if (std::abs(data(i) - mean) > min_spike_sigma * stddev) {
                    flags(i) = 1;  // mark as spike
                    spikes_found = true;
                    ++spike_count;  // increment spike count
                }
            }
        } while (spikes_found);

        // region flagging loop
        for (int i = 0; i < data.size(); ++i) {
            if (flags(i) == 1) {
                // flag the surrounding region based on window_size_2
                int region_start = std::max(0, i - window_size);
                int region_end = std::min(static_cast<int>(data.size()), i + window_size + 1);

                for (int j = region_start; j < region_end; ++j) {
                    flags(j) = 1;
                }
            }
        }

        flags_expr = flags.template cast<bool>();

        // return the total number of spikes found
        return spike_count;
    }
};
