# pragma once

// TodFlagging
template <typename TCDataType>
class TodFlagging : public PipelineComponent<TCDataType> {
public:
    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    Instrument& toltec;
    Telescope& telescope;
    double lower_weight_factor, upper_weight_factor;
    int max_iters;

    std::string type_;

    template <typename ConfigType>
    TodFlagging(std::string type, Instrument& toltec_ref, Telescope& telescope_ref, ConfigType& config)
        : type_(type), toltec(toltec_ref), telescope(telescope_ref) {

        config.get(lower_weight_factor, std::tuple{"timestream",type_,"flagging","lower_weight_factor"});
        config.get(upper_weight_factor, std::tuple{"timestream",type_,"flagging","upper_weight_factor"});
        config.get(max_iters, std::tuple{"timestream",type_,"flagging","max_iters"});
    }

    void init() {}
    void process(TCDataType& tcdata) override {
        logger->info("tod flagging processing");

        // only run if one factor is greater than zero
        if (lower_weight_factor > 0 || upper_weight_factor > 0) {
            flag_by_variance(tcdata);
        }
    }

    void flag_by_variance(TCDataType&);
};

template <typename TCDataType>
void TodFlagging<TCDataType>::flag_by_variance(TCDataType& tcdata) {
    int n_iter = 0;
    bool keep_going = true;

    while (n_iter < max_iters && keep_going) {
        int n_dets_iter = 0;

        int i = 0;
        for (const auto& [start, end]: toltec.apt.array_indices) {
            int n_good = 0;
            // find number of good detectors
            for (int det = start; det <= end; ++det) {
                if (!tcdata.apt_flag(det) && (tcdata.flag.col(det).array() == false).any()) {
                    n_good++;
                }
            }

            Eigen::VectorXd weights(n_good);

            // find weights (1/variance) of good detectors
            int j = 0;
            for (int det = start; det <= end; ++det) {
                if (!tcdata.apt_flag(det) && (tcdata.flag.col(det).array() == false).any()) {
                    double variance = flagged_variance(tcdata.signal.col(det), tcdata.flag.col(det));
                    weights(j++) = variance > 0 ? 1. / variance : 0;
                }
            }

            // median weight of good detectors
            double median_weights = tula::alg::median(weights);

            double lower_bound = lower_weight_factor * median_weights;
            double upper_bound = upper_weight_factor * median_weights;

            int n_dets_low = 0;
            int n_dets_high = 0;

            // loop through detectors and flag outliers
            j = 0;
            for (int det = start; det <= end; ++det) {
                if (!tcdata.apt_flag(det) && (tcdata.flag.col(det).array() == false).any()) {
                    bool low_weight = weights(j) < lower_bound;
                    bool high_weight  = (weights(j) > upper_bound) && (upper_weight_factor > 0) ;
                    if (low_weight) {
                        tcdata.apt_flag(det) = 1;
                        n_dets_low++;
                    }
                    if (high_weight) {
                        tcdata.apt_flag(det) = 1;
                        n_dets_high++;
                    }
                    j++;
                }
            }

            logger->info("array {} iter {}: {}/{} low weights | {}/{} high weights.", toltec.apt.arrays(i), n_iter,
                         n_dets_low, n_good, n_dets_high, n_good);

            n_dets_iter += (n_dets_low + n_dets_high);
            i++;
        }

        n_iter++;
        if (n_dets_iter == 0) {
            keep_going = false;
        }
    }
}
