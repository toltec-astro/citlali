# pragma once

// Weights
template <typename TCDataType>
class Weights : public PipelineComponent<TCDataType> {
public:
    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    std::string weight_type;
    double median_weight_factor;

    Instrument& toltec;
    Telescope& telescope;

    template <typename ConfigType>
    Weights(Instrument& toltec_ref, Telescope& telescope_ref, ConfigType& config)
        : toltec(toltec_ref), telescope(telescope_ref) {

        config.get(weight_type, std::tuple{"timestream","processed_time_chunk","weighting","type"});
        config.get(median_weight_factor, std::tuple{"timestream","processed_time_chunk","weighting","median_weight_factor"});
    }

    void init() {}
    void process(TCDataType& tcdata) override {
        logger->info("weight processing");
        calc_weights(tcdata);

        if (median_weight_factor >= 1 && weight_type != "constant") {
            reset_weights_to_median(tcdata.weight, tcdata.flag, tcdata.apt_flag);

            if (tcdata.weight_q) {
                reset_weights_to_median(tcdata.weight_q.value(), tcdata.flag, tcdata.apt_flag);
            }

            if (tcdata.weight_u) {
                reset_weights_to_median(tcdata.weight_u.value(), tcdata.flag, tcdata.apt_flag);
            }
        }
    }

    void calc_weights(TCDataType&);

    template <typename DerivedA, typename DerivedB, typename DerivedC>
    void reset_weights_to_median(Eigen::DenseBase<DerivedA>&, Eigen::DenseBase<DerivedB>&, Eigen::DenseBase<DerivedC>&);
};

template <typename TCDataType>
void Weights<TCDataType>::calc_weights(TCDataType& tcdata) {
    int n_dets = tcdata.n_dets();
    tcdata.weight.resize(n_dets);

    // initialize q and u weight timestreams
    if (tcdata.signal_q.has_value()) {
        tcdata.weight_q.emplace(n_dets);
    }
    if (tcdata.signal_u.has_value()) {
        tcdata.weight_u.emplace(n_dets);
    }

    if (weight_type == "full") {
        // loop over detectors
        for (int det = 0; det < n_dets; ++det) {
            if (!tcdata.apt_flag(det)) {
                double variance = flagged_variance(tcdata.signal.col(det), tcdata.flag.col(det));
                tcdata.weight(det) = variance > 0 ? 1 / variance : 0;

                if (tcdata.weight_q.has_value()) {
                    double q_variance = flagged_variance(tcdata.signal_q.value().col(det), tcdata.flag.col(det));
                    tcdata.weight_q.value()(det) = q_variance > 0 ? 1 / q_variance : 0;
                }
                if (tcdata.weight_u.has_value()) {
                    double u_variance = flagged_variance(tcdata.signal_u.value().col(det), tcdata.flag.col(det));
                    tcdata.weight_u.value()(det) = u_variance > 0 ? 1 / u_variance : 0;
                }
            }
            else {
                tcdata.weight(det) = 0;
                if (tcdata.weight_q.has_value()) {
                    tcdata.weight_q.value()(det) = 0;
                }
                if (tcdata.weight_u.has_value()) {
                    tcdata.weight_u.value()(det) = 0;
                }
            }
        }
    } else if (weight_type == "approximate") {
        // loop over detectors
        for (int det = 0; det < n_dets; ++det) {
            if (!tcdata.apt_flag(det)) {
                // calculate weight based on sensitivity while applying fcf for current chunk
                tcdata.weight(det) = pow(sqrt(tcdata.data_fs_hz) * toltec.apt["sens"].data(det) * tcdata.fcf(det), -2.0);

                if (tcdata.weight_q.has_value()) {
                    tcdata.weight_q.value()(det) = tcdata.weight(det);
                }
                if (tcdata.weight_u.has_value()) {
                    tcdata.weight_u.value()(det) = tcdata.weight(det);
                }
            }
            else {
                tcdata.weight(det) = 0;
                if (tcdata.weight_q.has_value()) {
                    tcdata.weight_q.value()(det) = 0;
                }
                if (tcdata.weight_u.has_value()) {
                    tcdata.weight_u.value()(det) = 0;
                }
            }
        }
    } else if (weight_type == "constant") {
        tcdata.weight = (toltec.apt["flag"].data.array() == 1).select(Eigen::ArrayXd::Zero(tcdata.n_dets()),
                                                                      Eigen::ArrayXd::Ones(tcdata.n_dets()));
        if (tcdata.weight_q.has_value()) {
            tcdata.weight_q.value() = (toltec.apt["flag"].data.array() == 1).select(Eigen::ArrayXd::Zero(tcdata.n_dets()),
                                                                                    Eigen::ArrayXd::Ones(tcdata.n_dets()));
        }
        if (tcdata.weight_u.has_value()) {
            tcdata.weight_u.value() = (toltec.apt["flag"].data.array() == 1).select(Eigen::ArrayXd::Zero(tcdata.n_dets()),
                                                                                    Eigen::ArrayXd::Ones(tcdata.n_dets()));
        }
    }
}

template <typename TCDataType>
template <typename DerivedA, typename DerivedB, typename DerivedC>
void Weights<TCDataType>::reset_weights_to_median(Eigen::DenseBase<DerivedA> &weight,
                                                  Eigen::DenseBase<DerivedB> &flag,
                                                  Eigen::DenseBase<DerivedC> &apt_flag) {

    // loop over arrays
    int i = 0;
    for (const auto& [start, end]: toltec.apt.array_indices) {
        int n_good = 0;
        int n_outliers = 0;

        // find weights of good detectors
        for (int det = start; det <= end ; ++det) {
            if (!apt_flag(det) && (flag.col(det).array() == false).any()) {
                n_good++;
            }
        }

        Eigen::VectorXd weights(n_good);

        // find weights of good detectors
        int j = 0;
        for (int det = start; det <= end; ++det) {
            if (!apt_flag(det) && (flag.col(det).array() == false).any()) {
                weights(j++) = weight(det);
            }
        }

        // median weight
        double median_weight = tula::alg::median(weights);

        // remove outliers
        for (int det = start; det <= end; ++det) {
            if (weight(det) > median_weight_factor * median_weight) {
                weight(det) = median_weight;
                n_outliers++;
            }
        }
        logger->info("array {} had {} outlier weights", toltec.apt.arrays(i++), n_outliers);
    }
}
