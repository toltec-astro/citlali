# pragma once

// Sensitivity
/*template <typename TCDataType, typename SensVectorType>
class Sensitivity : public PipelineComponent<std::vector<TCDataType>, SensVectorType> {
public:
    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    // Constructor takes the relevant YAML node for configuration
    template <typename ConfigType, typename KeyType>
    Sensitivity(Instrument& toltec_ref, Telescope& telescope_ref, const ConfigType& config, KeyType& missing_keys, KeyType& invalid_keys)
        : toltec(toltec_ref), telescope(telescope_ref) {

        // frequency range to calculate tod psd over
        //get_config_value(config, bmp_sens_calc_limits_hz, missing_keys, invalid_keys,
        //                 std::tuple{"beammap","flagging","sens_psd_limits_Hz"});
    }

    void init() {}
    void process(std::vector<TCDataType>& tc_vector, SensVectorType& sensitivity) override {
        logger->info("sensitivity processing");
        sensitivity = calc_sens(tc_vector);
    }

private:
    Instrument& toltec;
    Telescope& telescope;

    std::vector<double> freq_range_hz;
    void get_common_grid(std::vector<TCDataType>&);
    void calc_sens(std::vector<TCDataType>&);
};

auto Sensitivity::get_common_grid(const std::vector<TCDataType>& tc_vector) {
    Eigen::Index n_chunks = tc_vector.size();
    Eigen::VectorXI chunk_lengths(n_chunks);

    // get chunk lengths (same for all detectors)
    for (Eigen::Index i = 0; i < n_chunks; ++i) {
        chunk_lengths(i) = tc_vector[i].n_pts();
    }

    // use the median length for computation
    Eigen::Index median_chunk_length = tula::alg::median(chunk_lengths);

    // ensure length is even
    if (median_chunk_length % 2 == 1) {
        median_chunk_length--;
    }

    // size of common psd
    Eigen::Index n_freqs = median_chunk_length / 2 + 1;
    double d_freq = fs_hz / median_chunk_length;

    // common frequencies
    Eigen::VectorXd freqs = Eigen::VectorXd::LinSpaced(n_freqs, 0, n_freqs - 1) * d_freq;

    return std::make_tuple(freqs, n_freqs, n_chunks);

}

auto Sensitivity::calc_sensitivity(std::vector<TCDataType>& tc_vector) {
    // get common grid parameters
    auto [freqs, n_freqs, n_chunks] = get_common_grid(tc_vector);
    double d_freq = freqs(1) - freqs(0); // freq resolution

    // vector to hold sensitivities for all detectors
    Eigen::VectorXd sensitivity(tc_vector[0].n_dets());

    for (int det = 0; det < tc_vector[0].n_dets(); ++det) {
        // store calculated psds
        Eigen::MatrixXd chunk_psds(n_freqs, n_chunks);

        // calculate detector psd for each chunk
        for (int i = 0; i < n_chunks; ++i) {
            auto [det_psd, det_freqs] = calc_psd_1d(tc_vector[i].signal.col(det), fs_hz);

            // perform interpolation
            Eigen::Matrix<Eigen::Index,1,1> nd;
            nd << det_freqs.size();

            mlinterp::interp(nd.data(), n_freqs, // nd, ni
                             det_psd.data(), chunk_psds.data() + i * n_freqs, // yd, yi
                             det_freqs.data(), freqs.data()); // xd, xi
        }

        // get sensitivity in V * s^(1/2)
        auto det_sens = (chunk_psds / 2).cwiseSqrt();

        // compute sensitivity with given freqrange
        Eigen::Index start = static_cast<Eigen::Index>(std::max(0.0, freq_range_hz[0] / d_freq));
        Eigen::Index end = static_cast<Eigen::Index>(std::min(static_cast<double>(n_freqs - 1), freq_range_hz[1] / d_freq));

        auto n_pts = end - start + 1;
        sensitivity(det) = det_sens.block(start, 0, n_pts, n_chunks).colwise().sum().mean() / n_pts;
    }

    return sensitivity;
}*/
