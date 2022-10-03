#pragma once

namespace timestream {

class Filter {
public:
    double a_gibbs, freq_low_Hz, freq_high_Hz;
    Eigen::Index n_terms;
};

} // namespace timestream
