#pragma once

#include <Eigen/Core>

#include <tula/logging.h>

#include <tula/algorithm/mlinterp/mlinterp.hpp>
#include <citlali/core/utils/utils.h>

namespace timestream {

class Despiker {
public:
    // spike sigma, time constant, sample rate
    double sigma, time_constant, fsmp;
    // despike window
    int despike_window;
    bool run_filter = false;

    // the main despiking routine
    template <typename DerivedA, typename DerivedB>
    void despike(Eigen::DenseBase<DerivedA>&, Eigen::DenseBase<DerivedB> &);

};

template <typename DerivedA, typename DerivedB>
void Despiker::despike(Eigen::DenseBase<DerivedA> &scans,
                     Eigen::DenseBase<DerivedB> &flags) {

}

} // namespace timestream
