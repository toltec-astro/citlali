#pragma once

namespace timestream {

namespace timestream_utils {


// Standard deviation calculator since there isn't one implemented in Eigen
template<typename DerivedA>
auto stddev(Eigen::DenseBase<DerivedA> &scans){
    //calculate the standard deviation
    double tmp = std::sqrt(
        (scans.derived().array() - scans.derived().mean()).square().sum()
        / (scans.derived().size() - 1.0));

    // Return stddev
    return tmp;
}

// Standard deviation calculator since there isn't one implemented in Eigen
// This template specialization of stddev takes a flag matrix and only
// calculates the stddev over the unflagged points
template <typename DerivedA, typename DerivedB>
auto stddev(Eigen::DenseBase<DerivedA> &scans,
            Eigen::DenseBase<DerivedB> &flags){

    // Number of unflagged samples
    Eigen::Index ngood = (flags.derived().array() == 1).count();

    //Calculate the standard deviation
    double tmp = std::sqrt(((scans.derived().array() *flags.derived().template cast<double>().array()) -
                            (scans.derived().array() * flags.derived().template cast<double>().array()).sum()/
                                ngood).square().sum() / (ngood - 1.0));

    // Return stddev and the number of unflagged samples
    return std::tuple<double, double>(tmp, ngood);
}

// Box-car smooth function
template<typename DerivedA, typename DerivedB>
void smooth(Eigen::DenseBase<DerivedA> &in, Eigen::DenseBase<DerivedB> &out, int w)
{
    //Ensure box-car width is even
    if (w % 2 == 0) {
        w++;
    }

    Eigen::Index npts = in.size();

    out.head((w - 1) / 2) = in.head((w - 1) / 2);
    out.tail(npts - (w + 1) / 2. + 1) = in.tail(npts - (w + 1) / 2. + 1);

    double winv = 1. / w;
    int wm1d2 = (w - 1) / 2.;
    int wp1d2 = (w + 1) / 2.;

    for (int i = wm1d2; i <= npts - wp1d2; i++) {
        out(i) = winv * in.segment(i - wm1d2, w).sum();
    }
}

} //namespace
}
