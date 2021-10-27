#pragma once

#include <Eigen/Core>

namespace engine_utils {

template<typename DerivedA>
auto stddev(Eigen::DenseBase<DerivedA> &vec) {
    double norm;
    if (vec.derived().size() == 1) {
        norm = 1;
    }
    else {
        norm = vec.derived().size() - 1;
    }
    double tmp = std::sqrt(
        (vec.derived().array() - vec.derived().mean()).square().sum()
        / norm);
    return tmp;
}

template <typename DerivedA, typename DerivedB>
auto stddev(Eigen::DenseBase<DerivedA> &scans,
            Eigen::DenseBase<DerivedB> &flags) {

    Eigen::Index ngood = (flags.derived().array() == 1).count();

    double norm;
    if (ngood == 1) {
        norm = 1;
    }
    else {
        norm = scans.derived().size() - 1;
    }

    //Calculate the standard deviation
    double tmp = std::sqrt(((scans.derived().array() *flags.derived().template cast<double>().array()) -
                            (scans.derived().array() * flags.derived().template cast<double>().array()).sum()/
                                ngood).square().sum() / norm);

    // Return stddev and the number of unflagged samples
    return std::tuple<double, double>(tmp, ngood);
}

// box car smooth function
template<typename DerivedA, typename DerivedB>
void smooth(Eigen::DenseBase<DerivedA> &in, Eigen::DenseBase<DerivedB> &out, int w) {
    //Ensure box-car width is odd
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

// iterate over tuple (https://stackoverflow.com/questions/26902633/how-to-iterate-over-a-stdtuple-in-c-11)
template<class F, class...Ts, std::size_t...Is>
void for_each_in_tuple(const std::tuple<Ts...> & tuple, F func, std::index_sequence<Is...>) {
    using expander = int[];
    (void)expander { 0, ((void)func(std::get<Is>(tuple)), 0)... };
}

template<class F, class...Ts>
void for_each_in_tuple(const std::tuple<Ts...> & tuple, F func){
    for_each_in_tuple(tuple, func, std::make_index_sequence<sizeof...(Ts)>());
}

} // namespace
