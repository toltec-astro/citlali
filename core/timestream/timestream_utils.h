#pragma once

namespace timestream {

namespace timestream_utils {

enum PointingType {
    RaDec = 0,
    AzEl = 1
};

// Standard deviation calculator since there isn't one implemented in Eigen
template<typename DerivedA>
auto stddev(Eigen::DenseBase<DerivedA> &scans){
    //calculate the standard deviation
    double norm;
    if (scans.derived().size() == 1) {
        norm = 1;
    } else {
        norm = scans.derived().size() - 1;
    }
    double tmp = std::sqrt(
        (scans.derived().array() - scans.derived().mean()).square().sum()
        / norm);

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

    double norm;
    if (ngood == 1) {
        norm = 1;
    } else {
        norm = scans.derived().size() - 1;
    }

    //Calculate the standard deviation
    double tmp = std::sqrt(((scans.derived().array() *flags.derived().template cast<double>().array()) -
                            (scans.derived().array() * flags.derived().template cast<double>().array()).sum()/
                                ngood).square().sum() / norm);

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

template<PointingType pointingtype, typename DerivedA, typename DerivedB>
void getDetectorPointing(Eigen::DenseBase<DerivedA> &lat,
                         Eigen::DenseBase<DerivedA> &lon,
                         Eigen::DenseBase<DerivedB> &telLat,
                         Eigen::DenseBase<DerivedB> &telLon,
                         Eigen::DenseBase<DerivedB> &TelElDes,
                         Eigen::DenseBase<DerivedB> &ParAng,
                         const double azOffset,
                         const double elOffset,
                         std::shared_ptr<lali::YamlConfig> config)
{
    // RaDec map
    if constexpr (pointingtype == RaDec) {
        auto azOfftmp = cos(TelElDes.derived().array()) * azOffset
                        - sin(TelElDes.derived().array()) * elOffset
                        + config->get_typed<double>("bsOffset_0");
        auto elOfftmp = cos(TelElDes.derived().array()) * elOffset
                        + sin(TelElDes.derived().array()) * azOffset
                        + config->get_typed<double>("bsOffset_1");
        auto pa2 = ParAng.derived().array() - pi;

        auto ratmp = -azOfftmp * cos(pa2) - elOfftmp * sin(pa2);
        auto dectmp = -azOfftmp * sin(pa2) + elOfftmp * cos(pa2);

        lat = ratmp * RAD_ASEC + telLat.derived().array();
        lon = dectmp * RAD_ASEC + telLon.derived().array();
    }
}

} //namespace
} //namespace
