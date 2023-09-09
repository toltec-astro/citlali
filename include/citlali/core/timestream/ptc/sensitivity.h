#pragma once

#include <tuple>

#include <Eigen/Core>
#include <unsupported/Eigen/FFT>

#include <tula/algorithm/mlinterp/mlinterp.hpp>
#include <tula/algorithm/ei_stats.h>

#include <citlali/core/utils/constants.h>

namespace internal {

template <typename Derived>
void smooth_edge_truncate(Eigen::DenseBase<Derived> &in, Eigen::DenseBase<Derived> &out, int w) {
    Eigen::Index n_pts = in.size();

    // ensure w is odd
    if (w % 2 == 0) {
        w++;
    }

    double w_inv = 1./w;
    int w_mid = (w - 1)/2;

    double sum;
    for (Eigen::Index i=0; i<n_pts; i++) {
        sum=0;
        for (int j=0; j<w; j++) {
            int add_index = i + j - w_mid;

            if (add_index < 0) {
                add_index=0;
            }
            else if (add_index > n_pts-1) {
                add_index = n_pts-1;
            }
            sum += in(add_index);
        }
        out(i) = w_inv*sum;
    }
}

template <typename T = Eigen::ArrayXd> T *ei_nullptr() {
    return static_cast<T *>(nullptr);
}

template <typename Derived>
using const_ref = typename Eigen::internal::ref_selector<Derived>::type;

// non-const return type of derived()
template <typename Derived>
using ref = typename Eigen::internal::ref_selector<Derived>::non_const_type;

// true if typename Derived manages its own data, e.g. MatrixXd, etc.
// false if typename Derived is an expression.
template <typename Derived>
struct has_storage
    : std::is_base_of<Eigen::PlainObjectBase<std::decay_t<Derived>>,
                      std::decay_t<Derived>> {};

template <typename UnaryOp> struct MoveEnabledUnaryOp {
    MoveEnabledUnaryOp(const UnaryOp &func = UnaryOp()) : m_func(func) {}

    template <typename T, typename... Args>
    decltype(auto) operator()(T &&in, Args &&... args) {
        if constexpr (std::is_lvalue_reference<T>::value ||
                      !internal::has_storage<T>::value) {
            // lvalue ref, either expression or non-expression
            // return expression that builds on top of input
            return m_func(std::forward<T>(in),
                          std::forward<decltype(args)>(args)...);
        }
        else {
            // rvalue ref
            // in this case we need to call the function and update inplace
            // first and move to return
            in = m_func(std::forward<T>(in),
                        std::forward<decltype(args)>(args)...);
            return std::forward<T>(in);
        }
    }

    protected:
        const UnaryOp &m_func;
};


template <typename _Scalar>
class Interval : public Eigen::Array<_Scalar, 2, 1> {
  public:
    using Scalar = _Scalar;
    using Base = Eigen::Array<_Scalar, 2, 1>;
    inline static const Scalar inf = std::numeric_limits<Scalar>::infinity();

    Interval() : Base() {
        this->operator[](0) = -inf;
        this->operator[](1) = inf;
    }
    Interval(std::initializer_list<Scalar> interval) : Interval() {
        if (interval.size() == 0)
            return;
        if (interval.size() == 2) {
            auto it = interval.begin();
            this->operator[](0) = *it;
            ++it;
            this->operator[](1) = *it;
            return;
        }
        throw std::invalid_argument(
            "empty initalize_list or {left, right} is required");
    }

    template <typename OtherDerived>
    Interval(const Eigen::ArrayBase<OtherDerived> &other) : Base(other) {}

    template <typename OtherDerived>
    Interval &operator=(const Eigen::ArrayBase<OtherDerived> &other) {
        this->Base::operator=(other);
        return *this;
    }
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE Scalar &left() { return this->x(); }
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE Scalar &right() { return this->y(); }
};


// npts   nfreqs, dfreq
using FreqStat = std::tuple<Eigen::Index, Eigen::Index, double>;

FreqStat stat(Eigen::Index scanlength, double fsmp);
Eigen::VectorXd freq(Eigen::Index npts, Eigen::Index nfreqs, double dfreq);

enum Window {
    NoWindow = 0,
    Hanning = 1
};

inline decltype(auto) hann(Eigen::Index npts) {
    // hann window
    // N numbers starting from 0 to (include) 2pi/N * (N-1)
    // 0.5 - 0.5 cos([0, 2pi/N,  2pi/N * 2, ... 2pi/N * (N - 1)])
    // NENBW = 1.5 * df therefore we devide by 1.5 here to get the
    // equivalent scale as if no window is used
    return (0.5 - 0.5 * Eigen::ArrayXd::LinSpaced(npts, 0, 2.0 * pi / npts * (npts - 1)).cos()).matrix() / 1.5;
}

// psd of individual scan npts/2 + 1 values with frequency [0, f/2];
template <Window win=Hanning, typename DerivedA, typename DerivedB, typename DerivedC>
FreqStat psd(const Eigen::DenseBase<DerivedA> &_scan,
             Eigen::DenseBase<DerivedB> &psd, Eigen::DenseBase<DerivedC> *freqs,
             double fsmp) {
    // decltype(auto) scan = _scan.derived();
    typename internal::const_ref<DerivedA> scan(_scan.derived());

    auto stat = internal::stat(scan.size(), fsmp);
    auto [npts, nfreqs, dfreq] = stat;

    // prepare fft
    Eigen::FFT<double> fft;
    fft.SetFlag(Eigen::FFT<double>::HalfSpectrum);
    fft.SetFlag(Eigen::FFT<double>::Unscaled);
    Eigen::VectorXcd freqdata;

    // branch according to whether applying hann
    if constexpr (win == Hanning) {
        //SPDLOG_INFO("apply hann window");
        Eigen::VectorXd scns = scan.block(0,0,npts,1);
        fft.fwd(freqdata, scns.cwiseProduct(
                   internal::hann(npts)).eval());
    }

    else if (win == NoWindow) {
        fft.fwd(freqdata, scan.head(npts));
    }

    // calcualte psd and normalize to frequency resolution
    psd = freqdata.cwiseAbs2() / dfreq / npts/ fsmp;  // V/Hz^0.5
    // accound for the negative frequencies by an extra factor of 2. note: first
    // and last are 0 and nquist freq, so they only appear once
    psd.segment(1, nfreqs - 2) *= 2.;

    Eigen::VectorXd smoothed_psd(psd.size());
    internal::smooth_edge_truncate(psd, smoothed_psd, 10);
    psd = std::move(smoothed_psd);

    // make the freqency array when requested
    if (freqs) {
        freqs->operator=(internal::freq(npts, nfreqs, dfreq));
    }
    return stat;
}

// a set of psd for multiple scans with potentialy different length.
// all individual psds are interpolated to a common frequency array
template <Window win=Hanning, typename DerivedA, typename DerivedB, typename DerivedC>
FreqStat psds(const std::vector<DerivedA> &ptcs,
              Eigen::DenseBase<DerivedB> &_psds,
              Eigen::DenseBase<DerivedC> *freqs, double fsmp, Eigen::Index det) {

    typename internal::ref<DerivedB> psds(_psds.derived());

    // prepare common freq grid
    Eigen::Index nscans = ptcs.size();
    Eigen::Matrix<Eigen::Index,Eigen::Dynamic,1> scanlengths;

    scanlengths.resize(nscans);
    for(Eigen::Index j=0; j<nscans; j++) {
        scanlengths(j) = ptcs[j].scans.data.rows();
    }

    // use the median length for computation
    Eigen::Index len = tula::alg::median(scanlengths);

    // get the common freq stat and freq array
    auto stat = internal::stat(len, fsmp);
    auto [npts, nfreqs, dfreq] = stat;
    Eigen::VectorXd _freqs = internal::freq(npts, nfreqs, dfreq);

    // compute psd for each scan and interpolate onto the common freq grid
    psds.resize(nfreqs, nscans);

    // make some temporaries
    Eigen::VectorXd tfreqs, tpsd;
    // need this to store array size for interp
    Eigen::Matrix<Eigen::Index, Eigen::Dynamic,1> td(1);

    // get the psds
    for (Eigen::Index i=0; i<nscans; ++i) {
        internal::psd<win>(ptcs[i].scans.data.block(0,det,scanlengths(i),1), tpsd,
                      &tfreqs, fsmp);
        // interpolate (tfreqs, tpsd) on to _freqs
        td << tfreqs.size();

        // interp (tfreq, tpsd) onto freq and store the result in column i of psds
        mlinterp::interp(td.data(), nfreqs,
                         tpsd.data(), psds.data() + i * nfreqs,
                         tfreqs.data(),_freqs.data());
    }

    // update the freqs array if requested
    if (freqs) {
        freqs->operator=(_freqs);
    }
    return stat;
}

// calc sens from psd
inline auto psd2sen = internal::MoveEnabledUnaryOp([](auto&& psd) {
    return (psd / 2.).cwiseSqrt();  // V * s^(1/2)
});


} // namespace internal

template <typename tc_t, typename DerivedA, typename DerivedB>
void calc_sensitivity(
    std::vector<tc_t> &ptcs,
    Eigen::DenseBase<DerivedA> &sensitivities, // V * s^(1/2)
    Eigen::DenseBase<DerivedB> &noisefluxes,   // V, = sqrt(\int PSD df)
    double fsmp, Eigen::Index det, internal::Interval<double> freqrange = {3., 5.}) {

    // get psds
    Eigen::MatrixXd tpsds;
    auto [npts, nfreqs, dfreq] = internal::psds<internal::Hanning>(
        ptcs, tpsds, internal::ei_nullptr(), fsmp, det);

    // compute noises by integrate over all frequencies
    noisefluxes = (tpsds * dfreq).colwise().sum().cwiseSqrt();
    auto meannoise = noisefluxes.mean();

    // get sensitivity in V * s^(1/2)
    Eigen::MatrixXd sens = internal::psd2sen(std::move(tpsds));

    // compute sensitivity with given freqrange
    // make use the fact that freqs = i * df to find Eigen::Index i
    auto i0 = static_cast<Eigen::Index>(freqrange.left() / dfreq);
    auto i1 = static_cast<Eigen::Index>(freqrange.right() / dfreq);

    auto nf = i1 + 1 - i0;
    sensitivities =
        sens.block(i0, 0, nf, ptcs.size()).colwise().sum() / nf;
}
