#pragma once

namespace timestream {

class Downsampler {
public:
    int factor;
    double downsampled_freq_Hz;

    template <typename DerivedA, typename DerivedB>
    void downsample(Eigen::DenseBase<DerivedA> &in, Eigen::DenseBase<DerivedB> &out) {
        // define to save space
        using Eigen::Matrix;
        using Eigen::Map;
        using Eigen::Stride;
        using Eigen::InnerStride;
        using Eigen::Dynamic;

        // use eigen stride to skip over points
        using EigenStrideMap = Map<Matrix<typename DerivedA::Scalar, Dynamic, Dynamic>,0, Stride<Dynamic,Dynamic>>;

        // saving space
        auto rows = in.rows();
        auto cols = in.cols();

        // sanity check to avoid invalid stride construction
        if (factor <= 0) {
            throw std::invalid_argument("downsample factor must be > 0");
        }

        // floor to avoid reading past the end when rows % factor != 0
        auto out_rows = rows / factor;

        // stride-map view that picks every `factor`-th sample; truncates tail samples
        out = EigenStrideMap (in.derived().data(), out_rows, cols,
                              Stride<Dynamic, Dynamic>(in.outerStride(), in.innerStride()*factor));
    }
};

} // namespace timestream
