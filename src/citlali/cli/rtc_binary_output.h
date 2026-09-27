// Private adapter serialization. Preserve row-major binary64/native integer bytes.
#pragma once
#include <Eigen/Core>
#include <filesystem>
#include <fstream>
#include <span>
#include <stdexcept>
#include <vector>
namespace citlali::cli::detail {
template<class T> void write_native_binary(const std::filesystem::path &path,std::span<const T> values) {
    std::ofstream out(path,std::ios::binary);
    out.write(reinterpret_cast<const char*>(values.data()),static_cast<std::streamsize>(values.size_bytes()));
    out.close();if(!out)throw std::runtime_error("required binary output failed: "+path.string());
}
template<class Derived> void write_row_major_binary(const std::filesystem::path &path,const Eigen::MatrixBase<Derived> &matrix) {
    if constexpr(Derived::IsRowMajor) {
        if(matrix.innerStride()==1 && matrix.outerStride()==matrix.cols()) {
            write_native_binary<double>(path,{matrix.derived().data(),static_cast<std::size_t>(matrix.size())});return;
        }
    }
    // Column-major bases/vectors and strided views keep the original i,j order.
    std::vector<double> packed;packed.reserve(matrix.size());
    for(Eigen::Index i=0;i<matrix.rows();++i)for(Eigen::Index j=0;j<matrix.cols();++j)packed.push_back(matrix(i,j));
    write_native_binary<double>(path,packed);
}
}
