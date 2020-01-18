#pragma once
#include <unsupported/Eigen/FFT>

// Arcseconds in 360 degrees
#define ASEC_CIRC 1296000.0
// rad per arcsecond
#define RAD_ASEC (2.0*pi / ASEC_CIRC)

namespace mapmaking {

/**
 * @brief Eigen only supports 1D FFTs.
 * This function runs a 2D FFT.
 */
template<typename DerivedA>
Eigen::MatrixXcd fft2(Eigen::DenseBase<DerivedA> &matIn){
    const int nRows = matIn.rows();
    const int nCols = matIn.cols();

    Eigen::FFT<double> fft;
    Eigen::MatrixXcd matOut(nRows, nCols);

    for (int k = 0; k < nRows; ++k) {
        Eigen::VectorXcd tmpOut(nCols);
        fft.fwd(tmpOut, matIn.row(k));
        matOut.row(k) = tmpOut;
    }

    for (int k = 0; k < nCols; ++k) {
        Eigen::VectorXcd tmpOut(nRows);
        fft.fwd(tmpOut, matOut.col(k));
        matOut.col(k) = tmpOut;
    }

    return matOut;
}


/**
 * @brief Converts pointing lat/lon
 * into row/col indices.
 */
template <typename DerivedA>
void latlonPhysToIndex(double &lat, double &lon, Eigen::Index &irow, Eigen::Index &icol,
                       DerivedA &mapstruct){
    double ps = mapstruct.pixelsize*RAD_ASEC;
    irow = lat / ps + (mapstruct.nrows + 1.) / 2.;
    icol = lon / ps + (mapstruct.ncols + 1.) / 2.;
}

/**
 * @brief generate the pointing data for this detector this version
 * calculates it from scratch and requires a telescope object
 */
template <typename DerivedA, typename DerivedB, typename DerivedC>
void getPointing(DerivedA &telescope_data, Eigen::DenseBase<DerivedB> &lat, Eigen::DenseBase<DerivedB> &lon,
                 const Eigen::DenseBase<DerivedC> &offsets, const Eigen::Index &det, Eigen::Index si = -99, Eigen::Index ei = -99, int dsf = 1){

    Eigen::Index azelMap = 1;
    double azOffset = offsets(0,det);
    double elOffset = offsets(1,det);

    Eigen::Index npts;

    if (si !=-99){
        npts = ei - si;
    }

    else{
        npts = telescope_data["TelAzPhys"].rows();
        si = 0;
    }

    Eigen::Map<Eigen::VectorXd,0,Eigen::InnerStride<Eigen::Dynamic>> TelAzPhys(telescope_data["TelAzPhys"].segment(si,npts).data(),(npts+(dsf - 1))/dsf,Eigen::InnerStride<Eigen::Dynamic>(telescope_data["TelAzPhys"].segment(si,npts).innerStride()*dsf));
    Eigen::Map<Eigen::VectorXd,0,Eigen::InnerStride<Eigen::Dynamic>> TelElPhys(telescope_data["TelElPhys"].segment(si,npts).data(),(npts+(dsf - 1))/dsf,Eigen::InnerStride<Eigen::Dynamic>(telescope_data["TelElPhys"].segment(si,npts).innerStride()*dsf));
    Eigen::Map<Eigen::VectorXd,0,Eigen::InnerStride<Eigen::Dynamic>> TelElDes(telescope_data["TelElDes"].segment(si,npts).data(),(npts+(dsf - 1))/dsf,Eigen::InnerStride<Eigen::Dynamic>(telescope_data["TelElDes"].segment(si,npts).innerStride()*dsf));
    Eigen::Map<Eigen::VectorXd,0,Eigen::InnerStride<Eigen::Dynamic>> ParAng(telescope_data["ParAng"].segment(si,npts).data(),(npts+(dsf - 1))/dsf,Eigen::InnerStride<Eigen::Dynamic>(telescope_data["ParAng"].segment(si,npts).innerStride()*dsf));
    Eigen::Map<Eigen::VectorXd,0,Eigen::InnerStride<Eigen::Dynamic>> TelRaPhys(telescope_data["TelRaPhys"].segment(si,npts).data(),(npts+(dsf - 1))/dsf,Eigen::InnerStride<Eigen::Dynamic>(telescope_data["TelRaPhys"].segment(si,npts).innerStride()*dsf));
    Eigen::Map<Eigen::VectorXd,0,Eigen::InnerStride<Eigen::Dynamic>> TelDecPhys(telescope_data["TelDecPhys"].segment(si,npts).data(),(npts+(dsf - 1))/dsf,Eigen::InnerStride<Eigen::Dynamic>(telescope_data["TelDecPhys"].segment(si,npts).innerStride()*dsf));

    if(!azelMap){
        auto azOfftmp = cos(TelElDes.array())*azOffset - sin(TelElDes.array())*elOffset;
        auto elOfftmp = cos(TelElDes.array())*elOffset + sin(TelElDes.array())*azOffset;
        auto pa2 = ParAng.array()-pi;

        auto ratmp = -azOfftmp*cos(pa2) - elOfftmp*sin(pa2);
        auto dectmp= -azOfftmp*sin(pa2) + elOfftmp*cos(pa2);

        lat = ratmp*RAD_ASEC + TelRaPhys.array();
        lon = dectmp*RAD_ASEC + TelDecPhys.array();
    }

    else {
      lat =  (cos(TelElDes.array())*azOffset - sin(TelElDes.array())*elOffset)*RAD_ASEC
              + TelAzPhys.array();
      lon = (cos(TelElDes.array())*elOffset + sin(TelElDes.array())*azOffset)*RAD_ASEC
              + TelElPhys.array();

        //lat = -elOffset*RAD_ASEC + TelAzPhys.array();//telescope_data["TelAzPhys"].block(si,0,npts,1).array();
        //lon = -azOffset*RAD_ASEC + TelElPhys.array();//telescope_data["TelElPhys"].block(si,0,npts,1).array();
    }
}
} //namespace
