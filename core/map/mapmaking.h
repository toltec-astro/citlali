#pragma once

//Needed for config file input

#include "../timestream/timestream.h"

//Eigen Includes
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream> // cout

#include <boost/math/constants/constants.hpp>
#include <cmath>

#include "../../common_utils/src/utils/algorithm/ei_stats.h"

// Arcseconds in 360 degrees
#define ASEC_CIRC 1296000.0
// rad per arcsecond
#define RAD_ASEC (2.0*pi / ASEC_CIRC)

typedef std::map<string,Eigen::Matrix<double,Eigen::Dynamic,1>> pointing;

using namespace std;
using timestream::PTCData;

namespace mapmaking{

class MapStruct
{
public:
    MapStruct() {}

    Eigen::Tensor<double, 2> signal;
    Eigen::Tensor<double, 2> wtt;
    Eigen::Tensor<double, 2> sigma;
    Eigen::Tensor<double, 2> kernel;
    Eigen::Tensor<double, 2> kernel_wtt;
    Eigen::Tensor<double, 2> inttime;
    Eigen::Tensor<double,3> noisemaps;

    double pixelsize;
    Eigen::Index nrows, ncols, npixels, NNoiseMapsPerObs;
    Eigen::VectorXd rowcoordphys, colcoordphys;

    void resize(Eigen::Index ndet);
};

void MapStruct::resize(Eigen::Index ndet){

    signal.resize(nrows,ncols);
    wtt.resize(nrows,ncols);
    sigma.resize(nrows,ncols);
    kernel.resize(nrows,ncols);
    kernel_wtt.resize(nrows,ncols);
    inttime.resize(nrows,ncols);

    npixels = nrows*ncols;

    signal.setZero();
    wtt.setZero();
    kernel_wtt.setZero();
    sigma.setZero();
    kernel.setZero();
    inttime.setZero();
}

namespace internal {

//generate the pointing data for this detector this version calculates it from scratch and requires a telescope object
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

  if(!azelMap){
      auto azOfftmp = cos(telescope_data["TelElDes"].array())*azOffset - sin(telescope_data["TelElDes"].array())*elOffset;
      auto elOfftmp = cos(telescope_data["TelElDes"].array())*elOffset + sin(telescope_data["TelElDes"].array())*azOffset;
      auto pa2 = telescope_data["ParAng"].array()-pi;

      auto ratmp = -azOfftmp*cos(pa2) - elOfftmp*sin(pa2);
      auto dectmp= -azOfftmp*sin(pa2) + elOfftmp*cos(pa2);
      lat = ratmp*RAD_ASEC + telescope_data["TelRaPhys"].array();
      lon = dectmp*RAD_ASEC + telescope_data["TelDecPhys"].array();
  }

  else {
      /*lat =  (cos(telescope_data["TelElDes"].block(si,0,npts,1).array())*azOffset - sin(telescope_data["TelElDes"].block(si,0,npts,1).array())*elOffset)*RAD_ASEC
              + telescope_data["TelAzPhys"].block(si,0,npts,1).array();
      lon = (cos(telescope_data["TelElDes"].block(si,0,npts,1).array())*elOffset + sin(telescope_data["TelElDes"].block(si,0,npts,1).array())*azOffset)*RAD_ASEC
              + telescope_data["TelElPhys"].block(si,0,npts,1).array();
*/

  lat = -elOffset*RAD_ASEC + TelAzPhys.array();//telescope_data["TelAzPhys"].block(si,0,npts,1).array();
  lon = -azOffset*RAD_ASEC + TelElPhys.array();//telescope_data["TelElPhys"].block(si,0,npts,1).array();
  }

}

void physToAbs(double &pra, double &pdec, double &cra, double &cdec,
           double &ara, double &adec){
    double rho = sqrt(pow(pra,2)+pow(pdec,2));
    double c = atan(rho);
    if(c == 0.){
      ara = cra;
      adec = cdec;
    } else {
      double ccwhn0 = cos(c);
      double scwhn0 = sin(c);
      double ccdec = cos(cdec);
      double scdec = sin(cdec);
      double a1;
      double a2;
      a1 = ccwhn0*scdec + pdec*scwhn0*ccdec/rho;
      adec = asin(a1);
      a2 = pra*scwhn0/(rho*ccdec*ccwhn0 - pdec*scdec*scwhn0);
      ara = cra + atan(a2);
    }
}

template <typename DerivedA>
void latlonPhysToIndex(double &lat, double &lon, Eigen::Index &irow, Eigen::Index &icol,
                      DerivedA &mapstruct){
    double ps = mapstruct.pixelsize*RAD_ASEC;
    irow = lat / ps + (mapstruct.nrows + 1.) / 2.;
    icol = lon / ps + (mapstruct.ncols + 1.) / 2.;
}

template <typename DerivedA>
void getRowCol(DerivedA &mapstruct, double mgrid_0, double mgrid_1,
                       double x_max_act, double x_min_act, double y_max_act, double y_min_act){
    // start with the map dimensions but copy IDL utilities
    // for aligning the pixels in various maps

    double ps = mapstruct.pixelsize*RAD_ASEC;

    // explicitly center on mastergrid
    // require that maps are even dimensioned with a few extra pixels.
    int xminpix = ceil(abs(x_min_act / ps));
    int xmaxpix = ceil(abs(x_max_act / ps));
    xmaxpix = max(xminpix, xmaxpix);

    mapstruct.nrows = 2 * xmaxpix + 4;
    int yminpix = ceil(abs(y_min_act / ps));
    int ymaxpix = ceil(abs(y_max_act / ps));
    ymaxpix = max(yminpix, ymaxpix);
    mapstruct.ncols = 2 * ymaxpix + 4;
    mapstruct.npixels = mapstruct.nrows * mapstruct.ncols;

    SPDLOG_INFO("nrows {}", mapstruct.nrows);
    SPDLOG_INFO("ncols {}", mapstruct.ncols);
    SPDLOG_INFO("npixels {}", mapstruct.npixels);

    // physical coordinates: this grid is set up so that physical
    // coordinates are 0 at center of pixel near center of map

    mapstruct.rowcoordphys.resize(mapstruct.nrows);
    mapstruct.colcoordphys.resize(mapstruct.ncols);

    for (int i = 0; i < mapstruct.nrows; i++)
        mapstruct.rowcoordphys[i] = (i - (mapstruct.nrows + 1.) / 2.) * ps;
    for (int i = 0; i < mapstruct.ncols; i++)
        mapstruct.colcoordphys[i] = (i - (mapstruct.ncols + 1.) / 2.) * ps;

    // matrices of absolute coordinates
   /* Eigen::MatrixXd xCoordsAbs(nrows, ncols);
    Eigen::MatrixXd yCoordsAbs(nrows, ncols);

    for (int i = 0; i < nrows; i++)
        for (int j = 0; j < ncols; j++) {
            physToAbs(rowCoordsPhys[i], colCoordsPhys[j], mgrid_0,
                      mgrid_1, xCoordsAbs(i,j), yCoordsAbs(i,j));
        }*/
}

template <typename DerivedA, typename DerivedB>
double calcScanWeight(const Eigen::DenseBase<DerivedA> &scans, const Eigen::DenseBase<DerivedB> &flags, const double samplerate){

  //Get number of good samples in scan
  Eigen::Index numgood;
  numgood = (flags.derived().array()==1).count();

  //calculate the standard deviation
  double tmp = std::sqrt(((scans.derived().array() * flags.derived().template cast<double>().array())
                   -  (scans.derived().array() * flags.derived().template cast<double>().array()).sum()/numgood).square().sum()/(numgood - 1.0));

  //insist on at least 1s of good samples
  if (tmp!=tmp || numgood < samplerate)
      return 0.0;
  else
      return 1.0/pow(tmp,2.0);
}

template<typename DerivedA>
DerivedA calculateWeights(std::vector<PTCData> &ptcs, const int samplerate){
    Eigen::Index ndet = ptcs[0].scans.cols();
    Eigen::Index nscans = ptcs.size();

    Eigen::MatrixXd tmpwt(ndet, nscans);

    for (Eigen::Index i = 0;i<nscans;i++) {
        for (Eigen::Index j = 0;j<ndet;j++) {
            Eigen::Map<Eigen::VectorXd> scns(ptcs[i].scans.col(j).data(),ptcs[i].scans.rows());
            Eigen::Map<Eigen::Matrix<bool,Eigen::Dynamic,1>> flgs(ptcs[i].flags.col(j).data(),ptcs[i].flags.rows());

            tmpwt(j,i) = internal::calcScanWeight(scns,flgs,samplerate);
        }
    }

      Eigen::Map<Eigen::VectorXd> tmpwts (tmpwt.data(),nscans*ndet);
      double medweight = alg::median(tmpwts);

      for(Eigen::Index i=0;i<nscans;i++){
          for(Eigen::Index j=0;j<ndet;j++){
            if(tmpwt(j,i) > 2.* medweight){
                tmpwt(j,i) = medweight / 2.0;
            }
          }
      }
    return tmpwt;
}
} //namespace internal

template <typename DerivedA, typename DerivedC>//, typename DerivedB>
void mapgen(std::vector<PTCData> &ptcs,
                 pointing &telescope_data,
                 double mgrid_0, double mgrid_1, const int samplerate,
                 DerivedC &mapstruct, Eigen::DenseBase<DerivedA> &offsets, Eigen::Index det,
                 Eigen::DenseBase<DerivedA> &tmpwts){

    //MapStruct map;

    double ps = mapstruct.pixelsize*RAD_ASEC;

    Eigen::Index nscans = ptcs.size();
    Eigen::VectorXd lat, lon;
    Eigen::MatrixXi nValues(mapstruct.nrows,mapstruct.ncols);
    nValues.setZero();

    internal::getPointing(telescope_data, lat, lon, offsets);

    for (Eigen::Index j = 0; j < nscans; j++) {
        Eigen::Index s = 0;
        Eigen::Index si = ptcs[j].scanindex(0);
        Eigen::Index ei = ptcs[j].scanindex(1);

        for (Eigen::Index k=si;k<ei;k++) {
            if(ptcs[j].flags(s,det)){
                double hx = ptcs[j].scans(s,det)*tmpwts(det,j);
                Eigen::Index irow = 0;
                Eigen::Index icol = 0;

                internal::latlonPhysToIndex(lat[k], lon[k], irow, icol, mapstruct);

                mapstruct.signal(det,irow,icol) += hx;
                mapstruct.wtt(det,irow,icol) += tmpwts(det,j);

                nValues(irow,icol)++;
            } //if flags is True
            s++;
        } //per scan loop
    } //scan loop

    for(Eigen::Index irow = 0; irow < mapstruct.nrows; irow++)
        for(Eigen::Index icol = 0; icol < mapstruct.ncols; icol++){
            if(nValues(irow,icol) != 0)
            {
                mapstruct.signal(det,irow,icol) = -mapstruct.signal(det,irow,icol)/mapstruct.wtt(det,irow,icol);
                mapstruct.sigma(det,irow,icol) = 1./std::sqrt(mapstruct.wtt(det,irow,icol));
            }
            else
            {
                mapstruct.signal(det,irow,icol) = 0.;//std::nan("");
                mapstruct.sigma(det,irow,icol) = 0.;//std::numeric_limits<double>::infinity();
            }
        }
}
} //namespace
