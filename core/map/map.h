#pragma once
#include "map_utils.h"

namespace mapmaking{

/**
 * @brief Generates science maps
 */
template <typename DerivedA, typename DerivedB, typename DerivedC, typename DerivedD>//, typename DerivedB>
void generate_scimaps(TCData<LaliDataKind::PTC,Eigen::MatrixXd> &ptc, pointing &telescope_data,
                 double mgrid_0, double mgrid_1, const double samplerate,
                 DerivedA &mapstruct, Eigen::DenseBase<DerivedB> &offsets,
                 const Eigen::DenseBase<DerivedD> &tmpwts, const int &NNoiseMapsPerObs,
                 Eigen::DenseBase<DerivedC> &sn,
                 const int dsf){

    Eigen::Index ndet = ptc.scans.data.cols();

    //std::mutex farm_mutex;

    /*if (a->detectors[di[0]].atmTemplate.size() > 0.0){
      atmTemplate = new Map(mname.assign("atmTemplate"), nrows, ncols, pixelSize,
      weight->image, rowCoordsPhys, colCoordsPhys);
    }*/

    //to be removed
    Eigen::Index si = ptc.scanindex.data(0);
    Eigen::Index ei = ptc.scanindex.data(1);

    //Loop throught detectors since we are parallized by scan
    for(Eigen::Index i=0;i<ndet;i++){
        {
        //logging::scoped_timeit timer("loops");
        //get pointing for each scan
        Eigen::VectorXd lat, lon;
        getPointing(telescope_data, lat, lon, offsets, i, si, ei, dsf);
        for(int s=0;s<ptc.scans.data.rows();s++){
            if(ptc.flags.data(s,i)){
                //get the row and column index corresponding to the ra and dec
                Eigen::Index irow = 0;
                Eigen::Index icol = 0;
                //get pixel row and column index in matrix.
                latlonPhysToIndex(lat[s], lon[s], irow, icol, mapstruct);

                //weight map
                {
                //std::scoped_lock lock(farm_mutex);
                   // logging::scoped_timeit timer("wtt");
                mapstruct.wtt(irow,icol) += tmpwts[i];
                }

                //inttime map
                {
                //std::scoped_lock lock(farm_mutex);
                   // logging::scoped_timeit timer("inttime");
                mapstruct.inttime(irow,icol) += 1./samplerate;
                }

                double hx = ptc.scans.data(s,i)*tmpwts[i];
                double hk = ptc.kernelscans.data(s,i)*tmpwts[i];

                /*double ha = 0.0;
                  if (atmTemplate)
                          ha = tmpwt[i][k]*a->detectors[di[i]].atmTemplate[j];
                */

                //check for NaNs
                if(hx != hx || hk != hk){
                      //cerr << "NaN detected on file: "<<ap->getMapFile() << endl;
                      //cerr << "tmpwt: " << tmpwt[i][k] << endl;
                      //cerr << "det: " << a->detectors[di[i]].hValues[j] << endl;
                      //cerr << "ker: " << a->detectors[di[i]].hKernel[j] << endl;
                      SPDLOG_INFO("  i = {}, hx {}, hk {}",i,hx,hk);
                      SPDLOG_INFO("  ktmp = {}, ptc.kernelscans.data(s,i) {}",tmpwts[i],ptc.kernelscans.data(s,i));

                      SPDLOG_INFO("  s = {}",s);
                      exit(1);
                }

              //signal map
              {
              //std::scoped_lock lock(farm_mutex);
                     //logging::scoped_timeit timer("signal");
              mapstruct.signal(irow,icol) += hx;
              }

              //kernel map
              {
              //std::scoped_lock lock(farm_mutex);
                     // logging::scoped_timeit timer("kernel");
              mapstruct.kernel(irow,icol) += hk;
              }

              //noise maps
              for(int kk=0;kk<NNoiseMapsPerObs;kk++){
                  {
                  //std::scoped_lock lock(farm_mutex);
                 // logging::scoped_timeit timer("noisemaps");
                  //mapstruct.noisemaps(kk,irow*mapstruct.ncols+icol) += sn(kk)*hx;
                  mapstruct.noisemaps(kk,irow,icol) = mapstruct.noisemaps(kk,irow,icol) + sn(kk)*hx;
                  }
              }

              /*if (atmTemplate)
                      atmTemplate->image[irow][icol] += ha;
                      */
            }
          }
    }

    }
}

/**
 * @brief Normalization for science maps.
 * Needs to be done separately due to
 * streaming.
 */
template <typename T>
void mapnormalize(T &lal){
    double wt;
    double atmpix=0;
    for(Eigen::Index i=0;i<lal.br.mapstruct.nrows;i++){
        for(Eigen::Index j=0;j<lal.br.mapstruct.ncols;j++){
            wt = lal.br.mapstruct.wtt(i,j);
            if(wt != 0.){
                //if (atmTemplate)
                //atmpix = atmTemplate->image(i,j);
                //br.mapstruct.signal(i,j) = -(br.mapstruct.signal(i,j)-atmpix)/wt;
                lal.br.mapstruct.signal(i,j) = -(lal.br.mapstruct.signal(i,j))/wt;
                lal.br.mapstruct.kernel(i,j) = (lal.br.mapstruct.kernel(i,j))/wt;

                for(Eigen::Index kk=0;kk<lal.br.mapstruct.NNoiseMapsPerObs;kk++){
                    lal.br.mapstruct.noisemaps(kk,i,j) = lal.br.mapstruct.noisemaps(kk,i,j)/wt;
                }
            }
            else{
                lal.br.mapstruct.signal(i,j) = 0.;
                lal.br.mapstruct.kernel(i,j) = 0.;
                for(Eigen::Index kk=0;kk<lal.br.mapstruct.NNoiseMapsPerObs;kk++){
                    lal.br.mapstruct.noisemaps(kk,i,j) = 0.;
                }
            }
        }
    }
}
} //namespace
