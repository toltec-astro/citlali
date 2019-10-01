#pragma once

namespace mapmaking{

template <typename DerivedA, typename DerivedB, typename DerivedC>//, typename DerivedB>
void generatemaps(PTCData &ptc,
                 pointing &telescope_data,
                 double mgrid_0, double mgrid_1, const double samplerate,
                 DerivedA &mapstruct, Eigen::DenseBase<DerivedB> &offsets,
                 const double &tmpwts, const int &NNoiseMapsPerObs,
                 Eigen::DenseBase<DerivedC> &sn,
                 const int dsf){

    int ndet = ptc.scans.cols();

    //boost::random::mt19937 rng;
    //boost::random::uniform_int_distribution<> rands(-1,1);

    //std::mutex farm_mutex;

    /*if (a->detectors[di[0]].atmTemplate.size() > 0.0){
      atmTemplate = new Map(mname.assign("atmTemplate"), nrows, ncols, pixelSize,
      weight->image, rowCoordsPhys, colCoordsPhys);
    }*/

    //put together the maps
    //Eigen::Tensor<double, 1> sn(NNoiseMapsPerObs);
    //sn.setConstant(1);

    Eigen::Index si = ptc.scanindex(0);
    Eigen::Index ei = ptc.scanindex(1);

    for(Eigen::Index i=0;i<ndet;i++){
        {
            //logging::scoped_timeit timer("loops");
        Eigen::VectorXd lat, lon;
        internal::getPointing(telescope_data, lat, lon, offsets, i, si, ei, dsf);
        for(int s=0;s<ptc.scans.rows();s++){
            if(ptc.flags(s,i)){
                //get the row and column index corresponding to the ra and dec
                Eigen::Index irow = 0;
                Eigen::Index icol = 0;
                internal::latlonPhysToIndex(lat[s], lon[s], irow, icol, mapstruct);

                //weight map
                {
                //std::scoped_lock lock(farm_mutex);
                   // logging::scoped_timeit timer("wtt");
                mapstruct.wtt(irow,icol) += tmpwts;
                }

                //inttime map
                {
                //std::scoped_lock lock(farm_mutex);
                   // logging::scoped_timeit timer("inttime");
                mapstruct.inttime(irow,icol) += 1./samplerate;
                }

                //check for NaN
                double hx = ptc.scans(s,i)*tmpwts;
                double hk = ptc.kernelscans(s,i)*tmpwts;

                /*double ha = 0.0;
                  if (atmTemplate)
                          ha = tmpwt[i][k]*a->detectors[di[i]].atmTemplate[j];
                */
                  /*if(hx != hx || hk != hk){
                      //cerr << "NaN detected on file: "<<ap->getMapFile() << endl;
                      //cerr << "tmpwt: " << tmpwt[i][k] << endl;
                      //cerr << "det: " << a->detectors[di[i]].hValues[j] << endl;
                      //cerr << "ker: " << a->detectors[di[i]].hKernel[j] << endl;
                      SPDLOG_INFO("  i = {}, hx {}, hk {}",i,hx,hk);
                      SPDLOG_INFO("  ktmp = {}, ptc.kernelscans(s,i) {}",tmpwts,ptc.kernelscans(s,i));

                      SPDLOG_INFO("  s = {}",s);
                      exit(1);
                  }*/

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
} //namespace
