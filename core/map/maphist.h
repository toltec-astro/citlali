#pragma once

namespace mapmaking {


//Structure to hold map histogram results and output them to a netcdf file
struct maphiststruct {
    Eigen::VectorXd histBins;
    Eigen::VectorXd histVals;

    maphiststruct() {}

    int toNcFile(const std::string &filepath){
        try{
            //Create NetCDF file
            netCDF::NcFile fo(filepath, netCDF::NcFile::replace);
            netCDF::NcDim nbins = fo.addDim("nbins", histBins.size());
            netCDF::NcDim nhists = fo.addDim("nhists", histVals.size());

            auto histBins_var = "histBins";
            auto histVals_var = "histVals";
            netCDF::NcVar histBins_data = fo.addVar(histBins_var, netCDF::ncDouble, nbins);
            netCDF::NcVar histVals_data = fo.addVar(histVals_var, netCDF::ncDouble, nhists);

            histBins_data.putVar(histBins.data());
            histVals_data.putVar(histVals.data());

            fo.close();
        }
        catch(netCDF::exceptions::NcException& e)
        {e.what();
            return NC_ERR;
        }
        return 0;
    }

};

namespace internal {

template<typename DerivedA, typename DerivedB>
void histogramImage(Eigen::DenseBase<DerivedA> &im, int nbins, Eigen::DenseBase<DerivedB> &binloc, Eigen::DenseBase<DerivedB> &hist){
  //allocate memory for the histogram, binloc, and hist
  binloc.derived().resize(nbins);
  hist.derived().resize(nbins-1);

  double min = im.minCoeff();
  double max = im.maxCoeff();

  //force the histogram to be symmetric about 0.
  double rg = (abs(min) > abs(max)) ? abs(min) : abs(max);

  binloc = Eigen::VectorXd::LinSpaced(nbins,-rg,rg);

  for(int i=0;i<nbins-1;i++){
      hist(i) = ((im.derived().array() >= binloc(i)) && (im.derived().array() < binloc(i+1))).count();
  }
}
} //internal

template <typename Derived>
std::tuple<Eigen::VectorXd,Eigen::VectorXd> calcMapHistogram(Derived &mapstruct, int nbins, double coverageCut) {
  // space for the results
  Eigen::VectorXd histBins, histVals;

  // set the coverage cut ranges
  double weightCut = mapmaking::internal::findWeightThresh(mapstruct,coverageCut);
  auto [cutXRange, cutYRange] = mapmaking::internal::setCoverageCutRanges(mapstruct,weightCut);

  // make the coverage cut
  int nx = cutXRange[1] - cutXRange[0] + 1;
  int ny = cutYRange[1] - cutYRange[0] + 1;
  //Eigen::MatrixXd im(nx, ny);
  //for (int i = 0; i < nx; i++)
    //for (int j = 0; j < ny; j++){
      //     im(i,j) = mapstruct.signal(cutXRange[0] + i,cutYRange[0] + j);
  //}

  //Create a map to the eigen signal tensor
   Eigen::Map<Eigen::MatrixXd> signal_matrix(mapstruct.signal.data(),mapstruct.signal.dimension(0),mapstruct.signal.dimension(1));

   //Get the desire region
   auto im = signal_matrix.block(cutXRange[0],cutYRange[0],nx,ny);

  // do the histogram
  internal::histogramImage(im, nbins, histBins, histVals);

  return std::make_tuple(histBins,histVals);
}
} //namespace
