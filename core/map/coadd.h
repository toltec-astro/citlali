#pragma once

/* This is a place holder class for coaddition until we get file input
and MPI implementation done.*/

namespace mapmaking {

class CoaddedMapStruct : public MapUtils, public Wiener{
public:

    int nrows, ncols;
    double pixelsize;

    Eigen::MatrixXd signal;
    Eigen::MatrixXd weight;
    Eigen::MatrixXd kernel;
    Eigen::MatrixXd intMap;
    Eigen::MatrixXd noiseMaps;

    void allocateMaps();
    template <class MS>
    void coaddMaps();
    void mapNormalize();
};

template <class MS>
void coaddMaps(){

}

} //namespace
