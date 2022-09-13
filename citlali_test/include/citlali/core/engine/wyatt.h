#pragma once

#include <vector>
#include <Eigen/Core>

#include <tula/grppi.h>

#include <citlali/core/engine/engine.h>
#include <citlali/core/utils/fitting.h>

using timestream::TCData;
using timestream::RTCProc;
using timestream::PTCProc;

namespace wyatt {

using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

class maps {
public:
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> x_onoff;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> x_off;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> r_onoff;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> r_off;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> xmap;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> rmap;

    Eigen::Matrix<double,Eigen::Dynamic,1> si;
    Eigen::Matrix<double,Eigen::Dynamic,1> ei;

    void resize(int nr, int nc, int tss, Eigen::Index xsd);
};

void maps::resize(int nr, int nc, int tss, Eigen::Index xsd) {
    int nrows = nr;
    int ncols = nc;
    Eigen::Index Xs_dim = xsd;
    int ts_size = tss;

    x_onoff = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Zero(nrows*ncols,Xs_dim);
    x_off = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Zero(nrows*ncols,Xs_dim);
    r_onoff = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Zero(nrows*ncols,Xs_dim);
    r_off = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Zero(nrows*ncols,Xs_dim);
    xmap = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Zero(nrows*ncols,Xs_dim);
    rmap = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Zero(nrows*ncols,Xs_dim);

    si.resize(ts_size);
    si.setZero();
    ei.resize(ts_size);
    ei.setZero();
};


template<typename ss>
Eigen::VectorXd get_wyatt(ss filepath) {
    std::ifstream infile;
    infile.open(filepath);
    std::string line;

    int lines;

    SPDLOG_INFO("Opening file {}", filepath);

    std::vector<double> t;

    int fj = 0;

    if (infile.is_open()){
        while (getline(infile,line)){
            t.push_back(std::stod(line));
            fj++;
        }
    }
    infile.close();

    Eigen::VectorXd t0s = Eigen::Map<Eigen::VectorXd>(&t[0],t.size());
    t0s = t0s.array()/1000000.;
    return std::move(t0s);
}


template<typename DerivedA, typename DerivedB, typename DerivedC>
void populate_indices(Eigen::DenseBase<DerivedA> &si, Eigen::DenseBase<DerivedA> &ei,
                      Eigen::DenseBase<DerivedB> &t0, Eigen::DenseBase<DerivedB> &t1,
                      Eigen::DenseBase<DerivedC> &time,
                      Eigen::Index Ts_dim_0){
    int start = 0;
    int end = 0;
    double si_min = 0;
    double ei_min = 0;

    for (int i=0;i<si.size();i++) {

        if (start < Ts_dim_0 - 1){
            si_min = abs(time(start)-t0[i]);
        }

        bool si_check = 1;

        while ((si_check) && (start < Ts_dim_0 - 1)) {
            if (si_min>=abs(time(start)-t0[i])){
                si_min = abs(time(start)-t0[i]);
                start = start + 1;
            }
            else{
                si_check=0;
            }
        }

        bool ei_check = 1;
        if (end < Ts_dim_0 - 1){
            ei_min = abs(time(end)-t1[i]);
        }

        end = start + 1;

        while ((ei_check) && (end < Ts_dim_0 - 1)) {

            if (ei_min>=abs(time(end)-t1[i])){
                ei_min = abs(time(end)-t1[i]);
                end = end + 1;
            }
            else{
                ei_check=0;
            }
        }

        si[i] = start;
        ei[i] = end;

        start = end + 1;
        //end = start+1;
    }
}


template<typename DerivedA>
Eigen::VectorXd calc_time(Eigen::DenseBase<DerivedA> &Ts, Eigen::Index Ts_dim_0, const double SampleFreqRaw,
                          const int start_time){
    Eigen::Matrix<double,Eigen::Dynamic,1> ppse;
    Eigen::Matrix<double,Eigen::Dynamic,1> time;
    Eigen::Matrix<double,Eigen::Dynamic,1> packet;

    ppse.resize(Ts_dim_0,1);
    packet.resize(Ts_dim_0,1);

    ppse = Ts.col(1).template cast <double> ();
    packet = Ts.col(3).template cast <double> ();

    ppse = ppse.array() - ppse(0);
    packet = packet.array() - packet(0);

    time.resize(Ts_dim_0,1);
    //time(0) = start_time;

    time = start_time + packet.array()/SampleFreqRaw;
    return std::move(time);
}


template<typename DerivedA>
Eigen::MatrixXd psd(Eigen::DenseBase<DerivedA> &in, maps &m, int n2, const double SampleFreq) {

    while(in.size()>n2){
        n2 = n2*2;
    }

    Eigen::Index npts = in.rows();

    Eigen::VectorXd tmpvec = Eigen::VectorXd::Zero(n2);
    Eigen::VectorXd hann = (0.5 - 0.5 * Eigen::ArrayXd::LinSpaced(npts, 0, 2.0 * pi / npts * (npts - 1)).cos()).matrix();
    Eigen::VectorXd filtered = hann.cwiseProduct(in.derived());

    tmpvec.head(hann.size()) = filtered;

    Eigen::FFT<double> fft;
    fft.SetFlag(Eigen::FFT<double>::HalfSpectrum);
    fft.SetFlag(Eigen::FFT<double>::Unscaled);

    Eigen::VectorXcd freqdata;

    freqdata.resize(tmpvec.size()/2 + 1);
    freqdata.setZero();

    fft.fwd(freqdata,tmpvec);

    Eigen::VectorXd psd;
    psd.resize(tmpvec.size()/2 + 1);
    psd.setZero();

    psd = freqdata.cwiseAbs2()/SampleFreq;
    psd = psd.array().sqrt();

    return std::move(psd);
}

template <typename ConfigType>
auto coadd(ConfigType &config) {
    auto ex_name = config.template get_typed<std::string>("pipeline_ex_policy");
    auto numfiles = config.template get_typed<int>("numfiles");
    auto nw = config.template get_typed<int>("nw");
    auto nrows = config.template get_typed<int>("nrows");
    auto ncols = config.template get_typed<int>("ncols");
    auto lowerfreq_on = config.template get_typed<double>("lowerfreq_on");
    auto upperfreq_on = config.template get_typed<double>("upperfreq_on");
    auto lowerfreq_off = config.template get_typed<double>("lowerfreq_off");
    auto upperfreq_off = config.template get_typed<double>("upperfreq_off");
    auto fwhm_upper_lim = config.template get_typed<double>("fwhm_upper_lim");
    auto fwhm_lower_lim = config.template get_typed<double>("fwhm_lower_lim");

    using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    int numdets = 1000;
    RowMatrixXd x_coadd_on(nrows*ncols,numdets);
    RowMatrixXd x_coadd_off(nrows*ncols,numdets);
    RowMatrixXd x_coadd_onoff(nrows*ncols,numdets);
    RowMatrixXd x_coadd_onoff_div(nrows*ncols,numdets);

    x_coadd_on.setZero();
    x_coadd_off.setZero();
    x_coadd_onoff.setZero();
    x_coadd_onoff_div.setZero();

    RowMatrixXd r_coadd_on(nrows*ncols,numdets);
    RowMatrixXd r_coadd_off(nrows*ncols,numdets);
    RowMatrixXd r_coadd_onoff(nrows*ncols,numdets);
    RowMatrixXd r_coadd_onoff_div(nrows*ncols,numdets);

    r_coadd_on.setZero();
    r_coadd_off.setZero();
    r_coadd_onoff.setZero();
    r_coadd_onoff_div.setZero();

    RowMatrixXd x2r2(nrows*ncols,numdets);

    x2r2.setZero();

    int dim0, dim1;

    for(int i=0;i<numfiles;i++){
        auto current_obsid = config.template get_typed<int>("obsid_"+std::to_string(i));
        auto input_filepath = "/data/data_contrib/wyatt/data/" + std::to_string(current_obsid) + "/" + std::to_string(current_obsid) + "_toltec" + std::to_string(nw) + ".nc";
        SPDLOG_INFO("input_filepath {}", input_filepath);

        netCDF::NcFile fo(input_filepath, netCDF::NcFile::read);
        auto vars = fo.getVars();

             //xmaps
             //const auto& x_on_var =  vars.find("x_on")->second;
        const auto& x_off_var =  vars.find("x_off")->second;

        Eigen::Index x_off_dim_0 = x_off_var.getDim(0).getSize();
        Eigen::Index x_off_dim_1 = x_off_var.getDim(1).getSize();

        dim0 = x_off_dim_0;
        dim1 = x_off_dim_1;

        SPDLOG_INFO("dim0 {}",dim0);
        SPDLOG_INFO("dim1 {}",dim1);

        RowMatrixXd x_on(x_off_dim_0,x_off_dim_1);
        RowMatrixXd x_off(x_off_dim_0,x_off_dim_1);
        RowMatrixXd x_onoff(x_off_dim_0,x_off_dim_1);

        RowMatrixXd r_on(x_off_dim_0,x_off_dim_1);
        RowMatrixXd r_off(x_off_dim_0,x_off_dim_1);
        RowMatrixXd r_onoff(x_off_dim_0,x_off_dim_1);

             //x_on_var.getVar(x_on.data());

        x_off_var.getVar(x_off.data());

        const auto& x_onoff_var =  vars.find("x_onoff")->second;
        x_onoff_var.getVar(x_onoff.data());

             //rmaps
             //const auto& r_on_var =  vars.find("r_on")->second;
             //r_on_var.getVar(r_on.data());

        const auto& r_off_var =  vars.find("r_off")->second;
        r_off_var.getVar(r_off.data());

        const auto& r_onoff_var =  vars.find("r_onoff")->second;
        r_onoff_var.getVar(r_onoff.data());

        x_on = x_onoff.array()*x_off.array();
        r_on = r_onoff.array()*r_off.array();

        x_coadd_on.block(0,0,x_off_dim_0,x_off_dim_1) = x_coadd_on.block(0,0,x_off_dim_0,x_off_dim_1) + x_on;
        x_coadd_off.block(0,0,x_off_dim_0,x_off_dim_1) = x_coadd_off.block(0,0,x_off_dim_0,x_off_dim_1) + x_off;
        x_coadd_onoff.block(0,0,x_off_dim_0,x_off_dim_1) = x_coadd_onoff.block(0,0,x_off_dim_0,x_off_dim_1) + x_onoff;

        r_coadd_on.block(0,0,x_off_dim_0,x_off_dim_1) = r_coadd_on.block(0,0,x_off_dim_0,x_off_dim_1) + r_on;
        r_coadd_off.block(0,0,x_off_dim_0,x_off_dim_1) = r_coadd_off.block(0,0,x_off_dim_0,x_off_dim_1) + r_off;
        r_coadd_onoff.block(0,0,x_off_dim_0,x_off_dim_1) = r_coadd_onoff.block(0,0,x_off_dim_0,x_off_dim_1) + r_onoff;
        x2r2.block(0,0,x_off_dim_0,x_off_dim_1) = x2r2.block(0,0,x_off_dim_0,x_off_dim_1).array() + (x_onoff.array().pow(2) + r_onoff.array().pow(2)).cwiseSqrt();
    }

    RowMatrixXd x_coadd_on_final = x_coadd_on.block(0,0,dim0,dim1)/numfiles;
    RowMatrixXd x_coadd_off_final = x_coadd_off.block(0,0,dim0,dim1)/numfiles;
    RowMatrixXd x_coadd_onoff_final = x_coadd_onoff.block(0,0,dim0,dim1)/numfiles;
    RowMatrixXd x_coadd_onoff_div_final = x_coadd_on_final.array()/x_coadd_off_final.array();

    RowMatrixXd r_coadd_on_final = r_coadd_on.block(0,0,dim0,dim1)/numfiles;
    RowMatrixXd r_coadd_off_final = r_coadd_off.block(0,0,dim0,dim1)/numfiles;
    RowMatrixXd r_coadd_onoff_final = r_coadd_onoff.block(0,0,dim0,dim1)/numfiles;
    RowMatrixXd r_coadd_onoff_div_final = r_coadd_on_final.array()/r_coadd_off_final.array();

    RowMatrixXd x2r2_final = x2r2.block(0,0,dim0,dim1)/numfiles;

    SPDLOG_INFO("Co-added all maps");

    Eigen::MatrixXd params(6,dim1);
    params.setZero();
    int nthreads = Eigen::nbThreads();

    grppi::pipeline(grppi::parallel_execution_omp(), [&]() -> std::optional<int> {
            static int dets = 0;
            while(dets<dim1){
                return dets++;
            }
            return {};
        },

        grppi::farm(nthreads,[&](auto &dets) {
            SPDLOG_INFO("Fitting det {}", dets);
            Eigen::VectorXd xrng = Eigen::VectorXd::LinSpaced(nrows, 0, nrows-1);
            Eigen::VectorXd yrng = Eigen::VectorXd::LinSpaced(ncols, 0, ncols-1);

            //Eigen::VectorXd onoff_col = x_coadd_onoff_final.col(dets);
            Eigen::VectorXd onoff_col = x_coadd_onoff_final.col(dets);
            //Eigen::VectorXd onoff_col = x_coadd_on_final.col(dets);
            Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> onoff_matrix = Eigen::Map<Eigen::MatrixXd> (onoff_col.data(),nrows,ncols);

                 //Eigen::VectorXd off_col = maps.off.col(dets);
                 //Eigen::VectorXd on_col(off_col.size());

                 //on_col = (onoff_col.array() * off_col.array()).eval();

                 //Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> on_matrix = Eigen::Map<Eigen::MatrixXd> (on_col.data(),nrows,ncols);

            int l = 0;
            for (int k=0;k<nrows;k++) {
                for (int m =0;m<ncols;m++) {
                    onoff_matrix(k,m) = onoff_col(l);
                    //on_matrix(k,m) = on_col(l);
                    l++;
                }
            }

            double temp = onoff_matrix.mean();
            onoff_matrix = onoff_matrix.array() - temp;

                 //on_matrix = (on_matrix.array()/on_matrix.mean()).eval();

            for (int i=0;i<nrows;i=i+2) {
                Eigen::VectorXd tmprv = onoff_matrix.row(i).reverse();
                //Eigen::VectorXd tmprv = on_matrix.row(i).reverse();
                onoff_matrix.row(i) = tmprv;
                //on_matrix.row(i) = tmprv;

            }
            double th = onoff_matrix.maxCoeff();
            //double th = on_matrix.maxCoeff();

            int tmppeak_i = 0;
            int tmppeak_j = 0;
            for (int k=0;k<nrows;k++) {
                for (int m=0;m<ncols;m++) {
                    if(onoff_matrix(k,m) == th){
                        //if(on_matrix(k,m) == th){
                        tmppeak_i = k;
                        tmppeak_j = m;
                    }
                    if(onoff_matrix(k,m) < 0.1*th){
                        onoff_matrix(k,m) = 0;
                    }
                }
            }

            Eigen::VectorXd init_p(6);
            double init_fwhm = 1.5;//fwhm_upper_lim;
            init_p << th, tmppeak_j,tmppeak_i, init_fwhm, init_fwhm, 0.;

            SPDLOG_INFO("Initial Peak: {}",th);
            SPDLOG_INFO("Initial x: {}",tmppeak_i);
            SPDLOG_INFO("Initial y: {}",tmppeak_j);
            SPDLOG_INFO("Initial fwhmx: {}",init_fwhm);

            Eigen::VectorXd lower_limits(6);
            Eigen::VectorXd upper_limits(6);

            Eigen::Matrix<int,4,1> a14_nws;
            a14_nws(0) = 7;
            a14_nws(1) = 8;
            a14_nws(2) = 9;
            a14_nws(3) = 10;

            Eigen::Matrix<int,2,1> a20_nws;
            a20_nws(0) = 11;
            a20_nws(1) = 12;

            fwhm_upper_lim = 1.75;
            fwhm_lower_lim = 1.0;

            for(int nw_i=0;nw_i<a14_nws.size();nw_i++){
                if(nw == a14_nws(nw_i)){
                    fwhm_upper_lim = 2.5;
                }
            }


            SPDLOG_INFO("NW {}",nw);

            for(int nw_i=0;nw_i<a20_nws.size();nw_i++){
                if(nw == a20_nws(nw_i)){
                    fwhm_upper_lim = 2.5;
                }
            }

            SPDLOG_INFO("fwhm upper limit {}",fwhm_upper_lim);

            auto g = gaussfit::modelgen<gaussfit::Gaussian2D>(init_p);

            Eigen::MatrixXd sigmamatrix(nrows,ncols);
            sigmamatrix.setOnes();

            auto _p = g.params;
            auto xy = g.meshgrid(yrng, xrng);

            int range = 100;
            int minsize = 0;

            fwhm_upper_lim = 3.0;
            fwhm_lower_lim = 1.0;

            lower_limits << 0.75*th, std::max(tmppeak_j - range,minsize),std::max(tmppeak_i - range,minsize),fwhm_lower_lim, fwhm_lower_lim, 0.;
            upper_limits << 1.25*th, std::min(tmppeak_j + range,ncols),std::min(tmppeak_i + range,nrows), fwhm_upper_lim, fwhm_upper_lim, 3.1415/4.;

            Eigen::MatrixXd limits(6, 2);

            limits.col(0) << 0.75*th, std::max(tmppeak_j - range,minsize),std::max(tmppeak_i - range,minsize),fwhm_lower_lim, fwhm_lower_lim, 0.;
            limits.col(1) << 1.25*th, std::min(tmppeak_j + range,ncols),std::min(tmppeak_i + range,nrows), fwhm_upper_lim, fwhm_upper_lim, 3.1415/4.;


            //Run the fit
            auto [g_fit,cov] = curvefit_ceres(g, _p, xy, onoff_matrix, sigmamatrix,limits);
            //auto g_fit = curvefit_ceres(g, _p, xy, on_matrix, sigmamatrix,lower_limits,upper_limits);
            params.col(dets) = g_fit.params;
            SPDLOG_INFO(g_fit.params);
            return dets;}));


    using namespace netCDF;
    using namespace netCDF::exceptions;

    auto output_filepath = config.template get_typed<std::string>("output_filepath");

    try{
        std::string output_filename = output_filepath+"coadded_toltec"+"_"+std::to_string(nw)+".nc";
        //Create NetCDF file
        SPDLOG_INFO("output_filename {}",output_filename);
        NcFile fo(output_filename, NcFile::replace);

        int nalls = ncols*nrows;

        NcDim nall = fo.addDim("nall", nalls);
        NcDim ndet_dim = fo.addDim("ndet", dim1);

        NcDim nrowsdim = fo.addDim("nrows", nrows);
        NcDim ncolsdim = fo.addDim("ncols", ncols);



        std::vector<NcDim> dims;
        dims.push_back(nall);
        dims.push_back(ndet_dim);

        auto x_onoff_var = "x_onoff";
        NcVar x_onoff_data = fo.addVar(x_onoff_var, ncDouble, dims);
        x_onoff_data.putVar(x_coadd_onoff_final.data());
        // x_onoff_data.putVar(x_coadd_on_final.data());

        auto x_off_var = "x_off";
        NcVar x_off_data = fo.addVar(x_off_var, ncDouble, dims);
        x_off_data.putVar(x_coadd_off_final.data());

        auto r_onoff_var = "r_onoff";
        NcVar r_onoff_data = fo.addVar(r_onoff_var, ncDouble, dims);
        r_onoff_data.putVar(r_coadd_onoff_final.data());
        //r_onoff_data.putVar(r_coadd_on_final.data());

        auto r_off_var = "r_off";
        NcVar r_off_data = fo.addVar(r_off_var, ncDouble, dims);
        r_off_data.putVar(r_coadd_off_final.data());

        /*
        auto xmap_var = "xmap";
        NcVar xmap_data = fo.addVar(xmap_var, ncDouble, dims);
        xmap_data.putVar(maps.xmap.data());

        auto rmap_var = "rmap";
        NcVar rmap_data = fo.addVar(rmap_var, ncDouble, dims);
        rmap_data.putVar(maps.rmap.data());

        NcVar si_var = fo.addVar("si",ncDouble,nall);
        NcVar ei_var = fo.addVar("ei",ncDouble,nall);

        si_var.putVar(maps.si.data());
        ei_var.putVar(maps.ei.data());
        */
            for (int i = 0;i<dim1;i++) {
            auto mapfitvar = "map_fits" + std::to_string(i);
            NcVar mapfitdata = fo.addVar(mapfitvar, ncDouble);

            mapfitdata.putAtt("amplitude",ncDouble,params(0,i));
            mapfitdata.putAtt("offset_x",ncDouble,params(1,i));
            mapfitdata.putAtt("offset_y",ncDouble,params(2,i));
            mapfitdata.putAtt("FWHM_x",ncDouble,params(3,i));
            mapfitdata.putAtt("FWHM_y",ncDouble,params(4,i));
        }

        fo.close();
        return EXIT_SUCCESS;
    }
    catch (NcException &e) {
        SPDLOG_ERROR("{}", e.what());
        throw DataIOError{fmt::format(
            "failed to load data from netCDF file {}", output_filepath)};
    }
}


} //namespace

// selects the type of TCData
using timestream::TCDataKind;

class Wyatt: public EngineBase {

    void setup();
    auto run_timestream();
    auto run_loop();

    template <class KidsProc, class RawObs>
    auto timestream_pipeline(KidsProc &, RawObs &);

    template <class KidsProc, class RawObs>
    auto loop_pipeline(KidsProc &, RawObs &);

    template <class KidsProc, class RawObs>
    auto pipeline(KidsProc &, RawObs &);

    void output();

};
