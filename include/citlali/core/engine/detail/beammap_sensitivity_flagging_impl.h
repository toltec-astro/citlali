#pragma once

// Beammap implementation detail.
// Include this only after Beammap has been declared.

std::map<Eigen::Index, double> Beammap::beammap_network_median_sensitivities() {
    std::map<Eigen::Index, double> nw_median_sens;

    logger->debug("calculating mean sensitivities");
    for (Eigen::Index i=0; i<calib.n_nws; ++i) {
        Eigen::Index nw = calib.nws(i);

        auto nw_sens = calib.apt["sens"](Eigen::seq(std::get<0>(calib.nw_limits[nw]),
                                                    std::get<1>(calib.nw_limits[nw])-1));

        Eigen::Index n_good_det =
            (calib.apt["flag"](Eigen::seq(std::get<0>(calib.nw_limits[nw]),
                                          std::get<1>(calib.nw_limits[nw])-1)).array()==0).count();

        if (n_good_det>0) {
            Eigen::VectorXd sens(n_good_det);

            Eigen::Index j = std::get<0>(calib.nw_limits[nw]);
            Eigen::Index k = 0;
            for (Eigen::Index m=0; m<nw_sens.size(); m++) {
                if (calib.apt["flag"](j)==0) {
                    sens(k) = nw_sens(m);
                    k++;
                }
                j++;
            }
            nw_median_sens[nw] = tula::alg::median(sens);
        }
        else {
            nw_median_sens[nw] = tula::alg::median(nw_sens);
        }
    }

    return nw_median_sens;
}

void Beammap::flag_beammap_sensitivity_outliers(
    std::map<Eigen::Index, double> &nw_median_sens,
    double lower_sens_factor,
    double upper_sens_factor,
    const std::string &runtime_parallel_policy,
    std::atomic<int> &n_flagged_dets) {
    logger->debug("flagging sensitivities");
    grppi::map(tula::grppi_utils::dyn_ex(runtime_parallel_policy), det_in_vec, det_out_vec, [&](auto i) {
        auto nw_index = calib.apt["nw"](i);

        if (calib.apt["sens"](i) < lower_sens_factor*nw_median_sens[nw_index] ||
            (calib.apt["sens"](i) > upper_sens_factor*nw_median_sens[nw_index] && upper_sens_factor > 0)) {
            mark_beammap_detector_flagged(i, AptFlags::Sens, n_flagged_dets);
        }

        return 0;
    });
}
