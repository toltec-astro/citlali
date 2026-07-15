#pragma once

#include <citlali/core/error/error.h>
#include <citlali/core/pipeline/rawobs_detector_inventory.h>

#include <fmt/core.h>
#include <netcdf>
#include <tula/eigen.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <tuple>

namespace citlali::pipeline {

using RawObsToneFrequencies =
    std::map<Eigen::Index, Eigen::MatrixXd>;

template <class RawObs, class Logger>
RawObsToneFrequencies read_rawobs_tone_frequencies(
    const RawObs &rawobs, const Logger &logger) {
    using namespace netCDF;
    using namespace netCDF::exceptions;

    RawObsToneFrequencies tone_freqs;
    for (const typename RawObs::DataItem &data_item : rawobs.kidsdata()) {
        try {
            NcFile fo(data_item.filepath(), NcFile::read);
            const auto interface_id =
                rawobs_interface_id(data_item.interface());
            const Eigen::Index n_sweeps = static_cast<Eigen::Index>(
                fo.getVar("Header.Toltec.ToneFreq").getDim(0).getSize());
            const Eigen::Index n_tones = static_cast<Eigen::Index>(
                fo.getVar("Header.Toltec.ToneFreq").getDim(1).getSize());

            double lo_freq = 0.0;
            fo.getVar("Header.Toltec.LoCenterFreq").getVar(&lo_freq);

            tone_freqs[interface_id].resize(n_tones, n_sweeps);
            fo.getVar("Header.Toltec.ToneFreq")
                .getVar(tone_freqs[interface_id].data());
            tone_freqs[interface_id] =
                tone_freqs[interface_id].array() + lo_freq;

            fo.close();
        }
        catch (NcException &e) {
            logger->error("{}", e.what());
            throw ::DataIOError{fmt::format(
                "failed to load data from netCDF file {}",
                data_item.filepath())};
        }
    }
    return tone_freqs;
}

template <class Calib, class Logger>
void assign_tone_frequencies_by_network(
    Calib &calib, const RawObsToneFrequencies &tone_freqs,
    const Logger &logger) {
    calib.apt["tone_freq"].resize(calib.n_dets);

    for (const auto& [nw, limits] : calib.nw_limits) {
        auto it = tone_freqs.find(nw);
        if (it == tone_freqs.end()) {
            logger->error("missing tone freqs for nw {}", nw);
            throw citlali::error::io(
                fmt::format("missing tone frequencies for network {}", nw));
        }

        const auto& tf = it->second;
        const Eigen::Index n_tones = tf.rows();
        const Eigen::Index n_sweeps = tf.cols();
        const auto start = std::get<0>(limits);
        const auto end = std::get<1>(limits);
        const auto expected = end - start;

        if (n_sweeps < 1) {
            logger->error("no tone freq sweeps for nw {}", nw);
            throw citlali::error::io(
                fmt::format("no tone frequency sweeps for network {}", nw));
        }
        if (n_tones != expected) {
            logger->error(
                "tone freq size mismatch for nw {} (tones={}, expected dets={})",
                nw, n_tones, expected);
            throw citlali::error::io(fmt::format(
                "tone frequency size mismatch for network {}: tones={}, expected detectors={}",
                nw, n_tones, expected));
        }
        if (n_sweeps > 1) {
            logger->warn(
                "tone freqs have {} sweeps for nw {}, using first sweep",
                n_sweeps, nw);
        }

        calib.apt["tone_freq"].segment(start, expected) = tf.col(0);
    }
}

template <class Calib, class Logger>
int flag_duplicate_tones(Calib &calib, double delta_f_min_Hz,
                         const Logger &logger) {
    Eigen::VectorXd dfreq(calib.n_dets);
    dfreq(0) = calib.apt["tone_freq"](1) - calib.apt["tone_freq"](0);

    for (Eigen::Index i=1; i<calib.apt["tone_freq"].size()-1; ++i) {
        dfreq(i) = std::min(
            std::abs(calib.apt["tone_freq"](i) -
                     calib.apt["tone_freq"](i-1)),
            std::abs(calib.apt["tone_freq"](i+1) -
                     calib.apt["tone_freq"](i)));
    }
    dfreq(dfreq.size()-1) =
        std::abs(calib.apt["tone_freq"](dfreq.size()-1) -
                 calib.apt["tone_freq"](dfreq.size()-2));

    int n_nearby_tones = 0;
    calib.apt["duplicate_tone"].setZero(calib.n_dets);
    for (Eigen::Index i=0; i<calib.n_dets; ++i) {
        if (dfreq(i) < delta_f_min_Hz) {
            calib.apt["duplicate_tone"](i) = 1;
            ++n_nearby_tones;
        }
    }
    logger->info("{} nearby tones found. these will be flagged.",
                 n_nearby_tones);
    return n_nearby_tones;
}

}  // namespace citlali::pipeline
