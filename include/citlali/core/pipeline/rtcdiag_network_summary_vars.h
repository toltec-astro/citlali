#pragma once

// Included by rtcdiag_network_outputs.h inside namespace citlali::pipeline.

template <class AddInt, class AddDouble>
void add_rtcdiag_network_step_summary_diag(const AddInt &add_int,
                                           const AddDouble &add_double) {
    add_double("rtc_network_step_score_median",
               "median detector step score within each RTC network block");
    add_double("rtc_network_step_score_max",
               "maximum detector step score within each RTC network block");
    add_double("rtc_network_step_det_frac",
               "fraction of diagnostic-used detectors with strong step-like score in each RTC network block");
    add_double("rtc_network_step_alignment_frac",
               "fraction of strong-step detectors aligned in the dominant step-time cluster");
    add_int("rtc_network_step_dominant_sample",
            "dominant aligned step sample within each RTC network block; -2147483647 means unavailable");
}

template <class AddInt, class AddDouble>
void add_rtcdiag_network_impulsive_summary_diag(
    const AddInt &add_int, const AddDouble &add_double) {
    add_double("rtc_network_impulsive_score_median",
               "median detector impulsive-event score within each RTC network block");
    add_double("rtc_network_impulsive_score_max",
               "maximum detector impulsive-event score within each RTC network block");
    add_double("rtc_network_impulsive_det_frac",
               "fraction of diagnostic-used detectors with impulsive-event score above the impulsive coincidence threshold");
    add_double("rtc_network_impulsive_alignment_frac",
               "fraction of impulsive-active detectors aligned in the dominant impulsive time cluster");
    add_int("rtc_network_impulsive_dominant_sample",
            "dominant aligned impulsive sample within each RTC network block; -2147483647 means unavailable");
}

template <class AddDouble>
void add_rtcdiag_network_common_mode_diag(const AddDouble &add_double) {
    add_double("rtc_network_cm_low_mid_ratio",
               "low-band to mid-band common-mode power ratio for each RTC network block");
    add_double("rtc_network_cm_peak_freq_hz",
               "frequency of the strongest common-mode spectral peak for each RTC network block");
    add_double("rtc_network_cm_peak_prominence",
               "prominence of the strongest common-mode spectral peak for each RTC network block");
}

