# Citlali Reduction Product Comparison

- Mode: `point`
- Baseline: `/Users/gwilson/work_toltec/local_data/citlali-validation/v1/point/pointings/reduced/redu00`
- Candidate: `/private/tmp/citlali-sci-align-001-phase1-c77105b9b1676ec1ec74a9d560765954c5f1d5dd/point/point/reduced/redu00`
- Strict gate: `True`
- Common products: 12
- Changed records: 673
- Skipped records: 0

## Product Counts

| kind   | baseline | candidate |
| ------ | -------- | --------- |
| ecsv   | 2        | 1         |
| fits   | 6        | 3         |
| netcdf | 11       | 8         |

## Product Set Differences

Missing from candidate:
- `152389/filtered/ppt_commissioning_pointing_152389_filtered_citlali.ecsv`
- `152389/filtered/toltec_commissioning_a1100_pointing_152389_filtered_citlali.fits`
- `152389/filtered/toltec_commissioning_a1400_pointing_152389_filtered_citlali.fits`
- `152389/filtered/toltec_commissioning_a2000_pointing_152389_filtered_citlali.fits`
- `152389/filtered/toltec_commissioning_pointing_152389_hist_filtered.nc`
- `152389/filtered/toltec_commissioning_pointing_152389_mapdiag_filtered.nc`
- `152389/filtered/toltec_commissioning_pointing_152389_psd_filtered.nc`

## Largest Numeric Differences

| product                                                            | item                                                            | status           | finite | max abs     | med abs     | max frac    |
| ------------------------------------------------------------------ | --------------------------------------------------------------- | ---------------- | ------ | ----------- | ----------- | ----------- |
| 152389/raw/toltec_commissioning_a1100_pointing_152389_citlali.fits | HDU 4:noise_variance_I                                          | different        | 133875 | 83466.6     | 2.35849     | 3.2016e+16  |
| 152389/raw/toltec_commissioning_a1400_pointing_152389_citlali.fits | HDU 4:noise_variance_I                                          | different        | 133875 | 45863.4     | 0.271129    | 1.11928e+16 |
| 152389/raw/toltec_commissioning_a2000_pointing_152389_citlali.fits | HDU 4:noise_variance_I                                          | different        | 133875 | 35559.8     | 0.917449    | 9.73059e+15 |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_impulsive_event_sample                                      | different        | 66216  | 282         | 0           | 2.82e+14    |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_impulsive_peak_abs_sample                                   | different        | 66216  | 282         | 0           | 2.82e+14    |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_impulsive_event_sample                                      | different        | 66216  | 282         | 0           | 2.82e+14    |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_impulsive_peak_abs_sample                                   | different        | 66216  | 282         | 0           | 2.82e+14    |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_step_sample                                                 | different        | 66216  | 272         | 0           | 2.72e+14    |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_step_sample                                                 | different        | 66216  | 272         | 0           | 2.72e+14    |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_impulsive_peak_delta_abs_sample                             | different        | 66216  | 270         | 0           | 2.7e+14     |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_impulsive_peak_delta_abs_sample                             | different        | 66216  | 270         | 0           | 2.7e+14     |
| 152389/raw/toltec_commissioning_a1100_pointing_152389_citlali.fits | HDU 1:signal_I                                                  | different        | 133875 | 402.617     | 0.304719    | 1.82699e+14 |
| 152389/raw/toltec_commissioning_a1400_pointing_152389_citlali.fits | HDU 1:signal_I                                                  | different        | 133875 | 357.224     | 0.0715751   | 1.25634e+14 |
| 152389/raw/toltec_commissioning_a2000_pointing_152389_citlali.fits | HDU 1:signal_I                                                  | different        | 133875 | 458.745     | 0.239433    | 1.01759e+14 |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_network_impulsive_dominant_sample                           | different        | 132    | 143         | 0           | 1.8e+13     |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_network_impulsive_dominant_sample                           | different        | 132    | 143         | 0           | 1.8e+13     |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_impulsive_near_abs_count                                    | different        | 66216  | 122         | 0           | 1.4e+13     |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_impulsive_near_abs_count                                    | different        | 66216  | 122         | 0           | 1.4e+13     |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_invvar_window_n_total                                       | different        | 66216  | 10          | 0           | 1e+13       |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_invvar_window_n_valid                                       | different        | 66216  | 10          | 0           | 1e+13       |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_impulsive_near_delta_count                                  | different        | 66216  | 8           | 0           | 4e+12       |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_impulsive_near_delta_count                                  | different        | 66216  | 8           | 0           | 4e+12       |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_final_region_count                                          | different        | 66216  | 4           | 2           | 4e+12       |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_final_region_count                                          | different        | 66216  | 4           | 2           | 4e+12       |
| 152389/raw/toltec_commissioning_a2000_pointing_152389_citlali.fits | HDU 6:coverage_I                                                | different        | 133875 | 3.3936      | 0.00512518  | 3.3936e+12  |
| 152389/raw/toltec_commissioning_a1400_pointing_152389_citlali.fits | HDU 8:sig2noise_I                                               | different        | 133875 | 6.8147      | 0.0115034   | 2.50643e+12 |
| 152389/raw/toltec_commissioning_a1400_pointing_152389_citlali.fits | HDU 9:sig2noise_pixel_I                                         | different        | 133875 | 6.8147      | 0.0115034   | 2.50643e+12 |
| 152389/raw/toltec_commissioning_a2000_pointing_152389_citlali.fits | HDU 8:sig2noise_I                                               | different        | 133875 | 10.508      | 0.0379784   | 2.41625e+12 |
| 152389/raw/toltec_commissioning_a2000_pointing_152389_citlali.fits | HDU 9:sig2noise_pixel_I                                         | different        | 133875 | 10.508      | 0.0379784   | 2.41625e+12 |
| 152389/raw/toltec_commissioning_pointing_152389_hist.nc            | a1100_I_hist                                                    | different        | 50     | 118         | 0           | 2e+12       |
| 152389/raw/toltec_commissioning_a1100_pointing_152389_citlali.fits | HDU 8:sig2noise_I                                               | different        | 133875 | 7.34455     | 0.0205442   | 1.87302e+12 |
| 152389/raw/toltec_commissioning_a1100_pointing_152389_citlali.fits | HDU 9:sig2noise_pixel_I                                         | different        | 133875 | 7.34455     | 0.0205442   | 1.87302e+12 |
| 152389/raw/toltec_commissioning_a1100_pointing_152389_citlali.fits | HDU 7:coverage_bool_I                                           | different        | 133875 | 1           | 0           | 1e+12       |
| 152389/raw/toltec_commissioning_a1400_pointing_152389_citlali.fits | HDU 7:coverage_bool_I                                           | different        | 133875 | 1           | 0           | 1e+12       |
| 152389/raw/toltec_commissioning_a2000_pointing_152389_citlali.fits | HDU 7:coverage_bool_I                                           | different        | 133875 | 1           | 0           | 1e+12       |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_despike_local_raw_accepted_event_count                      | different        | 66216  | 1           | 0           | 1e+12       |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_impulsive_event_kind                                        | different        | 66216  | 1           | 0           | 1e+12       |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_network_impulsive_alignment_frac                            | different        | 132    | 1           | 0           | 1e+12       |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_despike_local_raw_accepted_event_count                      | different        | 66216  | 1           | 0           | 1e+12       |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_impulsive_event_kind                                        | different        | 66216  | 1           | 0           | 1e+12       |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_network_impulsive_alignment_frac                            | different        | 132    | 1           | 0           | 1e+12       |
| 152389/raw/toltec_commissioning_a1400_pointing_152389_citlali.fits | HDU 6:coverage_I                                                | different        | 133875 | 1.43633     | 1.33342e-08 | 6.00481e+11 |
| 152389/raw/toltec_commissioning_a1100_pointing_152389_citlali.fits | HDU 6:coverage_I                                                | different        | 133875 | 1.23184     | 0.00119985  | 5.00468e+11 |
| 152389/raw/toltec_commissioning_a2000_pointing_152389_citlali.fits | HDU 3:weight_formal_I                                           | different        | 133875 | 0.0136766   | 4.02558e-05 | 3.28975e+09 |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_network_impulsive_det_frac                                  | different        | 132    | 0.00475059  | 0           | 3.10559e+09 |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_network_impulsive_det_frac                                  | different        | 132    | 0.00475059  | 0           | 3.10559e+09 |
| 152389/raw/toltec_commissioning_a1400_pointing_152389_citlali.fits | HDU 5:kernel_I                                                  | different        | 133875 | 0.00937954  | 3.56093e-07 | 2.38409e+09 |
| 152389/raw/toltec_commissioning_a1400_pointing_152389_citlali.fits | HDU 3:weight_formal_I                                           | different        | 133875 | 0.0117542   | 1.29894e-07 | 1.64967e+09 |
| 152389/raw/toltec_commissioning_a2000_pointing_152389_citlali.fits | HDU 2:weight_I                                                  | different        | 133875 | 0.00557985  | 0.00017037  | 1.55781e+09 |
| 152389/raw/toltec_commissioning_a1400_pointing_152389_citlali.fits | HDU 2:weight_I                                                  | different        | 133875 | 0.00702493  | 5.39362e-05 | 9.58171e+08 |
| 152389/raw/toltec_commissioning_a2000_pointing_152389_citlali.fits | HDU 5:kernel_I                                                  | different        | 133875 | 0.00499441  | 2.87596e-06 | 9.32406e+08 |
| 152389/raw/toltec_commissioning_a1100_pointing_152389_citlali.fits | HDU 3:weight_formal_I                                           | different        | 133875 | 0.00370784  | 6.33618e-06 | 7.68432e+08 |
| 152389/raw/toltec_commissioning_a1100_pointing_152389_citlali.fits | HDU 2:weight_I                                                  | different        | 133875 | 0.000864178 | 1.13261e-05 | 1.85137e+08 |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_detector_median                                             | different        | 66216  | 101618      | 1.87583e-12 | 1.05956e+06 |
| 152389/raw/toltec_commissioning_pointing_152389_stats.nc           | median                                                          | different        | 66216  | 101618      | 1.87583e-12 | 1.05956e+06 |
| 152389/raw/toltec_commissioning_pointing_152389_psd.nc             | a1400_I_psd_2d                                                  | different        | 102785 | 5786.47     | 0.0379524   | 9745.95     |
| 152389/raw/toltec_commissioning_a1100_pointing_152389_citlali.fits | HDU 5:kernel_I                                                  | different        | 133875 | 0.00377885  | 1.07338e-09 | 2605.26     |
| 152389/raw/toltec_commissioning_pointing_152389_psd.nc             | a2000_I_psd_2d                                                  | different        | 115599 | 9245.44     | 0.000658749 | 2062.29     |
| 152389/raw/toltec_commissioning_pointing_152389_psd.nc             | a1100_I_psd_2d                                                  | different        | 106913 | 15255.8     | 0.0214131   | 1704.97     |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_detector_rms                                                | different        | 66216  | 101494      | 1.13687e-13 | 925.341     |
| 152389/raw/toltec_commissioning_pointing_152389_stats.nc           | rms                                                             | different        | 66216  | 101494      | 1.13687e-13 | 925.341     |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_network_cm_peak_freq_hz                                     | different        | 132    | 10.7952     | 0           | 51.1148     |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_network_cm_peak_freq_hz                                     | different        | 132    | 10.7952     | 0           | 51.1148     |
| 152389/raw/toltec_commissioning_pointing_152389_psd.nc             | a1400_I_noise_psd_2d                                            | different        | 102785 | 2375.69     | 0.0358352   | 36.7048     |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_invvar_window_q90                                           | different        | 56244  | 0.0114504   | 8.13152e-20 | 22.2361     |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_final_region_len_max                                        | different        | 66216  | 288         | 0           | 16.9412     |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_final_region_len_median                                     | different        | 66216  | 288         | 0           | 16.9412     |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_final_region_len_max                                        | different        | 66216  | 288         | 0           | 16.9412     |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_final_region_len_median                                     | different        | 66216  | 288         | 0           | 16.9412     |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_detector_stddev                                             | different        | 66216  | 15905.5     | 1.13687e-13 | 11.1916     |
| 152389/raw/toltec_commissioning_pointing_152389_stats.nc           | stddev                                                          | different        | 66216  | 15905.5     | 1.13687e-13 | 11.1916     |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_detector_flagged_fraction                                   | different        | 66216  | 0.888525    | 0           | 7.5         |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_final_flagged_frac                                          | different        | 66216  | 0.888525    | 0           | 7.5         |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_final_flagged_frac                                          | different        | 66216  | 0.888525    | 0           | 7.5         |
| 152389/raw/toltec_commissioning_pointing_152389_stats.nc           | flagged_frac                                                    | different        | 66216  | 0.888525    | 0           | 7.5         |
| 152389/raw/toltec_commissioning_pointing_152389_psd.nc             | a1100_I_noise_psd_2d                                            | different        | 106913 | 8238.68     | 0.0177638   | 6.75247     |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_invvar_window_q10                                           | different        | 53730  | 0.000500187 | 1.05032e-19 | 6.05419     |
| 152389/raw/toltec_commissioning_pointing_152389_hist.nc            | a1100_I_noise_hist                                              | different        | 50     | 659.2       | 0.5         | 5           |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_invvar_window_q10                                           | different        | 56244  | 0.000628167 | 1.69407e-20 | 4.962       |
| 152389/raw/toltec_commissioning_pointing_152389_psd.nc             | a2000_I_noise_psd_2d                                            | different        | 115599 | 1388.57     | 0.000463793 | 4.86374     |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_invvar_window_q90                                           | different        | 53730  | 0.00130588  | 3.998e-19   | 4.6897      |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_invvar_window_median                                        | different        | 56244  | 0.00347624  | 2.71051e-20 | 4.05076     |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_impulsive_event_score                                       | different        | 53730  | 25.6084     | 3.81917e-14 | 2.76971     |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_impulsive_event_score                                       | different        | 53730  | 25.6084     | 3.81917e-14 | 2.76971     |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_impulsive_peak_delta_abs_z                                  | different        | 53730  | 8.50204     | 3.81917e-14 | 2.76971     |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_impulsive_peak_delta_abs_z                                  | different        | 53730  | 8.50204     | 3.81917e-14 | 2.76971     |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_impulsive_peak_abs_z                                        | different        | 53730  | 25.6084     | 2.88658e-14 | 2.46554     |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_impulsive_peak_abs_z                                        | different        | 53730  | 25.6084     | 2.88658e-14 | 2.46554     |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_invvar_window_median                                        | different        | 53730  | 0.000356567 | 1.49078e-19 | 1.8409      |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_step_score                                                  | different        | 53730  | 17.8074     | 1.05471e-14 | 1.80134     |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_step_score                                                  | different        | 53730  | 17.8074     | 1.05471e-14 | 1.80134     |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_network_cm_low_mid_ratio                                    | different        | 132    | 51.1366     | 8.08242e-13 | 1.05346     |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_network_cm_low_mid_ratio                                    | different        | 132    | 51.1366     | 8.08242e-13 | 1.05346     |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_network_impulsive_score_max                                 | different        | 132    | 5.85381     | 7.72715e-14 | 1.02377     |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_network_impulsive_score_max                                 | different        | 132    | 5.85381     | 7.72715e-14 | 1.02377     |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_despike_added_region_len_max                                | different        | 66216  | 64          | 0           | 1           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_despike_local_exceed_count                                  | different        | 66216  | 64          | 0           | 1           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_despike_local_flagged_sample_count                          | different        | 66216  | 64          | 0           | 1           |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_despike_added_region_len_max                                | different        | 66216  | 64          | 0           | 1           |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_despike_local_exceed_count                                  | different        | 66216  | 64          | 0           | 1           |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_despike_local_flagged_sample_count                          | different        | 66216  | 64          | 0           | 1           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_despike_added_region_count                                  | different        | 66216  | 1           | 0           | 1           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_despike_local_raw_candidate_count                           | different        | 66216  | 1           | 0           | 1           |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_despike_added_region_count                                  | different        | 66216  | 1           | 0           | 1           |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_despike_local_raw_candidate_count                           | different        | 66216  | 1           | 0           | 1           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_invvar_window_flagged_frac_median                           | different        | 53730  | 0.274194    | 0           | 1           |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_invvar_window_flagged_frac_median                           | different        | 56244  | 0.274194    | 0           | 1           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_despike_added_flagged_frac                                  | different        | 56244  | 0.0995334   | 0           | 1           |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_despike_added_flagged_frac                                  | different        | 56244  | 0.0995334   | 0           | 1           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | ptc_second_pass_max_unflagged_residual_uid                      | different        | 132    | 404         | 0           | 0.955224    |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_second_pass_max_unflagged_residual_uid                      | different        | 132    | 404         | 0           | 0.955224    |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_network_step_score_max                                      | different        | 132    | 17.1985     | 3.35287e-14 | 0.838109    |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_network_step_score_max                                      | different        | 132    | 17.1985     | 3.35287e-14 | 0.838109    |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | ptc_second_pass_top_candidate_cluster_sample                    | different        | 132    | 179         | 0           | 0.733607    |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | ptc_second_pass_top_event_sample                                | different        | 132    | 179         | 0           | 0.733607    |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_second_pass_top_candidate_cluster_sample                    | different        | 132    | 179         | 0           | 0.733607    |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_second_pass_top_event_sample                                | different        | 132    | 179         | 0           | 0.733607    |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_network_step_det_frac                                       | different        | 132    | 0.0455635   | 0           | 0.710336    |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_network_step_det_frac                                       | different        | 132    | 0.0455635   | 0           | 0.710336    |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_despike_max_local_abs_z                                     | different        | 56244  | 1.99264     | 0           | 0.696962    |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_despike_max_local_abs_z                                     | different        | 56244  | 1.99264     | 0           | 0.696962    |
| 152389/raw/toltec_commissioning_pointing_152389_hist.nc            | a1400_I_hist                                                    | different        | 50     | 329         | 0           | 0.5         |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | ptc_second_pass_n_accepted_clusters                             | different        | 132    | 1           | 0           | 0.5         |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | ptc_second_pass_n_accepted_events                               | different        | 132    | 1           | 0           | 0.5         |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | ptc_second_pass_n_candidate_clusters                            | different        | 132    | 1           | 0           | 0.5         |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | ptc_second_pass_n_candidate_events                              | different        | 132    | 1           | 0           | 0.5         |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_second_pass_n_accepted_clusters                             | different        | 132    | 1           | 0           | 0.5         |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_second_pass_n_accepted_events                               | different        | 132    | 1           | 0           | 0.5         |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_second_pass_n_candidate_clusters                            | different        | 132    | 1           | 0           | 0.5         |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_second_pass_n_candidate_events                              | different        | 132    | 1           | 0           | 0.5         |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_despike_local_raw_reject_count                              | different        | 66216  | 1           | 0           | 0.5         |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_despike_local_raw_reject_count                              | different        | 66216  | 1           | 0           | 0.5         |
| 152389/raw/toltec_commissioning_pointing_152389_hist.nc            | a2000_I_noise_hist                                              | different        | 50     | 430.6       | 1.2         | 0.433333    |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_network_cm_peak_prominence                                  | different        | 132    | 101.413     | 7.87992e-12 | 0.420235    |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_network_cm_peak_prominence                                  | different        | 132    | 101.413     | 7.87992e-12 | 0.420235    |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_network_step_alignment_frac                                 | different        | 132    | 0.2         | 0           | 0.4         |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_network_step_alignment_frac                                 | different        | 132    | 0.2         | 0           | 0.4         |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | ptc_second_pass_top_event_uid                                   | different        | 132    | 278         | 0           | 0.389902    |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_second_pass_top_event_uid                                   | different        | 132    | 278         | 0           | 0.389902    |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_invvar_window_flagged_frac_max                              | different        | 53730  | 0.346154    | 0           | 0.346154    |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_invvar_window_flagged_frac_max                              | different        | 56244  | 0.346154    | 0           | 0.346154    |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_invvar_window_heavy_flagged_fraction                        | different        | 53730  | 0.1         | 0           | 0.333333    |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_invvar_window_heavy_flagged_fraction                        | different        | 56244  | 0.1         | 0           | 0.333333    |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_despike_max_local_delta_abs_z                               | different        | 56244  | 1.41913     | 0           | 0.324817    |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_despike_max_local_delta_abs_z                               | different        | 56244  | 1.41913     | 0           | 0.324817    |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_despike_added_region_len_median                             | different        | 1438   | 20          | 0           | 0.3125      |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_despike_added_region_len_median                             | different        | 1438   | 20          | 0           | 0.3125      |
| 152389/raw/toltec_commissioning_pointing_152389_hist.nc            | a2000_I_hist                                                    | different        | 50     | 193         | 0.5         | 0.285714    |
| 152389/raw/toltec_commissioning_pointing_152389_psd.nc             | a1400_I_noise_psd                                               | different        | 170    | 73.7777     | 0.00832395  | 0.263509    |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | ptc_second_pass_newly_flagged_fraction                          | different        | 132    | 1.30957e-05 | 0           | 0.263388    |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | ptc_second_pass_proposed_flagged_fraction                       | different        | 132    | 1.30957e-05 | 0           | 0.263388    |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_second_pass_newly_flagged_fraction                          | different        | 132    | 1.30957e-05 | 0           | 0.263388    |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_second_pass_proposed_flagged_fraction                       | different        | 132    | 1.30957e-05 | 0           | 0.263388    |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_invvar_window_n_valid                                       | different        | 66216  | 2           | 0           | 0.25        |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_invvar_window_valid_fraction                                | different        | 53730  | 0.2         | 0           | 0.25        |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_invvar_window_valid_fraction                                | different        | 56244  | 0.2         | 0           | 0.25        |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | ptc_second_pass_max_unflagged_residual_z                        | different        | 132    | 1.77111     | 1.94511e-13 | 0.239713    |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_second_pass_max_unflagged_residual_z                        | different        | 132    | 1.77111     | 1.94511e-13 | 0.239713    |
| 152389/raw/toltec_commissioning_pointing_152389_psd.nc             | a1400_I_psd                                                     | different        | 170    | 944.325     | 0.0799583   | 0.233604    |
| 152389/raw/toltec_commissioning_pointing_152389_hist.nc            | a1400_I_noise_hist                                              | different        | 50     | 122.8       | 0.6         | 0.2         |
| 152389/raw/ppt_commissioning_pointing_152389_citlali.ecsv          | y_t                                                             | different        | 3      | 0.0274618   | 0.00834692  | 0.134853    |
| 152389/raw/toltec_commissioning_pointing_152389_psd.nc             | a2000_I_psd                                                     | different        | 172    | 1320.85     | 0.00102797  | 0.13063     |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | ptc_second_pass_top_candidate_cluster_peak_score                | different        | 31     | 1.27872     | 3.05533e-13 | 0.122231    |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | ptc_second_pass_top_event_score                                 | different        | 31     | 1.27872     | 3.05533e-13 | 0.122231    |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_second_pass_top_candidate_cluster_peak_score                | different        | 31     | 1.27872     | 3.05533e-13 | 0.122231    |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_second_pass_top_event_score                                 | different        | 31     | 1.27872     | 3.05533e-13 | 0.122231    |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | scan_speed_altaz_p50_arcsec_s                                   | different        | 12     | 2.48503     | 4.82145e-09 | 0.119723    |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_network_step_dominant_sample                                | different        | 132    | 27          | 0           | 0.105882    |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_network_step_dominant_sample                                | different        | 132    | 27          | 0           | 0.105882    |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_noise_sig2noise_skew                                        | different        | 3      | 0.621454    | 0.437769    | 0.068617    |
| 152389/raw/toltec_commissioning_pointing_152389_psd.nc             | a2000_I_noise_psd                                               | different        | 172    | 208.193     | 0.000928977 | 0.0633769   |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_network_step_score_median                                   | different        | 132    | 0.0873195   | 9.76996e-15 | 0.0604236   |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_network_step_score_median                                   | different        | 132    | 0.0873195   | 9.76996e-15 | 0.0604236   |
| 152389/raw/toltec_commissioning_pointing_152389_psd.nc             | a1100_I_psd                                                     | different        | 167    | 1108.52     | 0.282311    | 0.0600264   |
| 152389/raw/toltec_commissioning_pointing_152389_psd.nc             | a1100_I_noise_psd                                               | different        | 167    | 576.008     | 0.511196    | 0.0588728   |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | scan_duration_s                                                 | different        | 12     | 0.270336    | 0           | 0.0572917   |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | raw_scan_indices                                                | different        | 48     | 32          | 16          | 0.0555556   |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | scan_indices                                                    | different        | 24     | 32          | 16          | 0.0555556   |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | raw_scan_indices                                                | different        | 48     | 32          | 16          | 0.0555556   |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | scan_indices                                                    | different        | 24     | 32          | 16          | 0.0555556   |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | tod_filter_edge_guard_flagged_frac                              | different        | 12     | 0.00618803  | 0           | 0.0540984   |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | tod_filter_edge_guard_flagged_frac                              | different        | 12     | 0.00618803  | 0           | 0.0540984   |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_core_tail_fraction_neg_lt3                                  | different        | 3      | 0.000694204 | 0.000357203 | 0.0452097   |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_core_tail_excess_neg_lt3                                    | different        | 3      | 0.514264    | 0.264615    | 0.0452097   |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_noise_tail_fraction_pos_gt3                                 | different        | 3      | 0.0008669   | 0.000220065 | 0.0445728   |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_noise_tail_excess_pos_gt3                                   | different        | 3      | 0.642196    | 0.163023    | 0.0445728   |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_core_sig2noise_skew                                         | different        | 3      | 28.3471     | 16.9047     | 0.0395403   |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | ptc_second_pass_existing_flagged_fraction                       | different        | 132    | 0.00886075  | 0           | 0.035354    |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_second_pass_existing_flagged_fraction                       | different        | 132    | 0.00886075  | 0           | 0.035354    |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_core_tail_fraction_abs_gt3                                  | different        | 3      | 0.000894619 | 0.000356414 | 0.0341583   |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_core_tail_excess_abs_gt3                                    | different        | 3      | 0.331365    | 0.132015    | 0.0341583   |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_network_impulsive_score_median                              | different        | 132    | 0.0677921   | 3.4639e-14  | 0.0231928   |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_network_impulsive_score_median                              | different        | 132    | 0.0677921   | 3.4639e-14  | 0.0231928   |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_noise_rms_p16                                               | different        | 3      | 0.621026    | 0.120384    | 0.0217922   |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_median_rms                                                  | different        | 3      | 0.548783    | 0.146547    | 0.0191913   |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_noise_rms_p84                                               | different        | 3      | 0.554891    | 0.0610342   | 0.0190704   |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_core_tail_fraction_pos_gt3                                  | different        | 3      | 0.000200414 | 0.000110941 | 0.0184966   |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_core_tail_excess_pos_gt3                                    | different        | 3      | 0.148466    | 0.0821846   | 0.0184966   |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | scan_tod_lowpass_to_source_power_half_ratio                     | different        | 36     | 0.0910313   | 2.02194e-11 | 0.0176852   |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | scan_source_power_half_bandwidth_hz                             | different        | 36     | 0.097886    | 2.72409e-11 | 0.0173778   |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | scan_speed_altaz_p995_arcsec_s                                  | different        | 12     | 2.12313     | 7.57225e-10 | 0.0173778   |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_noise_weight_median_ratio                                   | different        | 3      | 0.0355802   | 0.0215607   | 0.0171372   |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_noise_weight_scale                                          | different        | 3      | 0.00811502  | 0.00521373  | 0.0168484   |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_empirical_to_formal_noise_ratio                             | different        | 3      | 0.0333064   | 0.0221955   | 0.0158682   |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_noise_tail_fraction_neg_lt3                                 | different        | 3      | 0.00032776  | 0.000197221 | 0.0156671   |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_noise_tail_excess_neg_lt3                                   | different        | 3      | 0.242803    | 0.146101    | 0.0156671   |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_noise_tail_fraction_abs_gt3                                 | different        | 3      | 0.000613276 | 0.000546452 | 0.0149994   |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_noise_tail_excess_abs_gt3                                   | different        | 3      | 0.227156    | 0.202405    | 0.0149994   |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | scan_speed_altaz_p95_arcsec_s                                   | different        | 12     | 1.49983     | 3.32376e-09 | 0.0147697   |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_core_coverage_median                                        | different        | 3      | 0.10823     | 0.0937965   | 0.0114593   |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_noise_products_s2n_sigma                                    | different        | 3      | 0.0430469   | 0.00884438  | 0.0113019   |
| 152389/raw/ppt_commissioning_pointing_152389_citlali.ecsv          | a_fwhm_err                                                      | different        | 3      | 0.000372998 | 0.000348959 | 0.0112354   |
| 152389/raw/ppt_commissioning_pointing_152389_citlali.ecsv          | x_t_err                                                         | different        | 3      | 0.00014433  | 0.000141641 | 0.0109196   |
| 152389/raw/ppt_commissioning_pointing_152389_citlali.ecsv          | amp_err                                                         | different        | 3      | 0.03701     | 0.034636    | 0.0105147   |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_coverage_sum                                                | different        | 3      | 10413.2     | 8091.43     | 0.010215    |
| 152389/raw/ppt_commissioning_pointing_152389_citlali.ecsv          | b_fwhm_err                                                      | different        | 3      | 0.000384379 | 0.000358529 | 0.00928991  |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_coverage_max                                                | different        | 3      | 0.254345    | 0.0977398   | 0.00854978  |
| 152389/raw/ppt_commissioning_pointing_152389_citlali.ecsv          | y_t_err                                                         | different        | 3      | 0.000153599 | 0.00015022  | 0.00845438  |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_core_peak_abs_sig2noise                                     | different        | 3      | 1.53649     | 1.45971     | 0.00791267  |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_peak_abs_sig2noise                                          | different        | 3      | 1.53649     | 1.45971     | 0.00791267  |
| 152389/raw/ppt_commissioning_pointing_152389_citlali.ecsv          | sig2noise                                                       | different        | 3      | 0.286541    | 0.073801    | 0.00789876  |
| 152389/raw/toltec_commissioning_pointing_152389_hist.nc            | a1400_I_bins                                                    | different        | 50     | 6.58637     | 3.36039     | 0.00762126  |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_peak_signal                                                 | different        | 3      | 6.58637     | 2.96944     | 0.00762126  |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_network_impulsive_n_det_used                                | different        | 132    | 3           | 0           | 0.00728155  |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_network_n_det_used                                          | different        | 132    | 3           | 0           | 0.00728155  |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_network_impulsive_n_det_used                                | different        | 132    | 3           | 0           | 0.00728155  |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_network_n_det_used                                          | different        | 132    | 3           | 0           | 0.00728155  |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_weight_threshold                                            | different        | 3      | 3.79885e-05 | 7.06905e-06 | 0.00725037  |
| 152389/raw/ppt_commissioning_pointing_152389_citlali.ecsv          | x_t                                                             | different        | 3      | 0.0321651   | 0.0234551   | 0.00671676  |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | coadd_obs_core_weight_sum                                       | different        | 3      | 17.6997     | 2.42944     | 0.0062727   |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_core_weight_sum                                             | different        | 3      | 17.6997     | 2.42944     | 0.0062727   |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | coadd_obs_weight_sum                                            | different        | 3      | 17.8926     | 2.50358     | 0.00621985  |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_weight_sum                                                  | different        | 3      | 17.8926     | 2.50358     | 0.00621985  |
| 152389/raw/ppt_commissioning_pointing_152389_citlali.ecsv          | amp                                                             | different        | 3      | 3.70099     | 3.16418     | 0.00512879  |
| 152389/raw/toltec_commissioning_pointing_152389_hist.nc            | a2000_I_bins                                                    | different        | 50     | 2.96944     | 1.51502     | 0.00444811  |
| 152389/raw/ppt_commissioning_pointing_152389_citlali.ecsv          | a_fwhm                                                          | different        | 3      | 0.0216012   | 0.0194721   | 0.00393012  |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_median_err                                                  | different        | 3      | 0.0460025   | 0.0155512   | 0.00337663  |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | OOF_RMS_a1100                                                   | different        | 1      | 0.0460025   | 0.0460025   | 0.00337663  |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | OOF_RMS_a1100                                                   | different        | 1      | 0.0460025   | 0.0460025   | 0.00337663  |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | OOF_RMS_a2000                                                   | different        | 1      | 0.0155512   | 0.0155512   | 0.00260672  |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | OOF_RMS_a2000                                                   | different        | 1      | 0.0155512   | 0.0155512   | 0.00260672  |
| 152389/raw/ppt_commissioning_pointing_152389_citlali.ecsv          | b_fwhm                                                          | different        | 3      | 0.012536    | 0.00924492  | 0.00141571  |
| 152389/raw/toltec_commissioning_pointing_152389_hist.nc            | a1100_I_bins                                                    | different        | 50     | 1.34139     | 0.684383    | 0.00110216  |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | coadd_obs_n_core_pixels                                         | different        | 3      | 91          | 72          | 0.0010922   |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_n_core_pixels                                               | different        | 3      | 91          | 72          | 0.0010922   |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_noise_products_valid_pixels                                 | different        | 3      | 91          | 72          | 0.0010922   |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | OOF_RMS_a1400                                                   | different        | 1      | 0.00679414  | 0.00679414  | 0.000936545 |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | OOF_RMS_a1400                                                   | different        | 1      | 0.00679414  | 0.00679414  | 0.000936545 |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | coadd_obs_n_valid_pixels                                        | different        | 3      | 52          | 33          | 0.000476561 |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_n_valid_pixels                                              | different        | 3      | 52          | 33          | 0.000476561 |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | MEAN_AZ                                                         | different        | 1      | 7.31859e-13 | 7.31859e-13 | 1.74785e-14 |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | MEAN_AZ                                                         | different        | 1      | 7.31859e-13 | 7.31859e-13 | 1.74785e-14 |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | MEAN_PA                                                         | different        | 1      | 7.67386e-13 | 7.67386e-13 | 6.13396e-15 |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | MEAN_PA                                                         | different        | 1      | 7.67386e-13 | 7.67386e-13 | 6.13396e-15 |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | MEAN_EL                                                         | different        | 1      | 1.13687e-13 | 1.13687e-13 | 1.91986e-15 |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | MEAN_EL                                                         | different        | 1      | 1.13687e-13 | 1.13687e-13 | 1.91986e-15 |
| 152389/raw/toltec_commissioning_pointing_152389_psd.nc             | a1400_I_noise_psd_freq                                          | different        | 170    | 1.45519e-11 | 0           | 2.21821e-16 |
| 152389/raw/toltec_commissioning_pointing_152389_psd.nc             | a1400_I_psd_freq                                                | different        | 170    | 1.45519e-11 | 0           | 2.21821e-16 |
| 152389/raw/toltec_commissioning_pointing_152389_psd.nc             | a2000_I_noise_psd_freq                                          | different        | 172    | 1.45519e-11 | 0           | 2.15428e-16 |
| 152389/raw/toltec_commissioning_pointing_152389_psd.nc             | a2000_I_psd_freq                                                | different        | 172    | 1.45519e-11 | 0           | 2.15428e-16 |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | weights                                                         | different        | 66216  | 6.77626e-21 | 0           | 1.8737e-16  |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_detector_weight                                             | different        | 66216  | 6.77626e-21 | 0           | 1.8737e-16  |
| 152389/raw/toltec_commissioning_pointing_152389_stats.nc           | weights                                                         | different        | 66216  | 6.77626e-21 | 0           | 1.8737e-16  |
| 152389/raw/toltec_commissioning_pointing_152389_psd.nc             | a1100_I_noise_psd_freq                                          | different        | 167    | 1.45519e-11 | 0           | 1.45935e-16 |
| 152389/raw/toltec_commissioning_pointing_152389_psd.nc             | a1100_I_psd_freq                                                | different        | 167    | 1.45519e-11 | 0           | 1.45935e-16 |
| 152389/raw/ppt_commissioning_pointing_152389_citlali.ecsv          | angle                                                           | within_tolerance | 3      | 0           | 0           | 0           |
| 152389/raw/ppt_commissioning_pointing_152389_citlali.ecsv          | angle_err                                                       | within_tolerance | 3      | 0           | 0           | 0           |
| 152389/raw/ppt_commissioning_pointing_152389_citlali.ecsv          | array                                                           | within_tolerance | 3      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | MAP_COVERAGE_CUT                                                | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | MAP_EDGE_GUARD_ENABLED                                          | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | MAP_EDGE_GUARD_HITS_CORE_FRACTION                               | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | MAP_EDGE_GUARD_RADIUS_FWHM                                      | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | MAP_EDGE_GUARD_TAPER_MIN_FRACTION                               | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | MAP_PIXEL_SIZE_RAD                                              | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | coadd_obs_core_weight_frac                                      | within_tolerance | 3      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | coadd_obs_weight_frac                                           | within_tolerance | 3      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_edge_guard_applied                                          | within_tolerance | 3      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_edge_guard_guardband_npix                                   | within_tolerance | 3      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_edge_guard_science_npix                                     | within_tolerance | 3      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_edge_guard_support_npix                                     | within_tolerance | 3      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_edge_guard_support_radius_pix                               | within_tolerance | 3      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_peak_col                                                    | within_tolerance | 3      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_peak_row                                                    | within_tolerance | 3      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | obsnum                                                          | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_psd.nc             | a1100_I_noise_psd_2d_freq                                       | within_tolerance | 106913 | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_psd.nc             | a1100_I_psd_2d_freq                                             | within_tolerance | 106913 | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_psd.nc             | a1400_I_noise_psd_2d_freq                                       | within_tolerance | 102785 | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_psd.nc             | a1400_I_psd_2d_freq                                             | within_tolerance | 102785 | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_psd.nc             | a2000_I_noise_psd_2d_freq                                       | within_tolerance | 115599 | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_psd.nc             | a2000_I_psd_2d_freq                                             | within_tolerance | 115599 | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | ActGalAng                                                       | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | ActParAng                                                       | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | BMAJ_a1100                                                      | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | BMAJ_a1400                                                      | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | BMAJ_a2000                                                      | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | BMIN_a1100                                                      | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | BMIN_a1400                                                      | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | BMIN_a2000                                                      | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | BPA_a1100                                                       | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | BPA_a1400                                                       | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | BPA_a2000                                                       | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.CALIBRATED                                               | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.CLEANED                                                  | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.CLEANED.ADAPT.CLIP_Z                                     | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.CLEANED.ADAPT.ENABLED                                    | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.CLEANED.ADAPT.LOG_CANDIDATES                             | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.CLEANED.ADAPT.LOWMAX_HZ                                  | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.CLEANED.ADAPT.LOWMIN_HZ                                  | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.CLEANED.ADAPT.LOW_WEIGHT                                 | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.CLEANED.ADAPT.MAX_DET                                    | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.CLEANED.ADAPT.MAX_PAIRS                                  | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.CLEANED.ADAPT.MAX_SAMPLES                                | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.CLEANED.ADAPT.MIDMAX_HZ                                  | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.CLEANED.ADAPT.MIDMIN_HZ                                  | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.CLEANED.ADAPT.MIN_GOOD_FRAC                              | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.CLEANED.ADAPT.REG_WEIGHT                                 | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.CLEANED.ADAPT.TAIL_WEIGHT                                | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.CLEANED.ADAPT.TOPMODE_WEIGHT                             | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.CLEANED.MP.BANDHIGH_HZ                                   | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.CLEANED.MP.BANDLOW_HZ                                    | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.CLEANED.MP.ENABLED                                       | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.CLEANED.MP.MAXMODES                                      | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.CLEANED.NEIG_a1100                                       | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.CLEANED.NEIG_a1400                                       | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.CLEANED.NEIG_a2000                                       | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.DESPIKE.LOCAL.DELTA_GATE.ENABLED                         | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.DESPIKE.LOCAL.DELTA_GATE.HALF_PEAK_FRAC                  | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.DESPIKE.LOCAL.DELTA_GATE.MAX_STEP_SHIFT_Z                | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.DESPIKE.LOCAL.DELTA_GATE.MAX_WIDTH_SEC                   | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.DESPIKE.LOCAL.DELTA_GATE.WINDOW_SEC                      | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.DESPIKE.LOCAL.DELTA_SIGMA_SCALE                          | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.DESPIKE.LOCAL.ENABLED                                    | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.DESPIKE.LOCAL.EVENT_PADDING_SEC                          | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.DESPIKE.LOCAL.EXPAND_WITH_FILTER                         | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.DESPIKE.LOCAL.HIGH_SCORE_EVENT_OVERRIDE                  | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.DESPIKE.LOCAL.MAX_ADDED_FLAGGED_FRAC                     | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.DESPIKE.LOCAL.RAW_GATE.CAND_REL_SIGMA_SCALE              | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.DESPIKE.LOCAL.RAW_GATE.CAND_SIGMA_SCALE                  | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.DESPIKE.LOCAL.RAW_GATE.ENABLED                           | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.DESPIKE.LOCAL.RAW_GATE.HALF_PEAK_FRAC                    | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.DESPIKE.LOCAL.RAW_GATE.MAX_STEP_SHIFT_Z                  | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.DESPIKE.LOCAL.RAW_GATE.MAX_WIDTH_SEC                     | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.DESPIKE.LOCAL.RAW_GATE.WINDOW_SEC                        | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.DESPIKE.LOCAL.SIGMA_SCALE                                | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.DESPIKE.LOCAL.WINDOW_SEC                                 | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.DESPIKED                                                 | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.DOWNSAMPLED                                              | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.EXTINCTION                                               | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.FRUITLOOPS                                               | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.FRUITLOOPS.ADAPT_SUPPORT_FWHM                            | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.FRUITLOOPS.ADAPT_SUPPORT_RAD                             | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.FRUITLOOPS.FLUX_a1100                                    | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.FRUITLOOPS.FLUX_a1400                                    | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.FRUITLOOPS.FLUX_a2000                                    | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.FRUITLOOPS.LOCALSIG_EDGE                                 | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.FRUITLOOPS.LOCALSIG_INNER                                | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.FRUITLOOPS.LOCALSIG_MINPIX                               | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.FRUITLOOPS.LOCALSIG_OUTER                                | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.FRUITLOOPS.LOCALSNR                                      | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.FRUITLOOPS.MAXITER                                       | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.FRUITLOOPS.PEAKFRAC                                      | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.FRUITLOOPS.S2N                                           | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.FRUITLOOPS.WEIGHT_FEEDBACK.ENABLED                       | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.FRUITLOOPS.WEIGHT_FEEDBACK.HIGH_RELATIVE_WEIGHT          | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.FRUITLOOPS.WEIGHT_FEEDBACK.LOW_RELATIVE_WEIGHT           | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.INV_VAR.PTC.WTHIGH                                       | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.INV_VAR.PTC.WTLOW                                        | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.INV_VAR.RTC.WTHIGH                                       | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.INV_VAR.RTC.WTLOW                                        | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.LEARNING.APPLY_START_ITER                                | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.LEARNING.BUSY_DETECTOR_EXCLUSION_ENABLED                 | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.LEARNING.DIAGNOSTICS_ENABLED                             | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.LEARNING.ENABLED                                         | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.LEARNING.LEARN_ITERS                                     | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.LEARNING.MAP_PIXEL_OUTLIER_DETECTOR_EXCLUSION_ENABLED    | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.LEARNING.MAP_PIXEL_OUTLIER_DETECTOR_EXCLUSION_MIN_PIXELS | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.LEARNING.MAX_RECORDS_PER_TYPE                            | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.LEARNING.SCAN_NETWORK_PATHOLOGY_APPLY_PRE_MAPMAKING      | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.LEARNING.SCAN_NETWORK_PATHOLOGY_APPLY_PRE_PTC            | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.LEARNING.SCAN_NETWORK_PATHOLOGY_APPLY_PRE_RTC            | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.LEARNING.SCAN_NETWORK_PATHOLOGY_ENABLED                  | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.LEARNING.SCAN_NETWORK_PATHOLOGY_MAX_NEW_FLAGGED_FRAC     | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.LEARNING.SCAN_NETWORK_PATHOLOGY_MIN_CLUSTERS             | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.LEARNING.SCAN_NETWORK_PATHOLOGY_MIN_EVENTS               | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.LEARNING.SCAN_NETWORK_PATHOLOGY_MIN_RESID_Z              | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.LEARNING.SCAN_NETWORK_PATHOLOGY_SEVERE_EVENTS            | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.LEARNING.SCAN_NETWORK_PATHOLOGY_SEVERE_RESID_Z           | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.POLARIZED                                                | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.PTC.SECOND_PASS.BASELINE_WINDOW_SEC                      | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.PTC.SECOND_PASS.CLUSTER_EVENTS_SEC                       | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.PTC.SECOND_PASS.DELTA_HALF_PEAK_FRAC                     | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.PTC.SECOND_PASS.DELTA_MAX_WIDTH_SEC                      | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.PTC.SECOND_PASS.DELTA_SIGMA_SCALE                        | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.PTC.SECOND_PASS.DELTA_WINDOW_SEC                         | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.PTC.SECOND_PASS.ENABLED                                  | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.PTC.SECOND_PASS.HIGH_SCORE_CLUSTER_OVERRIDE              | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.PTC.SECOND_PASS.HIGH_SCORE_EVENT_OVERRIDE                | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.PTC.SECOND_PASS.MAX_AUTO_FLAG_CLUSTERS                   | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.PTC.SECOND_PASS.MAX_STEP_SHIFT_Z                         | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.PTC.SECOND_PASS.MERGE_WITHIN_DET_SEC                     | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.PTC.SECOND_PASS.MIN_CLUSTER_DETECTORS                    | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.PTC.SECOND_PASS.MIN_GOOD_FRAC                            | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.PTC.SECOND_PASS.MIN_SPIKE_SIGMA                          | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.PTC.SECOND_PASS.RAW_CAND_REL_SIGMA_SCALE                 | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.PTC.SECOND_PASS.RAW_HALF_PEAK_FRAC                       | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.PTC.SECOND_PASS.RAW_MAX_WIDTH_SEC                        | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.PTC.SECOND_PASS.RAW_WINDOW_SEC                           | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.PTC.SECOND_PASS.SIGMA_SCALE                              | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.IMPULSIVE.ENABLED                                    | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.IMPULSIVE.MAX_EVENTS                                 | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.IMPULSIVE.MIN_EVENT_Z                                | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.IMPULSIVE.MIN_GOOD_FRAC                              | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.IMPULSIVE.NEAR_EVENT_Z                               | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.IMPULSIVE.POST_WINDOW_SEC                            | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.IMPULSIVE.PRE_WINDOW_SEC                             | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.IMPULSIVE_COINCIDENCE.CLUSTER_TOL_SEC                | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.IMPULSIVE_COINCIDENCE.ENABLED                        | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.IMPULSIVE_COINCIDENCE.EVENT_SCORE_THRESH             | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.IMPULSIVE_COINCIDENCE.HIGH_SCORE_MIN_NETWORKS        | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.IMPULSIVE_COINCIDENCE.HIGH_SCORE_OVERRIDE_THRESH     | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.IMPULSIVE_COINCIDENCE.MAX_FLAGGED_FRAC               | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.IMPULSIVE_COINCIDENCE.MIN_ALIGNMENT_FRAC             | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.IMPULSIVE_COINCIDENCE.MIN_DET_FRAC                   | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.IMPULSIVE_COINCIDENCE.MIN_DET_USED                   | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.IMPULSIVE_COINCIDENCE.MIN_GOOD_FRAC                  | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.IMPULSIVE_COINCIDENCE.MIN_NETWORKS_ALIGNED           | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.IMPULSIVE_COINCIDENCE.POST_WINDOW_SEC                | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.IMPULSIVE_COINCIDENCE.PRE_WINDOW_SEC                 | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.LINE_AUDIT.APPLY_CLUSTER_TOL_HZ                      | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.LINE_AUDIT.APPLY_MAX_NOTCHES                         | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.LINE_AUDIT.APPLY_MAX_WIDTH_HZ                        | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.LINE_AUDIT.APPLY_MIN_CM_PROMINENCE                   | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.LINE_AUDIT.APPLY_MIN_DETECTOR_FRAC                   | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.LINE_AUDIT.APPLY_MIN_SUPPORT_NETWORKS                | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.LINE_AUDIT.APPLY_MIN_WIDTH_HZ                        | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.LINE_AUDIT.APPLY_SHARED_NOTCHES                      | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.LINE_AUDIT.APPLY_WIDTH_SCALE                         | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.LINE_AUDIT.BAD_DETECTOR_MAX_CLUSTER_FRAC             | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.LINE_AUDIT.CLUSTER_TOL_HZ                            | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.LINE_AUDIT.CM_PROMINENCE_THRESH                      | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.LINE_AUDIT.CONTINUUM_RADIUS_BINS                     | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.LINE_AUDIT.DETECTOR_MIN_LINE_POWER_FRAC              | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.LINE_AUDIT.DETECTOR_MIN_PROMINENCE                   | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.LINE_AUDIT.DETECTOR_NOTCH_CONTEXT_SAMPLES            | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.LINE_AUDIT.DETECTOR_NOTCH_MAX_NOTCHES                | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.LINE_AUDIT.DETECTOR_NOTCH_MAX_WIDTH_HZ               | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.LINE_AUDIT.DETECTOR_NOTCH_MIN_LINE_POWER_FRAC        | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.LINE_AUDIT.DETECTOR_NOTCH_MIN_PROMINENCE             | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.LINE_AUDIT.DETECTOR_NOTCH_MIN_WIDTH_HZ               | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.LINE_AUDIT.DETECTOR_NOTCH_WIDTH_SCALE                | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.LINE_AUDIT.ENABLED                                   | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.LINE_AUDIT.FIXED_NOTCH_COUNT                         | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.LINE_AUDIT.FIXED_NOTCH_ENABLED                       | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.LINE_AUDIT.FIXED_NOTCH_EXCLUSION_HALF_WIDTH_HZ       | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.LINE_AUDIT.FIXED_NOTCH_WIDTH_COUNT                   | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.LINE_AUDIT.LINE_MAX_HZ                               | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.LINE_AUDIT.LINE_MIN_HZ                               | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.LINE_AUDIT.MAX_DET                                   | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.LINE_AUDIT.MAX_PEAKS_PER_DETECTOR                    | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.LINE_AUDIT.MIN_DET_FOR_NETWORK                       | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.LINE_AUDIT.MIN_GOOD_FRAC                             | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.LINE_AUDIT.MIN_SEGMENT_SEC                           | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.LINE_AUDIT.MIN_WINDOWS                               | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.LINE_AUDIT.NOTCH_MIN_CM_PROMINENCE                   | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.LINE_AUDIT.NOTCH_MIN_DETECTORS                       | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.LINE_AUDIT.NOTCH_MIN_DETECTOR_FRAC                   | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.LINE_AUDIT.OVERLAP_FRAC                              | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.LINE_AUDIT.POST_FILTER_APPLY_DETECTOR_NOTCHES        | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.LINE_AUDIT.POST_FILTER_APPLY_ITERATIONS              | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.LINE_AUDIT.POST_FILTER_APPLY_SHARED_NOTCHES          | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.LINE_AUDIT.POST_FILTER_ENABLED                       | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.LINE_AUDIT.PRE_FILTER_ENABLED                        | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.LINE_AUDIT.PROMINENCE_THRESH                         | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.LINE_AUDIT.PTC_APPLY_DETECTOR_NOTCHES                | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.LINE_AUDIT.PTC_APPLY_FIXED_NOTCHES                   | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.LINE_AUDIT.PTC_APPLY_ITERATIONS                      | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.LINE_AUDIT.PTC_APPLY_SHARED_NOTCHES                  | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.LINE_AUDIT.PTC_MODEL_PROTECTED_ENABLED               | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.LINE_AUDIT.PTC_REQUIRE_MODEL_SUBTRACTED              | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.LINE_AUDIT.SEGMENT_SEC                               | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.STEP_MASK.CLUSTER_TOL_SEC                            | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.STEP_MASK.ENABLED                                    | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.STEP_MASK.HALF_WIDTH_SEC                             | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.STEP_MASK.MAX_FLAGGED_FRAC                           | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.STEP_MASK.MIN_ALIGNMENT_FRAC                         | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.STEP_MASK.MIN_DET_USED                               | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.STEP_MASK.MIN_GOOD_FRAC                              | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.STEP_MASK.MIN_STEP_DET_FRAC                          | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.STEP_MASK.STEP_SCORE_THRESH                          | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.RTC.STEP_MASK.STEP_WINDOW_SEC                            | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.TOD.OUTER_CONTEXT_SAMPLES                                | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.TOD.OUTPUT_OUTER_CONTEXT_SAMPLES                         | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.TODFILTER.EDGE_GUARD.CONTEXT_SAMPLES                     | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.TODFILTER.EDGE_GUARD.ENABLED                             | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.TODFILTER.EDGE_GUARD.EXTRA_SAMPLES                       | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.TODFILTER.EDGE_GUARD.GUARD_SAMPLES                       | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.TODFILTER.EDGE_GUARD.IIR_SETTLE_ATTENUATION              | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.TODFILTER.EDGE_GUARD.MAX_SAMPLES                         | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.TODFILTER.EDGE_GUARD.MIN_SAMPLES                         | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.TODFILTERED                                              | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.TODIIRHP                                                 | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.TODIIRHP.FREQ_HZ                                         | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.TODIIRHP.ORDER                                           | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.TODIIRHP.ZEROPHASE                                       | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.TODNOTCH                                                 | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.VERBOSE                                                  | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.WEIGHT.BUSY_ROW_SUPPRESS.ENABLED                         | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.WEIGHT.BUSY_ROW_SUPPRESS.FACTOR                          | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.WEIGHT.BUSY_ROW_SUPPRESS.MIN_CAND_CLUSTERS               | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.WEIGHT.BUSY_ROW_SUPPRESS.MIN_MAX_RESID_Z                 | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.WEIGHT.BUSY_ROW_SUPPRESS.REQUIRE_BUSY_VETO               | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.WEIGHT.CORR_PENALTY.CM_EL.ENABLED                        | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.WEIGHT.CORR_PENALTY.CM_EL.REF                            | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.WEIGHT.CORR_PENALTY.CM_EL.SPAN                           | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.WEIGHT.CORR_PENALTY.CM_EL.WEIGHT                         | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.WEIGHT.CORR_PENALTY.ENABLED                              | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.WEIGHT.CORR_PENALTY.EXPONENT                             | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.WEIGHT.CORR_PENALTY.FLOOR                                | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.WEIGHT.CORR_PENALTY.LOWMID.ENABLED                       | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.WEIGHT.CORR_PENALTY.LOWMID.LOWMAX_HZ                     | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.WEIGHT.CORR_PENALTY.LOWMID.LOWMIN_HZ                     | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.WEIGHT.CORR_PENALTY.LOWMID.MIDMAX_HZ                     | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.WEIGHT.CORR_PENALTY.LOWMID.MIDMIN_HZ                     | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.WEIGHT.CORR_PENALTY.LOWMID.REF                           | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.WEIGHT.CORR_PENALTY.LOWMID.SPAN                          | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.WEIGHT.CORR_PENALTY.LOWMID.WEIGHT                        | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.WEIGHT.CORR_PENALTY.MAX_PAIRS                            | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.WEIGHT.CORR_PENALTY.MAX_SAMPLES                          | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.WEIGHT.CORR_PENALTY.MIN_GOOD_FRAC                        | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.WEIGHT.CORR_PENALTY.MIN_OVERLAP                          | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.WEIGHT.CORR_PENALTY.PAIR.ENABLED                         | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.WEIGHT.CORR_PENALTY.PAIR.REF                             | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.WEIGHT.CORR_PENALTY.PAIR.SPAN                            | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.WEIGHT.CORR_PENALTY.PAIR.WEIGHT                          | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.WEIGHT.HYBRID_MAX_FACTOR                                 | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.WEIGHT.HYBRID_MIN_FACTOR                                 | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.WEIGHT.MEDWTFACTOR                                       | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.WEIGHT.PTC.WTHIGH                                        | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.WEIGHT.PTC.WTLOW                                         | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.WEIGHT.SOURCE_MASK_RADIUS_ARCSEC                         | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.WEIGHT.VALIDATION.ACCUMULATION_ITERS                     | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.WEIGHT.VALIDATION.APPLY_START_ITER                       | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.WEIGHT.VALIDATION.ENABLED                                | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.WEIGHT.VALIDATION.MIN_FACTOR                             | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.WEIGHT.VALIDATION.UPWARD_ENABLED                         | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.WEIGHT.VALIDATION.UPWARD_MAX_FACTOR                      | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.WEIGHT.VALIDATION.UPWARD_MIN_ATM_FACTOR                  | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.WEIGHT.VALIDATION.UPWARD_MIN_BASE_FACTOR                 | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.WEIGHT.VALIDATION.UPWARD_POWER                           | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.WEIGHT.VALIDATION.UPWARD_REQUIRE_ATM                     | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | EXPTIME                                                         | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | FRUITLOOPS_ITER                                                 | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | HWPR                                                            | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Dcs.CalMode                                              | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Dcs.IntegrationTime                                      | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Dcs.ObsMode                                              | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Dcs.ObsNum                                               | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Dcs.ObsType                                              | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Dcs.RequestedTime                                        | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Dcs.ScanNum                                              | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Dcs.SubObsNum                                            | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Gps.IgnoreLock                                           | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Lissajous.ExecMode                                       | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Lissajous.ScanRate                                       | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Lissajous.TScan                                          | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Lissajous.XDelta                                         | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Lissajous.XDeltaMinor                                    | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Lissajous.XLength                                        | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Lissajous.XLengthMinor                                   | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Lissajous.XOmega                                         | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Lissajous.XOmegaMinor                                    | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Lissajous.XOmegaMinorNorm                                | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Lissajous.XOmegaNorm                                     | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Lissajous.YLength                                        | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Lissajous.YLengthMinor                                   | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Lissajous.YOmega                                         | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Lissajous.YOmegaMinor                                    | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Lissajous.YOmegaMinorNorm                                | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Lissajous.YOmegaNorm                                     | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M1.ActPos                                                | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M1.CmdPos                                                | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M1.ModelEnabled                                          | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M1.ModelMode                                             | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M1.ZernikeC                                              | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M1.ZernikeEnabled                                        | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.AcuHeartbeat                                          | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.Alive                                                 | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.AzPcor                                                | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.CorEnabled                                            | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.ElCmd                                                 | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.ElPcor                                                | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.Follow                                                | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.Hold                                                  | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.M2Heartbeat                                           | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.ModelMode                                             | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.TiltAct                                               | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.TiltCmd                                               | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.TiltDes                                               | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.TiltPcor                                              | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.TiltReq                                               | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.TipAct                                                | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.TipCmd                                                | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.TipDes                                                | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.TipPcor                                               | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.TipReq                                                | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.XAct                                                  | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.XCmd                                                  | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.XDes                                                  | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.XPcor                                                 | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.XReq                                                  | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.YAct                                                  | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.YCmd                                                  | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.YDes                                                  | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.YPcor                                                 | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.YReq                                                  | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.ZAct                                                  | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.ZCmd                                                  | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.ZDes                                                  | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.ZPcor                                                 | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.ZReq                                                  | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M3.AcuHeartbeat                                          | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M3.Alive                                                 | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M3.ElDesEnabled                                          | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M3.Fault                                                 | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M3.M3Heartbeat                                           | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M3.M3OffPos                                              | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.PointModel.AzM2Cor                                       | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.PointModel.AzPaddleOff                                   | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.PointModel.AzPointModelCor                               | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.PointModel.AzReceiverCor                                 | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.PointModel.AzReceiverOff                                 | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.PointModel.AzTiltCor                                     | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.PointModel.AzTotalCor                                    | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.PointModel.AzUserOff                                     | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.PointModel.ElM2Cor                                       | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.PointModel.ElPaddleOff                                   | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.PointModel.ElPointModelCor                               | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.PointModel.ElReceiverCor                                 | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.PointModel.ElReceiverOff                                 | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.PointModel.ElRefracCor                                   | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.PointModel.ElTiltCor                                     | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.PointModel.ElTotalCor                                    | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.PointModel.ElUserOff                                     | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.PointModel.M2CorEnabled                                  | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.PointModel.ModRev                                        | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.PointModel.PointModelCorEnabled                          | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.PointModel.ReceiverOffEnabled                            | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.PointModel.RefracCorEnabled                              | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.PointModel.TiltCorEnabled                                | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Radiometer.Tau                                           | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Radiometer.Tau2                                          | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.ScanFile.Valid                                           | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Sky.BaryVel                                              | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Sky.ObsVel                                               | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Sky.ParAng                                               | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Sky.RaOffsetSys                                          | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Source.B                                                 | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Source.CoordSys                                          | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Source.Dec                                               | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Source.DecProperMotionCor                                | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Source.ElObsMax                                          | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Source.ElObsMin                                          | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Source.Epoch                                             | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Source.L                                                 | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Source.Planet                                            | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Source.Ra                                                | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Source.RaProperMotionCor                                 | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Source.VelSys                                            | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Source.Velocity                                          | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Telescope.AzActPos                                       | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Telescope.AzDesPos                                       | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Telescope.CraneInBeam                                    | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Telescope.ElActPos                                       | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Telescope.ElDesPos                                       | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Telescope.PointingTolerance                              | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.TelescopeBackend.CalObsNum                               | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.TelescopeBackend.Master                                  | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.TelescopeBackend.NumPixels                               | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.TelescopeBackend.ObsNum                                  | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.TelescopeBackend.ScanNum                                 | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.TelescopeBackend.SubObsNum                               | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Tiltmeter_0_.Temp                                        | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Tiltmeter_0_.TiltX                                       | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Tiltmeter_0_.TiltY                                       | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Tiltmeter_1_.Temp                                        | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Tiltmeter_1_.TiltX                                       | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Tiltmeter_1_.TiltY                                       | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.TimePlace.LST                                            | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.TimePlace.ObsElevation                                   | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.TimePlace.ObsLatitude                                    | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.TimePlace.ObsLongitude                                   | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.TimePlace.UT1                                            | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.TimePlace.UTDate                                         | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Toltec.AzPointCor                                        | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Toltec.AzPointOff                                        | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Toltec.BeamSelected                                      | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Toltec.ElPointCor                                        | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Toltec.ElPointOff                                        | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Toltec.M3Dir                                             | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Toltec.NumBands                                          | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Toltec.NumBeams                                          | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Toltec.NumPixels                                         | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Toltec.Remote                                            | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Weather.Humidity                                         | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Weather.Precipitation                                    | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Weather.Pressure                                         | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Weather.Radiation                                        | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Weather.Temperature                                      | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Weather.TimeOfDay                                        | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Weather.WindDir1                                         | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Weather.WindDir2                                         | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Weather.WindSpeed1                                       | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Weather.WindSpeed2                                       | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Hold                                                            | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | JINC_A_a1100                                                    | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | JINC_A_a1400                                                    | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | JINC_A_a2000                                                    | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | JINC_B_a1100                                                    | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | JINC_B_a1400                                                    | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | JINC_B_a2000                                                    | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | JINC_C_a1100                                                    | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | JINC_C_a1400                                                    | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | JINC_C_a2000                                                    | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | JINC_R                                                          | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | MEAN_TAU_a1100                                                  | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | MEAN_TAU_a1400                                                  | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | MEAN_TAU_a2000                                                  | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | OOF_ID_a1100                                                    | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | OOF_ID_a1400                                                    | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | OOF_ID_a2000                                                    | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | OOF_M2X                                                         | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | OOF_M2Y                                                         | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | OOF_M2Z                                                         | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | OOF_RI                                                          | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | OOF_RO                                                          | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | OOF_T                                                           | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | OOF_W_a1100                                                     | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | OOF_W_a1400                                                     | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | OOF_W_a2000                                                     | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | SAMPRATE                                                        | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | SourceAz                                                        | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | SourceDec                                                       | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | SourceEl                                                        | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | SourceRa                                                        | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | TAN_DEC                                                         | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | TAN_RA                                                          | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | TelAzAct                                                        | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | TelAzCor                                                        | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | TelAzDes                                                        | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | TelAzMap                                                        | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | TelB                                                            | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | TelDec                                                          | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | TelElAct                                                        | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | TelElCor                                                        | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | TelElDes                                                        | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | TelElMap                                                        | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | TelL                                                            | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | TelRa                                                           | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | TelTime                                                         | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | TelUTC                                                          | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | alt_phys                                                        | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | apt_a_fwhm                                                      | within_tolerance | 5518   | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | apt_a_fwhm_err                                                  | within_tolerance | 5518   | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | apt_amp                                                         | within_tolerance | 5518   | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | apt_amp_err                                                     | within_tolerance | 5518   | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | apt_angle                                                       | within_tolerance | 5518   | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | apt_angle_err                                                   | within_tolerance | 5518   | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | apt_array                                                       | within_tolerance | 5518   | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | apt_b_fwhm                                                      | within_tolerance | 5518   | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | apt_b_fwhm_err                                                  | within_tolerance | 5518   | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | apt_converge_iter                                               | within_tolerance | 5518   | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | apt_derot_elev                                                  | within_tolerance | 5518   | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | apt_duplicate_tone                                              | within_tolerance | 5518   | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | apt_fg                                                          | within_tolerance | 5518   | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | apt_flag                                                        | within_tolerance | 5518   | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | apt_flxscale                                                    | within_tolerance | 5518   | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | apt_loc                                                         | within_tolerance | 5518   | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | apt_nw                                                          | within_tolerance | 5518   | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | apt_ori                                                         | within_tolerance | 5518   | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | apt_pg                                                          | within_tolerance | 5518   | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | apt_responsivity                                                | within_tolerance | 5518   | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | apt_sens                                                        | within_tolerance | 5518   | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | apt_sig2noise                                                   | within_tolerance | 5518   | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | apt_tone_freq                                                   | within_tolerance | 5518   | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | apt_uid                                                         | within_tolerance | 5518   | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | apt_x_t                                                         | within_tolerance | 5518   | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | apt_x_t_derot                                                   | within_tolerance | 5518   | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | apt_x_t_err                                                     | within_tolerance | 5518   | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | apt_x_t_raw                                                     | within_tolerance | 5518   | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | apt_y_t                                                         | within_tolerance | 5518   | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | apt_y_t_derot                                                   | within_tolerance | 5518   | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | apt_y_t_err                                                     | within_tolerance | 5518   | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | apt_y_t_raw                                                     | within_tolerance | 5518   | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | az_phys                                                         | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | b_phys                                                          | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | dec_phys                                                        | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | flags                                                           | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | l_phys                                                          | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | lat_phys                                                        | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | lon_phys                                                        | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | obsnum                                                          | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | output_scan_index                                               | within_tolerance | 12     | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | pointing_offset_alt                                             | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | pointing_offset_az                                              | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | ptc_second_pass_added_flag                                      | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | ptc_second_pass_busy_network_vetoed                             | within_tolerance | 132    | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | ptc_second_pass_n_det_with_added_flags                          | within_tolerance | 132    | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | ptc_second_pass_n_rejected_clusters                             | within_tolerance | 132    | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | ptc_second_pass_n_rejected_events                               | within_tolerance | 132    | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | ptc_second_pass_n_source_protected_clusters                     | within_tolerance | 132    | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | ptc_second_pass_n_source_protected_events                       | within_tolerance | 132    | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | ptc_second_pass_network_ids                                     | within_tolerance | 11     | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | ptc_second_pass_top_candidate_cluster_n_detectors               | within_tolerance | 132    | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | ptc_second_pass_top_candidate_cluster_n_events                  | within_tolerance | 132    | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | ptc_second_pass_top_event_kind                                  | within_tolerance | 132    | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | ra_phys                                                         | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | signal                                                          | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | to_Jy_pixel_a1100                                               | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | to_Jy_pixel_a1400                                               | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | to_Jy_pixel_a2000                                               | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | to_MJy_sr_a1100                                                 | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | to_MJy_sr_a1400                                                 | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | to_MJy_sr_a2000                                                 | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | to_mJy_beam_a1100                                               | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | to_mJy_beam_a1400                                               | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | to_mJy_beam_a2000                                               | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | to_uK_a1100                                                     | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | to_uK_a1400                                                     | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | to_uK_a2000                                                     | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | tod_filter_edge_guard_flagged_samples                           | within_tolerance | 12     | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | tod_filter_edge_guard_post_samples                              | within_tolerance | 12     | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | tod_filter_edge_guard_pre_samples                               | within_tolerance | 12     | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.CLEANED                                                  | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.CLEANED.ADAPT.ENABLED                                    | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.FRUITLOOPS                                               | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.FRUITLOOPS.ADAPT_SUPPORT_FWHM                            | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.FRUITLOOPS.ADAPT_SUPPORT_RAD                             | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.FRUITLOOPS.LOCALSIG_EDGE                                 | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.FRUITLOOPS.LOCALSIG_INNER                                | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.FRUITLOOPS.LOCALSIG_MINPIX                               | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.FRUITLOOPS.LOCALSIG_OUTER                                | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.FRUITLOOPS.LOCALSNR                                      | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.FRUITLOOPS.PEAKFRAC                                      | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.FRUITLOOPS.S2N                                           | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.FRUITLOOPS.WEIGHT_FEEDBACK.ENABLED                       | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.FRUITLOOPS.WEIGHT_FEEDBACK.HIGH_RELATIVE_WEIGHT          | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.FRUITLOOPS.WEIGHT_FEEDBACK.LOW_RELATIVE_WEIGHT           | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.INV_VAR.PTC.WTHIGH                                       | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.INV_VAR.PTC.WTLOW                                        | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.INV_VAR.WINDOW_SEC                                       | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.LEARNING.APPLY_START_ITER                                | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.LEARNING.BUSY_DETECTOR_EXCLUSION_ENABLED                 | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.LEARNING.DIAGNOSTICS_ENABLED                             | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.LEARNING.ENABLED                                         | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.LEARNING.LEARN_ITERS                                     | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.LEARNING.MAP_PIXEL_OUTLIER_DETECTOR_EXCLUSION_ENABLED    | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.LEARNING.MAP_PIXEL_OUTLIER_DETECTOR_EXCLUSION_MIN_PIXELS | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.LEARNING.MAX_RECORDS_PER_TYPE                            | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.LEARNING.SCAN_NETWORK_PATHOLOGY_APPLY_PRE_MAPMAKING      | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.LEARNING.SCAN_NETWORK_PATHOLOGY_APPLY_PRE_PTC            | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.LEARNING.SCAN_NETWORK_PATHOLOGY_APPLY_PRE_RTC            | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.LEARNING.SCAN_NETWORK_PATHOLOGY_ENABLED                  | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.LEARNING.SCAN_NETWORK_PATHOLOGY_MAX_NEW_FLAGGED_FRAC     | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.LEARNING.SCAN_NETWORK_PATHOLOGY_MIN_CLUSTERS             | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.LEARNING.SCAN_NETWORK_PATHOLOGY_MIN_EVENTS               | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.LEARNING.SCAN_NETWORK_PATHOLOGY_MIN_RESID_Z              | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.LEARNING.SCAN_NETWORK_PATHOLOGY_SEVERE_EVENTS            | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.LEARNING.SCAN_NETWORK_PATHOLOGY_SEVERE_RESID_Z           | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.PTC.SECOND_PASS.ENABLED                                  | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.PTC.SECOND_PASS.HIGH_SCORE_EVENT_OVERRIDE                | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.PTC.SECOND_PASS.MAX_AUTO_FLAG_CLUSTERS                   | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.PTC.SECOND_PASS.MIN_CLUSTER_DETECTORS                    | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.PTC.SECOND_PASS.MIN_SPIKE_SIGMA                          | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.WEIGHT.BUSY_ROW_SUPPRESS.ENABLED                         | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.WEIGHT.CORR_PENALTY.ENABLED                              | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.WEIGHT.HYBRID_MAX_FACTOR                                 | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.WEIGHT.HYBRID_MIN_FACTOR                                 | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.WEIGHT.MEDWTFACTOR                                       | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.WEIGHT.PTC.WTHIGH                                        | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.WEIGHT.PTC.WTLOW                                         | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.WEIGHT.SOURCE_MASK_RADIUS_ARCSEC                         | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.WEIGHT.VALIDATION.ACCUMULATION_ITERS                     | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.WEIGHT.VALIDATION.APPLY_START_ITER                       | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.WEIGHT.VALIDATION.ENABLED                                | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.WEIGHT.VALIDATION.MIN_FACTOR                             | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.WEIGHT.VALIDATION.UPWARD_ENABLED                         | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.WEIGHT.VALIDATION.UPWARD_MAX_FACTOR                      | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.WEIGHT.VALIDATION.UPWARD_MIN_ATM_FACTOR                  | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.WEIGHT.VALIDATION.UPWARD_MIN_BASE_FACTOR                 | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.WEIGHT.VALIDATION.UPWARD_POWER                           | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.WEIGHT.VALIDATION.UPWARD_REQUIRE_ATM                     | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | SAMPRATE                                                        | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | SourceDec                                                       | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | SourceRa                                                        | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | adaptive_pca_baseline_k                                         | within_tolerance | 132    | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | adaptive_pca_chosen_k                                           | within_tolerance | 132    | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | adaptive_pca_n_candidates                                       | within_tolerance | 132    | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | adaptive_pca_n_det_input                                        | within_tolerance | 132    | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | adaptive_pca_n_det_used                                         | within_tolerance | 132    | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | adaptive_pca_n_time_used                                        | within_tolerance | 132    | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | adaptive_pca_network_ids                                        | within_tolerance | 11     | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | adaptive_pca_runnerup_k                                         | within_tolerance | 132    | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | adaptive_pca_sample_step                                        | within_tolerance | 132    | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | adaptive_pca_selector_fallback                                  | within_tolerance | 132    | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | adaptive_pca_selector_used                                      | within_tolerance | 132    | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | corr_nw_n_det_candidates                                        | within_tolerance | 132    | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | corr_nw_n_det_grouped                                           | within_tolerance | 132    | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | corr_nw_n_det_input                                             | within_tolerance | 132    | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | corr_nw_n_det_ungrouped                                         | within_tolerance | 132    | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | corr_nw_n_det_used                                              | within_tolerance | 132    | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | corr_nw_n_groups                                                | within_tolerance | 132    | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | corr_nw_n_groups_raw                                            | within_tolerance | 132    | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | corr_nw_network_ids                                             | within_tolerance | 11     | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | corr_nw_sample_step                                             | within_tolerance | 132    | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | obsnum                                                          | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | output_scan_index                                               | within_tolerance | 12     | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_diag_apt_flag                                               | within_tolerance | 5518   | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_diag_array                                                  | within_tolerance | 5518   | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_diag_network                                                | within_tolerance | 5518   | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_diag_uid                                                    | within_tolerance | 5518   | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_second_pass_busy_network_vetoed                             | within_tolerance | 132    | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_second_pass_n_det_with_added_flags                          | within_tolerance | 132    | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_second_pass_n_rejected_clusters                             | within_tolerance | 132    | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_second_pass_n_rejected_events                               | within_tolerance | 132    | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_second_pass_n_source_protected_clusters                     | within_tolerance | 132    | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_second_pass_n_source_protected_events                       | within_tolerance | 132    | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_second_pass_network_ids                                     | within_tolerance | 11     | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_second_pass_top_candidate_cluster_n_detectors               | within_tolerance | 132    | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_second_pass_top_candidate_cluster_n_events                  | within_tolerance | 132    | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_second_pass_top_event_kind                                  | within_tolerance | 132    | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | weight_busy_row_suppression_applied                             | within_tolerance | 132    | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | weight_busy_row_suppression_busy_network_vetoed                 | within_tolerance | 132    | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | weight_busy_row_suppression_n_candidate_clusters                | within_tolerance | 132    | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | weight_busy_row_suppression_n_det_weighted                      | within_tolerance | 132    | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | weight_busy_row_suppression_network_ids                         | within_tolerance | 11     | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | weight_corr_penalty_n_det_candidates                            | within_tolerance | 132    | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | weight_corr_penalty_n_det_input                                 | within_tolerance | 132    | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | weight_corr_penalty_n_det_used                                  | within_tolerance | 132    | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | weight_corr_penalty_n_det_weighted                              | within_tolerance | 132    | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | weight_corr_penalty_network_ids                                 | within_tolerance | 11     | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | weight_corr_penalty_sample_step                                 | within_tolerance | 132    | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | ActGalAng                                                       | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | ActParAng                                                       | shape_changed    |        |             |             |             |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | BMAJ_a1100                                                      | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | BMAJ_a1400                                                      | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | BMAJ_a2000                                                      | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | BMIN_a1100                                                      | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | BMIN_a1400                                                      | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | BMIN_a2000                                                      | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | BPA_a1100                                                       | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | BPA_a1400                                                       | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | BPA_a2000                                                       | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | CONFIG.CALIBRATED                                               | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | CONFIG.CLEANED                                                  | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | CONFIG.CLEANED.ADAPT.CLIP_Z                                     | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | CONFIG.CLEANED.ADAPT.ENABLED                                    | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | CONFIG.CLEANED.ADAPT.LOG_CANDIDATES                             | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | CONFIG.CLEANED.ADAPT.LOWMAX_HZ                                  | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | CONFIG.CLEANED.ADAPT.LOWMIN_HZ                                  | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | CONFIG.CLEANED.ADAPT.LOW_WEIGHT                                 | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | CONFIG.CLEANED.ADAPT.MAX_DET                                    | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | CONFIG.CLEANED.ADAPT.MAX_PAIRS                                  | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | CONFIG.CLEANED.ADAPT.MAX_SAMPLES                                | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | CONFIG.CLEANED.ADAPT.MIDMAX_HZ                                  | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | CONFIG.CLEANED.ADAPT.MIDMIN_HZ                                  | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | CONFIG.CLEANED.ADAPT.MIN_GOOD_FRAC                              | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | CONFIG.CLEANED.ADAPT.REG_WEIGHT                                 | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | CONFIG.CLEANED.ADAPT.TAIL_WEIGHT                                | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | CONFIG.CLEANED.ADAPT.TOPMODE_WEIGHT                             | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | CONFIG.CLEANED.MP.BANDHIGH_HZ                                   | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | CONFIG.CLEANED.MP.BANDLOW_HZ                                    | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | CONFIG.CLEANED.MP.ENABLED                                       | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | CONFIG.CLEANED.MP.MAXMODES                                      | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | CONFIG.CLEANED.NEIG_a1100                                       | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | CONFIG.CLEANED.NEIG_a1400                                       | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | CONFIG.CLEANED.NEIG_a2000                                       | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | CONFIG.DESPIKE.LOCAL.DELTA_GATE.ENABLED                         | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | CONFIG.DESPIKE.LOCAL.DELTA_GATE.HALF_PEAK_FRAC                  | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | CONFIG.DESPIKE.LOCAL.DELTA_GATE.MAX_STEP_SHIFT_Z                | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | CONFIG.DESPIKE.LOCAL.DELTA_GATE.MAX_WIDTH_SEC                   | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | CONFIG.DESPIKE.LOCAL.DELTA_GATE.WINDOW_SEC                      | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | CONFIG.DESPIKE.LOCAL.DELTA_SIGMA_SCALE                          | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | CONFIG.DESPIKE.LOCAL.ENABLED                                    | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | CONFIG.DESPIKE.LOCAL.EVENT_PADDING_SEC                          | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | CONFIG.DESPIKE.LOCAL.EXPAND_WITH_FILTER                         | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | CONFIG.DESPIKE.LOCAL.HIGH_SCORE_EVENT_OVERRIDE                  | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | CONFIG.DESPIKE.LOCAL.MAX_ADDED_FLAGGED_FRAC                     | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | CONFIG.DESPIKE.LOCAL.RAW_GATE.CAND_REL_SIGMA_SCALE              | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | CONFIG.DESPIKE.LOCAL.RAW_GATE.CAND_SIGMA_SCALE                  | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | CONFIG.DESPIKE.LOCAL.RAW_GATE.ENABLED                           | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | CONFIG.DESPIKE.LOCAL.RAW_GATE.HALF_PEAK_FRAC                    | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | CONFIG.DESPIKE.LOCAL.RAW_GATE.MAX_STEP_SHIFT_Z                  | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | CONFIG.DESPIKE.LOCAL.RAW_GATE.MAX_WIDTH_SEC                     | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | CONFIG.DESPIKE.LOCAL.RAW_GATE.WINDOW_SEC                        | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | CONFIG.DESPIKE.LOCAL.SIGMA_SCALE                                | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | CONFIG.DESPIKE.LOCAL.WINDOW_SEC                                 | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | CONFIG.DESPIKED                                                 | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | CONFIG.DOWNSAMPLED                                              | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | CONFIG.EXTINCTION                                               | within_tolerance | 1      | 0           | 0           | 0           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | CONFIG.FRUITLOOPS                                               | within_tolerance | 1      | 0           | 0           | 0           |

## Non-Tolerance Changes

| product                                                            | item                                                            | status             | detail                               |
| ------------------------------------------------------------------ | --------------------------------------------------------------- | ------------------ | ------------------------------------ |
| 152389/raw/ppt_commissioning_pointing_152389_citlali.ecsv          | fit_sig2noise                                                   | extra_column       |                                      |
| 152389/raw/ppt_commissioning_pointing_152389_citlali.ecsv          | peak_over_full_map_rms                                          | extra_column       |                                      |
| 152389/raw/ppt_commissioning_pointing_152389_citlali.ecsv          | a_fwhm                                                          | different          | [3] -> [3]                           |
| 152389/raw/ppt_commissioning_pointing_152389_citlali.ecsv          | a_fwhm_err                                                      | different          | [3] -> [3]                           |
| 152389/raw/ppt_commissioning_pointing_152389_citlali.ecsv          | amp                                                             | different          | [3] -> [3]                           |
| 152389/raw/ppt_commissioning_pointing_152389_citlali.ecsv          | amp_err                                                         | different          | [3] -> [3]                           |
| 152389/raw/ppt_commissioning_pointing_152389_citlali.ecsv          | b_fwhm                                                          | different          | [3] -> [3]                           |
| 152389/raw/ppt_commissioning_pointing_152389_citlali.ecsv          | b_fwhm_err                                                      | different          | [3] -> [3]                           |
| 152389/raw/ppt_commissioning_pointing_152389_citlali.ecsv          | sig2noise                                                       | different          | [3] -> [3]                           |
| 152389/raw/ppt_commissioning_pointing_152389_citlali.ecsv          | x_t                                                             | different          | [3] -> [3]                           |
| 152389/raw/ppt_commissioning_pointing_152389_citlali.ecsv          | x_t_err                                                         | different          | [3] -> [3]                           |
| 152389/raw/ppt_commissioning_pointing_152389_citlali.ecsv          | y_t                                                             | different          | [3] -> [3]                           |
| 152389/raw/ppt_commissioning_pointing_152389_citlali.ecsv          | y_t_err                                                         | different          | [3] -> [3]                           |
| 152389/raw/toltec_commissioning_a1100_pointing_152389_citlali.fits | HDU 1:signal_I                                                  | different          | [1, 1, 375, 357] -> [1, 1, 375, 357] |
| 152389/raw/toltec_commissioning_a1100_pointing_152389_citlali.fits | HDU 2:weight_I                                                  | different          | [1, 1, 375, 357] -> [1, 1, 375, 357] |
| 152389/raw/toltec_commissioning_a1100_pointing_152389_citlali.fits | HDU 3:weight_formal_I                                           | different          | [1, 1, 375, 357] -> [1, 1, 375, 357] |
| 152389/raw/toltec_commissioning_a1100_pointing_152389_citlali.fits | HDU 4:noise_variance_I                                          | different          | [1, 1, 375, 357] -> [1, 1, 375, 357] |
| 152389/raw/toltec_commissioning_a1100_pointing_152389_citlali.fits | HDU 5:kernel_I                                                  | different          | [1, 1, 375, 357] -> [1, 1, 375, 357] |
| 152389/raw/toltec_commissioning_a1100_pointing_152389_citlali.fits | HDU 6:coverage_I                                                | different          | [1, 1, 375, 357] -> [1, 1, 375, 357] |
| 152389/raw/toltec_commissioning_a1100_pointing_152389_citlali.fits | HDU 7:coverage_bool_I                                           | different          | [1, 1, 375, 357] -> [1, 1, 375, 357] |
| 152389/raw/toltec_commissioning_a1100_pointing_152389_citlali.fits | HDU 8:sig2noise_I                                               | different          | [1, 1, 375, 357] -> [1, 1, 375, 357] |
| 152389/raw/toltec_commissioning_a1100_pointing_152389_citlali.fits | HDU 9:sig2noise_pixel_I                                         | different          | [1, 1, 375, 357] -> [1, 1, 375, 357] |
| 152389/raw/toltec_commissioning_a1400_pointing_152389_citlali.fits | HDU 1:signal_I                                                  | different          | [1, 1, 375, 357] -> [1, 1, 375, 357] |
| 152389/raw/toltec_commissioning_a1400_pointing_152389_citlali.fits | HDU 2:weight_I                                                  | different          | [1, 1, 375, 357] -> [1, 1, 375, 357] |
| 152389/raw/toltec_commissioning_a1400_pointing_152389_citlali.fits | HDU 3:weight_formal_I                                           | different          | [1, 1, 375, 357] -> [1, 1, 375, 357] |
| 152389/raw/toltec_commissioning_a1400_pointing_152389_citlali.fits | HDU 4:noise_variance_I                                          | different          | [1, 1, 375, 357] -> [1, 1, 375, 357] |
| 152389/raw/toltec_commissioning_a1400_pointing_152389_citlali.fits | HDU 5:kernel_I                                                  | different          | [1, 1, 375, 357] -> [1, 1, 375, 357] |
| 152389/raw/toltec_commissioning_a1400_pointing_152389_citlali.fits | HDU 6:coverage_I                                                | different          | [1, 1, 375, 357] -> [1, 1, 375, 357] |
| 152389/raw/toltec_commissioning_a1400_pointing_152389_citlali.fits | HDU 7:coverage_bool_I                                           | different          | [1, 1, 375, 357] -> [1, 1, 375, 357] |
| 152389/raw/toltec_commissioning_a1400_pointing_152389_citlali.fits | HDU 8:sig2noise_I                                               | different          | [1, 1, 375, 357] -> [1, 1, 375, 357] |
| 152389/raw/toltec_commissioning_a1400_pointing_152389_citlali.fits | HDU 9:sig2noise_pixel_I                                         | different          | [1, 1, 375, 357] -> [1, 1, 375, 357] |
| 152389/raw/toltec_commissioning_a2000_pointing_152389_citlali.fits | HDU 1:signal_I                                                  | different          | [1, 1, 375, 357] -> [1, 1, 375, 357] |
| 152389/raw/toltec_commissioning_a2000_pointing_152389_citlali.fits | HDU 2:weight_I                                                  | different          | [1, 1, 375, 357] -> [1, 1, 375, 357] |
| 152389/raw/toltec_commissioning_a2000_pointing_152389_citlali.fits | HDU 3:weight_formal_I                                           | different          | [1, 1, 375, 357] -> [1, 1, 375, 357] |
| 152389/raw/toltec_commissioning_a2000_pointing_152389_citlali.fits | HDU 4:noise_variance_I                                          | different          | [1, 1, 375, 357] -> [1, 1, 375, 357] |
| 152389/raw/toltec_commissioning_a2000_pointing_152389_citlali.fits | HDU 5:kernel_I                                                  | different          | [1, 1, 375, 357] -> [1, 1, 375, 357] |
| 152389/raw/toltec_commissioning_a2000_pointing_152389_citlali.fits | HDU 6:coverage_I                                                | different          | [1, 1, 375, 357] -> [1, 1, 375, 357] |
| 152389/raw/toltec_commissioning_a2000_pointing_152389_citlali.fits | HDU 7:coverage_bool_I                                           | different          | [1, 1, 375, 357] -> [1, 1, 375, 357] |
| 152389/raw/toltec_commissioning_a2000_pointing_152389_citlali.fits | HDU 8:sig2noise_I                                               | different          | [1, 1, 375, 357] -> [1, 1, 375, 357] |
| 152389/raw/toltec_commissioning_a2000_pointing_152389_citlali.fits | HDU 9:sig2noise_pixel_I                                         | different          | [1, 1, 375, 357] -> [1, 1, 375, 357] |
| 152389/raw/toltec_commissioning_pointing_152389_hist.nc            | a1100_I_bins                                                    | different          | [50] -> [50]                         |
| 152389/raw/toltec_commissioning_pointing_152389_hist.nc            | a1100_I_hist                                                    | different          | [50] -> [50]                         |
| 152389/raw/toltec_commissioning_pointing_152389_hist.nc            | a1100_I_noise_hist                                              | different          | [50] -> [50]                         |
| 152389/raw/toltec_commissioning_pointing_152389_hist.nc            | a1400_I_bins                                                    | different          | [50] -> [50]                         |
| 152389/raw/toltec_commissioning_pointing_152389_hist.nc            | a1400_I_hist                                                    | different          | [50] -> [50]                         |
| 152389/raw/toltec_commissioning_pointing_152389_hist.nc            | a1400_I_noise_hist                                              | different          | [50] -> [50]                         |
| 152389/raw/toltec_commissioning_pointing_152389_hist.nc            | a2000_I_bins                                                    | different          | [50] -> [50]                         |
| 152389/raw/toltec_commissioning_pointing_152389_hist.nc            | a2000_I_hist                                                    | different          | [50] -> [50]                         |
| 152389/raw/toltec_commissioning_pointing_152389_hist.nc            | a2000_I_noise_hist                                              | different          | [50] -> [50]                         |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | coadd_obs_core_weight_sum                                       | different          | [3, 1] -> [3, 1]                     |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | coadd_obs_n_core_pixels                                         | different          | [3, 1] -> [3, 1]                     |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | coadd_obs_n_valid_pixels                                        | different          | [3, 1] -> [3, 1]                     |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | coadd_obs_weight_sum                                            | different          | [3, 1] -> [3, 1]                     |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_core_coverage_median                                        | different          | [3] -> [3]                           |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_core_peak_abs_sig2noise                                     | different          | [3] -> [3]                           |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_core_sig2noise_skew                                         | different          | [3] -> [3]                           |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_core_tail_excess_abs_gt3                                    | different          | [3] -> [3]                           |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_core_tail_excess_neg_lt3                                    | different          | [3] -> [3]                           |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_core_tail_excess_pos_gt3                                    | different          | [3] -> [3]                           |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_core_tail_fraction_abs_gt3                                  | different          | [3] -> [3]                           |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_core_tail_fraction_neg_lt3                                  | different          | [3] -> [3]                           |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_core_tail_fraction_pos_gt3                                  | different          | [3] -> [3]                           |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_core_weight_sum                                             | different          | [3] -> [3]                           |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_coverage_max                                                | different          | [3] -> [3]                           |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_coverage_sum                                                | different          | [3] -> [3]                           |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_empirical_to_formal_noise_ratio                             | different          | [3] -> [3]                           |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_median_err                                                  | different          | [3] -> [3]                           |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_median_rms                                                  | different          | [3] -> [3]                           |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_n_core_pixels                                               | different          | [3] -> [3]                           |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_n_valid_pixels                                              | different          | [3] -> [3]                           |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_noise_products_s2n_sigma                                    | different          | [3] -> [3]                           |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_noise_products_valid_pixels                                 | different          | [3] -> [3]                           |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_noise_rms_p16                                               | different          | [3] -> [3]                           |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_noise_rms_p84                                               | different          | [3] -> [3]                           |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_noise_sig2noise_skew                                        | different          | [3] -> [3]                           |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_noise_tail_excess_abs_gt3                                   | different          | [3] -> [3]                           |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_noise_tail_excess_neg_lt3                                   | different          | [3] -> [3]                           |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_noise_tail_excess_pos_gt3                                   | different          | [3] -> [3]                           |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_noise_tail_fraction_abs_gt3                                 | different          | [3] -> [3]                           |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_noise_tail_fraction_neg_lt3                                 | different          | [3] -> [3]                           |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_noise_tail_fraction_pos_gt3                                 | different          | [3] -> [3]                           |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_noise_weight_median_ratio                                   | different          | [3] -> [3]                           |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_noise_weight_scale                                          | different          | [3] -> [3]                           |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_peak_abs_sig2noise                                          | different          | [3] -> [3]                           |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_peak_signal                                                 | different          | [3] -> [3]                           |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_weight_sum                                                  | different          | [3] -> [3]                           |
| 152389/raw/toltec_commissioning_pointing_152389_mapdiag.nc         | map_weight_threshold                                            | different          | [3] -> [3]                           |
| 152389/raw/toltec_commissioning_pointing_152389_psd.nc             | a1100_I_noise_psd                                               | different          | [167] -> [167]                       |
| 152389/raw/toltec_commissioning_pointing_152389_psd.nc             | a1100_I_noise_psd_2d                                            | different          | [323, 331] -> [323, 331]             |
| 152389/raw/toltec_commissioning_pointing_152389_psd.nc             | a1100_I_noise_psd_freq                                          | different          | [167] -> [167]                       |
| 152389/raw/toltec_commissioning_pointing_152389_psd.nc             | a1100_I_psd                                                     | different          | [167] -> [167]                       |
| 152389/raw/toltec_commissioning_pointing_152389_psd.nc             | a1100_I_psd_2d                                                  | different          | [323, 331] -> [323, 331]             |
| 152389/raw/toltec_commissioning_pointing_152389_psd.nc             | a1100_I_psd_freq                                                | different          | [167] -> [167]                       |
| 152389/raw/toltec_commissioning_pointing_152389_psd.nc             | a1400_I_noise_psd                                               | different          | [170] -> [170]                       |
| 152389/raw/toltec_commissioning_pointing_152389_psd.nc             | a1400_I_noise_psd_2d                                            | different          | [305, 337] -> [305, 337]             |
| 152389/raw/toltec_commissioning_pointing_152389_psd.nc             | a1400_I_noise_psd_freq                                          | different          | [170] -> [170]                       |
| 152389/raw/toltec_commissioning_pointing_152389_psd.nc             | a1400_I_psd                                                     | different          | [170] -> [170]                       |
| 152389/raw/toltec_commissioning_pointing_152389_psd.nc             | a1400_I_psd_2d                                                  | different          | [305, 337] -> [305, 337]             |
| 152389/raw/toltec_commissioning_pointing_152389_psd.nc             | a1400_I_psd_freq                                                | different          | [170] -> [170]                       |
| 152389/raw/toltec_commissioning_pointing_152389_psd.nc             | a2000_I_noise_psd                                               | different          | [172] -> [172]                       |
| 152389/raw/toltec_commissioning_pointing_152389_psd.nc             | a2000_I_noise_psd_2d                                            | different          | [341, 339] -> [341, 339]             |
| 152389/raw/toltec_commissioning_pointing_152389_psd.nc             | a2000_I_noise_psd_freq                                          | different          | [172] -> [172]                       |
| 152389/raw/toltec_commissioning_pointing_152389_psd.nc             | a2000_I_psd                                                     | different          | [172] -> [172]                       |
| 152389/raw/toltec_commissioning_pointing_152389_psd.nc             | a2000_I_psd_2d                                                  | different          | [341, 339] -> [341, 339]             |
| 152389/raw/toltec_commissioning_pointing_152389_psd.nc             | a2000_I_psd_freq                                                | different          | [172] -> [172]                       |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | dimensions                                                      | dimensions_changed |                                      |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | PpsTime                                                         | missing_variable   |                                      |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.FRUITLOOPS.INJECTED_SOURCE_TEST.AMPLITUDE_MJY_BEAM_a1100 | extra_variable     |                                      |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.FRUITLOOPS.INJECTED_SOURCE_TEST.AMPLITUDE_MJY_BEAM_a1400 | extra_variable     |                                      |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.FRUITLOOPS.INJECTED_SOURCE_TEST.AMPLITUDE_MJY_BEAM_a2000 | extra_variable     |                                      |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.FRUITLOOPS.INJECTED_SOURCE_TEST.ENABLED                  | extra_variable     |                                      |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.FRUITLOOPS.INJECTED_SOURCE_TEST.START_ITERATION          | extra_variable     |                                      |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.FRUITLOOPS.RESTART_PATH                                  | extra_variable     |                                      |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | CONFIG.FRUITLOOPS.SOURCE_CENTER_MODE                            | extra_variable     |                                      |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | ActGalAng                                                       | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | ActParAng                                                       | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Dcs.CalMode                                              | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Dcs.IntegrationTime                                      | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Dcs.ObsMode                                              | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Dcs.ObsNum                                               | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Dcs.ObsType                                              | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Dcs.RequestedTime                                        | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Dcs.ScanNum                                              | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Dcs.SubObsNum                                            | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Gps.IgnoreLock                                           | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Lissajous.ExecMode                                       | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Lissajous.ScanRate                                       | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Lissajous.TScan                                          | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Lissajous.XDelta                                         | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Lissajous.XDeltaMinor                                    | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Lissajous.XLength                                        | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Lissajous.XLengthMinor                                   | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Lissajous.XOmega                                         | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Lissajous.XOmegaMinor                                    | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Lissajous.XOmegaMinorNorm                                | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Lissajous.XOmegaNorm                                     | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Lissajous.YLength                                        | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Lissajous.YLengthMinor                                   | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Lissajous.YOmega                                         | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Lissajous.YOmegaMinor                                    | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Lissajous.YOmegaMinorNorm                                | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Lissajous.YOmegaNorm                                     | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M1.ActPos                                                | shape_changed      | [1] -> [720]                         |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M1.CmdPos                                                | shape_changed      | [1] -> [720]                         |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M1.ModelEnabled                                          | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M1.ModelMode                                             | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M1.ZernikeC                                              | shape_changed      | [1] -> [18]                          |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M1.ZernikeEnabled                                        | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.AcuHeartbeat                                          | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.Alive                                                 | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.AzPcor                                                | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.CorEnabled                                            | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.ElCmd                                                 | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.ElPcor                                                | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.Follow                                                | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.Hold                                                  | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.M2Heartbeat                                           | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.ModelMode                                             | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.TiltAct                                               | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.TiltCmd                                               | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.TiltDes                                               | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.TiltPcor                                              | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.TiltReq                                               | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.TipAct                                                | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.TipCmd                                                | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.TipDes                                                | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.TipPcor                                               | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.TipReq                                                | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.XAct                                                  | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.XCmd                                                  | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.XDes                                                  | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.XPcor                                                 | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.XReq                                                  | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.YAct                                                  | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.YCmd                                                  | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.YDes                                                  | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.YPcor                                                 | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.YReq                                                  | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.ZAct                                                  | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.ZCmd                                                  | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.ZDes                                                  | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.ZPcor                                                 | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M2.ZReq                                                  | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M3.AcuHeartbeat                                          | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M3.Alive                                                 | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M3.ElDesEnabled                                          | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M3.Fault                                                 | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M3.M3Heartbeat                                           | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.M3.M3OffPos                                              | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.PointModel.AzM2Cor                                       | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.PointModel.AzPaddleOff                                   | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.PointModel.AzPointModelCor                               | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.PointModel.AzReceiverCor                                 | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.PointModel.AzReceiverOff                                 | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.PointModel.AzTiltCor                                     | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.PointModel.AzTotalCor                                    | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.PointModel.AzUserOff                                     | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.PointModel.ElM2Cor                                       | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.PointModel.ElPaddleOff                                   | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.PointModel.ElPointModelCor                               | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.PointModel.ElReceiverCor                                 | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.PointModel.ElReceiverOff                                 | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.PointModel.ElRefracCor                                   | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.PointModel.ElTiltCor                                     | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.PointModel.ElTotalCor                                    | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.PointModel.ElUserOff                                     | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.PointModel.M2CorEnabled                                  | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.PointModel.ModRev                                        | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.PointModel.PointModelCorEnabled                          | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.PointModel.ReceiverOffEnabled                            | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.PointModel.RefracCorEnabled                              | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.PointModel.TiltCorEnabled                                | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Radiometer.Tau                                           | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Radiometer.Tau2                                          | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.ScanFile.Valid                                           | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Sky.BaryVel                                              | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Sky.ObsVel                                               | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Sky.ParAng                                               | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Sky.RaOffsetSys                                          | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Source.B                                                 | shape_changed      | [1] -> [2]                           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Source.CoordSys                                          | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Source.Dec                                               | shape_changed      | [1] -> [2]                           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Source.DecProperMotionCor                                | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Source.ElObsMax                                          | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Source.ElObsMin                                          | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Source.Epoch                                             | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Source.L                                                 | shape_changed      | [1] -> [2]                           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Source.Planet                                            | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Source.Ra                                                | shape_changed      | [1] -> [2]                           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Source.RaProperMotionCor                                 | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Source.VelSys                                            | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Source.Velocity                                          | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Telescope.AzActPos                                       | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Telescope.AzDesPos                                       | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Telescope.CraneInBeam                                    | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Telescope.ElActPos                                       | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Telescope.ElDesPos                                       | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Telescope.PointingTolerance                              | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.TelescopeBackend.CalObsNum                               | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.TelescopeBackend.Master                                  | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.TelescopeBackend.NumPixels                               | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.TelescopeBackend.ObsNum                                  | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.TelescopeBackend.ScanNum                                 | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.TelescopeBackend.SubObsNum                               | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Tiltmeter_0_.Temp                                        | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Tiltmeter_0_.TiltX                                       | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Tiltmeter_0_.TiltY                                       | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Tiltmeter_1_.Temp                                        | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Tiltmeter_1_.TiltX                                       | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Tiltmeter_1_.TiltY                                       | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.TimePlace.LST                                            | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.TimePlace.ObsElevation                                   | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.TimePlace.ObsLatitude                                    | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.TimePlace.ObsLongitude                                   | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.TimePlace.UT1                                            | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.TimePlace.UTDate                                         | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Toltec.AzPointCor                                        | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Toltec.AzPointOff                                        | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Toltec.BeamSelected                                      | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Toltec.ElPointCor                                        | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Toltec.ElPointOff                                        | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Toltec.M3Dir                                             | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Toltec.NumBands                                          | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Toltec.NumBeams                                          | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Toltec.NumPixels                                         | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Toltec.Remote                                            | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Weather.Humidity                                         | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Weather.Precipitation                                    | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Weather.Pressure                                         | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Weather.Radiation                                        | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Weather.Temperature                                      | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Weather.TimeOfDay                                        | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Weather.WindDir1                                         | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Weather.WindDir2                                         | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Weather.WindSpeed1                                       | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Header.Weather.WindSpeed2                                       | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | Hold                                                            | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | MEAN_AZ                                                         | different          | [1] -> [1]                           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | MEAN_EL                                                         | different          | [1] -> [1]                           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | MEAN_PA                                                         | different          | [1] -> [1]                           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | OOF_RMS_a1100                                                   | different          | [1] -> [1]                           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | OOF_RMS_a1400                                                   | different          | [1] -> [1]                           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | OOF_RMS_a2000                                                   | different          | [1] -> [1]                           |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | SourceAz                                                        | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | SourceEl                                                        | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | TelAzAct                                                        | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | TelAzCor                                                        | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | TelAzDes                                                        | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | TelAzMap                                                        | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | TelB                                                            | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | TelDec                                                          | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | TelElAct                                                        | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | TelElCor                                                        | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | TelElDes                                                        | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | TelElMap                                                        | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | TelL                                                            | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | TelRa                                                           | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | TelTime                                                         | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | TelUTC                                                          | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | alt_phys                                                        | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | az_phys                                                         | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | b_phys                                                          | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | dec_phys                                                        | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | flags                                                           | shape_changed      | [3628, 5518] -> [3660, 5518]         |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | l_phys                                                          | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | lat_phys                                                        | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | lon_phys                                                        | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | pointing_offset_alt                                             | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | pointing_offset_az                                              | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | ptc_second_pass_added_flag                                      | shape_changed      | [3628, 5518] -> [3660, 5518]         |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | ptc_second_pass_existing_flagged_fraction                       | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | ptc_second_pass_max_unflagged_residual_uid                      | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | ptc_second_pass_max_unflagged_residual_z                        | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | ptc_second_pass_n_accepted_clusters                             | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | ptc_second_pass_n_accepted_events                               | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | ptc_second_pass_n_candidate_clusters                            | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | ptc_second_pass_n_candidate_events                              | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | ptc_second_pass_newly_flagged_fraction                          | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | ptc_second_pass_proposed_flagged_fraction                       | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | ptc_second_pass_top_candidate_cluster_peak_score                | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | ptc_second_pass_top_candidate_cluster_sample                    | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | ptc_second_pass_top_event_sample                                | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | ptc_second_pass_top_event_score                                 | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | ptc_second_pass_top_event_uid                                   | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | ra_phys                                                         | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | raw_scan_indices                                                | different          | [12, 4] -> [12, 4]                   |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | scan_indices                                                    | different          | [12, 2] -> [12, 2]                   |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | signal                                                          | shape_changed      | [3628, 5518] -> [3660, 5518]         |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | tod_filter_edge_guard_flagged_frac                              | different          | [12] -> [12]                         |
| 152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc  | weights                                                         | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | dimensions                                                      | dimensions_changed |                                      |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.FRUITLOOPS.INJECTED_SOURCE_TEST.ENABLED                  | extra_variable     |                                      |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.FRUITLOOPS.INJECTED_SOURCE_TEST.START_ITERATION          | extra_variable     |                                      |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.FRUITLOOPS.RESTART_PATH                                  | extra_variable     |                                      |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | CONFIG.FRUITLOOPS.SOURCE_CENTER_MODE                            | extra_variable     |                                      |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_detector_flagged_fraction                                   | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_detector_median                                             | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_detector_rms                                                | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_detector_stddev                                             | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_detector_weight                                             | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_invvar_window_flagged_frac_max                              | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_invvar_window_flagged_frac_median                           | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_invvar_window_heavy_flagged_fraction                        | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_invvar_window_median                                        | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_invvar_window_n_total                                       | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_invvar_window_n_valid                                       | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_invvar_window_q10                                           | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_invvar_window_q90                                           | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_invvar_window_valid_fraction                                | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_second_pass_existing_flagged_fraction                       | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_second_pass_max_unflagged_residual_uid                      | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_second_pass_max_unflagged_residual_z                        | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_second_pass_n_accepted_clusters                             | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_second_pass_n_accepted_events                               | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_second_pass_n_candidate_clusters                            | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_second_pass_n_candidate_events                              | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_second_pass_newly_flagged_fraction                          | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_second_pass_proposed_flagged_fraction                       | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_second_pass_top_candidate_cluster_peak_score                | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_second_pass_top_candidate_cluster_sample                    | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_second_pass_top_event_sample                                | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_second_pass_top_event_score                                 | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_ptcdiag.nc         | ptc_second_pass_top_event_uid                                   | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | dimensions                                                      | dimensions_changed |                                      |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | PpsTime                                                         | missing_variable   |                                      |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | CONFIG.FRUITLOOPS.INJECTED_SOURCE_TEST.AMPLITUDE_MJY_BEAM_a1100 | extra_variable     |                                      |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | CONFIG.FRUITLOOPS.INJECTED_SOURCE_TEST.AMPLITUDE_MJY_BEAM_a1400 | extra_variable     |                                      |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | CONFIG.FRUITLOOPS.INJECTED_SOURCE_TEST.AMPLITUDE_MJY_BEAM_a2000 | extra_variable     |                                      |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | CONFIG.FRUITLOOPS.INJECTED_SOURCE_TEST.ENABLED                  | extra_variable     |                                      |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | CONFIG.FRUITLOOPS.INJECTED_SOURCE_TEST.START_ITERATION          | extra_variable     |                                      |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | CONFIG.FRUITLOOPS.RESTART_PATH                                  | extra_variable     |                                      |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | CONFIG.FRUITLOOPS.SOURCE_CENTER_MODE                            | extra_variable     |                                      |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | ActGalAng                                                       | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | ActParAng                                                       | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Dcs.CalMode                                              | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Dcs.IntegrationTime                                      | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Dcs.ObsMode                                              | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Dcs.ObsNum                                               | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Dcs.ObsType                                              | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Dcs.RequestedTime                                        | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Dcs.ScanNum                                              | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Dcs.SubObsNum                                            | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Gps.IgnoreLock                                           | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Lissajous.ExecMode                                       | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Lissajous.ScanRate                                       | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Lissajous.TScan                                          | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Lissajous.XDelta                                         | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Lissajous.XDeltaMinor                                    | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Lissajous.XLength                                        | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Lissajous.XLengthMinor                                   | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Lissajous.XOmega                                         | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Lissajous.XOmegaMinor                                    | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Lissajous.XOmegaMinorNorm                                | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Lissajous.XOmegaNorm                                     | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Lissajous.YLength                                        | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Lissajous.YLengthMinor                                   | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Lissajous.YOmega                                         | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Lissajous.YOmegaMinor                                    | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Lissajous.YOmegaMinorNorm                                | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Lissajous.YOmegaNorm                                     | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.M1.ActPos                                                | shape_changed      | [1] -> [720]                         |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.M1.CmdPos                                                | shape_changed      | [1] -> [720]                         |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.M1.ModelEnabled                                          | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.M1.ModelMode                                             | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.M1.ZernikeC                                              | shape_changed      | [1] -> [18]                          |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.M1.ZernikeEnabled                                        | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.M2.AcuHeartbeat                                          | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.M2.Alive                                                 | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.M2.AzPcor                                                | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.M2.CorEnabled                                            | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.M2.ElCmd                                                 | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.M2.ElPcor                                                | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.M2.Follow                                                | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.M2.Hold                                                  | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.M2.M2Heartbeat                                           | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.M2.ModelMode                                             | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.M2.TiltAct                                               | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.M2.TiltCmd                                               | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.M2.TiltDes                                               | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.M2.TiltPcor                                              | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.M2.TiltReq                                               | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.M2.TipAct                                                | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.M2.TipCmd                                                | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.M2.TipDes                                                | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.M2.TipPcor                                               | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.M2.TipReq                                                | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.M2.XAct                                                  | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.M2.XCmd                                                  | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.M2.XDes                                                  | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.M2.XPcor                                                 | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.M2.XReq                                                  | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.M2.YAct                                                  | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.M2.YCmd                                                  | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.M2.YDes                                                  | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.M2.YPcor                                                 | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.M2.YReq                                                  | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.M2.ZAct                                                  | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.M2.ZCmd                                                  | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.M2.ZDes                                                  | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.M2.ZPcor                                                 | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.M2.ZReq                                                  | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.M3.AcuHeartbeat                                          | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.M3.Alive                                                 | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.M3.ElDesEnabled                                          | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.M3.Fault                                                 | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.M3.M3Heartbeat                                           | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.M3.M3OffPos                                              | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.PointModel.AzM2Cor                                       | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.PointModel.AzPaddleOff                                   | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.PointModel.AzPointModelCor                               | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.PointModel.AzReceiverCor                                 | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.PointModel.AzReceiverOff                                 | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.PointModel.AzTiltCor                                     | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.PointModel.AzTotalCor                                    | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.PointModel.AzUserOff                                     | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.PointModel.ElM2Cor                                       | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.PointModel.ElPaddleOff                                   | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.PointModel.ElPointModelCor                               | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.PointModel.ElReceiverCor                                 | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.PointModel.ElReceiverOff                                 | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.PointModel.ElRefracCor                                   | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.PointModel.ElTiltCor                                     | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.PointModel.ElTotalCor                                    | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.PointModel.ElUserOff                                     | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.PointModel.M2CorEnabled                                  | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.PointModel.ModRev                                        | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.PointModel.PointModelCorEnabled                          | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.PointModel.ReceiverOffEnabled                            | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.PointModel.RefracCorEnabled                              | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.PointModel.TiltCorEnabled                                | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Radiometer.Tau                                           | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Radiometer.Tau2                                          | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.ScanFile.Valid                                           | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Sky.BaryVel                                              | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Sky.ObsVel                                               | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Sky.ParAng                                               | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Sky.RaOffsetSys                                          | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Source.B                                                 | shape_changed      | [1] -> [2]                           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Source.CoordSys                                          | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Source.Dec                                               | shape_changed      | [1] -> [2]                           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Source.DecProperMotionCor                                | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Source.ElObsMax                                          | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Source.ElObsMin                                          | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Source.Epoch                                             | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Source.L                                                 | shape_changed      | [1] -> [2]                           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Source.Planet                                            | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Source.Ra                                                | shape_changed      | [1] -> [2]                           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Source.RaProperMotionCor                                 | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Source.VelSys                                            | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Source.Velocity                                          | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Telescope.AzActPos                                       | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Telescope.AzDesPos                                       | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Telescope.CraneInBeam                                    | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Telescope.ElActPos                                       | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Telescope.ElDesPos                                       | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Telescope.PointingTolerance                              | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.TelescopeBackend.CalObsNum                               | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.TelescopeBackend.Master                                  | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.TelescopeBackend.NumPixels                               | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.TelescopeBackend.ObsNum                                  | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.TelescopeBackend.ScanNum                                 | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.TelescopeBackend.SubObsNum                               | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Tiltmeter_0_.Temp                                        | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Tiltmeter_0_.TiltX                                       | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Tiltmeter_0_.TiltY                                       | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Tiltmeter_1_.Temp                                        | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Tiltmeter_1_.TiltX                                       | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Tiltmeter_1_.TiltY                                       | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.TimePlace.LST                                            | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.TimePlace.ObsElevation                                   | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.TimePlace.ObsLatitude                                    | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.TimePlace.ObsLongitude                                   | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.TimePlace.UT1                                            | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.TimePlace.UTDate                                         | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Toltec.AzPointCor                                        | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Toltec.AzPointOff                                        | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Toltec.BeamSelected                                      | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Toltec.ElPointCor                                        | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Toltec.ElPointOff                                        | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Toltec.M3Dir                                             | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Toltec.NumBands                                          | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Toltec.NumBeams                                          | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Toltec.NumPixels                                         | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Toltec.Remote                                            | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Weather.Humidity                                         | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Weather.Precipitation                                    | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Weather.Pressure                                         | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Weather.Radiation                                        | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Weather.Temperature                                      | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Weather.TimeOfDay                                        | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Weather.WindDir1                                         | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Weather.WindDir2                                         | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Weather.WindSpeed1                                       | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Header.Weather.WindSpeed2                                       | shape_changed      | [1] -> []                            |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | Hold                                                            | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | MEAN_AZ                                                         | different          | [1] -> [1]                           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | MEAN_EL                                                         | different          | [1] -> [1]                           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | MEAN_PA                                                         | different          | [1] -> [1]                           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | OOF_RMS_a1100                                                   | different          | [1] -> [1]                           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | OOF_RMS_a1400                                                   | different          | [1] -> [1]                           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | OOF_RMS_a2000                                                   | different          | [1] -> [1]                           |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | SourceAz                                                        | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | SourceEl                                                        | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | TelAzAct                                                        | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | TelAzCor                                                        | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | TelAzDes                                                        | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | TelAzMap                                                        | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | TelB                                                            | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | TelDec                                                          | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | TelElAct                                                        | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | TelElCor                                                        | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | TelElDes                                                        | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | TelElMap                                                        | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | TelL                                                            | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | TelRa                                                           | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | TelTime                                                         | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | TelUTC                                                          | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | alt_phys                                                        | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | az_phys                                                         | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | b_phys                                                          | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | dec_phys                                                        | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | flags                                                           | shape_changed      | [3628, 5518] -> [3660, 5518]         |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | l_phys                                                          | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | lat_phys                                                        | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | lon_phys                                                        | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | pointing_offset_alt                                             | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | pointing_offset_az                                              | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | ra_phys                                                         | shape_changed      | [3628] -> [3660]                     |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | raw_scan_indices                                                | different          | [12, 4] -> [12, 4]                   |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_despike_added_flagged_frac                                  | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_despike_added_region_count                                  | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_despike_added_region_len_max                                | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_despike_added_region_len_median                             | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_despike_local_exceed_count                                  | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_despike_local_flagged_sample_count                          | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_despike_local_raw_accepted_event_count                      | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_despike_local_raw_candidate_count                           | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_despike_local_raw_reject_count                              | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_despike_max_local_abs_z                                     | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_despike_max_local_delta_abs_z                               | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_final_flagged_frac                                          | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_final_region_count                                          | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_final_region_len_max                                        | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_final_region_len_median                                     | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_impulsive_event_kind                                        | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_impulsive_event_sample                                      | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_impulsive_event_score                                       | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_impulsive_near_abs_count                                    | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_impulsive_near_delta_count                                  | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_impulsive_peak_abs_sample                                   | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_impulsive_peak_abs_z                                        | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_impulsive_peak_delta_abs_sample                             | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_impulsive_peak_delta_abs_z                                  | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_network_cm_low_mid_ratio                                    | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_network_cm_peak_freq_hz                                     | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_network_cm_peak_prominence                                  | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_network_impulsive_alignment_frac                            | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_network_impulsive_det_frac                                  | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_network_impulsive_dominant_sample                           | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_network_impulsive_n_det_used                                | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_network_impulsive_score_max                                 | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_network_impulsive_score_median                              | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_network_n_det_used                                          | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_network_step_alignment_frac                                 | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_network_step_det_frac                                       | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_network_step_dominant_sample                                | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_network_step_score_max                                      | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_network_step_score_median                                   | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_step_sample                                                 | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | rtc_step_score                                                  | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | scan_indices                                                    | different          | [12, 2] -> [12, 2]                   |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | signal                                                          | shape_changed      | [3628, 5518] -> [3660, 5518]         |
| 152389/raw/toltec_commissioning_pointing_152389_rtc_timestream.nc  | tod_filter_edge_guard_flagged_frac                              | different          | [12] -> [12]                         |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_despike_added_flagged_frac                                  | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_despike_added_region_count                                  | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_despike_added_region_len_max                                | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_despike_added_region_len_median                             | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_despike_local_exceed_count                                  | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_despike_local_flagged_sample_count                          | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_despike_local_raw_accepted_event_count                      | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_despike_local_raw_candidate_count                           | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_despike_local_raw_reject_count                              | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_despike_max_local_abs_z                                     | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_despike_max_local_delta_abs_z                               | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_final_flagged_frac                                          | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_final_region_count                                          | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_final_region_len_max                                        | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_final_region_len_median                                     | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_impulsive_event_kind                                        | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_impulsive_event_sample                                      | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_impulsive_event_score                                       | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_impulsive_near_abs_count                                    | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_impulsive_near_delta_count                                  | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_impulsive_peak_abs_sample                                   | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_impulsive_peak_abs_z                                        | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_impulsive_peak_delta_abs_sample                             | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_impulsive_peak_delta_abs_z                                  | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_invvar_window_flagged_frac_max                              | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_invvar_window_flagged_frac_median                           | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_invvar_window_heavy_flagged_fraction                        | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_invvar_window_median                                        | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_invvar_window_n_valid                                       | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_invvar_window_q10                                           | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_invvar_window_q90                                           | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_invvar_window_valid_fraction                                | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_network_cm_low_mid_ratio                                    | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_network_cm_peak_freq_hz                                     | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_network_cm_peak_prominence                                  | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_network_impulsive_alignment_frac                            | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_network_impulsive_det_frac                                  | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_network_impulsive_dominant_sample                           | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_network_impulsive_n_det_used                                | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_network_impulsive_score_max                                 | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_network_impulsive_score_median                              | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_network_n_det_used                                          | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_network_step_alignment_frac                                 | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_network_step_det_frac                                       | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_network_step_dominant_sample                                | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_network_step_score_max                                      | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_network_step_score_median                                   | different          | [12, 11] -> [12, 11]                 |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_step_sample                                                 | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | rtc_step_score                                                  | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | scan_duration_s                                                 | different          | [12] -> [12]                         |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | scan_source_power_half_bandwidth_hz                             | different          | [12, 3] -> [12, 3]                   |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | scan_speed_altaz_p50_arcsec_s                                   | different          | [12] -> [12]                         |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | scan_speed_altaz_p95_arcsec_s                                   | different          | [12] -> [12]                         |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | scan_speed_altaz_p995_arcsec_s                                  | different          | [12] -> [12]                         |
| 152389/raw/toltec_commissioning_pointing_152389_rtcdiag.nc         | scan_tod_lowpass_to_source_power_half_ratio                     | different          | [12, 3] -> [12, 3]                   |
| 152389/raw/toltec_commissioning_pointing_152389_stats.nc           | flagged_frac                                                    | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_stats.nc           | median                                                          | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_stats.nc           | rms                                                             | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_stats.nc           | stddev                                                          | different          | [12, 5518] -> [12, 5518]             |
| 152389/raw/toltec_commissioning_pointing_152389_stats.nc           | weights                                                         | different          | [12, 5518] -> [12, 5518]             |
