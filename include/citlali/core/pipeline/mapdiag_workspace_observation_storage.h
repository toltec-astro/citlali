#pragma once

// Included by mapdiag_workspace.h inside namespace citlali::pipeline.

struct MapdiagObservationWorkspace {
    explicit MapdiagObservationWorkspace(std::size_t table_size,
                                         double fill_double, int fill_int)
        : weight_sum(table_size, fill_double),
          weight_frac(table_size, fill_double),
          core_weight_sum(table_size, fill_double),
          core_weight_frac(table_size, fill_double),
          valid_pixels(table_size, fill_int),
          core_pixels(table_size, fill_int),
          tables{weight_sum, core_weight_sum, valid_pixels, core_pixels},
          double_values{
              weight_sum, weight_frac, core_weight_sum, core_weight_frac},
          int_values{valid_pixels, core_pixels} {}

    std::vector<double> weight_sum;
    std::vector<double> weight_frac;
    std::vector<double> core_weight_sum;
    std::vector<double> core_weight_frac;
    std::vector<int> valid_pixels;
    std::vector<int> core_pixels;
    MapdiagObsTableRefs tables;
    MapdiagObservationDoubleValues double_values;
    MapdiagObservationIntValues int_values;
};

struct MapdiagOutlierMaskContext {
    MapdiagSourceDistanceContext source_distance;
    Eigen::ArrayXXd off_source_core_mask;
};

