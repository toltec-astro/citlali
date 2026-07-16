#include <citlali/core/pipeline/wiener_filter_validation.h>

void wiener_filter_validation_header_compiles_in_isolation() {
    citlali::pipeline::require_wiener_template_geometry(2, 2, 2, 2);
}
