#include <citlali/core/pipeline/fruit_loop_map_input_validation.h>

void fruit_loop_map_input_validation_header_compiles_in_isolation() {
    citlali::pipeline::require_fruit_loop_map_input(
        true, "header isolation check");
}
