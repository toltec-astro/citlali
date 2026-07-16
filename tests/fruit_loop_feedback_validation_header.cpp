#include <citlali/core/pipeline/fruit_loop_feedback_validation.h>

void fruit_loop_feedback_validation_header_compiles_in_isolation() {
    citlali::pipeline::require_fruit_loop_map_index(0, 1);
}
