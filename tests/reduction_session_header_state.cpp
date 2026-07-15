#include <citlali/core/session/reduction_session.h>

#include <cstddef>

std::size_t reduction_session_header_state_from_translation_unit() {
    const citlali::session::ReductionSession session;
    return session.runs_started();
}
