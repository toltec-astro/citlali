# pragma once

namespace citlali::utils::threads {

std::string chunk_exec_mode = "seq";
std::string det_exec_mode = "seq";

int n_chunk_threads = 1;
int n_det_threads = 1;

void set_optimal_threads(int n_threads, int n_chunks, std::string exec_mode) {
    chunk_exec_mode = exec_mode;
    det_exec_mode = exec_mode;

    if (exec_mode != "seq") {
        n_chunk_threads = std::min(n_threads, n_chunks);
        n_det_threads = n_threads / n_chunk_threads;

        if (n_det_threads == 1) {
            det_exec_mode = "seq";
        }
    }
}
} // namespace
