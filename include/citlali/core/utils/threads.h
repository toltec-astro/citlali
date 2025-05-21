# pragma once

namespace citlali::utils::threads {

std::string chunk_exec_mode = "seq";
std::string chunk_remainder_exec_mode = "seq";
std::string map_exec_mode = "seq";
std::string map_remainder_exec_mode = "seq";

int n_chunk_threads = 1;
int n_chunk_remainder_threads = 1;
int n_map_threads = 1;
int n_map_remainder_threads = 1;

void set_optimal_threads(int n_threads, int n_chunks, int n_maps, std::string exec_mode) {
    chunk_exec_mode = exec_mode;
    chunk_remainder_exec_mode = exec_mode;
    map_exec_mode = exec_mode;
    map_remainder_exec_mode = exec_mode;

    if (exec_mode != "seq") {
        n_chunk_threads = std::min(n_threads, n_chunks);
        n_chunk_remainder_threads = n_threads / n_chunk_threads;
        n_map_threads = std::min(n_threads, n_maps);
        n_map_remainder_threads = n_threads / n_maps;

        if (n_chunk_remainder_threads == 1) {
            chunk_remainder_exec_mode = "seq";
        }
        if (n_map_remainder_threads == 1) {
            map_remainder_exec_mode = "seq";
        }
    }
}

auto get_grppi_vectors(const int n_pts) {
    std::vector<int> in(n_pts), out(n_pts);
    std::iota(in.begin(), in.end(), 0);

    return std::make_tuple(in, out);
}

auto get_seq_exec_mode() {
    return tula::grppi_utils::dyn_ex("seq");
}

auto get_chunk_exec_mode() {
    return tula::grppi_utils::dyn_ex(chunk_exec_mode, n_chunk_threads);
}

auto get_chunk_remainder_exec_mode() {
    return tula::grppi_utils::dyn_ex(chunk_remainder_exec_mode, n_chunk_remainder_threads);
}

auto get_map_exec_mode() {
    return tula::grppi_utils::dyn_ex(map_exec_mode, n_map_threads);
}

auto get_map_remainder_exec_mode() {
    return tula::grppi_utils::dyn_ex(map_remainder_exec_mode, n_map_remainder_threads);
}

} // namespace
