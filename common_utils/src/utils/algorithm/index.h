#pragma once
#include "../logging.h"
#include <vector>
#include <algorithm>

namespace alg {

/**
 * @brief Generate a list of index chunks from given index range.
 * @param start The begin index.
 * @param end The past-last index.
 * @param nchunks Number of chunks to be generated
 * @param overlap The overlap between consecutive chunks.
 */
template <typename Index=int>
auto indexchunks(Index start, Index end, Index nchunks,
                        Index overlap = 0) {
    std::vector<std::pair<Index, Index>> chunks;
    chunks.resize(nchunks);
    auto size = end - start;
    // handle overlap by working with a streched-out size
    auto s_size = size + overlap * (nchunks - 1);
    auto s_chunk_size = s_size / nchunks; // this is the min chunk size
    auto s_leftover =
        s_size % nchunks; // this will be distributed among the first few chunks
    SPDLOG_TRACE("chunk s_size={} s_chunk_size={} s_leftover={}", s_size,
                 s_chunk_size, s_leftover);
    std::generate(chunks.begin(), chunks.end(),
                  [&, curr = start, i = 0]() mutable {
                      auto step = s_chunk_size + (i < s_leftover ? 1 : 0);
                      // SPDLOG_TRACE("gen with curr={} i={} step={}", curr, i,
                      // step);
                      auto chunk = std::make_pair(curr, curr + step);
                      ++i;
                      curr += step - overlap;
                      // SPDLOG_TRACE("done with curr={} i={} step={}", curr, i,
                      // step);
                      return chunk;
                  });
    return chunks;
}

} //namespace alg


