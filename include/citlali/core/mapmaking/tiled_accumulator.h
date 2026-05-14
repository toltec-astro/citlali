#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Core>

namespace mapmaking {

class TiledMapAccumulator {
public:
    explicit TiledMapAccumulator(Eigen::Index tile_size = 64)
        : m_tile_size(std::max<Eigen::Index>(1, tile_size)) {}

    void reset(std::size_t n_maps, Eigen::Index n_rows, Eigen::Index n_cols) {
        if (n_rows < 0 || n_cols < 0) {
            throw std::runtime_error("negative map dimensions in TiledMapAccumulator");
        }
        m_n_rows = n_rows;
        m_n_cols = n_cols;
        m_maps.clear();
        m_maps.resize(n_maps);
    }

    bool empty() const {
        return m_maps.empty();
    }

    void add(Eigen::Index map_index, Eigen::Index row, Eigen::Index col, double value) {
        if (map_index < 0 || map_index >= static_cast<Eigen::Index>(m_maps.size())) {
            throw std::runtime_error("map index out of range in TiledMapAccumulator");
        }
        auto &tile = tile_for(m_maps[static_cast<std::size_t>(map_index)], row, col);
        tile.values(row - tile.row0, col - tile.col0) += value;
    }

    void merge_into(std::vector<Eigen::MatrixXd> &target) const {
        if (target.size() != m_maps.size()) {
            throw std::runtime_error("target map count mismatch in TiledMapAccumulator");
        }

        for (std::size_t i = 0; i < m_maps.size(); ++i) {
            if (target[i].rows() != m_n_rows || target[i].cols() != m_n_cols) {
                throw std::runtime_error("target map dimensions mismatch in TiledMapAccumulator");
            }
            for (const auto &tile : m_maps[i].tiles) {
                target[i].block(tile.row0, tile.col0, tile.values.rows(), tile.values.cols()) +=
                    tile.values;
            }
        }
    }

private:
    struct Tile {
        Eigen::Index row0 = 0;
        Eigen::Index col0 = 0;
        Eigen::MatrixXd values;
    };

    struct MapTiles {
        std::vector<Tile> tiles;
        std::unordered_map<std::uint64_t, std::size_t> tile_index;
    };

    std::uint64_t tile_key(Eigen::Index tile_row, Eigen::Index tile_col) const {
        return (static_cast<std::uint64_t>(tile_row) << 32U) |
               (static_cast<std::uint64_t>(tile_col) & 0xffffffffULL);
    }

    Tile &tile_for(MapTiles &map_tiles, Eigen::Index row, Eigen::Index col) {
        if (row < 0 || row >= m_n_rows || col < 0 || col >= m_n_cols) {
            throw std::runtime_error("pixel index out of range in TiledMapAccumulator");
        }

        const Eigen::Index tile_row = row / m_tile_size;
        const Eigen::Index tile_col = col / m_tile_size;
        const auto key = tile_key(tile_row, tile_col);

        auto it = map_tiles.tile_index.find(key);
        if (it != map_tiles.tile_index.end()) {
            return map_tiles.tiles[it->second];
        }

        const Eigen::Index row0 = tile_row * m_tile_size;
        const Eigen::Index col0 = tile_col * m_tile_size;
        const Eigen::Index rows = std::min(m_tile_size, m_n_rows - row0);
        const Eigen::Index cols = std::min(m_tile_size, m_n_cols - col0);

        Tile tile;
        tile.row0 = row0;
        tile.col0 = col0;
        tile.values.setZero(rows, cols);

        const auto index = map_tiles.tiles.size();
        map_tiles.tiles.push_back(std::move(tile));
        map_tiles.tile_index.emplace(key, index);
        return map_tiles.tiles.back();
    }

    Eigen::Index m_tile_size;
    Eigen::Index m_n_rows = 0;
    Eigen::Index m_n_cols = 0;
    std::vector<MapTiles> m_maps;
};

} // namespace mapmaking
