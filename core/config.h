#pragma once

namespace lali{

namespace predefs {

// we need to be careful about the int type used here as
// we may have very large array in the memory.

// check eigen index type
static_assert(std::is_same_v<std::ptrdiff_t, Eigen::Index>,
              "UNEXPECTED EIGEN INDEX TYPE");
using index_t = std::ptrdiff_t;
using shape_t = Eigen::Matrix<index_t, 2, 1>;
using data_t = double;
} // namespace predefs


/**
 * @brief The YamlConfig struct
 * This is a thin wrapper around YAML::Node.
 */
struct YamlConfig {
    using key_t = std::string;
    using value_t = YAML::Node;
    using storage_t = YAML::Node;
    YamlConfig() = default;
    explicit YamlConfig(storage_t node) : m_node(std::move(node)) {}
    std::string dump_to_str() const { return fmt::format("{}", m_node); }
    static auto load_from_str(std::string s) {
        return YamlConfig(YAML::LoadFile(s));
    }
    friend std::ostream &operator<<(std::ostream &os,
                                    const YamlConfig &config) {
        return os << config.dump_to_str();
    }

    void list_all(){
        for (YAML::const_iterator it = m_node.begin(); it != m_node.end(); ++it){
            std::cerr << it->second.as<std::string>() << std::endl;
        }
    }

    template <typename T> auto get_typed(const key_t &key) const {
        //return m_node[key].as<T>(key);
        return  m_node[key].as<T>();
    }

    template <typename T> auto get_typed(const key_t &key, T &&defval) const {
        decltype(auto) node = m_node[key];
        if (node.IsDefined() && !node.IsNull()) {
            return node.as<T>(key);
        }
        return FWD(defval);
    }

    auto get_str(const key_t &key) const { return get_typed<std::string>(key); }
    auto get_str(const key_t &key, const std::string &defval) const {
        return get_typed<std::string>(key, std::string(defval));
    }

    decltype(auto) operator[](const key_t &key) const { return m_node[key]; }

    bool has(const key_t &key) const { return m_node[key].IsDefined(); }
    template <typename T> bool has_typed(const key_t &key) const {
        if (!has(key)) {
            return false;
        }
        try {
            get_typed<T>(key);
        } catch (YAML::BadConversion) {
            return false;
        }
        return true;
    }

    template<typename T>
    void set(const key_t &key, T &&val) {
        m_node[key] = val;
    }

private:
    storage_t m_node;
};
}//namespace
