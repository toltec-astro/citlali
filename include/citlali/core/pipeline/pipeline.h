#include <iostream>
#include <vector>
#include <memory>
#include <string>

// base class for pipeline components (components)
template <typename... Args>
class PipelineComponent {
public:
    virtual void process(Args&... args) = 0;  // all components process the same types of inputs by reference
    virtual void init() = 0;  // all components must implement an init() function

    //virtual void output(Args&...args) = 0; // all components must implement an output() function

    virtual ~PipelineComponent() = default; // Virtual destructor for safe deletion
};

// pipeline class that runs multiple components
template <typename... Args>
class Pipeline : public PipelineComponent<Args...> {
public:
    // add a new component to the pipeline with a key
    void add_component(const std::string& key, std::shared_ptr<PipelineComponent<Args...>> component) {
        components_.emplace_back(key, std::move(component));
    }

    void remove_component(const std::string& key) {
        components_.erase(
            std::remove_if(components_.begin(), components_.end(),
                           [&](const auto& pair) { return pair.first == key; }),
            components_.end()
            );
    }

    // clear all components from the pipeline
    void clear_components() {
        components_.clear();  // remove all elements from the components_ vector
    }

    std::optional<std::pair<int, PipelineComponent<Args...>*>> get_component(const std::string& key) {
        for (int i = 0; i < components_.size(); ++i) {
            auto& [k, component] = components_[i];
            if (k == key) {
                return std::make_pair(i, component.get());
            }
        }
        return std::nullopt;  // return nullopt if the key is not found
    }

    // insert or update a component with a specific key
    void insert_component(const std::string& key, std::shared_ptr<PipelineComponent<Args...>> component) {
        for (auto& [k, comp] : components_) {
            if (k == key) {
                comp = std::move(component);
                return;
            }
        }
        // if the key wasn't found, add the component
        components_.emplace_back(key, std::move(component));
    }

    // initialize all components in the pipeline
    void init() override {
        for (auto& [key, component] : components_) {
            component->init();  // call init() on each component
        }
    }

    // process the pipeline by invoking each component with the same input references
    void process(Args&... args) override {
        for (auto& [key, component] : components_) {
            component->process(args...);  // modify the inputs in place
        }
    }

    // process a subset of components
    void process_subset(const std::vector<std::string>& keys, Args&... args) {
        for (const auto& key : keys) {
            for (auto& [k, component] : components_) {
                if (k == key) {
                    component->process(args...);
                }
            }
        }
    }

    // Provide access to components from outside the pipeline
    const auto& get_components() const { return components_; }

private:
    std::vector<std::pair<std::string, std::shared_ptr<PipelineComponent<Args...>>>> components_;
};
