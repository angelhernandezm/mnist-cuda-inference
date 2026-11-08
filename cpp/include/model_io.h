// Readers for the artifacts produced by python/export.py (format documented there).
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

struct Tensor {
    std::vector<uint32_t> dims;
    std::vector<float> data;
};

class Weights {
public:
    explicit Weights(const std::string& path);

    // Returns the tensor, throwing if it is missing or its shape differs from `dims`.
    const Tensor& get(const std::string& name, const std::vector<uint32_t>& dims) const;

private:
    std::map<std::string, Tensor> tensors_;
};

struct TestSet {
    explicit TestSet(const std::string& path);

    uint32_t count = 0, rows = 0, cols = 0, classes = 0;
    std::vector<uint8_t> labels;    // [count]
    std::vector<float> images;      // [count, 1, rows, cols], already normalised
    std::vector<float> ref_logits;  // [count, classes], computed by PyTorch
};
