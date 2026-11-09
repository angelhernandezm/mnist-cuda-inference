// Inference engines for the MNIST CNN. Both take normalized images
// [n, 1, 28, 28] and write logits [n, 10], with the same maths as python/model.py.
#pragma once

#include <memory>
#include <string>

#include "model_io.h"

namespace mnist {

constexpr int kIn = 28;        // input height/width
constexpr int kC1 = 16;        // conv1 output channels
constexpr int kH1 = 26;        // conv1 output height/width
constexpr int kC2 = 32;        // conv2 output channels
constexpr int kH2 = 24;        // conv2 output height/width
constexpr int kP = 12;         // after 2x2 max pool
constexpr int kFlat = kC2 * kP * kP;  // 4608
constexpr int kHidden = 64;
constexpr int kClasses = 10;

}  // namespace mnist

class Engine {
public:
    virtual ~Engine() = default;
    virtual std::string name() const = 0;
    virtual void infer(const float* images, int n, float* logits) = 0;
    // Per-stage timing breakdown, if the engine collected one.
    virtual std::string profile_report() const { return {}; }
};

std::unique_ptr<Engine> make_cpu_engine(const Weights& w);

#ifdef MNIST_HAVE_CUDA
struct CudaOptions {
    int max_batch = 256;
    bool fuse_pool = true;  // conv2 + ReLU + max pool in one kernel
    bool profile = false;   // time each stage with CUDA events
};
std::unique_ptr<Engine> make_cuda_engine(const Weights& w, const CudaOptions& opts);
#endif
