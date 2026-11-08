// Reference CPU engine: plain loops, written so the innermost loop walks
// contiguous memory and the compiler can vectorise it. Parallel over the batch
// with OpenMP when it is available.
#include <algorithm>
#include <vector>

#include "engine.h"

using namespace mnist;

namespace {

// Valid 3x3 convolution (stride 1, no padding) followed by ReLU, for one sample.
void conv3x3_relu(const float* in, const float* w, const float* b, float* out, int cin, int h, int wd, int cout) {
    const int ho = h - 2, wo = wd - 2;
    for (int oc = 0; oc < cout; ++oc) {
        float* o = out + oc * ho * wo;
        std::fill(o, o + ho * wo, b[oc]);
        for (int ic = 0; ic < cin; ++ic) {
            const float* src = in + ic * h * wd;
            const float* k = w + (oc * cin + ic) * 9;
            for (int ky = 0; ky < 3; ++ky)
                for (int kx = 0; kx < 3; ++kx) {
                    const float kv = k[ky * 3 + kx];
                    for (int oy = 0; oy < ho; ++oy) {
                        const float* srow = src + (oy + ky) * wd + kx;
                        float* orow = o + oy * wo;
                        for (int ox = 0; ox < wo; ++ox) orow[ox] += kv * srow[ox];
                    }
                }
        }
        for (int i = 0; i < ho * wo; ++i) o[i] = std::max(o[i], 0.0f);
    }
}

void maxpool2(const float* in, float* out, int c, int h, int wd) {
    const int ph = h / 2, pw = wd / 2;
    for (int ch = 0; ch < c; ++ch)
        for (int y = 0; y < ph; ++y)
            for (int x = 0; x < pw; ++x) {
                const float* p = in + (ch * h + 2 * y) * wd + 2 * x;
                out[(ch * ph + y) * pw + x] = std::max(std::max(p[0], p[1]), std::max(p[wd], p[wd + 1]));
            }
}

void linear(const float* in, const float* w, const float* b, float* out, int nin, int nout, bool relu) {
    for (int o = 0; o < nout; ++o) {
        const float* row = w + o * nin;
        float acc = 0.0f;
        for (int i = 0; i < nin; ++i) acc += row[i] * in[i];
        acc += b[o];
        out[o] = relu ? std::max(acc, 0.0f) : acc;
    }
}

class CpuEngine : public Engine {
public:
    explicit CpuEngine(const Weights& w)
        : c1w_(w.get("conv1.weight", {kC1, 1, 3, 3}).data), c1b_(w.get("conv1.bias", {kC1}).data),
          c2w_(w.get("conv2.weight", {kC2, kC1, 3, 3}).data), c2b_(w.get("conv2.bias", {kC2}).data),
          f1w_(w.get("fc1.weight", {kHidden, kFlat}).data), f1b_(w.get("fc1.bias", {kHidden}).data),
          f2w_(w.get("fc2.weight", {kClasses, kHidden}).data), f2b_(w.get("fc2.bias", {kClasses}).data) {}

    std::string name() const override { return "CPU reference"; }

    void infer(const float* images, int n, float* logits) override {
#pragma omp parallel
        {
            // Per-thread scratch buffers, reused for every sample the thread handles
            std::vector<float> a1(kC1 * kH1 * kH1), a2(kC2 * kH2 * kH2), pooled(kFlat), hidden(kHidden);
#pragma omp for schedule(static)
            for (int i = 0; i < n; ++i) {
                conv3x3_relu(images + i * kIn * kIn, c1w_.data(), c1b_.data(), a1.data(), 1, kIn, kIn, kC1);
                conv3x3_relu(a1.data(), c2w_.data(), c2b_.data(), a2.data(), kC1, kH1, kH1, kC2);
                maxpool2(a2.data(), pooled.data(), kC2, kH2, kH2);
                // Flatten is free: pooled is already laid out as [C, H, W], like torch.flatten
                linear(pooled.data(), f1w_.data(), f1b_.data(), hidden.data(), kFlat, kHidden, true);
                linear(hidden.data(), f2w_.data(), f2b_.data(), logits + i * kClasses, kHidden, kClasses, false);
            }
        }
    }

private:
    std::vector<float> c1w_, c1b_, c2w_, c2b_, f1w_, f1b_, f2w_, f2b_;
};

}  // namespace

std::unique_ptr<Engine> make_cpu_engine(const Weights& w) { return std::make_unique<CpuEngine>(w); }
