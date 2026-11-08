// CUDA engine: four hand-written kernels for the whole network.
//
//   conv1 + ReLU             conv3x3_relu        (input tile and weights in shared memory)
//   conv2 + ReLU + max pool  conv3x3_relu_pool2  (fused: the 24x24x32 map never reaches global memory)
//   fc1 + ReLU               linear              (tiled matrix multiply)
//   fc2                      linear
//
// With --no-fusion, conv2 runs as conv3x3_relu followed by a separate maxpool2
// kernel, so the effect of fusion can be measured.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine.h"

using namespace mnist;

#define CUDA_CHECK(call)                                                                       \
    do {                                                                                       \
        cudaError_t err_ = (call);                                                             \
        if (err_ != cudaSuccess)                                                               \
            throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(err_));    \
    } while (0)

namespace {

constexpr int kConvTile = 16;  // conv3x3_relu: 16x16 outputs per block
constexpr int kPoolTile = 8;   // conv3x3_relu_pool2: 8x8 pooled outputs (16x16 conv outputs) per block
constexpr int kGemmTile = 16;

// One block computes a TILE x TILE patch of one output channel for one sample.
// Weights for that channel (cin*9 floats) and a (TILE+2)^2 input tile per input
// channel are staged in shared memory, so each input value is read from global
// memory once per block instead of nine times.
template <int TILE>
__global__ void conv3x3_relu(const float* __restrict__ in, const float* __restrict__ w,
                             const float* __restrict__ b, float* __restrict__ out,
                             int cin, int h, int wd, int cout) {
    extern __shared__ float smem[];
    float* sw = smem;              // [cin * 9]
    float* tile = smem + cin * 9;  // [(TILE+2) * (TILE+2)]
    constexpr int IT = TILE + 2;

    const int n = blockIdx.z / cout, oc = blockIdx.z % cout;
    const int ho = h - 2, wo = wd - 2;
    const int tx = threadIdx.x, ty = threadIdx.y, tid = ty * TILE + tx;
    const int ox = blockIdx.x * TILE + tx, oy = blockIdx.y * TILE + ty;

    for (int i = tid; i < cin * 9; i += TILE * TILE) sw[i] = w[oc * cin * 9 + i];

    float acc = b[oc];
    for (int ic = 0; ic < cin; ++ic) {
        __syncthreads();  // weights loaded / previous tile no longer in use
        const float* src = in + (size_t(n) * cin + ic) * h * wd;
        for (int i = tid; i < IT * IT; i += TILE * TILE) {
            const int gy = blockIdx.y * TILE + i / IT, gx = blockIdx.x * TILE + i % IT;
            tile[i] = (gy < h && gx < wd) ? src[gy * wd + gx] : 0.0f;
        }
        __syncthreads();

        const float* k = sw + ic * 9;
#pragma unroll
        for (int ky = 0; ky < 3; ++ky)
#pragma unroll
            for (int kx = 0; kx < 3; ++kx) acc += tile[(ty + ky) * IT + tx + kx] * k[ky * 3 + kx];
    }

    if (oy < ho && ox < wo) out[(size_t(n) * cout + oc) * ho * wo + oy * wo + ox] = fmaxf(acc, 0.0f);
}

// Fused conv3x3 + ReLU + 2x2 max pool. Each thread computes the four conv
// outputs under one pooling window and writes only their maximum. ReLU and max
// commute (both are monotonic), so relu(max(a,b,c,d)) == max(relu(a),...).
template <int TILE>
__global__ void conv3x3_relu_pool2(const float* __restrict__ in, const float* __restrict__ w,
                                   const float* __restrict__ b, float* __restrict__ out,
                                   int cin, int h, int wd, int cout) {
    extern __shared__ float smem[];
    float* sw = smem;
    float* tile = smem + cin * 9;
    constexpr int IT = 2 * TILE + 2;

    const int n = blockIdx.z / cout, oc = blockIdx.z % cout;
    const int ph = (h - 2) / 2, pw = (wd - 2) / 2;
    const int tx = threadIdx.x, ty = threadIdx.y, tid = ty * TILE + tx;
    const int px = blockIdx.x * TILE + tx, py = blockIdx.y * TILE + ty;

    for (int i = tid; i < cin * 9; i += TILE * TILE) sw[i] = w[oc * cin * 9 + i];

    float acc[4];
    for (auto& a : acc) a = b[oc];

    for (int ic = 0; ic < cin; ++ic) {
        __syncthreads();
        const float* src = in + (size_t(n) * cin + ic) * h * wd;
        for (int i = tid; i < IT * IT; i += TILE * TILE) {
            const int gy = blockIdx.y * 2 * TILE + i / IT, gx = blockIdx.x * 2 * TILE + i % IT;
            tile[i] = (gy < h && gx < wd) ? src[gy * wd + gx] : 0.0f;
        }
        __syncthreads();

        const float* k = sw + ic * 9;
#pragma unroll
        for (int dy = 0; dy < 2; ++dy)
#pragma unroll
            for (int dx = 0; dx < 2; ++dx)
#pragma unroll
                for (int ky = 0; ky < 3; ++ky)
#pragma unroll
                    for (int kx = 0; kx < 3; ++kx)
                        acc[dy * 2 + dx] += tile[(2 * ty + dy + ky) * IT + 2 * tx + dx + kx] * k[ky * 3 + kx];
    }

    if (py < ph && px < pw) {
        const float m = fmaxf(fmaxf(acc[0], acc[1]), fmaxf(acc[2], acc[3]));
        out[(size_t(n) * cout + oc) * ph * pw + py * pw + px] = fmaxf(m, 0.0f);
    }
}

__global__ void maxpool2(const float* __restrict__ in, float* __restrict__ out, int nc, int h, int wd) {
    const int ph = h / 2, pw = wd / 2;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nc * ph * pw) return;
    const int x = idx % pw, y = (idx / pw) % ph, c = idx / (pw * ph);
    const float* p = in + (size_t(c) * h + 2 * y) * wd + 2 * x;
    out[idx] = fmaxf(fmaxf(p[0], p[1]), fmaxf(p[wd], p[wd + 1]));
}

// y[n, o] = x[n, :] . W[o, :] + b[o]   (W is [out, in], like nn.Linear)
// Classic shared-memory tiling: each block loads a TILE x TILE slice of x and of
// W per step, and each thread accumulates one output.
template <int TILE>
__global__ void linear(const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b,
                       float* __restrict__ y, int n, int nin, int nout, bool relu) {
    __shared__ float xs[TILE][TILE];
    __shared__ float ws[TILE][TILE + 1];  // +1 column avoids shared-memory bank conflicts on ws[tx][k]

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;  // sample
    const int col = blockIdx.x * TILE + tx;  // output feature
    const int wrow = blockIdx.x * TILE + ty;

    float acc = 0.0f;
    for (int k0 = 0; k0 < nin; k0 += TILE) {
        xs[ty][tx] = (row < n && k0 + tx < nin) ? x[size_t(row) * nin + k0 + tx] : 0.0f;
        ws[ty][tx] = (wrow < nout && k0 + tx < nin) ? w[size_t(wrow) * nin + k0 + tx] : 0.0f;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TILE; ++k) acc += xs[ty][k] * ws[tx][k];
        __syncthreads();
    }

    if (row < n && col < nout) {
        const float v = acc + b[col];
        y[size_t(row) * nout + col] = relu ? fmaxf(v, 0.0f) : v;
    }
}

int div_up(int a, int b) { return (a + b - 1) / b; }

class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t floats) { CUDA_CHECK(cudaMalloc(&ptr_, floats * sizeof(float))); }
    DeviceBuffer(const std::vector<float>& host) : DeviceBuffer(host.size()) {
        CUDA_CHECK(cudaMemcpy(ptr_, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice));
    }
    ~DeviceBuffer() { cudaFree(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    float* get() const { return ptr_; }

private:
    float* ptr_ = nullptr;
};

class CudaEngine : public Engine {
public:
    CudaEngine(const Weights& w, const CudaOptions& opts)
        : opts_(opts),
          c1w_(w.get("conv1.weight", {kC1, 1, 3, 3}).data), c1b_(w.get("conv1.bias", {kC1}).data),
          c2w_(w.get("conv2.weight", {kC2, kC1, 3, 3}).data), c2b_(w.get("conv2.bias", {kC2}).data),
          f1w_(w.get("fc1.weight", {kHidden, kFlat}).data), f1b_(w.get("fc1.bias", {kHidden}).data),
          f2w_(w.get("fc2.weight", {kClasses, kHidden}).data), f2b_(w.get("fc2.bias", {kClasses}).data),
          in_(size_t(opts.max_batch) * kIn * kIn),
          a1_(size_t(opts.max_batch) * kC1 * kH1 * kH1),
          a2_(opts.fuse_pool ? 1 : size_t(opts.max_batch) * kC2 * kH2 * kH2),
          pooled_(size_t(opts.max_batch) * kFlat),
          hidden_(size_t(opts.max_batch) * kHidden),
          out_(size_t(opts.max_batch) * kClasses) {
        CUDA_CHECK(cudaGetDeviceProperties(&props_, 0));
        CUDA_CHECK(cudaStreamCreate(&stream_));
        for (auto& e : events_) CUDA_CHECK(cudaEventCreate(&e));
    }

    ~CudaEngine() override {
        for (auto& e : events_) cudaEventDestroy(e);
        cudaStreamDestroy(stream_);
    }

    std::string name() const override {
        return std::string("CUDA (") + props_.name + (opts_.fuse_pool ? ", fused conv2+pool)" : ", unfused)");
    }

    void infer(const float* images, int n, float* logits) override {
        for (int start = 0; start < n; start += opts_.max_batch) {
            const int bn = std::min(opts_.max_batch, n - start);
            run_batch(images + size_t(start) * kIn * kIn, bn, logits + size_t(start) * kClasses);
        }
    }

    std::string profile_report() const override {
        if (!opts_.profile || batches_ == 0) return {};
        static const char* names[kStages] = {"copy in", "conv1+relu", "conv2+relu+pool", "fc1+relu", "fc2",
                                             "copy out"};
        float total = 0;
        for (float t : stage_ms_) total += t;
        std::ostringstream s;
        s << "  per-stage GPU time over " << batches_ << " batches:\n";
        for (int i = 0; i < kStages; ++i) {
            char line[128];
            std::snprintf(line, sizeof line, "    %-16s %9.3f ms  %5.1f%%\n", names[i], stage_ms_[i],
                          100.0f * stage_ms_[i] / total);
            s << line;
        }
        return s.str();
    }

private:
    static constexpr int kStages = 6;

    void mark(int i) {
        if (opts_.profile) CUDA_CHECK(cudaEventRecord(events_[i], stream_));
    }

    void run_batch(const float* images, int n, float* logits) {
        mark(0);
        CUDA_CHECK(cudaMemcpyAsync(in_.get(), images, size_t(n) * kIn * kIn * sizeof(float),
                                   cudaMemcpyHostToDevice, stream_));
        mark(1);

        const dim3 conv_block(kConvTile, kConvTile);
        const size_t smem1 = (1 * 9 + (kConvTile + 2) * (kConvTile + 2)) * sizeof(float);
        conv3x3_relu<kConvTile><<<dim3(div_up(kH1, kConvTile), div_up(kH1, kConvTile), n * kC1), conv_block, smem1,
                                  stream_>>>(in_.get(), c1w_.get(), c1b_.get(), a1_.get(), 1, kIn, kIn, kC1);
        CUDA_CHECK(cudaGetLastError());
        mark(2);

        if (opts_.fuse_pool) {
            const size_t smem2 = (kC1 * 9 + (2 * kPoolTile + 2) * (2 * kPoolTile + 2)) * sizeof(float);
            conv3x3_relu_pool2<kPoolTile>
                <<<dim3(div_up(kP, kPoolTile), div_up(kP, kPoolTile), n * kC2), dim3(kPoolTile, kPoolTile), smem2,
                   stream_>>>(a1_.get(), c2w_.get(), c2b_.get(), pooled_.get(), kC1, kH1, kH1, kC2);
        } else {
            const size_t smem2 = (kC1 * 9 + (kConvTile + 2) * (kConvTile + 2)) * sizeof(float);
            conv3x3_relu<kConvTile><<<dim3(div_up(kH2, kConvTile), div_up(kH2, kConvTile), n * kC2), conv_block,
                                      smem2, stream_>>>(a1_.get(), c2w_.get(), c2b_.get(), a2_.get(), kC1, kH1,
                                                        kH1, kC2);
            CUDA_CHECK(cudaGetLastError());
            const int total = n * kC2 * kP * kP;
            maxpool2<<<div_up(total, 256), 256, 0, stream_>>>(a2_.get(), pooled_.get(), n * kC2, kH2, kH2);
        }
        CUDA_CHECK(cudaGetLastError());
        mark(3);

        const dim3 gemm_block(kGemmTile, kGemmTile);
        linear<kGemmTile><<<dim3(div_up(kHidden, kGemmTile), div_up(n, kGemmTile)), gemm_block, 0, stream_>>>(
            pooled_.get(), f1w_.get(), f1b_.get(), hidden_.get(), n, kFlat, kHidden, true);
        CUDA_CHECK(cudaGetLastError());
        mark(4);

        linear<kGemmTile><<<dim3(div_up(kClasses, kGemmTile), div_up(n, kGemmTile)), gemm_block, 0, stream_>>>(
            hidden_.get(), f2w_.get(), f2b_.get(), out_.get(), n, kHidden, kClasses, false);
        CUDA_CHECK(cudaGetLastError());
        mark(5);

        CUDA_CHECK(cudaMemcpyAsync(logits, out_.get(), size_t(n) * kClasses * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream_));
        mark(6);
        CUDA_CHECK(cudaStreamSynchronize(stream_));

        if (opts_.profile) {
            for (int i = 0; i < kStages; ++i) {
                float ms;
                CUDA_CHECK(cudaEventElapsedTime(&ms, events_[i], events_[i + 1]));
                stage_ms_[i] += ms;
            }
            ++batches_;
        }
    }

    CudaOptions opts_;
    cudaDeviceProp props_{};
    cudaStream_t stream_{};
    cudaEvent_t events_[kStages + 1]{};
    float stage_ms_[kStages]{};
    int batches_ = 0;

    DeviceBuffer c1w_, c1b_, c2w_, c2b_, f1w_, f1b_, f2w_, f2b_;
    DeviceBuffer in_, a1_, a2_, pooled_, hidden_, out_;
};

}  // namespace

std::unique_ptr<Engine> make_cuda_engine(const Weights& w, const CudaOptions& opts) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
        throw std::runtime_error("no CUDA device available");
    // The conv kernels put (sample, channel) on grid dimension z, which is capped at 65535.
    // Larger requests are still served; infer() just splits them into more launches.
    CudaOptions o = opts;
    o.max_batch = std::max(1, std::min(o.max_batch, 65535 / kC2));
    return std::make_unique<CudaEngine>(w, o);
}
