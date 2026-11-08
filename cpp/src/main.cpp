// mnist_infer: runs the exported CNN on the MNIST test set with the CPU and/or
// CUDA engine, checks the logits against PyTorch, and reports accuracy and speed.
//
//   mnist_infer [--artifacts DIR] [--engine cpu|cuda|all] [--batch N] [--repeat R]
//               [--no-fusion] [--profile]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine.h"

namespace {

struct Args {
    std::string artifacts = "artifacts";
    std::string engine = "all";
    int batch = 256;
    int repeat = 3;
    bool fusion = true;
    bool profile = false;
};

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(s + " needs a value");
            return argv[++i];
        };
        if (s == "--artifacts") a.artifacts = next();
        else if (s == "--engine") a.engine = next();
        else if (s == "--batch") a.batch = std::stoi(next());
        else if (s == "--repeat") a.repeat = std::stoi(next());
        else if (s == "--no-fusion") a.fusion = false;
        else if (s == "--profile") a.profile = true;
        else throw std::runtime_error("unknown argument " + s);
    }
    if (a.batch < 1 || a.repeat < 1) throw std::runtime_error("--batch and --repeat must be positive");
    return a;
}

// Returns false if the engine's logits disagree with PyTorch's.
bool evaluate(Engine& engine, const TestSet& ts, const Args& args) {
    const int n = static_cast<int>(ts.count);
    std::vector<float> logits(size_t(n) * ts.classes);

    engine.infer(ts.images.data(), std::min(n, args.batch), logits.data());  // warm-up

    double best_ms = 1e30;
    for (int r = 0; r < args.repeat; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        // Feed the test set in batches, like a server receiving requests
        for (int start = 0; start < n; start += args.batch) {
            const int bn = std::min(args.batch, n - start);
            engine.infer(ts.images.data() + size_t(start) * ts.rows * ts.cols, bn,
                         logits.data() + size_t(start) * ts.classes);
        }
        const auto t1 = std::chrono::steady_clock::now();
        best_ms = std::min(best_ms, std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    int correct = 0, agree = 0;
    float max_diff = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float* mine = &logits[size_t(i) * ts.classes];
        const float* ref = &ts.ref_logits[size_t(i) * ts.classes];
        const int pred = int(std::max_element(mine, mine + ts.classes) - mine);
        const int ref_pred = int(std::max_element(ref, ref + ts.classes) - ref);
        correct += pred == ts.labels[i];
        agree += pred == ref_pred;
        for (uint32_t c = 0; c < ts.classes; ++c) max_diff = std::max(max_diff, std::fabs(mine[c] - ref[c]));
    }

    const int batches = (n + args.batch - 1) / args.batch;
    std::printf("%s\n", engine.name().c_str());
    std::printf("  accuracy           %.4f  (%d / %d)\n", double(correct) / n, correct, n);
    std::printf("  matches PyTorch    %d / %d predictions, max |logit diff| = %.2e\n", agree, n, max_diff);
    std::printf("  time (best of %d)   %.2f ms for %d images  ->  %.0f images/s, %.3f ms per batch of %d\n",
                args.repeat, best_ms, n, n / (best_ms / 1e3), best_ms / batches, args.batch);
    std::printf("%s", engine.profile_report().c_str());

    // Float sums run in a different order than PyTorch's, so allow tiny differences
    const bool ok = max_diff < 1e-3f && agree == n;
    std::printf("  verification       %s\n\n", ok ? "PASS" : "FAIL");
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse(argc, argv);
        const Weights weights(args.artifacts + "/mnist_cnn.bin");
        const TestSet test(args.artifacts + "/mnist_test.bin");
        std::printf("loaded %u test images (%ux%u, %u classes)\n\n", test.count, test.rows, test.cols, test.classes);

        bool ok = true;
        if (args.engine == "cpu" || args.engine == "all") ok &= evaluate(*make_cpu_engine(weights), test, args);

        if (args.engine == "cuda" || args.engine == "all") {
#ifdef MNIST_HAVE_CUDA
            CudaOptions opts;
            opts.max_batch = args.batch;
            opts.fuse_pool = args.fusion;
            opts.profile = args.profile;
            ok &= evaluate(*make_cuda_engine(weights, opts), test, args);
#else
            if (args.engine == "cuda") throw std::runtime_error("this build has no CUDA support (nvcc not found)");
            std::printf("CUDA engine skipped: built without CUDA\n");
#endif
        }
        return ok ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return EXIT_FAILURE;
    }
}
