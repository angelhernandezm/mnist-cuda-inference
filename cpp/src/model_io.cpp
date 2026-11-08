#include "model_io.h"

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace {

class Reader {
public:
    explicit Reader(const std::string& path) : in_(path, std::ios::binary), path_(path) {
        if (!in_) throw std::runtime_error("cannot open " + path);
    }

    void bytes(void* dst, size_t n) {
        in_.read(static_cast<char*>(dst), static_cast<std::streamsize>(n));
        if (!in_) throw std::runtime_error("unexpected end of file in " + path_);
    }

    uint32_t u32() {
        uint32_t v;
        bytes(&v, sizeof v);  // files are little-endian, as are x86-64 and ARM64
        return v;
    }

    void magic(const char* expected) {
        char m[4];
        bytes(m, 4);
        if (std::memcmp(m, expected, 4) != 0) throw std::runtime_error(path_ + ": not a " + expected + " file");
        if (u32() != 1) throw std::runtime_error(path_ + ": unsupported version");
    }

    template <typename T>
    std::vector<T> array(size_t n) {
        std::vector<T> v(n);
        bytes(v.data(), n * sizeof(T));
        return v;
    }

private:
    std::ifstream in_;
    std::string path_;
};

std::string dims_str(const std::vector<uint32_t>& dims) {
    std::string s = "[";
    for (size_t i = 0; i < dims.size(); ++i) s += (i ? ", " : "") + std::to_string(dims[i]);
    return s + "]";
}

}  // namespace

Weights::Weights(const std::string& path) {
    Reader r(path);
    r.magic("MNCN");
    uint32_t count = r.u32();
    for (uint32_t i = 0; i < count; ++i) {
        std::string name(r.u32(), '\0');
        r.bytes(&name[0], name.size());
        Tensor t;
        t.dims.resize(r.u32());
        size_t elems = 1;
        for (auto& d : t.dims) elems *= (d = r.u32());
        t.data = r.array<float>(elems);
        tensors_[name] = std::move(t);
    }
}

const Tensor& Weights::get(const std::string& name, const std::vector<uint32_t>& dims) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) throw std::runtime_error("weights file has no tensor " + name);
    if (it->second.dims != dims)
        throw std::runtime_error(name + " has shape " + dims_str(it->second.dims) + ", expected " + dims_str(dims));
    return it->second;
}

TestSet::TestSet(const std::string& path) {
    Reader r(path);
    r.magic("MNTS");
    count = r.u32();
    rows = r.u32();
    cols = r.u32();
    classes = r.u32();
    labels = r.array<uint8_t>(count);
    images = r.array<float>(size_t(count) * rows * cols);
    ref_logits = r.array<float>(size_t(count) * classes);
}
