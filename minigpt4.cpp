#include "minigpt4.h"

#include <iostream>
#include <algorithm>
#include <iterator>
#include <vector>
#include <filesystem>
#include <sstream>
#include <csignal>
#include <fstream>
#include <codecvt>
#include <numeric>
#include <optional>
#include <thread>
#include <span>
#include <variant>
#include <any>
#include <ranges>
#include <cstring>
#include <map>
#include <chrono>

#include "llama.h"
#include "ggml.h"

#include "fmt/core.h"
#include "fmt/ranges.h"
#include "ankerl/unordered_dense.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>
#include <spdlog/fmt/bin_to_hex.h>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include <expected.hpp>

#include <magic_enum.hpp>

#ifdef MINIGPT4_BUILD_WITH_OPENCV
    #include <opencv2/opencv.hpp>
    #include <PillowResize.hpp>
#endif

/////////////////////
/// PLATFORM INCLUDE
/////////////////////

#ifdef __has_include
#if __has_include(<unistd.h>)
#include <unistd.h>
#if defined(_POSIX_MAPPED_FILES)
#include <sys/mman.h>
#endif
#if defined(_POSIX_MEMLOCK_RANGE)
#include <sys/resource.h>
#endif
#endif
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#include <stdio.h>
#endif

/////////////////////
/// FORWARDS
/////////////////////

namespace fs = std::filesystem;
using namespace std::chrono_literals;

template <typename K, typename V>
using HashMap = ankerl::unordered_dense::map<K, V>;

constexpr auto PAGE_SIZE = 4096u;

/////////////////////
/// DEFINITIONS
/////////////////////

constexpr std::string_view EXPECTED_HEADER = "ggml";
constexpr auto MB = 1024u * 1024u;
constexpr auto GB = 1024u * MB;
constexpr auto bytes_to_mb = [](auto bytes)
{ return static_cast<double>(bytes) / MB; };

enum MiniGPT4Error : int
{
    None,
    LoadModelFileHeader,
    LoadModelFileVersion,
    LoadModelMiniGPT4DataType,
    LoadLanguageModel,
    OpenImage,
    ImageSize,
    MmapSupport,
    FailedToAddString,
    LLamaProjectionEmbeddingInvalidSize,
    FailedToAddEmbedding,
    EosToken,
    Eos,
    ImageNot224_244_3,
    ImageNotF32,
    ImageChannelsExpectedRGB,
    ImageFormatExpectedU8,
    PathDoesNotExist,
    DumpModelFileOpen,
    OpenCVNotLinked,
};

/////////////////////
/// CONSTANT GLOBALS
/////////////////////

constexpr std::size_t PATCH_SIZE = 16;

constexpr std::size_t NUM_ATTENTION_HEADS = 12;
constexpr std::size_t ATTENTION_HEAD_SIZE = 64;
constexpr std::size_t ALL_HEAD_SIZE = 768;

constexpr std::size_t IMAGE_RESIZE = 224;

constexpr std::size_t LLAMA_PROJECTION_EMBEDDING_SIZE1 = 32;
constexpr std::size_t LLAMA_PROJECTION_HIDDEN_SIZE_7B = 4096;
constexpr std::size_t LLAMA_PROJECTION_HIDDEN_SIZE_13B = 5120;
constexpr std::size_t LLAMA_PROJECTION_EMBEDDING_SIZE_7B = LLAMA_PROJECTION_HIDDEN_SIZE_7B * LLAMA_PROJECTION_EMBEDDING_SIZE1;
constexpr std::size_t LLAMA_PROJECTION_EMBEDDING_SIZE_13B = LLAMA_PROJECTION_HIDDEN_SIZE_13B * LLAMA_PROJECTION_EMBEDDING_SIZE1;

constexpr std::string_view SYSTEM_PROMPT = R"(Give the following image: <Img>ImageContent</Img>. You will be able to see the image once I provide it to you. Please answer my questions.###)";
constexpr std::string_view EOS_TOKEN_SUFFIX = "##";
constexpr std::string_view EOS_SUFFIX = "###";

constexpr float TORCH_FLOAT_FIFO_MIN = -3.40282e+38;

constexpr std::size_t RGB_CHANNELS = 3;
constexpr static std::size_t MAX_SCRATCH_BUFFERS = 1;

/////////////////////
/// MUTABLE GLOBALS
/////////////////////

static MiniGPT4Verbosity global_verbosity;

/////////////////////
/// Memory sizes
/////////////////////

enum class ModelType
{
    Unknown,
    Vicuna7B,
    Vicuna13B,
};

// TODO: dynamically determine sizes
const static HashMap<ModelType, std::size_t> model_type_to_compute_size = {
    {ModelType::Vicuna7B, 100 * MB},
    {ModelType::Vicuna13B, 100 * MB},
};

const static HashMap<ModelType, std::size_t> model_type_to_scratch_size = {
    {ModelType::Vicuna7B, 2814 * MB},
    {ModelType::Vicuna13B, 2815 * MB},
};

/////////////////////
/// UTILS
/////////////////////

#define CCAT(a, b) a##b
#define CAT(a, b) CCAT(a, b)

#define STRINGIFY2(x) #x
#define STRINGIFY(x) STRINGIFY2(x)

#define UNIQUIFY2(x) CAT(x, __LINE__)
#define UNIQUIFY(x) UNIQUIFY2(x)

#ifdef USE_PREFIX
#define PREFIX "{}:{}:{} "
#define PREFIX_ENTRIES __FILE__, __FUNCTION__, __LINE__
#else
#define PREFIX
#define PREFIX_ENTRIES __FILE__
#endif

#define DEBUG(...)                                                                     \
    do                                                                                 \
    {                                                                                  \
        if (global_verbosity >= MiniGPT4Verbosity::MINIGPT4_VERBOSITY_DEBUG)           \
        {                                                                              \
            auto UNIQUIFY(log_header) = fmt::format(PREFIX "DEBUG: ", PREFIX_ENTRIES); \
            auto UNIQUIFY(other_info) = fmt::format(__VA_ARGS__);                      \
            std::cout << UNIQUIFY(log_header) << UNIQUIFY(other_info) << "\n";         \
        }                                                                              \
    } while (0)

#define INFO(...)                                                                     \
    do                                                                                \
    {                                                                                 \
        if (global_verbosity >= MiniGPT4Verbosity::MINIGPT4_VERBOSITY_INFO)           \
        {                                                                             \
            auto UNIQUIFY(log_header) = fmt::format(PREFIX "INFO: ", PREFIX_ENTRIES); \
            auto UNIQUIFY(other_info) = fmt::format(__VA_ARGS__);                     \
            std::cout << UNIQUIFY(log_header) << UNIQUIFY(other_info) << "\n";        \
        }                                                                             \
    } while (0)

#define ERR(...)                                                                       \
    do                                                                                 \
    {                                                                                  \
        if (global_verbosity >= MiniGPT4Verbosity::MINIGPT4_VERBOSITY_ERROR)           \
        {                                                                              \
            auto UNIQUIFY(log_header) = fmt::format(PREFIX "ERROR: ", PREFIX_ENTRIES); \
            auto UNIQUIFY(other_info) = fmt::format(__VA_ARGS__);                      \
            std::cerr << UNIQUIFY(log_header) << UNIQUIFY(other_info) << "\n";         \
        }                                                                              \
    } while (0)

#define PANIC(...)    \
    ERR(__VA_ARGS__); \
    exit(-1);

#ifndef NDEBUG

#define ASSERT(result, ...)                                                                                     \
    do                                                                                                          \
    {                                                                                                           \
        if (!(result))                                                                                          \
        {                                                                                                       \
            auto UNIQUIFY(log_header) = fmt::format(PREFIX "ASSERT: [{}] ", PREFIX_ENTRIES, STRINGIFY(result)); \
            auto UNIQUIFY(other_info) = fmt::format(__VA_ARGS__);                                               \
            std::cerr << UNIQUIFY(log_header) << UNIQUIFY(other_info) << "\n";                                  \
            exit(-1);                                                                                           \
        }                                                                                                       \
    } while (0)

#else
#define ASSERT(result, ...)
#endif

struct BufferView
{
    explicit BufferView(uint8_t *addr = nullptr, std::size_t size = 0) : addr(addr), size(size) {}

    bool valid() const
    {
        return addr != nullptr && size != 0;
    }

    template <typename T>
    T *As()
    {
        return reinterpret_cast<T *>(addr);
    }

    uint8_t *addr{};
    std::size_t size{};
};

struct Buffer : public BufferView
{
    explicit Buffer() = default;
    explicit Buffer(std::size_t size_)
    {
        size = size_;
        if (size)
        {
            buf.resize(size);
            addr = buf.data();
        }
    }

    std::vector<uint8_t> buf{};
};

struct Timer
{
    explicit Timer() {}
    double elapsed_us()
    {
        auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
        return diff;
    }

    const std::chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();
};

struct LoggingTimer : public Timer
{
    explicit LoggingTimer(std::string_view s_ = "") : s(std::string(s_)) {}
    ~LoggingTimer()
    {
        auto diff = elapsed_us();
        if (global_verbosity >= MiniGPT4Verbosity::MINIGPT4_VERBOSITY_INFO)
        {
            INFO("{} took {} ms to complete", s, diff);
        }
    }

    std::string s;
};

/////////////////////
/// FILE UTILS
/////////////////////

class MMappedFile
{
public:
    explicit MMappedFile() = default;
#ifdef _POSIX_MAPPED_FILES
    static constexpr bool SUPPORTED = true;
    void load(fs::path p, bool prefetch = true)
    {
        fp = std::fopen(p.string().c_str(), "rb");
        ASSERT(fp != nullptr, "file does not exist {}", p.string());
        std::fseek(fp, 0, SEEK_END);
        view.size = std::ftell(fp);
        std::fseek(fp, 0, SEEK_SET);

        int fd = fileno(fp);
        int flags = MAP_SHARED;
#ifdef __linux__
        flags |= MAP_POPULATE;
#endif
        view.addr = reinterpret_cast<uint8_t *>(mmap(NULL, view.size, PROT_READ, flags, fd, 0));
        if (view.addr == MAP_FAILED)
        {
            ERR("mmap failed: {}", strerror(errno));
        }

        if (prefetch)
        {
            // Advise the kernel to preload the mapped memory
            if (madvise(view.addr, view.size, MADV_WILLNEED))
            {
                ERR("warning: madvise(.., MADV_WILLNEED) failed: {}\n",
                    strerror(errno));
            }
        }
    }

    ~MMappedFile()
    {
        fclose(fp);
        munmap(view.addr, view.size);
    }
#elif defined(_WIN32)
    static constexpr bool SUPPORTED = true;

    void load(fs::path p, bool prefetch = true)
    {
        fp = std::fopen(p.string().c_str(), "rb");
        ASSERT(fp != nullptr, "file does not exist {}", p.string());
        std::fseek(fp, 0, SEEK_END);
        view.size = _ftelli64(fp);
        std::fseek(fp, 0, SEEK_SET);

        HANDLE hFile = (HANDLE)_get_osfhandle(_fileno(fp));

        HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
        DWORD error = GetLastError();

        if (hMapping == NULL)
        {
            PANIC("CreateFileMappingA failed: {}", error);
        }

        view.addr = reinterpret_cast<uint8_t *>(MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0));
        error = GetLastError();

        if (view.addr == NULL)
        {
            PANIC("MapViewOfFile failed: {}", error);
        }

#if _WIN32_WINNT >= _WIN32_WINNT_WIN8
        if (prefetch)
        {
            // Advise the kernel to preload the mapped memory
            WIN32_MEMORY_RANGE_ENTRY range;
            range.VirtualAddress = view.addr;
            range.NumberOfBytes = (SIZE_T)view.size;
            if (!PrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0))
            {
                INFO("PrefetchVirtualMemory failed: {}", GetLastError());
            }
        }
#else
#pragma message("warning: You are building for pre-Windows 8; prefetch not supported")
#endif // _WIN32_WINNT >= _WIN32_WINNT_WIN8
        CloseHandle(hMapping);
    }

    ~MMappedFile()
    {
        fclose(fp);
        if (!UnmapViewOfFile(view.addr))
        {
            PANIC("UnmapViewOfFile failed: {}", GetLastError());
        }
    }
#else
    static constexpr bool SUPPORTED = false;

    void load(fs::path p, bool prefetch = true)
    {
        PANIC("mmap not supported");
    }
#endif
protected:
    BufferView view;
    FILE *fp{};
};

class MMapReader : public MMappedFile
{
public:
    explicit MMapReader() = default;

    template <typename T = uint8_t *>
    T base_addr()
    {
        return reinterpret_cast<T>(view.addr);
    }

    template <typename T = uint8_t *>
    T current_addr()
    {
        return reinterpret_cast<T>(view.addr + pos);
    }

    std::size_t tell()
    {
        return pos;
    }

    void seek(std::size_t new_pos)
    {
        pos = new_pos;
        ASSERT(pos <= view.size, "Out of bounds for seeking {} > {}", pos, view.size);
    }

    void seek_to_alignment(std::size_t alignment)
    {
        if ((alignment - 1) & pos)
        {
            pos = (pos + alignment) & ~(alignment - 1);
        }
    }

    bool is_eof() const
    {
        ASSERT(pos <= view.size, "Out of bounds for eof {} > {}", pos, view.size);
        return pos == view.size;
    }

    void add_pos(std::size_t amount)
    {
        pos += amount;
        ASSERT(pos <= view.size, "Out of bounds for reading {} > {}", pos, view.size);
    }

    template <typename T>
    T &read_as()
    {
        T *t = current_addr<T *>();
        add_pos(sizeof(T));
        return *t;
    }

    int32_t read_s4()
    {
        return read_as<int32_t>();
    }

    std::string_view read_bytes(std::size_t len)
    {
        auto start = current_addr<const char *>();
        std::string_view s(start, len);
        add_pos(len);
        return s;
    }

    std::string_view read_string()
    {
        auto string_length = read_s4();
        auto s = read_bytes(string_length);
        return s;
    }

    template <typename T>
    void read_bytes_into(T buf, std::size_t len)
    {
        static_assert(std::is_pointer_v<T>, "T must be a pointer");
        auto start = current_addr();
        std::copy(start, start + len, buf);
        add_pos(len);
    }

private:
    std::size_t pos{};
};

/////////////////////
/// Debug
/////////////////////

void WriteDump(ggml_tensor *t)
{
    std::ofstream f("out.txt", std::ios::trunc | std::ios::ate);
    std::vector<std::size_t> sizes{(size_t *)&t->ne[0], (size_t *)&t->ne[4]};

    auto total = sizes[0] * sizes[1] * sizes[2] * sizes[3];
    for (auto i = 0; i < total; i++)
    {
        auto *d = (float *)t->data;
        auto dd = d[i];
        f << fmt::format("{},", dd);
    }
    fmt::print("TOTAL {}\n", total);
    f.close();
    exit(-2);
}

#define DUMP_TENSOR(cur)                         \
    {                                            \
        auto xxx = cur;                          \
        xxx = ggml_cont(ctx0, xxx);              \
        ggml_set_name(xxx, "dump");              \
        use_scratch(-1);                         \
        struct ggml_cgraph gf = {};              \
        gf.n_threads = 16;                       \
        ggml_build_forward_expand(&gf, xxx);     \
        ggml_graph_compute(ctx0, &gf);           \
        auto *t = ggml_get_tensor(ctx0, "dump"); \
        WriteDump(t);                            \
    }

/////////////////////
/// Tensors
/////////////////////

tl::expected<ggml_type, MiniGPT4Error> data_type_to_ggml_type(MiniGPT4DataType data_type)
{
    ggml_type type;
    switch (data_type)
    {
    case MiniGPT4DataType::F16:
    {
        type = GGML_TYPE_F16;
        break;
    }
    case MiniGPT4DataType::F32:
    {
        type = GGML_TYPE_F32;
        break;
    }
    case MiniGPT4DataType::I32:
    {
        type = GGML_TYPE_I32;
        break;
    }
    case MiniGPT4DataType::L64:
    {
        ERR("Unsupported MiniGPT4DataType {}", magic_enum::enum_name(data_type));
        return tl::unexpected(MiniGPT4Error::LoadModelMiniGPT4DataType);
        break;
    }
    case MiniGPT4DataType::Q4_0:
    {
        type = GGML_TYPE_Q4_0;
        break;
    }
    case MiniGPT4DataType::Q4_1:
    {
        type = GGML_TYPE_Q4_1;
        break;
    }
    case MiniGPT4DataType::Q5_0:
    {
        type = GGML_TYPE_Q5_0;
        break;
    }
    case MiniGPT4DataType::Q5_1:
    {
        type = GGML_TYPE_Q5_1;
        break;
    }
    case MiniGPT4DataType::Q8_0:
    {
        type = GGML_TYPE_Q8_0;
        break;
    }
    case MiniGPT4DataType::Q8_1:
    {
        type = GGML_TYPE_Q8_1;
        break;
    }
    case MiniGPT4DataType::Q2_K:
    {
        type = GGML_TYPE_Q2_K;
        break;
    }
    case MiniGPT4DataType::Q3_K:
    {
        type = GGML_TYPE_Q3_K;
        break;
    }
    case MiniGPT4DataType::Q4_K:
    {
        type = GGML_TYPE_Q4_K;
        break;
    }
    case MiniGPT4DataType::Q5_K:
    {
        type = GGML_TYPE_Q5_K;
        break;
    }
    case MiniGPT4DataType::Q6_K:
    {
        type = GGML_TYPE_Q6_K;
        break;
    }
    case MiniGPT4DataType::Q8_K:
    {
        type = GGML_TYPE_Q8_K;
        break;
    }
    default:
    {
        ERR("Unsupported MiniGPT4DataType {}", magic_enum::enum_name(data_type));
        return tl::unexpected(MiniGPT4Error::LoadModelMiniGPT4DataType);
        break;
    }
    }
    return type;
}

tl::expected<MiniGPT4DataType, MiniGPT4Error> ggml_type_to_data_type(ggml_type t)
{
    MiniGPT4DataType data_type;
    switch (t)
    {
    case GGML_TYPE_F16:
    {
        data_type = MiniGPT4DataType::F16;
        break;
    }
    case GGML_TYPE_F32:
    {
        data_type = MiniGPT4DataType::F32;
        break;
    }
    case GGML_TYPE_I32:
    {
        data_type = MiniGPT4DataType::I32;
        break;
    }
    case GGML_TYPE_Q4_0:
    {
        data_type = MiniGPT4DataType::Q4_0;
        break;
    }
    case GGML_TYPE_Q4_1:
    {
        data_type = MiniGPT4DataType::Q4_1;
        break;
    }
    case GGML_TYPE_Q5_0:
    {
        data_type = MiniGPT4DataType::Q5_0;
        break;
    }
    case GGML_TYPE_Q5_1:
    {
        data_type = MiniGPT4DataType::Q5_1;
        break;
    }
    case GGML_TYPE_Q8_0:
    {
        data_type = MiniGPT4DataType::Q8_0;
        break;
    }
    case GGML_TYPE_Q8_1:
    {
        data_type = MiniGPT4DataType::Q8_1;
        break;
    }
    case GGML_TYPE_Q2_K:
    {
        data_type = MiniGPT4DataType::Q2_K;
        break;
    }
    case GGML_TYPE_Q3_K:
    {
        data_type = MiniGPT4DataType::Q3_K;
        break;
    }
    case GGML_TYPE_Q4_K:
    {
        data_type = MiniGPT4DataType::Q4_K;
        break;
    }
    case GGML_TYPE_Q5_K:
    {
        data_type = MiniGPT4DataType::Q5_K;
        break;
    }
    case GGML_TYPE_Q6_K:
    {
        data_type = MiniGPT4DataType::Q6_K;
        break;
    }
    case GGML_TYPE_Q8_K:
    {
        data_type = MiniGPT4DataType::Q8_K;
        break;
    }
    default:
    {
        ERR("Unsupported MiniGPT4DataType {}", magic_enum::enum_name(t));
        return tl::unexpected(MiniGPT4Error::LoadModelMiniGPT4DataType);
        break;
    }
    }
    return data_type;
}

struct LazyLoadTensor
{
    MMapReader *reader;
    std::string name;
    std::vector<uint32_t> shape;
    ggml_type type = ggml_type::GGML_TYPE_COUNT;

    std::size_t pos = 0;

    struct ggml_tensor *tensor = nullptr;
    BufferView tensor_buf;

    std::size_t type_size() const
    {
        switch (type)
        {
        case ggml_type::GGML_TYPE_F16:
            return sizeof(float) / 2;
        case ggml_type::GGML_TYPE_F32:
            return sizeof(float);
        case ggml_type::GGML_TYPE_I32:
            return sizeof(int32_t);
        default:
            return ggml_type_size(type);
        }
        return 0;
    }

    std::size_t total_shape() const
    {
        std::size_t size = 1;
        for (auto i = 0; i < shape.size(); i++)
        {
            size *= shape[i];
        }
        return size;
    }

    std::size_t total_size() const
    {
        if (shape.empty())
        {
            return type_size();
        }
        std::size_t size = 1;
        for (auto i = 0; i < shape.size(); i++)
        {
            size *= shape[i];
        }
        size *= type_size();
        return size;
    }

    auto get_size_in_bytes() const
    {
        // Calculate the size
        struct ggml_tensor temp
        {
        };
        temp.type = type;
        auto k = 0;
        for (; k < shape.size(); k++)
        {
            temp.ne[k] = shape[k];
        }
        for (; k < 4; k++)
        {
            temp.ne[k] = 1;
        }
        return ggml_nbytes(&temp);
    }

    auto get_file_address() const
    {
        return reader->base_addr() + pos;
    }

    struct ggml_tensor *operator()(ggml_context *ctx)
    {
        // Cached
        if (tensor)
        {
            return tensor;
        }

        // Create tensors
        const auto shape_size = shape.size();
        if (shape_size == 1)
        {
            tensor = ggml_new_tensor_1d(ctx, type, shape[0]);
        }
        else if (shape_size == 2)
        {
            tensor = ggml_new_tensor_2d(ctx, type, shape[0], shape[1]);
        }
        else if (shape_size == 3)
        {
            tensor = ggml_new_tensor_3d(ctx, type, shape[0], shape[1], shape[2]);
        }
        else if (shape_size == 4)
        {
            tensor = ggml_new_tensor_4d(ctx, type, shape[0], shape[1], shape[2], shape[3]);
        }
        else
        {
            PANIC("Layer: {}, didn't expect shape of size {}", name, shape_size);
        }

        // Just reference it
        tensor_buf.addr = get_file_address();
        tensor_buf.size = get_size_in_bytes();

        tensor->data = tensor_buf.addr;
        return tensor;
    }
};

class TorchModel
{
public:
    void set_name(std::string_view s)
    {
        name = s;
    }
    const std::string &get_name() const
    {
        return name;
    }

    void add_tensor(std::string_view name, LazyLoadTensor tensor)
    {
        tensors.try_emplace(std::string(name), tensor);
    }

    template <typename... Args>
    LazyLoadTensor &get(Args &&...args)
    {
        const auto tensor_name = fmt::format(std::forward<Args>(args)...);
        return operator[](tensor_name);
    }

    std::optional<LazyLoadTensor *> get_tensor(const std::string &tensor_name)
    {
        if (auto found = tensors.find(tensor_name); found != std::end(tensors))
        {
            auto &[_, tensor] = *found;
            return &tensor;
        }
        return std::nullopt;
    }

    LazyLoadTensor &operator[](const std::string &tensor_name)
    {
        if (auto tensor = get_tensor(tensor_name))
        {
            return **tensor;
        }
        PANIC("Couldn't find tensor {}", name);
        return tensors.begin()->second;
    }

    const LazyLoadTensor &operator[](const std::string &tensor_name) const
    {
        return const_cast<TorchModel *>(this)->operator[](tensor_name);
    }

    auto &get_tensors() { return tensors; }
    const auto &get_tensors() const { return tensors; }

private:
    std::string name;
    HashMap<std::string, LazyLoadTensor> tensors;
};

struct ContextBuffer
{
    void init_context(std::size_t buf_compute_size,
                      std::size_t buf_scratch_size,
                      std::size_t num_scratch_buffers = MAX_SCRATCH_BUFFERS)
    {
        buf_scratch.resize(num_scratch_buffers);
        buf_max_size.resize(num_scratch_buffers);
        reset_scratch_usage();

        buf_compute = Buffer(buf_compute_size);
        if (buf_scratch_size)
        {
            for (auto i = 0; i < num_scratch_buffers; i++)
            {
                buf_scratch[i] = Buffer(buf_scratch_size);
            }
        }
    }

    void use_scratch(int i)
    {
        size_t last_size = 0;

        if (i == -1)
        {
            last_size = ggml_set_scratch(ctx, {0, 0, nullptr});
        }
        else
        {
            auto &buf = buf_scratch[i];
            last_size = ggml_set_scratch(ctx, {0, buf.size, buf.addr});
        }

        if (buf_last >= 0)
        {
            buf_max_size[buf_last] = std::max(buf_max_size[buf_last], last_size);
        }

        buf_last = i;
    }

    auto get_memory_usage(int i)
    {
        if (i == -1)
        {
            return ggml_used_mem(ctx);
        }
        return buf_max_size[static_cast<std::size_t>(i)];
    }

    void reset_scratch_usage()
    {
        buf_last = 0;
        for (auto &s : buf_max_size)
        {
            s = 0;
        }
    }

    Buffer buf_compute;
    std::vector<Buffer> buf_scratch;
    int buf_last = 0;
    std::vector<size_t> buf_max_size;

    ggml_context *ctx{};
};

template <typename Derived>
struct HasContext
{
    ggml_context *data_ctx = nullptr;

    template <typename... Args>
    auto operator()(ggml_context *ctx, ggml_tensor *x, Args &&...args)
    {
        return static_cast<Derived *>(this)->forward(ctx, x, std::forward<Args>(args)...);
    }
};

struct HasContextBase;

template <template <typename> class THIS, typename IMPL, template <typename> class SUPERCLASS>
using HasContextFix = SUPERCLASS<std::conditional_t<
    std::is_same_v<IMPL, HasContextBase>,
    THIS<HasContextBase>, IMPL>>;

template <typename Derived = HasContextBase>
struct NNParameter : public HasContextFix<NNParameter, Derived, HasContext>
{
    ggml_tensor *weight{};

    ggml_tensor *forward(ggml_context *ctx, ggml_tensor *x)
    {
        return x;
    }
};
using Parameter = NNParameter<>;

template <typename Derived = HasContextBase>
struct NNLinear : public HasContextFix<NNLinear, Derived, HasContext>
{
    ggml_tensor *weight{};
    ggml_tensor *bias{};

    ggml_tensor *forward(ggml_context *ctx, ggml_tensor *x)
    {
        auto *result = ggml_mul_mat(ctx, weight, x);
        if (bias)
        {
            ggml_tensor *bias_repeated = bias;
            bias_repeated = ggml_repeat(ctx, bias, result);
            result = ggml_add(ctx, bias_repeated, result);
        }
        return result;
    }
};
using Linear = NNLinear<>;

template <typename Derived = HasContextBase>
struct NNEmbedding : public HasContextFix<NNEmbedding, Derived, HasContext>
{
    ggml_tensor *weight{};

    ggml_tensor *forward(ggml_context *ctx, ggml_tensor *x)
    {
        auto *result = ggml_get_rows(ctx, weight, x);
        return result;
    }
};
using Embedding = NNEmbedding<>;

template <typename Derived = HasContextBase>
struct NNConv2d : public HasContextFix<NNConv2d, Derived, NNLinear>
{
    ggml_tensor *forward(ggml_context *ctx, ggml_tensor *x)
    {
        const auto stride1 = this->weight->ne[0];
        const auto stride2 = this->weight->ne[1];
        const auto padding1 = 0;
        const auto padding2 = 0;
        const auto dialation1 = 1;
        const auto dialation2 = 1;

        x = ggml_conv_2d(ctx, this->weight, x, stride1, stride2, padding1, padding2, dialation1, dialation2);

        if (this->bias)
        {
            ggml_tensor *bias_repeated = this->bias;
            bias_repeated = ggml_reshape_4d(ctx, bias_repeated, bias_repeated->ne[2], bias_repeated->ne[1], bias_repeated->ne[0], bias_repeated->ne[3]);
            bias_repeated = ggml_repeat(ctx, bias_repeated, x);
            x = ggml_add(ctx, bias_repeated, x);
        }

        return x;
    }
};
using Conv2d = NNConv2d<>;

template <typename Derived = HasContextBase>
struct NNLayerNorm : public HasContextFix<NNLayerNorm, Derived, NNLinear>
{
    ggml_tensor *forward(ggml_context *ctx, ggml_tensor *x)
    {
        x = ggml_norm(ctx, x);

        // out = w * x + b
        auto w = ggml_repeat(ctx, this->weight, x);
        auto result = ggml_mul(ctx, w, x);
        if (this->bias)
        {
            auto b = ggml_repeat(ctx, this->bias, x);
            result = ggml_add(ctx, result, b);
        }

        return result;
    }
};
using LayerNorm = NNLayerNorm<>;

template <typename Derived = HasContextBase>
struct NNSelfAttention : public HasContextFix<NNSelfAttention, Derived, HasContext>
{
    Linear query;
    Linear key;
    Linear value;
    Linear dense;
    LayerNorm layer_norm;

    struct Output
    {
        ggml_tensor *context_layer;
        ggml_tensor *attention_probs;
        ggml_tensor *key_layer;
        ggml_tensor *value_layer;
    };

    Output forward(ggml_context *ctx,
                   ggml_tensor *hidden_states = nullptr,
                   ggml_tensor *attention_mask = nullptr,
                   ggml_tensor *head_mask = nullptr,
                   ggml_tensor *encoder_hidden_states = nullptr,
                   ggml_tensor *encoder_attention_mask = nullptr,
                   ggml_tensor *past_key_value = nullptr,
                   bool output_attentions = false)
    {
        auto transpose_for_scores = [&](ggml_tensor *x)
        {
            // new_x_shape = x.size()[:-1] + (
            //         self.num_attention_heads,
            //         self.attention_head_size,
            // )
            // x = x.view(*new_x_shape)
            // return x.permute(0, 2, 1, 3)

            auto x_shape1 = x->ne[1];
            auto x_shape2 = x->ne[2];
            std::array<int64_t, 4> new_shape{ATTENTION_HEAD_SIZE, NUM_ATTENTION_HEADS, x_shape1, x_shape2};

            x = ggml_reshape_4d(ctx, x, new_shape[0], new_shape[1], new_shape[2], new_shape[3]);
            x = ggml_permute(ctx, x, 0, 2, 1, 3);
            x = ggml_cont(ctx, x);

            return x;
        };

        // BertAttention -> forward
        ggml_tensor *context_layer;
        ggml_tensor *attention_probs;
        ggml_tensor *key_layer;
        ggml_tensor *value_layer;
        {
            // BertSelfAttention -> forward
            bool is_cross_attention = encoder_hidden_states != nullptr;

            if (is_cross_attention)
            {
                key_layer = transpose_for_scores(key(ctx, encoder_hidden_states));
                value_layer = transpose_for_scores(value(ctx, encoder_hidden_states));
                attention_mask = encoder_attention_mask;
            }
            else if (past_key_value != nullptr)
            {
                // TODO: implement
                key_layer = transpose_for_scores(key(ctx, hidden_states));
                value_layer = transpose_for_scores(value(ctx, hidden_states));
                exit(-1);
                // key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
                // value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
            }
            else
            {
                auto k = key(ctx, hidden_states);
                auto v = value(ctx, hidden_states);
                key_layer = transpose_for_scores(k);
                value_layer = transpose_for_scores(v);
            }

            // BertSelfAttention | mixed_query_layer = self.query(hidden_states)
            ggml_tensor *mixed_query_layer = query(ctx, hidden_states);

            // BertSelfAttention | query_layer = self.transpose_for_scores(mixed_query_layer)
            ggml_tensor *query_layer = transpose_for_scores(mixed_query_layer);

            // attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

            ggml_tensor *key_layer_permuted = key_layer;

            // ggml_tensor* attention_scores = ggml_mul_mat(ctx, query_layer, key_layer);
            ggml_tensor *attention_scores = ggml_mul_mat(ctx, key_layer, query_layer);

            // attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            auto sqrt_attention_head_size = std::sqrt(ATTENTION_HEAD_SIZE);
            ggml_tensor *sqrt_attention_head_size_tensor = ggml_new_tensor_4d(ctx,
                                                                              attention_scores->type,
                                                                              attention_scores->ne[0],
                                                                              attention_scores->ne[1],
                                                                              attention_scores->ne[2],
                                                                              attention_scores->ne[3]);
            sqrt_attention_head_size_tensor = ggml_set_f32(sqrt_attention_head_size_tensor, sqrt_attention_head_size);
            attention_scores = ggml_div_inplace(ctx, attention_scores, sqrt_attention_head_size_tensor);

            if (attention_mask != nullptr)
            {
                // attention_scores = attention_scores + attention_mask
                attention_scores = ggml_add_inplace(ctx, attention_scores, ggml_repeat(ctx, attention_mask, attention_scores));
            }

            // attention_probs = nn.Softmax(dim=-1)(attention_scores)
            attention_probs = ggml_soft_max_inplace(ctx, attention_scores);

            // attention_probs_dropped = attention_probs
            ggml_tensor *attention_probs_dropped = attention_probs;

            // context_layer = torch.matmul(attention_probs_dropped, value_layer)
            value_layer = ggml_cont(ctx, ggml_transpose(ctx, value_layer));
            context_layer = ggml_mul_mat(ctx, value_layer, attention_probs_dropped);

            // context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            context_layer = ggml_cont(ctx, ggml_permute(ctx, context_layer, 0, 2, 1, 3));

            // new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            // context_layer = context_layer.view(*new_context_layer_shape)
            context_layer = ggml_reshape_3d(ctx, context_layer, context_layer->ne[0] * context_layer->ne[1], context_layer->ne[2], context_layer->ne[3]);
        }

        ggml_tensor *attention_output;
        {
            ggml_tensor *input_tensor = hidden_states;
            ggml_tensor *hidden_states_c = context_layer;

            // hidden_states = self.dense(hidden_states)
            hidden_states_c = dense(ctx, hidden_states_c);

            // hidden_states = self.LayerNorm(hidden_states + input_tensor)
            hidden_states_c = ggml_add_inplace(ctx, hidden_states_c, input_tensor);
            hidden_states_c = layer_norm(ctx, hidden_states_c);

            attention_output = hidden_states_c;
        }

        //        outputs = (
        //                (context_layer, attention_probs) if output_attentions else (context_layer,)
        //        )
        //        outputs = outputs + (past_key_value,)

        return Output{attention_output, attention_probs, key_layer, value_layer};
    }
};
using SelfAttention = NNSelfAttention<>;

template <typename Derived = HasContextBase>
struct NNQKVAttention : public HasContextFix<NNQKVAttention, Derived, HasContext>
{
    Parameter q_bias;
    Parameter v_bias;
    Linear qkv;
    Linear proj;

    ggml_tensor *forward(ggml_context *ctx, ggml_tensor *x, std::size_t hidden_size = 0, std::size_t n_state = 0, std::size_t B = 0)
    {
        auto d_head = hidden_size / B;

        // qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        ggml_tensor *qkv_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, q_bias.weight->ne[0] * 3);
        ggml_set_zero(qkv_bias);
        qkv_bias = ggml_acc(ctx, qkv_bias, q_bias.weight, qkv_bias->nb[1], qkv_bias->nb[2], qkv_bias->nb[3], 0);
        qkv_bias = ggml_acc(ctx, qkv_bias, v_bias.weight, qkv_bias->nb[1], qkv_bias->nb[2], qkv_bias->nb[3], ggml_element_size(q_bias.weight) * 2 * hidden_size);

        // qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        auto *result = ggml_mul_mat(ctx, qkv.weight, x);
        auto *bias_repeated = ggml_repeat(ctx, qkv_bias, result);
        result = ggml_add(ctx, bias_repeated, result);

        // qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        x = result;
        x = ggml_reshape_4d(ctx, x, 88, B, 3, x->ne[1]);
        x = ggml_permute(ctx, x, 0, 2, 3, 1);
        x = ggml_cont(ctx, x);

        const auto ne0 = 88;
        const auto ne1 = 257;
        const auto ne2 = 16;

        x = ggml_reshape_4d(ctx, x, ne0, ne1, ne2, 3);

        const auto nb1 = x->nb[1];
        const auto nb2 = x->nb[2];
        const auto nb3 = x->nb[3];

        const auto offset = nb3;

        //  q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        ggml_tensor *q = ggml_view_4d(ctx, x, ne0, ne1, ne2, 1, nb1, nb2, nb3, offset * 0);
        ggml_tensor *k = ggml_view_4d(ctx, x, ne0, ne1, ne2, 1, nb1, nb2, nb3, offset * 1);
        ggml_tensor *v = ggml_view_4d(ctx, x, ne0, ne1, ne2, 1, nb1, nb2, nb3, offset * 2);

        // q = q * self.scale
        const auto scale = 1.0f / sqrt(static_cast<float>(d_head));
        q = ggml_scale_inplace(ctx, q, ggml_new_f32(ctx, scale));

        // attn = (q @ k.transpose(-2, -1))
        ggml_tensor *kq = ggml_mul_mat(ctx, k, q);
        kq = ggml_cont(ctx, kq);

        // attn = attn.softmax(dim=-1)
        kq = ggml_soft_max_inplace(ctx, kq);

        // x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        v = ggml_cont(ctx, ggml_transpose(ctx, v));
        ggml_tensor *kqv = ggml_mul_mat(ctx, v, kq);
        kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));
        x = ggml_reshape_2d(ctx, kqv, hidden_size, n_state);

        // x = self.proj(x)
        x = proj(ctx, x);

        return x;
    }
};
using QKVAttention = NNQKVAttention<>;

template <typename T, typename... Args>
T make_layer(ggml_context *ctx, Args &&...args)
{
    // static_assert(std::is_base_of_v<HasContext<T>, T>, "Has to have layer as base");
    return T{ctx, std::forward<Args>(args)...};
}

template <typename Derived = HasContextBase>
struct NNBertEncoderLayer : public HasContextFix<NNBertEncoderLayer, Derived, HasContext>
{
    SelfAttention self_attention;
    std::optional<SelfAttention> cross_attention;
    Linear intermediate_query;
    struct OutputQuery
    {
        Linear dense;
        LayerNorm layer_norm;
    } output_query;

    struct Output
    {
        ggml_tensor *layer_output;
        ggml_tensor *attention_probs;
        ggml_tensor *cross_attention_probs;
        ggml_tensor *key_layer;
        ggml_tensor *value_layer;
    };

    ggml_tensor *feed_forward_chunk_query(ggml_context *ctx, ggml_tensor *attention_output)
    {
        ggml_tensor *intermediate_output = intermediate_query(ctx, attention_output);
        intermediate_output = ggml_gelu_inplace(ctx, intermediate_output);

        // BertOutput
        // ->     def forward(self, hidden_states, input_tensor):
        //        hidden_states = self.dense(hidden_states)
        //        hidden_states = self.dropout(hidden_states)
        //        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        ggml_tensor *hidden_states = intermediate_output;
        {
            hidden_states = output_query.dense(ctx, hidden_states);
            hidden_states = ggml_add_inplace(ctx, hidden_states, attention_output);
            hidden_states = output_query.layer_norm(ctx, hidden_states);
        }
        ggml_tensor *layer_output = hidden_states;
        return layer_output;
    }

    Output forward(ggml_context *ctx,
                   ggml_tensor *x,
                   ggml_tensor *attention_mask,
                   ggml_tensor *head_mask,
                   ggml_tensor *encoder_hidden_states,
                   ggml_tensor *encoder_attention_mask,
                   ggml_tensor *past_key_value,
                   std::size_t query_length)
    {
        Output o{};
        auto &[layer_output,
               attention_probs,
               cross_attention_probs,
               key_layer,
               value_layer] = o;

        // hidden_states,
        // attention_mask=None,
        // head_mask=None,
        // encoder_hidden_states=None,
        // encoder_attention_mask=None,
        // past_key_value=None,
        // output_attentions=False,
        // query_length=0,
        ggml_tensor *hidden_states = x;

        // self_attn_past_key_value = (
        //     past_key_value[:2] if past_key_value is not None else None
        // )
        ggml_tensor *self_attn_past_key_value = past_key_value;

        // self_attention_outputs = self.attention(
        //         hidden_states,
        //         attention_mask,
        //         head_mask,
        //         output_attentions=output_attentions,
        //         past_key_value=self_attn_past_key_value,
        // )
        SelfAttention::Output self_attention_outputs = self_attention(ctx,
                                                                      hidden_states,
                                                                      attention_mask,
                                                                      head_mask,
                                                                      nullptr,
                                                                      nullptr,
                                                                      self_attn_past_key_value);

        // attention_output = self_attention_outputs[0]
        ggml_tensor *attention_output = self_attention_outputs.context_layer;
        key_layer = self_attention_outputs.key_layer;
        value_layer = self_attention_outputs.value_layer;
        attention_probs = self_attention_outputs.attention_probs;

        if (query_length > 0)
        {
            // query_attention_output = attention_output[:, :query_length, :]
            // ggml_tensor* query_attention_output = ggml_reshape_3d(ctx, attention_output, attention_output->ne[0], query_length, attention_output->ne[2]);
            ggml_tensor *query_attention_output = ggml_reshape_2d(ctx, attention_output, attention_output->ne[0], query_length);
            if (cross_attention)
            {
                // cross_attention_outputs = self.crossattention(
                //    query_attention_output,
                //    attention_mask,
                //    head_mask,
                //    encoder_hidden_states,
                //    encoder_attention_mask,
                //    output_attentions=output_attentions,
                // )
                SelfAttention::Output cross_attention_outputs = (*cross_attention)(ctx,
                                                                                   query_attention_output,
                                                                                   attention_mask,
                                                                                   head_mask,
                                                                                   encoder_hidden_states,
                                                                                   encoder_attention_mask);
                query_attention_output = cross_attention_outputs.context_layer;
            }

            // layer_output = apply_chunking_to_forward(
            //     self.feed_forward_chunk_query,
            //     self.chunk_size_feed_forward,
            //     self.seq_len_dim,
            //     query_attention_output,
            // )
            layer_output = feed_forward_chunk_query(ctx, query_attention_output);
            if (attention_output->ne[1] > query_length)
            {
                // TODO: implement
                assert(false);
            }
        }
        else
        {
            layer_output = feed_forward_chunk_query(ctx, attention_output);
        }
        // outputs = self_attention_outputs[1:-1]

        return o;
    }
};
using BertEncoderLayer = NNBertEncoderLayer<>;

/////////////////////
/// LOADERS
/////////////////////

enum class FileVersion : uint32_t
{
    UNKNOWN,
    V0
};

class MiniGPT4ModelLoader
{
public:
    MiniGPT4Error load(const fs::path &path)
    {
        reader.load(path);

        file_header = reader.read_bytes(EXPECTED_HEADER.length());
        ASSERT(file_header == EXPECTED_HEADER, "File header not matching {} != {}", spdlog::to_hex(file_header), EXPECTED_HEADER);
        if (file_header != EXPECTED_HEADER)
        {
            ERR("Unepected file header {}", file_header);
            return MiniGPT4Error::LoadModelFileHeader;
        }

        file_version = FileVersion(reader.read_s4());
        ASSERT(file_version != FileVersion::UNKNOWN, "File unknown");
        if (file_version == FileVersion::UNKNOWN)
        {
            ERR("Unepected file version {}", magic_enum::enum_name(file_version));
            return MiniGPT4Error::LoadModelFileVersion;
        }

        file_data_type = *data_type_to_ggml_type(MiniGPT4DataType(reader.read_s4()));

        auto config_json = reader.read_string();
        config = json::parse(config_json);

        // Model
        auto ParseModel = [&]() -> tl::expected<TorchModel, MiniGPT4Error>
        {
            TorchModel model;

            // Get model name and number of layers
            auto model_name = reader.read_string();
            INFO("Model name: {}", model_name);
            auto num_layers = reader.read_s4();

            model.set_name(model_name);

            std::vector<LazyLoadTensor> lazy_load_tensors(num_layers);
            for (auto i = 0; i < num_layers; i++)
            {
                auto &lazy_load_tensor = lazy_load_tensors[i];

                // Tensor
                auto &tensor_reader = lazy_load_tensor.reader;
                auto &layer_name = lazy_load_tensor.name;
                auto &shape = lazy_load_tensor.shape;
                auto &type = lazy_load_tensor.type;
                auto &tensor_pos = lazy_load_tensor.pos;
                tensor_reader = &reader;

                // Layer name
                layer_name = reader.read_string();

                // Shape
                auto num_shape = reader.read_s4();
                shape.reserve(num_shape);

                for (auto j = 0; j < num_shape; j++)
                {
                    auto size = reader.read_s4();
                    shape.emplace_back(size);
                }

                // Data type
                auto data_type = MiniGPT4DataType(reader.read_s4());
                if (auto type_ = data_type_to_ggml_type(data_type); type_.has_value())
                {
                    type = type_.value();
                }
                else
                {
                    return tl::unexpected(type_.error());
                }

                if (type == GGML_TYPE_F32)
                {
                    num_f32_tensors++;
                }

                model_context_size += lazy_load_tensor.get_size_in_bytes();
            }

            for (auto i = 0; i < num_layers; i++)
            {
                auto &lazy_load_tensor = lazy_load_tensors[i];
                auto &layer_name = lazy_load_tensor.name;
                auto &tensor_pos = lazy_load_tensor.pos;

                // seek to next bound
                reader.seek_to_alignment(PAGE_SIZE);

                // save position and size for loading the actual tensor later lazily
                tensor_pos = reader.tell();

                auto tensor_size = lazy_load_tensor.get_size_in_bytes();
                reader.seek(tensor_pos + tensor_size);

                // add mapping
                model.add_tensor(layer_name, lazy_load_tensor);
            }

            return model;
        };

        // Parse each model
        while (!reader.is_eof())
        {
            if (auto model = ParseModel(); model.has_value())
            {
                models.try_emplace(model->get_name(), std::move(*model));
            }
            else
            {
                ERR("Error parsing model {}", magic_enum::enum_name(model.error()));
                return model.error();
            }
        }
        return MiniGPT4Error::None;
    }

    TorchModel &operator[](const std::string &name)
    {
        if (auto found = models.find(name); found != std::end(models))
        {
            auto &[_, model] = *found;
            return model;
        }
        ERR("Couldn't find model {}", name);
        return models.begin()->second;
    }

    auto &get_models() { return models; }
    const auto &get_models() const { return models; }
    const auto &get_config() const { return config; }
    auto get_model_context_size() { return model_context_size; }

    ModelType get_model_type()
    {
        auto &llama_proj_model = operator[]("llama_proj");
        auto &weight = llama_proj_model["weight"];
        if (weight.shape[1] == LLAMA_PROJECTION_HIDDEN_SIZE_7B)
        {
            return ModelType::Vicuna7B;
        }
        else if (weight.shape[1] == LLAMA_PROJECTION_HIDDEN_SIZE_13B)
        {
            return ModelType::Vicuna13B;
        }
        return ModelType::Unknown;
    }

    void set_file_data_type(ggml_type data_type) { file_data_type = data_type; }
    auto get_file_data_type() const { return file_data_type; }

    MiniGPT4Error dump(fs::path path)
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);

        if (!out.is_open())
        {
            ERR("Couldn't open file {}", path.string());
            return MiniGPT4Error::DumpModelFileOpen;
        }

        auto WriteString = [&](std::string_view str)
        {
            auto str_size = static_cast<int32_t>(str.size());
            out.write(reinterpret_cast<const char *>(&str_size), sizeof(str_size));
            out.write(str.data(), str_size);
        };

        auto WriteInt = [&](int32_t i)
        {
            out.write(reinterpret_cast<const char *>(&i), sizeof(i));
        };

        // Write header
        out.write(reinterpret_cast<const char *>(file_header.c_str()), file_header.size());

        // Write file version
        WriteInt(magic_enum::enum_integer(file_version));

        // Write file data type
        auto data_type = *ggml_type_to_data_type(file_data_type);
        WriteInt(magic_enum::enum_integer(data_type));

        // Write config
        auto config_str = config.dump();
        WriteString(config_str);

        // Write models
        for (auto &[model_name, model] : get_models())
        {
            // Write model name
            WriteString(model_name);

            // Write number of tensors
            WriteInt(model.get_tensors().size());

            for (const auto &[name, t] : model.get_tensors())
            {
                // Write tensor name
                WriteString(name);

                // Write shape
                WriteInt(t.shape.size());

                // Write shape elements
                for (const auto &s : t.shape)
                {
                    WriteInt(s);
                }

                // Write MiniGPT4DataType
                WriteInt(magic_enum::enum_integer(*ggml_type_to_data_type(t.type)));
            }

            for (const auto &[name, t] : model.get_tensors())
            {
                // Write data, aligning it
                auto align_to_next_page = [](std::size_t pos)
                {
                    if ((PAGE_SIZE - 1) & pos)
                    {
                        return (pos + PAGE_SIZE) & ~(PAGE_SIZE - 1);
                    }
                    else
                    {
                        return pos;
                    }
                };

                auto pos = align_to_next_page(out.tellp());
                out.seekp(pos, std::ios::beg);
                out.write(reinterpret_cast<const char *>(t.tensor_buf.addr), t.tensor_buf.size);
            }
        }

        return MiniGPT4Error::None;
    }

private:
    std::unique_ptr<std::ifstream> file_stream = nullptr;
    MMapReader reader;

    std::string file_header;
    FileVersion file_version;
    ggml_type file_data_type;

    HashMap<std::string, TorchModel> models;
    json config;
    std::size_t model_context_size{};
    std::size_t num_f32_tensors{};
};

#define RETURN_IF_ERROR(call)  \
    auto UNIQUIFY(err) = call; \
    if (UNIQUIFY(err))         \
    {                          \
        return UNIQUIFY(err);  \
    }

class MiniGPT4 : public ContextBuffer
{
public:
    template <typename T, typename... Args>
    T make_layer_ctx(Args &&...args)
    {
        auto try_transform_ggml_tensor = [&]<typename Entry>(Entry &&e) -> ggml_tensor *
        {
            if constexpr (std::is_same_v<std::decay_t<Entry>, LazyLoadTensor>)
            {
                return e(model_ctx);
            }
            else if constexpr (std::is_same_v<std::decay_t<Entry>, ggml_tensor *>)
            {
                return e;
            }
            PANIC("Not transformation could be made...");
        };
        return make_layer<T>(model_ctx, try_transform_ggml_tensor(std::forward<Args>(args))...);
    };

    template <typename T, typename... Args>
    auto make_linear(T &model, std::string_view layer_name, Args &&...args)
    {
        auto weight_name = fmt::format("{}.weight", layer_name);
        auto bias_name = fmt::format("{}.bias", layer_name);
        auto linear = make_layer_ctx<Linear>(model[weight_name], model[bias_name]);
        return linear;
    }

    MiniGPT4Error init(const fs::path &path, const fs::path &llm_path, MiniGPT4Verbosity verbosity, int seed, int n_ctx, int n_batch, bool numa)
    {
        global_verbosity = verbosity;

        {
            LoggingTimer timer("LLM model init");
            llama_init_backend(numa);
            llm_params = llama_context_default_params();
            llm_params.n_ctx = n_ctx;
            llm_params.n_batch = n_batch;
            llm_params.seed = seed;
            llm_params.use_mmap = true;
            // llm_params.use_mlock = true;
            llm_model = LLMModel(llama_load_model_from_file(llm_path.string().c_str(), llm_params));
            llm_ctx = LLMContext(llama_new_context_with_model(&*llm_model, llm_params));
            llama_print_system_info();
        }
        {
            LoggingTimer timer("Load file");
            RETURN_IF_ERROR(minigpt4_model_loader.load(path));
        }
        auto model_type = minigpt4_model_loader.get_model_type();
        INFO("Model type: {}", magic_enum::enum_name(model_type));

        const auto model_context_size = minigpt4_model_loader.get_model_context_size();
        INFO("Model size: {} MB", bytes_to_mb(model_context_size));

        struct ggml_init_params params
        {
            .mem_size = model_context_size,
            .mem_buffer = nullptr,
            .no_alloc = true,
        };

        model_ctx = ggml_init(params);
        ASSERT(model_ctx != nullptr, "Context should be valid");

        auto compute_size = model_type_to_compute_size.at(model_type);
        auto scratch_size = model_type_to_scratch_size.at(model_type);
        if (minigpt4_model_loader.get_file_data_type() == GGML_TYPE_F32)
        {
            compute_size *= 2;
            scratch_size *= 2;
        }
        init_context(compute_size, scratch_size);

        {
            LoggingTimer timer("Loading minigpt4 model");
            load_minigpt4_model();
        }

        return MiniGPT4Error::None;
    }

    struct VisualEncoderModel
    {
        Parameter cls_token;
        Parameter pos_embed;
        Conv2d patch_embed;
        struct Block
        {
            LayerNorm norm1;
            QKVAttention attn;
            LayerNorm norm2;
            struct MLP
            {
                Linear fc1;
                Linear fc2;
            } mlp;
        };
        std::vector<Block> blocks;
    };

    using LnVisionModel = LayerNorm;

    using QueryTokensModel = Parameter;

    struct QFormerModel
    {
        struct Bert
        {
            struct Embeddings
            {
                Parameter position_ids;
                LayerNorm layer_norm;
            } embeddings;
            struct Encoder
            {
                std::vector<BertEncoderLayer> layer;
            } encoder;
        } bert;
    };

    using LLamaProjectionModel = Linear;

    struct MiniGPT4Model
    {
        VisualEncoderModel visual_encoder_model;
        LnVisionModel ln_vision_model;
        QueryTokensModel query_tokens_model;
        QFormerModel qformer_model;
        LLamaProjectionModel llama_projection_model;
    };

    void load_visual_encoder()
    {
        const auto &config = minigpt4_model_loader.get_config();

        auto &c = minigpt4_model_loader["visual_encoder"];
        auto &visual_encoder = minigpt4_model.visual_encoder_model;

        visual_encoder.cls_token = make_layer_ctx<Parameter>(c["cls_token"]);
        visual_encoder.pos_embed = make_layer_ctx<Parameter>(c["pos_embed"]);
        visual_encoder.patch_embed = make_layer_ctx<Conv2d>(c["patch_embed.proj.weight"], c["patch_embed.proj.bias"]);

        // Get number of visual encoder blocks
        auto num_visual_encoder_blocks = 0;
        while (true)
        {
            if (c.get_tensor(fmt::format("blocks.{}.norm1.weight", num_visual_encoder_blocks)).has_value())
            {
                num_visual_encoder_blocks++;
            }
            else
            {
                break;
            }
        }

        visual_encoder.blocks.resize(num_visual_encoder_blocks);
        for (auto i = 0; i < num_visual_encoder_blocks; i++)
        {
            auto &block = visual_encoder.blocks[i];
            constexpr std::string_view prefix = "blocks";
            auto take_name = [&]<typename... Args>(Args &&...suffix)
            {
                return fmt::format("{}.{}.{}", prefix, i, std::forward<Args>(suffix)...);
            };

            block = VisualEncoderModel::Block{
                .norm1 = make_layer_ctx<LayerNorm>(c[take_name("norm1.weight")], c[take_name("norm1.bias")]),
                .attn = QKVAttention{
                    .q_bias = make_layer_ctx<Parameter>(c[take_name("attn.q_bias")]),
                    .v_bias = make_layer_ctx<Parameter>(c[take_name("attn.v_bias")]),
                    .qkv = make_layer_ctx<Linear>(c[take_name("attn.qkv.weight")]),
                    .proj = make_layer_ctx<Linear>(c[take_name("attn.proj.weight")], c[take_name("attn.proj.bias")]),
                },
                .norm2 = make_layer_ctx<LayerNorm>(c[take_name("norm2.weight")], c[take_name("norm2.bias")]),
                .mlp = {
                    .fc1 = make_layer_ctx<Linear>(c[take_name("mlp.fc1.weight")], c[take_name("mlp.fc1.bias")]),
                    .fc2 = make_layer_ctx<Linear>(c[take_name("mlp.fc2.weight")], c[take_name("mlp.fc2.bias")]),
                },
            };
        }

        ggml_set_name(visual_encoder.cls_token.weight, "visual_encoder.cls_token");
        ggml_set_name(visual_encoder.pos_embed.weight, "visual_encoder.pos_embed");
        ggml_set_name(visual_encoder.patch_embed.weight, "visual_encoder.patch_embed.proj.weight");
        ggml_set_name(visual_encoder.patch_embed.bias, "visual_encoder.patch_embed.proj.bias");
        for (const auto &block : visual_encoder.blocks)
        {
            ggml_set_name(block.norm1.weight, "visual_encoder.blocks.norm1.weight");
            ggml_set_name(block.norm1.bias, "visual_encoder.blocks.norm1.bias");
            ggml_set_name(block.attn.q_bias.weight, "visual_encoder.blocks.attn.q_bias");
            ggml_set_name(block.attn.v_bias.weight, "visual_encoder.blocks.attn.v_bias");
            ggml_set_name(block.attn.qkv.weight, "visual_encoder.blocks.qkv.weight");
            ggml_set_name(block.attn.proj.weight, "visual_encoder.blocks.attn.proj.weight");
            ggml_set_name(block.attn.proj.bias, "visual_encoder.blocks.attn.proj.bias");
            ggml_set_name(block.norm2.weight, "visual_encoder.blocks.norm2.weight");
            ggml_set_name(block.norm2.bias, "visual_encoder.blocks.norm2.bias");
            ggml_set_name(block.mlp.fc1.weight, "visual_encoder.blocks.mlp.fc1.weight");
            ggml_set_name(block.mlp.fc1.bias, "visual_encoder.blocks.mlp.fc1.bias");
            ggml_set_name(block.mlp.fc2.weight, "visual_encoder.blocks.mlp.fc2.weight");
            ggml_set_name(block.mlp.fc2.bias, "visual_encoder.blocks.mlp.fc2.bias");
        }
    }

    void load_ln_vision()
    {
        const auto &config = minigpt4_model_loader.get_config();

        auto &c = minigpt4_model_loader["ln_vision"];
        auto &ln_vision = minigpt4_model.ln_vision_model;

        ln_vision = make_layer_ctx<LayerNorm>(c["weight"], c["bias"]);

        ggml_set_name(ln_vision.weight, "ln_vision.weight");
        ggml_set_name(ln_vision.bias, "ln_vision.bias");
    }

    void load_query_tokens()
    {
        const auto &config = minigpt4_model_loader.get_config();

        auto &c = minigpt4_model_loader["query_tokens"];
        auto &query_tokens = minigpt4_model.query_tokens_model;

        query_tokens = make_layer_ctx<Parameter>(c["weight"]);

        ggml_set_name(query_tokens.weight, "query_tokens.weight");
    }

    void load_qformer()
    {
        const auto &config = minigpt4_model_loader.get_config();

        auto &c = minigpt4_model_loader["Qformer"];
        auto &qformer = minigpt4_model.qformer_model;

        qformer.bert.embeddings = QFormerModel::Bert::Embeddings{
            .position_ids = make_layer_ctx<Parameter>(c["bert.embeddings.position_ids"]),
            .layer_norm = make_layer_ctx<LayerNorm>(c["bert.embeddings.LayerNorm.weight"], c["bert.embeddings.LayerNorm.bias"]),
        };

        // Get the number of encoder blocks
        auto num_bert_encoder_blocks = 0;
        while (true)
        {
            if (c.get_tensor(fmt::format("bert.encoder.layer.{}.attention.self.query.weight", num_bert_encoder_blocks)).has_value())
            {
                num_bert_encoder_blocks++;
            }
            else
            {
                break;
            }
        }

        qformer.bert.encoder.layer.resize(num_bert_encoder_blocks);
        for (auto i = 0; i < num_bert_encoder_blocks; i++)
        {
            auto &encoder = qformer.bert.encoder.layer[i];
            constexpr std::string_view prefix = "bert.encoder.layer";
            auto take_name = [&]<typename... Args>(Args &&...suffix)
            {
                return fmt::format("{}.{}.{}", prefix, i, std::forward<Args>(suffix)...);
            };

            bool layer_has_crossattention = c.get_tensor(take_name("crossattention.self.query.weight")).has_value();
            encoder = BertEncoderLayer{
                .self_attention = SelfAttention{
                    .query = make_layer_ctx<Linear>(c[take_name("attention.self.query.weight")],
                                                    c[take_name("attention.self.query.bias")]),
                    .key = make_layer_ctx<Linear>(c[take_name("attention.self.key.weight")],
                                                  c[take_name("attention.self.key.bias")]),
                    .value = make_layer_ctx<Linear>(c[take_name("attention.self.value.weight")],
                                                    c[take_name("attention.self.value.bias")]),
                    .dense = make_layer_ctx<Linear>(c[take_name("attention.output.dense.weight")],
                                                    c[take_name("attention.output.dense.bias")]),
                    .layer_norm = make_layer_ctx<LayerNorm>(c[take_name("attention.output.LayerNorm.weight")],
                                                            c[take_name("attention.output.LayerNorm.bias")]),
                },
                .cross_attention = layer_has_crossattention ? std::make_optional<SelfAttention>(SelfAttention{
                                                                  .query = make_layer_ctx<Linear>(c[take_name("crossattention.self.query.weight")], c[take_name("crossattention.self.query.bias")]),
                                                                  .key = make_layer_ctx<Linear>(c[take_name("crossattention.self.key.weight")], c[take_name("crossattention.self.key.bias")]),
                                                                  .value = make_layer_ctx<Linear>(c[take_name("crossattention.self.value.weight")], c[take_name("crossattention.self.value.bias")]),
                                                                  .dense = make_layer_ctx<Linear>(c[take_name("crossattention.output.dense.weight")], c[take_name("crossattention.output.dense.bias")]),
                                                                  .layer_norm = make_layer_ctx<LayerNorm>(c[take_name("crossattention.output.LayerNorm.weight")], c[take_name("crossattention.output.LayerNorm.bias")]),
                                                              })
                                                            : std::nullopt,
                .intermediate_query = make_layer_ctx<Linear>(c[take_name("intermediate_query.dense.weight")], c[take_name("intermediate_query.dense.bias")]),
                .output_query = BertEncoderLayer::OutputQuery{
                    .dense = make_layer_ctx<Linear>(c[take_name("output_query.dense.weight")], c[take_name("output_query.dense.bias")]),
                    .layer_norm = make_layer_ctx<LayerNorm>(c[take_name("output_query.LayerNorm.weight")], c[take_name("output_query.LayerNorm.bias")]),
                },
            };
        }

        ggml_set_name(qformer.bert.embeddings.position_ids.weight, "Qformer.bert.embeddings.position_ids");
        ggml_set_name(qformer.bert.embeddings.layer_norm.weight, "Qformer.bert.embeddings.LayerNorm.weight");
        ggml_set_name(qformer.bert.embeddings.layer_norm.bias, "Qformer.bert.embeddings.LayerNorm.bias");
        for (const auto &layer : qformer.bert.encoder.layer)
        {
            ggml_set_name(layer.self_attention.query.weight, "Qformer.bert.attention.self.query.weight");
            ggml_set_name(layer.self_attention.query.bias, "Qformer.bert.attention.self.query.bias");
            ggml_set_name(layer.self_attention.key.weight, "Qformer.bert.attention.self.key.weight");
            ggml_set_name(layer.self_attention.key.bias, "Qformer.bert.attention.self.key.bias");
            ggml_set_name(layer.self_attention.value.weight, "Qformer.bert.attention.self.value.weight");
            ggml_set_name(layer.self_attention.value.bias, "Qformer.bert.attention.self.value.bias");
            ggml_set_name(layer.self_attention.dense.weight, "Qformer.bert.attention.output.dense.weight");
            ggml_set_name(layer.self_attention.dense.bias, "Qformer.bert.attention.output.dense.bias");
            ggml_set_name(layer.self_attention.layer_norm.weight, "Qformer.bert.attention.output.LayerNorm.weight");
            ggml_set_name(layer.self_attention.layer_norm.bias, "Qformer.bert.attention.output.LayerNorm.bias");
            if (layer.cross_attention)
            {
                ggml_set_name(layer.cross_attention->query.weight, "Qformer.bert.crossattention.self.query.weight");
                ggml_set_name(layer.cross_attention->query.bias, "Qformer.bert.crossattention.self.query.bias");
                ggml_set_name(layer.cross_attention->key.weight, "Qformer.bert.crossattention.self.key.weight");
                ggml_set_name(layer.cross_attention->key.bias, "Qformer.bert.crossattention.self.key.bias");
                ggml_set_name(layer.cross_attention->value.weight, "Qformer.bert.crossattention.self.value.weight");
                ggml_set_name(layer.cross_attention->value.bias, "Qformer.bert.crossattention.self.value.bias");
                ggml_set_name(layer.cross_attention->dense.weight, "Qformer.bert.crossattention.output.dense.weight");
                ggml_set_name(layer.cross_attention->dense.bias, "Qformer.bert.crossattention.output.dense.bias");
                ggml_set_name(layer.cross_attention->layer_norm.weight, "Qformer.bert.crossattention.output.LayerNorm.weight");
                ggml_set_name(layer.cross_attention->layer_norm.bias, "Qformer.bert.crossattention.output.LayerNorm.bias");
            }
            ggml_set_name(layer.intermediate_query.weight, "Qformer.bert.intermediate_query.weight");
            ggml_set_name(layer.intermediate_query.bias, "Qformer.bert.intermediate_query.bias");
            ggml_set_name(layer.output_query.dense.weight, "Qformer.bert.output_query.output.dense.weight");
            ggml_set_name(layer.output_query.dense.bias, "Qformer.bert.output_query.output.dense.bias");
            ggml_set_name(layer.output_query.layer_norm.weight, "Qformer.bert.output_query.output.LayerNorm.weight");
            ggml_set_name(layer.output_query.layer_norm.bias, "Qformer.bert.output_query.output.LayerNorm.bias");
        }
    }

    void load_llama_projection()
    {
        const auto &config = minigpt4_model_loader.get_config();

        auto &c = minigpt4_model_loader["llama_proj"];
        auto &llama_proj = minigpt4_model.llama_projection_model;

        llama_proj = make_layer_ctx<Linear>(c["weight"], c["bias"]);
    }

    void load_minigpt4_model()
    {
        load_visual_encoder();
        load_ln_vision();
        load_query_tokens();
        load_qformer();
        load_llama_projection();
    }

    MiniGPT4Error encode_image(struct MiniGPT4Image *image, OUT struct MiniGPT4Embedding *minigpt4_embedding, int n_threads)
    {
        const auto &config = minigpt4_model_loader.get_config();

        auto &visual_encoder = minigpt4_model.visual_encoder_model;
        auto &ln_vision = minigpt4_model.ln_vision_model;
        auto &qformer = minigpt4_model.qformer_model;
        auto &query_tokens = minigpt4_model.query_tokens_model;
        auto &llama_proj = minigpt4_model.llama_projection_model;

        const std::size_t image_size = IMAGE_RESIZE;

        const auto num_positions = visual_encoder.pos_embed.weight->ne[1];

        struct ggml_init_params params = {
            .mem_size = buf_compute.size,
            .mem_buffer = buf_compute.addr,
            .no_alloc = false,
        };

        ctx = ggml_init(params);

        reset_scratch_usage();

        auto ctx0 = ctx;
        struct ggml_cgraph gf = {};
        gf.n_threads = n_threads;

        ggml_tensor *cur;

        ggml_tensor *inp = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, image_size, image_size, 3, 1);
        {
            float *data = (float *)ggml_get_data(inp);

            float *src_data = reinterpret_cast<float *>(image->data);

            if (image->width * image->height * image->channels != image_size * image_size * 3)
            {
                return MiniGPT4Error::ImageNot224_244_3;
            }

            if (image->format != MiniGPT4ImageFormat::MINIGPT4_IMAGE_FORMAT_F32)
            {
                return MiniGPT4Error::ImageNotF32;
            }

            std::copy(src_data, src_data + (image->width * image->height * image->channels), data);
        }
        ggml_set_name(inp, "inp");

        ggml_tensor *residual{};

        static const std::size_t embed_dim = config["Qformer"]["encoder_width"];
        const std::size_t num_heads = embed_dim / 88;

        use_scratch(0);

        // Embeddings
        {
            cur = visual_encoder.patch_embed(ctx0, inp);

            cur = ggml_reshape_2d(ctx0, cur, PATCH_SIZE * PATCH_SIZE, embed_dim);
            cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

            ggml_tensor *embeddings = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, embed_dim, num_positions);
            ggml_set_zero(embeddings);
            embeddings = ggml_acc(ctx0, embeddings, visual_encoder.cls_token.weight, embeddings->nb[1], embeddings->nb[2], embeddings->nb[3], 0);
            embeddings = ggml_acc(ctx0, embeddings, cur, embeddings->nb[1], embeddings->nb[2], embeddings->nb[3], ggml_element_size(visual_encoder.cls_token.weight) * embed_dim);

            embeddings = ggml_add_inplace(ctx0, embeddings, visual_encoder.pos_embed.weight);

            residual = embeddings;
        }

        cur = residual;

        // Weights
        auto &blocks = visual_encoder.blocks;
        for (auto i = 0; i < blocks.size(); i++)
        {
            auto &block = blocks[i];

            residual = cur;

            cur = block.norm1(ctx0, cur);
            cur = block.attn(ctx0, cur, embed_dim, num_positions, num_heads);
            cur = ggml_add_inplace(ctx0, residual, cur);

            residual = cur;
            cur = block.norm2(ctx0, cur);

            // MLP
            {
                cur = block.mlp.fc1(ctx0, cur);
                cur = ggml_gelu_inplace(ctx0, cur);
                cur = block.mlp.fc2(ctx0, cur);
            }

            cur = ggml_add_inplace(ctx0, residual, cur);
        }

        // image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
        cur = ln_vision(ctx0, cur);
        ggml_tensor *image_embeds = cur;

        // image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
        ggml_tensor *image_atts = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, image_embeds->ne[1], image_embeds->ne[2]);
        image_atts = ggml_set_f32(image_atts, 1.0f);

        ggml_tensor *last_hidden_state;
        {
            ggml_tensor *encoder_hidden_states = image_embeds;
            ggml_tensor *encoder_attention_mask = image_atts;

            // bert
            auto &bert = qformer.bert;
            auto &encoder = bert.encoder.layer;
            auto &embeddings = bert.embeddings;

            ggml_tensor *query_embeds = query_tokens.weight;
            ggml_tensor *past_key_values;

            // TODO: change to pull from config query length
            auto past_key_values_length = 0;
            //  past_key_values_length = (
            //          past_key_values[0][0].shape[2] - self.config.query_length
            //  if past_key_values is not None
            //  else 0
            //  )
            // TODO: what is past key values...? chat
            //            auto past_key_values_length = 0;

            auto query_embeds_length = query_embeds->ne[1];
            static const std::size_t query_length = config["Qformer"]["query_length"];
            if (query_embeds_length != query_length)
            {
                PANIC("query_embeds_length != query_length {} {}", query_embeds_length, query_length);
            }

            // BertEmbeddings
            // embedding_output = self.embeddings(
            //         input_ids=input_ids,
            //         position_ids=position_ids,
            //         query_embeds=query_embeds,
            //         past_key_values_length=past_key_values_length,
            // )
            // ->
            // embeddings = self.LayerNorm(embeddings)
            // embeddings = self.dropout(embeddings)
            // return embeddings

            // BertEmbeddings | embeddings = self.LayerNorm(embeddings)
            ggml_tensor *embedding_output = embeddings.layer_norm(ctx0, query_tokens.weight);

            // BertModel forward
            auto batch_size = embedding_output->ne[2];
            auto seq_length = embedding_output->ne[1];

            ggml_tensor *attention_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, seq_length + past_key_values_length, batch_size);

            // get_extended_attention_mask
            // -> extended_attention_mask = attention_mask[:, None, None, :]
            ggml_tensor *extended_attention_mask = ggml_reshape_4d(ctx0, attention_mask, attention_mask->ne[0], 1, 1, attention_mask->ne[1]);

            // invert_attention_mask
            //  encoder_extended_attention_mask = self.invert_attention_mask(
            //          encoder_attention_mask
            //  )
            //  -> encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * torch.finfo(self.dtype).min
            ggml_tensor *ones = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, encoder_attention_mask->ne[0], encoder_attention_mask->ne[1], encoder_attention_mask->ne[2], encoder_attention_mask->ne[3]);
            ones = ggml_set_f32(ones, 1.0f);
            ggml_tensor *finfo_min = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, encoder_attention_mask->ne[0], encoder_attention_mask->ne[1], encoder_attention_mask->ne[2], encoder_attention_mask->ne[3]);
            finfo_min = ggml_set_f32(finfo_min, TORCH_FLOAT_FIFO_MIN);
            encoder_attention_mask = ggml_sub_inplace(ctx0, ones, encoder_attention_mask);
            ggml_tensor *encoder_extended_attention_mask = ggml_mul_inplace(ctx0, finfo_min, encoder_attention_mask);

            // head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
            // -> [None, None, None, None, None, None, None, None, None, None, None, None], 12
            // encoder_outputs = self.encoder(
            //         embedding_output,
            //         attention_mask=extended_attention_mask,
            //         head_mask=head_mask,
            //         encoder_hidden_states=encoder_hidden_states,
            //         encoder_attention_mask=encoder_extended_attention_mask,
            //         past_key_values=past_key_values,
            //         use_cache=use_cache,
            //         output_attentions=output_attentions,
            //         output_hidden_states=output_hidden_states,
            //         return_dict=return_dict,
            //         query_length=query_length,
            // )

            {
                ggml_tensor *hidden_states = embedding_output;
                ggml_tensor *attention_mask = extended_attention_mask;
                ggml_tensor *head_mask = nullptr;
                ggml_tensor *encoder_attention_mask = encoder_extended_attention_mask;
                ggml_tensor *past_key_values = past_key_values;

                static const std::size_t num_hidden_layers = config["Qformer"]["num_hidden_layers"];
                for (auto i = 0; i < num_hidden_layers; i++)
                {
                    auto &layer_module = encoder[i];
                    auto &layer_head_mask = head_mask;
                    ggml_tensor *past_key_value = nullptr;

                    // BertLayer -> forward
                    // def forward(
                    //         self,
                    //         hidden_states,
                    //         attention_mask=None,
                    //         head_mask=None,
                    //         encoder_hidden_states=None,
                    //         encoder_attention_mask=None,
                    //         past_key_value=None,
                    //         output_attentions=False,
                    //         query_length=0,
                    // ):
                    // layer_outputs = layer_module(
                    //         hidden_states,
                    //         attention_mask,
                    //         layer_head_mask,
                    //         encoder_hidden_states,
                    //         encoder_attention_mask,
                    //         past_key_value,
                    //         output_attentions,
                    //         query_length,
                    // )
                    {
                        ggml_tensor *self_attn_past_key_value = past_key_value;
                        ggml_tensor *past_key_value = self_attn_past_key_value;
                        BertEncoderLayer::Output layer_output = layer_module(ctx0,
                                                                             hidden_states,
                                                                             attention_mask,
                                                                             layer_head_mask,
                                                                             encoder_hidden_states,
                                                                             encoder_attention_mask,
                                                                             past_key_value,
                                                                             query_length);

                        hidden_states = layer_output.layer_output;
                    }
                }

                last_hidden_state = hidden_states;
            }
        }

        ggml_tensor *inputs_llama = llama_proj(ctx0, last_hidden_state);
        cur = inputs_llama;

        ggml_set_name(cur, "output");

        use_scratch(-1);

        ggml_build_forward_expand(&gf, cur);
        ggml_graph_compute(ctx0, &gf);

        INFO("Compute buffer uses {} MB", bytes_to_mb(get_memory_usage(-1)));
        INFO("Scratch buffer uses {} MB", bytes_to_mb(get_memory_usage(0)));

        minigpt4_embedding->elements = cur->ne[0] * cur->ne[1] * cur->ne[2] * cur->ne[3];
        minigpt4_embedding->data = new float[minigpt4_embedding->elements];
        auto src = ggml_get_data_f32(cur);
        std::copy(src, src + minigpt4_embedding->elements, minigpt4_embedding->data);

        ggml_free(ctx0);

        return MiniGPT4Error::None;
    }

    MiniGPT4Error add_tokens(const std::vector<llama_token> &tokens, int n_threads)
    {
        int cur_n_past = n_past;

        for (int i = 0; i < static_cast<int>(tokens.size()); i += llm_params.n_batch)
        {
            int n_eval = static_cast<int>(tokens.size()) - i;
            n_eval = std::min(n_eval, static_cast<decltype(n_eval)>(llm_params.n_batch));
            if (llama_eval(llm_ctx.get(), &tokens[i], n_eval, cur_n_past, n_threads))
            {
                ERR("Failed to add string");
                return MiniGPT4Error::FailedToAddString;
            }
            cur_n_past += n_eval;
        }
        n_past = cur_n_past;
        return MiniGPT4Error::None;
    }

    MiniGPT4Error add_strings(const char *s_, int n_threads)
    {
        std::string_view s(s_);
        bool add_bos = true;
        std::vector<llama_token> tokens(s.length() + (int)add_bos);
        auto num_tokens = llama_tokenize(llm_ctx.get(), s.data(), tokens.data(), tokens.size(), add_bos);
        tokens.resize(num_tokens);

        if (auto err = add_tokens(tokens, n_threads))
        {
            return err;
        }
        return MiniGPT4Error::None;
    }

    MiniGPT4Error add_embedding(MiniGPT4Embedding *embedding, int n_threads)
    {
        int n_embd = llama_n_embd(llm_ctx.get());

        auto N = embedding->elements;
        auto data = embedding->data;
        int n_batch = N;

        int cur_n_past = n_past;
        for (auto i = 0; i < static_cast<int>(N); i += n_batch)
        {
            int n_eval = static_cast<int>(N) - i;
            n_eval = std::min(n_eval, static_cast<decltype(n_eval)>(n_batch));
            if (llama_eval_embd(llm_ctx.get(), (data + i * n_embd), n_eval, cur_n_past, n_threads))
            {
                ERR("Failed to add embedding");
                return MiniGPT4Error::FailedToAddEmbedding;
            }
            cur_n_past += n_eval;
        }
        n_past = cur_n_past;

        return MiniGPT4Error::None;
    }

    // mostly from embd-input
    llama_token sample_token(int n_threads, float temp, int32_t top_k, float top_p, float tfs_z, float typical_p, int32_t repeat_last_n, float repeat_penalty, float alpha_presence, float alpha_frequency, int mirostat, float mirostat_tau, float mirostat_eta, int penalize_nl)
    {
        // out of user input, sample next token
        top_k = top_k <= 0 ? llama_n_vocab(llm_ctx.get()) : top_k;

        llama_token id = 0;
        {
            auto logits = llama_get_logits(llm_ctx.get());
            auto n_vocab = llama_n_vocab(llm_ctx.get());

            // Apply params.logit_bias map
            //            for (auto it = params.logit_bias.begin(); it != params.logit_bias.end(); it++) {
            //                logits[it->first] += it->second;
            //            }

            std::vector<llama_token_data> candidates;
            candidates.reserve(n_vocab);
            for (llama_token token_id = 0; token_id < n_vocab; token_id++)
            {
                candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
            }

            llama_token_data_array candidates_p = {candidates.data(), candidates.size(), false};

            if (temp <= 0)
            {
                // Greedy sampling
                id = llama_sample_token_greedy(llm_ctx.get(), &candidates_p);
            }
            else
            {
                if (mirostat == 1)
                {
                    static float mirostat_mu = 2.0f * mirostat_tau;
                    const int mirostat_m = 100;
                    llama_sample_temperature(llm_ctx.get(), &candidates_p, temp);
                    id = llama_sample_token_mirostat(llm_ctx.get(), &candidates_p, mirostat_tau, mirostat_eta, mirostat_m, &mirostat_mu);
                }
                else if (mirostat == 2)
                {
                    static float mirostat_mu = 2.0f * mirostat_tau;
                    llama_sample_temperature(llm_ctx.get(), &candidates_p, temp);
                    id = llama_sample_token_mirostat_v2(llm_ctx.get(), &candidates_p, mirostat_tau, mirostat_eta, &mirostat_mu);
                }
                else
                {
                    // Temperature sampling
                    llama_sample_top_k(llm_ctx.get(), &candidates_p, top_k, 1);
                    llama_sample_tail_free(llm_ctx.get(), &candidates_p, tfs_z, 1);
                    llama_sample_typical(llm_ctx.get(), &candidates_p, typical_p, 1);
                    llama_sample_top_p(llm_ctx.get(), &candidates_p, top_p, 1);
                    llama_sample_temperature(llm_ctx.get(), &candidates_p, temp);
                    id = llama_sample_token(llm_ctx.get(), &candidates_p);
                }
            }
        }

        return id;
    }

    const char *id_to_token(llama_token id)
    {
        const char *token = nullptr;
        if (id == llama_token_eos())
        {
            token = "</s>";
        }
        else
        {
            token = llama_token_to_str(llm_ctx.get(), id);
        }
        return token;
    }

    void reset()
    {
        n_past = 0;
    }

    const auto &get_config() { return minigpt4_model_loader.get_config(); }

private:
    using LLMModel = std::unique_ptr<struct llama_model, decltype([](struct llama_model *c)
                                                                  { llama_free_model(c); })>;

    using LLMContext = std::unique_ptr<struct llama_context, decltype([](struct llama_context *c)
                                                                      { llama_free(c); })>;

    MiniGPT4ModelLoader minigpt4_model_loader;

    ggml_context *model_ctx;

    MiniGPT4Model minigpt4_model;
    LLMModel llm_model = nullptr;
    LLMContext llm_ctx = nullptr;
    llama_context_params llm_params;
    std::size_t n_past = 0;
};

int limit_threads_in_bound(int threads)
{
    if (threads <= 0)
    {
        threads = static_cast<int>(std::thread::hardware_concurrency());
    }
    return threads;
}

bool ends_with(std::string_view s, std::string_view suffix)
{
    return s.size() >= suffix.size() && !s.compare(s.size() - suffix.size(), suffix.size(), suffix);
}

bool contains(std::string_view s, std::string_view v)
{
    return s.find(v) != std::string_view::npos;
}

struct MiniGPT4Context *minigpt4_model_load(const char *path, const char *llm_model, int verbosity, int seed, int n_ctx, int n_batch, bool numa)
{
    LoggingTimer timer("Load model from file");

    fs::path model_path(path);
    if (!fs::exists(model_path))
    {
        ERR("{} does not exist", model_path.string());
        return nullptr;
    }

    fs::path llm_model_path(llm_model);
    if (!fs::exists(llm_model_path))
    {
        ERR("{} does not exist", llm_model_path.string());
        return nullptr;
    }

    INFO("Running from path {}", fs::current_path().string());

    MiniGPT4 *minigpt4 = new MiniGPT4();
    if (auto err = minigpt4->init(model_path, llm_model_path, MiniGPT4Verbosity(verbosity), seed, n_ctx, n_batch, numa))
    {
        ERR("Failed to initialize MiniGPT4: {}", magic_enum::enum_name(err));
        minigpt4_free(reinterpret_cast<struct MiniGPT4Context *>(minigpt4));
        return nullptr;
    }

    struct MiniGPT4Context *ctx = reinterpret_cast<struct MiniGPT4Context *>(minigpt4);

    return ctx;
}

int minigpt4_image_load_from_file(struct MiniGPT4Context *ctx, const char *path, IN struct MiniGPT4Image *image, int flags)
{
#ifdef MINIGPT4_BUILD_WITH_OPENCV
    LoggingTimer timer("Load image from file");

    auto m = cv::imread(path, cv::IMREAD_COLOR);
    cv::cvtColor(m, m, cv::COLOR_BGR2RGB);

    image->width = m.cols;
    image->height = m.rows;
    image->channels = m.channels();
    image->data = new uint8_t[image->width * image->height * image->channels];
    image->format = MiniGPT4ImageFormat::MINIGPT4_IMAGE_FORMAT_U8;
    memcpy(image->data, m.data, m.total() * m.elemSize());

    return MiniGPT4Error::None;
#else
    return MiniGPT4Error::OpenCVNotLinked;
#endif
}

int minigpt4_preprocess_image(struct MiniGPT4Context *ctx, IN const struct MiniGPT4Image *image, OUT struct MiniGPT4Image *preprocessed_image, int flags)
{
#ifdef MINIGPT4_BUILD_WITH_OPENCV
    LoggingTimer timer("Preprocess image");

    MiniGPT4 *minigpt4 = reinterpret_cast<MiniGPT4 *>(ctx);
    const auto &config = minigpt4->get_config();

    if (image->channels != RGB_CHANNELS)
    {
        ERR("Image must have {} channels", RGB_CHANNELS);
        return MiniGPT4Error::ImageChannelsExpectedRGB;
    }

    if (image->format != MiniGPT4ImageFormat::MINIGPT4_IMAGE_FORMAT_U8)
    {
        ERR("Image must be in U8 format");
        return MiniGPT4Error::ImageFormatExpectedU8;
    }

    auto m = cv::Mat(image->height, image->width, CV_8UC3, image->data);

    // TODO: This isn't pytorch resize, so use pillow instead
    // cv::resize(m, m2, cv::Size(IMAGE_RESIZE, IMAGE_RESIZE), 0, 0, cv::INTER_CUBIC);

    // Resize
    m = PillowResize::resize(m, cv::Size(IMAGE_RESIZE, IMAGE_RESIZE), PillowResize::InterpolationMethods::INTERPOLATION_BICUBIC);
    auto mean = cv::Scalar(0.48145466, 0.4578275, 0.40821073);
    auto std = cv::Scalar(0.26862954, 0.26130258, 0.27577711);

    m.convertTo(m, CV_32FC3, 1.0f / 255.0f);
    m = (m - mean) / std;

    // HWC -> CHW
    std::vector<cv::Mat> channels;
    cv::split(m, channels);
    for (auto &c : channels)
    {
        c = c.reshape(1, 1);
    }
    cv::hconcat(channels, m);

    preprocessed_image->width = m.rows;
    preprocessed_image->height = m.cols;
    preprocessed_image->channels = m.channels();
    preprocessed_image->data = new float[preprocessed_image->width * preprocessed_image->height * preprocessed_image->channels];
    preprocessed_image->format = MiniGPT4ImageFormat::MINIGPT4_IMAGE_FORMAT_F32;

    memcpy(preprocessed_image->data, m.data, m.total() * m.elemSize());

    return MiniGPT4Error::None;
#else
    return MiniGPT4Error::OpenCVNotLinked;
#endif
}

int minigpt4_encode_image(struct MiniGPT4Context *ctx, IN struct MiniGPT4Image *image, OUT struct MiniGPT4Embedding *embedding, size_t n_threads)
{
    LoggingTimer timer("Encoding image");

    n_threads = limit_threads_in_bound(n_threads);
    MiniGPT4 *minigpt4 = reinterpret_cast<MiniGPT4 *>(ctx);
    RETURN_IF_ERROR(minigpt4->encode_image(image, embedding, n_threads));

    return MiniGPT4Error::None;
}

#define ADD_STRINGS_CHECK_ERR(...)                                      \
    auto UNIQUIFY(err) = minigpt4->add_strings(__VA_ARGS__, n_threads); \
    if (UNIQUIFY(err))                                                  \
    {                                                                   \
        return UNIQUIFY(err);                                           \
    }

int minigpt4_begin_chat_image(struct MiniGPT4Context *ctx, IN struct MiniGPT4Embedding *image_embedding, const char *s, std::size_t n_threads)
{
    // LoggingTimer timer("Begin chat image");
    
    MiniGPT4 *minigpt4 = reinterpret_cast<MiniGPT4 *>(ctx);
    const auto &config = minigpt4->get_config();

    n_threads = limit_threads_in_bound(n_threads);

    ADD_STRINGS_CHECK_ERR("Human: <Img>");

    if (image_embedding->elements != LLAMA_PROJECTION_EMBEDDING_SIZE_13B && image_embedding->elements != LLAMA_PROJECTION_EMBEDDING_SIZE_7B)
    {
        ERR("LLAMA projection image embedding size not equal {} != {}", image_embedding->elements, LLAMA_PROJECTION_EMBEDDING_SIZE_13B);
        return MiniGPT4Error::LLamaProjectionEmbeddingInvalidSize;
    }

    MiniGPT4Embedding updated_embedding = *image_embedding;
    updated_embedding.elements = LLAMA_PROJECTION_EMBEDDING_SIZE1;

    if (auto err = minigpt4->add_embedding(&updated_embedding, n_threads))
    {
        ERR("Failed to add image embedding: {}", magic_enum::enum_name(err));
        return err;
    }

    ADD_STRINGS_CHECK_ERR("</Img> ");
    ADD_STRINGS_CHECK_ERR(s);
    ADD_STRINGS_CHECK_ERR("### Assistant:");

    return MiniGPT4Error::None;
}

int minigpt4_end_chat_image(struct MiniGPT4Context *ctx, const char **token, std::size_t n_threads, float temp, int32_t top_k, float top_p, float tfs_z, float typical_p, int32_t repeat_last_n, float repeat_penalty, float alpha_presence, float alpha_frequency, int mirostat, float mirostat_tau, float mirostat_eta, int penalize_nl)
{
    // LoggingTimer timer("End chat");

    MiniGPT4 *minigpt4 = reinterpret_cast<MiniGPT4 *>(ctx);
    const auto &config = minigpt4->get_config();

    n_threads = limit_threads_in_bound(n_threads);

    auto id = minigpt4->sample_token(n_threads, temp, top_k, top_p, tfs_z, typical_p, repeat_last_n, repeat_penalty, alpha_presence, alpha_frequency, mirostat, mirostat_tau, mirostat_eta, penalize_nl);
    *token = minigpt4->id_to_token(id);
    minigpt4->add_tokens({id}, n_threads);

    return MiniGPT4Error::None;
}

int minigpt4_system_prompt(struct MiniGPT4Context *ctx, std::size_t n_threads)
{
    // LoggingTimer timer("System prompt");

    MiniGPT4 *minigpt4 = reinterpret_cast<MiniGPT4 *>(ctx);
    const auto &config = minigpt4->get_config();

    n_threads = limit_threads_in_bound(n_threads);

    ADD_STRINGS_CHECK_ERR(SYSTEM_PROMPT.data());

    return MiniGPT4Error::None;
}

int minigpt4_begin_chat(struct MiniGPT4Context *ctx, const char *s, std::size_t n_threads)
{
    // LoggingTimer timer("Begin chat");

    MiniGPT4 *minigpt4 = reinterpret_cast<MiniGPT4 *>(ctx);
    const auto &config = minigpt4->get_config();

    n_threads = limit_threads_in_bound(n_threads);

    ADD_STRINGS_CHECK_ERR("Human: ");
    ADD_STRINGS_CHECK_ERR(s);
    ADD_STRINGS_CHECK_ERR("### Assistant:");

    return MiniGPT4Error::None;
}

int minigpt4_end_chat(struct MiniGPT4Context *ctx, const char **token, std::size_t n_threads, float temp, int32_t top_k, float top_p, float tfs_z, float typical_p, int32_t repeat_last_n, float repeat_penalty, float alpha_presence, float alpha_frequency, int mirostat, float mirostat_tau, float mirostat_eta, int penalize_nl)
{
    return minigpt4_end_chat_image(ctx, token, n_threads, temp, top_k, top_p, tfs_z, typical_p, repeat_last_n, repeat_penalty, alpha_presence, alpha_frequency, mirostat, mirostat_tau, mirostat_eta, penalize_nl);
}

int minigpt4_reset_chat(struct MiniGPT4Context *ctx)
{
    MiniGPT4 *minigpt4 = reinterpret_cast<MiniGPT4 *>(ctx);
    const auto &config = minigpt4->get_config();

    minigpt4->reset();
    return MiniGPT4Error::None;
}

int minigpt4_contains_eos_token(const char *s)
{
    auto str = std::string_view(s);
    if (str == EOS_TOKEN_SUFFIX)
    {
        return MiniGPT4Error::EosToken;
    }
    return MiniGPT4Error::None;
}

int minigpt4_is_eos(const char *s)
{
    auto str = std::string_view(s);
    if (ends_with(str, EOS_SUFFIX))
    {
        return MiniGPT4Error::Eos;
    }
    return MiniGPT4Error::None;
}

int minigpt4_free(struct MiniGPT4Context *ctx)
{
    MiniGPT4 *minigpt4 = reinterpret_cast<MiniGPT4 *>(ctx);
    delete minigpt4;
    return MiniGPT4Error::None;
}

int minigpt4_free_image(struct MiniGPT4Image *image)
{
    if (image->data)
    {
        delete (float *)image->data;
        image->data = nullptr;
    }
    return MiniGPT4Error::None;
}

int minigpt4_free_embedding(struct MiniGPT4Embedding *embedding)
{
    if (embedding->data)
    {
        delete (float *)embedding->data;
        embedding->data = nullptr;
    }
    return MiniGPT4Error::None;
}

const char *minigpt4_error_code_to_string(int error_code)
{
    std::string_view error_string = magic_enum::enum_name(MiniGPT4Error(error_code));
    return error_string.data();
}

int minigpt4_quantize_model(const char *in_path, const char *out_path, int data_type_)
{
    fs::path in_path_fs(in_path);
    fs::path out_path_fs(out_path);
    MiniGPT4DataType data_type = static_cast<MiniGPT4DataType>(data_type_);

    if (!fs::exists(in_path_fs))
    {
        return MiniGPT4Error::PathDoesNotExist;
    }

    MiniGPT4ModelLoader model_loader;
    RETURN_IF_ERROR(model_loader.load(in_path_fs));

    ggml_type out_type;
    if (auto type_ = data_type_to_ggml_type(data_type); type_.has_value())
    {
        out_type = type_.value();
    }
    else
    {
        return magic_enum::enum_integer(type_.error());
    }

    size_t max_in_size = 0;
    size_t max_out_size = 0;

    // Init tables
    ggml_free(ggml_init({0, NULL, true}));

    for (auto &[model_name, model] : model_loader.get_models())
    {
        for (auto &[name, t] : model.get_tensors())
        {
            auto orig_tensor_size = t.get_size_in_bytes();
            max_in_size = std::max(max_in_size, orig_tensor_size);

            if (t.type == GGML_TYPE_F16)
            {
                max_out_size = std::max(max_out_size, max_in_size * 2);
            }

            // Overwrite the type
            auto orig_type = t.type;
            t.type = out_type;

            auto new_tensor_size = t.get_size_in_bytes();

            max_out_size = std::max(max_out_size, new_tensor_size);
            t.type = orig_type;
        }
    }

    std::array<int64_t, 16> entire_history;

    Buffer in_scratch(max_in_size);
    Buffer out_scratch(max_out_size);
    Buffer other_scratch(max_out_size);

    uint8_t *in_buf = in_scratch.addr;
    uint8_t *out_buf = out_scratch.addr;
    uint8_t *other_buf = other_scratch.addr;

    INFO("Quantizing model: {}", in_path_fs.string());

    size_t orig_total_size = 0;
    size_t new_total_size = 0;

    std::vector<Buffer> new_model_buffers;

    for (auto &[model_name, model] : model_loader.get_models())
    {
        for (auto &[name, t] : model.get_tensors())
        {
            auto orig_size = t.get_size_in_bytes();
            auto new_size = orig_size;

            memcpy(in_buf, t.get_file_address(), orig_size);

            auto data_type = t.type;
            if ((data_type == GGML_TYPE_F16 || data_type == GGML_TYPE_F32) &&
                                                  ends_with(t.name, "weight") && t.shape.size() >= 2 &&

                                                  !contains(t.name, "norm") &&
                                                  !contains(t.name, "Norm") &&
                                                  
                                                //   model_name != "visual_encoder" &&

                                                //   !contains(t.name, ".norm1") &&
                                                //   !contains(t.name, ".attn") &&
                                                //   !contains(t.name, ".norm2") &&
                                                //   !contains(t.name, ".mlp") &&

                                                  model_name != "ln_vision" &&

                                                  model_name != "query_tokens" &&
                                                
                                                //   model_name != "Qformer" &&

                                                //   !contains(t.name, ".self") &&
                                                //   !contains(t.name, ".attention") &&
                                                //   !contains(t.name, ".crossattention") &&
                                                //   !contains(t.name, ".intermediate_query") &&
                                                //   !contains(t.name, ".output_query") &&
                                                
                                                  model_name != "llama_proj" &&
                                                  t.name != "patch_embed.proj.weight")
            {
                auto nelements = t.total_shape();
                if (data_type == GGML_TYPE_F16)
                {
                    ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t *>(in_buf), reinterpret_cast<float *>(other_buf), nelements);
                    in_buf = other_buf;
                }
                std::array<int64_t, 16> current_history;
                new_size = ggml_quantize_chunk(out_type, (const float *)in_buf, out_buf, 0, nelements, current_history.data());

                INFO("{}.{} | Original {:10.2f} MB -> New {:10.2f} MB", model_name, t.name, bytes_to_mb(orig_size), bytes_to_mb(new_size));
                std::array<float, 16> normalized_history;
                for (auto i = 0; i < current_history.size(); i++)
                {
                    const auto &h = current_history[i];
                    entire_history[i] += h;
                    normalized_history[i] = static_cast<float>(h) / nelements;
                }
                INFO("History : {:6.4f}", fmt::join(normalized_history, ", "));
    
                t.type = out_type;
            }
            else
            {
                INFO("{}.{} | Original {:10.2f} MB -> New {:10.2f} MB", model_name, t.name, bytes_to_mb(orig_size), bytes_to_mb(new_size));
                out_buf = in_buf;
            }

            orig_total_size += orig_size;
            new_total_size += new_size;

            Buffer new_model_buffer(new_size);
            memcpy(new_model_buffer.addr, out_buf, new_size);

            // Now update the reference
            t.tensor_buf.addr = new_model_buffer.addr;
            t.tensor_buf.size = new_size;

            new_model_buffers.emplace_back(std::move(new_model_buffer));
        }
    }

    model_loader.set_file_data_type(out_type);

    INFO("Original size {:10.2f} MB", bytes_to_mb(orig_total_size));
    INFO("Quantized size {:10.2f} MB", bytes_to_mb(new_total_size));
    INFO("Compression ratio {:10.2f}", bytes_to_mb(orig_total_size) / bytes_to_mb(new_total_size));

    auto history_sum = std::accumulate(std::begin(entire_history), std::end(entire_history), 0);

    std::array<float, 16> normalized_history;
    std::transform(std::begin(entire_history), std::end(entire_history), std::begin(normalized_history), [history_sum](auto h)
                   { return static_cast<float>(h) / history_sum; });
    INFO("Entire history: {}", fmt::join(normalized_history, ", "));

    RETURN_IF_ERROR(model_loader.dump(out_path_fs));

    return MiniGPT4Error::None;
}

void minigpt4_set_verbosity(int verbosity)
{
    global_verbosity = MiniGPT4Verbosity(verbosity);
}