#pragma once

#include <stdint.h>
#include <stdlib.h>

#ifdef MINIGPT4_SHARED
#ifdef _WIN32
#ifdef MINIGPT4_BUILD
#define MINIGPT4_API __declspec(dllexport)
#else
#define MINIGPT4_API __declspec(dllimport)
#endif
#else
#define MINIGPT4_API __attribute__((visibility("default")))
#endif
#else
#define MINIGPT4_API
#endif

#define IN
#define OUT

#ifdef __cplusplus
extern "C"
{
#endif

struct MiniGPT4Context;

enum MiniGPT4DataType
{
    F16,
    F32,
    I32,
    L64,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
};

enum MiniGPT4Verbosity
{
    MINIGPT4_VERBOSITY_NONE,
    MINIGPT4_VERBOSITY_ERROR,
    MINIGPT4_VERBOSITY_INFO,
    MINIGPT4_VERBOSITY_DEBUG,
};

enum MiniGPT4ImageFormat
{
    MINIGPT4_IMAGE_FORMAT_UNKNOWN,
    MINIGPT4_IMAGE_FORMAT_F32,
    MINIGPT4_IMAGE_FORMAT_U8,
};

struct MiniGPT4Image
{
    void *data;
    int width;
    int height;
    int channels;
    MiniGPT4ImageFormat format;
};

struct MiniGPT4Embedding
{
    float *data;
    size_t elements;
};

struct MiniGPT4Embeddings
{
    struct MiniGPT4Embedding *embeddings;
    size_t n_embeddings;
};

struct MiniGPT4Images
{
    struct MiniGPT4Image *images;
    size_t n_images;
};

enum MiniGPT4ImageLoadFlags
{
    MINIGPT4_IMAGE_LOAD_FLAG_NONE,
};

MINIGPT4_API struct MiniGPT4Context *minigpt4_model_load(const char *path, const char *llm_model, int verbosity, int seed, int n_ctx, int n_batch, bool numa);
MINIGPT4_API int minigpt4_image_load_from_file(struct MiniGPT4Context *ctx, const char *path, IN struct MiniGPT4Image *image, int flags);
MINIGPT4_API int minigpt4_preprocess_image(struct MiniGPT4Context *ctx, IN const struct MiniGPT4Image *image, OUT struct MiniGPT4Image *preprocessed_image, int flags);
MINIGPT4_API int minigpt4_encode_image(struct MiniGPT4Context *ctx, IN struct MiniGPT4Image *image, OUT struct MiniGPT4Embedding *embedding, size_t n_threads);
MINIGPT4_API int minigpt4_begin_chat_image(struct MiniGPT4Context *ctx, IN struct MiniGPT4Embedding *image_embedding, const char *s, size_t n_threads);
MINIGPT4_API int minigpt4_end_chat_image(struct MiniGPT4Context *ctx, const char **token, size_t n_threads, float temp, int32_t top_k, float top_p, float tfs_z, float typical_p, int32_t repeat_last_n, float repeat_penalty, float alpha_presence, float alpha_frequency, int mirostat, float mirostat_tau, float mirostat_eta, int penalize_nl);
MINIGPT4_API int minigpt4_system_prompt(struct MiniGPT4Context *ctx, size_t n_threads);
MINIGPT4_API int minigpt4_begin_chat(struct MiniGPT4Context *ctx, const char *s, size_t n_threads);
MINIGPT4_API int minigpt4_end_chat(struct MiniGPT4Context *ctx, const char **token, size_t n_threads, float temp, int32_t top_k, float top_p, float tfs_z, float typical_p, int32_t repeat_last_n, float repeat_penalty, float alpha_presence, float alpha_frequency, int mirostat, float mirostat_tau, float mirostat_eta, int penalize_nl);
MINIGPT4_API int minigpt4_reset_chat(struct MiniGPT4Context *ctx);
MINIGPT4_API int minigpt4_contains_eos_token(const char *s);
MINIGPT4_API int minigpt4_is_eos(const char *s);
MINIGPT4_API int minigpt4_free(struct MiniGPT4Context *ctx);
MINIGPT4_API int minigpt4_free_image(struct MiniGPT4Image *image);
MINIGPT4_API int minigpt4_free_embedding(struct MiniGPT4Embedding *embedding);
MINIGPT4_API const char *minigpt4_error_code_to_string(int error_code);
MINIGPT4_API int minigpt4_quantize_model(const char *in_path, const char *out_path, int data_type);
MINIGPT4_API void minigpt4_set_verbosity(int verbosity);

#ifdef __cplusplus
}
#endif