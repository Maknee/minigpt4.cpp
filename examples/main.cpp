#include <filesystem>

#include "minigpt4.h"
#include <argparse/argparse.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

#define INFO(...) spdlog::info(__VA_ARGS__)
#define ERR(...)                \
    spdlog::error(__VA_ARGS__); \
    std::cerr << std::endl;
#define ERR_EXIT(...) \
    ERR(__VA_ARGS__); \
    exit(-1);
#define CHECK_ERR_EXIT(x, ...)                                      \
    if (x)                                                          \
    {                                                               \
        ERR("ERROR MESSAGE: {}", minigpt4_error_code_to_string(x)); \
        ERR_EXIT(__VA_ARGS__)                                       \
    }

namespace fs = std::filesystem;

int main(int argc, char **argv)
{
    spdlog::set_pattern("[%H:%M:%S %z] [%n] [%^---%L---%$] [thread %t] %v");
    spdlog::stopwatch sw;

    argparse::ArgumentParser args("MiniGPT4.cpp", "1.0", argparse::default_arguments::help, false);

    args.add_argument("-v", "--verbose")
        .help("increase output verbosity")
        .default_value(0)
        .scan<'i', int>();

    args.add_argument("-m", "--model")
        .required()
        .help("Path to the model file")
        .default_value(std::string("minigpt4-13B-f16.bin"));

    args.add_argument("-lm", "--llm_model")
        .required()
        .help("Path to language model")
        .default_value(std::string("ggml-vicuna-13b-v0-q4_1.bin"));

    args.add_argument("-t", "--threads")
        .help("Number of threads to use")
        .default_value(0)
        .scan<'i', int>();

    args.add_argument("--image")
        .required()
        .help("Images to encode")
        .nargs(argparse::nargs_pattern::at_least_one)
        .default_value(std::string{"../minigpt4/images/llama.png"});

    args.add_argument("--texts")
        .required()
        .help("Texts to encode")
        .nargs(argparse::nargs_pattern::at_least_one)
        .default_value(std::vector<std::string>{"what is the text in the picture?", "what is the color of it?"});

    args.add_argument("--temp")
        .help("temperature")
        .default_value(0.80f)
        .scan<'f', float>();

    args.add_argument("--top_k")
        .help("top_k")
        .default_value(40)
        .scan<'i', int>();

    args.add_argument("--top_p")
        .help("top_p")
        .default_value(0.90f)
        .scan<'f', float>();

    args.add_argument("--tfs_z")
        .help("tfs_z")
        .default_value(1.00f)
        .scan<'f', float>();

    args.add_argument("--typical_p")
        .help("typical_p")
        .default_value(1.00f)
        .scan<'f', float>();

    args.add_argument("--repeat_last_n")
        .help("repeat_last_n")
        .default_value(64)
        .scan<'i', int>();

    args.add_argument("--repeat_penalty")
        .help("repeat_penalty")
        .default_value(1.10f)
        .scan<'f', float>();

    args.add_argument("--alpha_presence")
        .help("alpha_presence")
        .default_value(1.00f)
        .scan<'f', float>();

    args.add_argument("--alpha_frequency")
        .help("alpha_frequency")
        .default_value(1.00f)
        .scan<'f', float>();

    args.add_argument("--mirostat")
        .help("mirostat")
        .default_value(0)
        .scan<'i', int>();

    args.add_argument("--mirostat_tau")
        .help("mirostat_tau")
        .default_value(5.00f)
        .scan<'f', float>();

    args.add_argument("--mirostat_eta")
        .help("mirostat_eta")
        .default_value(1.00f)
        .scan<'f', float>();

    args.add_argument("--penalize_nl")
        .help("penalize_nl")
        .default_value(1)
        .scan<'i', int>();

    args.add_argument("--n_ctx")
        .help("n_ctx")
        .default_value(2048)
        .scan<'i', int>();

    args.add_argument("--n_batch_size")
        .help("n_batch_size")
        .default_value(512)
        .scan<'i', int>();

    args.add_argument("--seed")
        .help("seed")
        .default_value(1337)
        .scan<'i', int>();

    args.add_argument("--numa")
        .help("numa")
        .default_value(0)
        .scan<'i', int>();

    args.parse_args(argc, argv);

    auto model = args.get<std::string>("model");
    auto llm_model = args.get<std::string>("llm_model");
    auto verbose = args.get<int>("verbose");
    auto threads = args.get<int>("threads");
    auto texts = args.get<std::vector<std::string>>("texts");
    auto image_path = args.get<std::string>("image");
    auto temp = args.get<float>("temp");
    auto top_k = args.get<int32_t>("top_k");
    auto top_p = args.get<float>("top_p");
    auto tfs_z = args.get<float>("tfs_z");
    auto typical_p = args.get<float>("typical_p");
    auto repeat_last_n = args.get<int32_t>("repeat_last_n");
    auto repeat_penalty = args.get<float>("repeat_penalty");
    auto alpha_presence = args.get<float>("alpha_presence");
    auto alpha_frequency = args.get<float>("alpha_frequency");
    auto mirostat = args.get<int32_t>("mirostat");
    auto mirostat_tau = args.get<float>("mirostat_tau");
    auto mirostat_eta = args.get<float>("mirostat_eta");
    auto penalize_nl = args.get<int>("penalize_nl");
    auto seed = args.get<int>("seed");
    auto n_ctx = args.get<int>("n_ctx");
    auto n_batch_size = args.get<int>("n_batch_size");
    auto numa = args.get<int>("numa");

    if (threads <= 0)
    {
        threads = static_cast<int>(std::thread::hardware_concurrency());
    }

    INFO("=== Args ===");
    INFO("Model: {}", model);
    INFO("LLM Model: {}", llm_model);
    INFO("Verbose: {}", verbose);
    INFO("Threads: {}", threads);
    INFO("Texts: {}", fmt::join(texts, ", "));
    INFO("Images: {}", image_path);
    INFO("============");
    INFO("Running from {}", fs::current_path().string());

    if (!fs::exists(model))
    {
        ERR("Model file '{}' does not exist", model);
        return 1;
    }

    if (!fs::exists(llm_model))
    {
        ERR("LLM Model file '{}' does not exist", llm_model);
        return 1;
    }

    if (!fs::exists(image_path))
    {
        ERR("Image file '{}' does not exist", image_path);
        return 1;
    }

    auto ctx = minigpt4_model_load(model.c_str(), llm_model.c_str(), verbose, seed, n_ctx, n_batch_size, numa);
    if (!ctx)
    {
        ERR("Failed to load model");
        return 1;
    }

    MiniGPT4Image image{};
    {
        auto err = minigpt4_image_load_from_file(ctx, image_path.c_str(), &image, 0);
        CHECK_ERR_EXIT(err, "Failed to load image for {}", image_path);
    }

    MiniGPT4Image preprocessed_image{};
    {
        auto err = minigpt4_preprocess_image(ctx, &image, &preprocessed_image, 0);
        CHECK_ERR_EXIT(err, "Failed to preprocess image for {}", image_path);
    }

    MiniGPT4Embedding image_embedding{};
    {
        auto err = minigpt4_encode_image(ctx, &preprocessed_image, &image_embedding, threads);
        CHECK_ERR_EXIT(err, "Failed to encode image for {}", image_path);
    }

    MiniGPT4Embeddings minigpt4_image_embeddings{
        .embeddings = &image_embedding,
        .n_embeddings = 1,
    };

    {
        int err = minigpt4_system_prompt(ctx, threads);
        CHECK_ERR_EXIT(err, "Failed have system prompt");
    }

    {
        const auto &text = texts[0];
        int err = minigpt4_begin_chat_image(ctx, &image_embedding, texts[0].c_str(), threads);
        CHECK_ERR_EXIT(err, "Failed to chat image {}", image_path);
        const char *token = nullptr;
        std::string response;
        response.reserve(2048);

        do
        {
            if (token && !minigpt4_contains_eos_token(token))
            {
                std::cout << token << std::flush;
            }
            int err = minigpt4_end_chat_image(ctx, &token, threads, temp, top_k, top_p, tfs_z, typical_p, repeat_last_n, repeat_penalty, alpha_presence, alpha_frequency, mirostat, mirostat_tau, mirostat_eta, penalize_nl);
            CHECK_ERR_EXIT(err, "Failed to generate chat image");
            response += token;
        } while (!minigpt4_is_eos(response.c_str()));
    }

    {
        if (texts.size() > 1)
        {
            for (auto i = 1; i < texts.size(); i++)
            {
                const auto &text = texts[i];
                int err = minigpt4_begin_chat(ctx, text.c_str(), threads);
                CHECK_ERR_EXIT(err, "Failed to begin chat");
                const char *token = nullptr;
                std::string response;
                response.reserve(2048);

                do
                {
                    if (token && !minigpt4_contains_eos_token(token))
                    {
                        std::cout << token << std::flush;
                    }
                    int err = minigpt4_end_chat(ctx, &token, threads, temp, top_k, top_p, tfs_z, typical_p, repeat_last_n, repeat_penalty, alpha_presence, alpha_frequency, mirostat, mirostat_tau, mirostat_eta, penalize_nl);
                    CHECK_ERR_EXIT(err, "Failed to generate chat");
                    response += token;
                } while (!minigpt4_is_eos(response.c_str()));
            }
        }
    }

    const auto entire_time = sw.elapsed();

    minigpt4_free_image(&image);
    minigpt4_free_image(&preprocessed_image);
    minigpt4_free_embedding(&image_embedding);
    minigpt4_free(ctx);

    if (verbose)
    {
        INFO("MiniGPT4");
        INFO("Entire session time spent: {:10.2f}", entire_time.count() * 1000);
    }

    return 0;
}