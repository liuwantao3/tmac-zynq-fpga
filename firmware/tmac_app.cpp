/*
 * T-MAC FPGA Application
 * End-to-end LLM inference with ARM+FPGA hybrid execution
 *
 * Build: arm-linux-gnueabihf-g++ -o tmac_app tmac_app.cpp tmac_runtime.cpp tmac_fpga.cpp -lpthread
 */

#include "tmac_runtime.hpp"
#include <stdio.h>
#include <vector>
#include <chrono>

using namespace tmac;
using namespace std::chrono;

struct AppConfig {
    const char* model_path;
    const char* weight_file;
    int max_seq_len;
    bool use_fpga;
    bool verbose;
};

void print_usage(const char* prog) {
    printf("T-MAC FPGA Inference Application\n");
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  -m <path>   Model path\n");
    printf("  -w <file>   Weight file\n");
    printf("  -l <len>    Max sequence length (default: 512)\n");
    printf("  -f          Use FPGA acceleration (default: on)\n");
    printf("  -v          Verbose mode\n");
    printf("  -h          Show this help\n");
}

int main(int argc, char** argv) {
    AppConfig config = {
        "/models/qwen2-0.5b",
        "weights.bin",
        512,
        true,
        false
    };

    (void)config; // suppress unused warning until FPGA integration

    // Parse args
    int opt;
    while ((opt = getopt(argc, argv, "m:w:l:fhv")) != -1) {
        switch (opt) {
            case 'm': config.model_path = optarg; break;
            case 'w': config.weight_file = optarg; break;
            case 'l': config.max_seq_len = atoi(optarg); break;
            case 'f': config.use_fpga = true; break;
            case 'v': config.verbose = true; break;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }

    printf("================================================\n");
    printf("T-MAC FPGA Inference Engine\n");
    printf("================================================\n");
    printf("Model: %s\n", config.model_path);
    printf("Weights: %s\n", config.weight_file);
    printf("Max seq: %d\n", config.max_seq_len);
    printf("FPGA: %s\n", config.use_fpga ? "enabled" : "disabled");
    printf("================================================\n\n");

    // Initialize runtime
    ModelConfig model_config = {
        .vocab_size = 151936,
        .hidden_dim = 896,
        .intermediate_dim = 4864,
        .num_layers = 24,
        .max_seq_len = config.max_seq_len,
        .num_heads = 14,
        .head_dim = 64
    };

    TMacRuntime runtime;
    int ret = runtime.init(model_config);
    if (ret != 0) {
        fprintf(stderr, "[ERROR] Failed to initialize runtime\n");
        return 1;
    }

    // Load weights
    ret = runtime.load_weights(config.weight_file);
    if (ret != 0) {
        fprintf(stderr, "[ERROR] Failed to load weights\n");
        return 1;
    }

    // Interactive inference loop
    printf("\nReady for inference. Type 'quit' to exit.\n\n");

    char input[4096];
    while (true) {
        printf("> ");
        if (!fgets(input, sizeof(input), stdin)) break;
        input[strcspn(input, "\n")] = 0;

        if (strcmp(input, "quit") == 0 || strcmp(input, "q") == 0) {
            printf("Exiting...\n");
            break;
        }

        if (strlen(input) == 0) continue;

        // Tokenize (simplified - would use actual tokenizer)
        std::vector<int> input_ids;
        for (size_t i = 0; i < strlen(input); i++) {
            input_ids.push_back(static_cast<int>(input[i]));
        }

        // Add EOS token
        input_ids.push_back(151643);

        // Forward pass
        auto start = high_resolution_clock::now();

        std::vector<int> output_ids;
        ret = runtime.forward(input_ids, output_ids);

        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start);

        if (ret == 0) {
            printf("[Output tokens: %zu, Time: %ld ms]\n",
                   output_ids.size(), duration.count());
            printf("Response: ");
            for (int id : output_ids) {
                // Simplified - would decode token IDs
                if (id < 128) printf("%c", id);
            }
            printf("\n\n");
        } else {
            printf("[ERROR] Inference failed with code %d\n\n", ret);
        }
    }

    return 0;
}