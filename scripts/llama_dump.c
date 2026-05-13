// Dump hidden states from llama.cpp for comparison
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "llama.h"

int main(int argc, char** argv) {
    const char* model_path = "/Users/arctic/Downloads/qwen2-0_5b-instruct-q4_k_m.gguf";
    
    // Initialize backend
    llama_backend_init();
    
    // Load model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    
    llama_model* model = llama_load_model(model_path, model_params);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    
    // Create context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 512;
    ctx_params.n_batch = 512;
    
    llama_context* ctx = llama_new_context_with_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        llama_free_model(model);
        return 1;
    }
    
    // Tokenize prompt
    const char* prompt = "<|im_start|>user\nHello! How are you?<|im_end|>\n<|im_start|>assistant\n";
    int n_tokens = llama_tokenize(model, prompt, strlen(prompt), NULL, 0, true, true);
    int* tokens = malloc(n_tokens * sizeof(int));
    llama_tokenize(model, prompt, strlen(prompt), tokens, n_tokens, true, true);
    
    printf("Prompt tokens (%d):", n_tokens);
    for (int i = 0; i < n_tokens; i++) {
        printf(" %d", tokens[i]);
    }
    printf("\n");
    
    // Process prompt
    if (llama_decode(ctx, n_tokens, tokens, n_tokens, 0) != 0) {
        fprintf(stderr, "Failed to process prompt\n");
        return 1;
    }
    
    // Get logits for last token
    float* logits = llama_get_logits_ith(ctx, n_tokens - 1);
    printf("\nLogits (first 20):");
    for (int i = 0; i < 20; i++) {
        printf(" %.4f", logits[i]);
    }
    printf("...\n");
    
    // Get embedding for last token (if available)
    // Note: llama.cpp doesn't expose hidden states directly
    // But we can get the embedding from the model if embedding mode is enabled
    
    printf("\nModel n_embd: %d\n", llama_n_embd(model));
    
    // Generate one token to verify
    int next_token = llama_sample_token_greedy(ctx, logits);
    printf("Next token: %d\n", next_token);
    
    // Decode next token
    const char* token_str = llama_token_to_piece(model, next_token);
    printf("Next token text: %s\n", token_str);
    
    // Cleanup
    free(tokens);
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    
    return 0;
}
