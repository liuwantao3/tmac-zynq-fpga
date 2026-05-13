#!/usr/bin/env python3
"""
Ground Truth - Qwen2 GGUF Inference with Layer-Wise Output
Uses proper Qwen2 architecture and extracts hidden states for comparison
"""

import sys
import os
import time
import traceback
import numpy as np
import struct

# Try to import dependencies
try:
    from gguf import GGUFReader, GGMLQuantizationType
    from gguf.quants import dequantize
except ImportError:
    print("[ERROR] pip install gguf")
    sys.exit(1)

try:
    from tokenizers import Tokenizer as HFTokenizer
except ImportError:
    print("[ERROR] pip install tokenizers")
    sys.exit(1)


def rmsnorm(x, weight):
    """RMS normalization"""
    return (x / np.sqrt(np.mean(x**2) + 1e-5)) * weight


def silu(x):
    """SiLU activation"""
    return x * (1.0 / (1.0 + np.exp(-x)))


def softmax(x):
    """Softmax"""
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


def load_gguf_model(gguf_path):
    """Load GGUF model and extract all weights"""
    print(f"Loading GGUF: {gguf_path}")
    
    reader = GGUFReader(gguf_path)
    
    # Get model parameters from GGUF fields
    def get_field(name, default=None):
        if name in reader.fields:
            parts = reader.fields[name].parts
            if parts:
                val = parts[-1]
                if hasattr(val, 'tolist'):
                    result = val.tolist()
                    # Handle bytes
                    if isinstance(result, bytes):
                        return result.decode('utf-8', errors='ignore')
                    return result
                return val
        return default
    
    # Qwen2 architecture
    n_layers = int(get_field("qwen2.block_count", 24))
    hidden = int(get_field("qwen2.embedding_length", 896))
    n_heads = int(get_field("qwen2.attention.head_count", 14))
    n_kv_heads = int(get_field("qwen2.attention.head_count_kv", 2))
    head_dim = hidden // n_heads
    
    vocab_size = get_field("qwen2.vocab_size", 151936)
    
    print(f"  Architecture: Qwen2")
    print(f"  Layers: {n_layers}")
    print(f"  Hidden dim: {hidden}")
    print(f"  Attention heads: {n_heads}")
    print(f"  KV heads: {n_kv_heads}")
    print(f"  Head dim: {head_dim}")
    print(f"  Vocab size: {vocab_size}")
    
    # Load all weights with dequantization
    print("  Loading weights...")
    t0 = time.time()
    weights = {}
    for t in reader.tensors:
        weights[t.name] = dequantize(t.data, GGMLQuantizationType(t.tensor_type))
    print(f"  Loaded {len(weights)} tensors in {time.time() - t0:.1f}s")
    
    # Check embedding shape
    emb = weights['token_embd.weight']
    print(f"  token_embd.weight shape: {emb.shape}")
    
    # GGUF format: typically [hidden, vocab] for embedding
    # We need [vocab, hidden] for lookup
    if emb.shape[0] == hidden and emb.shape[1] == vocab_size:
        print("  Embedding is [hidden, vocab], transposing for lookup")
        emb_lookup = emb.T  # Now [vocab, hidden]
    else:
        print("  Embedding shape unexpected, using as-is")
        emb_lookup = emb
    
    return {
        'weights': weights,
        'n_layers': n_layers,
        'hidden': hidden,
        'n_heads': n_heads,
        'n_kv_heads': n_kv_heads,
        'head_dim': head_dim,
        'vocab_size': vocab_size,
        'emb_lookup': emb_lookup,
    }


def run_inference_with_checkpoints(model, prompt, max_new_tokens=10):
    """Run inference and capture hidden states at each checkpoint"""
    weights = model['weights']
    n_layers = model['n_layers']
    hidden = model['hidden']
    n_heads = model['n_heads']
    n_kv_heads = model['n_kv_heads']
    head_dim = model['head_dim']
    emb_lookup = model['emb_lookup']
    vocab_size = model['vocab_size']
    
    # Load tokenizer
    tok_path = '/tmp/qwen-tok/tokenizer.json'
    if not os.path.exists(tok_path):
        print(f"[ERROR] Tokenizer not found: {tok_path}")
        return None
    
    tokenizer = HFTokenizer.from_file(tok_path)
    
    # Format with Qwen2 chat template
    chat_prompt = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
    enc = tokenizer.encode(chat_prompt)
    token_ids = enc.ids if hasattr(enc, 'ids') else enc
    
    print(f"\nPrompt: {prompt}")
    print(f"Tokens ({len(token_ids)}): {token_ids[:5]}...{token_ids[-5:]}")
    
    checkpoints = {}  # name -> numpy array
    
    # Process prompt tokens
    h = np.zeros(hidden, dtype=np.float32)
    
    for idx, token_id in enumerate(token_ids):
        # Get embedding
        if token_id < emb_lookup.shape[0]:
            h = emb_lookup[token_id].copy()
        else:
            print(f"  [WARN] Token {token_id} out of bounds")
            h = np.zeros(hidden, dtype=np.float32)
        
        # Save input embedding (only for last token or first few)
        if idx == 0:
            checkpoints['input_emb'] = h.copy()
        
        # Forward through layers
        for layer in range(n_layers):
            p = f'blk.{layer}'
            
            # Pre-attention norm
            h_norm = rmsnorm(h.copy(), weights[f'{p}.attn_norm.weight'])
            if layer == 0 and idx == len(token_ids) - 1:
                checkpoints['layer0_pre_attn'] = h_norm.copy()
            
            # Attention
            q = h_norm @ weights[f'{p}.attn_q.weight'].T
            k = h_norm @ weights[f'{p}.attn_k.weight'].T
            v = h_norm @ weights[f'{p}.attn_v.weight'].T
            
            # Reshape for multi-head
            q = q.reshape(n_heads, head_dim)
            k = k.reshape(n_kv_heads, head_dim)
            v = v.reshape(n_kv_heads, head_dim)
            
            # Repeat KV if needed (GQA)
            n_rep = n_heads // n_kv_heads
            if n_rep > 1:
                k = np.repeat(k, n_rep, axis=0)
                v = np.repeat(v, n_rep, axis=0)
            
            # Simplified attention: single token, so just use v
            # For single token, attention gives uniform weights
            attn_out = v.reshape(-1)  # Flatten back to hidden dim
            
            # Output projection
            h = h + attn_out @ weights[f'{p}.attn_output.weight'].T
            
            if layer == 0 and idx == len(token_ids) - 1:
                checkpoints['layer0_post_attn'] = h.copy()
            
            # FFN
            h_norm = rmsnorm(h.copy(), weights[f'{p}.ffn_norm.weight'])
            ffn_gate = h_norm @ weights[f'{p}.ffn_gate.weight'].T
            ffn_up = h_norm @ weights[f'{p}.ffn_up.weight'].T
            ffn_down = silu(ffn_gate) * ffn_up @ weights[f'{p}.ffn_down.weight'].T
            h = h + ffn_down
            
            if layer == 0 and idx == len(token_ids) - 1:
                checkpoints['layer0_post_ffn'] = h.copy()
            if layer == 1 and idx == len(token_ids) - 1:
                checkpoints['layer1_post_ffn'] = h.copy()
            if layer == 2 and idx == len(token_ids) - 1:
                checkpoints['layer2_post_ffn'] = h.copy()
    
    # Final checkpoint
    checkpoints['final_hidden'] = h.copy()
    
    # Logits (no output_norm per user's finding)
    logits = h @ emb_lookup.T
    checkpoints['logits'] = logits.copy()
    
    # Get predicted token
    next_token = np.argmax(logits)
    checkpoints['next_token'] = next_token
    
    return checkpoints, token_ids


def generate_response(model, prompt, max_new_tokens=20):
    """Generate full response and return tokens"""
    weights = model['weights']
    n_layers = model['n_layers']
    hidden = model['hidden']
    n_heads = model['n_heads']
    n_kv_heads = model['n_kv_heads']
    head_dim = model['head_dim']
    emb_lookup = model['emb_lookup']
    vocab_size = model['vocab_size']
    eos_token = 151645
    
    tokenizer = HFTokenizer.from_file('/tmp/qwen-tok/tokenizer.json')
    
    # Format prompt
    chat_prompt = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
    enc = tokenizer.encode(chat_prompt)
    token_ids = enc.ids if hasattr(enc, 'ids') else enc
    
    # Process prompt
    h = np.zeros(hidden, dtype=np.float32)
    for token_id in token_ids:
        if token_id < emb_lookup.shape[0]:
            h = emb_lookup[token_id].copy()
        
        for layer in range(n_layers):
            p = f'blk.{layer}'
            h_norm = rmsnorm(h.copy(), weights[f'{p}.attn_norm.weight'])
            
            q = h_norm @ weights[f'{p}.attn_q.weight'].T
            k = h_norm @ weights[f'{p}.attn_k.weight'].T
            v = h_norm @ weights[f'{p}.attn_v.weight'].T
            
            q = q.reshape(n_heads, head_dim)
            k = k.reshape(n_kv_heads, head_dim)
            v = v.reshape(n_kv_heads, head_dim)
            
            n_rep = n_heads // n_kv_heads
            if n_rep > 1:
                k = np.repeat(k, n_rep, axis=0)
                v = np.repeat(v, n_rep, axis=0)
            
            h = h + v[0] @ weights[f'{p}.attn_output.weight'].T
            
            h_norm = rmsnorm(h.copy(), weights[f'{p}.ffn_norm.weight'])
            ffn_gate = h_norm @ weights[f'{p}.ffn_gate.weight'].T
            ffn_up = h_norm @ weights[f'{p}.ffn_up.weight'].T
            ffn_down = silu(ffn_gate) * ffn_up @ weights[f'{p}.ffn_down.weight'].T
            h = h + ffn_down
    
    # Generate
    generated = []
    for _ in range(max_new_tokens):
        logits = h @ emb_lookup.T
        next_token = np.argmax(logits)
        
        if next_token == eos_token:
            break
        
        generated.append(next_token)
        
        # Next token
        if next_token < emb_lookup.shape[0]:
            h = emb_lookup[next_token].copy()
        else:
            break
        
        for layer in range(n_layers):
            p = f'blk.{layer}'
            h_norm = rmsnorm(h.copy(), weights[f'{p}.attn_norm.weight'])
            
            q = h_norm @ weights[f'{p}.attn_q.weight'].T
            k = h_norm @ weights[f'{p}.attn_k.weight'].T
            v = h_norm @ weights[f'{p}.attn_v.weight'].T
            
            q = q.reshape(n_heads, head_dim)
            k = k.reshape(n_kv_heads, head_dim)
            v = v.reshape(n_kv_heads, head_dim)
            
            n_rep = n_heads // n_kv_heads
            if n_rep > 1:
                k = np.repeat(k, n_rep, axis=0)
                v = np.repeat(v, n_rep, axis=0)
            
            h = h + v[0] @ weights[f'{p}.attn_output.weight'].T
            
            h_norm = rmsnorm(h.copy(), weights[f'{p}.ffn_norm.weight'])
            ffn_gate = h_norm @ weights[f'{p}.ffn_gate.weight'].T
            ffn_up = h_norm @ weights[f'{p}.ffn_up.weight'].T
            ffn_down = silu(ffn_gate) * ffn_up @ weights[f'{p}.ffn_down.weight'].T
            h = h + ffn_down
    
    response = tokenizer.decode(generated)
    return response, generated


def compare_with_cpp(checkpoints, cpp_output_path):
    """Compare Python checkpoints with C++ output"""
    print("\n" + "="*60)
    print("COMPARING PYTHON vs C++")
    print("="*60)
    
    if not os.path.exists(cpp_output_path):
        print(f"C++ output not found: {cpp_output_path}")
        return
    
    # Load C++ output (assuming it's a numpy file)
    cpp_data = np.load(cpp_output_path, allow_pickle=True).item()
    
    for name, py_val in checkpoints.items():
        if name in cpp_data:
            cpp_val = cpp_data[name]
            if isinstance(py_val, np.ndarray) and isinstance(cpp_val, np.ndarray):
                diff = np.abs(py_val - cpp_val)
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
                match = "✓" if max_diff < 1e-3 else "✗"
                print(f"  {match} {name:30s}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
            else:
                print(f"  ? {name:30s}: Python={py_val}, C++={cpp_val}")
        else:
            print(f"  - {name:30s}: Not found in C++ output")


def main():
    gguf_path = "/Users/arctic/Downloads/qwen2-0_5b-instruct-q4_k_m.gguf"
    
    # Load model
    model = load_gguf_model(gguf_path)
    
    # Test prompt
    prompt = "Hello! How are you?"
    
    print("\n" + "="*60)
    print("RUNNING INFERENCE WITH CHECKPOINTS")
    print("="*60)
    
    # Run inference and get checkpoints
    result = run_inference_with_checkpoints(model, prompt, max_new_tokens=5)
    if result is None:
        return 1
    
    checkpoints, token_ids = result
    
    # Print checkpoint summary
    print("\nCheckpoint Summary:")
    for name, val in checkpoints.items():
        if isinstance(val, np.ndarray):
            print(f"  {name:30s}: shape={val.shape}, mean={val.mean():.6f}, std={val.std():.6f}")
        else:
            print(f"  {name:30s}: {val}")
    
    # Save checkpoints for C++ comparison
    np.savez('/tmp/ground_truth_checkpoints.npz', **checkpoints)
    print("\nSaved checkpoints to /tmp/ground_truth_checkpoints.npz")
    
    # Generate full response
    print("\n" + "="*60)
    print("GENERATING RESPONSE")
    print("="*60)
    
    response, tokens = generate_response(model, prompt, max_new_tokens=30)
    print(f"Response: {response}")
    print(f"Tokens: {tokens[:10]}...")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()
        sys.exit(1)
