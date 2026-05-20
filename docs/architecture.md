# Qwen2-0.5B TMAC Inference — Architecture & Knowledge

## 1. Model Architecture (Qwen2-0.5B)

| Parameter | Value |
|-----------|-------|
| Hidden dim | 896 |
| Intermediate dim (FFN) | 4864 |
| Vocab size | 151936 |
| Num layers | 24 |
| Num attention heads | 14 |
| Head dim | 64 |
| Num KV heads | 2 (GQA) |
| RMS norm eps | 1e-6 |
| RoPE theta | 1000000.0 |
| Tie word embeddings | true (lm_head = token_embd.weight) |

### Layer composition

Each transformer layer (blk.N):
1. **RMS Norm** → attn_norm.weight (F32)
2. **QKV projections** → attn_q/k/v.weight + bias
   - Q: [896×896] Q5_0
   - K: [128×896] Q5_0
   - V: [128×896] Q8_0
3. **RoPE** applied to Q (14 heads) and K (2 heads)
4. **GQA attention**: 14 Q heads, 2 KV heads, 7 queries per KV
5. **Output projection** → attn_output.weight [896×896] Q5_0
6. **Residual** + →
7. **RMS Norm** → ffn_norm.weight (F32)
8. **SwiGLU FFN**:
   - gate_proj [4864×896] Q5_0
   - up_proj [4864×896] Q5_0
   - down_proj [896×4864] Q6_K (some layers) or Q5_0
9. **Residual** + → output

### Weight tensor count breakdown (290 total)
- 12× Q6_K: selected ffn_down layers (0,1,3,6,7,8,9,10,13,16,19,21)
- 132× Q5_0: most weight tensors
- Q8_0: token_embd.weight, attn_v.weight
- F32: norm weights, biases, output_norm.weight

---

## 2. GGUF / GGML Quantization Formats

### Column-major storage (CRITICAL)

GGUF stores 2D tensors column-major (FORTRAN order):
- `ne[0]` = input_dim (fastest varying in storage)
- `ne[1]` = output_dim (slowest varying)
- Element at logical `W[output][input]` is at flat index: `input + output * input_dim`

This means **row-major access** (`idx = row * cols + col`) is **wrong** for non-square matrices. The correct access is:
```
idx = input + output * input_dim    // = col + row * ne[0]
```

For square matrices this is irrelevant because `i*N + j == j + i*N` is false—wait, actually `i*N + j == j + i*N` only when `i*N + j = j + i*N`, which means `i*N = i*N` → always true. So row-major and column-major give the same flat index for square matrices. This is why Q (896×896) worked even with wrong indexing.

### Q5_0 (block size: 32 elements, 22 bytes)

```
[d: f16][qh: u32][qs: u8×16]
 0      1-2     3-6     7-22
```

- `d` = f16 scale factor
- `qh` = 32-bit mask (high bit for each element)
- `qs` = 16 bytes = 32 nibbles (low 4 bits of each element)

**Element layout:**
- j=0..15: `qs[j]` lower nibble, `qh` bit j
- j=16..31: `qs[j-16]` upper nibble, `qh` bit j+16

**Dequant:** `q = ((qh_bit << 4) | ql_nibble) - 16; val = d * q`

**BUG FIXED:** qh shift for elements 16-31 was `offset-4` (extracting bit j+12), but should be `offset` (extracting bit j+16). The ggml source `(qh >> (j+12)) & 0x10` extracts bit j+16 via the `& 0x10` mask.

### Q6_K (block size: 256 elements, 210 bytes)

```
[ql: u8×128][qh: u8×64][scales: i8×16][d: f16]
 0         127  128     191  192       207  208-209
```

Split into two 128-element halves. Each half:
- ql[64]: 32 lower nibbles + 32 upper nibbles (4-bit each)
- qh[32]: 2-bit high parts (4 elements per byte, 2 bits each)
- scales[8]: 8 int8 scales

For element within half (pos=0..127):
- l = pos % 32 (group of 4), sub = pos / 32 (0..3)
- q1 (sub=0): ql[l] lower nibble, qh[l] bits[1:0]
- q2 (sub=1): ql[l+32] lower nibble, qh[l] bits[3:2]
- q3 (sub=2): ql[l] upper nibble, qh[l] bits[5:4]
- q4 (sub=3): ql[l+32] upper nibble, qh[l] bits[7:6]
- scale_idx = l/16 + sub*2 (scales[0..7])

**Dequant:** `q = ((qh_bits << 4) | ql_nibble) - 32; val = d * scales[scale_idx] * q`

### Q4_K (block size: 256 elements, 144 bytes)

[scale factors (f16), min (f16), sub-block scales (u8×8), qs (u8×128)]

### Q8_0 (block size: 32 elements, 34 bytes)

```
[d: f16][q: i8×32]
 0-1     2-33
```

Straightforward: `val = d * q`

---

## 3. Inference Architecture (C++)

### Forward pass (`tmac_gguf.cpp`)

```
embedding → [24× transformer layers] → output_norm → lm_head → logits
```

### Per-layer computation

1. **RMS Norm**: `out = x * w / sqrt(mean(x²) + 1e-6)`
2. **QKV matmuls**: `y[i] = sum_j W[j + i*rows] * x[j]` (column-major access)
3. **RoPE**: standard rotary embeddings with theta=1e6
4. **GQA Attention** (with KV cache):
   - Single token: `context = V` (no softmax needed, single element)
   - Multi-token: softmax over Q·K scores, weighted sum of V
5. **Output projection**: matmul
6. **Residual connection**: `h = h + attn_out`
7. **FFN**: RMS norm → SwiGLU(gate, up) → down projection → residual
8. **SiLU**: `x / (1 + exp(-x))`

### KV Cache
- K cache: `[24×256×128]` (layers × seq_len × K_dim)
- V cache: `[24×256×128]`
- Fixed max length of 256 tokens

### Logit computation
```
h = output_norm(h)                      // RMS norm
logits[v] = sum_f emb_lookup[v][f] * h[f]   // lm_head = embedding transposed
```

Top-k sampling with temperature (default top_k=40).

### Memory budget
- DDR: 512 MB
- Model: 373.7 MB (290 tensors)
- Remaining: ~138 MB for KV cache, intermediates, code

---

## 4. Ground Truth Verification Pipeline

### Chain of trust
```
llama.cpp (patched main.cpp)
    ↓ max diff 0.0018
gguf Python library (gguf.quants.dequantize)
    ↓ max diff < 0.002
C++ TMAC inference (tmac_gguf.cpp)
```

### Getting ground truth from llama.cpp

Patch `llama.cpp/examples/main/main.cpp` to dump logits after prompt eval:

```c
{ static bool dumped = false; if (!dumped && n_past == 0 && n_eval > 0) {
    float* logits = llama_get_logits(ctx);
    fwrite(logits, sizeof(float), n_vocab, lf);
    dumped = true;
}}
```

### Getting ground truth from gguf Python library

```python
from gguf import GGUFReader, GGMLQuantizationType
from gguf.quants import dequantize

reader = GGUFReader(model_path)
weights = {}
for t in reader.tensors:
    weights[t.name] = dequantize(t.data, GGMLQuantizationType(t.tensor_type))
```

This gives the authoritative dequantized weights. The gguf library's `dequantize` is the reference implementation that exactly matches ggml's C dequant functions.

---

## 5. Debug Methodology

### Layer-by-layer comparison

1. Dump hidden state after each layer from C++: `/tmp/cpp_layer_{0..24}.bin`
2. Compute ground truth with gguf library: match computation exactly
3. Compare: `np.abs(cpp - gt).max()`

### Intermediate dump points (layer 0)
```
/tmp/cpp_inter/attn_norm_out.bin  → after RMS norm
/tmp/cpp_inter/q_after_bias.bin   → after Q projection + bias
/tmp/cpp_inter/k_after_bias.bin   → after K projection + bias
/tmp/cpp_inter/v_after_bias.bin   → after V projection + bias
/tmp/cpp_inter/context.bin        → attention context (before output proj)
/tmp/cpp_inter/attn_out.bin       → after attention output projection
/tmp/cpp_inter/after_attn_res.bin → after attention residual
/tmp/cpp_inter/ffn_norm_out.bin   → after FFN RMS norm
/tmp/cpp_inter/gate.bin           → after gate projection
/tmp/cpp_inter/up.bin             → after up projection
/tmp/cpp_inter/gate_x_up.bin      → after SwiGLU gate
/tmp/cpp_inter/ffn_out.bin        → after FFN down projection
/tmp/cpp_inter/final.bin          → after FFN residual
```

### Weight-level comparison

Compare dequantized weights element-by-element:

```python
w_gguf = dequantize(t.data, GGMLQuantizationType(t.tensor_type))
# vs element-by-element C++ dequant
for idx in range(n_elements):
    w_cpp[idx] = dequant_q5_0_cpp(data, idx)
diff = np.abs(w_gguf - w_cpp)
```

---

## 6. Bugs Found & Fixed

| Bug | File | Impact | Fix |
|-----|------|--------|-----|
| RMS norm formula: `x*w/(sqrt(mean)+eps)` vs `x*w/sqrt(mean+eps)` | tmac_gguf.cpp | Wrong norm values | Moved eps inside sqrt |
| RMS norm eps: 1e-5 vs correct 1e-6 | tmac_gguf.cpp | Small numerical diff | Changed to 1e-6 |
| Q5_0 dequant qh shift for elements 16-31: `offset-4` instead of `offset` | tmac_gguf.cpp, py_tmac_vec.py | Wrong high-bit extraction for 16+ | Changed to `qh >> offset` |
| Q5_0 qs byte indexing: `offset//2` (pairing even/odd) instead of `offset-16` (pairing j, j+16) | py_tmac_vec.py | Wrong nibble assignment | Fixed to use ggml pairing |
| Q4_K dequant: ignored dmin/sub-block scales, wrong byte offsets | tmac_gguf.cpp, py_tmac_vec.py | Wrong Q4_K values | Rewrote to match ggml |
| Q6_K dequant: flat structure not matching ggml's two-half layout | tmac_gguf.cpp, py_tmac_vec.py | Every element wrong | Rewrote per ggml's 128-element half structure |
| `read_tensor` row-major indexing for column-major stored data | tmac_gguf.cpp | Wrong for non-square tensors | Changed to `col + row * rows` for 2D tensors |
| `matmul_transposed` used for non-square projections | tmac_gguf.cpp | Wrong K/V/FFN | Changed all to `matmul` with corrected read_tensor |

---

## 7. TMAC File Format

```
Header:
  [magic: 4B]
  [n_tensors: u64]

Per tensor:
  [name_len: u64]
  [name: name_len bytes]
  [rows: u64]           = GGUF ne[0]
  [cols: u64]           = GGUF ne[1]
  [type: u32]           = GGMLQuantizationType value
  [n_bytes: u64]
  [data: n_bytes bytes]
```

Type codes:
- 0 = F32, 1 = F16, 6 = Q5_0, 8 = Q8_0, 12 = Q4_K, 14 = Q6_K

---

## 8. Key Files

| File | Purpose |
|------|---------|
| `sim/tmac_gguf.cpp` | **C++ inference engine** (3 dispatch branches: Q4K→FFN, Q8→logits, else→INT16) |
| `sim/fpga_sim.hpp` | FPGA simulator: MatmulAccel, AXI-Lite buffer API, timing model |
| `sim/test_integration.cpp` | C++ integration test (address map / Q4K compute / 5-tile roundtrip — all pass) |
| `verilog/matmul_top.v` | **Dual-core Verilog top**: Q8_0 + Q4_K, AXI4-Lite, 8192-byte weight_buf |
| `verilog/matmul_q8_core.v` | Q8_0 compute core (3-stage FSM, LUT dequant, 1 BRAM) |
| `verilog/matmul_q4k_core.v` | Q4_K compute core (INT16×INT16 pipeline, 512×128-bit wmem, 2 BRAMs) |
| `scripts/extract_tmac.py` | Converts GGUF to TMAC format |
| `scripts/py_tmac_vec.py` | **Python reference** (matches C++ exactly) |
| `scripts/ground_truth_v2.py` | Ground truth with tokenizer/chat |
| `scripts/verify_layers_fast.py` | Layer-by-layer comparison tool |
| `sim/chat.py` | Chat interface |
| `scripts/feedback_parser.py` | FPGA HLS/Vivado feedback parser |
| `scripts/design_iteration.sh` | FPGA design iteration workflow |

### Build & Run
```bash
# Build inference engine
cd sim
g++ -std=c++17 -pthread -O2 -o tmac_gguf tmac_gguf.cpp matmul_q8.cpp
echo "9707" | ./tmac_gguf /tmp/model.tmac                # single token, FP32
echo "9707" | ./tmac_gguf /tmp/model.tmac --fpga-q4k     # Q4_K path (FFN layers)
echo "9707" | ./tmac_gguf /tmp/model.tmac --fpga-q8      # Q8_0 path (logits)
echo "9707" | ./tmac_gguf /tmp/model.tmac --generate 10  # generate tokens
echo "9707" | ./tmac_gguf /tmp/model.tmac --dump-layers  # + layer dumps

# Build and run integration test
g++ -std=c++14 -O2 -I. -I../gguf -I.. test_integration.cpp -lpthread -o /tmp/test_integration
/tmp/test_integration

# Chat pipeline:
python3 -c "
from tokenizers import Tokenizer; import subprocess
tok = Tokenizer.from_file('/tmp/qwen-tok/tokenizer.json')
prompt = '<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n'
inp = '\n'.join(str(t) for t in tok.encode(prompt).ids)
proc = subprocess.run(['./tmac_gguf', '/tmp/model.tmac', '--generate', '20'],
                     input=inp, capture_output=True, text=True, timeout=600)
gen = [int(l) for l in proc.stdout.strip().split('\n') if l and not l.startswith('[')]
print(tok.decode(gen))
"
```
