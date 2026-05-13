#pragma once

#include <cstdint>
#include <cstdio>
#include <chrono>
#include <vector>
#include <map>
#include <string>
#include <algorithm>

namespace fpga_sim {

// ===========================================================================
// Inference Pipeline Profiler
//
// Instrumentation-based profiler that tracks where CPU time is spent across
// the inference pipeline stages, per-layer, and per-tile. Identifies the
// bottleneck at each level of granularity.
//
// Stages tracked:
//   quantize    — float → int8/int16 conversion
//   ddr_copy    — memcpy to/from simulated DDR
//   axi_setup   — AXI-Lite register writes for control/status
//   compute     — FPGA compute (simulated systolic array)
//   dequantize  — int32/int64 → float + scale multiplication
//   overhead    — loop control, branches, other
// ===========================================================================
class Profiler {
public:
    enum Stage {
        STAGE_QUANTIZE   = 0,
        STAGE_DDR_COPY   = 1,
        STAGE_AXI_SETUP  = 2,
        STAGE_COMPUTE    = 3,
        STAGE_DEQUANTIZE = 4,
        STAGE_OVERHEAD   = 5,
        STAGE_COUNT
    };

    static const char* stage_name(Stage s) {
        static const char* names[] = {
            "quantize", "ddr_copy", "axi_setup", "compute",
            "dequantize", "overhead"
        };
        return (s < STAGE_COUNT) ? names[s] : "unknown";
    }

    Profiler() {
        layers_.emplace_back(); // layer 0 sentinel
    }

    static auto tnow() { return std::chrono::high_resolution_clock::now(); }
    static int64_t telapsed(const std::chrono::high_resolution_clock::time_point& t) {
        return std::chrono::duration<int64_t, std::nano>(tnow() - t).count();
    }

    // -- Start/stop a stage timer -------------------------------------------
    void begin(Stage stage, const char* label = "") {
        if (depth_ < MAX_DEPTH) {
            stack_[depth_].stage = stage;
            stack_[depth_].label = label;
            stack_[depth_].start = tnow();
            stack_[depth_].active = true;
            depth_++;
        }
    }

    void end() {
        if (depth_ == 0) return;
        depth_--;
        if (!stack_[depth_].active) return;

        auto ns = telapsed(stack_[depth_].start);
        Stage s = stack_[depth_].stage;

        // Per-stage accumulation
        stage_ns_[s] += ns;
        stage_count_[s]++;

        // Per-tile if we're tracking
        if (current_tile_row_ >= 0) {
            tile_times_.push_back({s, ns, current_tile_row_, current_tile_col_,
                                   current_dim_m_, current_dim_n_});
        }
    }

    // -- RAII scope guard ----------------------------------------------------
    struct Scope {
        Profiler& p;
        Scope(Profiler& p, Stage s, const char* label = "") : p(p) {
            p.begin(s, label);
        }
        ~Scope() { p.end(); }
    };

    // -- Tile tracking -------------------------------------------------------
    void begin_tile(int row, int col, int dimM, int dimN) {
        current_tile_row_ = row;
        current_tile_col_ = col;
        current_dim_m_ = dimM;
        current_dim_n_ = dimN;
    }

    void end_tile() {
        current_tile_row_ = -1;
    }

    // -- Layer tracking ------------------------------------------------------
    void begin_layer(int layer) {
        if ((size_t)layer >= layers_.size())
            layers_.resize(layer + 1);
        active_layer_ = layer;
        layer_start_ = tnow();
    }

    void end_layer() {
        if (active_layer_ >= 0) {
            layers_[active_layer_].ns += telapsed(layer_start_);
            layers_[active_layer_].count++;
            active_layer_ = -1;
        }
    }

    // -- Output --------------------------------------------------------------
    void report() const {
        printf("\n[PROFILER — PIPELINE BOTTLENECK ANALYSIS]\n");

        // 1. Stage breakdown
        int64_t total_ns = 0;
        for (int i = 0; i < STAGE_COUNT; i++)
            total_ns += stage_ns_[i];

        if (total_ns == 0) {
            printf("  (no data — profiling not enabled)\n");
            return;
        }

        printf("  %-14s %12s %8s %12s\n", "Stage", "Time (ms)", "Calls", "Avg (us)");
        printf("  %s\n", std::string(50, '-').c_str());

        // Find bottleneck
        int64_t max_ns = 0;
        Stage bottleneck = STAGE_OVERHEAD;

        for (int i = 0; i < STAGE_COUNT; i++) {
            double ms = stage_ns_[i] / 1e6;
            double avg_us = stage_count_[i] ? (stage_ns_[i] / 1e3 / (double)stage_count_[i]) : 0.0;
            const char* marker = "";
            if (stage_ns_[i] > max_ns && stage_ns_[i] > 0) {
                max_ns = stage_ns_[i];
                bottleneck = (Stage)i;
                marker = " <-- BOTTLENECK";
            }
            printf("  %-14s %10.3f %8llu %10.1f%s\n",
                   stage_name((Stage)i), ms,
                   (unsigned long long)stage_count_[i], avg_us, marker);
        }
        printf("  %s\n", std::string(50, '-').c_str());
        printf("  %-14s %10.3f %8llu\n", "TOTAL",
               total_ns / 1e6, (unsigned long long)stage_count_[STAGE_QUANTIZE]
               + stage_count_[STAGE_DDR_COPY] + stage_count_[STAGE_COMPUTE]);

        printf("\n  >> Bottleneck: %s (%.1f%% of total time)\n",
               stage_name(bottleneck),
               max_ns * 100.0 / total_ns);

        // 2. Layer breakdown (top 5 hottest layers)
        if (layers_.size() > 1) {
            printf("\n  %s\n", std::string(50, '-').c_str());
            printf("  %-8s %12s %8s\n", "Layer", "Time (ms)", "Calls");
            printf("  %s\n", std::string(50, '-').c_str());

            // Sort layers by time
            std::vector<size_t> idx(layers_.size() - 1);
            for (size_t i = 0; i < idx.size(); i++) idx[i] = i + 1;
            std::sort(idx.begin(), idx.end(), [this](size_t a, size_t b) {
                return layers_[a].ns > layers_[b].ns;
            });

            int n_show = std::min((size_t)5, idx.size());
            for (int i = 0; i < n_show; i++) {
                size_t li = idx[i];
                printf("  Layer %-3zu %10.3f %8llu\n",
                       li, layers_[li].ns / 1e6,
                       (unsigned long long)layers_[li].count);
            }

            // Show max/min layers
            if (idx.size() > 1) {
                printf("  %s\n", std::string(50, '-').c_str());
                printf("  Hottest:  Layer %zu (%.1f ms)\n",
                       idx[0], layers_[idx[0]].ns / 1e6);
                printf("  Coolest:  Layer %zu (%.1f ms)\n",
                       idx.back(), layers_[idx.back()].ns / 1e6);
                double avg = 0;
                for (size_t i = 1; i < layers_.size(); i++)
                    avg += layers_[i].ns;
                avg /= (layers_.size() - 1);
                printf("  Average:  %.1f ms\n", avg / 1e6);
            }
        }

        // 3. Recommendations
        printf("\n  >> Recommendations:\n");
        if (bottleneck == STAGE_QUANTIZE)
            printf("     Quantization is the bottleneck. Consider lookup tables\n");
        else if (bottleneck == STAGE_DDR_COPY)
            printf("     DDR copy dominates. Check tile size vs bus width.\n");
        else if (bottleneck == STAGE_AXI_SETUP)
            printf("     AXI register overhead! Consider reducing tiles or\n"
                   "     batching multiple operations per START.\n");
        else if (bottleneck == STAGE_COMPUTE)
            printf("     Compute bound — expected for FPGA simulation.\n");
        else if (bottleneck == STAGE_DEQUANTIZE)
            printf("     Dequantization bottleneck. Consider SIMD or LUT.\n");
        printf("\n");
    }

    // -- CSV export for further analysis ------------------------------------
    void dump_csv(const char* path) const {
        FILE* f = fopen(path, "w");
        if (!f) return;
        fprintf(f, "stage,time_ns,row,col,dimM,dimN\n");
        for (auto& t : tile_times_) {
            fprintf(f, "%s,%lld,%d,%d,%d,%d\n",
                    stage_name(t.stage),
                    (long long)t.ns,
                    t.row, t.col, t.dimM, t.dimN);
        }
        fclose(f);
        printf("[CSV] Wrote %s\n", path);
    }

private:
    struct Timer {
        Stage stage;
        const char* label;
        std::chrono::high_resolution_clock::time_point start;
        bool active = false;
    };

    struct TileEntry {
        Stage stage;
        int64_t ns;
        int row, col, dimM, dimN;
    };

    struct LayerStats {
        int64_t ns = 0;
        uint64_t count = 0;
    };

    static constexpr int MAX_DEPTH = 16;

    Timer stack_[MAX_DEPTH];
    int depth_ = 0;

    int64_t stage_ns_[STAGE_COUNT] = {0};
    uint64_t stage_count_[STAGE_COUNT] = {0};

    int active_layer_ = -1;
    std::chrono::high_resolution_clock::time_point layer_start_;
    std::vector<LayerStats> layers_;

    int current_tile_row_ = -1;
    int current_tile_col_ = -1;
    int current_dim_m_ = 0;
    int current_dim_n_ = 0;

    std::vector<TileEntry> tile_times_;
};

// Global profiler instance (optional — created on demand)
inline Profiler& prof() {
    static Profiler instance;
    return instance;
}

} // namespace fpga_sim
