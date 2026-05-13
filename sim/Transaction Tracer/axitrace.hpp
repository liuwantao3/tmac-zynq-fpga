#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>

namespace fpga_sim {

// ===========================================================================
// AXI-Lite transaction entry
// ===========================================================================
struct AxiEntry {
    uint64_t cycle;
    uint8_t  type;   // 0=write, 1=read, 2=state, 3=interrupt_on,
                     // 4=interrupt_off, 5=compute_start, 6=compute_end
    uint8_t  mode;   // 0=INT8, 1=INT16 (for compute entries)
    uint32_t addr;
    uint32_t data;
    union {
        uint32_t state_val;
        uint32_t dim_m;
    };
    uint32_t dim_n;
    uint32_t dim_k;
};

// ===========================================================================
// AXI-Lite Signal-Level Transaction Tracer
//
// Models cycle-accurate AXI-Lite handshake timing and logs every signal
// transition to a VCD waveform (viewable in GTKWave) plus a CSV summary.
//
// Timing model (Zynq PS→PL AXI-Lite GP port, approximate):
//   Register write: 5 cycles  (AW+W handshake → B response)
//   Register read:  5 cycles  (AR handshake → R response)
//   Compute/tile:   64 cycles (1×64×64 systolic at 150 MHz)
// ===========================================================================
class AxiTrace {
public:
    AxiTrace() {
        wall_start_ = std::chrono::steady_clock::now();
    }

    // -- Configuration -------------------------------------------------------
    void set_write_latency(uint32_t cyc) { write_latency_ = cyc; }
    void set_read_latency(uint32_t cyc)  { read_latency_  = cyc; }
    void set_compute_cycles(uint32_t cyc){ compute_cycles_ = cyc; }
    void set_max_entries(uint32_t max)   { max_entries_ = max; }

    // -- Cycle counter -------------------------------------------------------
    uint64_t cycle() const { return cycle_; }
    void advance(uint64_t n = 1) { cycle_ += n; }

    // -- AXI-Lite transaction logging (with automatic cycle advance) --------
    void log_write(uint32_t addr, uint32_t data) {
        if (entries_.size() >= max_entries_) return;

        // Model write handshake: AW→W→B = write_latency_ cycles
        entries_.push_back({cycle_, 0, 0, addr, data, {0}, 0, 0});

        // Record signal transitions for VCD
        vcd_signals_.push_back({cycle_, "awaddr", addr, 32});
        vcd_signals_.push_back({cycle_, "awvalid", 1, 1});
        vcd_signals_.push_back({cycle_, "wdata", data, 32});
        vcd_signals_.push_back({cycle_, "wvalid", 1, 1});

        advance(1);
        vcd_signals_.push_back({cycle_, "awready", 1, 1});
        vcd_signals_.push_back({cycle_, "wready", 1, 1});
        vcd_signals_.push_back({cycle_, "awvalid", 0, 1});
        vcd_signals_.push_back({cycle_, "wvalid", 0, 1});

        advance(1);
        vcd_signals_.push_back({cycle_, "awready", 0, 1});
        vcd_signals_.push_back({cycle_, "wready", 0, 1});

        // B response channel
        vcd_signals_.push_back({cycle_, "bvalid", 1, 1});
        vcd_signals_.push_back({cycle_, "bresp", 0, 2});

        advance(1);
        vcd_signals_.push_back({cycle_, "bvalid", 0, 1});
        vcd_signals_.push_back({cycle_, "bresp", 0, 2});

        // Fill remaining write_latency_ cycles with idle
        for (uint32_t i = 3; i < write_latency_; i++)
            advance(1);
    }

    void log_read(uint32_t addr, uint32_t result) {
        if (entries_.size() >= max_entries_) return;
        entries_.push_back({cycle_, 1, 0, addr, result, {0}, 0, 0});

        vcd_signals_.push_back({cycle_, "araddr", addr, 32});
        vcd_signals_.push_back({cycle_, "arvalid", 1, 1});

        advance(1);
        vcd_signals_.push_back({cycle_, "arready", 1, 1});
        vcd_signals_.push_back({cycle_, "arvalid", 0, 1});

        advance(1);
        vcd_signals_.push_back({cycle_, "arready", 0, 1});
        vcd_signals_.push_back({cycle_, "rdata", result, 32});
        vcd_signals_.push_back({cycle_, "rvalid", 1, 1});
        vcd_signals_.push_back({cycle_, "rresp", 0, 2});

        advance(1);
        vcd_signals_.push_back({cycle_, "rvalid", 0, 1});

        for (uint32_t i = 3; i < read_latency_; i++)
            advance(1);
    }

    // -- Internal event logging ----------------------------------------------
    void log_state(uint32_t state) {
        entries_.push_back({cycle_, 2, 0, 0, 0, {state}, 0, 0});
        vcd_signals_.push_back({cycle_, "state", state, 8});

        vcd_signals_.push_back({cycle_, "ap_idle",
            (state == 0 || state == 2) ? 1u : 0u, 1});
        vcd_signals_.push_back({cycle_, "ap_start",
            (state == 1) ? 1u : 0u, 1});
        vcd_signals_.push_back({cycle_, "ap_done",
            (state == 2) ? 1u : 0u, 1});
    }

    void log_interrupt(bool asserted) {
        if (asserted)
            entries_.push_back({cycle_, 3, 0, 0, 0, {0}, 0, 0});
        else
            entries_.push_back({cycle_, 4, 0, 0, 0, {0}, 0, 0});
        vcd_signals_.push_back({cycle_, "interrupt", asserted ? 1u : 0u, 1});
    }

    void log_compute_start(uint32_t dimM, uint32_t dimN, uint32_t dimK, bool int16) {
        entries_.push_back({cycle_, 5, (uint8_t)(int16 ? 1u : 0u), 0, 0, {dimM}, dimN, dimK});
        vcd_signals_.push_back({cycle_, "ap_start", 1, 1});
        vcd_signals_.push_back({cycle_, "ap_idle", 0, 1});

        // Compute takes compute_cycles_ (systolic array)
        advance(compute_cycles_);
    }

    void log_compute_end() {
        entries_.push_back({cycle_, 6, 0, 0, 0, {0}, 0, 0});
        vcd_signals_.push_back({cycle_, "ap_done", 1, 1});
        vcd_signals_.push_back({cycle_, "ap_start", 0, 1});
        vcd_signals_.push_back({cycle_, "ap_idle", 1, 1});
    }

    // -- Summary statistics --------------------------------------------------
    void print_summary() const {
        uint64_t n_write = 0, n_read = 0, n_compute = 0;
        uint64_t compute_cycles_total = 0;

        for (auto& e : entries_) {
            if (e.type == 0) n_write++;
            if (e.type == 1) n_read++;
            if (e.type == 5) compute_cycles_total += compute_cycles_;
            if (e.type == 6) n_compute++;
        }

        double wall_ns = std::chrono::duration<double, std::nano>(
            std::chrono::steady_clock::now() - wall_start_).count();

        printf("\n[AXI TRACE SUMMARY]\n");
        printf("  Total cycles:  %-12llu  (%.3f ms at 150 MHz)\n",
               (unsigned long long)cycle_, cycle_ * 6.667 / 1e6);
        printf("  Wall time:     %-12.3f ms\n", wall_ns / 1e6);
        printf("  AXI writes:    %-12llu  AXI reads:     %llu\n",
               (unsigned long long)n_write, (unsigned long long)n_read);
        printf("  Compute tiles: %-12llu  (avg %.0f cycles/tile)\n",
               (unsigned long long)n_compute,
               n_compute ? (double)compute_cycles_total / n_compute : 0.0);
        printf("  Trace entries: %zu\n", entries_.size());

        if (n_write + n_read > 0) {
            double axi_pct = (double)(n_write + n_read) * write_latency_ * 100.0
                           / cycle_;
            double comp_pct = (double)compute_cycles_total * 100.0 / cycle_;
            printf("  AXI overhead:  %-12.1f%%  Compute:       %.1f%%\n",
                   axi_pct, comp_pct);
        }
    }

    static void append_bits(char* buf, int bits, uint32_t val) {
        for (int i = bits - 1; i >= 0; i--)
            *buf++ = (val >> i) & 1 ? '1' : '0';
        *buf = '\0';
    }

    void dump_vcd(const char* path) const {
        FILE* f = fopen(path, "w");
        if (!f) { perror("dump_vcd"); return; }

        auto now = std::chrono::system_clock::to_time_t(
            std::chrono::system_clock::now());
        struct tm tmbuf;
        localtime_r(&now, &tmbuf);
        char datebuf[64];
        strftime(datebuf, sizeof(datebuf), "%b %d %Y %H:%M:%S", &tmbuf);

        fprintf(f, "$date\n    %s\n$end\n", datebuf);
        fprintf(f, "$version\n    FPGA AXI-Lite Transaction Tracer v1.0\n$end\n");
        fprintf(f, "$timescale\n    1 ns\n$end\n");
        fprintf(f, "$scope module fpga_sim $end\n");

        fprintf(f, "$var wire  1 ! aclk         $end\n");
        fprintf(f, "$var wire 32 \" awaddr [31:0] $end\n");
        fprintf(f, "$var wire  1 # awvalid       $end\n");
        fprintf(f, "$var wire  1 $ awready       $end\n");
        fprintf(f, "$var wire 32 %% wdata  [31:0] $end\n");
        fprintf(f, "$var wire  1 & wvalid        $end\n");
        fprintf(f, "$var wire  1 ' wready        $end\n");
        fprintf(f, "$var wire  2 ( bresp  [1:0]  $end\n");
        fprintf(f, "$var wire  1 ) bvalid        $end\n");
        fprintf(f, "$var wire  1 * bready        $end\n");
        fprintf(f, "$var wire 32 + araddr [31:0] $end\n");
        fprintf(f, "$var wire  1 , arvalid       $end\n");
        fprintf(f, "$var wire  1 - arready       $end\n");
        fprintf(f, "$var wire 32 . rdata  [31:0] $end\n");
        fprintf(f, "$var wire  1 / rvalid        $end\n");
        fprintf(f, "$var wire  1 0 rready        $end\n");
        fprintf(f, "$var wire  2 1 rresp  [1:0]  $end\n");
        fprintf(f, "$var wire  1 2 ap_start      $end\n");
        fprintf(f, "$var wire  1 3 ap_done       $end\n");
        fprintf(f, "$var wire  1 4 ap_idle       $end\n");
        fprintf(f, "$var wire  1 5 ap_ready      $end\n");
        fprintf(f, "$var wire  1 6 interrupt     $end\n");
        fprintf(f, "$var wire  8 7 state   [7:0] $end\n");
        fprintf(f, "$var wire 32 8 mode    [31:0]$end\n");
        fprintf(f, "$upscope $end\n");
        fprintf(f, "$enddefinitions $end\n");

        char bits_buf[128];
        auto vcd_line = [&](uint32_t val, int bits, const char* id) {
            if (bits == 1) {
                fprintf(f, "b%u %s\n", val, id);
            } else {
                append_bits(bits_buf, bits, val);
                fprintf(f, "b%s %s\n", bits_buf, id);
            }
        };

        fprintf(f, "#0\n");
        vcd_line(0, 1, "!");                     // aclk = 0
        vcd_line(0, 32, "\""); vcd_line(0, 1, "#"); vcd_line(0, 1, "$");
        vcd_line(0, 32, "%%"); vcd_line(0, 1, "&"); vcd_line(0, 1, "'");
        vcd_line(0, 2, "(");  vcd_line(0, 1, ")"); vcd_line(1, 1, "*");
        vcd_line(0, 32, "+"); vcd_line(0, 1, ","); vcd_line(1, 1, "-");
        vcd_line(0, 32, "."); vcd_line(0, 1, "/"); vcd_line(1, 1, "0");
        vcd_line(0, 2, "1");  vcd_line(0, 1, "2"); vcd_line(0, 1, "3");
        vcd_line(1, 1, "4");  vcd_line(1, 1, "5"); vcd_line(0, 1, "6");
        vcd_line(0, 8, "7");  vcd_line(0, 32, "8");

        uint64_t last_cycle = 0;
        for (auto& s : vcd_signals_) {
            for (uint64_t c = last_cycle; c < s.cycle; c++) {
                uint64_t t_ns = c * 7;
                fprintf(f, "#%llu\n", (unsigned long long)t_ns);
                fprintf(f, "b0 !\n");
                fprintf(f, "#%llu\n", (unsigned long long)(t_ns + 3));
                fprintf(f, "b1 !\n");
            }
            last_cycle = s.cycle;

            uint64_t t_ns = s.cycle * 7 + 3;
            fprintf(f, "#%llu\n", (unsigned long long)t_ns);
            vcd_line(s.value, s.bits, s.signal_id.c_str());
        }

        fclose(f);
        printf("[VCD] Wrote %s\n", path);
    }

    // -- CSV transaction dump -----------------------------------------------
    void dump_csv(const char* path) const {
        FILE* f = fopen(path, "w");
        if (!f) { perror("dump_csv"); return; }

        fprintf(f, "cycle,type,addr,data,metadata\n");
        for (auto& e : entries_) {
            const char* tstr = "???";
            switch (e.type) {
                case 0: tstr = "WRITE"; break;
                case 1: tstr = "READ"; break;
                case 2: tstr = "STATE"; break;
                case 3: tstr = "IRQ_ON"; break;
                case 4: tstr = "IRQ_OFF"; break;
                case 5: tstr = "COMP_START"; break;
                case 6: tstr = "COMP_END"; break;
            }
            fprintf(f, "%llu,%s,0x%X,0x%X,0x%X\n",
                    (unsigned long long)e.cycle, tstr, e.addr, e.data,
                    e.state_val);
        }
        fclose(f);
        printf("[CSV] Wrote %s\n", path);
    }

    const std::vector<AxiEntry>& entries() const { return entries_; }

private:
    struct VcdSignal {
        uint64_t cycle;
        std::string signal_id;
        uint32_t value;
        int bits;
    };

    uint64_t cycle_ = 0;
    uint32_t write_latency_ = 5;
    uint32_t read_latency_  = 5;
    uint32_t compute_cycles_ = 64;
    uint32_t max_entries_ = 1000000;

    std::vector<AxiEntry> entries_;
    std::vector<VcdSignal> vcd_signals_;
    std::chrono::steady_clock::time_point wall_start_;
};

} // namespace fpga_sim
