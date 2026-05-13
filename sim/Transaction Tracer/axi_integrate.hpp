#pragma once

// Integration header: bridges fpga_sim::MatmulAccel with AxiTrace.
//
// Usage:
//   #include "fpga_sim.hpp"
//   #include "Transaction Tracer/axitrace.hpp"
//   #include "Transaction Tracer/axi_integrate.hpp"
//
//   auto& accel = fpga_sim::accel();
//   AxiTraceScope trace(accel);              // starts tracing
//   // ... do inference ...
//   trace.summary("/tmp/trace.vcd");          // prints + dumps VCD

#include "axitrace.hpp"

namespace fpga_sim {

// Callback adapter: AxiTraceFn → AxiTrace::log_write/log_read
inline void axi_trace_callback(void* ctx, uint32_t cycle, int dir,
                                uint32_t addr, uint32_t data) {
    auto* trace = static_cast<AxiTrace*>(ctx);
    if (dir == 0) {
        trace->log_write(addr, data);
    } else {
        trace->log_read(addr, data);
    }
}

// RAII scope that traces all AXI-Lite transactions on a MatmulAccel
class AxiTraceScope {
public:
    AxiTraceScope(MatmulAccel& accel, uint32_t compute_cycles = 64)
        : accel_(accel) {
        trace_.set_compute_cycles(compute_cycles);
        accel_.set_tracer(axi_trace_callback, &trace_);
    }

    ~AxiTraceScope() {
        accel_.set_tracer(nullptr, nullptr);
    }

    AxiTrace& trace() { return trace_; }

    void summary(const char* vcd_path = nullptr, const char* csv_path = nullptr) {
        trace_.print_summary();
        if (vcd_path) trace_.dump_vcd(vcd_path);
        if (csv_path) trace_.dump_csv(csv_path);
    }

private:
    MatmulAccel& accel_;
    AxiTrace trace_;
};

} // namespace fpga_sim
