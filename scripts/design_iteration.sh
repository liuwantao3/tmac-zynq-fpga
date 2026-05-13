#!/bin/bash
# FPGA Design Iteration Script
# Usage: ./scripts/design_iteration.sh [hls|vivado|all]
# Workflow: Run -> Capture Layer1 (console) -> Layer2 (reports) -> Structured feedback

set -e

WORKSPACE="/Users/arctic/fpga"
LOG_DIR="${WORKSPACE}/logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/iteration_${TIMESTAMP}.log"

HLS_DIR="${WORKSPACE}/hls"
VIVADO_DIR="${WORKSPACE}/vivado"
VITIS_HLS="/opt/Xilinx/Vitis_HLS/2023.1/bin/vitis_hls"
VIVADO="/opt/Xilinx/Vivado/2023.1/bin/vivado"

report_status() {
    echo "[$(date +%H:%M:%S)] $1"
    echo "[$(date +%H:%M:%S)] $1" >> "${LOG_FILE}"
}

parse_feedback() {
    python3 "${WORKSPACE}/scripts/feedback_parser.py" "${LOG_DIR}" >> "${LOG_FILE}" 2>&1
}

run_hls() {
    report_status "=== Starting HLS Synthesis ==="

    if [ ! -f "${HLS_DIR}/script.tcl" ]; then
        echo "ERROR: HLS script.tcl not found in ${HLS_DIR}"
        exit 1
    fi

    cd "${HLS_DIR}"
    "${VITIS_HLS}" -s script.tcl -l hls_${TIMESTAMP}.log

    report_status "HLS Synthesis completed"

    local latency=$(grep -i "Latency" "${HLS_DIR}/solution/solution.xml" 2>/dev/null | head -1 || echo "N/A")
    local dsp=$(grep -i "DSP" "${HLS_DIR}/solution/solution.xml" 2>/dev/null | head -1 || echo "N/A")

    report_status "HLS Latency: ${latency}"
    report_status "HLS DSP usage: ${dsp}"
}

run_vivado() {
    report_status "=== Starting Vivado Implementation ==="

    if [ ! -f "${VIVADO_DIR}/block_design.tcl" ]; then
        echo "ERROR: block_design.tcl not found in ${VIVADO_DIR}"
        exit 1
    fi

    cd "${VIVADO_DIR}"
    "${VIVADO}" -mode batch -source block_design.tcl -log vivado_${TIMESTAMP}.log

    report_status "Vivado Implementation completed"

    local util=$(grep -i "CLB LUT" "${VIVADO_DIR}/design_2_utilization_routed.rpt" 2>/dev/null || echo "N/A")

    report_status "Vivado Utilization: ${util}"
}

main() {
    local cmd="${1:-all}"

    report_status "Starting FPGA design iteration - mode: ${cmd}"

    case "${cmd}" in
        hls)
            run_hls
            parse_feedback
            ;;
        vivado)
            run_vivado
            parse_feedback
            ;;
        all)
            run_hls
            run_vivado
            parse_feedback
            ;;
        *)
            echo "Usage: $0 [hls|vivado|all]"
            exit 1
            ;;
    esac

    report_status "=== Iteration Complete ==="
    echo ""
    echo "Log saved to: ${LOG_FILE}"
    echo "Run feedback_parser.py for detailed analysis"
}

main "$@"