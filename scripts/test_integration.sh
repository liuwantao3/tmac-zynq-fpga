#!/bin/bash
# System integration test: exercises Q8_0 + Q4_K FPGA paths
# Usage: ./scripts/test_integration.sh [--model /path/to/model.tmac]
#
# Prerequisites:
#   - iverilog (Icarus Verilog)
#   - g++ with C++14 support
#   - A .tmac model file (default: /tmp/model.tmac)

set -euo pipefail

MODEL="${2:-/tmp/model.tmac}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SIM_DIR="$PROJ_DIR/sim"
VERILOG_DIR="$PROJ_DIR/verilog"
PASS=0
FAIL=0

green() { printf "\033[32m%s\033[0m\n" "$1"; }
red()   { printf "\033[31m%s\033[0m\n" "$1"; }

# ===========================================================================
# Phase 0: Check prerequisites
# ===========================================================================
echo "========================================"
echo "Phase 0: Prerequisites"
echo "========================================"

if ! command -v iverilog &>/dev/null; then
    red "  FAIL: iverilog not found (brew install icarus-verilog)"
    exit 1
fi
echo "  [OK] iverilog found"

if ! g++ -std=c++14 --version &>/dev/null; then
    red "  FAIL: g++ not found"
    exit 1
fi
echo "  [OK] g++ found"

if [ ! -f "$MODEL" ]; then
    if [ -f /Users/arctic/fpga/models/model.tmac ]; then
        MODEL=/Users/arctic/fpga/models/model.tmac
        echo "  [OK] Model: $MODEL"
    else
        echo "  [WARN] No model file found at $MODEL or /Users/arctic/fpga/models/model.tmac"
        echo "  [WARN] Skipping inference tests (phases 4-5)"
        MODEL=""
    fi
else
    echo "  [OK] Model: $MODEL"
fi

# ===========================================================================
# Phase 1: Build C++ integration test
# ===========================================================================
echo ""
echo "========================================"
echo "Phase 1: Build C++ integration test"
echo "========================================"

g++ -std=c++14 -O2 -I"$SIM_DIR" -o /tmp/test_integration "$SIM_DIR/test_integration.cpp" -lpthread
echo "  [OK] Built /tmp/test_integration"

# ===========================================================================
# Phase 2: Run C++ integration test
# ===========================================================================
echo ""
echo "========================================"
echo "Phase 2: C++ integration test"
echo "========================================"

if /tmp/test_integration; then
    green "  PASS: C++ integration test"
    PASS=$((PASS + 1))
else
    red "  FAIL: C++ integration test"
    FAIL=$((FAIL + 1))
fi

# ===========================================================================
# Phase 3: Verilog Q4K core unit test
# ===========================================================================
echo ""
echo "========================================"
echo "Phase 3: Verilog Q4K core testbench"
echo "========================================"

cd "$VERILOG_DIR"

iverilog -Wall -g2012 -o tb_matmul_q4k.vvp matmul_q4k_core.v tb_matmul_q4k.v
if vvp tb_matmul_q4k.vvp 2>&1 | grep -q "ALL 7 TESTS PASSED"; then
    green "  PASS: Q4K core testbench"
    PASS=$((PASS + 1))
else
    red "  FAIL: Q4K core testbench"
    vvp tb_matmul_q4k.vvp 2>&1 | tail -5
    FAIL=$((FAIL + 1))
fi

# ===========================================================================
# Phase 4: Verilog Q8_0 core unit test
# ===========================================================================
echo ""
echo "========================================"
echo "Phase 4: Verilog Q8_0 core testbench"
echo "========================================"

iverilog -Wall -g2012 -o tb_matmul_q8.vvp matmul_q8_core.v dequant_lut.v systolic_8x8.v tb_matmul_q8.v
if vvp tb_matmul_q8.vvp 2>&1 | grep -q "ALL 6 TESTS PASSED"; then
    green "  PASS: Q8_0 core testbench"
    PASS=$((PASS + 1))
else
    red "  FAIL: Q8_0 core testbench"
    vvp tb_matmul_q8.vvp 2>&1 | tail -5
    FAIL=$((FAIL + 1))
fi

# ===========================================================================
# Phase 5: Full inference — Q8_0 path
# ===========================================================================
if [ -n "$MODEL" ]; then
    echo ""
    echo "========================================"
    echo "Phase 5: Full inference — Q8_0 FPGA path"
    echo "========================================"

    cd "$SIM_DIR"
    printf "1 2 3 4 5" | ./tmac_gguf "$MODEL" --fpga-q8 2>&1 | grep -q "token 5" \
        && green "  PASS: Q8_0 inference produces token 5" \
        || { red "  FAIL: Q8_0 inference"; FAIL=$((FAIL + 1)); }

    # ===========================================================================
    # Phase 6: Full inference — Q4_K path
    # ===========================================================================
    echo ""
    echo "========================================"
    echo "Phase 6: Full inference — Q4_K FPGA path"
    echo "========================================"

    printf "1 2 3 4 5" | ./tmac_gguf "$MODEL" --fpga-q4k 2>&1 | grep -q "token 5" \
        && green "  PASS: Q4_K inference produces token 5" \
        || { red "  FAIL: Q4_K inference"; FAIL=$((FAIL + 1)); }

    # ===========================================================================
    # Phase 7: Output consistency check
    # ===========================================================================
    echo ""
    echo "========================================"
    echo "Phase 7: Output consistency — Q8_0 vs Q4_K vs CPU"
    echo "========================================"

    CPU_OUT=$(printf "1 2 3 4 5" | ./tmac_gguf "$MODEL" 2>&1 | grep "token 5" || true)
    Q8_OUT=$(printf "1 2 3 4 5" | ./tmac_gguf "$MODEL" --fpga-q8 2>&1 | grep "token 5" || true)
    Q4K_OUT=$(printf "1 2 3 4 5" | ./tmac_gguf "$MODEL" --fpga-q4k 2>&1 | grep "token 5" || true)

    if [ "$Q8_OUT" = "$CPU_OUT" ] && [ "$Q4K_OUT" = "$CPU_OUT" ]; then
        green "  PASS: All paths produce same token 5 ($CPU_OUT)"
        PASS=$((PASS + 3))
    else
        echo "  CPU:  $CPU_OUT"
        echo "  Q8:   $Q8_OUT"
        echo "  Q4K:  $Q4K_OUT"
        if [ "$Q8_OUT" = "$CPU_OUT" ]; then
            green "  PASS: Q8_0 matches CPU"
            PASS=$((PASS + 1))
        else
            red "  FAIL: Q8_0 differs from CPU"
            FAIL=$((FAIL + 1))
        fi
        if [ "$Q4K_OUT" = "$CPU_OUT" ]; then
            green "  PASS: Q4_K matches CPU"
            PASS=$((PASS + 1))
        else
            red "  FAIL: Q4_K differs from CPU"
            FAIL=$((FAIL + 1))
        fi
    fi
else
    echo ""
    echo "========================================"
    echo "Phases 5-7: SKIPPED (no model file)"
    echo "========================================"
fi

# ===========================================================================
# Phase 8: Cleanup
# ===========================================================================
cd "$VERILOG_DIR"
rm -f tb_matmul_q4k.vvp tb_matmul_q4k.vcd tb_matmul_q8.vvp tb_matmul_q8.vcd 2>/dev/null || true
rm -f /tmp/test_integration 2>/dev/null || true

# ===========================================================================
# Summary
# ===========================================================================
echo ""
echo "========================================"
echo "Integration Test Summary"
echo "========================================"
echo "  PASS: $PASS"
echo "  FAIL: $FAIL"
echo ""

if [ "$FAIL" -eq 0 ]; then
    green "ALL TESTS PASSED"
    exit 0
else
    red "$FAIL TEST(S) FAILED"
    exit 1
fi
