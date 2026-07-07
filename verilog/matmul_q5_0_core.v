`timescale 1ns / 1ps

// Q5_0 compute core — 2 rows × 896 columns, 56 blocks (28 per row, 32 elements/block)
//
// Pipeline: SETUP_D(1) + COMPUTE(32) + DRAIN(1) = 34 cycles/block
// 56 blocks: 56 × 34 = 1904 cycles (all blocks pass through SETUP_D)
// 1 DSP MAC/cycle, 16 LUTs for q5 decode (no multipliers)
//
// Q5_0 block format (GGUF): d[15:0] (f16) + qh[31:0] + qs[127:0] = 22 bytes
// row_norm is PhaseB-only metadata (UQ8.8), not part of Q5_0 format.
//
// Interface: per-block inputs (d/qh/qs) loaded by FSM via blk_valid pulse.
// No byte-at-a-time header loading, no hdr_packed LUTRAM.
// Each blk_valid carries ONE block's data for ONE row.

module matmul_q5_0_core (
    input  wire         clk, rst_n,

    // ── Core ID (0=rows 0-1, 1=rows 2-3) ──
    input  wire [0:0]   core_id,

    // ── Per-row normalization (PhaseB metadata, UQ8.8) ──
    // row_norm = 32767/max_abs, computed by CPU from block data.
    // Written by FSM once per descriptor, persists across all blocks.
    // 4 entries: core0=norm[0:1], core1=norm[2:3]
    // ** NOT part of the Q5_0 quantization format **
    input  wire         norm_we,
    input  wire [1:0]   norm_addr,    // 0..3
    input  wire [15:0]  norm_din,     // UQ8.8

    // ── Per-block data (pushed by FSM, one block at a time) ──
    input  wire [15:0]  blk_d,        // f16 scale (GGUF Q5_0 block header)
    input  wire [31:0]  blk_qh,       // 32 high bits
    input  wire [127:0] blk_qs,       // 32 nibbles in 16 bytes
    input  wire         blk_valid,    // pulse: latch d/qh/qs and compute

    // ── Activation BRAM (pre-loaded by FSM, 896 × INT16) ──
    input  wire         act_we,
    input  wire [9:0]   act_addr,
    input  wire [15:0]  act_din,

    // ── Control ──
    input  wire         clr_acc,      // pulse: clear accumulators before tile

    // ── Result (both rows accessible combinatorially when done=1) ──
    output wire [47:0]  res0, res1,
    output reg          done,
    output reg          busy
);

    // ── Block compute FSM ──
    localparam IDLE    = 2'd0;
    localparam SETUP_D = 2'd1;
    localparam COMPUTE = 2'd2;
    localparam DRAIN   = 2'd3;

    reg [1:0] state;
    reg [5:0] blk_counter;            // 0..55, tracks current block
    reg [4:0] wi;                     // element index within block (0..31)

    // Latched block data (written on blk_valid)
    reg [15:0]  blk_d_r;
    reg [31:0]  blk_qh_r;
    reg [127:0] blk_qs_r;

    // Per-row normalization registers (4 entries: 2 rows × 2 cores)
    reg [15:0] row_norm [0:3];

    // Fixed-point: d_pre = f16_decode(d) × norm >> 8, clamped S16
    reg signed [15:0] d_pre;

    // Activation BRAM (pre-loaded by FSM, shared across both rows)
    (* ram_style = "block" *) reg [15:0] act_mem [0:1023];

    // Pipeline: act_r holds activation for CURRENT wi (pre-loaded previous cycle)
    reg [15:0] act_r;

    // Accumulators: one per row, S48 signed
    reg signed [47:0] acc [0:1];

    assign res0 = acc[0];
    assign res1 = acc[1];

    // Row tracking: blocks 0..27 → row0, 28..55 → row1
    wire       row_high   = (blk_counter >= 6'd28);

    // Act column tracking: same 28 column blocks for both rows
    wire [5:0] act_blk    = row_high ? (blk_counter - 6'd28) : blk_counter;

    // ── Act BRAM pre-load pipeline ──
    // BRAM read: address clocked at posedge N, data available at posedge N+1.
    // We capture BRAM output at posedge N+1 into act_r for the MAC.
    //
    // SETUP_D/entry: pre-load act[act_blk*32 + 0] for wi=0
    // COMPUTE[n]:    pre-load act[act_blk*32 + n+1] for next wi
    //                MAC uses act_r (loaded previous cycle) = act[blk*32+n]
    // wi==31:        pre-load act[act_blk*32 + 0] for next block's wi=0
    //                (overwritten by SETUP_D if back-to-back)
    wire blk_entry = blk_valid && (state == IDLE || state == DRAIN);
    wire [4:0]  wi_preload = blk_entry    ? 5'd0 :
                             (state == SETUP_D) ? 5'd0 :
                             (wi == 5'd31)      ? 5'd0 : wi + 5'd1;
    wire [9:0]  act_addr_pre = act_blk * 6'd32 + wi_preload;

    // ── Q5 element decode (combinational from latched block data) ──
    // qs nibble: wi[4] selects upper/lower nibble of byte qs[wi[3:0]]
    wire [7:0] qs_byte = blk_qs_r[wi[3:0]*8 +: 8];
    wire [3:0] ql = wi[4] ? qs_byte[7:4] : qs_byte[3:0];
    wire       qh = blk_qh_r[wi];
    wire signed [4:0] q5 = $signed({qh, ql}) - 5'd16;

    // ── f16 decode: (1.mant) × 2^(exp-15) scaled to S24.8 ──
    // Output is S24.8 fixed-point: 1.0 f16 → 256
    function signed [31:0] f16_decode;
        input [15:0] f16;
        reg [4:0] exp;
        reg [9:0] mant;
        begin
            exp = f16[14:10];
            mant = f16[9:0];
            if (exp == 5'd0 || exp == 5'd31)
                f16_decode = 32'sd0;
            else if (exp >= 5'd17)
                f16_decode = $signed((32'd1024 + mant) << (exp - 5'd17));
            else
                f16_decode = $signed(({1'b0, 32'd1024 + mant} +
                    (32'd1 << (5'd17 - exp - 5'd1))) >>> (5'd17 - exp));
        end
    endfunction

    // ── d_pre = f16_decode(d) × norm >> 8, clamped to S16 ──
    // f16_decode output is S24.8 (1.0 = 256).
    // norm is UQ8.8 (1.0 = 256).
    // d_pre = d_fp × norm / 256 = f16_decode(d) × norm >> 8.
    wire signed [31:0] d_fp       = f16_decode(blk_d_r);
    wire [1:0] norm_idx           = {core_id, row_high};
    wire signed [16:0] norm_s     = $signed({1'b0, row_norm[norm_idx]});
    wire signed [47:0] d_pre_full = $signed(d_fp) * norm_s;
    wire signed [47:0] d_pre_shr = d_pre_full >>> 8;
    wire signed [15:0] d_pre_next =
        (d_pre_shr > 32'sd32767)  ? 16'sh7FFF :
        (d_pre_shr < -32'sd32768) ? 16'sh8000 :
        d_pre_shr[15:0];

    // ── dq = d_pre × q5 (purely LUTs, 16×5 multiply) ──
    wire signed [20:0] dq = d_pre * $signed(q5);

    // ── MAC: dq × act_r → acc (one DSP48E1 per core) ──
    wire signed [36:0] prod = $signed(dq) * $signed(act_r);

    // ===== Synchronous logic =====
    always @(posedge clk) begin
        // Activation BRAM write (pre-loaded by FSM)
        if (act_we) act_mem[act_addr] <= act_din;

        // BRAM pre-load pipeline: act_r gets next wi's activation
        act_r <= act_mem[act_addr_pre];
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= IDLE;
            blk_counter  <= 6'd0;
            wi           <= 5'd0;
            done         <= 1'b0;
            busy         <= 1'b0;
            acc[0]       <= 48'd0;
            acc[1]       <= 48'd0;
            d_pre        <= 16'd0;
            blk_d_r      <= 16'd0;
            blk_qh_r     <= 32'd0;
            blk_qs_r     <= 128'd0;
        end else begin
            done <= 1'b0;

            // Clear accumulators at tile start
            if (clr_acc) begin
                acc[0] <= 48'd0;
                acc[1] <= 48'd0;
                blk_counter <= 6'd0;
            end

            // Per-row normalization write
            if (norm_we) row_norm[norm_addr] <= norm_din;

            case (state)
                IDLE: begin
                    busy <= 1'b0;
                    if (blk_valid) begin
                        blk_d_r  <= blk_d;
                        blk_qh_r <= blk_qh;
                        blk_qs_r <= blk_qs;
                        busy <= 1'b1;
                        state <= SETUP_D;
                    end
                end

                SETUP_D: begin
                    d_pre <= d_pre_next;
                    wi    <= 5'd0;
                    state <= COMPUTE;
                end

                COMPUTE: begin
                    // MAC: acc[row] += d_pre × q5(wi) × act[wi]
                    acc[row_high] <= acc[row_high] + prod;

                    if (wi == 5'd31) begin
                        state <= DRAIN;
                    end else begin
                        wi <= wi + 5'd1;
                    end
                end

                DRAIN: begin
                    done <= 1'b1;
                    busy <= 1'b0;
                    blk_counter <= (blk_counter == 6'd55) ? 6'd0 : blk_counter + 6'd1;

                    if (blk_valid) begin
                        // Back-to-back: next block data already available
                        blk_d_r  <= blk_d;
                        blk_qh_r <= blk_qh;
                        blk_qs_r <= blk_qs;
                        state <= SETUP_D;
                    end else begin
                        state <= IDLE;
                    end
                end
            endcase
        end
    end

    // ── iVerilog initial block (simulation only) ──
    `ifdef __ICARUS__
    integer k;
    initial begin
        for (k = 0; k < 1024; k = k + 1)
            act_mem[k] = 16'd0;
        for (k = 0; k < 4; k = k + 1)
            row_norm[k] = 16'd0;
    end
    `endif

endmodule
