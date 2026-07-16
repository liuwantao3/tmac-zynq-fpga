`timescale 1ns / 1ps

// Q5_0 compute core — 2 rows × 896 columns, 56 blocks (28 per row, 32 elements/block)
//
// Pipeline: SETUP_D(1) + SETUP_D2(1) + SETUP_D3(1) + SETUP_D4(1) + COMPUTE(32) + DRAIN(1) = 37 cycles/block
// 56 blocks: 56 × 37 = 2072 cycles.
// d_pre_full_r register breaks 2-deep DSP48E1 cascade (was 13.17ns critical path).
// dq_r register breaks d_pre→q5→dq→DSP critical path (was 12.258ns, WNS=-2.258ns).
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
    output reg          busy,

    // ── Debug outputs (free-running, not registered in core) ──
    output wire [15:0]  dbg_d_pre,        // current d_pre (S16)
    output wire [5:0]   dbg_blk_counter,  // current block counter
    output wire [31:0]  dbg_d_fp,         // registered f16 decode output (S24.8)
    output wire [15:0]  dbg_norm,         // row_norm for current block (UQ8.8)
    output wire [2:0]   dbg_state,        // core FSM state
    output wire [15:0]  dbg_act_r,        // current activation value (from BRAM pipeline)
    output wire [4:0]   dbg_q5,           // current Q5 decode value
    output wire [4:0]   dbg_wi            // current wi (element index within block)
);

    // ── Block compute FSM ──
    localparam IDLE      = 3'd0;
    localparam SETUP_D   = 3'd1;
    localparam SETUP_D2  = 3'd2;
    localparam SETUP_D3  = 3'd3;
    localparam SETUP_D4  = 3'd4;
    localparam COMPUTE   = 3'd5;
    localparam DRAIN     = 3'd6;

    reg [2:0]  state;
    reg [5:0] blk_counter;            // 0..55, tracks current block
    reg [4:0] wi;                     // element index within block (0..31)

    // Latched block data (written on blk_valid)
    reg [15:0]  blk_d_r;
    reg signed [31:0] d_fp_r;       // pipeline: f16_decode output before DSP multiply
    reg signed [20:0] dq_r;         // pipeline: q5×d_pre result before DSP MAC
    reg [31:0]  blk_qh_r;
    reg [127:0] blk_qs_r;

    // Per-row normalization registers (4 entries: 2 rows × 2 cores)
    reg [15:0] row_norm [0:3];

    // Fixed-point: d_pre = f16_decode(d) × norm >> 8, clamped S16
    reg signed [15:0] d_pre;
    reg signed [47:0] d_pre_full_r;   // pipeline: registered DSP product before shift+clamp

    // Activation BRAM (pre-loaded by FSM, shared across both rows)
    (* ram_style = "block" *) reg [15:0] act_mem [0:1023];

    // Pipeline: act_r holds activation for CURRENT wi (pre-loaded previous cycle)
    reg [15:0] act_r;

    // Pipeline register: breaks DSP48E1 combinational A→P path (was 4.018ns critical)
    reg signed [36:0] prod_r;

    // Accumulators: one per row, S48 signed
    reg signed [47:0] acc [0:1];

    assign res0 = acc[0];
    assign res1 = acc[1];

    // Debug: free-running signal visibility
    assign dbg_d_pre       = d_pre;
    assign dbg_blk_counter = blk_counter;
    assign dbg_d_fp        = d_fp_r;
    assign dbg_norm        = row_norm[norm_idx];
    assign dbg_state       = state;
    assign dbg_act_r       = act_r;
    assign dbg_q5          = q5;
    assign dbg_wi          = wi;

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
    // wi_preload: act_r gets act[wi_preload] at end of current cycle.
    // In SETUP_D4, wi=0 so act_r loads act[0] (paired with dq_r set to d_pre×q5(0)).
    // In COMPUTE(wi=n), wi_preload=n+1 pre-loads act[n+1] for next cycle.
    // With dq_r pipeline, COMPUTE wi=0 uses stale prod_r (skip acc), then
    // COMPUTE wi=1..31 + DRAIN accumulate 32 elements with correct pairing.
    wire [4:0]  wi_preload = blk_entry    ? 5'd0 :
                             (state == SETUP_D)  ? 5'd0 :
                             (state == SETUP_D2) ? 5'd0 :
                             (state == SETUP_D3) ? 5'd0 :
                             (state == SETUP_D4) ? wi :
                             (wi == 5'd31)       ? 5'd0 : wi + 5'd1;
    wire [9:0]  act_addr_pre = act_blk * 6'd32 + wi_preload;

    // ── Q5 element decode (combinational from latched block data) ──
    // wi_for_prod = wi in SETUP_D4 (pre-charge), wi_preload in all other states.
    // This pairs q5 correctly with BRAM-pipelined act_r:
    //   SETUP_D4: wi_for_prod=0, act_r=act[0]      → first element correct
    //   COMPUTE(n): wi_for_prod=n+1, act_r=act[n+1] → subsequent elements correct
    wire [4:0] wi_for_prod = (state == SETUP_D4) ? wi : wi_preload;
    wire [7:0] qs_byte = blk_qs_r[wi_for_prod[3:0]*8 +: 8];
    wire [3:0] ql = wi_for_prod[4] ? qs_byte[7:4] : qs_byte[3:0];
    wire       qh = blk_qh_r[wi_for_prod];
    wire signed [4:0] q5 = $signed({1'b0, {qh, ql}}) - $signed(6'd16);

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
    // d_pre_full_r is registered in SETUP_D2 (breaks 2-DSP cascade critical path)
    wire signed [47:0] d_pre_shr = d_pre_full_r >>> 8;
    wire signed [15:0] d_pre_next =
        (d_pre_shr > 32'sd32767)  ? 16'sh7FFF :
        (d_pre_shr < -32'sd32768) ? 16'sh8000 :
        d_pre_shr[15:0];

    // ── dq = d_pre × q5 (purely LUTs, 16×5 multiply) ──
    wire signed [20:0] dq = d_pre * $signed(q5);

    // ── dq_r pipeline register: breaks d_pre→q5→dq→DSP critical path ──
    // dq (LUT multiply, ~6ns) → dq_r (register) → DSP multiply (~6ns)
    // Without this, d_pre→q5→dq→DSP→prod_r is 12.258ns (WNS=-2.258ns at 10ns).

    // ── MAC: dq_r × act_r → acc (one DSP48E1 per core) ──
    // Force prod=0 during clr_acc to prevent DSP pipeline residue
    wire signed [36:0] prod = clr_acc ? 37'sd0 : $signed(dq_r) * $signed(act_r);

    // ===== Synchronous logic =====
    always @(posedge clk) begin
        if (clr_acc) begin
            // Flush all DSP pipeline registers: DSP48E1 internal AREG/BREG/MREG/PREG
            // hold stale (d_pre × q5) × act for multiple cycles after clr_acc deasserts.
            // Combinatorial prod=0 mux only forces PREG=0; AREG/BREG/MREG retain stale
            // values that propagate through the pipeline → 2-3 stale accumulations.
            dq_r   <= 21'd0;
            prod_r <= 37'd0;
            act_r  <= 16'd0;
        end else begin
            // Activation BRAM write (pre-loaded by FSM)
            if (act_we) act_mem[act_addr] <= act_din;

            // BRAM pre-load pipeline: act_r gets next wi's activation
            act_r <= act_mem[act_addr_pre];

            // DSP pipeline registers: dq_r breaks d_pre→q5→dq→DSP critical path
            dq_r   <= $signed(dq);   // register LUT multiply output (d_pre × q5)
            prod_r <= prod;          // register DSP multiply output (dq_r × act_r)
        end
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
            d_pre_full_r <= 48'd0;
            d_fp_r       <= 32'd0;
            blk_d_r      <= 16'd0;
            blk_qh_r     <= 32'd0;
            blk_qs_r     <= 128'd0;
        end else begin
            done <= 1'b0;

            // Clear accumulators at tile start (pipeline flush also clears dq_r/prod_r/act_r)
            if (clr_acc) begin
                acc[0] <= 48'd0;
                acc[1] <= 48'd0;
                blk_counter <= 6'd0;
                wi <= 5'd0;
                d_pre        <= 16'd0;
                d_pre_full_r <= 48'd0;
                d_fp_r       <= 32'd0;
                blk_d_r      <= 16'd0;
                blk_qh_r     <= 32'd0;
                blk_qs_r     <= 128'd0;
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
                    d_fp_r <= d_fp;      // register f16_decode output (long combinational path)
                    state  <= SETUP_D2;
                end

                SETUP_D2: begin
                    d_pre_full_r <= $signed(d_fp_r) * norm_s; // register DSP product (break cascade)
                    wi    <= 5'd0;
                    state <= SETUP_D3;
                end

                SETUP_D3: begin
                    d_pre <= d_pre_next; // shift+clamp from registered d_pre_full_r
                    state <= SETUP_D4;
                end

                SETUP_D4: begin
                    // dq_r loaded with d_pre × q5(wi=0); prod_r still has stale value.
                    // First element (wi=0) will be accumulated in COMPUTE(wi=1).
                    state <= COMPUTE;
                end

                COMPUTE: begin
                    // MAC: acc[row] += prod_r (DSP result from previous cycle, dq_r × act_r).
                    // Skip wi=0: prod_r from SETUP_D4 has stale dq_r (pre-pipeline residue).
                    // dq_r pipeline adds 1-cycle latency: first real element in COMPUTE(wi=1).
                    if (wi != 5'd0)
                        acc[row_high] <= acc[row_high] + prod_r;

                    if (wi == 5'd31) begin
                        state <= DRAIN;
                    end else begin
                        wi <= wi + 5'd1;
                    end
                end

                DRAIN: begin
                    // Last element accumulated here (dq_r adds 1-cycle latency vs combinational dq)
                    acc[row_high] <= acc[row_high] + prod_r;
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
