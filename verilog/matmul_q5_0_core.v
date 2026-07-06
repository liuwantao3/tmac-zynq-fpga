`timescale 1ns / 1ps

module matmul_q5_0_core (
    input  wire         clk, rst_n,

    // Pre-loaded headers (f16+qh, banks 0-5, LUTRAM, 56 blocks * 6 bytes)
    input  wire         hdr_we,
    input  wire [2:0]   hdr_bank,
    input  wire [5:0]   hdr_addr,
    input  wire  [7:0]  hdr_din,

    // Pre-loaded scales (4 * UQ16.8)
    input  wire         sc_we,
    input  wire [2:0]   sc_addr,
    input  wire [15:0]  sc_din,

    // Per-block qs word (4 * 32-bit = 16 bytes, driven by FSM per block)
    input  wire [127:0] qs_word,

    // Shared act BRAM write (pre-loaded once per descriptor)
    input  wire         act_we,
    input  wire [9:0]   act_addr,
    input  wire [15:0]  act_din,

    // Block number (set by FSM before each start)
    input  wire [5:0]   blk_num,
    // Clear accumulators (pulsed once per descriptor before first block)
    input  wire         clr_acc,

    // Control (shared: both cores start/stop together)
    input  wire         start,
    output reg          done,
    output reg          busy,

    // Result readback
    input  wire [0:0]   res_addr,
    output wire [47:0]  res_dout,

    input  wire [1:0]   core_id,
    input  wire         dbg_tile_start,
    output reg  [31:0]  dbg_tile_cycles,
    output reg  [7:0]   dbg_tile_id,
    input  wire         dbg_verbose
);

    // Block compute FSM (33 cycles/block: 1 SETUP_D + 32 COMPUTE)
    // blk_counter tracks which of 56 blocks we're computing (indexes LUTRAM banks 0-5)
    localparam IDLE    = 3'd0;
    localparam SETUP_D = 3'd1;
    localparam COMPUTE = 3'd2;
    localparam DRAIN   = 3'd3;

    reg [2:0] state;
    reg [5:0] blk_counter;
    reg [4:0] wi;
    reg [31:0] cycle_cnt;
    reg [31:0] tile_start_cycle;
    reg [7:0]  tile_counter;

    // LUTRAM banks 0-5: f16 + qh (pre-loaded once per descriptor)
    // Stored as single packed reg (2688 bits) to avoid iVerilog array aliasing
    // Layout: hdr_packed[blk*48 + bank*8 +: 8] = byte for block blk, bank bank
    reg [2687:0] hdr_packed;
    wire [47:0] hdr_word_w;
    // hdr_word_w assigned after hdr_addr_w declaration (below)

    // qs word — driven by FSM as 128-bit input, sampled combinatorially

    // Shared activation BRAM (written by FSM, read by both cores during compute)
    // Accessed via dedicated port -- shared across both cores
    (* ram_style = "block" *) reg [15:0] act_mem [0:1023];

    // Scales (4 * UQ16.8, pre-loaded once per descriptor)
    reg [15:0] row_scale [0:7];

    // Accumulator (2 rows per core)
    reg signed [47:0] acc [0:1];
    assign res_dout = (res_addr == 0) ? acc[0] : acc[1];

    // Pipeline regs
    reg signed [15:0] d_pre;
    reg [7:0]  hdr0_r, hdr1_r, hdr2_r, hdr3_r, hdr4_r, hdr5_r;
    reg [7:0]  qs_r;
    reg [15:0] act_r;

    // Combinational helpers
    wire        row_high_w  = (blk_counter >= 6'd28);
    wire [5:0]  hdr_addr_w  = blk_counter;
    assign hdr_word_w = hdr_packed[hdr_addr_w*48 +: 48];
    // Act address: same act data reused for both rows (28 unique column positions)
    // blk 0..27 -> act[blk*32 .. blk*32+31]; blk 28..55 -> same as blk-28
    wire [5:0]  act_blk_w   = row_high_w ? (blk_counter - 6'd28) : blk_counter;
    wire [9:0]  act_addr_w  = act_blk_w * 6'd32 + {5'b0, wi};

    // Q5 decode helpers
    wire [15:0] f16_w       = {hdr1_r, hdr0_r};
    wire [31:0] qh_w        = {hdr5_r, hdr4_r, hdr3_r, hdr2_r};
    wire [4:0]  exp_d_w     = f16_w[14:10];
    wire [9:0]  mant_d_w    = f16_w[9:0];
    wire        qh_bit_w    = qh_w[wi];
    // qs read from buffer (4 words * 4 bytes, 2 elements per byte)
    // wi[4:3] = word index (0-3), wi[2:1] = byte within word (0-3), wi[0] = nibble select
    wire [1:0]  wi_word_w = wi[4:3];
    wire [1:0]  wi_byte_w = wi[2:1];
    wire [31:0] qs_word_w = (wi_word_w == 0) ? qs_word[31:0] :
                            (wi_word_w == 1) ? qs_word[63:32] :
                            (wi_word_w == 2) ? qs_word[95:64] : qs_word[127:96];
    wire [7:0]  qs_byte_w = qs_word_w[8*wi_byte_w +: 8];
    wire [3:0]  ql_nibble_w = wi[0] ? qs_byte_w[7:4] : qs_byte_w[3:0];
    wire signed [4:0] q5_w  = ((qh_bit_w << 4) | ql_nibble_w) - 5'd16;

    // f16 decode
    wire signed [31:0] d_fp_w =
        (exp_d_w == 5'd0 || exp_d_w == 5'd31) ? 32'sd0 :
        (exp_d_w >= 5'd17) ? $signed((32'd1024 + mant_d_w) << (exp_d_w - 5'd17)) :
        $signed(({1'b0, 32'd1024 + mant_d_w} + (32'd1 << (5'd17 - exp_d_w - 5'd1)))
                >>> (5'd17 - exp_d_w));

    // d_pre = d_fp * scale >> 8, clamped to S16
    wire [2:0]  scale_idx_w = {core_id[0], row_high_w};
    wire signed [16:0] scale_w  = $signed({1'b0, row_scale[scale_idx_w]});
    wire signed [47:0] d_pre_norm_w = $signed(d_fp_w) * scale_w;
    wire signed [47:0] d_pre_shr_w  = d_pre_norm_w >>> 8;
    wire signed [15:0] d_pre_next =
        (d_pre_shr_w > 32'sd32767)  ? 16'sh7FFF :
        (d_pre_shr_w < -32'sd32768) ? 16'sh8000 :
        d_pre_shr_w[15:0];

    // dq = d_pre * q5 (LUT multiply)
    wire signed [20:0] dq_w = d_pre * $signed(q5_w);

    // MAC
    wire signed [36:0] prod_w = $signed(dq_w) * $signed(act_r);

    // ===== Synchronous writes =====
    always @(posedge clk) begin
        // Header banks (LUTRAM) — packed reg, unconditional write (hdr_we gates bank only)
        if (hdr_we && hdr_bank == 0) hdr_packed[hdr_addr*48 + 0*8 +: 8] <= hdr_din;
        if (hdr_we && hdr_bank == 1) hdr_packed[hdr_addr*48 + 1*8 +: 8] <= hdr_din;
        if (hdr_we && hdr_bank == 2) hdr_packed[hdr_addr*48 + 2*8 +: 8] <= hdr_din;
        if (hdr_we && hdr_bank == 3) hdr_packed[hdr_addr*48 + 3*8 +: 8] <= hdr_din;
        if (hdr_we && hdr_bank == 4) hdr_packed[hdr_addr*48 + 4*8 +: 8] <= hdr_din;
        if (hdr_we && hdr_bank == 5) hdr_packed[hdr_addr*48 + 5*8 +: 8] <= hdr_din;

        // Pipeline reads (registered)
        if (hdr_we && core_id == 1 && hdr_addr < 2)
            $display("[CORE%0d] HDR_WR: we=%b bank=%d addr=%d din=%0h t=%0t",
                core_id, hdr_we, hdr_bank, hdr_addr, hdr_din, $time);
        hdr0_r <= hdr_word_w[7:0];
        hdr1_r <= hdr_word_w[15:8];
        hdr2_r <= hdr_word_w[23:16];
        hdr3_r <= hdr_word_w[31:24];
        hdr4_r <= hdr_word_w[39:32];
        hdr5_r <= hdr_word_w[47:40];
        qs_r   <= qs_byte_w;
        act_r  <= act_mem[act_addr_w];

        // Shared activation BRAM (pre-loaded)
        if (act_we) act_mem[act_addr] <= act_din;
        // Scales
        if (sc_we) row_scale[sc_addr] <= sc_din;
    end

    // ===== Compute FSM =====
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state    <= IDLE;
            blk_counter <= 6'd0;
            wi       <= 5'd0;
            done     <= 1'b0; busy <= 1'b0;
            acc[0]   <= 48'd0; acc[1] <= 48'd0;
            cycle_cnt <= 32'd0; tile_counter <= 8'd0;
            tile_start_cycle <= 32'd0;
            dbg_tile_cycles <= 32'd0; dbg_tile_id <= 8'd0;
            d_pre    <= 16'd0;
        end else begin
            cycle_cnt <= cycle_cnt + 1;
            done <= 1'b0;
            if (clr_acc) begin
                acc[0] <= 48'd0; acc[1] <= 48'd0;
                if (core_id == 1) $display("[CORE1] CLR_ACC fired t=%0t", $time);
            end
            case (state)
                IDLE: begin
                    busy <= 1'b0;
                    if (core_id == 1 && blk_counter == 55 && $time > 45000000 && $time < 46000000)
                        $display("[CORE1] IDLE blk55 acc=[%0d,%0d] t=%0t", acc[0], acc[1], $time);
                    if (start) begin
                        blk_counter <= blk_num;
                        wi <= 5'd0;
                        busy <= 1'b1;
                        tile_start_cycle <= cycle_cnt;
                        tile_counter <= tile_counter + 8'd1;
                        state <= SETUP_D;
                    end
                end

                SETUP_D: begin
                    d_pre <= d_pre_next;
                    state <= COMPUTE;
                    if (blk_num < 4)
                        $display("[CORE%0d] SETUP_D: blk=%0d d_pre=%0d hdr=%0h_%0h_%0h_%0h_%0h_%0h qsw=%0h_%0h_%0h_%0h",
                            core_id, blk_num, d_pre_next,
                            hdr_word_w[7:0], hdr_word_w[15:8],
                            hdr_word_w[23:16], hdr_word_w[31:24],
                            hdr_word_w[39:32], hdr_word_w[47:40],
                            qs_word[127:96], qs_word[95:64], qs_word[63:32], qs_word[31:0]);
                end

                COMPUTE: begin
                    acc[row_high_w] <= acc[row_high_w] + prod_w;
                    if (wi == 5'd31) begin
                        wi <= 5'd0;
                        state <= DRAIN;
                    end else begin
                        wi <= wi + 5'd1;
                    end
                end

                DRAIN: begin
                    if (core_id == 1) $display("[CORE1] DRAIN entry acc=[%0d,%0d] t=%0t", acc[0], acc[1], $time);
                    dbg_tile_cycles <= cycle_cnt - tile_start_cycle;
                    dbg_tile_id <= tile_counter;
                    if (blk_counter < 4 || blk_counter >= 54)
                    $display("[CORE%0d] blk=%0d DONE acc=[%0d,%0d] d_pre=%0d f16=%0h qh=%0h",
                        core_id, blk_counter, acc[0], acc[1], d_pre, f16_w, qh_w);
                if (core_id == 1 && blk_counter == 55)
                    $display("[CORE1] FINAL BLK55: acc[0]=%0d acc[1]=%0d", acc[0], acc[1]);
                    done <= 1'b1; busy <= 1'b0; state <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

    `ifdef __ICARUS__
    integer k, m;
    initial begin
        hdr_packed = 2688'd0;
        for (k = 0; k < 1024; k = k + 1)
            act_mem[k] = 16'd0;
        for (k = 0; k < 8; k = k + 1)
            row_scale[k] = 16'd0;

    end
    `endif

endmodule
