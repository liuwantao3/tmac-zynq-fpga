`timescale 1ns / 1ps

module matmul_top (
    input  wire         clk,
    input  wire         rst_n,

    // AXI4-Lite slave interface
    input  wire         s_axil_awvalid,
    output reg          s_axil_awready,
    input  wire [15:0]  s_axil_awaddr,
    input  wire         s_axil_wvalid,
    output reg          s_axil_wready,
    input  wire [31:0]  s_axil_wdata,
    input  wire [3:0]   s_axil_wstrb,
    output reg          s_axil_bvalid,
    input  wire         s_axil_bready,
    output reg  [1:0]   s_axil_bresp,
    input  wire         s_axil_arvalid,
    output reg          s_axil_arready,
    input  wire [15:0]  s_axil_araddr,
    output reg          s_axil_rvalid,
    input  wire         s_axil_rready,
    output reg  [31:0]  s_axil_rdata,
    output reg  [1:0]   s_axil_rresp,

    // Interrupt
    output reg          interrupt
);

    // ======================================================================
    // Parameters
    // ======================================================================
    localparam REG_AP_CTRL   = 16'h0000;
    localparam REG_GIE       = 16'h0004;
    localparam REG_IER       = 16'h0008;
    localparam REG_ISR       = 16'h000C;
    localparam REG_CTRL_USER = 16'h0010;
    localparam REG_STATUS    = 16'h0014;

    // Data buffer base addresses (word-aligned)
    localparam BUF_WEIGHT_BASE    = 16'h0400;  // AXI addr 0x1000 >> 2
    localparam BUF_WEIGHT_END     = 16'h07FF;  // AXI addr 0x1FFF >> 2 (Q8: 4KB)
    localparam BUF_WEIGHT_EXT_END = 16'h0BFF;  // AXI addr 0x2FFF >> 2 (INT16: 8192 bytes)
    localparam BUF_SCALE_BASE  = 16'h0800;  // AXI addr 0x2000 >> 2
    localparam BUF_SCALE_END   = 16'h0803;  // 128 entries = 64 words
    localparam BUF_ACT_BASE    = 16'h0840;  // AXI addr 0x2100 >> 2
    localparam BUF_ACT_END     = 16'h0847;  // 64 entries = 32 words
    localparam BUF_RES_BASE    = 16'h1000;  // AXI addr 0x4000 >> 2
    localparam BUF_RES_END     = 16'h103F;  // 64 entries = 64 words (lower 32b)
    localparam BUF_RES_HI_BASE = 16'h1080;  // AXI addr 0x4200 >> 2
    localparam BUF_RES_HI_END  = 16'h109F;  // upper 16b of 64 entries

    // ======================================================================
    // Register file
    // ======================================================================
    reg [31:0] reg_ap_ctrl;
    reg [31:0] reg_gie;
    reg [31:0] reg_ier;
    reg [31:0] reg_isr;
    reg [31:0] reg_ctrl_user;
    reg [31:0] reg_status;

    // ======================================================================
    // Mode selection (mutually exclusive in practice)
    //   bit 4 = CTRL_MODE_INT16, bit 5 = CTRL_MODE_Q8PATH, bit 6 = CTRL_MODE_Q4K
    //   Default (no bits) = Q8 for backward compat
    // ======================================================================
    wire mode_int16;
    wire mode_q8;
    wire mode_q4k;
    assign mode_q4k   = reg_ctrl_user[6];
    assign mode_q8    = reg_ctrl_user[5] || (!reg_ctrl_user[4] && !reg_ctrl_user[6]);
    assign mode_int16 = reg_ctrl_user[4];

    // ======================================================================
    // Data buffer memories
    // ======================================================================
    reg [7:0]  weight_buf [0:8191];  // Shared: INT16=8192B, Q8=4096B, Q4K=32256B
    reg [15:0] scale_buf  [0:895];  // Q8: 128 entries, Q4K: 896 entries for row_scale
    reg [15:0] act_buf    [0:63];
    reg [47:0] result_buf [0:63];

    integer bi;

    // ======================================================================
    // Core connections
    // ======================================================================
    wire        core_start;
    wire        core_op_vecmul;
    wire        core_done;
    wire        core_busy;

    // INT16 core signals
    reg         int16_wt_we;
    reg [12:0]  int16_wt_addr;
    reg [7:0]   int16_wt_din;
    wire        int16_done, int16_busy;
    wire [47:0] int16_res_dout;

    // Q8 core signals
    wire        q8_done, q8_busy;
    wire [47:0] q8_res_dout;

    // Q4K core signals
    reg         q4k_wt_we;
    reg [12:0]  q4k_wt_addr;
    reg [7:0]   q4k_wt_din;
    wire        q4k_done, q4k_busy, q4k_decode_busy;
    wire [47:0] q4k_res_dout;

    // Core shared interfaces (Q8 core uses these directly; INT16/Q4K use dedicated signals)
    reg         wt_we;
    reg [11:0]  wt_addr;
    reg [7:0]   wt_din;
    reg         sc_we;
    reg [6:0]   sc_addr;
    reg [15:0]  sc_din;
    reg         act_we;
    reg [5:0]   act_addr;
    reg [15:0]  act_din;
    wire [5:0]  res_addr;
    wire [47:0] res_dout;

    // ======================================================================
    // Core instantiation 1: matmul_int16_core — general INT16×INT16
    // ======================================================================
    matmul_int16_core u_core_int16 (
        .clk       (clk),
        .rst_n     (rst_n),
        .start     (core_start & mode_int16),
        .op_vecmul (core_op_vecmul),
        .done      (int16_done),
        .busy      (int16_busy),
        .wt_we     (int16_wt_we),
        .wt_addr   (int16_wt_addr),
        .wt_din    (int16_wt_din),
        .sc_we     (1'b0),
        .sc_addr   (7'd0),
        .sc_din    (16'd0),
        .act_we    (act_we),
        .act_addr  (act_addr),
        .act_din   (act_din),
        .res_addr  (res_addr),
        .res_dout  (int16_res_dout)
    );

    // ======================================================================
    // Core instantiation 2: matmul_q8_core — Q8_0 dequant + INT16 compute
    // ======================================================================
    matmul_q8_core u_core_q8 (
        .clk       (clk),
        .rst_n     (rst_n),
        .start     (core_start & mode_q8),
        .op_vecmul (core_op_vecmul),
        .done      (q8_done),
        .busy      (q8_busy),
        .wt_we     (wt_we),
        .wt_addr   (wt_addr),
        .wt_din    (wt_din),
        .sc_we     (sc_we),
        .sc_addr   (sc_addr),
        .sc_din    (sc_din),
        .act_we    (act_we),
        .act_addr  (act_addr),
        .act_din   (act_din),
        .res_addr  (res_addr),
        .res_dout  (q8_res_dout)
    );

    // ======================================================================
    // Core instantiation 3: matmul_q4k_core — Q4_K block decode + INT16 compute
    // ======================================================================
    matmul_q4k_core u_core_q4k (
        .clk       (clk),
        .rst_n     (rst_n),
        .start     (core_start & mode_q4k),
        .op_vecmul (core_op_vecmul),
        .done      (q4k_done),
        .busy      (q4k_busy),
        .wt_we     (q4k_wt_we),
        .wt_addr   (q4k_wt_addr),
        .wt_din    (q4k_wt_din),
        .sc_we     (1'b0),
        .sc_addr   (7'd0),
        .sc_din    (16'd0),
        .act_we    (act_we),
        .act_addr  (act_addr),
        .act_din   (act_din),
        .res_addr  (res_addr),
        .res_dout  (q4k_res_dout),
        .mode_block_load (mode_q4k),
        .decode_busy     (q4k_decode_busy)
    );

    // ======================================================================
    // 3-way mux: select done/busy/results from active core
    // ======================================================================
    assign core_done = mode_q4k  ? q4k_done  :
                       mode_q8   ? q8_done   : int16_done;
    assign core_busy = mode_q4k  ? (q4k_busy | q4k_decode_busy) :
                       mode_q8   ? q8_busy   : int16_busy;
    assign res_dout  = mode_q4k  ? q4k_res_dout  :
                       mode_q8   ? q8_res_dout   : int16_res_dout;

    // ======================================================================
    // AXI-Lite write transaction
    // ======================================================================
    reg [1:0] wstate;
    localparam W_IDLE = 0, W_WAIT = 1, W_RESP = 2;

    wire is_ctrl_write, is_buf_write;
    assign is_ctrl_write = (s_axil_awaddr[15:12] == 0);
    assign is_buf_write  = (s_axil_awaddr[15:12] == 1 || s_axil_awaddr[15:12] == 2);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wstate <= W_IDLE;
            s_axil_awready <= 0;
            s_axil_wready  <= 0;
            s_axil_bvalid  <= 0;
            s_axil_bresp   <= 0;
            reg_ap_ctrl    <= 32'h0000_0004;
            reg_gie        <= 0;
            reg_ier        <= 0;
            reg_isr        <= 0;
            reg_ctrl_user  <= 0;
        end else begin
            case (wstate)
                W_IDLE: begin
                    s_axil_awready <= 1;
                    s_axil_wready  <= 1;
                    if (s_axil_awvalid && s_axil_wvalid) begin
                        s_axil_awready <= 0;
                        s_axil_wready  <= 0;

                        // Control register writes
                        if (is_ctrl_write) begin
                            case (s_axil_awaddr)
                                REG_AP_CTRL:   reg_ap_ctrl   <= s_axil_wdata;
                                REG_GIE:       reg_gie       <= s_axil_wdata;
                                REG_IER:       reg_ier       <= s_axil_wdata;
                                REG_CTRL_USER: reg_ctrl_user <= s_axil_wdata;
                                REG_ISR: begin
                                    if (s_axil_wdata[0])
                                        reg_isr[0] <= 0;
                                end
                            endcase
                        end

                        // Buffer writes (into weight/scale/act memories)
                        if (is_buf_write) begin
                            write_buffer(s_axil_awaddr, s_axil_wdata, s_axil_wstrb);
                        end

                        wstate <= W_RESP;
                    end
                end

                W_RESP: begin
                    s_axil_bvalid <= 1;
                    s_axil_bresp  <= 2'b00;
                    if (s_axil_bready) begin
                        s_axil_bvalid <= 0;
                        wstate <= W_IDLE;
                    end
                end
            endcase
        end
    end

    // ======================================================================
    // Buffer write helper
    //   In INT16/Q4K mode the full 0x1000-0x2FFF range maps to weight_buf;
    //   scale_buf and act_buf at 0x2000-0x217F are unused.
    //   In Q8 mode (default), 0x2000-0x20FF → scale_buf, 0x2100-0x217F → act_buf.
    // ======================================================================
    wire mode_bypass_guard = mode_int16 | mode_q4k;

    task write_buffer(input [15:0] addr, input [31:0] data, input [3:0] strb);
        reg [11:0] word_off;
        begin
            word_off = addr[13:2];

            // Scales: 0x2000-0x23FF (Q8: 128 entries, Q4K: 896 entries for row_scale)
            if (mode_q8 &&
                word_off >= BUF_SCALE_BASE && word_off <= BUF_SCALE_END) begin
                scale_buf[(word_off - BUF_SCALE_BASE) * 2 + 0] <= data[15:0];
                scale_buf[(word_off - BUF_SCALE_BASE) * 2 + 1] <= data[31:16];
            end
            // Q4K row_scale: 896 entries at 0x2000-0x23FF
            if (mode_q4k &&
                word_off >= BUF_SCALE_BASE && word_off <= (BUF_SCALE_BASE + 223)) begin
                scale_buf[(word_off - BUF_SCALE_BASE) * 2 + 0] <= data[15:0];
                scale_buf[(word_off - BUF_SCALE_BASE) * 2 + 1] <= data[31:16];
            end

            // Activations: 0x2100-0x217F (Q8 and INT16 modes)
            if (!(mode_q4k) &&
                word_off >= BUF_ACT_BASE && word_off <= BUF_ACT_END) begin
                act_buf[(word_off - BUF_ACT_BASE) * 2 + 0] <= data[15:0];
                act_buf[(word_off - BUF_ACT_BASE) * 2 + 1] <= data[31:16];
            end

            // Weights: 0x1000-0x1FFF (Q8) or 0x1000-0x2FFF (INT16/Q4K)
            if (word_off >= BUF_WEIGHT_BASE && word_off <= BUF_WEIGHT_EXT_END) begin
                if (mode_bypass_guard ||
                    !(word_off >= BUF_SCALE_BASE && word_off <= BUF_SCALE_END) &&
                    !(word_off >= BUF_ACT_BASE && word_off <= BUF_ACT_END)) begin
                    if (strb[0]) weight_buf[(word_off - BUF_WEIGHT_BASE) * 4 + 0] <= data[7:0];
                    if (strb[1]) weight_buf[(word_off - BUF_WEIGHT_BASE) * 4 + 1] <= data[15:8];
                    if (strb[2]) weight_buf[(word_off - BUF_WEIGHT_BASE) * 4 + 2] <= data[23:16];
                    if (strb[3]) weight_buf[(word_off - BUF_WEIGHT_BASE) * 4 + 3] <= data[31:24];
                end
            end
        end
    endtask

    // ======================================================================
    // AXI-Lite read transaction
    // ======================================================================
    reg [1:0] rstate;
    localparam R_IDLE = 0, R_DATA = 1;

    wire [11:0] rd_word;
    assign rd_word = s_axil_araddr[13:2];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rstate <= R_IDLE;
            s_axil_arready <= 0;
            s_axil_rvalid  <= 0;
            s_axil_rdata   <= 0;
            s_axil_rresp   <= 0;
        end else begin
            case (rstate)
                R_IDLE: begin
                    s_axil_arready <= 1;
                    if (s_axil_arvalid) begin
                        s_axil_arready <= 0;
                        rstate <= R_DATA;

                        case (s_axil_araddr)
                            REG_AP_CTRL:   s_axil_rdata <= reg_ap_ctrl;
                            REG_GIE:       s_axil_rdata <= reg_gie;
                            REG_IER:       s_axil_rdata <= reg_ier;
                            REG_ISR:       s_axil_rdata <= reg_isr;
                            REG_CTRL_USER: s_axil_rdata <= reg_ctrl_user;
                            REG_STATUS:    s_axil_rdata <= reg_status;
                            default: begin
                                if (rd_word >= BUF_SCALE_BASE && rd_word <= BUF_SCALE_END) begin
                                    s_axil_rdata <= {scale_buf[(rd_word - BUF_SCALE_BASE) * 2 + 1],
                                                     scale_buf[(rd_word - BUF_SCALE_BASE) * 2 + 0]};
                                end else if (rd_word >= BUF_ACT_BASE && rd_word <= BUF_ACT_END) begin
                                    s_axil_rdata <= {act_buf[(rd_word - BUF_ACT_BASE) * 2 + 1],
                                                     act_buf[(rd_word - BUF_ACT_BASE) * 2 + 0]};
                                end else if (rd_word >= BUF_WEIGHT_BASE && rd_word <= BUF_WEIGHT_EXT_END) begin
                                    s_axil_rdata <= {weight_buf[(rd_word - BUF_WEIGHT_BASE) * 4 + 3],
                                                     weight_buf[(rd_word - BUF_WEIGHT_BASE) * 4 + 2],
                                                     weight_buf[(rd_word - BUF_WEIGHT_BASE) * 4 + 1],
                                                     weight_buf[(rd_word - BUF_WEIGHT_BASE) * 4 + 0]};
                                end else if (rd_word >= BUF_RES_BASE && rd_word <= BUF_RES_END) begin
                                    s_axil_rdata <= result_buf[rd_word - BUF_RES_BASE][31:0];
                                end else if (rd_word >= BUF_RES_HI_BASE && rd_word <= BUF_RES_HI_END) begin
                                    s_axil_rdata <= {16'b0, result_buf[rd_word - BUF_RES_HI_BASE][47:32]};
                                end else begin
                                    s_axil_rdata <= 32'h0;
                                end
                            end
                        endcase
                    end
                end

                R_DATA: begin
                    s_axil_rvalid <= 1;
                    s_axil_rresp  <= 2'b00;
                    if (s_axil_rready) begin
                        s_axil_rvalid <= 0;
                        rstate <= R_IDLE;
                    end
                end
            endcase
        end
    end

    // ======================================================================
    // Data loading FSM — copies from weight_buf/scale_buf/act_buf into core
    // ======================================================================
    reg        loading;
    reg [12:0] load_addr;
    reg [2:0]  load_phase;

    localparam LP_IDLE   = 0;
    localparam LP_WEIGHT = 1;
    localparam LP_SCALE  = 2;
    localparam LP_ACT    = 3;
    localparam LP_DONE   = 4;

    // INT16 weight bytes:  8192 (64 cols × 64 rows × 2 bytes)
    // Q8   weight bytes:  4096 (64 cols × 64 rows × 1 byte)
    // Q4K  weight bytes: 32256 (224 blocks × 144 bytes = 896 rows × 64 cols)
    localparam INT16_WEIGHT_BYTES = 14'd8192;
    localparam Q8_WEIGHT_BYTES    = 13'd4096;
    localparam Q4K_WEIGHT_BYTES  = 15'd32256;

    assign core_start     = reg_ap_ctrl[0] && !core_busy && !loading;
    assign core_op_vecmul = reg_ctrl_user[3];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            loading    <= 0;
            load_phase <= LP_IDLE;
            load_addr  <= 0;
            int16_wt_we <= 0;
            wt_we      <= 0;
            q4k_wt_we  <= 0;
            sc_we      <= 0;
            act_we     <= 0;
        end else if (reg_ap_ctrl[0] && !core_busy && !loading) begin
            reg_ap_ctrl[0] <= 0;
            loading    <= 1;
            load_phase <= LP_WEIGHT;
            load_addr  <= 0;
        end else if (loading) begin
            int16_wt_we <= 0;
            wt_we <= 0;
            q4k_wt_we <= 0;
            sc_we <= 0;
            act_we <= 0;

            case (load_phase)
                LP_WEIGHT: begin
                    if (mode_q4k) begin
                        if (load_addr < Q4K_WEIGHT_BYTES) begin
                            q4k_wt_we   <= 1;
                            q4k_wt_addr <= {5'd0, load_addr[11:0]};
                            q4k_wt_din  <= weight_buf[load_addr];
                            load_addr <= load_addr + 1;
                        end else begin
                            load_addr <= 0;
                            load_phase <= LP_SCALE;
                        end
                    end else if (mode_int16) begin
                        if (load_addr < INT16_WEIGHT_BYTES) begin
                            int16_wt_we <= 1;
                            int16_wt_addr <= {load_addr[3:0], load_addr[12:4]};
                            int16_wt_din <= weight_buf[load_addr];
                            load_addr <= load_addr + 1;
                        end else begin
                            load_addr <= 0;
                            load_phase <= LP_ACT;
                        end
                    end else begin
                        // Q8 mode (default)
                        if (load_addr < Q8_WEIGHT_BYTES) begin
                            wt_we   <= 1;
                            wt_addr <= load_addr[11:0];
                            wt_din  <= weight_buf[load_addr[11:0]];
                            load_addr <= load_addr + 1;
                        end else begin
                            load_addr <= 0;
                            load_phase <= LP_SCALE;
                        end
                    end
                end

                LP_SCALE: begin
                    if (mode_q4k) begin
                        if (load_addr < 896) begin
                            sc_we   <= 1;
                            sc_addr <= load_addr[9:0];
                            sc_din  <= scale_buf[load_addr[9:0]];
                            load_addr <= load_addr + 1;
                        end else begin
                            load_addr <= 0;
                            load_phase <= LP_ACT;
                        end
                    end else if (!mode_int16) begin
                        // Q8 mode: load 128 combined scales
                        if (load_addr < 128) begin
                            sc_we   <= 1;
                            sc_addr <= load_addr[6:0];
                            sc_din  <= scale_buf[load_addr[6:0]];
                            load_addr <= load_addr + 1;
                        end else begin
                            load_addr <= 0;
                            load_phase <= LP_ACT;
                        end
                    end else begin
                        load_addr <= 0;
                        load_phase <= LP_ACT;
                    end
                end

                LP_ACT: begin
                    if (load_addr < 64) begin
                        act_we   <= 1;
                        act_addr <= load_addr[5:0];
                        act_din  <= act_buf[load_addr[5:0]];
                        load_addr <= load_addr + 1;
                    end else begin
                        load_addr <= 0;
                        load_phase <= LP_DONE;
                    end
                end

                LP_DONE: begin
                    loading    <= 0;
                    load_phase <= LP_IDLE;
                end
            endcase
        end
    end

    // ======================================================================
    // Capture core results
    // ======================================================================
    reg [9:0] result_idx;
    reg       draining;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            draining   <= 0;
            result_idx <= 0;
        end else if (core_done && !draining) begin
            draining   <= 1;
            result_idx <= 0;
        end else if (draining) begin
            if (result_idx < 64) begin
                result_buf[result_idx] <= res_dout;
                result_idx <= result_idx + 1;
            end else begin
                draining <= 0;
            end
        end
    end

    assign res_addr = draining ? result_idx[5:0] : 0;

    // ======================================================================
    // STATUS register
    // ======================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            reg_status <= 0;
        else if (core_busy)
            reg_status <= 1;
        else if (core_done && !draining)
            reg_status <= 2;
        else
            reg_status <= 0;
    end

    // ======================================================================
    // AP_CTRL done/idle/ready bits (read-only)
    // ======================================================================
    always @(*) begin
        reg_ap_ctrl[3:1] = {~core_busy, ~core_busy, core_done};
    end

    // ======================================================================
    // Interrupt
    // ======================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            reg_isr <= 0;
        else if (core_done && reg_gie[0] && reg_ier[0])
            reg_isr[0] <= 1;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            interrupt <= 0;
        else
            interrupt <= reg_gie[0] && reg_ier[0] && reg_isr[0];
    end

endmodule
