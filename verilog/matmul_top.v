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
    output reg          interrupt,

    // AXI HP read master (DDR → PL)
    output wire [31:0]  m_axi_araddr,
    output wire         m_axi_arvalid,
    input  wire         m_axi_arready,
    output wire [7:0]   m_axi_arlen,
    output wire [2:0]   m_axi_arsize,
    output wire [1:0]   m_axi_arburst,
    input  wire [63:0]  m_axi_rdata,
    input  wire         m_axi_rvalid,
    output wire         m_axi_rready,
    input  wire         m_axi_rlast,

    // AXI HP write master (PL → DDR)
    output wire [31:0]  m_axi_awaddr,
    output wire         m_axi_awvalid,
    input  wire         m_axi_awready,
    output wire [7:0]   m_axi_awlen,
    output wire [2:0]   m_axi_awsize,
    output wire [1:0]   m_axi_awburst,
    output wire [63:0]  m_axi_wdata,
    output wire         m_axi_wvalid,
    input  wire         m_axi_wready,
    output wire         m_axi_wlast,
    output wire [7:0]   m_axi_wstrb,
    input  wire         m_axi_bvalid,
    output wire         m_axi_bready,
    input  wire  [1:0]  m_axi_bresp
);

    // ======================================================================
    // Parameters — register addresses
    // ======================================================================
    localparam REG_AP_CTRL   = 16'h0000;
    localparam REG_GIE       = 16'h0004;
    localparam REG_IER       = 16'h0008;
    localparam REG_ISR       = 16'h000C;
    localparam REG_CTRL_USER = 16'h0010;
    localparam REG_STATUS    = 16'h0014;
    localparam REG_DESC_BASE = 16'h0018;
    localparam REG_DESC_TAIL = 16'h001C;
    localparam REG_DESC_HEAD = 16'h0020;
    localparam REG_CHAIN_CTRL= 16'h0024;

    // ======================================================================
    // Register file
    // ======================================================================
    reg [31:0] reg_ap_ctrl;
    reg [31:0] reg_gie;
    reg [31:0] reg_ier;
    reg [31:0] reg_isr;
    reg [31:0] reg_ctrl_user;
    reg [31:0] reg_status;
    reg [31:0] reg_desc_base;
    reg [31:0] reg_desc_tail;
    reg [31:0] reg_desc_head;
    reg [31:0] reg_chain_ctrl;

    // ======================================================================
    // Mode wires  — bit[8:4] of reg_ctrl_user
    //   4=INT16, 5=Q8_0, 6=Q4_K, 7=Q5_0, 8=Q6_K
    //   Default (none set) = Q8 for backward compat
    // ======================================================================
    wire mode_int16 = reg_ctrl_user[4];
    wire mode_q8    = reg_ctrl_user[5] || (!reg_ctrl_user[4] && !reg_ctrl_user[6]
                                        && !reg_ctrl_user[7] && !reg_ctrl_user[8]);
    wire mode_q4k   = reg_ctrl_user[6];
    wire mode_q5_0  = reg_ctrl_user[7];
    wire mode_q6_k  = reg_ctrl_user[8];

    // ======================================================================
    // Internal buffers
    // ======================================================================
    reg [15:0] act_buf    [0:895];  // max columns (Q5_0)
    reg [47:0] result_buf [0:63];

    // ======================================================================
    // Core shared signals
    // ======================================================================
    wire        core_start;
    wire        core_done;
    wire        core_busy;
    reg  [7:0]  wt_din;
    reg  [12:0] wt_addr;
    reg         wt_we;
    reg  [15:0] sc_din;
    reg  [9:0]  sc_addr;
    reg         sc_we;
    reg  [15:0] act_din;
    reg  [9:0]  act_addr;
    reg         act_we;
    wire [5:0]  res_addr;
    wire [47:0] res_dout;

    // ======================================================================
    // INT16 core signals
    // ======================================================================
    reg  [12:0] int16_wt_addr;
    reg  [7:0]  int16_wt_din;
    reg         int16_wt_we;
    wire        int16_done, int16_busy;
    wire [47:0] int16_res_dout;

    matmul_int16_core u_core_int16 (
        .clk       (clk),
        .rst_n     (rst_n),
        .start     (core_start & mode_int16),
        .op_vecmul (1'b1),
        .done      (int16_done),
        .busy      (int16_busy),
        .wt_we     (int16_wt_we),
        .wt_addr   (int16_wt_addr),
        .wt_din    (int16_wt_din),
        .sc_we     (1'b0),
        .sc_addr   (7'd0),
        .sc_din    (16'd0),
        .act_we    (act_we),
        .act_addr  (act_addr[5:0]),
        .act_din   (act_din),
        .res_addr  (res_addr[5:0]),
        .res_dout  (int16_res_dout)
    );

    // ======================================================================
    // Q8 core signals
    // ======================================================================
    wire        q8_done, q8_busy;
    wire [47:0] q8_res_dout;

    matmul_q8_core u_core_q8 (
        .clk       (clk),
        .rst_n     (rst_n),
        .start     (core_start & mode_q8),
        .op_vecmul (1'b1),
        .done      (q8_done),
        .busy      (q8_busy),
        .wt_we     (wt_we),
        .wt_addr   (wt_addr[11:0]),
        .wt_din    (wt_din),
        .sc_we     (sc_we),
        .sc_addr   (sc_addr[6:0]),
        .sc_din    (sc_din),
        .act_we    (act_we),
        .act_addr  (act_addr[5:0]),
        .act_din   (act_din),
        .res_addr  (res_addr[5:0]),
        .res_dout  (q8_res_dout)
    );

    // ======================================================================
    // Q4K core signals (byte-wide block_buf, auto-increment write_ptr)
    // ======================================================================
    wire        q4k_done, q4k_busy, q4k_decode_busy;
    wire [47:0] q4k_res_dout;

    matmul_q4k_core u_core_q4k (
        .clk       (clk),
        .rst_n     (rst_n),
        .start     (core_start & mode_q4k),
        .op_vecmul (1'b1),
        .done      (q4k_done),
        .busy      (q4k_busy),
        .wt_we     (wt_we),
        .wt_addr   (wt_addr),
        .wt_din    (wt_din),
        .sc_we     (1'b0),
        .sc_addr   (7'd0),
        .sc_din    (16'd0),
        .act_we    (act_we),
        .act_addr  (act_addr[7:0]),
        .act_din   (act_din),
        .res_addr  (res_addr[5:0]),
        .res_dout  (q4k_res_dout),
        .mode_block_load (mode_q4k),
        .decode_busy     (q4k_decode_busy)
    );

    // ======================================================================
    // Q5_0 core  (8×896 tile, 224 × 22-byte blocks)
    // ======================================================================
    wire        q5_0_done, q5_0_busy;
    wire [47:0] q5_0_res_dout;

    matmul_q5_0_core u_core_q5_0 (
        .clk       (clk),
        .rst_n     (rst_n),
        .start     (core_start & mode_q5_0),
        .done      (q5_0_done),
        .busy      (q5_0_busy),
        .wt_we     (wt_we),
        .wt_addr   (wt_addr),
        .wt_din    (wt_din),
        .sc_we     (sc_we),
        .sc_addr   (sc_addr[2:0]),
        .sc_din    (sc_din),
        .act_we    (act_we),
        .act_addr  (act_addr),
        .act_din   (act_din),
        .res_addr  (res_addr[2:0]),
        .res_dout  (q5_0_res_dout)
    );

    // ======================================================================
    // Q6_K core (32×256 tile, 32 × 210-byte blocks, auto-increment write_ptr)
    // ======================================================================
    wire        q6_k_done, q6_k_busy, q6_k_decode_busy;
    wire [47:0] q6_k_res_dout;

    matmul_q6_k_core u_core_q6_k (
        .clk       (clk),
        .rst_n     (rst_n),
        .start     (core_start & mode_q6_k),
        .op_vecmul (1'b1),
        .done      (q6_k_done),
        .busy      (q6_k_busy),
        .wt_we     (wt_we),
        .wt_addr   (wt_addr),
        .wt_din    (wt_din),
        .sc_we     (sc_we),
        .sc_addr   (sc_addr[4:0]),
        .sc_din    (sc_din),
        .act_we    (act_we),
        .act_addr  (act_addr[7:0]),
        .act_din   (act_din),
        .res_addr  (res_addr[4:0]),
        .res_dout  (q6_k_res_dout),
        .mode_block_load (mode_q6_k),
        .decode_busy     (q6_k_decode_busy)
    );

    // ======================================================================
    // 5-way mux: select done/busy/results from active core
    // ======================================================================
    assign core_done = mode_q5_0  ? q5_0_done        :
                       mode_q6_k  ? q6_k_done         :
                       mode_q4k   ? q4k_done          :
                       mode_q8    ? q8_done           : int16_done;
    assign core_busy = mode_q5_0  ? q5_0_busy         :
                       mode_q6_k  ? (q6_k_busy | q6_k_decode_busy) :
                       mode_q4k   ? (q4k_busy | q4k_decode_busy) :
                       mode_q8    ? q8_busy           : int16_busy;
    assign res_dout  = mode_q5_0  ? q5_0_res_dout     :
                       mode_q6_k  ? q6_k_res_dout     :
                       mode_q4k   ? q4k_res_dout      :
                       mode_q8    ? q8_res_dout       : int16_res_dout;

    // ======================================================================
    // AXI4-Lite write FSM
    // ======================================================================
    reg [1:0] wstate;
    localparam W_IDLE = 0, W_WAIT_W = 1, W_RESP = 2, W_RESP2 = 3;

    reg [15:0] waddr_buf;
    reg [31:0] wdata_buf;
    reg [3:0]  wstrb_buf;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wstate         <= W_IDLE;
            s_axil_awready <= 0;
            s_axil_wready  <= 0;
            s_axil_bvalid  <= 0;
            s_axil_bresp   <= 0;
            reg_ap_ctrl    <= 32'h0000_0004;
            reg_gie        <= 0;
            reg_ier        <= 0;
            reg_isr        <= 0;
            reg_ctrl_user  <= 0;
            reg_desc_base  <= 0;
            reg_desc_tail  <= 0;
            reg_chain_ctrl <= 0;
            waddr_buf      <= 0;
            wdata_buf      <= 0;
            wstrb_buf      <= 0;
        end else begin
            case (wstate)
                W_IDLE: begin
                    s_axil_awready <= 1;
                    s_axil_wready  <= 1;
                    if (s_axil_awvalid && s_axil_wvalid) begin
                        // Both valid same cycle — write immediately
                        s_axil_awready <= 0;
                        s_axil_wready  <= 0;
                        write_reg(s_axil_awaddr, s_axil_wdata, s_axil_wstrb);
                        wstate <= W_RESP;
                    end else if (s_axil_awvalid) begin
                        // Address first, wait for data
                        s_axil_awready <= 0;
                        waddr_buf <= s_axil_awaddr;
                        wstate <= W_WAIT_W;
                    end else if (s_axil_wvalid) begin
                        // Data first, wait for address
                        s_axil_wready <= 0;
                        wdata_buf <= s_axil_wdata;
                        wstrb_buf <= s_axil_wstrb;
                        // Stay in IDLE but with ready deasserted for data channel
                        wstate <= W_WAIT_W;
                    end
                end

                W_WAIT_W: begin
                    s_axil_awready <= 1;
                    s_axil_wready  <= 1;
                    if (s_axil_awvalid && s_axil_wvalid) begin
                        // Both arrived (one was already buffered, other just arrived)
                        s_axil_awready <= 0;
                        s_axil_wready  <= 0;
                        write_reg(s_axil_awaddr, s_axil_wdata, s_axil_wstrb);
                        wstate <= W_RESP;
                    end else if (s_axil_awvalid) begin
                        // Address arrived (data already buffered)
                        s_axil_awready <= 0;
                        write_reg(s_axil_awaddr, wdata_buf, wstrb_buf);
                        wstate <= W_RESP;
                    end else if (s_axil_wvalid) begin
                        // Data arrived (address already buffered)
                        s_axil_wready <= 0;
                        write_reg(waddr_buf, s_axil_wdata, s_axil_wstrb);
                        wstate <= W_RESP;
                    end
                end

                W_RESP: begin
                    s_axil_bvalid <= 1;
                    s_axil_bresp  <= 2'b00;
                    wstate <= W_RESP2;
                end

                W_RESP2: begin
                    if (s_axil_bready) begin
                        s_axil_bvalid <= 0;
                        wstate <= W_IDLE;
                    end
                end
            endcase
        end
    end

    // Write register helper task (compatible with Verilog-2001)
    task write_reg;
        input [15:0] addr;
        input [31:0] data;
        input [3:0]  strb;
        begin
            case (addr)
                REG_AP_CTRL:   reg_ap_ctrl   <= data;
                REG_GIE:       reg_gie       <= data;
                REG_IER:       reg_ier       <= data;
                REG_CTRL_USER: reg_ctrl_user <= data;
                REG_DESC_BASE: reg_desc_base <= data;
                REG_DESC_TAIL: reg_desc_tail <= data;
                REG_CHAIN_CTRL: begin
                    reg_chain_ctrl <= data;
                    if (data[1]) begin
                        reg_desc_head <= 0;
                        reg_chain_ctrl[2] <= 0;
                    end
                end
                REG_ISR: begin
                    if (data[0])
                        reg_isr[0] <= 0;
                end
            endcase
        end
    endtask

    // Combinational: wstrb_buf feeds into write_reg but not used for byte-level
    // writes in the current design (full-word only). The strb input is accepted
    // for AXI compliance but not gated in the current register file.

    // ======================================================================
    // AXI4-Lite read FSM
    // ======================================================================
    reg [1:0] rstate;
    localparam R_IDLE = 0, R_DATA = 1, R_DATA2 = 2;

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
                            REG_DESC_BASE: s_axil_rdata <= reg_desc_base;
                            REG_DESC_TAIL: s_axil_rdata <= reg_desc_tail;
                            REG_DESC_HEAD: s_axil_rdata <= reg_desc_head;
                            REG_CHAIN_CTRL:s_axil_rdata <= reg_chain_ctrl;
                            default:       s_axil_rdata <= 32'h0;
                        endcase
                    end
                end
                R_DATA: begin
                    s_axil_rvalid <= 1;
                    s_axil_rresp  <= 2'b00;
                    rstate <= R_DATA2;
                end
                R_DATA2: begin
                    if (s_axil_rready) begin
                        s_axil_rvalid <= 0;
                        rstate <= R_IDLE;
                    end
                end
            endcase
        end
    end

    // ======================================================================
    // HP Read Master — byte-stream output, one byte per data_valid cycle
    // ======================================================================
    wire        hp_read_busy;
    wire        hp_read_done;
    reg  [31:0] hp_read_addr;
    reg  [7:0]  hp_read_len;
    wire [7:0]  hp_read_data;
    wire        hp_read_valid;
    reg         hp_read_ready;
    reg         hp_read_start_raw;

    axihp_read_master u_hp_read (
        .clk         (clk),
        .rst_n       (rst_n),
        .start       (hp_read_start_raw),
        .src_addr    (hp_read_addr),
        .burst_len   (hp_read_len),
        .done        (hp_read_done),
        .busy        (hp_read_busy),
        .data_out    (hp_read_data),
        .data_valid  (hp_read_valid),
        .data_ready  (hp_read_ready),
        .m_axi_araddr (m_axi_araddr),
        .m_axi_arvalid(m_axi_arvalid),
        .m_axi_arready(m_axi_arready),
        .m_axi_arlen  (m_axi_arlen),
        .m_axi_arsize (m_axi_arsize),
        .m_axi_arburst(m_axi_arburst),
        .m_axi_rdata  (m_axi_rdata),
        .m_axi_rvalid (m_axi_rvalid),
        .m_axi_rready (m_axi_rready),
        .m_axi_rlast  (m_axi_rlast)
    );

    // hp_read_start_raw is a single-cycle pulse from the FSM — connect directly

    // ======================================================================
    // HP Write Master — 64-bit word input
    // ======================================================================
    reg         hp_write_start;
    reg  [31:0] hp_write_addr;
    reg  [15:0] hp_write_count;
    wire        hp_write_busy;
    wire        hp_write_done;
    reg  [63:0] hp_write_data;
    reg         hp_write_valid;
    wire        hp_write_ready;

    axihp_write_master u_hp_write (
        .clk          (clk),
        .rst_n        (rst_n),
        .start        (hp_write_start),
        .dst_addr     (hp_write_addr),
        .word_count   (hp_write_count),
        .busy         (hp_write_busy),
        .done         (hp_write_done),
        .wdata        (hp_write_data),
        .wvalid       (hp_write_valid),
        .wready       (hp_write_ready),
        .m_axi_awaddr (m_axi_awaddr),
        .m_axi_awvalid(m_axi_awvalid),
        .m_axi_awready(m_axi_awready),
        .m_axi_awlen  (m_axi_awlen),
        .m_axi_awsize (m_axi_awsize),
        .m_axi_awburst(m_axi_awburst),
        .m_axi_wdata  (m_axi_wdata),
        .m_axi_wvalid (m_axi_wvalid),
        .m_axi_wready (m_axi_wready),
        .m_axi_wlast  (m_axi_wlast),
        .m_axi_wstrb  (m_axi_wstrb),
        .m_axi_bvalid (m_axi_bvalid),
        .m_axi_bready (m_axi_bready),
        .m_axi_bresp  (m_axi_bresp)
    );

    // ======================================================================
    // Descriptor buffer — 32 bytes, read from DDR
    // ======================================================================
    reg [7:0]  desc_buf [0:31];
    reg [4:0]  desc_byte_idx;

    // Parsed descriptor fields (latched from desc_buf)
    reg [31:0] desc_next_addr;
    reg [31:0] desc_weight_addr;
    reg [31:0] desc_act_addr;
    reg [31:0] desc_result_addr;
    reg [15:0] desc_num_tiles;
    reg [15:0] desc_tile_bytes;
    reg [7:0]  desc_tensor_type;
    reg [7:0]  desc_tile_res_rows;
    reg [7:0]  desc_flags;
    reg [15:0] desc_act_total_bytes;
    reg [7:0]  desc_num_col_groups;

    // ======================================================================
    // Descriptor chain FSM
    // ======================================================================
    localparam PH_IDLE          = 5'd0;
    localparam PH_FETCH_DESC    = 5'd1;
    localparam PH_FETCH_WAIT    = 5'd2;
    localparam PH_LOAD_ACT      = 5'd3;
    localparam PH_LOAD_ACT_WAIT = 5'd4;
    localparam PH_LOAD_WEIGHT   = 5'd5;
    localparam PH_WEIGHT_START  = 5'd6;
    localparam PH_WEIGHT_WAIT   = 5'd7;
    localparam PH_TILE_START    = 5'd8;
    localparam PH_TILE_WAIT     = 5'd9;
    localparam PH_WRITE_ACT     = 5'd10;
    localparam PH_WRITE_RESULT  = 5'd11;
    localparam PH_WRITE_WAIT    = 5'd12;
    localparam PH_ADVANCE_DESC  = 5'd13;
    localparam PH_CPU_OP_WAIT   = 5'd14;

    reg [4:0]  ph_state;
    reg        desc_chain_busy;
    reg        desc_chain_enable;
    reg        desc_irq;

    // Tile tracking
    reg [15:0] tile_count;        // tile within current descriptor
    reg [15:0] burst_bytes_rem;   // remaining bytes in current tile's weight
    reg [31:0] burst_addr;        // current read address for multi-burst transfer
    reg [15:0] write_count;       // bytes written to weight buffer so far
    reg        weight_loading;    // in weight-loading phase (vs scale-loading phase)
    reg [1:0]  scale_byte_cnt;
    reg [15:0] scale_accum;
    reg [7:0]  scale_wr_ptr;

    // Result write
    reg        res_write_active;
    reg  [5:0] result_wr_idx;
    reg  [1:0] res_wr_state;
    reg [63:0] res_write_data_reg;
    reg  [9:0] draining_result_idx;
    reg        draining;
    reg        accumulate;

    // Act loading
    reg [15:0] act_byte_cnt;
    reg [9:0]  act_wr_ptr;

    // Misc
    reg        core_start_pulse;
    reg        hp_read_done_ff;

    // Tile config per type
    reg [13:0] tile_block_bytes;  // bytes of block data (excluding scales)
    reg [9:0]  cols_per_tile;
    reg [9:0]  act_base;

    // Tile config lookup
    always @(*) begin
        case (desc_tensor_type)
            0: begin  // INT16
                tile_block_bytes = 14'd8192;
                cols_per_tile = 10'd64;
            end
            8: begin  // Q8_0
                tile_block_bytes = 14'd4096;
                cols_per_tile = 10'd64;
            end
            12: begin  // Q4_K
                tile_block_bytes = 14'd8064;
                cols_per_tile = 10'd256;
            end
            6: begin  // Q5_0
                tile_block_bytes = 14'd4928;
                cols_per_tile = 10'd896;
            end
            14: begin  // Q6_K
                tile_block_bytes = 14'd6720;
                cols_per_tile = 10'd256;
            end
            default: begin
                tile_block_bytes = 14'd8192;
                cols_per_tile = 10'd64;
            end
        endcase
    end

    assign core_start = core_start_pulse;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ph_state          <= PH_IDLE;
            desc_chain_busy   <= 0;
            desc_chain_enable <= 0;
            desc_irq          <= 0;
            hp_read_start_raw <= 0;
            tile_count        <= 0;
            write_count       <= 0;
            weight_loading    <= 0;
            scale_byte_cnt    <= 0;
            scale_accum       <= 0;
            scale_wr_ptr      <= 0;
            core_start_pulse  <= 0;
            wt_we             <= 0;
            sc_we             <= 0;
            act_we            <= 0;
            draining          <= 0;
            draining_result_idx <= 0;
            accumulate        <= 0;
            res_write_active  <= 0;
            result_wr_idx     <= 0;
            act_wr_ptr        <= 0;
            act_byte_cnt      <= 0;
            reg_desc_head     <= 0;
            burst_bytes_rem   <= 0;
            desc_byte_idx     <= 0;
            hp_read_ready     <= 0;
            hp_write_start    <= 0;
            hp_write_valid    <= 0;
        end else begin
            // Default clears
            wt_we             <= 0;
            sc_we             <= 0;
            act_we            <= 0;
            hp_read_start_raw <= 0;
    hp_write_start    <= 0;
    core_start_pulse  <= 0;
    desc_irq          <= 0;

            case (ph_state)
                // ==========================================================
                // PH_IDLE: wait for chain enable
                // ==========================================================
                PH_IDLE: begin
                    if (reg_chain_ctrl[0]) begin
                        $display("[TB] PH_IDLE: chain started head=%0d tail=%0d",
                            reg_desc_head, reg_desc_tail);
                        desc_chain_enable <= 1;
                        reg_chain_ctrl[0] <= 0;
                        if (reg_desc_head == reg_desc_tail) begin
                            $display("[TB] PH_IDLE: no descriptors");
                        end else begin
                            desc_chain_busy <= 1;
                            ph_state <= PH_FETCH_DESC;
                        end
                    end
                end

                // ==========================================================
                // PH_FETCH_DESC: read 32-byte descriptor from DDR
                // ==========================================================
                PH_FETCH_DESC: begin
                    desc_byte_idx <= 0;
                    hp_read_addr <= reg_desc_base + reg_desc_head * 32;
                    hp_read_len  <= 8'd3;  // 4 beats = 32 bytes
                    hp_read_start_raw <= 1;
                    ph_state <= PH_FETCH_WAIT;
                end

                // ==========================================================
                // PH_FETCH_WAIT: drain descriptor bytes from HP read master
                // ==========================================================
                PH_FETCH_WAIT: begin
                    if (hp_read_valid) begin
                        hp_read_ready <= 1;
                        if (desc_byte_idx < 32) begin
                            desc_buf[desc_byte_idx] <= hp_read_data;
                            desc_byte_idx <= desc_byte_idx + 1;
                        end
                    end else begin
                        hp_read_ready <= 0;
                    end
                    if (hp_read_done) begin
                        // Descriptor fully read — parse fields
                        desc_next_addr <= {desc_buf[3], desc_buf[2], desc_buf[1], desc_buf[0]};
                        desc_weight_addr <= {desc_buf[7], desc_buf[6], desc_buf[5], desc_buf[4]};
                        desc_act_addr <= {desc_buf[11],desc_buf[10],desc_buf[9], desc_buf[8]};
                        desc_result_addr <= {desc_buf[15],desc_buf[14],desc_buf[13],desc_buf[12]};
                        desc_num_tiles <= {desc_buf[17],desc_buf[16]};
                        desc_tile_bytes <= {desc_buf[19],desc_buf[18]};
                        desc_tensor_type <= desc_buf[20];
                        desc_tile_res_rows <= desc_buf[21];
                        desc_flags <= desc_buf[22];
                        desc_act_total_bytes <= {desc_buf[24], desc_buf[23]};
                        desc_num_col_groups <= desc_buf[25];
                        tile_count <= 0;

                        $strobe("[TB] DESC %0d: next=0x%08x wt=0x%08x act=0x%08x res=0x%08x tiles=%0d tileB=%0d type=%0d rows=%0d flags=0x%02x actB=%0d colGrp=%0d",
                            reg_desc_head,
                            desc_next_addr,
                            desc_weight_addr,
                            desc_act_addr,
                            desc_result_addr,
                            desc_num_tiles,
                            desc_tile_bytes,
                            desc_tensor_type,
                            desc_tile_res_rows,
                            desc_flags,
                            desc_act_total_bytes,
                            desc_num_col_groups);

                        // Set mode from tensor type (GGML type values)
                        case (desc_tensor_type)
                            0:  reg_ctrl_user <= 32'h0000_0010;  // INT16
                            6:  reg_ctrl_user <= 32'h0000_0080;  // Q5_0
                            8:  reg_ctrl_user <= 32'h0000_0020;  // Q8_0
                            12: reg_ctrl_user <= 32'h0000_0040;  // Q4_K
                            14: reg_ctrl_user <= 32'h0000_0100;  // Q6_K
                        endcase

                        // CPU-OP (type=15): signal CPU and wait for resume
                        if (desc_tensor_type == 8'd15) begin
                            reg_chain_ctrl[2] <= 1;
                            desc_irq <= 1;
                            ph_state <= PH_CPU_OP_WAIT;
                        end else begin
                            ph_state <= PH_LOAD_ACT;
                        end
                    end
                end

                // ==========================================================
                // PH_LOAD_ACT: read activation vector from DDR → act_buf
                // ==========================================================
                PH_LOAD_ACT: begin
                    act_wr_ptr <= 0;
                    act_byte_cnt <= 0;
                    if (desc_act_total_bytes > 0) begin
                        hp_read_addr <= desc_act_addr;
                        hp_read_len  <= (desc_act_total_bytes >> 3) - 1;
                        hp_read_start_raw <= 1;
                        ph_state <= PH_LOAD_ACT_WAIT;
                    end else begin
                        ph_state <= PH_LOAD_WEIGHT;
                    end
                end

                // ==========================================================
                // PH_LOAD_ACT_WAIT: drain activation bytes from HP read master
                // ==========================================================
                PH_LOAD_ACT_WAIT: begin
                    if (hp_read_valid) begin
                        hp_read_ready <= 1;
                        // Accumulate 2 bytes into 16-bit act value
                        if (act_byte_cnt[0] == 0) begin
                            scale_accum[7:0] <= hp_read_data;
                            act_byte_cnt <= act_byte_cnt + 1;
                        end else begin
                            scale_accum[15:8] <= hp_read_data;
                            act_buf[act_wr_ptr] <= {hp_read_data, scale_accum[7:0]};
                            act_wr_ptr <= act_wr_ptr + 1;
                            act_byte_cnt <= act_byte_cnt + 1;
                        end
                    end else begin
                        hp_read_ready <= 0;
                    end
                    if (hp_read_done) begin
                        ph_state <= PH_LOAD_WEIGHT;
                    end
                end

                // ==========================================================
                // PH_LOAD_WEIGHT: start weight burst for current tile
                // ==========================================================
                PH_LOAD_WEIGHT: begin
                    burst_addr <= desc_weight_addr + tile_count * desc_tile_bytes;
                    burst_bytes_rem <= desc_tile_bytes;
                    write_count <= 0;
                    weight_loading <= 1;
                    scale_wr_ptr <= 0;
                    scale_byte_cnt <= 0;
                    ph_state <= PH_WEIGHT_START;
                end

                // ==========================================================
                // PH_WEIGHT_START: issue one HP read burst for weight data
                // ==========================================================
                PH_WEIGHT_START: begin
                    if (burst_bytes_rem > 2048) begin
                        hp_read_addr <= burst_addr;
                        hp_read_len  <= 8'd255;  // 256 beats = 2048 bytes
                        burst_addr   <= burst_addr + 2048;
                        burst_bytes_rem <= burst_bytes_rem - 2048;
                    end else begin
                        hp_read_addr <= burst_addr;
                        hp_read_len  <= (burst_bytes_rem >> 3) - 1;
                        burst_bytes_rem <= 0;
                    end
                    hp_read_start_raw <= 1;
                    hp_read_ready <= 1;
                    ph_state <= PH_WEIGHT_WAIT;
                end

                // ==========================================================
                // PH_WEIGHT_WAIT: drain weight/scale bytes from HP read master
                // ==========================================================
                PH_WEIGHT_WAIT: begin
                    if (hp_read_valid) begin
                        hp_read_ready <= 1;
                        if (weight_loading) begin
                            // Still in block data phase
                            wt_we   <= 1;
                            wt_addr <= write_count[12:0];
                            wt_din  <= hp_read_data;
                            write_count <= write_count + 1;
                            if (write_count >= tile_block_bytes - 1) begin
                                weight_loading <= 0;
                                scale_byte_cnt <= 0;
                            end
                        end else begin
                            // Scale data phase — accumulate 2 bytes per 16-bit scale
                            // Only load desc_tile_res_rows scales; skip padding bytes
                            if (scale_wr_ptr < desc_tile_res_rows) begin
                                if (scale_byte_cnt == 0) begin
                                    scale_accum[7:0] <= hp_read_data;
                                    scale_byte_cnt <= 1;
                                end else begin
                                    scale_accum[15:8] <= hp_read_data;
                                    sc_we   <= 1;
                                    sc_din  <= {hp_read_data, scale_accum[7:0]};
                                    sc_addr <= scale_wr_ptr;
                                    scale_wr_ptr <= scale_wr_ptr + 1;
                                    scale_byte_cnt <= 0;
                                end
                            end
                            // else: skip padding bytes (discard)
                        end
                    end else begin
                        hp_read_ready <= 0;
                    end
                    if (hp_read_done) begin
                        if (burst_bytes_rem > 0) begin
                            // More bursts needed for this tile
                            ph_state <= PH_WEIGHT_START;
                        end else begin
                            // All weight+scale data loaded — write activations to core
                            act_wr_ptr <= 0;
                            ph_state <= PH_WRITE_ACT;
                        end
                    end
                end

                // ==========================================================
                // PH_WRITE_ACT: write activations from act_buf to core
                // ==========================================================
                PH_WRITE_ACT: begin
                    if (act_wr_ptr < cols_per_tile) begin
                        act_we <= 1;
                        act_addr <= act_wr_ptr;
                        act_din  <= act_buf[act_wr_ptr];
                        act_wr_ptr <= act_wr_ptr + 1;
                    end else begin
                        core_start_pulse <= 1;
                        ph_state <= PH_TILE_WAIT;
                    end
                end

                // ==========================================================
                // PH_TILE_START: activate core compute
                // ==========================================================
                PH_TILE_START: begin
                    $display("[TB] PH_TILE_START: type=%0d rows=%0d %s",
                        desc_tensor_type, desc_tile_res_rows,
                        mode_q5_0 ? "Q5_0" : mode_q6_k ? "Q6_K" :
                        mode_q4k  ? "Q4_K" : mode_q8 ? "Q8_0" : "INT16");
                    core_start_pulse <= 1;
                    ph_state <= PH_TILE_WAIT;
                end

                // ==========================================================
                // PH_TILE_WAIT: wait for core done + drain results
                // ==========================================================
                PH_TILE_WAIT: begin
                    if (draining && draining_result_idx >=
                        (desc_tensor_type == 6  ? 8  :   // Q5_0
                         desc_tensor_type == 14 ? 32 :   // Q6_K
                         desc_tensor_type == 12 ? 56 :   // Q4_K
                         desc_tensor_type == 8  ? 64 :   // Q8_0
                         64)) begin
                        // Drain complete — write results in PH_WRITE_RESULT
                    ph_state <= PH_WRITE_RESULT;
                    end
                end

                // ==========================================================
                // PH_WRITE_RESULT: write result_buf to DDR via HP write master
                // ==========================================================
                PH_WRITE_RESULT: begin
                    $display("[TB] PH_WRITE_RESULT: addr=0x%08x rows=%0d tile=%0d",
                        desc_result_addr + (tile_count * desc_tile_res_rows * 8),
                        desc_tile_res_rows, tile_count);
                    hp_write_addr <= desc_result_addr + (tile_count * desc_tile_res_rows * 8);
                    hp_write_count <= desc_tile_res_rows;
                    hp_write_start <= 1;
                    result_wr_idx <= 0;
                    res_write_active <= 1;
                    res_wr_state <= 0;
                    hp_write_valid <= 0;
                    ph_state <= PH_WRITE_WAIT;
                end

                // ==========================================================
                // PH_WRITE_WAIT: stream results to HP write master
                // Sub-state machine for reliable handshake:
                //   0: present data, wait for capture
                //   1: wait for write master ready, advance to next word
                // ==========================================================
                PH_WRITE_WAIT: begin
                    if (res_write_active) begin
                        if (result_wr_idx < desc_tile_res_rows) begin
                            case (res_wr_state)
                                0: begin
                                    // Present data word
                                    res_write_data_reg <= {48'b0, result_buf[result_wr_idx]};
                                    hp_write_data <= {48'b0, result_buf[result_wr_idx]};
                                    hp_write_valid <= 1;
                                    res_wr_state <= 1;
                                end
                                1: begin
                                    // Wait for write master to accept
                                    hp_write_valid <= 1;
                                    if (hp_write_ready) begin
                                        // Master is ready — capture happens next cycle
                                        // Advance to next word
                                        hp_write_valid <= 0;
                                        result_wr_idx <= result_wr_idx + 1;
                                        res_wr_state <= 0;
                                    end
                                end
                            endcase
                        end else begin
                            hp_write_valid <= 0;
                        end
                    end else begin
                        hp_write_valid <= 0;
                    end
                    if (hp_write_done) begin
                        $display("[TB] PH_WRITE_WAIT: done tile=%0d", tile_count);
                        res_write_active <= 0;
                        hp_write_valid <= 0;
                        tile_count <= tile_count + 1;
                        if (tile_count + 1 >= desc_num_tiles) begin
                            ph_state <= PH_ADVANCE_DESC;
                        end else begin
                            ph_state <= PH_LOAD_WEIGHT;
                        end
                    end
                end

                // ==========================================================
                // PH_ADVANCE_DESC: advance descriptor chain head
                // ==========================================================
                PH_ADVANCE_DESC: begin
                    reg_desc_head <= reg_desc_head + 1;
                    if (reg_desc_head + 1 >= reg_desc_tail) begin
                        // Chain complete
                        reg_chain_ctrl[2] <= 1;
                        desc_chain_busy <= 0;
                        desc_irq <= 1;
                        ph_state <= PH_IDLE;
                    end else begin
                        ph_state <= PH_FETCH_DESC;
                    end
                end

                // ==========================================================
                // PH_CPU_OP_WAIT: pause until CPU resumes via CHAIN_CTRL[0]
                // ==========================================================
                PH_CPU_OP_WAIT: begin
                    if (reg_chain_ctrl[0]) begin
                        reg_chain_ctrl[0] <= 0;
                        reg_chain_ctrl[2] <= 0;
                        desc_irq <= 0;
                        ph_state <= PH_ADVANCE_DESC;
                    end
                end

                default: ph_state <= PH_IDLE;
            endcase
        end
    end

    // ======================================================================
    // Result drain (from core → result_buf)
    // ======================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            draining   <= 0;
            draining_result_idx <= 0;
            accumulate <= 0;
        end else if (core_done && !draining) begin
            draining   <= 1;
            draining_result_idx <= 0;
        end else if (draining) begin
            if ((mode_q5_0 && draining_result_idx < 8) ||
                (mode_q6_k && draining_result_idx < 32) ||
                (mode_q4k && draining_result_idx < 56) ||
                (!mode_q5_0 && !mode_q6_k && !mode_q4k && draining_result_idx < 64)) begin
                result_buf[draining_result_idx] <= res_dout;
                draining_result_idx <= draining_result_idx + 1;
            end else begin
                draining <= 0;
            end
        end
    end

    assign res_addr = draining ? draining_result_idx[5:0] : 0;

    // ======================================================================
    // STATUS register
    // ======================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            reg_status <= 0;
        else if (desc_chain_busy)
            reg_status <= 3;
        else if (core_busy)
            reg_status <= 1;
        else if (core_done && !draining)
            reg_status <= 2;
        else
            reg_status <= 0;
    end

    // ======================================================================
    // AP_CTRL done/idle/ready bits
    // ======================================================================
    always @(*) begin
        reg_ap_ctrl[3:1] = {~desc_chain_busy && ~core_busy,
                            ~desc_chain_busy && ~core_busy,
                            core_done && !draining};
    end

    // ======================================================================
    // Interrupt generation (chain completion)
    // ======================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            reg_isr <= 0;
        else if (desc_irq && reg_gie[0] && reg_ier[0])
            reg_isr[0] <= 1;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            interrupt <= 0;
        else
            interrupt <= reg_gie[0] && reg_ier[0] && reg_isr[0];
    end

endmodule
