`timescale 1ns / 1ps
module hp_fsm_top (
    input  wire         clk, rst_n,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWADDR" *)
    input  wire [15:0]  s_axil_awaddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWVALID" *)
    input  wire         s_axil_awvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWREADY" *)
    output wire         s_axil_awready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WDATA" *)
    input  wire [31:0]  s_axil_wdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WSTRB" *)
    input  wire [3:0]   s_axil_wstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WVALID" *)
    input  wire         s_axil_wvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WREADY" *)
    output wire         s_axil_wready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BRESP" *)
    output wire [1:0]   s_axil_bresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BVALID" *)
    output wire         s_axil_bvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BREADY" *)
    input  wire         s_axil_bready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARADDR" *)
    input  wire [15:0]  s_axil_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARVALID" *)
    input  wire         s_axil_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARREADY" *)
    output reg          s_axil_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RDATA" *)
    output reg  [31:0]  s_axil_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RRESP" *)
    output reg  [1:0]   s_axil_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RVALID" *)
    output reg          s_axil_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RREADY" *)
    input  wire         s_axil_rready,
    // AXI HP master
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME M_AXI_HP, PROTOCOL AXI3, ID_WIDTH 6, DATA_WIDTH 64" *)
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP AWID" *)
    output wire [5:0]   m_axi_awid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP AWADDR" *)
    output wire [31:0]  m_axi_awaddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP AWVALID" *)
    output wire         m_axi_awvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP AWREADY" *)
    input  wire         m_axi_awready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP AWLEN" *)
    output wire [7:0]   m_axi_awlen,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP AWSIZE" *)
    output wire [2:0]   m_axi_awsize,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP AWBURST" *)
    output wire [1:0]   m_axi_awburst,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP AWLOCK" *)
    output wire [1:0]   m_axi_awlock,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP AWCACHE" *)
    output wire [3:0]   m_axi_awcache,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP AWPROT" *)
    output wire [2:0]   m_axi_awprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP WDATA" *)
    output wire [63:0]  m_axi_wdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP WVALID" *)
    output wire         m_axi_wvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP WREADY" *)
    input  wire         m_axi_wready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP WLAST" *)
    output wire         m_axi_wlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP WSTRB" *)
    output wire [7:0]   m_axi_wstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP WID" *)
    output wire [5:0]   m_axi_wid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP BVALID" *)
    input  wire         m_axi_bvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP BREADY" *)
    output wire         m_axi_bready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP BRESP" *)
    input  wire  [1:0]  m_axi_bresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP BID" *)
    input  wire  [5:0]  m_axi_bid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP ARID" *)
    output wire [5:0]   m_axi_arid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP ARADDR" *)
    output wire [31:0]  m_axi_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP ARVALID" *)
    output wire         m_axi_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP ARREADY" *)
    input  wire         m_axi_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP ARLEN" *)
    output wire [7:0]   m_axi_arlen,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP ARSIZE" *)
    output wire [2:0]   m_axi_arsize,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP ARBURST" *)
    output wire [1:0]   m_axi_arburst,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP ARLOCK" *)
    output wire [1:0]   m_axi_arlock,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP ARCACHE" *)
    output wire [3:0]   m_axi_arcache,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP ARPROT" *)
    output wire [2:0]   m_axi_arprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP RDATA" *)
    input  wire [63:0]  m_axi_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP RRESP" *)
    input  wire [1:0]   m_axi_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP RID" *)
    input  wire [5:0]   m_axi_rid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP RVALID" *)
    input  wire         m_axi_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP RREADY" *)
    output wire         m_axi_rready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP RLAST" *)
    input  wire         m_axi_rlast
);

    // ===== Registers =====
    reg        reg_start;          // 0x00[0]: start chain
    reg [31:0] reg_status;         // 0x14: [8]=rd_done, [9]=wr_done, [15]=busy
    reg [31:0] reg_desc_base;      // 0x18: descriptor base address
    reg [31:0] reg_desc_tail;      // 0x1C: tail index
    reg [31:0] reg_desc_head;      // 0x20: head index (read-only)
    reg [31:0] reg_debug;          // 0x28: debug
    reg [31:0] reg_clk_cnt;        // 0x2C: free-running clock cycle counter
    reg [31:0] reg_clk_cnt_slow;   // 0x30: clock counter divided by 1024
    reg [31:0] reg_act_info;       // 0x34: act_addr from descriptor
    reg [31:0] reg_desc_info;      // 0x38: {8'h0, act_total_bytes[23:0]}
    reg [31:0] reg_q8_debug;       // 0x3C: Q8 core debug status
    reg [31:0] reg_q5_debug;       // 0x40: Q5 core debug status
    reg [31:0] reg_q5_dbg_cap0;    // 0x44: Q5 debug capture 0 (core0/1 d_pre)
    reg [31:0] reg_q5_dbg_cap1;    // 0x48: Q5 debug capture 1 (d_fp, norm, blk_counter)
    reg [31:0] reg_q5_dbg_trig;    // 0x4C: Q5 debug trigger (W: arm, R: status)
    reg [31:0] reg_q5_dbg_live;    // 0x50: Q5 live debug (always updated)
    reg [31:0] reg_q5_dbg_cap2;    // 0x54: Q5 acc capture 2 (core1.acc[1][31:0])
    reg [31:0] reg_q5_dbg_cap3;    // 0x58: Q5 acc capture 3 (core1.acc[0] lo, core1 state)
    reg [31:0] reg_q5_dbg_cap4;    // 0x5C: Q5 debug capture 4 (core0_act_r[15:0], core0_q5[4:0])
    reg [31:0] reg_q5_dbg_cap5;    // 0x60: Q5 debug capture 5 (core1_act_r[15:0], core1_q5[4:0])
    reg [31:0] reg_q5_dbg_snap;    // 0x64: Q5 snapshot on CPU write
    reg [31:0] reg_q5_dbg_wi_start; // 0x68: wi captured on first blk_entry after clr_acc
    reg        q5_wi_arm;           // armed after clr_acc deasserts, captures wi on next blk_entry
    reg        q5_clr_acc_d;        // delayed clr_acc for edge detection
    reg        q5_blk_valid_r;      // delayed blk_valid for edge detection
    reg  [3:0] reg_q8_num_groups;  // 0x10[3:0]: number of column groups (0=single)
    // Snapshot trigger: combinatorial from AXI4-Lite write decode
    wire q5_snap_trig = aw_got && w_got && !bvalid_r && awaddr_r[15:0] == 16'h64;

    // ===== HP Read Master (64-bit word output) =====
    wire       rd_busy, rd_done, rd_valid;
    wire [63:0] rd_data;
    wire [2:0] rd_dbg_state;
    wire [7:0] rd_beat_cnt;
    reg        rd_start, rd_ready;
    reg        q5_start_pending;   // Q5: hold rd_start request across default clear
    reg [31:0] rd_addr;
    reg [7:0]  rd_len;

    // Byte-unpack shift registers for scale and Q5_0 data loads
    // (Q8 weights use direct 64-bit word write in LOAD_WEIGHT_W)
    reg [63:0] rd_unpack_buf;
    reg [2:0]  rd_unpack_cnt;
    reg        rd_unpack_active;
    reg        wt_burst_done;   // weight/Q8 load: rd_done captured (1-cycle delay for burst completion)
    reg        sc_burst_done;   // scale load: rd_done captured during byte unpack

    axihp_read_master u_rd (
        .clk(clk), .rst_n(rst_n),
        .start(rd_start), .src_addr(rd_addr),
        .burst_len(rd_len),
        .done(rd_done), .busy(rd_busy),
        .dbg_state(rd_dbg_state),
        .dbg_beat_cnt(rd_beat_cnt),
        .rdata(rd_data), .rvalid(rd_valid), .rready(rd_ready),
        .m_axi_arid(m_axi_arid), .m_axi_araddr(m_axi_araddr),
        .m_axi_arvalid(m_axi_arvalid), .m_axi_arready(m_axi_arready),
        .m_axi_arlen(m_axi_arlen), .m_axi_arsize(m_axi_arsize),
        .m_axi_arburst(m_axi_arburst), .m_axi_arlock(m_axi_arlock),
        .m_axi_arcache(m_axi_arcache), .m_axi_arprot(m_axi_arprot),
        .m_axi_rdata(m_axi_rdata), .m_axi_rresp(m_axi_rresp),
        .m_axi_rid(m_axi_rid), .m_axi_rvalid(m_axi_rvalid),
        .m_axi_rready(m_axi_rready), .m_axi_rlast(m_axi_rlast)
    );

    // ===== HP Write Master =====
    wire       wr_busy, wr_done, wr_wready;
    wire [2:0] wr_dbg_state;
    reg        wr_start;
    wire [63:0] wr_wdata;
    reg         wr_wvalid;
    reg [31:0]  wr_addr;
    reg [15:0]  wr_word_cnt;

    axihp_write_master u_wr (
        .clk(clk), .rst_n(rst_n),
        .start(wr_start), .dst_addr(wr_addr),
        .word_count(wr_word_cnt),
        .busy(wr_busy), .done(wr_done),
        .dbg_state(wr_dbg_state),
        .wdata(wr_wdata), .wvalid(wr_wvalid), .wready(wr_wready),
        .m_axi_awid(m_axi_awid), .m_axi_awaddr(m_axi_awaddr),
        .m_axi_awvalid(m_axi_awvalid), .m_axi_awready(m_axi_awready),
        .m_axi_awlen(m_axi_awlen), .m_axi_awsize(m_axi_awsize),
        .m_axi_awburst(m_axi_awburst), .m_axi_awlock(m_axi_awlock),
        .m_axi_awcache(m_axi_awcache), .m_axi_awprot(m_axi_awprot),
        .m_axi_wid(m_axi_wid), .m_axi_wdata(m_axi_wdata),
        .m_axi_wvalid(m_axi_wvalid), .m_axi_wready(m_axi_wready),
        .m_axi_wlast(m_axi_wlast), .m_axi_wstrb(m_axi_wstrb),
        .m_axi_bvalid(m_axi_bvalid), .m_axi_bready(m_axi_bready),
        .m_axi_bresp(m_axi_bresp), .m_axi_bid(m_axi_bid)
    );

    // ===== Q8 Compute Core =====
    wire       q8_busy, q8_done;
    wire [47:0] q8_res_dout;
    wire [2:0]  q8_core_state;
    wire [5:0]  q8_core_k;
    wire [2:0]  q8_core_g;

    reg        q8_start;
    reg        q8_wt_we;
    reg [8:0]  q8_wt_addr;
    reg [63:0] q8_wt_din;
    reg        q8_sc_we;
    reg [6:0]  q8_sc_addr;
    reg [15:0] q8_sc_din;
    reg        q8_act_we;
    reg [5:0]  q8_act_addr;
    reg [15:0] q8_act_din;
    reg [5:0]  q8_res_addr;

    matmul_q8_core u_q8 (
        .clk(clk), .rst_n(rst_n),
        .start(q8_start), .op_vecmul(1'b1),
        .done(q8_done), .busy(q8_busy),
        .wt_we(q8_wt_we), .wt_addr(q8_wt_addr), .wt_din(q8_wt_din),
        .sc_we(q8_sc_we), .sc_addr(q8_sc_addr), .sc_din(q8_sc_din),
        .act_we(q8_act_we), .act_addr(q8_act_addr), .act_din(q8_act_din),
        .res_addr(q8_res_addr), .res_dout(q8_res_dout),
        .dbg_state(q8_core_state), .dbg_k(q8_core_k), .dbg_g(q8_core_g)
    );

    // ===== Q5_0 Compute Cores (2 parallel, each handles 2 of 4 rows) =====
    wire       q5_done0, q5_done1;
    wire       q5_busy0, q5_busy1;
    reg        q5_clr_acc;
    reg  [4:0] q5_clr_acc_cnt;
    reg        q5_norm_we;
    reg [1:0]  q5_norm_addr;
    reg [15:0] q5_norm_din;
    reg [15:0] q5_blk_d0;
    reg [31:0] q5_blk_qh0;
    reg [127:0] q5_blk_qs0;
    reg [15:0] q5_blk_d1;
    reg [31:0] q5_blk_qh1;
    reg [127:0] q5_blk_qs1;
    reg        q5_blk_valid;
    reg        q5_blk_valid_pulsed;
    reg        q5_act_we;
    reg [9:0]  q5_act_addr;
    reg [15:0] q5_act_din;
    reg [5:0]  q5_blk_counter;      // block counter (0..55)
    reg [15:0] q5_num_tiles;        // total tiles per descriptor
    reg [15:0] q5_tile_counter;     // current tile index

    reg [15:0] q8_num_tiles;        // total tiles per Q8 descriptor
    reg [15:0] q8_tile_counter;     // current Q8 tile index
    reg [15:0] q8_tile_stride;      // bytes per Q8 tile

    // Q5 debug wires from cores
    wire [15:0] q5_core0_d_pre,  q5_core1_d_pre;
    wire [5:0]  q5_core0_blk_cnt, q5_core1_blk_cnt;
    wire [31:0] q5_core0_d_fp,   q5_core1_d_fp;
    wire [15:0] q5_core0_norm,   q5_core1_norm;
    wire [2:0]  q5_core0_state,  q5_core1_state;
    wire [15:0] q5_core0_act_r,  q5_core1_act_r;
    wire [4:0]  q5_core0_q5,     q5_core1_q5;
    wire [4:0]  q5_core0_wi,     q5_core1_wi;
    wire [47:0] q5_core0_res0, q5_core0_res1, q5_core1_res0, q5_core1_res1;

    matmul_q5_0_core u_q5_core0 (
        .clk(clk), .rst_n(rst_n),
        .core_id(1'b0),
        .done(q5_done0), .busy(q5_busy0),
        .norm_we(q5_norm_we), .norm_addr(q5_norm_addr), .norm_din(q5_norm_din),
        .blk_d(q5_blk_d0), .blk_qh(q5_blk_qh0), .blk_qs(q5_blk_qs0),
        .blk_valid(q5_blk_valid),
        .act_we(q5_act_we), .act_addr(q5_act_addr), .act_din(q5_act_din),
        .clr_acc(q5_clr_acc),
        .res0(q5_core0_res0), .res1(q5_core0_res1),
        .dbg_d_pre(q5_core0_d_pre), .dbg_blk_counter(q5_core0_blk_cnt),
        .dbg_d_fp(q5_core0_d_fp), .dbg_norm(q5_core0_norm),
        .dbg_state(q5_core0_state),
        .dbg_act_r(q5_core0_act_r), .dbg_q5(q5_core0_q5),
        .dbg_wi(q5_core0_wi)
    );
    matmul_q5_0_core u_q5_core1 (
        .clk(clk), .rst_n(rst_n),
        .core_id(1'b1),
        .done(q5_done1), .busy(q5_busy1),
        .norm_we(q5_norm_we), .norm_addr(q5_norm_addr), .norm_din(q5_norm_din),
        .blk_d(q5_blk_d1), .blk_qh(q5_blk_qh1), .blk_qs(q5_blk_qs1),
        .blk_valid(q5_blk_valid),
        .act_we(q5_act_we), .act_addr(q5_act_addr), .act_din(q5_act_din),
        .clr_acc(q5_clr_acc),
        .res0(q5_core1_res0), .res1(q5_core1_res1),
        .dbg_d_pre(q5_core1_d_pre), .dbg_blk_counter(q5_core1_blk_cnt),
        .dbg_d_fp(q5_core1_d_fp), .dbg_norm(q5_core1_norm),
        .dbg_state(q5_core1_state),
        .dbg_act_r(q5_core1_act_r), .dbg_q5(q5_core1_q5),
        .dbg_wi(q5_core1_wi)
    );

    wire q5_all_done  = q5_done0 & q5_done1;
    wire q5_any_busy  = q5_busy0 | q5_busy1;
    wire q5_busy_both = q5_busy0 & q5_busy1;
    reg q5_done_d;
    wire q5_done_rise;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) q5_done_d <= 0;
        else q5_done_d <= q5_all_done;
    end
    assign q5_done_rise = q5_all_done && !q5_done_d;

    // ── Q5 debug capture — continuous track with trigger freeze ──
    // reg_q5_dbg_cap0 (0x44): [31:16]=core0_d_pre, [15:0]=core1_d_pre
    // reg_q5_dbg_cap1 (0x48): [31:16]=core0_d_fp[15:0], [15:9]=norm[15:9],
    //                         [8:6]=core0_state, [5:0]=blk_counter
    // reg_q5_dbg_trig (0x4C): W:arm, R:live core0_dpre[31:16]+status
    reg q5_dbg_frozen;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            reg_q5_dbg_cap0 <= 0;
            reg_q5_dbg_cap1 <= 0;
            reg_q5_dbg_live <= 0;
            reg_q5_dbg_cap4 <= 0;
            reg_q5_dbg_cap5 <= 0;
            reg_q5_dbg_snap <= 0;
            q5_dbg_frozen   <= 0;
        end else begin
            // Always update live debug (unconditional, every cycle)
            reg_q5_dbg_live <= {5'b0,                          // [31:27] reserved
                                state,                          // [26:22] FSM state (5 bits)
                                q5_dbg_frozen,                  // [21] frozen
                                q5_any_busy,                    // [20] busy
                                q5_tile_counter[5:0],           // [19:14] tile counter (6 bits)
                                q5_blk_counter[5:0],            // [13:8]  block counter (6 bits)
                                3'b0,                           // [7:5]  reserved
                                q5_core0_state};                // [4:2]  core0 state (3 bits)
            if (q5_snap_trig) begin
                // Snapshot on CPU write to 0x64
                reg_q5_dbg_snap <= {q5_core0_wi, q5_core1_wi,
                                    q5_core0_q5[4:0], q5_core1_q5[4:0],
                                    q5_core0_state, q5_core1_state,
                                    q5_blk_counter[5:0]};
                q5_dbg_frozen <= 1;
            end else if (q5_dbg_frozen) begin
                // Hold: keep last captured values (do nothing)
            end else begin
                // Continuous track: update with live values every cycle
                reg_q5_dbg_cap0 <= {q5_core0_d_pre[15:0], q5_core1_d_pre[15:0]};
                reg_q5_dbg_cap1 <= {q5_core0_d_fp[15:0], q5_core0_norm[15:9],
                                    q5_core0_state, q5_blk_counter[5:0]};
                reg_q5_dbg_cap4 <= {q5_core0_q5[4:0], 11'b0, q5_core0_act_r[15:0]};
                reg_q5_dbg_cap5 <= {q5_core1_q5[4:0], 11'b0, q5_core1_act_r[15:0]};
                // Freeze on trigger match (CPU clears armed bit manually via write to 0x4C)
                if (reg_q5_dbg_trig[0] && q5_core0_blk_cnt == reg_q5_dbg_trig[15:10]) begin
                    q5_dbg_frozen <= 1;
                end
            end
            // Re-arm (unfreeze) when CPU writes to trig register
            if (aw_got && w_got && awaddr_r[15:0] == 16'h4C) begin
                q5_dbg_frozen <= 0;
            end
        end
    end

    // ── wi_start capture: captures wi on first blk_entry after clr_acc ──
    // reg_q5_dbg_wi_start (0x68): [31:27]=core0_wi, [26:22]=core1_wi,
    //                             [21:19]=core0_state, [18:16]=core1_state,
    //                             [15:10]=reserved, [5:0]=blk_counter
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            q5_clr_acc_d     <= 0;
            q5_wi_arm        <= 0;
            reg_q5_dbg_wi_start <= 0;
            q5_blk_valid_r   <= 0;
        end else begin
            q5_blk_valid_r   <= q5_blk_valid;
            q5_clr_acc_d     <= q5_clr_acc;
            // Falling edge of clr_acc: arm wi capture
            if (!q5_clr_acc && q5_clr_acc_d) begin
                q5_wi_arm <= 1;
            end
            // On blk_valid rising edge, if armed: capture wi (before blk_entry resets it)
            if (q5_blk_valid && !q5_blk_valid_r && q5_wi_arm) begin
                reg_q5_dbg_wi_start <= {q5_core0_wi, q5_core1_wi,
                                        3'b0, 3'b0,
                                        10'b0, q5_blk_counter[5:0]};
                q5_wi_arm <= 0;
            end
        end
    end

    // ===== Compute type: 0=Q8, 1=Q5_0 =====
    reg [1:0] compute_type;

    // ===== Q5_0 helper regs =====
    reg [7:0]  q5_unpack_cnt;       // counter for packing rd_data words (0..5)
    reg [9:0]  q5_copy_act_idx;     // activation copy index (0..895)
    reg [3:0]  q5_unpack_word;      // which 64-bit rd_data word being captured (0..5)

    // ===== Buffer =====
    reg [7:0] desc_buf [0:31];    // 32-byte descriptor
    reg [63:0] act_buf [0:63];    // 64 x 64-bit = 512 bytes result buffer (Q8: 64 rows x 8 bytes)
    reg [31:0] desc_addr;         // current descriptor DDR address
    reg [31:0] next_addr;
    reg [31:0] weight_addr;
    reg [31:0] act_addr;
    reg [31:0] result_addr;
    reg [15:0] tensor_type;       // descriptor type (15=CPU_OP)
    reg  [3:0] q8_num_groups;     // column groups from descriptor (offset 20 bits 3:0)
    reg [23:0] act_total_bytes;
    reg [23:0] act_remaining;     // bytes left to read (multi-burst tracking)

    // Q8 core helper regs
    reg [5:0]  q8_res_idx;        // result read index (0..63)
    reg [15:0] wt_byte_idx;       // weight byte counter (0..4095)
    reg [15:0] wt_remaining;      // weight bytes remaining
    reg [8:0]  sc_byte_idx;       // scale byte counter (0..255)
    reg [8:0]  sc_remaining;      // scale bytes remaining (multi-burst)
    reg [7:0]  sc_din_lo;         // captured low byte for scale packing
    reg [5:0]  copy_act_idx;      // activation copy index (0..63)

    // Write-side burst tracking
    reg [23:0] wr_remaining;      // total bytes left to write across all bursts
    reg [31:0] wr_burst_addr;     // DDR address for next burst
    reg [8:0]  wr_burst_bytes;    // bytes in current burst (max 512)
    reg [8:0]  wr_byte_offset;    // cumulative byte offset into act_buf

    // Multi-group accumulator
    reg signed [47:0] acc_buf [0:63];  // running accumulator across column groups
    reg [3:0]  col_group;              // current column group (0..13)
    reg [5:0]  copy_acc_idx;           // acc_buf ? act_buf copy index
    reg [15:0] timeout_cnt;            // shared timeout counter for all wait states
    reg [4:0]  timeout_src;            // state that triggered timeout (latched at timeout)

    // ===== FSM =====
    localparam IDLE           = 5'd0;
    localparam FETCH_DESC     = 5'd1;
    localparam FETCH_DESC_W   = 5'd2;
    localparam LOAD_ACT       = 5'd3;
    localparam LOAD_ACT_W     = 5'd4;
    localparam WRITE_RES      = 5'd5;
    localparam WRITE_RES_W    = 5'd6;
    localparam DONE           = 5'd7;
    localparam LOAD_WEIGHT    = 5'd8;
    localparam LOAD_WEIGHT_W  = 5'd9;
    localparam LOAD_SCALES    = 5'd10;
    localparam LOAD_SCALES_W  = 5'd11;
    localparam COPY_ACT_TO_CORE = 5'd12;
    localparam COMPUTE        = 5'd13;
    localparam COMPUTE_W      = 5'd14;
    localparam READ_RES       = 5'd15;
    localparam READ_RES_ACC   = 5'd16;
    localparam COPY_ACC_TO_BUF = 5'd17;
    localparam TIMEOUT_ERROR  = 5'd18;
    localparam WRITE_RES_BURST  = 5'd19;
    // Q5_0 compute path states (per-block design)
    localparam Q5_LOAD_NORM     = 5'd20;
    localparam Q5_LOAD_NORM_W   = 5'd21;
    localparam Q5_COPY_ACT      = 5'd22;
    localparam Q5_COPY_ACT_W    = 5'd23;
    localparam Q5_BLOCK_COMPUTE = 5'd24;
    localparam Q5_BLOCK_COMPUTE_W = 5'd25;
    localparam Q5_READ_RES      = 5'd26;

    // Multi-group flag: non-zero when q8_num_groups > 1 (from descriptor, fallback to GP0 reg)
    wire multi_group = (q8_num_groups > 1);

    reg [4:0] state;
    reg [8:0] byte_idx;             // write byte index (0..511)
    reg [7:0] desc_byte_idx;
    reg [8:0] act_byte_idx;

    // Edge detection for done signals
    reg rd_done_d, wr_done_d, q8_done_d;
    wire rd_done_rise, wr_done_rise, q8_done_rise;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin rd_done_d <= 0; wr_done_d <= 0; q8_done_d <= 0; end
        else begin rd_done_d <= rd_done; wr_done_d <= wr_done; q8_done_d <= q8_done; end
    end
    assign rd_done_rise = rd_done && !rd_done_d;
    assign wr_done_rise = wr_done && !wr_done_d;

    // Rising-edge detection for reg_start (to avoid multi-driver)
    reg reg_start_d;
    wire reg_start_rise;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) reg_start_d <= 0;
        else reg_start_d <= reg_start;
    end
    assign reg_start_rise = reg_start && !reg_start_d;
    assign wr_wdata = act_buf[wr_byte_offset[8:3]];
    assign q8_done_rise = q8_done && !q8_done_d;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            reg_status <= 0;
            reg_desc_head <= 0;
            reg_debug <= 0;
            reg_q8_debug <= 0;
            reg_q5_debug <= 0;
            reg_q5_dbg_trig <= 0;
            rd_start <= 0; rd_ready <= 0;
            rd_addr <= 0; rd_len <= 0;
            wr_start <= 0; wr_wvalid <= 0;
            wr_addr <= 0; wr_word_cnt <= 0;
            byte_idx <= 0;
            desc_byte_idx <= 0;
            act_byte_idx <= 0;
            desc_addr <= 0;
            next_addr <= 0;
            weight_addr <= 0;
            act_addr <= 0;
            result_addr <= 0;
            tensor_type <= 0;
            act_total_bytes <= 0;
            act_remaining <= 0;
            reg_clk_cnt <= 0;
            reg_clk_cnt_slow <= 0;
            reg_act_info <= 0;
            reg_desc_info <= 0;
            q8_res_idx <= 0;
            wt_byte_idx <= 0;
            wt_remaining <= 0;
            sc_byte_idx <= 0;
            sc_remaining <= 0;
            sc_din_lo <= 0;
            copy_act_idx <= 0;
            q8_start <= 0;
            q8_wt_we <= 0;
            q8_wt_addr <= 0;
            q8_wt_din <= 0;
            q8_sc_we <= 0;
            q8_sc_addr <= 0;
            q8_sc_din <= 0;
            q8_act_we <= 0;
            q8_act_addr <= 0;
            q8_act_din <= 0;
            q8_res_addr <= 0;
            col_group <= 0;
            copy_acc_idx <= 0;
            wr_remaining <= 0;
            wr_burst_addr <= 0;
            wr_burst_bytes <= 0;
            timeout_cnt <= 0;
            timeout_src <= 0;
            rd_unpack_buf   <= 0;
            rd_unpack_cnt   <= 0;
            rd_unpack_active <= 0;
            wt_burst_done   <= 0;
            sc_burst_done   <= 0;
            q5_clr_acc      <= 0;
            q5_clr_acc_cnt  <= 0;
            q5_norm_we      <= 0; q5_norm_addr <= 0; q5_norm_din <= 0;
            q5_blk_d0       <= 16'd0; q5_blk_qh0 <= 32'd0; q5_blk_qs0 <= 128'd0;
            q5_blk_d1       <= 16'd0; q5_blk_qh1 <= 32'd0; q5_blk_qs1 <= 128'd0;
            q5_blk_valid    <= 0;
            q5_blk_valid_pulsed <= 0;
            q5_act_we       <= 0; q5_act_addr <= 0; q5_act_din <= 0;
            q5_blk_counter  <= 0;
            q5_unpack_cnt   <= 0;
            q5_copy_act_idx <= 0;
            q5_unpack_word  <= 0;
            reg_q5_dbg_cap2 <= 0;
            reg_q5_dbg_cap3 <= 0;
            q8_num_tiles    <= 0;
            q8_tile_counter <= 0;
            q8_tile_stride  <= 0;
            compute_type    <= 0;
        end else begin
            reg_clk_cnt <= reg_clk_cnt + 1;
            reg_clk_cnt_slow <= reg_clk_cnt_slow + (|reg_clk_cnt[9:0] ? 0 : 1);
            rd_start <= 0; rd_ready <= 0;
            wr_start <= 0; wr_wvalid <= 0;
            reg_status[15] <= 1'b1;  // default busy
            // Default-off for Q8 core control signals
            q8_start <= 0;
            q8_wt_we <= 0;
            q8_sc_we <= 0;
            q8_act_we <= 0;
            // Default-off for Q5_0 core control signals
            q5_norm_we <= 0;
            q5_blk_valid <= 0;
            q5_act_we <= 0;

            case (state)
                IDLE: begin
                    reg_status[15:8] <= 0;
                    if (reg_start_rise) begin
                        reg_desc_head <= 0;
                        desc_addr <= reg_desc_base;
                        state <= FETCH_DESC;
                    end
                end

                FETCH_DESC: begin
                    desc_byte_idx <= 0;
                    rd_addr <= desc_addr;
                    rd_len  <= 8'd7;
                    rd_start <= 1;
                    state <= FETCH_DESC_W;
                end

                FETCH_DESC_W: begin
                    rd_ready <= rd_valid;
                    if (rd_valid && rd_ready) begin
                        desc_buf[desc_byte_idx]   <= rd_data[7:0];
                        desc_buf[desc_byte_idx+1] <= rd_data[15:8];
                        desc_buf[desc_byte_idx+2] <= rd_data[23:16];
                        desc_buf[desc_byte_idx+3] <= rd_data[31:24];
                        desc_buf[desc_byte_idx+4] <= rd_data[39:32];
                        desc_buf[desc_byte_idx+5] <= rd_data[47:40];
                        desc_buf[desc_byte_idx+6] <= rd_data[55:48];
                        desc_buf[desc_byte_idx+7] <= rd_data[63:56];
                        desc_byte_idx <= desc_byte_idx + 8;
                    end
                    if (rd_done_rise) begin
                        timeout_cnt <= 0;
                        next_addr    <= {desc_buf[3], desc_buf[2], desc_buf[1], desc_buf[0]};
                        weight_addr  <= {desc_buf[7], desc_buf[6], desc_buf[5], desc_buf[4]};
                        act_addr     <= {desc_buf[11],desc_buf[10],desc_buf[9], desc_buf[8]};
                        result_addr  <= {desc_buf[15],desc_buf[14],desc_buf[13],desc_buf[12]};
                        tensor_type  <= {desc_buf[17], desc_buf[16]};
                        act_total_bytes <= {desc_buf[26], desc_buf[25], desc_buf[24]};
                        act_remaining  <= {desc_buf[26], desc_buf[25], desc_buf[24]};
                        reg_act_info  <= {desc_buf[11],desc_buf[10],desc_buf[9], desc_buf[8]};
                        reg_desc_info <= {8'h0, desc_buf[26], desc_buf[25], desc_buf[24]};
                        // num_groups from descriptor byte offset 20, fallback to GP0 register
                        q8_num_groups <= (desc_buf[20][3:0] != 0) ? desc_buf[20][3:0] : reg_q8_num_groups;
                        act_byte_idx   <= 0;
                        byte_idx       <= 0;
                        // Branch by descriptor type
                        // Use desc_buf directly (tensor_type register lags by 1 cycle)
                        if ({desc_buf[17], desc_buf[16]} == 15) begin
                            // CPU_OP: passthrough activation to DDR
                            col_group <= 0;
                            state <= LOAD_ACT;
                        end else if ({desc_buf[17], desc_buf[16]} == 1) begin
                            // Q5_0 compute path (per-block, tile iteration)
                            compute_type <= 1;
                            q5_num_tiles <= ({desc_buf[23], desc_buf[22]} == 0) ? 16'd1 : {desc_buf[23], desc_buf[22]};
                            q5_tile_counter <= 0;
                            q5_blk_counter <= 0;
                            state <= Q5_LOAD_NORM;
                        end else begin
                            // Q8 compute path (default)
                            compute_type <= 0;
                            q8_num_tiles <= ({desc_buf[23], desc_buf[22]} == 0) ? 16'd1 : {desc_buf[23], desc_buf[22]};
                            q8_tile_counter <= 0;
                            q8_tile_stride <= 4096 + ((desc_buf[20][3:0] != 0) ? desc_buf[20][3:0] : reg_q8_num_groups) * 256;
                            wt_byte_idx   <= 0;
                            wt_remaining  <= 4096;
                            state <= LOAD_WEIGHT;
                        end
                    end else if (&timeout_cnt) begin
                        timeout_cnt <= 0;
                        timeout_src <= state;
                        state <= TIMEOUT_ERROR;
                    end else begin
                        timeout_cnt <= timeout_cnt + 1;
                    end
                end

                // === Load activation data from DDR (multi-burst) ===
                LOAD_ACT: begin
                    reg_status[8] <= 0;
                    if (act_remaining > 64) begin
                        rd_addr <= act_addr + col_group * 128 + (act_total_bytes - act_remaining);
                        rd_start <= 1;
                        rd_len <= 8'd15;
                        state <= LOAD_ACT_W;
                    end else if (act_remaining >= 4) begin
                        rd_addr <= act_addr + col_group * 128 + (act_total_bytes - act_remaining);
                        rd_start <= 1;
                        rd_len <= (act_remaining >> 2) - 1;
                        state <= LOAD_ACT_W;
                    end else if (act_remaining > 0) begin
                        rd_addr <= act_addr + col_group * 128 + (act_total_bytes - act_remaining);
                        rd_start <= 1;
                        rd_len <= 8'd0;
                        state <= LOAD_ACT_W;
                    end else begin
                        byte_idx <= 0;
                        reg_status[8] <= 1;
                        state <= WRITE_RES;
                    end
                end

                LOAD_ACT_W: begin
                    rd_ready <= rd_valid;
                    if (rd_valid && rd_ready) begin
                        act_buf[act_byte_idx[8:3]] <= rd_data;
                        act_byte_idx <= act_byte_idx + 8;
                    end
                    if (rd_done_rise) begin
                        timeout_cnt <= 0;
                        if (act_remaining > 64) begin
                            act_remaining <= act_remaining - 64;
                            state <= LOAD_ACT;
                        end else begin
                            act_remaining <= 24'd0;
                            byte_idx <= 0;
                            reg_status[8] <= 1;
                            if (tensor_type == 15) begin
                                state <= WRITE_RES;
                            end else begin
                                copy_act_idx <= 0;
                                state <= COPY_ACT_TO_CORE;
                            end
                        end
                    end else if (&timeout_cnt) begin
                        timeout_cnt <= 0;
                        timeout_src <= state;
                        state <= TIMEOUT_ERROR;
                    end else begin
                        timeout_cnt <= timeout_cnt + 1;
                    end
                end

                // === Load Q8 weights: 4096 bytes from DDR to core wmem ===
                LOAD_WEIGHT: begin
                    if (wt_remaining > 64) begin
                        rd_addr <= weight_addr + (4096 - wt_remaining);
                        rd_start <= 1;
                        rd_len <= 8'd15;
                        state <= LOAD_WEIGHT_W;
                    end else if (wt_remaining > 0) begin
                        rd_addr <= weight_addr + (4096 - wt_remaining);
                        rd_start <= 1;
                        rd_len <= (wt_remaining >> 2) - 1;
                        state <= LOAD_WEIGHT_W;
                    end else begin
                        sc_byte_idx <= 0;
                        state <= LOAD_SCALES;
                    end
                end

                LOAD_WEIGHT_W: begin
                    // Direct 64-bit word write (DDR is pre-transposed to column-major)
                    rd_ready <= rd_valid;   // same delayed-handshake as LOAD_ACT_W
                    q8_wt_we <= 0;
                    if (rd_valid && rd_ready) begin
                        q8_wt_we    <= 1;
                        q8_wt_addr  <= wt_byte_idx[11:3];
                        q8_wt_din   <= rd_data;
                        wt_byte_idx <= wt_byte_idx + 8;
                    end
                    if (rd_done_rise) begin
                        wt_burst_done <= 1;
                    end
                    if (wt_burst_done) begin
                        wt_burst_done <= 0;
                        timeout_cnt   <= 0;
                        if (wt_remaining > 64) begin
                            wt_remaining <= wt_remaining - 64;
                            state <= LOAD_WEIGHT;
                        end else begin
                            wt_remaining <= 0;
                            sc_remaining <= 256;
                            sc_byte_idx  <= 0;
                            col_group    <= 0;
                            state        <= LOAD_SCALES;
                        end
                    end else if (&timeout_cnt) begin
                        timeout_cnt <= 0;
                        timeout_src <= state;
                        state <= TIMEOUT_ERROR;
                    end else begin
                        timeout_cnt <= timeout_cnt + 1;
                    end
                end

                // === Load Q8 scales: 256 bytes from DDR to core smem (packed as 128 x 16-bit) ===
                // Uses same pattern as LOAD_WEIGHT: sc_remaining tracks bytes left,
                // address = weight_addr + 4096 + (256 - sc_remaining) before each burst.
                LOAD_SCALES: begin
                    if (sc_remaining > 64) begin
                        rd_addr <= weight_addr + 4096 + col_group * 256 + (256 - sc_remaining);
                        rd_start <= 1;
                        rd_len <= 8'd15;
                        state <= LOAD_SCALES_W;
                    end else if (sc_remaining > 0) begin
                        rd_addr <= weight_addr + 4096 + col_group * 256 + (256 - sc_remaining);
                        rd_start <= 1;
                        rd_len <= (sc_remaining >> 2) - 1;
                        state <= LOAD_SCALES_W;
                    end else begin
                        act_byte_idx <= 0;
                        state <= LOAD_ACT;
                    end
                end

                LOAD_SCALES_W: begin
                    if (!rd_unpack_active) begin
                        rd_ready <= rd_valid;
                        q8_sc_we <= 0;
                        if (rd_valid && rd_ready) begin
                            rd_unpack_buf   <= rd_data;
                            rd_unpack_active <= 1;
                            rd_unpack_cnt   <= 0;
                        end
                    end else begin
                        rd_ready <= 0;
                        q8_sc_we <= 1;
                        q8_sc_addr <= sc_byte_idx[7:1];
                        case (rd_unpack_cnt)
                            0: q8_sc_din <= {rd_unpack_buf[15:8], rd_unpack_buf[7:0]};
                            1: q8_sc_din <= {rd_unpack_buf[31:24], rd_unpack_buf[23:16]};
                            2: q8_sc_din <= {rd_unpack_buf[47:40], rd_unpack_buf[39:32]};
                            3: q8_sc_din <= {rd_unpack_buf[63:56], rd_unpack_buf[55:48]};
                        endcase
                        sc_byte_idx <= sc_byte_idx + 2;
                        if (rd_unpack_cnt == 3) begin
                            rd_unpack_active <= 0;
                            rd_unpack_cnt   <= 0;
                        end else begin
                            rd_unpack_cnt <= rd_unpack_cnt + 1;
                        end
                    end
                    if (rd_done_rise) sc_burst_done <= 1;
                    if (sc_burst_done && !rd_unpack_active) begin
                        sc_burst_done <= 0;
                        timeout_cnt   <= 0;
                        if (sc_remaining > 64) begin
                            sc_remaining <= sc_remaining - 64;
                            state <= LOAD_SCALES;
                        end else begin
                            sc_remaining <= 0;
                            act_byte_idx <= 0;
                            state <= LOAD_ACT;
                        end
                    end else if (&timeout_cnt && !rd_unpack_active) begin
                        timeout_cnt   <= 0;
                        timeout_src   <= state;
                        state <= TIMEOUT_ERROR;
                    end else if (!rd_unpack_active) begin
                        timeout_cnt <= timeout_cnt + 1;
                    end
                end

                // =====================================================================
                // Q5_0 Compute Path (per-block wide register interface)
                // =====================================================================
                //
                // DDR layout per block (48 bytes):
                //   [0:1]   core0_d[15:0]       (f16)
                //   [2:5]   core0_qh[31:0]
                //   [6:21]  core0_qs[127:0]
                //   [22:23] core1_d[15:0]
                //   [24:27] core1_qh[31:0]
                //   [28:43] core1_qs[127:0]
                //   [44:47] padding
                // 56 blocks × 48 = 2688 bytes, then 8 bytes row_norm at offset 2688
                //
                // Flow:
                //   1. Q5_LOAD_NORM:  8B DDR read → 4 × UQ8.8 norm → core norm_we ports
                //   2. Q5_COPY_ACT:   1792B DDR read → act_mem BRAM (shared across cores)
                //   3. Q5_BLOCK_COMPUTE loop (56×):
                //      a. 48B DDR read (12 AXI beats) → unpack into blk_d/qh/qs
                //      b. pulse blk_valid → both cores compute block
                //   4. Q5_READ_RES: capture res0/res1 from both cores into act_buf

                Q5_LOAD_NORM: begin
                    // Load 4 row_norm values (8 bytes = 2 AXI beats) from end of current tile
                    rd_addr <= weight_addr + q5_tile_counter * 2696 + 56 * 48;
                    rd_len <= 8'd1;     // 2 AXI beats = 8 bytes = 4 × UQ8.8
                    rd_start <= 1;
                    q5_unpack_cnt <= 0;
                    sc_burst_done <= 0;
                    state <= Q5_LOAD_NORM_W;
                end

                Q5_LOAD_NORM_W: begin
                    if (!rd_unpack_active) begin
                        rd_ready <= rd_valid;
                        if (rd_valid && rd_ready) begin
                            rd_unpack_buf <= rd_data;
                            rd_unpack_active <= 1;
                            q5_unpack_cnt <= 0;
                        end
                    end else begin
                        rd_ready <= 0;
                        q5_norm_we <= 1;
                        case (q5_unpack_cnt)
                            0: begin q5_norm_addr <= 2'd0; q5_norm_din <= rd_unpack_buf[15:0];  end
                            1: begin q5_norm_addr <= 2'd1; q5_norm_din <= rd_unpack_buf[31:16]; end
                            2: begin q5_norm_addr <= 2'd2; q5_norm_din <= rd_unpack_buf[47:32]; end
                            3: begin q5_norm_addr <= 2'd3; q5_norm_din <= rd_unpack_buf[63:48]; end
                        endcase
                        if (q5_unpack_cnt == 3) rd_unpack_active <= 0;
                        q5_unpack_cnt <= q5_unpack_cnt + 1;
                    end
                    if (rd_done_rise) sc_burst_done <= 1;
                    if (sc_burst_done && !rd_unpack_active) begin
                        sc_burst_done <= 0;
                        timeout_cnt <= 0;
                        if (q5_tile_counter == 0) begin
                            q5_copy_act_idx <= 0;
                            state <= Q5_COPY_ACT;
                        end else begin
                            state <= Q5_BLOCK_COMPUTE;
                        end
                    end else if (&timeout_cnt && !rd_unpack_active) begin
                        timeout_cnt <= 0;
                        timeout_src <= state;
                        state <= TIMEOUT_ERROR;
                    end else if (!rd_unpack_active) begin
                        timeout_cnt <= timeout_cnt + 1;
                    end
                end

                Q5_COPY_ACT: begin
                    // Read 64 bytes from DDR = 8 × 64-bit words = 32 activation values
                    rd_addr <= act_addr + q5_copy_act_idx * 2;
                    rd_len <= 8'd15;
                    rd_start <= 1;
                    sc_burst_done <= 0;
                    state <= Q5_COPY_ACT_W;
                end

                Q5_COPY_ACT_W: begin
                    if (!rd_unpack_active) begin
                        rd_ready <= rd_valid;
                        if (rd_valid && rd_ready) begin
                            rd_unpack_buf <= rd_data;
                            rd_unpack_active <= 1;
                            q5_unpack_cnt <= 0;
                        end
                    end else begin
                        rd_ready <= 0;
                        q5_act_we <= 1;
                        case (q5_unpack_cnt)
                            0: begin q5_act_din <= rd_unpack_buf[15:0];  q5_act_addr <= q5_copy_act_idx + 0; end
                            1: begin q5_act_din <= rd_unpack_buf[31:16]; q5_act_addr <= q5_copy_act_idx + 1; end
                            2: begin q5_act_din <= rd_unpack_buf[47:32]; q5_act_addr <= q5_copy_act_idx + 2; end
                            3: begin q5_act_din <= rd_unpack_buf[63:48]; q5_act_addr <= q5_copy_act_idx + 3; end
                        endcase
                        if (q5_unpack_cnt == 3) begin
                            rd_unpack_active <= 0;
                            q5_copy_act_idx <= q5_copy_act_idx + 4;
                        end
                        q5_unpack_cnt <= q5_unpack_cnt + 1;
                    end
                    if (rd_done_rise) sc_burst_done <= 1;
                    if (sc_burst_done && !rd_unpack_active) begin
                        sc_burst_done <= 0;
                        timeout_cnt <= 0;
                        if (q5_copy_act_idx < 896) begin
                            state <= Q5_COPY_ACT;
                        end else begin
                            q5_blk_counter <= 0;
                            q5_clr_acc <= 1;
                            q5_clr_acc_cnt <= 16;
                            state <= Q5_BLOCK_COMPUTE;
                        end
                    end else if (&timeout_cnt && !rd_unpack_active) begin
                        timeout_cnt <= 0;
                        timeout_src <= state;
                        state <= TIMEOUT_ERROR;
                    end else if (!rd_unpack_active) begin
                        timeout_cnt <= timeout_cnt + 1;
                    end
                end

                Q5_BLOCK_COMPUTE: begin
                    // Read 48 bytes (12 AXI beats = 6 × 64-bit) per-block data from DDR
                    rd_addr <= weight_addr + q5_tile_counter * 2696 + q5_blk_counter * 48;
                    rd_len <= 8'd11;    // 12 beats × 4 bytes = 48 bytes
                    rd_start <= 1;      // pulse start for read master
                    q5_unpack_cnt <= 0;
                    q5_unpack_word <= 0;
                    q5_blk_valid_pulsed <= 0;
                    sc_burst_done <= 0;
                    timeout_cnt <= 0;   // reset timeout counter at entry
                    if (q5_blk_counter == 0) begin
                        q5_clr_acc <= 1;  // DSP flush: prod forced to 0 while set
                        q5_clr_acc_cnt <= 16;  // hold for 16 more cycles
                    end
                    state <= Q5_BLOCK_COMPUTE_W;
                end

                Q5_BLOCK_COMPUTE_W: begin
                    if (q5_clr_acc_cnt > 0) begin
                        q5_clr_acc_cnt <= q5_clr_acc_cnt - 1;
                        q5_clr_acc <= 1;  // hold clr_acc for DSP pipeline flush
                    end else begin
                        q5_clr_acc <= 0;
                    end
                    // rd_start stays high until read master accepts (default clears it next cycle)
                    // Capture rd_data words and unpack into core d/qh/qs
                    if (!rd_unpack_active && q5_unpack_word < 6) begin
                        rd_ready <= rd_valid;
                        if (rd_valid && rd_ready) begin
                            rd_unpack_buf <= rd_data;
                            rd_unpack_active <= 1;
                        end
                    end else if (rd_unpack_active && q5_unpack_word < 6) begin
                        rd_ready <= 0;
                        case (q5_unpack_word)
                            0: begin q5_blk_d0   <= rd_unpack_buf[15:0];
                                   q5_blk_qh0[15:0]  <= rd_unpack_buf[31:16];
                                   q5_blk_qh0[31:16] <= rd_unpack_buf[47:32];
                                   q5_blk_qs0[15:0]  <= rd_unpack_buf[63:48]; end
                            1: begin q5_blk_qs0[31:16] <= rd_unpack_buf[15:0];
                                   q5_blk_qs0[47:32]  <= rd_unpack_buf[31:16];
                                   q5_blk_qs0[63:48]  <= rd_unpack_buf[47:32];
                                   q5_blk_qs0[79:64]  <= rd_unpack_buf[63:48]; end
                            2: begin q5_blk_qs0[95:80]  <= rd_unpack_buf[15:0];
                                   q5_blk_qs0[111:96] <= rd_unpack_buf[31:16];
                                   q5_blk_qs0[127:112]<= rd_unpack_buf[47:32];
                                   q5_blk_d1        <= rd_unpack_buf[63:48]; end
                            3: begin q5_blk_qh1         <= rd_unpack_buf[31:0];
                                   q5_blk_qs1[15:0]  <= rd_unpack_buf[47:32];
                                   q5_blk_qs1[31:16] <= rd_unpack_buf[63:48]; end
                            4: begin q5_blk_qs1[47:32]  <= rd_unpack_buf[15:0];
                                   q5_blk_qs1[63:48]  <= rd_unpack_buf[31:16];
                                   q5_blk_qs1[79:64]  <= rd_unpack_buf[47:32];
                                   q5_blk_qs1[95:80]  <= rd_unpack_buf[63:48]; end
                            5: begin q5_blk_qs1[111:96] <= rd_unpack_buf[15:0];
                                   q5_blk_qs1[127:112]<= rd_unpack_buf[31:16]; end
                        endcase
                        q5_unpack_word <= q5_unpack_word + 1;
                        rd_unpack_active <= 0;
                    end
                    // Burst tracking: reset timeout on read completion (like LOAD_ACT_W)
                    if (rd_done_rise) begin
                        sc_burst_done <= 1;
                        timeout_cnt <= 0;
                    end
                    // After all 6 words captured, pulse blk_valid to Q5 core
                    if (q5_unpack_word == 6 && !q5_blk_valid && !q5_blk_valid_pulsed) begin
                        q5_blk_valid <= 1;
                        q5_blk_valid_pulsed <= 1;
                        timeout_cnt <= 0;
                    end
                    // Wait for compute done
                    if (q5_done_rise) begin
                        q5_blk_counter <= q5_blk_counter + 1;
                        q5_blk_valid <= 0;  // clear so next block can re-pulse
                        timeout_cnt <= 0;
                        if (q5_blk_counter == 55) begin
                            q5_unpack_word <= 0;
                            state <= Q5_READ_RES;
                        end else begin
                            state <= Q5_BLOCK_COMPUTE;
                        end
                    end else if (&timeout_cnt && !rd_unpack_active) begin
                        timeout_cnt <= 0;
                        timeout_src <= state;
                        state <= TIMEOUT_ERROR;
                    end else if (!rd_unpack_active) begin
                        timeout_cnt <= timeout_cnt + 1;
                    end
                end

                Q5_READ_RES: begin
                    // Results available directly from core wires
                    act_buf[0] <= {16'd0, q5_core0_res0};
                    act_buf[1] <= {16'd0, q5_core0_res1};
                    act_buf[2] <= {16'd0, q5_core1_res0};
                    act_buf[3] <= {16'd0, q5_core1_res1};
                    // Debug capture: core1 acc values
                    reg_q5_dbg_cap2 <= {q5_core1_res1[31:0]};  // row 3 low 32 bits
                    reg_q5_dbg_cap3 <= {q5_core1_res1[47:32], q5_core1_res0[15:0]};
                    byte_idx <= 0;
                    state <= WRITE_RES;
                end

                // === Copy activation bytes from act_buf to core act_reg (64 x 16-bit) ===
                COPY_ACT_TO_CORE: begin
                    q8_act_we <= 1;
                    q8_act_addr <= copy_act_idx;
                    q8_act_din <= act_buf[copy_act_idx >> 2][{copy_act_idx[1:0], 4'd0} +: 16];
                    if (copy_act_idx == 63) begin
                        copy_act_idx <= 0;
                        state <= COMPUTE;
                    end else begin
                        copy_act_idx <= copy_act_idx + 1;
                    end
                end

                // === Start Q8 core computation ===
                COMPUTE: begin
                    q8_start <= 1;
                    state <= COMPUTE_W;
                end

                COMPUTE_W: begin
                    q8_start <= 0;
                    if (q8_done_rise) begin
                        timeout_cnt <= 0;
                        q8_res_idx <= 0;
                        q8_res_addr <= 0;
                        if (multi_group) begin
                            state <= READ_RES_ACC;
                        end else begin
                            state <= READ_RES;
                        end
                    end else if (&timeout_cnt) begin
                        timeout_cnt <= 0;
                        timeout_src <= state;
                        state <= TIMEOUT_ERROR;
                    end else begin
                        timeout_cnt <= timeout_cnt + 1;
                    end
                end

                // === Read Q8 core results into act_buf (64 rows x 8 bytes = 512 bytes) ===
                READ_RES: begin
                    act_buf[q8_res_idx] <= {16'd0, q8_res_dout};
                    if (q8_res_idx == 63) begin
                        q8_res_idx <= 0;
                        byte_idx <= 0;
                        state <= WRITE_RES;
                    end else begin
                        q8_res_addr <= q8_res_idx + 1;
                        q8_res_idx <= q8_res_idx + 1;
                    end
                end

                // === Read Q8 results and accumulate into acc_buf (multi-group) ===
                READ_RES_ACC: begin
                    if (col_group == 0) begin
                        acc_buf[q8_res_idx] <= q8_res_dout;
                    end else begin
                        acc_buf[q8_res_idx] <= acc_buf[q8_res_idx] + q8_res_dout;
                    end
                    if (q8_res_idx == 63) begin
                        q8_res_idx <= 0;
                        if (col_group == q8_num_groups - 1) begin
                            copy_acc_idx <= 0;
                            state <= COPY_ACC_TO_BUF;
                        end else begin
                            col_group <= col_group + 1;
                            sc_remaining <= 256;
                            sc_byte_idx <= 0;
                            act_byte_idx <= 0;
                            act_remaining <= act_total_bytes;
                            sc_din_lo <= 0;
                            state <= LOAD_SCALES;
                        end
                    end else begin
                        q8_res_addr <= q8_res_idx + 1;
                        q8_res_idx <= q8_res_idx + 1;
                    end
                end

                // === Copy acc_buf to act_buf for DDR writeback ===
                COPY_ACC_TO_BUF: begin
                    act_buf[copy_acc_idx] <= {16'd0, acc_buf[copy_acc_idx]};
                    if (copy_acc_idx == 63) begin
                        copy_acc_idx <= 0;
                        byte_idx <= 0;
                        state <= WRITE_RES;
                    end else begin
                        copy_acc_idx <= copy_acc_idx + 1;
                    end
                end

                // === Write result to DDR (64-bit word, multi-burst) ===
                // Initiates the first burst; subsequent bursts loop through WRITE_RES_BURST.
                WRITE_RES: begin
                    reg_status[9] <= 0;
                    if (tensor_type == 15) wr_remaining <= act_total_bytes;
                    else if (tensor_type == 1) wr_remaining <= 24'd32;  // Q5_0: 4 rows x 8 bytes
                    else wr_remaining <= 24'd512;  // Q8: 64 rows x 8 bytes
                    wr_burst_addr <= result_addr;
                    byte_idx <= 0;
                    wr_byte_offset <= 0;
                    state <= WRITE_RES_BURST;
                end

                // Start next INCR burst
                WRITE_RES_BURST: begin
                    byte_idx <= 0;
                    if (wr_remaining >= 64) begin
                        wr_addr <= wr_burst_addr;
                        wr_word_cnt <= 16'd8;    // 8 words x 8 bytes = 64 bytes
                        wr_start <= 1;
                        wr_burst_bytes <= 9'd64;
                        state <= WRITE_RES_W;
                    end else if (wr_remaining >= 8) begin
                        wr_addr <= wr_burst_addr;
                        wr_word_cnt <= {8'd0, wr_remaining[7:3]};  // remaining / 8
                        wr_start <= 1;
                        wr_burst_bytes <= wr_remaining[8:0];
                        state <= WRITE_RES_W;
                    end else begin
                        reg_status[9] <= 1;
                        reg_desc_head <= reg_desc_head + 1;
                        if (next_addr != 0) begin
                            desc_addr <= next_addr;
                            state <= FETCH_DESC;
                        end else begin
                            state <= DONE;
                        end
                    end
                end

                // Feed 64-bit words to write master for current burst
                WRITE_RES_W: begin
                    wr_start <= 0;
                    wr_wvalid <= 1;
                    if (wr_wready && wr_wvalid) begin
                        byte_idx <= byte_idx + 1;
                        wr_byte_offset <= wr_byte_offset + 8;
                    end
                    if (wr_done_rise) begin
                        timeout_cnt <= 0;
                        wr_wvalid <= 0;
                        wr_burst_addr <= wr_burst_addr + wr_burst_bytes;
                        if (wr_remaining > wr_burst_bytes) begin
                            wr_remaining <= wr_remaining - wr_burst_bytes;
                            state <= WRITE_RES_BURST;
                        end else begin
                            reg_status[9] <= 1;
                            // Q5_0 multi-tile: loop back for next tile
                            if (tensor_type == 1 && q5_tile_counter + 1 < q5_num_tiles) begin
                                q5_tile_counter <= q5_tile_counter + 1;
                                q5_blk_counter <= 6'd0;
                                result_addr <= result_addr + 24'd32;
                                state <= Q5_LOAD_NORM;
                            end else if (tensor_type == 0 && q8_tile_counter + 1 < q8_num_tiles) begin
                                q8_tile_counter   <= q8_tile_counter + 1;
                                weight_addr       <= weight_addr + q8_tile_stride;
                                result_addr       <= result_addr + 24'd512;
                                col_group         <= 0;
                                sc_remaining      <= 256;
                                sc_byte_idx       <= 0;
                                act_byte_idx      <= 0;
                                act_remaining     <= act_total_bytes;
                                wt_byte_idx       <= 0;
                                wt_remaining      <= 4096;
                                state             <= LOAD_WEIGHT;
                            end else begin
                                reg_desc_head <= reg_desc_head + 1;
                                if (next_addr != 0) begin
                                    desc_addr <= next_addr;
                                    state <= FETCH_DESC;
                                end else begin
                                    state <= DONE;
                                end
                            end
                        end
                    end else if (&timeout_cnt) begin
                        timeout_cnt <= 0;
                        timeout_src <= state;
                        state <= TIMEOUT_ERROR;
                    end else begin
                        timeout_cnt <= timeout_cnt + 1;
                    end
                end

                DONE: begin
                    reg_status[15] <= 1'b0;
                    if (reg_start_rise) begin
                        reg_status[15:8] <= 0;
                        reg_desc_head <= 0;
                        desc_addr <= reg_desc_base;
                        state <= FETCH_DESC;
                    end
                end

                TIMEOUT_ERROR: begin
                    reg_status[15] <= 1'b0;
                    reg_act_info[4:0] <= timeout_src;  // expose source state
                    // Stay here until reg_start re-triggers
                    if (reg_start_rise) begin
                        reg_status[15:8] <= 0;
                        reg_desc_head <= 0;
                        desc_addr <= reg_desc_base;
                        state <= FETCH_DESC;
                    end
                end
            endcase

            // DEBUG: expose state and status continuously
            reg_debug[31:27] <= state;       // 5-bit state (was 4 bits, lost state[4])
            reg_debug[26]    <= rd_done;
            reg_debug[25]    <= wr_done;
            reg_debug[24]    <= rd_busy;
            reg_debug[23]    <= wr_busy;
            reg_debug[22]    <= q8_busy;
            reg_debug[21:19] <= wr_dbg_state;
            reg_debug[18:16] <= rd_dbg_state;
            reg_debug[15]    <= q8_done;
            reg_debug[14:11] <= col_group;
            reg_debug[10:8]  <= timeout_cnt[15:13];
            reg_debug[7:0]   <= sc_byte_idx[7:0];

            reg_q8_debug[31:27] <= state;       // 5-bit state
            reg_q8_debug[26]    <= q8_busy;
            reg_q8_debug[25]    <= q8_done;
            reg_q8_debug[24]    <= q8_start;
            reg_q8_debug[23]    <= q8_act_we;
            reg_q8_debug[22:20] <= q8_core_state;     // Q8 core's internal FSM state
            reg_q8_debug[19:17] <= q8_core_g;         // Q8 core's bank counter
            reg_q8_debug[16:11] <= q8_core_k;         // Q8 core's column counter
            reg_q8_debug[10:7]  <= {copy_act_idx[1:0], q8_sc_we, sc_byte_idx[0]};
            reg_q8_debug[6:0]   <= wt_byte_idx[6:0];

            reg_q5_debug[31:27] <= state;
            reg_q5_debug[26]    <= rd_unpack_active;
            reg_q5_debug[25]    <= q5_blk_valid;
            reg_q5_debug[24]    <= q5_any_busy;
            reg_q5_debug[23]    <= q5_busy_both;
            reg_q5_debug[22]    <= q5_done0;
            reg_q5_debug[21]    <= q5_done1;
            reg_q5_debug[20:15] <= {2'b0, q5_unpack_word[3:0]};
            reg_q5_debug[14:9]  <= q5_blk_counter;
            reg_q5_debug[7:0]   <= rd_beat_cnt;      // read master AXI beats received
        end
    end

    // ===== AXI4-Lite write (pulsed-ready, matches proven hp_loopback_top.v) =====
    reg awready_r, wready_r, bvalid_r;
    reg [1:0] bresp_r;
    reg [15:0] awaddr_r;
    reg [31:0] wdata_r;
    reg aw_got, w_got;

    assign s_axil_awready = awready_r;
    assign s_axil_wready  = wready_r;
    assign s_axil_bvalid  = bvalid_r;
    assign s_axil_bresp   = bresp_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            awready_r <= 0; wready_r <= 0; bvalid_r <= 0; bresp_r <= 0;
            awaddr_r <= 0; wdata_r <= 0; aw_got <= 0; w_got <= 0;
            reg_start <= 0; reg_desc_base <= 0; reg_desc_tail <= 0; reg_q8_num_groups <= 0;
        end else begin
            awready_r <= 0; wready_r <= 0;

            if (s_axil_awvalid && !awready_r) begin
                awready_r <= 1;
                if (!aw_got) begin awaddr_r <= s_axil_awaddr; aw_got <= 1; end
            end
            if (s_axil_wvalid && !wready_r) begin
                wready_r <= 1;
                if (!w_got) begin wdata_r <= s_axil_wdata; w_got <= 1; end
            end

            reg_start <= 0;

            if (aw_got && w_got && !bvalid_r) begin
                bvalid_r <= 1; bresp_r <= 0;
      case (awaddr_r[15:0])
           0:       reg_start     <= wdata_r[0];
           16'h10:  reg_q8_num_groups <= wdata_r[3:0];
           16'h18:  reg_desc_base <= wdata_r;
           16'h1C:  reg_desc_tail <= wdata_r;
           16'h4C:  reg_q5_dbg_trig <= wdata_r;  // [15:10]=trig_blk, [0]=arm
       endcase
                aw_got <= 0; w_got <= 0;
            end
            if (bvalid_r && s_axil_bready) bvalid_r <= 0;
        end
    end

    // ===== AXI4-Lite read =====
    reg [1:0] rstate;
    localparam R_IDLE = 0, R_DATA = 1, R_DATA2 = 2;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rstate <= R_IDLE;
            s_axil_arready <= 0; s_axil_rvalid <= 0;
            s_axil_rdata <= 0; s_axil_rresp <= 0;
        end else begin
            case (rstate)
                R_IDLE: begin
                    s_axil_arready <= 1;
                    if (s_axil_arvalid) begin
                        s_axil_arready <= 0;
                        rstate <= R_DATA;
                     case (s_axil_araddr)
                         0:       s_axil_rdata <= {28'h0, reg_start};
                          16'h10:  s_axil_rdata <= {28'h0, reg_q8_num_groups};
                         16'h14:  s_axil_rdata <= reg_status;
                         16'h18:  s_axil_rdata <= reg_desc_base;
                         16'h1C:  s_axil_rdata <= reg_desc_tail;
                         16'h20:  s_axil_rdata <= reg_desc_head;
                         16'h28:  s_axil_rdata <= reg_debug;
                         16'h2C:  s_axil_rdata <= reg_clk_cnt;
                         16'h30:  s_axil_rdata <= reg_clk_cnt_slow;
                         16'h34:  s_axil_rdata <= reg_act_info;
                         16'h38:  s_axil_rdata <= reg_desc_info;
                  16'h3C:  s_axil_rdata <= reg_q8_debug;
                   16'h40:  s_axil_rdata <= reg_q5_debug;
                   16'h44:  s_axil_rdata <= reg_q5_dbg_cap0;
                   16'h48:  s_axil_rdata <= reg_q5_dbg_cap1;
                    16'h4C:  s_axil_rdata <= {q5_core0_d_pre[15:0],       // [31:16] live d_pre
                                              reg_q5_dbg_trig[15:10],    // [15:10] trigger block
                                              q5_dbg_frozen,             // [9] frozen
                                              q5_any_busy,               // [8] busy
                                              q5_core0_state,            // [7:5] core0 state
                                              1'b0,                      // [4] reserved
                                              2'b0,                      // [3:2] reserved
                                              reg_q5_dbg_trig[0]};       // [0] armed
                     16'h50:  s_axil_rdata <= reg_q5_dbg_live;
                     16'h54:  s_axil_rdata <= reg_q5_dbg_cap2;
                      16'h58:  s_axil_rdata <= reg_q5_dbg_cap3;
                      16'h5C:  s_axil_rdata <= reg_q5_dbg_cap4;
                       16'h60:  s_axil_rdata <= reg_q5_dbg_cap5;
                        16'h64:  s_axil_rdata <= reg_q5_dbg_snap;
                        16'h68:  s_axil_rdata <= reg_q5_dbg_wi_start;
                             default: s_axil_rdata <= 32'h0;
                     endcase
                    end
                end
                R_DATA: begin
                    s_axil_rvalid <= 1;
                    rstate <= R_DATA2;
                end
                R_DATA2: begin
                    s_axil_rvalid <= 1;
                    if (s_axil_rready) begin
                        s_axil_rvalid <= 0;
                        rstate <= R_IDLE;
                    end
                end
            endcase
        end
    end

endmodule
