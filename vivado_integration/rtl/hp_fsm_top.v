`timescale 1ns / 1ps
// Minimal descriptor-chain FSM — fetch desc from DDR, read act, write result.
// Based on hp_loopback_top.v (proven HP0 working). No compute core.
module hp_fsm_top (
    input  wire         clk, rst_n,
    // AXI4-Lite slave (GP0)
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

    // ===== HP Read Master =====
    wire       rd_busy, rd_done, rd_valid;
    wire [7:0] rd_data;
    wire [2:0] rd_dbg_state;
    reg        rd_start, rd_ready;
    reg [31:0] rd_addr;
    reg [7:0]  rd_len;

    axihp_read_master u_rd (
        .clk(clk), .rst_n(rst_n),
        .start(rd_start), .src_addr(rd_addr),
        .burst_len(rd_len),
        .done(rd_done), .busy(rd_busy),
        .dbg_state(rd_dbg_state),
        .data_out(rd_data), .data_valid(rd_valid), .data_ready(rd_ready),
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
    reg [63:0] wr_wdata;
    reg        wr_wvalid;
    reg [31:0] wr_addr;
    reg [15:0] wr_count;

    axihp_write_master u_wr (
        .clk(clk), .rst_n(rst_n),
        .start(wr_start), .dst_addr(wr_addr),
        .word_count(wr_count),
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

    // ===== Buffer =====
    reg [7:0] desc_buf [0:31];    // 32-byte descriptor
    reg [7:0] act_buf [0:255];    // 256-byte activation buffer
    reg [31:0] desc_addr;         // current descriptor DDR address
    reg [31:0] next_addr;
    reg [31:0] weight_addr;
    reg [31:0] act_addr;
    reg [31:0] result_addr;
    reg [15:0] tensor_type;       // descriptor type (15=CPU_OP)
    reg [23:0] act_total_bytes;
    reg [23:0] act_remaining;     // bytes left to read (multi-burst tracking)

    // ===== FSM =====
    localparam IDLE           = 4'd0;
    localparam FETCH_DESC     = 4'd1;
    localparam FETCH_DESC_W   = 4'd2;
    localparam LOAD_ACT       = 4'd3;
    localparam LOAD_ACT_W     = 4'd4;
    localparam WRITE_RES      = 4'd5;
    localparam WRITE_RES_W    = 4'd6;
    localparam DONE           = 4'd7;

    reg [3:0] state;
    reg [5:0] byte_idx;
    reg [7:0] desc_byte_idx;
    reg [7:0] act_byte_idx;

    // Edge detection for done signals
    reg rd_done_d, wr_done_d;
    wire rd_done_rise, wr_done_rise;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin rd_done_d <= 0; wr_done_d <= 0; end
        else begin rd_done_d <= rd_done; wr_done_d <= wr_done; end
    end
    assign rd_done_rise = rd_done && !rd_done_d;
    assign wr_done_rise = wr_done && !wr_done_d;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            reg_status <= 0;
            reg_desc_head <= 0;
            reg_debug <= 0;
            rd_start <= 0; rd_ready <= 0;
            rd_addr <= 0; rd_len <= 0;
            wr_start <= 0; wr_wvalid <= 0; wr_wdata <= 0;
            wr_addr <= 0; wr_count <= 0;
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
        end else begin
            reg_clk_cnt <= reg_clk_cnt + 1;
            reg_clk_cnt_slow <= reg_clk_cnt_slow + (|reg_clk_cnt[9:0] ? 0 : 1);
            rd_start <= 0; rd_ready <= 0;
            wr_start <= 0; wr_wvalid <= 0;
            reg_status[15] <= 1'b1;  // default busy

            case (state)
                IDLE: begin
                    reg_status[15:8] <= 0;
                    if (reg_start) begin
                        reg_desc_head <= 0;
                        desc_addr <= reg_desc_base;
                        state <= FETCH_DESC;
                    end
                end

                // === Descriptor fetch: read 32 bytes from desc_base + head*32 ===
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
                        desc_buf[desc_byte_idx] <= rd_data;
                        desc_byte_idx <= desc_byte_idx + 1;
                    end
                    if (rd_done_rise) begin
                        // Parse descriptor (little-endian DDR layout)
                        // bytes 0-3:  next_addr
                        // bytes 4-7:  weight_addr
                        // bytes 8-11: act_addr
                        // bytes 12-15:result_addr
                        // bytes 16-23: reserved (16-17 = tensor_type)
                        // bytes 24-27: act_total_bytes (24-bit LE)
                        next_addr    <= {desc_buf[3], desc_buf[2], desc_buf[1], desc_buf[0]};
                        weight_addr  <= {desc_buf[7], desc_buf[6], desc_buf[5], desc_buf[4]};
                        act_addr     <= {desc_buf[11],desc_buf[10],desc_buf[9], desc_buf[8]};
                        result_addr  <= {desc_buf[15],desc_buf[14],desc_buf[13],desc_buf[12]};
                        tensor_type  <= {desc_buf[17], desc_buf[16]};
                        act_total_bytes <= {desc_buf[26], desc_buf[25], desc_buf[24]};
                        act_remaining  <= {desc_buf[26], desc_buf[25], desc_buf[24]};
                        reg_act_info  <= {desc_buf[11],desc_buf[10],desc_buf[9], desc_buf[8]};
                        reg_desc_info <= {8'h0, desc_buf[26], desc_buf[25], desc_buf[24]};
                        act_byte_idx   <= 0;
                        byte_idx       <= 0;
                        state          <= LOAD_ACT;
                    end
                end

                // === Load activation data from DDR (multi-burst, AXI3 max 16 beats) ===
                LOAD_ACT: begin
                    reg_status[8] <= 0;  // clear prior rd_done
                    if (act_remaining > 64) begin
                        rd_addr <= act_addr + (act_total_bytes - act_remaining);
                        rd_start <= 1;
                        rd_len <= 8'd15;
                        state <= LOAD_ACT_W;
                    end else if (act_remaining >= 4) begin
                        rd_addr <= act_addr + (act_total_bytes - act_remaining);
                        rd_start <= 1;
                        rd_len <= (act_remaining >> 2) - 1;
                        state <= LOAD_ACT_W;
                    end else if (act_remaining > 0) begin
                        rd_addr <= act_addr + (act_total_bytes - act_remaining);
                        rd_start <= 1;
                        rd_len <= 8'd0;          // 1 beat (guard against >> 2 wrap)
                        state <= LOAD_ACT_W;
                    end else begin
                        byte_idx <= 0;
                        reg_status[8] <= 1;
                        state <= WRITE_RES;      // nothing to read
                    end
                end

                LOAD_ACT_W: begin
                    rd_ready <= rd_valid;
                    if (rd_valid && rd_ready && act_byte_idx < act_total_bytes) begin
                        act_buf[act_byte_idx] <= rd_data;
                        act_byte_idx <= act_byte_idx + 1;
                    end
                    if (rd_done_rise) begin
                        if (act_remaining > 64) begin
                            act_remaining <= act_remaining - 64;
                            state <= LOAD_ACT;      // another burst needed
                        end else begin
                            act_remaining <= 24'd0;
                            byte_idx <= 0;
                            reg_status[8] <= 1;     // all bursts done
                            // Check for CPU_OP descriptor (tensor_type == 15)
                            if (tensor_type == 15) begin
                                reg_status[15] <= 1'b1;  // keep busy
                                // TODO: trigger CPU interrupt and wait
                                // For now, treat CPU_OP as pass-through (go next or done)
                                state <= DONE;
                            end else begin
                                state <= WRITE_RES;
                            end
                        end
                    end
                end

                // === Write result to DDR (copy act data to result addr) ===
                WRITE_RES: begin
                    reg_status[9] <= 0;  // clear prior wr_done
                    wr_addr  <= result_addr;
                    wr_count <= {8'd0, act_total_bytes[15:3]};  // bytes/8 = words
                    wr_wdata <= {act_buf[7], act_buf[6], act_buf[5], act_buf[4],
                                 act_buf[3], act_buf[2], act_buf[1], act_buf[0]};
                    wr_wvalid <= 1;
                    wr_start <= 1;
                    byte_idx <= 0;
                    state <= WRITE_RES_W;
                end

                WRITE_RES_W: begin
                    wr_start   <= 0;   // pulsed: cleared after write master latches it
                    wr_wvalid <= (byte_idx < wr_count);
                    wr_wdata <= {act_buf[byte_idx*8+7], act_buf[byte_idx*8+6],
                                 act_buf[byte_idx*8+5], act_buf[byte_idx*8+4],
                                 act_buf[byte_idx*8+3], act_buf[byte_idx*8+2],
                                 act_buf[byte_idx*8+1], act_buf[byte_idx*8+0]};
                    if (wr_wready && wr_wvalid)
                        byte_idx <= byte_idx + 1;
                    if (wr_done_rise) begin
                        reg_status[9] <= 1;  // write done
                        reg_desc_head <= reg_desc_head + 1;
                        // Chain traversal: if next_addr != 0, advance to next descriptor
                        if (next_addr != 0) begin
                            desc_addr <= next_addr;
                            state <= FETCH_DESC;
                        end else begin
                            state <= DONE;
                        end
                    end
                end

                DONE: begin
                    reg_status[15] <= 1'b0;
                    if (reg_start) begin
                        reg_status[15:8] <= 0;      // clear prior done bits
                        reg_desc_head <= 0;
                        desc_addr <= reg_desc_base;
                        state <= FETCH_DESC;        // restart chain from base
                    end
                end
            endcase

            // DEBUG: expose state and status continuously
            reg_debug[31:28] <= state;
            reg_debug[27]    <= rd_done;
            reg_debug[26]    <= wr_done;
            reg_debug[25]    <= rd_busy;
            reg_debug[24]    <= wr_busy;
            reg_debug[23]    <= 1'b0;
            reg_debug[22:20] <= wr_dbg_state;
            reg_debug[19:17] <= rd_dbg_state;
            reg_debug[16]    <= 1'b0;
            reg_debug[15:8]  <= rd_len;
            reg_debug[7:0]   <= act_byte_idx;
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
            reg_start <= 0; reg_desc_base <= 0; reg_desc_tail <= 0;
        end else begin
            awready_r <= 0; wready_r <= 0;  // default LOW (pulsed-ready style)

            if (s_axil_awvalid && !awready_r) begin
                awready_r <= 1;
                if (!aw_got) begin awaddr_r <= s_axil_awaddr; aw_got <= 1; end
            end
            if (s_axil_wvalid && !wready_r) begin
                wready_r <= 1;
                if (!w_got) begin wdata_r <= s_axil_wdata; w_got <= 1; end
            end

            reg_start <= 0;  // auto-clear

            if (aw_got && w_got && !bvalid_r) begin
                bvalid_r <= 1; bresp_r <= 0;
                case (awaddr_r[15:0])
                    0:       reg_start     <= wdata_r[0];
                    16'h18:  reg_desc_base <= wdata_r;
                    16'h1C:  reg_desc_tail <= wdata_r;
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
                            16'h14:  s_axil_rdata <= reg_status;
                            16'h18:  s_axil_rdata <= reg_desc_base;
                            16'h1C:  s_axil_rdata <= reg_desc_tail;
                            16'h20:  s_axil_rdata <= reg_desc_head;
                            16'h28:  s_axil_rdata <= reg_debug;
                            16'h2C:  s_axil_rdata <= reg_clk_cnt;
                            16'h30:  s_axil_rdata <= reg_clk_cnt_slow;
                            16'h34:  s_axil_rdata <= reg_act_info;
                            16'h38:  s_axil_rdata <= reg_desc_info;
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
