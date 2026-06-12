`timescale 1ns / 1ps
// AXI HP + INT16 core: CPU writes DDR pointers, FPGA reads/writes DDR directly
module axi_hp_int16_top (
    input  wire         clk, rst_n,

    // AXI4-Lite (CPU control) �?interface S_AXI
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWADDR" *)   input  wire [15:0]  S_AXI_AWADDR,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWVALID" *)  input  wire         S_AXI_AWVALID,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWREADY" *)  output wire         S_AXI_AWREADY,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WDATA"  *)   input  wire [31:0]  S_AXI_WDATA,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WSTRB"  *)   input  wire [3:0]   S_AXI_WSTRB,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WVALID" *)   input  wire         S_AXI_WVALID,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WREADY" *)   output wire         S_AXI_WREADY,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BRESP"  *)   output wire [1:0]   S_AXI_BRESP,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BVALID" *)   output wire         S_AXI_BVALID,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BREADY" *)   input  wire         S_AXI_BREADY,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARADDR" *)   input  wire [15:0]  S_AXI_ARADDR,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARVALID" *)  input  wire         S_AXI_ARVALID,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARREADY" *)  output wire         S_AXI_ARREADY,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RDATA"  *)   output wire [31:0]  S_AXI_RDATA,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RRESP"  *)   output wire [1:0]   S_AXI_RRESP,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RVALID" *)   output wire         S_AXI_RVALID,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RREADY" *)   input  wire         S_AXI_RREADY,
    (* X_INTERFACE_INFO = "xilinx.com:signal:interrupt:1.0 INTERRUPT INTERRUPT" *)
    (* X_INTERFACE_PARAMETER = "SENSITIVITY EDGE_RISING" *)
    output wire         interrupt,

    // AXI HP master (PL �?DDR, read + write) �?interface M_AXI_HP
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP ARADDR"  *) output wire [31:0]  m_axi_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP ARVALID" *) output wire         m_axi_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP ARREADY" *) input  wire         m_axi_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP ARLEN"  *) output wire [7:0]   m_axi_arlen,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP ARSIZE" *) output wire [2:0]   m_axi_arsize,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP ARBURST"*) output wire [1:0]   m_axi_arburst,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP RDATA"  *) input  wire [63:0]  m_axi_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP RVALID" *) input  wire         m_axi_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP RREADY" *) output wire         m_axi_rready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP RLAST"  *) input  wire         m_axi_rlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP AWADDR" *) output wire [31:0]  m_axi_awaddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP AWVALID"*) output wire         m_axi_awvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP AWREADY"*) input  wire         m_axi_awready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP AWLEN"  *) output wire [7:0]   m_axi_awlen,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP AWSIZE" *) output wire [2:0]   m_axi_awsize,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP AWBURST"*) output wire [1:0]   m_axi_awburst,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP WDATA"  *) output wire [63:0]  m_axi_wdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP WVALID" *) output wire         m_axi_wvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP WREADY" *) input  wire         m_axi_wready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP WLAST"  *) output wire         m_axi_wlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP WSTRB"  *) output wire [7:0]   m_axi_wstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP BVALID" *) input  wire         m_axi_bvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP BREADY" *) output wire         m_axi_bready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HP BRESP"  *) input  wire  [1:0]  m_axi_bresp
);
    // ======== Internal Registers ========
    reg [31:0] reg_ap_ctrl;     // [0]=start
    reg [31:0] reg_status;
    reg [31:0] reg_wt_addr;     // DDR address of weights (8192 bytes)
    reg [31:0] reg_act_addr;    // DDR address of activations (128 bytes)
    reg [31:0] reg_res_addr;    // DDR address for results (512 bytes)
    reg [31:0] reg_debug;       // debug: {hp_done, hp_busy, state, byte_cnt}

    wire core_done, core_busy;
    wire [47:0] core_res_dout;

    // ======== INT16 core ========
    wire        core_start;
    reg         act_we;
    reg  [5:0]  act_addr;
    reg  [15:0] act_din;
    reg         wt_we;
    reg  [12:0] wt_addr;
    reg  [7:0]  wt_din;
    wire [5:0]  res_addr;
    wire [47:0] res_dout;

    matmul_int16_core u_core (
        .clk(clk), .rst_n(rst_n),
        .start(core_start), .op_vecmul(1'b1),
        .done(core_done), .busy(core_busy),
        .wt_we(wt_we), .wt_addr(wt_addr), .wt_din(wt_din),
        .act_we(act_we), .act_addr(act_addr), .act_din(act_din),
        .res_addr(res_addr), .res_dout(res_dout),
        .sc_we(1'b0), .sc_addr(7'd0), .sc_din(16'd0)
    );

    // ======== AXI HP Read Master ========
    wire        hp_read_busy, hp_read_done, hp_read_valid;
    wire [7:0]  hp_read_data;
    reg         hp_read_ready, hp_read_start;
    reg  [31:0] hp_read_addr;
    reg  [7:0]  hp_read_len;

    axihp_read_master u_hpread (
        .clk(clk), .rst_n(rst_n),
        .start(hp_read_start), .src_addr(hp_read_addr), .burst_len(hp_read_len),
        .done(hp_read_done), .busy(hp_read_busy),
        .data_out(hp_read_data), .data_valid(hp_read_valid), .data_ready(hp_read_ready),
        .m_axi_araddr(m_axi_araddr), .m_axi_arvalid(m_axi_arvalid),
        .m_axi_arready(m_axi_arready),
        .m_axi_arlen(m_axi_arlen), .m_axi_arsize(m_axi_arsize),
        .m_axi_arburst(m_axi_arburst),
        .m_axi_rdata(m_axi_rdata), .m_axi_rvalid(m_axi_rvalid),
        .m_axi_rready(m_axi_rready), .m_axi_rlast(m_axi_rlast)
    );

    // ======== AXI HP Write Master ========
    wire        hp_write_busy, hp_write_done, hp_write_ready;
    reg         hp_write_start, hp_write_valid;
    reg  [31:0] hp_write_addr;
    reg  [15:0] hp_write_count;
    reg  [63:0] hp_write_data;

    axihp_write_master u_hpwrite (
        .clk(clk), .rst_n(rst_n),
        .start(hp_write_start), .dst_addr(hp_write_addr), .word_count(hp_write_count),
        .busy(hp_write_busy), .done(hp_write_done),
        .wdata(hp_write_data), .wvalid(hp_write_valid), .wready(hp_write_ready),
        .m_axi_awaddr(m_axi_awaddr), .m_axi_awvalid(m_axi_awvalid),
        .m_axi_awready(m_axi_awready),
        .m_axi_awlen(m_axi_awlen), .m_axi_awsize(m_axi_awsize),
        .m_axi_awburst(m_axi_awburst),
        .m_axi_wdata(m_axi_wdata), .m_axi_wvalid(m_axi_wvalid),
        .m_axi_wready(m_axi_wready), .m_axi_wlast(m_axi_wlast),
        .m_axi_wstrb(m_axi_wstrb),
        .m_axi_bvalid(m_axi_bvalid), .m_axi_bready(m_axi_bready),
        .m_axi_bresp(m_axi_bresp)
    );

    // ======== FSM ========
    localparam IDLE  = 0, RD_WT = 1, RD_ACT = 2, COMP = 3, DRAIN = 4,
               START_WR = 5, WR_RES = 6;
    reg [2:0] state;
    reg [13:0] byte_cnt;    // bytes read/written in current phase
    reg [5:0]  idx;
    reg        core_start_pulse;
    reg        start_clear;
    reg        hp_read_done_prev;  // edge detection for hp_read_done
    wire       hp_read_done_rise = hp_read_done && !hp_read_done_prev;

    assign core_start = core_start_pulse;
    assign res_addr = idx;
    assign res_dout = core_res_dout;

    // Debug
    always @(*) begin
        reg_debug = {hp_read_busy, hp_read_done, hp_write_busy, hp_write_done,
                     state, byte_cnt[7:0], idx};
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE; reg_status <= 0;
            hp_read_start <= 0; hp_read_ready <= 0;
            hp_write_start <= 0; hp_write_valid <= 0;
            wt_we <= 0; act_we <= 0; core_start_pulse <= 0;
            byte_cnt <= 0; idx <= 0;
        end else begin
            hp_read_start <= 0; hp_write_start <= 0; hp_write_valid <= 0;
            wt_we <= 0; act_we <= 0; core_start_pulse <= 0;
            hp_read_done_prev <= hp_read_done;

            case (state)
                IDLE: begin
                    if (reg_ap_ctrl[0] && !core_busy) begin
                        reg_status <= 1; state <= RD_WT;
                        byte_cnt <= 0;
                        // Start reading weights from DDR
                        hp_read_addr <= reg_wt_addr;
                        hp_read_len <= 8'd15;   // 16 beats = 128 bytes per burst (AXI3 max)
                        hp_read_start <= 1;
                    end
                end

                RD_WT: begin
                    if (hp_read_valid && !hp_read_ready) begin
                        hp_read_ready <= 1;
                        wt_we <= 1;
                        wt_addr <= {byte_cnt[3:0], byte_cnt[12:4]};
                        wt_din  <= hp_read_data;
                        byte_cnt <= byte_cnt + 1;
                    end else begin
                        hp_read_ready <= 0;
                    end

                    if (hp_read_done_rise) begin
                        if (byte_cnt < 8192) begin
                            // Next burst
                            hp_read_addr <= reg_wt_addr + byte_cnt;
                            hp_read_len <= 8'd15;   // 16 beats = 128 bytes per burst (AXI3 max)
                            hp_read_start <= 1;
                        end else begin
                            // Weights done, start reading activations
                            state <= RD_ACT;
                            byte_cnt <= 0;
                            hp_read_addr <= reg_act_addr;
                            hp_read_len <= 8'd15;  // 16 beats = 128 bytes
                            hp_read_start <= 1;
                        end
                    end
                end

                RD_ACT: begin
                    if (hp_read_valid && !hp_read_ready) begin
                        hp_read_ready <= 1;
                        // Accumulate pairs of bytes into 16-bit activations
                        if (byte_cnt[0] == 0)
                            act_din[7:0] <= hp_read_data;
                        else begin
                            act_din[15:8] <= hp_read_data;
                            act_we <= 1;
                            act_addr <= byte_cnt[6:1];
                        end
                        byte_cnt <= byte_cnt + 1;
                    end else begin
                        hp_read_ready <= 0;
                    end

                    if (hp_read_done_rise) begin
                        state <= COMP;
                        core_start_pulse <= 1;
                    end
                end

                COMP: begin
                    if (core_done) begin
                        state <= DRAIN;
                        idx <= 0;
                    end
                end

                DRAIN: begin
                    if (idx < 64) begin
                        idx <= idx + 1;
                    end else begin
                        state <= START_WR;
                        idx <= 0;
                        hp_write_addr <= reg_res_addr;
                        hp_write_count <= 64;
                    end
                end

                START_WR: begin
                    hp_write_start <= 1;
                    state <= WR_RES;
                end

                WR_RES: begin
                    if (!hp_write_busy && hp_write_done) begin
                        state <= IDLE;
                        reg_status <= 0;
                    end
                    if (hp_write_valid && hp_write_ready) begin
                        hp_write_valid <= 0;
                        idx <= idx + 1;
                    end
                    if (!hp_write_valid && idx < 64) begin
                        hp_write_data <= {16'd0, 16'd0, res_dout[31:0]};
                        hp_write_valid <= 1;
                    end
                end
            endcase
        end
    end

    // ======== AXI4-Lite (from verified axi_wrap_int16.v pattern) ========
    reg awready_r, wready_r, bvalid_r, aw_got, w_got;
    reg [15:0] awaddr_r;
    reg [31:0] wdata_r;
    reg [1:0]  bresp_r;

    assign S_AXI_AWREADY = awready_r;
    assign S_AXI_WREADY  = wready_r;
    assign S_AXI_BVALID  = bvalid_r;
    assign S_AXI_BRESP   = bresp_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            awready_r <= 0; wready_r <= 0; bvalid_r <= 0; bresp_r <= 0;
            awaddr_r <= 0; wdata_r <= 0; aw_got <= 0; w_got <= 0;
            reg_ap_ctrl <= 32'h4; reg_status <= 0;
            reg_wt_addr <= 0; reg_act_addr <= 0; reg_res_addr <= 0;
        end else begin
            reg_ap_ctrl[3:1] <= {~core_busy, ~core_busy, core_done};
            if (start_clear) reg_ap_ctrl[0] <= 0;

            awready_r <= 0; wready_r <= 0;

            // AW
            if (S_AXI_AWVALID && !awready_r) begin
                awready_r <= 1;
                if (!aw_got) begin awaddr_r <= S_AXI_AWADDR; aw_got <= 1; end
            end
            // W
            if (S_AXI_WVALID && !wready_r) begin
                wready_r <= 1;
                if (!w_got) begin wdata_r <= S_AXI_WDATA; w_got <= 1; end
            end
            // Process
            if (aw_got && w_got && !bvalid_r) begin
                bvalid_r <= 1; bresp_r <= 2'b00;
                case (awaddr_r[15:0])
                    16'h0000: if (!core_busy && state==IDLE) reg_ap_ctrl[0] <= wdata_r[0];
                    16'h0018: reg_wt_addr  <= wdata_r;
                    16'h001C: reg_act_addr <= wdata_r;
                    16'h0020: reg_res_addr <= wdata_r;
                endcase
                aw_got <= 0; w_got <= 0;
            end
            // Clear BVALID when BREADY
            if (bvalid_r && S_AXI_BREADY) bvalid_r <= 0;
        end
    end

    // Read
    reg arready_r, rvalid_r;
    reg [31:0] rdata_r;
    reg [1:0]  rresp_r;
    assign S_AXI_ARREADY = arready_r;
    assign S_AXI_RVALID  = rvalid_r;
    assign S_AXI_RDATA   = rdata_r;
    assign S_AXI_RRESP   = rresp_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            arready_r <= 0; rvalid_r <= 0; rdata_r <= 0; rresp_r <= 0;
        end else begin
            arready_r <= 0;
            if (S_AXI_ARVALID && !arready_r) begin
                arready_r <= 1;
                case (S_AXI_ARADDR)
                    16'h0000: rdata_r <= reg_ap_ctrl;
                    16'h0014: rdata_r <= reg_status;
                    16'h0018: rdata_r <= reg_wt_addr;
                    16'h001C: rdata_r <= reg_act_addr;
                    16'h0020: rdata_r <= reg_res_addr;
                    16'h0024: rdata_r <= reg_debug;
                    default:  rdata_r <= 32'h0;
                endcase
                rresp_r <= 2'b00;
            end
            if (!rvalid_r && arready_r) rvalid_r <= 1;
            if (rvalid_r && S_AXI_RREADY) rvalid_r <= 0;
        end
    end
    assign interrupt = 1'b0;
endmodule
