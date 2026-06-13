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
    reg [31:0] reg_debug;
    reg [31:0] reg_wr_test;     // [0]=write test trigger, [14:8]=test idx
    reg        wr_sticky_start, wr_sticky_aw, wr_sticky_w, wr_sticky_b;
    // reg_debug layout:
    // [2:0]   = FSM state
    // [3]     = fsm_busy
    // [4]     = hp_write_start (LIVE)
    // [5]     = wr_sticky_aw
    // [6]     = wr_sticky_w
    // [7]     = wr_sticky_b
    // [9:8]   = idx[1:0]
    // [10]    = hp_read_busy
    // [11]    = hp_read_done
    // [12]    = hp_write_busy
    // [13]    = hp_write_done
    // [14]    = core_busy
    // [15]    = core_done
    // [16]    = wr_sticky_start
    // [17]    = hp_write_busy_fall
    // [20:18] = prev_state (state from which IDLE was entered)
    // [23:21] = 000
    // [31:24] = 0xDB (marker)

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
    reg  [7:0]  act_lo;
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
    reg         hp_read_start;
    wire        hp_read_ready;
    assign hp_read_ready = (state == RD_WT || state == RD_ACT) && hp_read_valid && hp_read_busy;
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
               START_WR = 5, WR_RES = 6, WRITE_TEST = 7;
    reg [2:0] state;  // 3 bits supports 0-7 (8 states)
    reg [13:0] byte_cnt;
    reg [6:0]  idx;
    reg        core_start_pulse;
    reg        fsm_busy;
    reg [2:0]  prev_state;
    reg        hp_read_busy_prev;
    wire       hp_read_busy_fall = !hp_read_busy && hp_read_busy_prev;
    reg        hp_write_busy_prev;
    wire       hp_write_busy_fall = !hp_write_busy && hp_write_busy_prev;
    reg [1:0]  wr_test_idx;
    reg        wr_test_active;

    assign core_start = core_start_pulse;
    assign res_addr = idx;
    assign res_dout = core_res_dout;

    // Diagnostic: capture FIRST hp_read_addr
    reg [31:0] dbg_first_addr;
    reg        dbg_first_done;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin dbg_first_addr <= 0; dbg_first_done <= 0; end
        else if (hp_read_start && !dbg_first_done) begin
            dbg_first_addr <= hp_read_addr;
            dbg_first_done <= 1;
        end
    end

    // Replace dedicated diagnostic block — fold into FSM block to avoid multi-driver
    // Write-sticky bits live in reg_debug[9:6]

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE; reg_status <= 0;
            hp_read_start <= 0;
            hp_write_start <= 0; hp_write_valid <= 0;
            wt_we <= 0; act_we <= 0; core_start_pulse <= 0;
            byte_cnt <= 0; idx <= 0; act_lo <= 0;
            fsm_busy <= 0; hp_read_busy_prev <= 0; hp_write_busy_prev <= 0;
            reg_debug <= 0;
            wr_sticky_aw <= 0; wr_sticky_w <= 0; wr_sticky_b <= 0; wr_sticky_start <= 0;
            wr_test_idx <= 0;
        end else begin
            // Capture write sticky events
            if (m_axi_awvalid && m_axi_awready) wr_sticky_aw <= 1;
            if (m_axi_wvalid  && m_axi_wready)  wr_sticky_w  <= 1;
            if (m_axi_bvalid  && m_axi_bready)  wr_sticky_b  <= 1;
            if (hp_write_start)                 wr_sticky_start <= 1;

            // Live diagnostic: expose FSM state via reg_debug
            reg_debug[2:0]   <= state;
            reg_debug[3]     <= fsm_busy;
            reg_debug[4]     <= hp_write_start;                // live, not sticky
            reg_debug[5]     <= wr_sticky_aw;
            reg_debug[6]     <= wr_sticky_w;
            reg_debug[7]     <= wr_sticky_b;
            reg_debug[9:8]   <= idx[1:0];
            reg_debug[10]    <= hp_read_busy;
            reg_debug[11]    <= hp_read_done;
            reg_debug[12]    <= hp_write_busy;
            reg_debug[13]    <= hp_write_done;
            reg_debug[14]    <= core_busy;
            reg_debug[15]    <= core_done;
            reg_debug[17:16] <= {wr_sticky_start, hp_write_busy_fall};
            reg_debug[20:18] <= prev_state;
            reg_debug[23:21] <= 3'b000;
            reg_debug[31:24] <= 8'hDB;

            hp_read_start <= 0; hp_write_start <= 0; hp_write_valid <= 0;
            wt_we <= 0; act_we <= 0; core_start_pulse <= 0;
            hp_read_busy_prev <= hp_read_busy;
            hp_write_busy_prev <= hp_write_busy;
            prev_state <= state;

            case (state)
                IDLE: begin
                    if (reg_wr_test[0] && !hp_write_busy && !wr_test_active) begin
                        // Direct write test: bypass core, write known pattern to DDR
                        wr_test_active <= 1;
                        hp_write_addr <= reg_res_addr;
                        hp_write_count <= 16'd4;
                        hp_write_start <= 1;
                        wr_test_idx <= 0;
                        state <= WRITE_TEST;
                    end else if (reg_ap_ctrl[0] && !core_busy && !fsm_busy) begin
                        fsm_busy <= 1;
                        reg_status <= 1; state <= RD_WT;
                        byte_cnt <= 0;
                        // Start reading weights from DDR
                        hp_read_addr <= reg_wt_addr;
                        hp_read_len <= 8'd15;   // 16 beats = 128 bytes per burst (AXI3 max)
                        hp_read_start <= 1;
                    end
                end

                RD_WT: begin
                    if (hp_read_busy)
                        hp_read_start <= 0;

                    if (hp_read_valid && hp_read_busy) begin
                        wt_we <= 1;
                        wt_addr <= {byte_cnt[3:0], byte_cnt[12:4]};
                        wt_din  <= hp_read_data;
                        byte_cnt <= byte_cnt + 1;
                    end

                    if (hp_read_busy_fall) begin
                        if (byte_cnt < 8192) begin
                            hp_read_addr <= reg_wt_addr + byte_cnt;
                            hp_read_len <= 8'd15;
                            hp_read_start <= 1;
                        end else begin
                            state <= RD_ACT;
                            byte_cnt <= 0;
                            hp_read_addr <= reg_act_addr;
                            hp_read_len <= 8'd15;
                            hp_read_start <= 1;
                        end
                    end
                end

                RD_ACT: begin
                    if (hp_read_busy)
                        hp_read_start <= 0;

                    if (hp_read_valid && hp_read_busy) begin
                        if (byte_cnt[0] == 0)
                            act_lo <= hp_read_data;
                        else begin
                            act_din <= {hp_read_data, act_lo};
                            act_we <= 1;
                            act_addr <= byte_cnt[6:1];
                        end
                        byte_cnt <= byte_cnt + 1;
                    end

                    if (hp_read_busy_fall) begin
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
                        hp_write_count <= 16;
                    end
                end

                START_WR: begin
                    hp_write_start <= 1;
                    state <= WR_RES;
                end

                WR_RES: begin
                    if (hp_write_busy_fall) begin
                        if (idx < 64) begin
                            // More results to write: start next burst
                            hp_write_addr <= hp_write_addr + 128;
                            hp_write_count <= 16;
                            state <= START_WR;
                        end else begin
                            state <= IDLE;
                            reg_status <= 0;
                            fsm_busy <= 0;
                        end
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

                WRITE_TEST: begin
                    if (hp_write_busy)
                        hp_write_start <= 0;
                    // Send test data: 4 words of known pattern
                    if (hp_write_valid && hp_write_ready) begin
                        hp_write_valid <= 0;
                        wr_test_idx <= wr_test_idx + 1;
                    end else if (!hp_write_valid && wr_test_idx < 4) begin
                        case (wr_test_idx)
                            0: hp_write_data <= 64'hDEADBEEF_CAFEBABE;
                            1: hp_write_data <= 64'h12345678_9ABCDEF0;
                            2: hp_write_data <= 64'h00000000_FFFFFFFF;
                            3: hp_write_data <= 64'hAAAAAAAA_55555555;
                        endcase
                        hp_write_valid <= 1;
                    end
                    if (hp_write_busy_fall) begin
                        state <= IDLE;
                        wr_test_active <= 0;
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
            reg_ap_ctrl <= 32'h4;
            reg_wt_addr <= 0; reg_act_addr <= 0; reg_res_addr <= 0;
            reg_wr_test <= 0;
        end else begin
            reg_ap_ctrl[3:1] <= {~core_busy, ~core_busy, core_done};
            if (state != IDLE) reg_ap_ctrl[0] <= 0;

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
                    16'h0028: reg_wr_test  <= wdata_r;
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
                    16'h0028: rdata_r <= reg_wr_test;
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
