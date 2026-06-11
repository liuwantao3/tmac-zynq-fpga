`timescale 1ns / 1ps

module axi_wrap_int16 (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 S_AXI_ACLK CLK" *)
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF S_AXI" *)
    input  wire         S_AXI_ACLK,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 S_AXI_ARESETN RST" *)
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  wire         S_AXI_ARESETN,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWADDR" *)
    input  wire [15:0]  S_AXI_AWADDR,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWVALID" *)
    input  wire         S_AXI_AWVALID,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWREADY" *)
    output wire         S_AXI_AWREADY,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WDATA" *)
    input  wire [31:0]  S_AXI_WDATA,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WSTRB" *)
    input  wire [3:0]   S_AXI_WSTRB,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WVALID" *)
    input  wire         S_AXI_WVALID,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WREADY" *)
    output wire         S_AXI_WREADY,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BRESP" *)
    output wire [1:0]   S_AXI_BRESP,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BVALID" *)
    output wire         S_AXI_BVALID,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BREADY" *)
    input  wire         S_AXI_BREADY,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARADDR" *)
    input  wire [15:0]  S_AXI_ARADDR,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARVALID" *)
    input  wire         S_AXI_ARVALID,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARREADY" *)
    output wire         S_AXI_ARREADY,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RDATA" *)
    output wire [31:0]  S_AXI_RDATA,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RRESP" *)
    output wire [1:0]   S_AXI_RRESP,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RVALID" *)
    output wire         S_AXI_RVALID,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RREADY" *)
    input  wire         S_AXI_RREADY,

    (* X_INTERFACE_INFO = "xilinx.com:signal:interrupt:1.0 INTERRUPT INTERRUPT" *)
    (* X_INTERFACE_PARAMETER = "SENSITIVITY EDGE_RISING" *)
    output wire         interrupt
);

    // ====================================================================
    // Internal registers
    // ====================================================================
    reg [31:0] reg_ap_ctrl;
    reg [31:0] reg_gie;
    reg [31:0] reg_ier;
    reg [31:0] reg_isr;
    reg [31:0] reg_ctrl_user;
    reg [31:0] reg_status;

    // Weight buffer (2048 x 32-bit = 8192 bytes, BRAM)
    (* ram_style = "block" *) reg [31:0] weight_buf [0:2047];
    reg [10:0] wb_waddr;
    reg [31:0] wb_wdata;
    reg        wb_we;

    always @(posedge S_AXI_ACLK) begin
        if (wb_we)
            weight_buf[wb_waddr] <= wb_wdata;
    end

    // Act and result buffers
    reg [15:0] act_buf [0:63];
    reg [47:0] result_buf [0:63];
    integer ai;

    // ====================================================================
    // Core instance
    // ====================================================================
    reg        core_wt_we;
    reg [12:0] core_wt_addr;
    reg [7:0]  core_wt_din;
    reg        core_act_we;
    reg [5:0]  core_act_addr;
    reg [15:0] core_act_din;
    wire [5:0] core_res_addr;
    reg        core_start;

    reg [13:0] load_addr;  // 14 bits to hold 8192 (2^13)
    reg [7:0]  core_wt_din_pre;

    always @(posedge S_AXI_ACLK) begin
        core_wt_din_pre <= weight_buf[load_addr[12:2]][{load_addr[1:0], 3'b000} +: 8];
    end

    wire core_done, core_busy;
    wire [47:0] core_res_dout;
    // Pre-fetch address for weight loading (BRAM read has 1-cycle latency)
    wire [13:0] prev_load = load_addr - 1;

    matmul_int16_core u_core (
        .clk       (S_AXI_ACLK),
        .rst_n     (S_AXI_ARESETN),
        .start     (core_start),
        .op_vecmul (1'b0),
        .done      (core_done),
        .busy      (core_busy),
        .wt_we     (core_wt_we),
        .wt_addr   (core_wt_addr),
        .wt_din    (core_wt_din),
        .sc_we     (1'b0),
        .sc_addr   (7'd0),
        .sc_din    (16'd0),
        .act_we    (core_act_we),
        .act_addr  (core_act_addr),
        .act_din   (core_act_din),
        .res_addr  (core_res_addr),
        .res_dout  (core_res_dout)
    );

    // ====================================================================
    // Compute FSM
    // ====================================================================
    localparam S_IDLE  = 0;
    localparam S_LOAD  = 1;
    localparam S_ACT   = 2;
    localparam S_COMP  = 3;
    localparam S_DRAIN = 4;

    reg [2:0] state;
    reg [6:0] idx;  // 7 bits to hold 64 (2^6)
    assign core_res_addr = idx;
    reg start_clear;

    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            state      <= S_IDLE;
            idx        <= 0;
            load_addr  <= 0;
            core_start <= 0;
            core_wt_we <= 0;
            core_act_we <= 0;
            reg_status <= 0;
            start_clear <= 0;
        end else begin
            core_start <= 0;
            core_wt_we <= 0;
            core_act_we <= 0;
            start_clear <= 0;

            case (state)
                S_IDLE: begin
                    if (reg_ap_ctrl[0] && !core_busy) begin
                        start_clear <= 1;
                        reg_status <= 1;
                        state <= S_LOAD;
                        load_addr <= 0;
                    end
                end

                S_LOAD: begin
                    core_wt_we <= 0;
                    if (load_addr == 0) begin
                        load_addr <= 1;
                    end else if (load_addr <= 8192) begin
                        core_wt_we   <= 1;
                        core_wt_addr <= {prev_load[3:0], prev_load[12:4]};
                        core_wt_din  <= core_wt_din_pre;
                        load_addr    <= load_addr + 1;
                    end else begin
                        state <= S_ACT;
                        idx <= 0;
                    end
                end

                S_ACT: begin
                    if (idx < 64) begin
                        core_act_we  <= 1;
                        core_act_addr <= idx;
                        core_act_din  <= act_buf[idx];
                        idx <= idx + 1;
                    end else begin
                        state <= S_COMP;
                        core_start <= 1;
                    end
                end

                S_COMP: begin
                    if (core_done) begin
                        state <= S_DRAIN;
                        idx <= 0;
                        reg_status <= 2;
                    end
                end

                S_DRAIN: begin
                    if (idx < 64) begin
                        result_buf[idx] <= core_res_dout;
                        idx <= idx + 1;
                    end else begin
                        state <= S_IDLE;
                        reg_status <= 0;
                    end
                end
            endcase
        end
    end

    // ====================================================================
    // AXI4-Lite write (standard template pattern)
    // ====================================================================
    reg awready_r;
    reg wready_r;
    reg bvalid_r;
    reg [1:0] bresp_r;

    reg [15:0] awaddr_r;
    reg [31:0] wdata_r;
    reg        aw_got;
    reg        w_got;

    assign S_AXI_AWREADY = awready_r;
    assign S_AXI_WREADY  = wready_r;
    assign S_AXI_BVALID  = bvalid_r;
    assign S_AXI_BRESP   = bresp_r;

    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            awready_r <= 0;
            wready_r  <= 0;
            bvalid_r  <= 0;
            bresp_r   <= 0;
            reg_ap_ctrl   <= 32'h0000_0004;
            reg_gie       <= 0;
            reg_ier       <= 0;
            reg_isr       <= 0;
            reg_ctrl_user <= 0;
            awaddr_r <= 0;
            wdata_r  <= 0;
            aw_got   <= 0;
            w_got    <= 0;
            wb_we    <= 0;
        end else begin
            reg_ap_ctrl[3:1] <= {~core_busy, ~core_busy, core_done};
            if (start_clear)
                reg_ap_ctrl[0] <= 0;

            wb_we <= 0;

            // Defaults
            awready_r <= 0;
            wready_r  <= 0;

            // ============================================================
            // AW: latch address when valid
            // ============================================================
            if (S_AXI_AWVALID && !awready_r) begin
                awready_r <= 1;
                if (!aw_got) begin
                    awaddr_r <= S_AXI_AWADDR;
                    aw_got   <= 1;
                end
            end

            // ============================================================
            // W: latch data when valid
            // ============================================================
            if (S_AXI_WVALID && !wready_r) begin
                wready_r <= 1;
                if (!w_got) begin
                    wdata_r <= S_AXI_WDATA;
                    w_got   <= 1;
                end
            end

            // ============================================================
            // B: when both AW and W received, process and send response
            // ============================================================
            if (aw_got && w_got && !bvalid_r) begin
                bvalid_r <= 1;
                bresp_r  <= 2'b00;

                // Process write
                case (awaddr_r[15:0])
                    16'h0000: if (!core_busy && state == S_IDLE)
                                 reg_ap_ctrl[0] <= wdata_r[0];
                    16'h0004: reg_gie       <= wdata_r;
                    16'h0008: reg_ier       <= wdata_r;
                    16'h000C: if (wdata_r[0]) reg_isr[0] <= 0;
                    16'h0010: reg_ctrl_user <= wdata_r;
                endcase

                if (awaddr_r[15:13] == 3'b001) begin
                    wb_we    <= 1;
                    wb_waddr <= awaddr_r[12:2];
                    wb_wdata <= wdata_r;
                end

                if (awaddr_r[15:7] == 9'h20) begin
                    ai = {awaddr_r[6:2], 1'b0};
                    if (ai < 63) begin
                        act_buf[ai]     <= wdata_r[15:0];
                        act_buf[ai + 1] <= wdata_r[31:16];
                    end
                end


                aw_got <= 0;
                w_got  <= 0;
            end

            // Clear BVALID when BREADY
            if (bvalid_r && S_AXI_BREADY) begin
                bvalid_r <= 0;
            end
        end
    end

    // ====================================================================
    // AXI4-Lite read
    // ====================================================================
    reg arready_r;
    reg rvalid_r;
    reg [31:0] rdata_r;
    reg [1:0] rresp_r;

    assign S_AXI_ARREADY = arready_r;
    assign S_AXI_RVALID  = rvalid_r;
    assign S_AXI_RDATA   = rdata_r;
    assign S_AXI_RRESP   = rresp_r;

    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            arready_r <= 0;
            rvalid_r  <= 0;
            rdata_r   <= 0;
            rresp_r   <= 0;
        end else begin
            arready_r <= 0;

            if (S_AXI_ARVALID && !arready_r) begin
                arready_r <= 1;
                rdata_r   <= axil_read(S_AXI_ARADDR);
                rresp_r   <= 2'b00;
            end

            if (!rvalid_r && arready_r) begin
                rvalid_r <= 1;
            end

            if (rvalid_r && S_AXI_RREADY) begin
                rvalid_r <= 0;
            end
        end
    end

    function [31:0] axil_read(input [15:0] addr);
        reg [5:0] ri;
        begin
            axil_read = 0;
            case (addr[15:12])
                4'h0: begin
                    case (addr[11:0])
                        12'h000: axil_read = reg_ap_ctrl;
                        12'h004: axil_read = reg_gie;
                        12'h008: axil_read = reg_ier;
                        12'h00C: axil_read = reg_isr;
                        12'h010: axil_read = reg_ctrl_user;
                        12'h014: axil_read = reg_status;
                    endcase
                end
                4'h5: begin
                    if (addr[8:7] == 2'b00) begin
                        ri = {addr[6:2], 1'b0};
                        axil_read = {act_buf[ri + 1], act_buf[ri]};
                    end
                end
                4'h4: begin
                    if (addr[9:8] == 2'b00) begin
                        ri = (addr - 16'h4000) >> 2;
                        if (ri < 64)
                            axil_read = result_buf[ri][31:0];
                    end else if (addr[9:8] == 2'b10) begin
                        ri = (addr - 16'h4200) >> 2;
                        if (ri < 64)
                            axil_read = {16'b0, result_buf[ri][47:32]};
                    end
                end
            endcase
        end
    endfunction

    // ====================================================================
    // Interrupt
    // ====================================================================
    reg int_r;
    assign interrupt = int_r;

    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            int_r <= 0;
        end else begin
            if (core_done && reg_gie[0] && reg_ier[0])
                reg_isr[0] <= 1;
            int_r <= reg_gie[0] && reg_ier[0] && reg_isr[0];
        end
    end

endmodule
