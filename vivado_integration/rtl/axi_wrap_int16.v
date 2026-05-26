`timescale 1ns / 1ps

module axi_wrap_int16 (
    input  wire         clk,
input  wire         rst_n,
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
output reg          interrupt
);

    reg [31:0] reg_ap_ctrl;
    reg [31:0] reg_gie;
    reg [31:0] reg_ier;
    reg [31:0] reg_isr;
    reg [31:0] reg_ctrl_user;
    reg [31:0] reg_status;

    // Weight buffer (2048 × 32-bit = 8192 bytes, BRAM)
    (* ram_style = "block" *) reg [31:0] weight_buf [0:2047];
    reg [10:0] wb_waddr;
    reg [31:0] wb_wdata;
    reg        wb_we;

    always @(posedge clk) begin
        if (wb_we)
            weight_buf[wb_waddr] <= wb_wdata;
    end

    // Act buffer
    reg [15:0] act_buf [0:63];
    reg [47:0] result_buf [0:63];
    integer ai;

    // Weight load sequencer: copy weight_buf → core wmem
    reg [12:0] load_addr;
    reg        core_wt_we;
    reg [12:0] core_wt_addr;
    reg [7:0]  core_wt_din;
    reg [7:0]  core_wt_din_pre;
    reg        core_act_we;
    reg [5:0]  core_act_addr;
    reg [15:0] core_act_din;
    reg [5:0]  core_res_addr;
    reg        core_start;

    // Separate weight_buf read register (no async reset for BRAM compatibility)
    always @(posedge clk) begin
        core_wt_din_pre <= weight_buf[load_addr[12:2]][{load_addr[1:0], 3'b000} +: 8];
    end

    wire core_done, core_busy;
    wire [47:0] core_res_dout;

    matmul_int16_core u_core (
        .clk       (clk),
        .rst_n     (rst_n),
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

    localparam S_IDLE  = 0;
    localparam S_LOAD  = 1;
    localparam S_ACT   = 2;
    localparam S_COMP  = 3;
    localparam S_DRAIN = 4;

    reg [2:0] state;
    reg [5:0] idx;

    always @(posedge clk) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            idx        <= 0;
            load_addr  <= 0;
            core_start <= 0;
            core_wt_we <= 0;
            core_act_we <= 0;
            reg_status <= 0;
        end else begin
            core_start <= 0;
            core_wt_we <= 0;
            core_act_we <= 0;
            core_res_addr <= idx;

            case (state)
                S_IDLE: begin
                    if (reg_ap_ctrl[0] && !core_busy) begin
                        reg_ap_ctrl[0] <= 0;
                        reg_status <= 1;
                        state <= S_LOAD;
                        load_addr <= 0;
                    end
                end

                S_LOAD: begin
                    if (load_addr < 8192) begin
                        core_wt_we   <= 1;
                        core_wt_addr <= {load_addr[3:0], load_addr[12:4]};
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

    always @(*) begin
        reg_ap_ctrl[3:1] = {~core_busy, ~core_busy, core_done};
    end

    // ==================================================================
    // AXI4-Lite write
    // ==================================================================
    reg [1:0] wstate;
    localparam W_IDLE = 0, W_RESP = 1;

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
            wb_we <= 0;
        end else begin
            wb_we <= 0;
            case (wstate)
                W_IDLE: begin
                    s_axil_awready <= 1;
                    s_axil_wready  <= 1;
                    if (s_axil_awvalid && s_axil_wvalid) begin
                        s_axil_awready <= 0;
                        s_axil_wready  <= 0;
                        wstate <= W_RESP;

                        case (s_axil_awaddr[15:0])
                            16'h0000: if (!core_busy && state == S_IDLE)
                                          reg_ap_ctrl[0] <= s_axil_wdata[0];
                            16'h0004: reg_gie       <= s_axil_wdata;
                            16'h0008: reg_ier       <= s_axil_wdata;
                            16'h000C: if (s_axil_wdata[0]) reg_isr[0] <= 0;
                            16'h0010: reg_ctrl_user <= s_axil_wdata;
                        endcase

                        // Weight write: AXI 0x2000-0x3FFF → weight_buf (2048 × 32-bit)
                        if (s_axil_awaddr[15:13] == 3'b001) begin
                            wb_we     <= 1;
                            wb_waddr  <= s_axil_awaddr[12:2];
                            wb_wdata  <= s_axil_wdata;
                        end

                        // Act write: AXI 0x1000-0x107C → act_buf
                        if (s_axil_awaddr[15:7] == 9'b000_1000_0) begin
                            ai = {s_axil_awaddr[6:2], 1'b0};
                            if (ai < 63) begin
                                act_buf[ai]     <= s_axil_wdata[15:0];
                                act_buf[ai + 1] <= s_axil_wdata[31:16];
                            end
                        end
                    end
                end

                W_RESP: begin
                    s_axil_bvalid <= 1;
                    s_axil_bresp  <= 2'b00;
                    s_axil_awready <= 0;
                    s_axil_wready  <= 0;
                    if (s_axil_bready) begin
                        s_axil_bvalid <= 0;
                        wstate <= W_IDLE;
                    end
                end
            endcase
        end
    end

    // ==================================================================
    // AXI4-Lite read
    // ==================================================================
    reg [1:0] rstate;
    localparam R_IDLE = 0, R_DATA = 1;

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
                        s_axil_rdata <= axil_read(s_axil_araddr);
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
                    // Act readback: 0x5000-0x507C
                    if (addr[8:7] == 2'b00) begin
                        ri = {addr[6:2], 1'b0};
                        axil_read = {act_buf[ri + 1], act_buf[ri]};
                    end
                end
                4'h4: begin
                    // Result: 0x4000-0x40FC (lo 32b), 0x4200-0x427C (hi 16b)
                    if (addr[11:10] == 2'b00) begin
                        // 0x4000-0x40FF → lo
                        ri = (addr - 16'h4000) >> 2;
                        if (ri < 64)
                            axil_read = result_buf[ri][31:0];
                    end else if (addr[11:10] == 2'b10) begin
                        // 0x4200-0x42FF → hi
                        ri = (addr - 16'h4200) >> 2;
                        if (ri < 64)
                            axil_read = {16'b0, result_buf[ri][47:32]};
                    end
                end
            endcase
        end
    endfunction

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            interrupt <= 0;
        end else begin
            if (core_done && reg_gie[0] && reg_ier[0])
                reg_isr[0] <= 1;
            interrupt <= reg_gie[0] && reg_ier[0] && reg_isr[0];
        end
    end

endmodule
