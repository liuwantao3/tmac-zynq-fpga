`timescale 1ns / 1ps

module axilite_slave (
    input  wire         clk,
    input  wire         rst_n,

    // AXI4-Lite write address
    input  wire         s_axil_awvalid,
    output reg          s_axil_awready,
    input  wire [15:0]  s_axil_awaddr,

    // AXI4-Lite write data
    input  wire         s_axil_wvalid,
    output reg          s_axil_wready,
    input  wire [31:0]  s_axil_wdata,
    input  wire [3:0]   s_axil_wstrb,

    // AXI4-Lite write response
    output reg          s_axil_bvalid,
    input  wire         s_axil_bready,
    output reg [1:0]   s_axil_bresp,

    // AXI4-Lite read address
    input  wire         s_axil_arvalid,
    output reg          s_axil_arready,
    input  wire [15:0]  s_axil_araddr,

    // AXI4-Lite read data
    output reg          s_axil_rvalid,
    input  wire         s_axil_rready,
    output reg [31:0]  s_axil_rdata,
    output reg [1:0]   s_axil_rresp,

    // Register interface
    output reg  [31:0]  reg_ap_ctrl,
    input  wire [31:0]  reg_ap_ctrl_rd,
    output reg  [31:0]  reg_gie,
    output reg  [31:0]  reg_ier,
    input  wire [31:0]  reg_isr,
    output reg  [31:0]  reg_ctrl_user,
    input  wire [31:0]  reg_status,

    // Data buffer write interface
    output reg          buf_we,
    output reg [11:0]   buf_addr,
    output reg [31:0]   buf_din,

    // Data buffer read interface
    input  wire [31:0]  buf_dout
);

    localparam REG_AP_CTRL   = 16'h0000;
    localparam REG_GIE       = 16'h0004;
    localparam REG_IER       = 16'h0008;
    localparam REG_ISR       = 16'h000C;
    localparam REG_CTRL_USER = 16'h0010;
    localparam REG_STATUS    = 16'h0014;
    localparam BUF_BASE      = 16'h1000;
    localparam BUF_LIMIT     = 16'h40FF;

    // Write transaction FSM
    reg [1:0] wstate;
    localparam W_IDLE = 0, W_ADDR = 1, W_DATA = 2, W_RESP = 3;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wstate <= W_IDLE;
            s_axil_awready <= 0;
            s_axil_wready  <= 0;
            s_axil_bvalid  <= 0;
            s_axil_bresp   <= 0;
            reg_ap_ctrl    <= 32'h0000_0004; // idle initially
            reg_gie        <= 0;
            reg_ier        <= 0;
            reg_ctrl_user  <= 0;
            buf_we         <= 0;
        end else begin
            buf_we <= 0;
            case (wstate)
                W_IDLE: begin
                    s_axil_awready <= 1;
                    s_axil_wready  <= 1;
                    if (s_axil_awvalid && s_axil_wvalid) begin
                        s_axil_awready <= 0;
                        s_axil_wready  <= 0;
                        wstate <= W_RESP;

                        if (s_axil_awaddr < BUF_BASE) begin
                            case (s_axil_awaddr)
                                REG_AP_CTRL:   reg_ap_ctrl   <= s_axil_wdata;
                                REG_GIE:       reg_gie       <= s_axil_wdata;
                                REG_IER:       reg_ier       <= s_axil_wdata;
                                REG_CTRL_USER: reg_ctrl_user <= s_axil_wdata;
                            endcase
                        end else if (s_axil_awaddr >= BUF_BASE && s_axil_awaddr <= BUF_LIMIT) begin
                            buf_we   <= 1;
                            buf_addr <= s_axil_awaddr[13:2]; // word-aligned addr / 4
                            buf_din  <= s_axil_wdata;
                        end
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

    // Read transaction FSM
    reg [1:0] rstate;
    localparam R_IDLE = 0, R_ADDR = 1, R_DATA = 2;

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

                        if (s_axil_araddr < BUF_BASE) begin
                            case (s_axil_araddr)
                                REG_AP_CTRL:   s_axil_rdata <= reg_ap_ctrl_rd;
                                REG_GIE:       s_axil_rdata <= reg_gie;
                                REG_IER:       s_axil_rdata <= reg_ier;
                                REG_ISR:       s_axil_rdata <= reg_isr;
                                REG_CTRL_USER: s_axil_rdata <= reg_ctrl_user;
                                REG_STATUS:    s_axil_rdata <= reg_status;
                                default:       s_axil_rdata <= 32'hDEADBEEF;
                            endcase
                        end else begin
                            s_axil_rdata <= buf_dout;
                        end
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

endmodule
