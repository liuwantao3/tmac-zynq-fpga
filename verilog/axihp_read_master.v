`timescale 1ns / 1ps

// AXI HP Read Master — 32-bit INCR burst, byte-stream output
// ARSIZE=2 (4 bytes/beat) on a 64-bit HP bus. Zynq-7010 with x16 DDR3
// caps HP0 at 32-bit: RDATA[63:32] is always 0, data returns on RDATA[31:0].
// ARSIZE=2 avoids the lost-upper-half issue.
module axihp_read_master (
    input  wire         clk,
    input  wire         rst_n,

    input  wire         start,
    input  wire [31:0]  src_addr,
    input  wire [7:0]   burst_len,    // beats - 1 (0-255)
    output reg          done,
    output reg          busy,
    output wire [2:0]   dbg_state,

    output wire [7:0]   data_out,
    output reg          data_valid,
    input  wire         data_ready,

    output reg  [5:0]   m_axi_arid,
    output reg  [31:0]  m_axi_araddr,
    output reg          m_axi_arvalid,
    input  wire         m_axi_arready,
    output reg  [7:0]   m_axi_arlen,
    output reg  [2:0]   m_axi_arsize,
    output reg  [1:0]   m_axi_arburst,
    output reg  [1:0]   m_axi_arlock,
    output reg  [3:0]   m_axi_arcache,
    output reg  [2:0]   m_axi_arprot,
    input  wire [63:0]  m_axi_rdata,
    input  wire [1:0]   m_axi_rresp,
    input  wire [5:0]   m_axi_rid,
    input  wire         m_axi_rvalid,
    output reg          m_axi_rready,
    input  wire         m_axi_rlast
);

    localparam IDLE       = 3'd0;
    localparam SEND_AR    = 3'd1;
    localparam WAIT_R     = 3'd2;
    localparam DRAIN_BYTE = 3'd3;

    reg [2:0]  state;
    reg [7:0]  beat_cnt;
    reg [31:0] rdata_hold;  // captured RDATA[31:0] per beat
    reg [1:0]  buf_idx;     // 0..3 (4 bytes per beat)

    assign data_out = rdata_hold[{buf_idx, 3'b0} +: 8];
    assign dbg_state = state;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state          <= IDLE;
            done           <= 0;
            busy           <= 0;
            data_valid     <= 0;
            m_axi_arvalid  <= 0;
            m_axi_rready   <= 0;
            buf_idx        <= 0;
            beat_cnt       <= 0;
        end else begin
            data_valid <= 0;
            done       <= 0;

            case (state)
                IDLE: begin
                    if (start) begin
                        busy          <= 1;
                        m_axi_arid    <= 6'd0;
                        m_axi_araddr  <= src_addr;
                        m_axi_arlen   <= burst_len;
                        m_axi_arsize  <= 3'd2;   // 4 bytes per beat
                        m_axi_arburst <= 2'd1;   // INCR
                        m_axi_arlock  <= 2'd0;
                        m_axi_arcache <= 4'b0011;
                        m_axi_arprot  <= 3'd0;
                        m_axi_arvalid <= 1;
                        beat_cnt      <= 0;
                        state         <= SEND_AR;
                    end
                end

                SEND_AR: begin
                    if (m_axi_arready) begin
                        m_axi_arvalid <= 0;
                        m_axi_rready  <= 1;
                        state         <= WAIT_R;
                    end
                end

                WAIT_R: begin
                    if (m_axi_rvalid) begin
                        m_axi_rready <= 0;
                        rdata_hold   <= m_axi_rdata[31:0];
                        buf_idx      <= 0;
                        state        <= DRAIN_BYTE;
                    end
                end

                DRAIN_BYTE: begin
                    data_valid <= 1;
                    if (data_ready) begin
                        if (buf_idx == 3) begin  // 4 bytes per beat
                            data_valid <= 0;
                            if (beat_cnt == burst_len) begin
                                m_axi_rready <= 0;
                                state <= IDLE;
                                busy  <= 0;
                                done  <= 1;
                            end else begin
                                beat_cnt <= beat_cnt + 1;
                                m_axi_rready <= 1;
                                state <= WAIT_R;
                            end
                        end else begin
                            buf_idx <= buf_idx + 1;
                        end
                    end
                end
            endcase
        end
    end

endmodule
