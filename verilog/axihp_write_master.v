`timescale 1ns / 1ps

// AXI HP Write Master — 32-bit mode, split 64-bit words into 2×32-bit AXI3 writes
// AWSIZE=2 (4 bytes/beat) on a 64-bit HP bus. Each 64-bit word from the
// top is split into two single-beat AXI3 transactions:
//   Lower half: A[2]=0, WDATA[31:0], WSTRB[3:0]
//   Upper half: A[2]=1, WDATA[63:32], WSTRB[7:4]
// wready asserted once per word (in W_L only, not in upper half).
module axihp_write_master (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         start,
    input  wire [31:0]  dst_addr,
    input  wire [15:0]  word_count,
    output reg          busy,
    output reg          done,
    input  wire [63:0]  wdata,
    input  wire         wvalid,
    output reg          wready,
    output wire [2:0]   dbg_state,
    output reg  [5:0]   m_axi_awid,
    output reg  [31:0]  m_axi_awaddr,
    output reg          m_axi_awvalid,
    input  wire         m_axi_awready,
    output reg  [7:0]   m_axi_awlen,
    output reg  [2:0]   m_axi_awsize,
    output reg  [1:0]   m_axi_awburst,
    output reg  [1:0]   m_axi_awlock,
    output reg  [3:0]   m_axi_awcache,
    output reg  [2:0]   m_axi_awprot,
    output reg  [5:0]   m_axi_wid,
    output reg  [63:0]  m_axi_wdata,
    output reg          m_axi_wvalid,
    input  wire         m_axi_wready,
    output reg          m_axi_wlast,
    output reg  [7:0]   m_axi_wstrb,
    input  wire         m_axi_bvalid,
    output reg          m_axi_bready,
    input  wire  [1:0]  m_axi_bresp,
    input  wire  [5:0]  m_axi_bid
);

    localparam [2:0] IDLE = 0, AW_L = 1, W_L = 2, B_L = 3, AW_U = 4, B_U = 5;
    reg [2:0] state;
    reg [15:0] words_rem;
    reg [31:0] sub_addr;
    reg [31:0] hold_data_hi;  // captured wdata[63:32] during W_L

    assign dbg_state = state;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state         <= IDLE;
            busy          <= 0;
            done          <= 0;
            m_axi_awvalid <= 0;
            m_axi_awid    <= 6'd0;
            m_axi_awlock  <= 2'd0;
            m_axi_awcache <= 4'b0011;
            m_axi_awprot  <= 3'd0;
            m_axi_wid     <= 6'd0;
            m_axi_wvalid  <= 0;
            m_axi_wstrb   <= 8'h00;
            m_axi_bready  <= 0;
            wready        <= 0;
            words_rem     <= 0;
            hold_data_hi  <= 0;
        end else begin
            done <= 0;

            case (state)
                IDLE: begin
                    if (start && !busy) begin
                        if (word_count == 0) begin
                            busy  <= 0;
                            done  <= 1;
                        end else begin
                            busy      <= 1;
                            words_rem <= word_count;
                            sub_addr  <= dst_addr;
                            m_axi_awaddr  <= dst_addr;
                            m_axi_awlen   <= 8'd0;
                            m_axi_awsize  <= 3'd2;   // 4 bytes per beat
                            m_axi_awburst <= 2'd1;
                            m_axi_awvalid <= 1;
                            state     <= AW_L;
                        end
                    end
                end

                // Lower half: send AW with A[2]=0
                AW_L: begin
                    if (m_axi_awready) begin
                        m_axi_awvalid <= 0;
                        // Send lower half data immediately
                        m_axi_wdata[31:0] <= wdata[31:0];
                        m_axi_wdata[63:32] <= 32'd0;
                        m_axi_wstrb    <= 8'h0F;
                        m_axi_wlast    <= 1'b1;
                        m_axi_wvalid   <= 1;
                        wready         <= 1;
                        state <= W_L;
                    end
                end

                // Lower half: complete write, assert wready, capture upper half
                W_L: begin
                    if (m_axi_wvalid && m_axi_wready) begin
                        hold_data_hi <= wdata[63:32];
                        m_axi_wvalid <= 0;
                        m_axi_bready <= 1;
                        wready       <= 0;
                        state <= B_L;
                    end
                end

                // Lower half: wait for write response
                B_L: begin
                    if (m_axi_bvalid) begin
                        m_axi_bready <= 0;
                        // Upper half: send AW with A[2]=1
                        m_axi_awaddr  <= sub_addr + 4;
                        m_axi_awlen   <= 8'd0;
                        m_axi_awsize  <= 3'd2;
                        m_axi_awburst <= 2'd1;
                        m_axi_awvalid <= 1;
                        // Upper half data from hold register (no wready)
                        m_axi_wdata[63:32] <= hold_data_hi;
                        m_axi_wdata[31:0]  <= 32'd0;
                        m_axi_wstrb    <= 8'hF0;
                        m_axi_wlast    <= 1'b1;
                        m_axi_wvalid   <= 1;
                        state <= AW_U;
                    end
                end

                // Upper half: combined AW+W (no separate W_U, no wready)
                AW_U: begin
                    if (m_axi_awready)
                        m_axi_awvalid <= 0;
                    if (m_axi_wready && m_axi_wvalid) begin
                        m_axi_wvalid <= 0;
                        m_axi_bready <= 1;
                        state <= B_U;
                    end
                end

                // Upper half: wait for write response, then advance to next word
                B_U: begin
                    if (m_axi_bvalid) begin
                        m_axi_bready <= 0;
                        words_rem <= words_rem - 1;
                        sub_addr  <= sub_addr + 8;
                        if (words_rem > 1) begin
                            // Next word: lower half
                            m_axi_awaddr  <= sub_addr + 8;
                            m_axi_awlen   <= 8'd0;
                            m_axi_awsize  <= 3'd2;
                            m_axi_awburst <= 2'd1;
                            m_axi_awvalid <= 1;
                            state <= AW_L;
                        end else begin
                            busy  <= 0;
                            done  <= 1;
                            state <= IDLE;
                        end
                    end
                end
            endcase
        end
    end

endmodule
