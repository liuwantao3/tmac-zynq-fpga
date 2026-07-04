`timescale 1ns/1ps
// Stub matching the 64-bit wdata axihp_write_master interface
module axihp_write_master (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         start,
    input  wire [31:0]  dst_addr,
    input  wire [15:0]  word_count,
    output reg          done,
    output reg          busy,
    output wire [2:0]   dbg_state,
    input  wire [63:0]  wdata,
    input  wire         wvalid,
    output reg          wready,
    output reg  [5:0]   m_axi_awid
);
  reg [2:0] state;
  reg [15:0] beat_cnt;
  assign dbg_state = state;
  always @(posedge clk) begin
    if (start && !busy) begin
      busy <= 1;
      beat_cnt <= 0;
      wready <= 1;
    end
    if (wvalid && wready) begin
      if (beat_cnt == word_count - 1) begin
        wready <= 0;
        done <= 1;
        busy <= 0;
      end
      beat_cnt <= beat_cnt + 1;
    end
  end
endmodule
