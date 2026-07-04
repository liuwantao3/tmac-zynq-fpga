`timescale 1ns/1ps
module axihp_write_master (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         start,
    input  wire [31:0]  dst_addr,
    input  wire [7:0]   burst_len,
    output reg          done,
    output reg          busy,
    output wire [2:0]   dbg_state,
    input  wire [7:0]   data_in,
    input  wire         data_valid,
    output reg          data_ready,
    output reg  [5:0]   m_axi_awid
);
  reg [2:0] state;
  reg [7:0] buf [0:63];
  always @(posedge clk) state <= 0;
endmodule
