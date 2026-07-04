`timescale 1ns/1ps
module test_plus;
  reg [511:0] buf;
  reg [5:0] idx;
  wire [7:0] byte_out = buf[idx*8 +: 8];
endmodule