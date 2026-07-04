`timescale 1ns/1ps
module test;
  reg [511:0] buf;
  reg [5:0] idx;
  reg [7:0] byte_sel;
  always @* begin
    byte_sel = buf[idx*8 +: 8];
  end
endmodule
