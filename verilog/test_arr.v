`timescale 1ns/1ps
module test;
  reg [7:0] buf [0:63];
  integer i;
  initial begin
    for (i=0;i<64;i=i+1) buf[i] = i;
    $display("buf[0]=%d buf[63]=%d", buf[0], buf[63]);
  end
endmodule
