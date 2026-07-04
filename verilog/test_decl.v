`timescale 1ns/1ps
module test_decl (
    input wire clk,
    output wire [2:0] dbg_state
);
    reg [2:0] state;
    reg [7:0] buf [0:63];
    assign dbg_state = state;
endmodule
