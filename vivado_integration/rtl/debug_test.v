`timescale 1ns / 1ps
module debug_test (
    input  wire         clk,
    input  wire         rst_n
);
    reg [31:0] counter = 0;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) counter <= 0;
        else        counter <= counter + 1;
    end

    (* MARK_DEBUG = "TRUE" *) wire dbg_tick = (counter[23:0] == 24'hFFFFFF);
endmodule
