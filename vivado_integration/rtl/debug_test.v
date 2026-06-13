`timescale 1ns / 1ps
module debug_test (
    input  wire         clk,
    input  wire         rst_n
);
    // Internal oscillator for debug hub clock when clk is not connected
    (* KEEP = "TRUE" *) wire int_clk;
    LUT1 #(.INIT(2'b10)) lut_osc (.I0(int_clk), .O(int_clk_w));
    BUFG bufg_osc (.I(int_clk_w), .O(int_clk));

    reg [31:0] counter = 0;
    always @(posedge int_clk or negedge rst_n) begin
        if (!rst_n) counter <= 0;
        else        counter <= counter + 1;
    end

    (* MARK_DEBUG = "TRUE" *) wire dbg_tick = (counter[23:0] == 24'hFFFFFF);
endmodule
