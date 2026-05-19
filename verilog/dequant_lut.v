`timescale 1ns / 1ps

module dequant_lut (
    input  wire signed  [7:0] q8_val,
    input  wire       [15:0] scale,
    output reg  signed [15:0] result
);

    wire signed [23:0] product;

    assign product = q8_val * $signed({8'b0, scale});

    always @(*) begin
        if (product > 24'sd8388607)
            result = 16'sd32767;
        else if (product < -24'sd8388608)
            result = -16'sd32768;
        else
            result = product[23:8];
    end

endmodule
