`timescale 1ns / 1ps

module tb_simple;

    reg clk = 0;

    initial forever #5 clk = ~clk;

    initial begin
        $display("Time 0");
        # 5;
        $display("Time 5, clk=%b", clk);
        @(posedge clk);
        $display("After first posedge, clk=%b", clk);
        @(posedge clk);
        $display("After second posedge, clk=%b", clk);
        #100;
        $finish;
    end

endmodule
