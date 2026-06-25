module tb_minimal;
    reg clk = 0;
    always #5 clk = ~clk;
    initial begin
        $display("Hello at time %0t", $time);
        #50;
        $display("Hello at time %0t", $time);
        #50 $finish;
    end
endmodule
