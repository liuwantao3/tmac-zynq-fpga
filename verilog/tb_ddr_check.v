`timescale 1ns / 1ps
module tb_ddr_check;
    reg [63:0] ddr_mem [0:524287];

    task ddr_write32(input [31:0] addr, input [31:0] val);
        if (addr[2])
            ddr_mem[addr[31:3]][63:32] = val;
        else
            ddr_mem[addr[31:3]][31:0] = val;
    endtask

    function [31:0] ddr_read32(input [31:0] addr);
        if (addr[2])
            ddr_read32 = ddr_mem[addr[31:3]][63:32];
        else
            ddr_read32 = ddr_mem[addr[31:3]][31:0];
    endfunction

    integer i;
    initial begin
        for (i = 0; i < 10; i = i + 1) ddr_mem[i] = 64'h0;

        ddr_write32(32'h00100000, 32'h00000000);
        ddr_write32(32'h00100004, 32'h00000000);
        ddr_write32(32'h00100008, 32'h00101000);
        ddr_write32(32'h0010000C, 32'h00102000);
        ddr_write32(32'h00100010, 32'h0000000F);
        ddr_write32(32'h00100014, 32'h00000000);
        ddr_write32(32'h00100018, 32'h00000040);
        ddr_write32(32'h0010001C, 32'h00000000);

        $display("--- DDR write/read consistency check ---");
        $display("addr 0x00100000: 0x%08x (expect 0x00000000)", ddr_read32(32'h00100000));
        $display("addr 0x00100004: 0x%08x (expect 0x00000000)", ddr_read32(32'h00100004));
        $display("addr 0x00100008: 0x%08x (expect 0x00101000)", ddr_read32(32'h00100008));
        $display("addr 0x0010000C: 0x%08x (expect 0x00102000)", ddr_read32(32'h0010000C));
        $display("addr 0x00100010: 0x%08x (expect 0x0000000F)", ddr_read32(32'h00100010));
        $display("addr 0x00100014: 0x%08x (expect 0x00000000)", ddr_read32(32'h00100014));
        $display("addr 0x00100018: 0x%08x (expect 0x00000040)", ddr_read32(32'h00100018));
        $display("addr 0x0010001C: 0x%08x (expect 0x00000000)", ddr_read32(32'h0010001C));

        $display("--- Raw ddr_mem content (64-bit) ---");
        for (i = 0; i < 4; i = i + 1)
            $display("ddr_mem[0x%x] = 0x%016x", 32'h00200000 + i, ddr_mem[32'h00200000 + i]);

        // Simulate read slave's INCR burst at ARSIZE=2
        $display("--- Simulated read slave (burst @ 0x00100000, ARSIZE=2) ---");
        for (i = 0; i < 8; i = i + 1) begin
            reg [31:0] addr;
            reg [31:0] rd_idx;
            reg [31:0] rd_half;
            addr = 32'h00100000 + i * 4;
            rd_idx = addr[31:2];
            rd_half = rd_idx[0] ? ddr_mem[rd_idx >> 1][63:32] : ddr_mem[rd_idx >> 1][31:0];
            $display("  beat %0d (addr 0x%08x): rd_idx=0x%x [0]=%b half=0x%08x",
                i, addr, rd_idx, rd_idx[0], rd_half);
        end

        $finish;
    end
endmodule
