`timescale 1ns / 1ps

// AXI HP Slave BFM + DDR Memory Model for iVerilog simulation
// Base address: 0x18000000 (Phase B region)
// Size: 256 MB (0x10000000 bytes)

module sim_ddr_axi_hp (
    input  wire         clk,
    input  wire         rst_n,

    // AXI HP read (slave)
    input  wire [31:0]  s_axi_araddr,
    input  wire         s_axi_arvalid,
    output reg          s_axi_arready,
    input  wire [7:0]   s_axi_arlen,
    input  wire [2:0]   s_axi_arsize,
    input  wire [1:0]   s_axi_arburst,
    output reg [63:0]   s_axi_rdata,
    output reg          s_axi_rvalid,
    input  wire         s_axi_rready,
    output reg          s_axi_rlast,

    // AXI HP write (slave)
    input  wire [31:0]  s_axi_awaddr,
    input  wire         s_axi_awvalid,
    output reg          s_axi_awready,
    input  wire [7:0]   s_axi_awlen,
    input  wire [2:0]   s_axi_awsize,
    input  wire [1:0]   s_axi_awburst,
    input  wire [63:0]  s_axi_wdata,
    input  wire         s_axi_wvalid,
    output reg          s_axi_wready,
    input  wire         s_axi_wlast,
    input  wire [7:0]   s_axi_wstrb,
    output reg          s_axi_bvalid,
    input  wire         s_axi_bready,
    output reg [1:0]    s_axi_bresp,
    output reg  [5:0]   s_axi_bid,

    // AXI HP response signals (optional, tied to 0)
    output reg  [1:0]   s_axi_rresp,
    output reg  [5:0]   s_axi_rid,

    // Debug readback (for testbench verification)
    input  wire [31:0]  dbg_addr,
    output reg [63:0]   dbg_data
);

    parameter BASE_ADDR = 32'h1800_0000;
    parameter MEM_BYTES = 256 * 1024 * 1024;

    reg [7:0] mem [0:MEM_BYTES-1];

    // Load DDR image from binary file
    integer fd;
    initial begin
        s_axi_rresp = 0;
        s_axi_rid   = 0;
        s_axi_bid   = 0;
        fd = $fopen("/tmp/tb_phaseb.bin", "rb");
        if (fd) begin
            $display("[DDR] Loading /tmp/tb_phaseb.bin...");
            $fread(mem, fd);
            $fclose(fd);
            $display("[DDR] Loaded /tmp/tb_phaseb.bin");
        end else begin
            $display("[DDR] WARNING: /tmp/tb_phaseb.bin not found");
        end
    end

    // Beat stride (bytes per beat) based on ARSIZE/AWSIZE
    function [2:0] beat_stride;
        input [2:0] size;
        begin
            case (size)
                3'd0: beat_stride = 1;
                3'd1: beat_stride = 2;
                3'd2: beat_stride = 4;
                3'd3: beat_stride = 8;
                default: beat_stride = 8;
            endcase
        end
    endfunction

    // Read state machine ? AXI burst BFM.
    //
    // Uses BLOCKING assignments (=) for rdata/rvalid so the read master
    // sees fresh values immediately within the same evaluation cycle.
    //
    // Flow per beat:
    //   R_IDLE ? R_START (present beat N data, rvalid=1)
    //         ? R_HOLD   (wait for rready handshake; on handshake: drop rvalid=0)
    //         ? R_NEXT   (present beat N+1 data, rvalid=1) or R_DONE
    reg [2:0] rstate;
    localparam R_IDLE  = 3'd0;
    localparam R_START = 3'd1;
    localparam R_HOLD  = 3'd2;
    localparam R_NEXT  = 3'd3;
    localparam R_DONE  = 3'd4;

    reg [7:0] r_beat_cnt;
    reg [7:0] r_burst_len;
    reg [31:0] r_base_off;  // latched address offset
    reg [2:0] r_stride;

    // Internal data signals (blocking-assigned)
    reg [63:0] rdata_int;
    reg        rvalid_int;

    // Drive outputs from internal signals using continuous assignments
    assign s_axi_rdata = rdata_int;
    assign s_axi_rvalid = rvalid_int;

    // Fetch 32-bit word from mem at byte offset
    function [31:0] mem_word;
        input [31:0] off;
        begin
            mem_word = {mem[off + 3], mem[off + 2], mem[off + 1], mem[off]};
        end
    endfunction

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rstate     <= R_IDLE;
            s_axi_arready <= 0;
            s_axi_rlast  <= 0;
            s_axi_rresp  <= 0;
            s_axi_rid    <= 0;
            r_beat_cnt   <= 0;
            r_burst_len  <= 0;
            r_base_off   <= 0;
            r_stride     <= 4;
            rdata_int    <= 0;
            rvalid_int   <= 0;
        end else begin
            case (rstate)
                R_IDLE: begin
                    s_axi_arready <= 1;
                    rvalid_int    <= 0;
                    if (s_axi_arvalid) begin
                        s_axi_arready <= 0;
                        r_base_off    <= s_axi_araddr - BASE_ADDR;
                        r_burst_len   <= s_axi_arlen;
                        r_stride      <= beat_stride(s_axi_arsize);
                        r_beat_cnt    <= 0;
                        rstate        <= R_START;
                    end
                end

                R_START: begin
                    // Present beat N data with rvalid=1
                    rdata_int  <= {32'd0, mem_word(r_base_off + r_beat_cnt * r_stride)};
                    rvalid_int <= 1;
                    s_axi_rlast <= (r_beat_cnt == r_burst_len);
                    if (r_beat_cnt == 0) begin
                        $display("[DDR] RD beat0: addr=0x%08x off=%0d data=%02x %02x %02x %02x",
                            s_axi_araddr, r_base_off,
                            mem[r_base_off + 3], mem[r_base_off + 2],
                            mem[r_base_off + 1], mem[r_base_off]);
                    end
                    $display("[DDR] START @%0t beat=%0d off=%0d data=0x%08x",
                        $time, r_beat_cnt, r_base_off + r_beat_cnt * r_stride,
                        mem_word(r_base_off + r_beat_cnt * r_stride));
                    rstate <= R_HOLD;
                end

                R_HOLD: begin
                    // Keep data stable; on rready=1 handshake, drop rvalid.
                    $display("[DDR] HOLD @%0t beat=%0d rready=%b rvalid=%b rstate=%d",
                        $time, r_beat_cnt, s_axi_rready, rvalid_int, rstate);
                    if (s_axi_rready) begin
                        rvalid_int <= 0;  // drop rvalid in next cycle
                        $display("[DDR] HOLD ADVANCE @%0t beat=%0d->%0d",
                            $time, r_beat_cnt, r_beat_cnt+1);
                        if (r_beat_cnt == r_burst_len) begin
                            rstate <= R_DONE;
                        end else begin
                            r_beat_cnt <= r_beat_cnt + 1;
                            rstate <= R_NEXT;
                        end
                    end
                end

                R_NEXT: begin
                    // Present next beat data with rvalid=1
                    rdata_int  <= {32'd0, mem_word(r_base_off + r_beat_cnt * r_stride)};
                    rvalid_int <= 1;
                    s_axi_rlast <= (r_beat_cnt == r_burst_len);
                    $display("[DDR] NEXT beat=%0d off=%0d data=0x%08x",
                        r_beat_cnt, r_base_off + r_beat_cnt * r_stride,
                        mem_word(r_base_off + r_beat_cnt * r_stride));
                    rstate <= R_HOLD;
                end

                R_DONE: begin
                    rvalid_int  <= 0;
                    s_axi_rlast <= 0;
                    rstate      <= R_IDLE;
                end
            endcase
        end
    end

    // Write state machine ? handles AWSIZE=2 single-beat writes
    reg [1:0] wstate;
    localparam W_IDLE = 0, W_DATA = 1, W_RESP = 2;

    reg [31:0] w_addr_base;
    reg [7:0] w_beat_cnt;
    reg [7:0] w_burst_len;
    reg [2:0] w_stride;
    wire [31:0] w_beat_addr = w_addr_base + w_beat_cnt * w_stride;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wstate <= W_IDLE;
            s_axi_awready <= 0;
            s_axi_wready <= 0;
            s_axi_bvalid <= 0;
            s_axi_bresp <= 0;
            s_axi_bid   <= 0;
            w_beat_cnt <= 0;
            w_burst_len <= 0;
            w_addr_base <= 0;
            w_stride <= 4;
        end else begin
            case (wstate)
                W_IDLE: begin
                    s_axi_awready <= 1;
                    if (s_axi_awvalid) begin
                        w_addr_base <= s_axi_awaddr - BASE_ADDR;
                        w_burst_len <= s_axi_awlen;
                        w_stride <= beat_stride(s_axi_awsize);
                        w_beat_cnt <= 0;
                        s_axi_wready <= 1;
                        wstate <= W_DATA;
                    end
                end

                W_DATA: begin
                    if (s_axi_wvalid) begin
                        // For AWSIZE=2 on 64-bit bus, per-beat address A[2] selects byte lanes:
                        //   A[2]=0 ? WDATA[31:0], WSTRB[3:0] ? write at beat_addr+0..3
                        //   A[2]=1 ? WDATA[63:32], WSTRB[7:4] ? write at beat_addr+0..3
                        // beat_addr = w_addr_base + w_beat_cnt * w_stride
                        if (w_beat_addr[2]) begin
                            if (s_axi_wstrb[4]) mem[w_beat_addr + 0] <= s_axi_wdata[39:32];
                            if (s_axi_wstrb[5]) mem[w_beat_addr + 1] <= s_axi_wdata[47:40];
                            if (s_axi_wstrb[6]) mem[w_beat_addr + 2] <= s_axi_wdata[55:48];
                            if (s_axi_wstrb[7]) mem[w_beat_addr + 3] <= s_axi_wdata[63:56];
                        end else begin
                            if (s_axi_wstrb[0]) mem[w_beat_addr + 0] <= s_axi_wdata[7:0];
                            if (s_axi_wstrb[1]) mem[w_beat_addr + 1] <= s_axi_wdata[15:8];
                            if (s_axi_wstrb[2]) mem[w_beat_addr + 2] <= s_axi_wdata[23:16];
                            if (s_axi_wstrb[3]) mem[w_beat_addr + 3] <= s_axi_wdata[31:24];
                        end
                        if (s_axi_wlast) begin
                            s_axi_wready <= 0;
                            s_axi_bvalid <= 1;
                            s_axi_bresp <= 2'b00;
                            wstate <= W_RESP;
                        end else begin
                            w_beat_cnt <= w_beat_cnt + 1;
                        end
                    end
                end

                W_RESP: begin
                    if (s_axi_bready) begin
                        s_axi_bvalid <= 0;
                        wstate <= W_IDLE;
                    end
                end
            endcase
        end
    end

    // Debug readback ? triggered only by dbg_addr changes
    always @(dbg_addr) begin
        if (dbg_addr >= BASE_ADDR) begin
            integer off;
            off = dbg_addr - BASE_ADDR;
            if (off + 7 < MEM_BYTES) begin
                dbg_data = {mem[off + 7], mem[off + 6], mem[off + 5], mem[off + 4],
                            mem[off + 3], mem[off + 2], mem[off + 1], mem[off]};
            end else begin
                dbg_data = 64'b0;
            end
        end else begin
            dbg_data = 64'b0;
        end
    end

endmodule
