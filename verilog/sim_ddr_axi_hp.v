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

    // Read state machine
    reg [1:0] rstate;
    localparam R_IDLE = 0, R_RDATA = 1, R_WAIT = 2;

    reg [7:0] r_beat_cnt;
    reg [7:0] r_burst_len;
    reg [31:0] r_base_off;  // latched address offset

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rstate <= R_IDLE;
            s_axi_arready <= 0;
            s_axi_rvalid <= 0;
            s_axi_rlast <= 0;
            s_axi_rdata <= 0;
            r_beat_cnt <= 0;
            r_burst_len <= 0;
            r_base_off <= 0;
        end else begin
            case (rstate)
                R_IDLE: begin
                    s_axi_arready <= 1;
                    s_axi_rvalid <= 0;
                    if (s_axi_arvalid) begin
                        // $display("[DDR] Read AR: addr=0x%08x len=%0d", s_axi_araddr, s_axi_arlen);
                        s_axi_arready <= 0;
                        r_base_off <= s_axi_araddr - BASE_ADDR;
                        r_burst_len <= s_axi_arlen;
                        r_beat_cnt <= 0;
                        rstate <= R_RDATA;
                    end
                end

                R_RDATA: begin
                    if (!s_axi_rvalid || s_axi_rready) begin
                        s_axi_rdata <= {mem[r_base_off + r_beat_cnt * 8 + 7],
                                        mem[r_base_off + r_beat_cnt * 8 + 6],
                                        mem[r_base_off + r_beat_cnt * 8 + 5],
                                        mem[r_base_off + r_beat_cnt * 8 + 4],
                                        mem[r_base_off + r_beat_cnt * 8 + 3],
                                        mem[r_base_off + r_beat_cnt * 8 + 2],
                                        mem[r_base_off + r_beat_cnt * 8 + 1],
                                        mem[r_base_off + r_beat_cnt * 8]};
                        s_axi_rvalid <= 1;
                        s_axi_rlast <= (r_beat_cnt == r_burst_len);
                        if (r_beat_cnt == r_burst_len) begin
                            rstate <= R_WAIT;
                        end else begin
                            r_beat_cnt <= r_beat_cnt + 1;
                        end
                    end
                end

                R_WAIT: begin
                    if (s_axi_rready) begin
                        // $display("[DDR] Read burst done (addr offset=0x%08x, len=%0d)", r_base_off, r_burst_len);
                        s_axi_rvalid <= 0;
                        s_axi_rlast <= 0;
                        rstate <= R_IDLE;
                    end
                end
            endcase
        end
    end

    // Write state machine
    reg [1:0] wstate;
    localparam W_IDLE = 0, W_DATA = 1, W_RESP = 2;

    reg [31:0] w_addr_base;
    reg [7:0] w_beat_cnt;
    reg [7:0] w_burst_len;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wstate <= W_IDLE;
            s_axi_awready <= 0;
            s_axi_wready <= 0;
            s_axi_bvalid <= 0;
            s_axi_bresp <= 0;
            w_beat_cnt <= 0;
            w_burst_len <= 0;
            w_addr_base <= 0;
        end else begin
            case (wstate)
                W_IDLE: begin
                    s_axi_awready <= 1;
                    if (s_axi_awvalid) begin
                        s_axi_awready <= 0;
                        w_addr_base <= s_axi_awaddr - BASE_ADDR;
                        w_burst_len <= s_axi_awlen;
                        w_beat_cnt <= 0;
                        s_axi_wready <= 1;
                        wstate <= W_DATA;
                    end
                end

                W_DATA: begin
                    if (s_axi_wvalid) begin
                        if (s_axi_wstrb[0]) mem[w_addr_base + w_beat_cnt * 8]     <= s_axi_wdata[7:0];
                        if (s_axi_wstrb[1]) mem[w_addr_base + w_beat_cnt * 8 + 1] <= s_axi_wdata[15:8];
                        if (s_axi_wstrb[2]) mem[w_addr_base + w_beat_cnt * 8 + 2] <= s_axi_wdata[23:16];
                        if (s_axi_wstrb[3]) mem[w_addr_base + w_beat_cnt * 8 + 3] <= s_axi_wdata[31:24];
                        if (s_axi_wstrb[4]) mem[w_addr_base + w_beat_cnt * 8 + 4] <= s_axi_wdata[39:32];
                        if (s_axi_wstrb[5]) mem[w_addr_base + w_beat_cnt * 8 + 5] <= s_axi_wdata[47:40];
                        if (s_axi_wstrb[6]) mem[w_addr_base + w_beat_cnt * 8 + 6] <= s_axi_wdata[55:48];
                        if (s_axi_wstrb[7]) mem[w_addr_base + w_beat_cnt * 8 + 7] <= s_axi_wdata[63:56];
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

    // Debug readback — triggered only by dbg_addr changes
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
