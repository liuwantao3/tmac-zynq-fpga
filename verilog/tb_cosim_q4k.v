`timescale 1ns / 1ps

module tb_cosim_q4k;

    reg         clk;
    reg         rst_n;
    reg         start;
    wire        done;
    wire        busy;
    wire        decode_busy;

    reg         wt_we;
    reg [12:0]  wt_addr;
    reg [7:0]   wt_din;
    reg         act_we;
    reg [5:0]   act_addr;
    reg [15:0]  act_din;
    reg [5:0]   res_addr;
    wire [47:0] res_dout;

    matmul_q4k_core u_core (
        .clk              (clk),
        .rst_n            (rst_n),
        .start            (start),
        .op_vecmul        (1'b1),
        .done             (done),
        .busy             (busy),
        .wt_we            (wt_we),
        .wt_addr          (wt_addr),
        .wt_din           (wt_din),
        .sc_we            (1'b0),
        .sc_addr          (7'd0),
        .sc_din           (16'd0),
        .act_we           (act_we),
        .act_addr         (act_addr),
        .act_din          (act_din),
        .res_addr         (res_addr),
        .res_dout         (res_dout),
        .mode_block_load  (1'b1),
        .decode_busy      (decode_busy)
    );

    always #5 clk = ~clk;

    // ======================================================================
    // Binary tile dump format for Q4_K (little-endian):
    //   Header: 4 × uint32_t = 16 B
    //     [0]: num_tiles
    //     [1..3]: reserved
    //   Per tile (2944 B):
    //     blocks[2304]: 16 × Q4_K blocks × 144 bytes (raw Q4_K data)
    //     vec[128]:     64 × int16_t  (activations, little-endian)
    //     expected[512]: 64 × int64_t (fpga_sim reference result)
    // ======================================================================

    localparam MAX_FILE_BYTES = 600000;
    localparam TILE_BYTES = 2304 + 128 + 512;  // 2944

    reg [7:0]  fdata [0:MAX_FILE_BYTES-1];
    integer    f, file_bytes, num_tiles, tile;
    integer    i, k, bi, byte_idx;
    integer    errors, total;
    integer    poll_count;
    reg [47:0] expected_val;
    reg [63:0] tmp64;

    initial begin
        $dumpfile("tb_cosim_q4k.vcd");
        $dumpvars(0, tb_cosim_q4k);

        clk = 0; rst_n = 0; start = 0;
        wt_we = 0; act_we = 0;
        errors = 0; total = 0;

        repeat (4) @(posedge clk);
        rst_n <= 1;
        repeat (2) @(posedge clk);

        // Read binary tile dump
        f = $fopen("/tmp/cosim_tiles_q4k.bin", "rb");
        if (f == 0) begin
            $display("========================================");
            $display("FATAL: Cannot open /tmp/cosim_tiles_q4k.bin");
            $display("Run: echo PROMPT | ./tmac_gguf model.tmac --fpga-q4k --dump-tiles-q4k N");
            $display("========================================");
            $finish;
        end
        file_bytes = $fread(fdata, f);
        $fclose(f);
        $display("Read %0d bytes from /tmp/cosim_tiles_q4k.bin", file_bytes);

        // Parse header
        num_tiles = {fdata[3], fdata[2], fdata[1], fdata[0]};
        $display("Number of Q4_K tiles: %0d", num_tiles);
        $display("========================================");

        if (file_bytes < 16 || file_bytes < 16 + num_tiles * TILE_BYTES) begin
            $display("FATAL: File too small (%0d bytes, need %0d)", file_bytes, 16 + num_tiles * TILE_BYTES);
            $finish;
        end

        // Process each tile
        for (tile = 0; tile < num_tiles; tile = tile + 1) begin
            integer base = 16 + tile * TILE_BYTES;
            integer ok;

            $display("Tile %0d/%0d:", tile + 1, num_tiles);

            // ---- Load Q4_K blocks: 2304 bytes sequential address write ----
            // wt_addr[11:0] maps to block_buf[blk_load_ptr[10:0]]
            // Sequential 0..2303 writes block_buf[0..2303]
            for (byte_idx = 0; byte_idx < 2304; byte_idx = byte_idx + 1) begin
                @(negedge clk);
                wt_we <= 1;
                wt_addr <= byte_idx[11:0];
                wt_din <= fdata[base + byte_idx];
            end
            @(negedge clk); wt_we <= 0;

            // ---- Load activations: 64 writes (addr = 0..63) ----
            for (i = 0; i < 64; i = i + 1) begin
                @(negedge clk);
                act_we <= 1;
                act_addr <= i;
                act_din <= {fdata[base + 2304 + i * 2 + 1], fdata[base + 2304 + i * 2]};
            end
            @(negedge clk); act_we <= 0;

            // ---- Start computation ----
            @(negedge clk); start <= 1;
            @(negedge clk); start <= 0;

            // ---- Wait for done (poll with timeout) ----
            poll_count = 2000;
            for (i = 0; i < 2000 && poll_count == 2000; i = i + 1) begin
                @(posedge clk);
                if (done) poll_count = i + 1;
            end

            // ---- Read and compare results ----
            ok = 1;
            for (i = 0; i < 64; i = i + 1) begin
                integer exp_base = base + 2304 + 128;
                @(negedge clk);
                res_addr <= i;
                @(posedge clk);
                #1;

                // Read int64_t expected value (little-endian), take low 48 bits
                tmp64 = {fdata[exp_base + i * 8 + 7], fdata[exp_base + i * 8 + 6],
                         fdata[exp_base + i * 8 + 5], fdata[exp_base + i * 8 + 4],
                         fdata[exp_base + i * 8 + 3], fdata[exp_base + i * 8 + 2],
                         fdata[exp_base + i * 8 + 1], fdata[exp_base + i * 8]};
                expected_val = tmp64[47:0];

                if (res_dout !== expected_val) begin
                    if (ok) $display("  FAIL: row[%0d] = %0d (expected %0d, raw %0d)", i, res_dout, expected_val, tmp64);
                    ok = 0;
                    errors = errors + 1;
                end
                total = total + 1;
            end
            if (ok) $display("  PASS (%0d poll cycles)", poll_count);
        end

        // ---- Summary ----
        $display("========================================");
        if (errors === 0)
            $display("ALL %0d Q4_K TILES PASSED (%0d checks)", num_tiles, total);
        else
            $display("%0d / %0d CHECKS FAILED across %0d tiles", errors, total, num_tiles);

        repeat (10) @(posedge clk);
        $finish;
    end

endmodule