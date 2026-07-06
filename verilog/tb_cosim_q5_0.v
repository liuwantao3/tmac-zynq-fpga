`timescale 1ns / 1ps

module tb_cosim_q5_0;

    reg         clk;
    reg         rst_n;
    reg         start;
    wire        done;
    wire        busy;

    reg         hdr_we;
    reg [2:0]   hdr_bank;
    reg [5:0]   hdr_addr;
    reg [7:0]   hdr_din;
    reg         qs_we;
    reg [1:0]   qs_addr;
    reg [31:0]  qs_din;
    reg         act_we;
    reg [9:0]   act_addr;
    reg [15:0]  act_din;
    reg [0:0]   res_addr;
    wire [47:0] res_dout;

    reg         sc_we;
    reg [2:0]   sc_addr;
    reg [15:0]  sc_din;

    reg [1:0]   core_id;

    matmul_q5_0_core u_core (
        .clk      (clk),
        .rst_n    (rst_n),
        .start    (start),
        .done     (done),
        .busy     (busy),
        .hdr_we   (hdr_we),
        .hdr_bank (hdr_bank),
        .hdr_addr (hdr_addr),
        .hdr_din  (hdr_din),
        .qs_we    (qs_we),
        .qs_addr  (qs_addr),
        .qs_din   (qs_din),
        .sc_we    (sc_we),
        .sc_addr  (sc_addr),
        .sc_din   (sc_din),
        .act_we   (act_we),
        .act_addr (act_addr),
        .act_din  (act_din),
        .res_addr (res_addr),
        .res_dout (res_dout),
        .core_id  (core_id),
        .dbg_tile_start(1'b0),
        .dbg_verbose(1'b0)
    );

    always #5 clk = ~clk;

    // ======================================================================
    // Binary tile dump format for Q5_0 (little-endian):
    //   Header: 4 × uint32_t = 16 B
    //     [0]: num_tiles
    //     [1..3]: reserved
    //   Per tile:
    //     blocks[4928]:  224 × Q5_0 blocks × 22 bytes (raw Q5_0 data)
    //     vec[1792]:     896 × int16_t   (activations, little-endian)
    //     row_inv[32]:   8 × float32      (row_inv, little-endian)
    //     expected[64]:  8 × int64_t      (fpga_sim reference result)
    // ======================================================================

    localparam MAX_FILE_BYTES = 8000000;
    localparam BLOCKS_PER_TILE = 224;  // 8 rows × 56 blocks
    localparam BLOCK_BYTES = 22;
    localparam TILE_BYTES = BLOCKS_PER_TILE * BLOCK_BYTES + 1792 + 32 + 64;

    reg [7:0]  fdata [0:MAX_FILE_BYTES-1];
    integer    f, file_bytes, num_tiles, tile;
    integer    i, b, w, core_run;
    integer    errors, total;
    integer    poll_count;
    reg [47:0] expected_val;
    reg [63:0] tmp64;
    reg [31:0] qs_word;

    initial begin
        $dumpfile("tb_cosim_q5_0.vcd");
        $dumpvars(0, tb_cosim_q5_0);

        clk = 0; rst_n = 0; start = 0;
        hdr_we = 0; qs_we = 0; act_we = 0; sc_we = 0; core_id = 0;
        errors = 0; total = 0;

        repeat (4) @(posedge clk);
        rst_n <= 1;
        repeat (2) @(posedge clk);

        f = $fopen("/tmp/cosim_tiles_q5_0.bin", "rb");
        if (f == 0) begin
            $display("========================================");
            $display("FATAL: Cannot open /tmp/cosim_tiles_q5_0.bin");
            $display("Run: echo PROMPT | ./tmac_gguf model.tmac --fpga-q5-0 --dump-tiles-q5-0 N");
            $display("========================================");
            $finish;
        end
        file_bytes = $fread(fdata, f);
        $fclose(f);
        $display("Read %0d bytes from /tmp/cosim_tiles_q5_0.bin", file_bytes);

        num_tiles = {fdata[3], fdata[2], fdata[1], fdata[0]};
        $display("Number of Q5_0 tiles: %0d", num_tiles);
        $display("========================================");

        if (file_bytes < 16 || file_bytes < 16 + num_tiles * TILE_BYTES) begin
            $display("FATAL: File too small (%0d bytes, need %0d)", file_bytes, 16 + num_tiles * TILE_BYTES);
            $finish;
        end

        for (tile = 0; tile < num_tiles; tile = tile + 1) begin
            integer base = 16 + tile * TILE_BYTES;
            integer ok;
            integer blocks_off = 0;
            integer vec_off    = BLOCKS_PER_TILE * BLOCK_BYTES;
            integer rowinv_off = vec_off + 1792;
            integer exp_off    = rowinv_off + 32;

            $display("Tile %0d/%0d:", tile + 1, num_tiles);

            // Pre-load acts (shared, same for all core runs)
            for (i = 0; i < 896; i = i + 1) begin
                @(negedge clk);
                act_we <= 1;
                act_addr <= i[9:0];
                act_din <= {fdata[base + vec_off + i * 2 + 1], fdata[base + vec_off + i * 2]};
            end
            @(negedge clk); act_we <= 0;

            // Pre-load scales (shared, same for all core runs)
            for (i = 0; i < 8; i = i + 1) begin
                @(negedge clk);
                sc_we <= 1;
                sc_addr <= i[2:0];
                sc_din <= 16'd256;
            end
            @(negedge clk); sc_we <= 0;

            ok = 1;

            // Run 4 cores (each handles 2 rows, 56 blocks)
            for (core_run = 0; core_run < 4; core_run = core_run + 1) begin
                integer blk_start = core_run * 56;

                core_id <= core_run;

                // Load headers for this core's 56 blocks
                for (b = 0; b < 56; b = b + 1) begin
                    integer blk_off = blk_start + b;

                    @(negedge clk);
                    hdr_we <= 1; hdr_bank <= 0; hdr_addr <= b[5:0];
                    hdr_din <= fdata[base + blocks_off + blk_off * 22];
                    @(posedge clk); hdr_we <= 0;
                    @(negedge clk);
                    hdr_we <= 1; hdr_bank <= 1; hdr_addr <= b[5:0];
                    hdr_din <= fdata[base + blocks_off + blk_off * 22 + 1];
                    @(posedge clk); hdr_we <= 0;
                    @(negedge clk);
                    hdr_we <= 1; hdr_bank <= 2; hdr_addr <= b[5:0];
                    hdr_din <= fdata[base + blocks_off + blk_off * 22 + 2];
                    @(posedge clk); hdr_we <= 0;
                    @(negedge clk);
                    hdr_we <= 1; hdr_bank <= 3; hdr_addr <= b[5:0];
                    hdr_din <= fdata[base + blocks_off + blk_off * 22 + 3];
                    @(posedge clk); hdr_we <= 0;
                    @(negedge clk);
                    hdr_we <= 1; hdr_bank <= 4; hdr_addr <= b[5:0];
                    hdr_din <= fdata[base + blocks_off + blk_off * 22 + 4];
                    @(posedge clk); hdr_we <= 0;
                    @(negedge clk);
                    hdr_we <= 1; hdr_bank <= 5; hdr_addr <= b[5:0];
                    hdr_din <= fdata[base + blocks_off + blk_off * 22 + 5];
                    @(posedge clk); hdr_we <= 0;
                end

                // Process each block: load qs + compute
                for (b = 0; b < 56; b = b + 1) begin
                    integer blk_off = blk_start + b;

                    // Load qs: 16 bytes → 4×32-bit words
                    for (w = 0; w < 4; w = w + 1) begin
                        qs_word = {fdata[base + blocks_off + blk_off * 22 + 6 + w*4 + 3],
                                   fdata[base + blocks_off + blk_off * 22 + 6 + w*4 + 2],
                                   fdata[base + blocks_off + blk_off * 22 + 6 + w*4 + 1],
                                   fdata[base + blocks_off + blk_off * 22 + 6 + w*4]};
                        @(negedge clk);
                        qs_we <= 1; qs_addr <= w[1:0]; qs_din <= qs_word;
                        @(posedge clk);
                        qs_we <= 0;
                    end

                    // Start compute
                    @(negedge clk); start <= 1;
                    @(negedge clk); start <= 0;

                    poll_count = 200000;
                    for (i = 0; i < 200000 && poll_count == 200000; i = i + 1) begin
                        @(posedge clk);
                        if (done) poll_count = i + 1;
                    end
                end

                // Check 2 rows
                for (i = 0; i < 2; i = i + 1) begin
                    integer row_idx = core_run * 2 + i;
                    @(negedge clk);
                    res_addr <= i[0:0];
                    @(posedge clk);
                    #1;

                    tmp64 = {fdata[base + exp_off + row_idx * 8 + 7], fdata[base + exp_off + row_idx * 8 + 6],
                            fdata[base + exp_off + row_idx * 8 + 5], fdata[base + exp_off + row_idx * 8 + 4],
                            fdata[base + exp_off + row_idx * 8 + 3], fdata[base + exp_off + row_idx * 8 + 2],
                            fdata[base + exp_off + row_idx * 8 + 1], fdata[base + exp_off + row_idx * 8]};
                    expected_val = tmp64[47:0];

                    if ($signed(res_dout) !== $signed(expected_val)) begin
                        if (ok) $display("  FAIL core_id=%0d: row[%0d] = %0d (expected %0d, diff %0d)",
                            core_run, i, $signed(res_dout), $signed(expected_val),
                            $signed(res_dout) - $signed(expected_val));
                        ok = 0;
                        errors = errors + 1;
                    end
                    total = total + 1;
                end
            end
            if (ok) $display("  PASS");
        end

        $display("========================================");
        if (errors === 0)
            $display("ALL %0d Q5_0 TILES PASSED (%0d checks)", num_tiles, total);
        else
            $display("%0d / %0d CHECKS FAILED across %0d tiles", errors, total, num_tiles);

        repeat (10) @(posedge clk);
        $finish;
    end

endmodule
