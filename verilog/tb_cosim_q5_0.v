`timescale 1ns / 1ps

module tb_cosim_q5_0;

    reg         clk;
    reg         rst_n;
    reg         start;
    wire        done;
    wire        busy;

    reg         wt_we;
    reg [12:0]  wt_addr;
    reg [7:0]   wt_din;
    reg         act_we;
    reg [9:0]   act_addr;
    reg [15:0]  act_din;
    reg [2:0]   res_addr;
    wire [47:0] res_dout;

    reg         sc_we;
    reg [2:0]   sc_addr;
    reg [15:0]  sc_din;

    matmul_q5_0_core u_core (
        .clk      (clk),
        .rst_n    (rst_n),
        .start    (start),
        .done     (done),
        .busy     (busy),
        .wt_we    (wt_we),
        .wt_addr  (wt_addr),
        .wt_din   (wt_din),
        .sc_we    (sc_we),
        .sc_addr  (sc_addr),
        .sc_din   (sc_din),
        .act_we   (act_we),
        .act_addr (act_addr),
        .act_din  (act_din),
        .res_addr (res_addr),
        .res_dout (res_dout)
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
    localparam BLOCKS_PER_TILE = 224;
    localparam BLOCK_BYTES = 22;
    localparam TILE_BYTES = BLOCKS_PER_TILE * BLOCK_BYTES + 1792 + 32 + 64;

    reg [7:0]  fdata [0:MAX_FILE_BYTES-1];
    integer    f, file_bytes, num_tiles, tile;
    integer    i, byte_idx;
    integer    errors, total;
    integer    poll_count;
    reg [47:0] expected_val;
    reg [63:0] tmp64;

    initial begin
        $dumpfile("tb_cosim_q5_0.vcd");
        $dumpvars(0, tb_cosim_q5_0);

        clk = 0; rst_n = 0; start = 0;
        wt_we = 0; act_we = 0; sc_we = 0;
        errors = 0; total = 0;

        repeat (4) @(posedge clk);
        rst_n <= 1;
        repeat (2) @(posedge clk);

        f = $fopen("/tmp/cosim_tiles_q5_0.bin", "rb");
        if (f == 0) begin
            $display("========================================");
            $display("FATAL: Cannot open /tmp/cosim_tiles_q5_0.bin");
            $display("Run: echo PROMPT | ./tmac_gguf model.tmac --fpga-q4k --dump-tiles-q5-0 N");
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

            for (byte_idx = 0; byte_idx < BLOCKS_PER_TILE * BLOCK_BYTES; byte_idx = byte_idx + 1) begin
                @(negedge clk);
                wt_we <= 1;
                wt_addr <= byte_idx[12:0];
                wt_din <= fdata[base + blocks_off + byte_idx];
            end
            @(negedge clk); wt_we <= 0;

            for (i = 0; i < 896; i = i + 1) begin
                @(negedge clk);
                act_we <= 1;
                act_addr <= i[9:0];
                act_din <= {fdata[base + vec_off + i * 2 + 1], fdata[base + vec_off + i * 2]};
            end
            @(negedge clk); act_we <= 0;

            for (i = 0; i < 8; i = i + 1) begin
                @(negedge clk);
                sc_we <= 1;
                sc_addr <= i[2:0];
                sc_din <= 16'd256;
            end
            @(negedge clk); sc_we <= 0;

            @(negedge clk); start <= 1;
            @(negedge clk); start <= 0;

            poll_count = 200000;
            for (i = 0; i < 200000 && poll_count == 200000; i = i + 1) begin
                @(posedge clk);
                if (done) poll_count = i + 1;
            end

            ok = 1;
            for (i = 0; i < 8; i = i + 1) begin
                @(negedge clk);
                res_addr <= i[2:0];
                @(posedge clk);
                #1;

                tmp64 = {fdata[base + exp_off + i * 8 + 7], fdata[base + exp_off + i * 8 + 6],
                        fdata[base + exp_off + i * 8 + 5], fdata[base + exp_off + i * 8 + 4],
                        fdata[base + exp_off + i * 8 + 3], fdata[base + exp_off + i * 8 + 2],
                        fdata[base + exp_off + i * 8 + 1], fdata[base + exp_off + i * 8]};
                expected_val = tmp64[47:0];

                if ($signed(res_dout) !== $signed(expected_val)) begin
                    if (ok) $display("  FAIL: row[%0d] = %0d (expected %0d, diff %0d)", i, $signed(res_dout), $signed(expected_val), $signed(res_dout) - $signed(expected_val));
                    ok = 0;
                    errors = errors + 1;
                end
                total = total + 1;
            end
            if (ok) $display("  PASS (%0d poll cycles)", poll_count);
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