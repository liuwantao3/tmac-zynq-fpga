`timescale 1ns / 1ps

module tb_cosim_q6_k;

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
    reg [7:0]   act_addr;
    reg [15:0]  act_din;
    reg [4:0]   res_addr;
    wire [47:0] res_dout;

    reg         sc_we;
    reg [4:0]   sc_addr;
    reg [15:0]  sc_din;

    wire        mode_block_load = 1'b1;

    matmul_q6_k_core u_core (
        .clk              (clk),
        .rst_n            (rst_n),
        .start            (start),
        .op_vecmul        (1'b1),
        .done             (done),
        .busy             (busy),
        .wt_we            (wt_we),
        .wt_addr          (wt_addr),
        .wt_din           (wt_din),
        .sc_we            (sc_we),
        .sc_addr          (sc_addr),
        .sc_din           (sc_din),
        .act_we           (act_we),
        .act_addr         (act_addr),
        .act_din           (act_din),
        .res_addr         (res_addr),
        .res_dout         (res_dout),
        .mode_block_load  (mode_block_load),
        .decode_busy      (decode_busy)
    );

    always #5 clk = ~clk;

    // ======================================================================
    // Binary tile dump format for Q6_K (little-endian):
    //   Header: 4 × uint32_t = 16 B
    //     [0]: num_tiles
    //     [1..3]: reserved
    //   Per tile:
    //     blocks[6720]:   32 × Q6_K blocks × 210 bytes (raw Q6_K data)
    //     vec[512]:       256 × int16_t  (activations, little-endian)
    //     row_inv[128]:   32 × float32    (row_inv, little-endian)
    //     expected[256]:  32 × int64_t    (fpga_sim reference result)
    // ======================================================================

    localparam MAX_FILE_BYTES = 8000000;
    localparam BLOCKS_PER_TILE = 32;
    localparam BLOCK_BYTES = 210;
    localparam TILE_BYTES = BLOCKS_PER_TILE * BLOCK_BYTES + 512 + 128 + 256;

    reg [7:0]  fdata [0:MAX_FILE_BYTES-1];
    integer    f, file_bytes, num_tiles, tile;
    integer    i, byte_idx;
    integer    errors, total;
    integer    poll_count;
    reg [47:0] expected_val;
    reg [63:0] tmp64;

    initial begin
        $dumpfile("tb_cosim_q6_k.vcd");
        $dumpvars(0, tb_cosim_q6_k);

        clk = 0; rst_n = 0; start = 0;
        wt_we = 0; act_we = 0; sc_we = 0;
        errors = 0; total = 0;

        repeat (4) @(posedge clk);
        rst_n <= 1;
        repeat (2) @(posedge clk);

        f = $fopen("/tmp/cosim_tiles_q6_k.bin", "rb");
        if (f == 0) begin
            $display("========================================");
            $display("FATAL: Cannot open /tmp/cosim_tiles_q6_k.bin");
            $display("Run: echo PROMPT | ./tmac_gguf model.tmac --fpga-q4k --dump-tiles-q6-k N");
            $display("========================================");
            $finish;
        end
        file_bytes = $fread(fdata, f);
        $fclose(f);
        $display("Read %0d bytes from /tmp/cosim_tiles_q6_k.bin", file_bytes);

        num_tiles = {fdata[3], fdata[2], fdata[1], fdata[0]};
        $display("Number of Q6_K tiles: %0d", num_tiles);
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
            integer rowinv_off = vec_off + 512;
            integer exp_off    = rowinv_off + 128;

            $display("Tile %0d/%0d:", tile + 1, num_tiles);

            for (byte_idx = 0; byte_idx < BLOCKS_PER_TILE * BLOCK_BYTES; byte_idx = byte_idx + 1) begin
                @(negedge clk);
                wt_we <= 1;
                wt_addr <= byte_idx[12:0];
                wt_din <= fdata[base + blocks_off + byte_idx];
            end
            @(negedge clk); wt_we <= 0;

            for (i = 0; i < 256; i = i + 1) begin
                @(negedge clk);
                act_we <= 1;
                act_addr <= i[7:0];
                act_din <= {fdata[base + vec_off + i * 2 + 1], fdata[base + vec_off + i * 2]};
            end
            @(negedge clk); act_we <= 0;

            for (i = 0; i < 32; i = i + 1) begin
                reg [31:0] inv_bits;
                inv_bits = {fdata[base + rowinv_off + i * 4 + 3],
                            fdata[base + rowinv_off + i * 4 + 2],
                            fdata[base + rowinv_off + i * 4 + 1],
                            fdata[base + rowinv_off + i * 4]};
                @(negedge clk);
                sc_we <= 1;
                sc_addr <= i[4:0];
                sc_din <= 16'd32767;
            end
            @(negedge clk); sc_we <= 0;

            @(negedge clk); start <= 1;
            @(negedge clk); start <= 0;

            poll_count = 0;
            for (i = 0; i < 200000; i = i + 1) begin
                if (done) begin
                    poll_count = i;
                    break;
                end
                @(posedge clk);
            end

            ok = 1;
            for (i = 0; i < 32; i = i + 1) begin
                @(negedge clk);
                res_addr <= i[4:0];
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
            $display("ALL %0d Q6_K TILES PASSED (%0d checks)", num_tiles, total);
        else
            $display("%0d / %0d CHECKS FAILED across %0d tiles", errors, total, num_tiles);

        repeat (10) @(posedge clk);
        $finish;
    end

endmodule