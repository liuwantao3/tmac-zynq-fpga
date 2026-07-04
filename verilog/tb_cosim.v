`timescale 1ns / 1ps

module tb_cosim;

    reg         clk;
    reg         rst_n;
    reg         start;
    wire        done;
    wire        busy;

    reg         wt_we;
    reg [8:0]   wt_addr;
    reg [63:0]  wt_din;
    reg         sc_we;
    reg [6:0]   sc_addr;
    reg [15:0]  sc_din;
    reg         act_we;
    reg [5:0]   act_addr;
    reg [15:0]  act_din;
    reg [5:0]   res_addr;
    wire [47:0] res_dout;

    matmul_q8_core u_core (
        .clk       (clk),
        .rst_n     (rst_n),
        .start     (start),
        .op_vecmul (1'b1),
        .done      (done),
        .busy      (busy),
        .wt_we     (wt_we),
        .wt_addr   (wt_addr),
        .wt_din    (wt_din),
        .sc_we     (sc_we),
        .sc_addr   (sc_addr),
        .sc_din    (sc_din),
        .act_we    (act_we),
        .act_addr  (act_addr),
        .act_din   (act_din),
        .res_addr  (res_addr),
        .res_dout  (res_dout)
    );

    always #5 clk = ~clk;

    // ======================================================================
    // Binary tile dump format (little-endian):
    //   Header: 4 × uint32_t = 16 B
    //     [0]: num_tiles
    //     [1..3]: reserved
    //   Per tile (4992 B):
    //     q8_W[4096]:  4096 × uint8_t  (col-major: [col][row], transposed
    //                   to bank-major 512×64-bit for BRAM load)
    //     scales[128]: 128 × uint16_t  (combined UQ8.8 per row×block)
    //     vec[64]:      64 × int16_t   (activations)
    //     expected[64]: 64 × int64_t   (fpga_sim reference result)
    // ======================================================================

    localparam MAX_FILE_BYTES = 600000;
    localparam TILE_BYTES = 4096 + 256 + 128 + 512;  // 4992

    reg [7:0]  fdata [0:MAX_FILE_BYTES-1];
    integer    f, file_bytes, num_tiles, tile;
    integer    i, k, row, col, addr, bank;
    integer    errors, total;
    integer    poll_count;
    reg [47:0] expected_val;
    reg [63:0] tmp64;

    initial begin
        $dumpfile("tb_cosim.vcd");
        $dumpvars(0, tb_cosim);

        clk = 0; rst_n = 0; start = 0;
        wt_we = 0; sc_we = 0; act_we = 0;
        errors = 0; total = 0;

        repeat (4) @(posedge clk);
        rst_n <= 1;
        repeat (2) @(posedge clk);

        // Read binary tile dump
        f = $fopen("/tmp/cosim_tiles.bin", "rb");
        if (f == 0) begin
            $display("========================================");
            $display("FATAL: Cannot open /tmp/cosim_tiles.bin");
            $display("Run: echo YOUR_PROMPT | ./tmac_gguf /tmp/model.tmac --fpga-q8 --dump-tiles N");
            $display("========================================");
            $finish;
        end
        file_bytes = $fread(fdata, f);
        $fclose(f);
        $display("Read %0d bytes from /tmp/cosim_tiles.bin", file_bytes);

        // Parse header
        num_tiles = {fdata[3], fdata[2], fdata[1], fdata[0]};
        $display("Number of tiles: %0d", num_tiles);
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

            // ---- Load weights: 512×64-bit words (bank-major) ----
            //   addr[8:0] = {bank[2:0], col[5:0]} = bank*64 + col
            //   word = {fdata[col*64 + bank*8 + 7], ..., fdata[col*64 + bank*8]}
            //   i.e. 8 consecutive rows for same column, packed into one 64-bit word
            for (bank = 0; bank < 8; bank = bank + 1) begin
                for (col = 0; col < 64; col = col + 1) begin
                    @(negedge clk);
                    wt_we <= 1;
                    wt_addr <= bank * 64 + col;
                    wt_din <= {fdata[base + col * 64 + bank * 8 + 7],
                               fdata[base + col * 64 + bank * 8 + 6],
                               fdata[base + col * 64 + bank * 8 + 5],
                               fdata[base + col * 64 + bank * 8 + 4],
                               fdata[base + col * 64 + bank * 8 + 3],
                               fdata[base + col * 64 + bank * 8 + 2],
                               fdata[base + col * 64 + bank * 8 + 1],
                               fdata[base + col * 64 + bank * 8]};
                end
            end
            @(negedge clk); wt_we <= 0;

            // ---- Load scales: 128 writes (addr = 0..127) ----
            for (i = 0; i < 128; i = i + 1) begin
                @(negedge clk);
                sc_we <= 1;
                sc_addr <= i;
                sc_din <= {fdata[base + 4096 + i * 2 + 1], fdata[base + 4096 + i * 2]};
            end
            @(negedge clk); sc_we <= 0;

            // ---- Load activations: 64 writes (addr = 0..63) ----
            for (i = 0; i < 64; i = i + 1) begin
                @(negedge clk);
                act_we <= 1;
                act_addr <= i;
                act_din <= {fdata[base + 4096 + 256 + i * 2 + 1], fdata[base + 4096 + 256 + i * 2]};
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
                integer exp_base = base + 4096 + 256 + 128;
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
            $display("ALL %0d TILES PASSED (%0d checks)", num_tiles, total);
        else
            $display("%0d / %0d CHECKS FAILED across %0d tiles", errors, total, num_tiles);

        repeat (10) @(posedge clk);
        $finish;
    end

endmodule
