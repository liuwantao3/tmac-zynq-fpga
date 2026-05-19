`timescale 1ns / 1ps

module systolic_8x8 (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         start,
    input  wire  [15:0] A [0:7][0:7],  // 8x8 INT16 matrix A
    input  wire  [15:0] B [0:7][0:7],  // 8x8 INT16 matrix B
    output reg   [63:0] C [0:7][0:7],  // 8x8 INT64 output
    output reg          done
);

    reg  [15:0] a_reg [0:7][0:7];
    reg  [15:0] b_reg [0:7][0:7];
    reg  [63:0] c_reg [0:7][0:7];
    reg  [31:0] mac [0:7][0:7];

    integer i, j, t;
    reg [3:0] state;
    reg [3:0] step;

    localparam IDLE     = 0;
    localparam LOAD_A   = 1;
    localparam LOAD_B   = 2;
    localparam COMPUTE  = 3;
    localparam FINISH   = 4;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            done  <= 0;
            for (i = 0; i < 8; i = i + 1)
                for (j = 0; j < 8; j = j + 1) begin
                    a_reg[i][j] <= 0;
                    b_reg[i][j] <= 0;
                    c_reg[i][j] <= 0;
                    mac[i][j]   <= 0;
                end
        end else begin
            case (state)
                IDLE: begin
                    done <= 0;
                    if (start) begin
                        // Load A matrix
                        for (i = 0; i < 8; i = i + 1)
                            for (j = 0; j < 8; j = j + 1)
                                a_reg[i][j] <= A[i][j];
                        // Load B matrix
                        for (i = 0; i < 8; i = i + 1)
                            for (j = 0; j < 8; j = j + 1)
                                b_reg[i][j] <= B[i][j];
                        // Clear accumulators
                        for (i = 0; i < 8; i = i + 1)
                            for (j = 0; j < 8; j = j + 1)
                                c_reg[i][j] <= 0;
                        step <= 0;
                        state <= COMPUTE;
                    end
                end

                COMPUTE: begin
                    // Systolic: at each step t, multiply column t of A by row t of B
                    // A[i][t] * B[t][j] accumulated into C[i][j]
                    for (i = 0; i < 8; i = i + 1)
                        for (j = 0; j < 8; j = j + 1)
                            c_reg[i][j] <= c_reg[i][j] +
                                $signed(a_reg[i][step]) * $signed(b_reg[step][j]);
                    if (step == 7)
                        state <= FINISH;
                    else
                        step <= step + 1;
                end

                FINISH: begin
                    for (i = 0; i < 8; i = i + 1)
                        for (j = 0; j < 8; j = j + 1)
                            C[i][j] <= c_reg[i][j];
                    done  <= 1;
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule
