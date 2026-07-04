`timescale 1ns / 1ps

// AXI HP Write Master — byte-stream input, INCR burst output, AWSIZE=2
module axihp_write_master (
    input  wire         clk,
    input  wire         rst_n,

    input  wire         start,
    input  wire [31:0]  dst_addr,
    input  wire [7:0]   burst_len,
    output reg          done,
    output reg          busy,
    output wire [2:0]   dbg_state,

    input  wire [7:0]   data_in,
    input  wire         data_valid,
    output reg          data_ready,

    output reg  [5:0]   m_axi_awid,
    output reg  [31:0]  m_axi_awaddr,
    output reg          m_axi_awvalid,
    input  wire         m_axi_awready,
    output reg  [7:0]   m_axi_awlen,
    output reg  [2:0]   m_axi_awsize,
    output reg  [1:0]   m_axi_awburst,
    output reg  [1:0]   m_axi_awlock,
    output reg  [3:0]   m_axi_awcache,
    output reg  [2:0]   m_axi_awprot,
    output reg  [5:0]   m_axi_wid,
    output reg  [63:0]  m_axi_wdata,
    output reg          m_axi_wvalid,
    input  wire         m_axi_wready,
    output reg          m_axi_wlast,
    output reg  [7:0]   m_axi_wstrb,
    input  wire         m_axi_bvalid,
    output reg          m_axi_bready,
    input  wire  [1:0]  m_axi_bresp,
    input  wire  [5:0]  m_axi_bid
);

    localparam [2:0] IDLE   = 0;
    localparam [2:0] FILL   = 1;
    localparam [2:0] XFER   = 2;
    localparam [2:0] B_WAIT = 3;
    localparam [2:0] FINISH = 4;

    reg [2:0] state;
    reg [7:0] bytebuf [0:63];
    reg [6:0] fill_cnt;
    reg [7:0] beat_cnt;
    reg [7:0] total_beats;
    reg [31:0] base_addr;
    reg       aw_grant;
    reg       w_last_done;

    assign dbg_state = state;

    wire [6:0] fill_target = {total_beats[5:0], 2'b00} - 7'd1;

    // cur_beat uses registered beat_cnt (no speculative advance from wready)
    wire [7:0]  cur_beat = beat_cnt;
    wire [31:0] wdata_beat = {bytebuf[{cur_beat[5:0], 2'b00} + 3],
                              bytebuf[{cur_beat[5:0], 2'b00} + 2],
                              bytebuf[{cur_beat[5:0], 2'b00} + 1],
                              bytebuf[{cur_beat[5:0], 2'b00} + 0]};
    wire [63:0] wdata_steer = (base_addr[2] ^ cur_beat[0]) ? {wdata_beat, 32'd0} : {32'd0, wdata_beat};
    wire [7:0]  wstrb_steer = (base_addr[2] ^ cur_beat[0]) ? 8'hF0 : 8'h0F;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= IDLE;
            busy         <= 0;
            done         <= 0;
            data_ready   <= 0;
            m_axi_awvalid <= 0;
            m_axi_wvalid <= 0;
            m_axi_bready <= 0;
            m_axi_awid   <= 6'd0;
            m_axi_awlock <= 2'd0;
            m_axi_awcache <= 4'b0011;
            m_axi_awprot <= 3'd0;
            m_axi_wid    <= 6'd0;
            m_axi_wlast  <= 0;
            m_axi_wdata  <= 0;
            m_axi_wstrb  <= 0;
            fill_cnt     <= 0;
            beat_cnt     <= 0;
            total_beats  <= 0;
            base_addr    <= 0;
            aw_grant     <= 0;
            w_last_done  <= 0;
        end else begin
            done <= 0;

            case (state)
                IDLE: begin
                    if (start && !busy) begin
                        if (burst_len == 0) begin
                            busy <= 0;
                            done <= 1;
                        end else begin
                            busy        <= 1;
                            total_beats <= burst_len + 1;
                            base_addr   <= dst_addr;
                            fill_cnt    <= 0;
                            data_ready  <= 1;
                            m_axi_awid  <= 6'd0;
                            m_axi_wid   <= 6'd0;
                            state       <= FILL;
                        end
                    end
                end

                // Fill byte buffer from data stream
                FILL: begin
                    if (data_valid && data_ready) begin
                        bytebuf[fill_cnt[5:0]] <= data_in;
                        fill_cnt <= fill_cnt + 1;
                    end
                    if (fill_cnt == fill_target) begin
                        data_ready    <= 0;
                        beat_cnt      <= 0;
                        aw_grant      <= 0;
                        w_last_done   <= 0;
                        m_axi_awaddr  <= base_addr;
                        m_axi_awlen   <= burst_len;
                        m_axi_awsize  <= 3'd2;
                        m_axi_awburst <= 2'd1;
                        m_axi_awvalid <= 1;
                        state <= XFER;
                    end
                end

                // Issue AW + send W beats (overlapped for efficiency)
                XFER: begin
                    // AW channel
                    if (!aw_grant) begin
                        if (m_axi_awready) begin
                            m_axi_awvalid <= 0;
                            aw_grant <= 1;
                        end
                    end

                    // W channel — data driven from registered beat_cnt
                    m_axi_wdata <= wdata_steer;
                    m_axi_wstrb <= wstrb_steer;
                    m_axi_wlast  <= (beat_cnt == burst_len);

                    if (!w_last_done) begin
                        m_axi_wvalid <= 1;
                        if (m_axi_wready) begin
                            if (beat_cnt == burst_len) begin
                                w_last_done <= 1;
                            end else begin
                                beat_cnt <= beat_cnt + 1;
                            end
                        end
                    end

                    // Both AW and W complete → wait for B
                    if (aw_grant && w_last_done) begin
                        m_axi_wvalid <= 0;
                        m_axi_wlast  <= 0;
                        m_axi_bready <= 1;
                        state <= B_WAIT;
                    end
                end

                B_WAIT: begin
                    if (m_axi_bvalid) begin
                        m_axi_bready <= 0;
                        state <= FINISH;
                    end
                end

                FINISH: begin
                    busy     <= 0;
                    done     <= 1;
                    beat_cnt <= 0;
                    state    <= IDLE;
                end
            endcase
        end
    end

endmodule
