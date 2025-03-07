// ==============================================================
// Generated by Vitis HLS v2024.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

(* CORE_GENERATION_INFO="assign_array_complete_assign_array_complete,hls_ip_2024_2,{HLS_INPUT_TYPE=cxx,HLS_INPUT_FLOAT=0,HLS_INPUT_FIXED=0,HLS_INPUT_PART=xcvu19p-fsvb3824-2-e,HLS_INPUT_CLOCK=3.300000,HLS_INPUT_ARCH=others,HLS_SYN_CLOCK=1.346000,HLS_SYN_LAT=6,HLS_SYN_TPT=none,HLS_SYN_MEM=0,HLS_SYN_DSP=0,HLS_SYN_FF=5,HLS_SYN_LUT=90,HLS_VERSION=2024_2}" *)

module assign_array_complete (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        ap_ce,
        ap_core,
        ap_part,
        ap_parent,
        arr_complete_0,
        arr_complete_0_ap_vld,
        arr_complete_1,
        arr_complete_1_ap_vld,
        arr_complete_2,
        arr_complete_2_ap_vld,
        arr_complete_3,
        arr_complete_3_ap_vld,
        arr_complete_4,
        arr_complete_4_ap_vld,
        base_r
);

parameter    ap_ST_fsm_state1 = 1'd1;

input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
output   ap_idle;
output   ap_ready;
input   ap_ce;
input  [7:0] ap_core;
input  [7:0] ap_part;
input  [7:0] ap_parent;
output  [31:0] arr_complete_0;
output   arr_complete_0_ap_vld;
output  [31:0] arr_complete_1;
output   arr_complete_1_ap_vld;
output  [31:0] arr_complete_2;
output   arr_complete_2_ap_vld;
output  [31:0] arr_complete_3;
output   arr_complete_3_ap_vld;
output  [31:0] arr_complete_4;
output   arr_complete_4_ap_vld;
input  [31:0] base_r;

reg ap_idle;
reg arr_complete_0_ap_vld;
reg arr_complete_1_ap_vld;
reg arr_complete_2_ap_vld;
reg arr_complete_3_ap_vld;
reg arr_complete_4_ap_vld;

(* fsm_encoding = "none" *) reg   [0:0] ap_CS_fsm;
wire    ap_CS_fsm_state1;
reg    ap_block_state1_pp0_stage0_iter0;
wire   [0:0] icmp_ln18_fu_134_p2;
reg    ap_condition_exit_pp0_iter0_stage0;
wire    ap_loop_exit_ready;
reg    ap_ready_int;
reg   [2:0] i1_fu_60;
wire   [2:0] i_fu_128_p2;
wire    ap_loop_init;
reg   [2:0] ap_sig_allocacmp_i1_load;
wire   [2:0] i1_load_load_fu_110_p1;
wire   [31:0] add_ln20_fu_117_p2;
wire   [31:0] zext_ln20_fu_113_p1;
reg    ap_done_reg;
wire    ap_continue_int;
reg    ap_done_int;
reg   [0:0] ap_NS_fsm;
reg    ap_ST_fsm_state1_blk;
wire    ap_start_int;
wire    ap_ready_sig;
wire    ap_done_sig;
reg    ap_condition_59;
wire    ap_ce_reg;

// power-on initialization
initial begin
#0 ap_CS_fsm = 1'd1;
#0 i1_fu_60 = 3'd0;
#0 ap_done_reg = 1'b0;
end

assign_array_complete_flow_control_loop_pipe flow_control_loop_pipe_U(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(ap_start),
    .ap_ready(ap_ready_sig),
    .ap_done(ap_done_sig),
    .ap_start_int(ap_start_int),
    .ap_loop_init(ap_loop_init),
    .ap_ready_int(ap_ready_int),
    .ap_loop_exit_ready(ap_condition_exit_pp0_iter0_stage0),
    .ap_loop_exit_done(ap_done_int),
    .ap_continue_int(ap_continue_int),
    .ap_done_int(ap_done_int),
    .ap_continue(1'b1)
);

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_fsm <= ap_ST_fsm_state1;
    end else begin
        if ((1'b1 == ap_ce)) begin
            ap_CS_fsm <= ap_NS_fsm;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_done_reg <= 1'b0;
    end else begin
        if ((ap_continue_int == 1'b1)) begin
            ap_done_reg <= 1'b0;
        end else if ((~((1'b0 == ap_ce) | (1'b1 == ap_block_state1_pp0_stage0_iter0)) & (ap_loop_exit_ready == 1'b1) & (1'b1 == ap_CS_fsm_state1))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_59)) begin
        i1_fu_60 <= i_fu_128_p2;
    end
end

always @ (*) begin
    if (((1'b0 == ap_ce) | (1'b1 == ap_block_state1_pp0_stage0_iter0))) begin
        ap_ST_fsm_state1_blk = 1'b1;
    end else begin
        ap_ST_fsm_state1_blk = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b0 == ap_ce) | (1'b1 == ap_block_state1_pp0_stage0_iter0)) & (icmp_ln18_fu_134_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b1;
    end else begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b0 == ap_ce) | (1'b1 == ap_block_state1_pp0_stage0_iter0)) & (ap_loop_exit_ready == 1'b1) & (1'b1 == ap_CS_fsm_state1))) begin
        ap_done_int = 1'b1;
    end else begin
        ap_done_int = ap_done_reg;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state1) & (ap_start_int == 1'b0))) begin
        ap_idle = 1'b1;
    end else begin
        ap_idle = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b0 == ap_ce) | (1'b1 == ap_block_state1_pp0_stage0_iter0)) & (1'b1 == ap_CS_fsm_state1))) begin
        ap_ready_int = 1'b1;
    end else begin
        ap_ready_int = 1'b0;
    end
end

always @ (*) begin
    if (((ap_loop_init == 1'b1) & (1'b1 == ap_CS_fsm_state1))) begin
        ap_sig_allocacmp_i1_load = 3'd0;
    end else begin
        ap_sig_allocacmp_i1_load = i1_fu_60;
    end
end

always @ (*) begin
    if (((i1_load_load_fu_110_p1 == 3'd0) & (1'b0 == ap_block_state1_pp0_stage0_iter0) & (1'b1 == ap_ce) & (1'b1 == ap_CS_fsm_state1))) begin
        arr_complete_0_ap_vld = 1'b1;
    end else begin
        arr_complete_0_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((i1_load_load_fu_110_p1 == 3'd1) & (1'b0 == ap_block_state1_pp0_stage0_iter0) & (1'b1 == ap_ce) & (1'b1 == ap_CS_fsm_state1))) begin
        arr_complete_1_ap_vld = 1'b1;
    end else begin
        arr_complete_1_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((i1_load_load_fu_110_p1 == 3'd2) & (1'b0 == ap_block_state1_pp0_stage0_iter0) & (1'b1 == ap_ce) & (1'b1 == ap_CS_fsm_state1))) begin
        arr_complete_2_ap_vld = 1'b1;
    end else begin
        arr_complete_2_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((i1_load_load_fu_110_p1 == 3'd3) & (1'b0 == ap_block_state1_pp0_stage0_iter0) & (1'b1 == ap_ce) & (1'b1 == ap_CS_fsm_state1))) begin
        arr_complete_3_ap_vld = 1'b1;
    end else begin
        arr_complete_3_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if ((~(i1_load_load_fu_110_p1 == 3'd0) & ~(i1_load_load_fu_110_p1 == 3'd1) & ~(i1_load_load_fu_110_p1 == 3'd2) & ~(i1_load_load_fu_110_p1 == 3'd3) & (1'b0 == ap_block_state1_pp0_stage0_iter0) & (1'b1 == ap_ce) & (1'b1 == ap_CS_fsm_state1))) begin
        arr_complete_4_ap_vld = 1'b1;
    end else begin
        arr_complete_4_ap_vld = 1'b0;
    end
end

always @ (*) begin
    case (ap_CS_fsm)
        ap_ST_fsm_state1 : begin
            ap_NS_fsm = ap_ST_fsm_state1;
        end
        default : begin
            ap_NS_fsm = 'bx;
        end
    endcase
end

assign add_ln20_fu_117_p2 = (zext_ln20_fu_113_p1 + base_r);

assign ap_CS_fsm_state1 = ap_CS_fsm[32'd0];

always @ (*) begin
    ap_block_state1_pp0_stage0_iter0 = (ap_start_int == 1'b0);
end

always @ (*) begin
    ap_condition_59 = ((1'b0 == ap_block_state1_pp0_stage0_iter0) & (1'b1 == ap_ce) & (1'b1 == ap_CS_fsm_state1));
end

assign ap_done = ap_done_sig;

assign ap_loop_exit_ready = ap_condition_exit_pp0_iter0_stage0;

assign ap_ready = ap_ready_sig;

assign arr_complete_0 = add_ln20_fu_117_p2;

assign arr_complete_1 = add_ln20_fu_117_p2;

assign arr_complete_2 = add_ln20_fu_117_p2;

assign arr_complete_3 = add_ln20_fu_117_p2;

assign arr_complete_4 = add_ln20_fu_117_p2;

assign i1_load_load_fu_110_p1 = ap_sig_allocacmp_i1_load;

assign i_fu_128_p2 = (ap_sig_allocacmp_i1_load + 3'd1);

assign icmp_ln18_fu_134_p2 = ((ap_sig_allocacmp_i1_load == 3'd4) ? 1'b1 : 1'b0);

assign zext_ln20_fu_113_p1 = ap_sig_allocacmp_i1_load;

endmodule //assign_array_complete
