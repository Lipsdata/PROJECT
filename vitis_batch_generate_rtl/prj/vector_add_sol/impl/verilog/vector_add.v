// ==============================================================
// Generated by Vitis HLS v2024.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

(* CORE_GENERATION_INFO="vector_add_vector_add,hls_ip_2024_2,{HLS_INPUT_TYPE=cxx,HLS_INPUT_FLOAT=0,HLS_INPUT_FIXED=0,HLS_INPUT_PART=xcvu19p-fsvb3824-2-e,HLS_INPUT_CLOCK=3.300000,HLS_INPUT_ARCH=others,HLS_SYN_CLOCK=0.880000,HLS_SYN_LAT=0,HLS_SYN_TPT=none,HLS_SYN_MEM=0,HLS_SYN_DSP=0,HLS_SYN_FF=0,HLS_SYN_LUT=78,HLS_VERSION=2024_2}" *)

module vector_add (
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        ap_core,
        ap_part,
        ap_parent,
        vec_d1_x,
        vec_d1_x_ap_vld,
        vec_d1_y,
        vec_d1_y_ap_vld,
        vec_s1_x,
        vec_s1_y,
        vec_s2_x,
        vec_s2_y,
        ap_ce,
        ap_rst
);


input   ap_start;
output   ap_done;
output   ap_idle;
output   ap_ready;
input  [7:0] ap_core;
input  [7:0] ap_part;
input  [7:0] ap_parent;
output  [31:0] vec_d1_x;
output   vec_d1_x_ap_vld;
output  [31:0] vec_d1_y;
output   vec_d1_y_ap_vld;
input  [31:0] vec_s1_x;
input  [31:0] vec_s1_y;
input  [31:0] vec_s2_x;
input  [31:0] vec_s2_y;
input   ap_ce;
input   ap_rst;

reg vec_d1_x_ap_vld;
reg vec_d1_y_ap_vld;

wire    ap_ce_reg;

always @ (*) begin
    if (((1'b1 == ap_ce) & (ap_start == 1'b1))) begin
        vec_d1_x_ap_vld = 1'b1;
    end else begin
        vec_d1_x_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_ce) & (ap_start == 1'b1))) begin
        vec_d1_y_ap_vld = 1'b1;
    end else begin
        vec_d1_y_ap_vld = 1'b0;
    end
end

assign ap_done = ap_start;

assign ap_idle = 1'b1;

assign ap_ready = ap_start;

assign vec_d1_x = (vec_s2_x + vec_s1_x);

assign vec_d1_y = (vec_s2_y + vec_s1_y);

endmodule //vector_add
