#include "hls_design_meta.h"
const Port_Property HLS_Design_Meta::port_props[]={
	Port_Property("ap_clk", 1, hls_in, -1, "", "", 1),
	Port_Property("ap_rst", 1, hls_in, -1, "", "", 1),
	Port_Property("ap_start", 1, hls_in, -1, "", "", 1),
	Port_Property("ap_done", 1, hls_out, -1, "", "", 1),
	Port_Property("ap_idle", 1, hls_out, -1, "", "", 1),
	Port_Property("ap_ready", 1, hls_out, -1, "", "", 1),
	Port_Property("ap_ce", 1, hls_in, -1, "", "", 1),
	Port_Property("ap_core", 8, hls_in, 0, "ap_none", "in_data", 1),
	Port_Property("ap_part", 8, hls_in, 1, "ap_none", "in_data", 1),
	Port_Property("ap_parent", 8, hls_in, 2, "ap_none", "in_data", 1),
	Port_Property("arr_complete_0", 32, hls_out, 3, "ap_vld", "out_data", 1),
	Port_Property("arr_complete_0_ap_vld", 1, hls_out, 3, "ap_vld", "out_vld", 1),
	Port_Property("arr_complete_1", 32, hls_out, 4, "ap_vld", "out_data", 1),
	Port_Property("arr_complete_1_ap_vld", 1, hls_out, 4, "ap_vld", "out_vld", 1),
	Port_Property("arr_complete_2", 32, hls_out, 5, "ap_vld", "out_data", 1),
	Port_Property("arr_complete_2_ap_vld", 1, hls_out, 5, "ap_vld", "out_vld", 1),
	Port_Property("arr_complete_3", 32, hls_out, 6, "ap_vld", "out_data", 1),
	Port_Property("arr_complete_3_ap_vld", 1, hls_out, 6, "ap_vld", "out_vld", 1),
	Port_Property("arr_complete_4", 32, hls_out, 7, "ap_vld", "out_data", 1),
	Port_Property("arr_complete_4_ap_vld", 1, hls_out, 7, "ap_vld", "out_vld", 1),
	Port_Property("base_r", 32, hls_in, 8, "ap_none", "in_data", 1),
};
const char* HLS_Design_Meta::dut_name = "assign_array_complete";
