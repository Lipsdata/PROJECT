-- ==============================================================
-- Generated by Vitis HLS v2024.2
-- Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
-- Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
-- ==============================================================

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity vector_add is
port (
    ap_start : IN STD_LOGIC;
    ap_done : OUT STD_LOGIC;
    ap_idle : OUT STD_LOGIC;
    ap_ready : OUT STD_LOGIC;
    ap_core : IN STD_LOGIC_VECTOR (7 downto 0);
    ap_part : IN STD_LOGIC_VECTOR (7 downto 0);
    ap_parent : IN STD_LOGIC_VECTOR (7 downto 0);
    vec_d1_x : OUT STD_LOGIC_VECTOR (31 downto 0);
    vec_d1_x_ap_vld : OUT STD_LOGIC;
    vec_d1_y : OUT STD_LOGIC_VECTOR (31 downto 0);
    vec_d1_y_ap_vld : OUT STD_LOGIC;
    vec_s1_x : IN STD_LOGIC_VECTOR (31 downto 0);
    vec_s1_y : IN STD_LOGIC_VECTOR (31 downto 0);
    vec_s2_x : IN STD_LOGIC_VECTOR (31 downto 0);
    vec_s2_y : IN STD_LOGIC_VECTOR (31 downto 0);
    ap_ce : IN STD_LOGIC;
    ap_rst : IN STD_LOGIC );
end;


architecture behav of vector_add is 
    attribute CORE_GENERATION_INFO : STRING;
    attribute CORE_GENERATION_INFO of behav : architecture is
    "vector_add_vector_add,hls_ip_2024_2,{HLS_INPUT_TYPE=cxx,HLS_INPUT_FLOAT=0,HLS_INPUT_FIXED=0,HLS_INPUT_PART=xcvu19p-fsvb3824-2-e,HLS_INPUT_CLOCK=3.300000,HLS_INPUT_ARCH=others,HLS_SYN_CLOCK=0.880000,HLS_SYN_LAT=0,HLS_SYN_TPT=none,HLS_SYN_MEM=0,HLS_SYN_DSP=0,HLS_SYN_FF=0,HLS_SYN_LUT=78,HLS_VERSION=2024_2}";
    constant ap_const_logic_1 : STD_LOGIC := '1';
    constant ap_const_logic_0 : STD_LOGIC := '0';
    constant ap_const_boolean_1 : BOOLEAN := true;

    signal ap_ce_reg : STD_LOGIC;


begin



    ap_done <= ap_start;
    ap_idle <= ap_const_logic_1;
    ap_ready <= ap_start;
    vec_d1_x <= std_logic_vector(unsigned(vec_s2_x) + unsigned(vec_s1_x));

    vec_d1_x_ap_vld_assign_proc : process(ap_start, ap_ce)
    begin
        if (((ap_const_logic_1 = ap_ce) and (ap_start = ap_const_logic_1))) then 
            vec_d1_x_ap_vld <= ap_const_logic_1;
        else 
            vec_d1_x_ap_vld <= ap_const_logic_0;
        end if; 
    end process;

    vec_d1_y <= std_logic_vector(unsigned(vec_s2_y) + unsigned(vec_s1_y));

    vec_d1_y_ap_vld_assign_proc : process(ap_start, ap_ce)
    begin
        if (((ap_const_logic_1 = ap_ce) and (ap_start = ap_const_logic_1))) then 
            vec_d1_y_ap_vld <= ap_const_logic_1;
        else 
            vec_d1_y_ap_vld <= ap_const_logic_0;
        end if; 
    end process;

end behav;
