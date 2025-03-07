-- ==============================================================
-- Generated by Vitis HLS v2024.2
-- Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
-- Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
-- ==============================================================

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity assign_array_complete is
port (
    ap_clk : IN STD_LOGIC;
    ap_rst : IN STD_LOGIC;
    ap_start : IN STD_LOGIC;
    ap_done : OUT STD_LOGIC;
    ap_idle : OUT STD_LOGIC;
    ap_ready : OUT STD_LOGIC;
    ap_ce : IN STD_LOGIC;
    ap_core : IN STD_LOGIC_VECTOR (7 downto 0);
    ap_part : IN STD_LOGIC_VECTOR (7 downto 0);
    ap_parent : IN STD_LOGIC_VECTOR (7 downto 0);
    arr_complete_0 : OUT STD_LOGIC_VECTOR (31 downto 0);
    arr_complete_0_ap_vld : OUT STD_LOGIC;
    arr_complete_1 : OUT STD_LOGIC_VECTOR (31 downto 0);
    arr_complete_1_ap_vld : OUT STD_LOGIC;
    arr_complete_2 : OUT STD_LOGIC_VECTOR (31 downto 0);
    arr_complete_2_ap_vld : OUT STD_LOGIC;
    arr_complete_3 : OUT STD_LOGIC_VECTOR (31 downto 0);
    arr_complete_3_ap_vld : OUT STD_LOGIC;
    arr_complete_4 : OUT STD_LOGIC_VECTOR (31 downto 0);
    arr_complete_4_ap_vld : OUT STD_LOGIC;
    base_r : IN STD_LOGIC_VECTOR (31 downto 0) );
end;


architecture behav of assign_array_complete is 
    attribute CORE_GENERATION_INFO : STRING;
    attribute CORE_GENERATION_INFO of behav : architecture is
    "assign_array_complete_assign_array_complete,hls_ip_2024_2,{HLS_INPUT_TYPE=cxx,HLS_INPUT_FLOAT=0,HLS_INPUT_FIXED=0,HLS_INPUT_PART=xcvu19p-fsvb3824-2-e,HLS_INPUT_CLOCK=3.300000,HLS_INPUT_ARCH=others,HLS_SYN_CLOCK=1.346000,HLS_SYN_LAT=6,HLS_SYN_TPT=none,HLS_SYN_MEM=0,HLS_SYN_DSP=0,HLS_SYN_FF=5,HLS_SYN_LUT=90,HLS_VERSION=2024_2}";
    constant ap_const_logic_1 : STD_LOGIC := '1';
    constant ap_const_logic_0 : STD_LOGIC := '0';
    constant ap_ST_fsm_state1 : STD_LOGIC_VECTOR (0 downto 0) := "1";
    constant ap_const_lv32_0 : STD_LOGIC_VECTOR (31 downto 0) := "00000000000000000000000000000000";
    constant ap_const_boolean_1 : BOOLEAN := true;
    constant ap_const_lv1_1 : STD_LOGIC_VECTOR (0 downto 0) := "1";
    constant ap_const_lv3_0 : STD_LOGIC_VECTOR (2 downto 0) := "000";
    constant ap_const_boolean_0 : BOOLEAN := false;
    constant ap_const_lv3_3 : STD_LOGIC_VECTOR (2 downto 0) := "011";
    constant ap_const_lv3_2 : STD_LOGIC_VECTOR (2 downto 0) := "010";
    constant ap_const_lv3_1 : STD_LOGIC_VECTOR (2 downto 0) := "001";
    constant ap_const_lv3_4 : STD_LOGIC_VECTOR (2 downto 0) := "100";

    signal ap_CS_fsm : STD_LOGIC_VECTOR (0 downto 0) := "1";
    attribute fsm_encoding : string;
    attribute fsm_encoding of ap_CS_fsm : signal is "none";
    signal ap_CS_fsm_state1 : STD_LOGIC;
    attribute fsm_encoding of ap_CS_fsm_state1 : signal is "none";
    signal ap_block_state1_pp0_stage0_iter0 : BOOLEAN;
    signal icmp_ln18_fu_134_p2 : STD_LOGIC_VECTOR (0 downto 0);
    signal ap_condition_exit_pp0_iter0_stage0 : STD_LOGIC;
    signal ap_loop_exit_ready : STD_LOGIC;
    signal ap_ready_int : STD_LOGIC;
    signal i1_fu_60 : STD_LOGIC_VECTOR (2 downto 0) := "000";
    signal i_fu_128_p2 : STD_LOGIC_VECTOR (2 downto 0);
    signal ap_loop_init : STD_LOGIC;
    signal ap_sig_allocacmp_i1_load : STD_LOGIC_VECTOR (2 downto 0);
    signal i1_load_load_fu_110_p1 : STD_LOGIC_VECTOR (2 downto 0);
    signal add_ln20_fu_117_p2 : STD_LOGIC_VECTOR (31 downto 0);
    signal zext_ln20_fu_113_p1 : STD_LOGIC_VECTOR (31 downto 0);
    signal ap_done_reg : STD_LOGIC := '0';
    signal ap_continue_int : STD_LOGIC;
    signal ap_done_int : STD_LOGIC;
    signal ap_NS_fsm : STD_LOGIC_VECTOR (0 downto 0);
    signal ap_ST_fsm_state1_blk : STD_LOGIC;
    signal ap_start_int : STD_LOGIC;
    signal ap_ready_sig : STD_LOGIC;
    signal ap_done_sig : STD_LOGIC;
    signal ap_condition_59 : BOOLEAN;
    signal ap_ce_reg : STD_LOGIC;

    component assign_array_complete_flow_control_loop_pipe IS
    port (
        ap_clk : IN STD_LOGIC;
        ap_rst : IN STD_LOGIC;
        ap_start : IN STD_LOGIC;
        ap_ready : OUT STD_LOGIC;
        ap_done : OUT STD_LOGIC;
        ap_start_int : OUT STD_LOGIC;
        ap_loop_init : OUT STD_LOGIC;
        ap_ready_int : IN STD_LOGIC;
        ap_loop_exit_ready : IN STD_LOGIC;
        ap_loop_exit_done : IN STD_LOGIC;
        ap_continue_int : OUT STD_LOGIC;
        ap_done_int : IN STD_LOGIC;
        ap_continue : IN STD_LOGIC );
    end component;



begin
    flow_control_loop_pipe_U : component assign_array_complete_flow_control_loop_pipe
    port map (
        ap_clk => ap_clk,
        ap_rst => ap_rst,
        ap_start => ap_start,
        ap_ready => ap_ready_sig,
        ap_done => ap_done_sig,
        ap_start_int => ap_start_int,
        ap_loop_init => ap_loop_init,
        ap_ready_int => ap_ready_int,
        ap_loop_exit_ready => ap_condition_exit_pp0_iter0_stage0,
        ap_loop_exit_done => ap_done_int,
        ap_continue_int => ap_continue_int,
        ap_done_int => ap_done_int,
        ap_continue => ap_const_logic_1);





    ap_CS_fsm_assign_proc : process(ap_clk)
    begin
        if (ap_clk'event and ap_clk =  '1') then
            if (ap_rst = '1') then
                ap_CS_fsm <= ap_ST_fsm_state1;
            else
                if ((ap_const_logic_1 = ap_ce)) then 
                    ap_CS_fsm <= ap_NS_fsm;
                end if; 
            end if;
        end if;
    end process;


    ap_done_reg_assign_proc : process(ap_clk)
    begin
        if (ap_clk'event and ap_clk =  '1') then
            if (ap_rst = '1') then
                ap_done_reg <= ap_const_logic_0;
            else
                if ((ap_continue_int = ap_const_logic_1)) then 
                    ap_done_reg <= ap_const_logic_0;
                elsif ((not(((ap_const_logic_0 = ap_ce) or (ap_const_boolean_1 = ap_block_state1_pp0_stage0_iter0))) and (ap_loop_exit_ready = ap_const_logic_1) and (ap_const_logic_1 = ap_CS_fsm_state1))) then 
                    ap_done_reg <= ap_const_logic_1;
                end if; 
            end if;
        end if;
    end process;


    i1_fu_60_assign_proc : process (ap_clk)
    begin
        if (ap_clk'event and ap_clk = '1') then
            if ((ap_const_boolean_1 = ap_condition_59)) then
                i1_fu_60 <= i_fu_128_p2;
            end if;
        end if;
    end process;

    ap_NS_fsm_assign_proc : process (ap_CS_fsm, ap_CS_fsm_state1, ap_block_state1_pp0_stage0_iter0, ap_ce)
    begin
        case ap_CS_fsm is
            when ap_ST_fsm_state1 => 
                ap_NS_fsm <= ap_ST_fsm_state1;
            when others =>  
                ap_NS_fsm <= "X";
        end case;
    end process;
    add_ln20_fu_117_p2 <= std_logic_vector(unsigned(zext_ln20_fu_113_p1) + unsigned(base_r));
    ap_CS_fsm_state1 <= ap_CS_fsm(0);

    ap_ST_fsm_state1_blk_assign_proc : process(ap_block_state1_pp0_stage0_iter0, ap_ce)
    begin
        if (((ap_const_logic_0 = ap_ce) or (ap_const_boolean_1 = ap_block_state1_pp0_stage0_iter0))) then 
            ap_ST_fsm_state1_blk <= ap_const_logic_1;
        else 
            ap_ST_fsm_state1_blk <= ap_const_logic_0;
        end if; 
    end process;


    ap_block_state1_pp0_stage0_iter0_assign_proc : process(ap_start_int)
    begin
                ap_block_state1_pp0_stage0_iter0 <= (ap_start_int = ap_const_logic_0);
    end process;


    ap_condition_59_assign_proc : process(ap_CS_fsm_state1, ap_block_state1_pp0_stage0_iter0, ap_ce)
    begin
                ap_condition_59 <= ((ap_const_boolean_0 = ap_block_state1_pp0_stage0_iter0) and (ap_const_logic_1 = ap_ce) and (ap_const_logic_1 = ap_CS_fsm_state1));
    end process;


    ap_condition_exit_pp0_iter0_stage0_assign_proc : process(ap_CS_fsm_state1, ap_block_state1_pp0_stage0_iter0, ap_ce, icmp_ln18_fu_134_p2)
    begin
        if ((not(((ap_const_logic_0 = ap_ce) or (ap_const_boolean_1 = ap_block_state1_pp0_stage0_iter0))) and (icmp_ln18_fu_134_p2 = ap_const_lv1_1) and (ap_const_logic_1 = ap_CS_fsm_state1))) then 
            ap_condition_exit_pp0_iter0_stage0 <= ap_const_logic_1;
        else 
            ap_condition_exit_pp0_iter0_stage0 <= ap_const_logic_0;
        end if; 
    end process;

    ap_done <= ap_done_sig;

    ap_done_int_assign_proc : process(ap_CS_fsm_state1, ap_block_state1_pp0_stage0_iter0, ap_ce, ap_loop_exit_ready, ap_done_reg)
    begin
        if ((not(((ap_const_logic_0 = ap_ce) or (ap_const_boolean_1 = ap_block_state1_pp0_stage0_iter0))) and (ap_loop_exit_ready = ap_const_logic_1) and (ap_const_logic_1 = ap_CS_fsm_state1))) then 
            ap_done_int <= ap_const_logic_1;
        else 
            ap_done_int <= ap_done_reg;
        end if; 
    end process;


    ap_idle_assign_proc : process(ap_CS_fsm_state1, ap_start_int)
    begin
        if (((ap_const_logic_1 = ap_CS_fsm_state1) and (ap_start_int = ap_const_logic_0))) then 
            ap_idle <= ap_const_logic_1;
        else 
            ap_idle <= ap_const_logic_0;
        end if; 
    end process;

    ap_loop_exit_ready <= ap_condition_exit_pp0_iter0_stage0;
    ap_ready <= ap_ready_sig;

    ap_ready_int_assign_proc : process(ap_CS_fsm_state1, ap_block_state1_pp0_stage0_iter0, ap_ce)
    begin
        if ((not(((ap_const_logic_0 = ap_ce) or (ap_const_boolean_1 = ap_block_state1_pp0_stage0_iter0))) and (ap_const_logic_1 = ap_CS_fsm_state1))) then 
            ap_ready_int <= ap_const_logic_1;
        else 
            ap_ready_int <= ap_const_logic_0;
        end if; 
    end process;


    ap_sig_allocacmp_i1_load_assign_proc : process(ap_CS_fsm_state1, i1_fu_60, ap_loop_init)
    begin
        if (((ap_loop_init = ap_const_logic_1) and (ap_const_logic_1 = ap_CS_fsm_state1))) then 
            ap_sig_allocacmp_i1_load <= ap_const_lv3_0;
        else 
            ap_sig_allocacmp_i1_load <= i1_fu_60;
        end if; 
    end process;

    arr_complete_0 <= add_ln20_fu_117_p2;

    arr_complete_0_ap_vld_assign_proc : process(ap_CS_fsm_state1, ap_block_state1_pp0_stage0_iter0, ap_ce, i1_load_load_fu_110_p1)
    begin
        if (((i1_load_load_fu_110_p1 = ap_const_lv3_0) and (ap_const_boolean_0 = ap_block_state1_pp0_stage0_iter0) and (ap_const_logic_1 = ap_ce) and (ap_const_logic_1 = ap_CS_fsm_state1))) then 
            arr_complete_0_ap_vld <= ap_const_logic_1;
        else 
            arr_complete_0_ap_vld <= ap_const_logic_0;
        end if; 
    end process;

    arr_complete_1 <= add_ln20_fu_117_p2;

    arr_complete_1_ap_vld_assign_proc : process(ap_CS_fsm_state1, ap_block_state1_pp0_stage0_iter0, ap_ce, i1_load_load_fu_110_p1)
    begin
        if (((i1_load_load_fu_110_p1 = ap_const_lv3_1) and (ap_const_boolean_0 = ap_block_state1_pp0_stage0_iter0) and (ap_const_logic_1 = ap_ce) and (ap_const_logic_1 = ap_CS_fsm_state1))) then 
            arr_complete_1_ap_vld <= ap_const_logic_1;
        else 
            arr_complete_1_ap_vld <= ap_const_logic_0;
        end if; 
    end process;

    arr_complete_2 <= add_ln20_fu_117_p2;

    arr_complete_2_ap_vld_assign_proc : process(ap_CS_fsm_state1, ap_block_state1_pp0_stage0_iter0, ap_ce, i1_load_load_fu_110_p1)
    begin
        if (((i1_load_load_fu_110_p1 = ap_const_lv3_2) and (ap_const_boolean_0 = ap_block_state1_pp0_stage0_iter0) and (ap_const_logic_1 = ap_ce) and (ap_const_logic_1 = ap_CS_fsm_state1))) then 
            arr_complete_2_ap_vld <= ap_const_logic_1;
        else 
            arr_complete_2_ap_vld <= ap_const_logic_0;
        end if; 
    end process;

    arr_complete_3 <= add_ln20_fu_117_p2;

    arr_complete_3_ap_vld_assign_proc : process(ap_CS_fsm_state1, ap_block_state1_pp0_stage0_iter0, ap_ce, i1_load_load_fu_110_p1)
    begin
        if (((i1_load_load_fu_110_p1 = ap_const_lv3_3) and (ap_const_boolean_0 = ap_block_state1_pp0_stage0_iter0) and (ap_const_logic_1 = ap_ce) and (ap_const_logic_1 = ap_CS_fsm_state1))) then 
            arr_complete_3_ap_vld <= ap_const_logic_1;
        else 
            arr_complete_3_ap_vld <= ap_const_logic_0;
        end if; 
    end process;

    arr_complete_4 <= add_ln20_fu_117_p2;

    arr_complete_4_ap_vld_assign_proc : process(ap_CS_fsm_state1, ap_block_state1_pp0_stage0_iter0, ap_ce, i1_load_load_fu_110_p1)
    begin
        if ((not((i1_load_load_fu_110_p1 = ap_const_lv3_0)) and not((i1_load_load_fu_110_p1 = ap_const_lv3_1)) and not((i1_load_load_fu_110_p1 = ap_const_lv3_2)) and not((i1_load_load_fu_110_p1 = ap_const_lv3_3)) and (ap_const_boolean_0 = ap_block_state1_pp0_stage0_iter0) and (ap_const_logic_1 = ap_ce) and (ap_const_logic_1 = ap_CS_fsm_state1))) then 
            arr_complete_4_ap_vld <= ap_const_logic_1;
        else 
            arr_complete_4_ap_vld <= ap_const_logic_0;
        end if; 
    end process;

    i1_load_load_fu_110_p1 <= ap_sig_allocacmp_i1_load;
    i_fu_128_p2 <= std_logic_vector(unsigned(ap_sig_allocacmp_i1_load) + unsigned(ap_const_lv3_1));
    icmp_ln18_fu_134_p2 <= "1" when (ap_sig_allocacmp_i1_load = ap_const_lv3_4) else "0";
    zext_ln20_fu_113_p1 <= std_logic_vector(IEEE.numeric_std.resize(unsigned(ap_sig_allocacmp_i1_load),32));
end behav;
