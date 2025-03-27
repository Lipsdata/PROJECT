#ifndef HLS_CNN_H
#define HLS_CNN_H

#include "common.h" // 确保common.h在前面
#if __VITIS_HLS__
    #include <ap_int.h>
#endif

#include <stdint.h>
#include <stddef.h>
#include "hls_config.h"
#include "common_with_hls.h"
#include <math.h>
#include <string.h>

#ifndef MAX_MAP_SIZE
#define MAX_MAP_SIZE 1024  // 最大特征图大小 (32x32)
#endif


// 声明 activation_func 结构体，实现在 hls.cpp 中
struct activation_func;

#ifdef __cplusplus
extern "C" {
#endif

// Connection table for C3 layer
extern bool connection_table[6*16];


// Function declarations

void convn_valid(float in_data[MAX_MAP_SIZE], int in_w, int in_h,
    float kernel[MAX_KERNEL_SIZE], int kernel_w, int kernel_h,
    float out_data[MAX_MAP_SIZE], int out_w, int out_h);

/*
void HLS_DECLARE(conv_fprop1)(HLS_COMMON_ARG Layer *input_layer, Layer *c1_conv_layer, bool pconnection[96]);
void HLS_DECLARE(max_pooling_fprop1)(HLS_COMMON_ARG Layer *c1_conv_layer, Layer *s2_pooling_layer);
void HLS_DECLARE(conv_fprop2)(HLS_COMMON_ARG Layer *s2_pooling_layer, Layer *c3_conv_layer, bool pconnection[96]);
void HLS_DECLARE(max_pooling_fprop2)(HLS_COMMON_ARG Layer *c3_conv_layer, Layer *s4_pooling_layer);
void HLS_DECLARE(conv_fprop3)(HLS_COMMON_ARG Layer *s4_pooling_layer, Layer *c5_conv_layer, bool pconnection[96]);
void HLS_DECLARE(fully_connected_fprop)(HLS_COMMON_ARG Layer *c5_conv_layer, Layer *output_layer); */
void HLS_DECLARE(conv_fprop1)(HLS_COMMON_ARG NetworkParams *input_params, InputLayer *input_layer, 
                             NetworkParams *c1_params, C1Layer *c1_layer, uint8_t pconnection[96]);
                             
void HLS_DECLARE(max_pooling_fprop1)(HLS_COMMON_ARG NetworkParams *c1_params, C1Layer *c1_layer,
                                    NetworkParams *s2_params, S2Layer *s2_layer);
                                    
void HLS_DECLARE(conv_fprop2)(HLS_COMMON_ARG NetworkParams *s2_params, S2Layer *s2_layer,
                             NetworkParams *c3_params, C3Layer *c3_layer, uint8_t pconnection[96]);
                             
void HLS_DECLARE(max_pooling_fprop2)(HLS_COMMON_ARG NetworkParams *c3_params, C3Layer *c3_layer,
                                    NetworkParams *s4_params, S4Layer *s4_layer);
                                    
void HLS_DECLARE(conv_fprop3)(HLS_COMMON_ARG NetworkParams *s4_params, S4Layer *s4_layer,
                             NetworkParams *c5_params, C5Layer *c5_layer, uint8_t pconnection[96]);
                             
void HLS_DECLARE(fully_connected_fprop)(HLS_COMMON_ARG NetworkParams *c5_params, C5Layer *c5_layer,
                                       NetworkParams *output_params, OutputLayer *output_layer);

//void HLS_DECLARE(max_pooling_fprop)(HLS_COMMON_ARG Layer *prev_layer, Layer *layer);

//void HLS_DECLARE(fully_connected_fprop)(HLS_COMMON_ARG Layer *prev_layer, Layer *layer);


#ifdef __cplusplus
}
#endif


#if __riscv
    #include "hls_apcall.h"
#endif



#endif // HLS_CNN_H
