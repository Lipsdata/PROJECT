#ifndef _XMEM_H_
#define _XMEM_H_

#include <stdint.h>
#include "common.h"
#include "common_with_hls.h"
#if __riscv
    #include "riscv/hls_long_tail.h"
#endif

#include <stddef.h>
#include "hls_config.h"
#include <math.h>
#include <string.h>

// 定义xmem_t结构体
typedef struct xmem_t {
    // Common arguments section
    //void* common_arg                                  
    
    // convn_valid arguments
  /*  double in_data[MAX_MAP_SIZE]                      _ALIGN; 
    int in_w                                          _ALIGN; 
    int in_h                                          _ALIGN; 
    double kernel[MAX_KERNEL_SIZE]                    _ALIGN; 
    int kernel_w                                      _ALIGN; 
    int kernel_h                                      _ALIGN; 
    double out_data[MAX_MAP_SIZE]                     _ALIGN; 
    int out_w                                         _ALIGN; 
    int out_h                                         _ALIGN; */
    
    // 网络参数结构体
    NetworkParams input_params                          _ALIGN; // 输入层参数
    NetworkParams c1_params                             _ALIGN; // C1卷积层参数
    NetworkParams s2_params                             _ALIGN; // S2池化层参数
    NetworkParams c3_params                             _ALIGN; // C3卷积层参数
    NetworkParams s4_params                             _ALIGN; // S4池化层参数
    NetworkParams c5_params                             _ALIGN; // C5全连接层参数
    NetworkParams output_params                         _ALIGN; // 输出层参数

    uint8_t xxxxx_paddingA[1880];

    // 各层数据结构体
    InputLayer input_layer                              _ALIGN; // 输入层数据
    C1Layer c1_layer                                    _ALIGN; // C1卷积层数据
    S2Layer s2_layer                                    _ALIGN; // S2池化层数据
    C3Layer c3_layer                                    _ALIGN; // C3卷积层数据
    S4Layer s4_layer                                    _ALIGN; // S4池化层数据
    C5Layer c5_layer                                    _ALIGN; // C5全连接层数据
    OutputLayer output_layer                            _ALIGN; // 输出层数据
    uint8_t pconnection[96]                             _ALIGN; // 连接表

// Total scalar size: 168
// Total array size: 19209440
// Total cyclic size: 0

} xmem_t;

#ifdef __cplusplus
extern "C" {
#endif


void xmem_rw_test(void);
/*封装函数
// Function declarations using the xmem_1_t structure
void xmem_1_convn_valid(xmem_1_t* mem);
void xmem_1_conv_fprop(xmem_1_t* mem);
void xmem_1_max_pooling_fprop(xmem_1_t* mem);
void xmem_1_fully_connected_fprop(xmem_1_t* mem);
*/
#ifdef __cplusplus
}
#endif

#ifndef ARRAY_SIZE
    #define ARRAY_SIZE(arr) (sizeof(arr)/sizeof((arr)[0]))
#endif

// Use volatile in xmem_1 here to prevent mismatch load / store instruction being used for memory copy
#ifdef __cplusplus

template <typename T>
static void inline xmem_assign(T *dst, T value)
{
	volatile T *dst_ptr = static_cast<volatile T *>(dst);
	*dst_ptr = value;
}

template <typename T>
static void inline xmem_copy(T *dst, const T *src, size_t num)
{
	for (size_t i = 0; i < num; i++)
	{
		volatile T *dst_ptr = static_cast<volatile T *>(dst);
		dst_ptr[i] = src[i];
	}
}

template <typename T>
static void inline xmem_set(T *dst, T value, size_t num)
{
	for (size_t i = 0; i < num; i++)
	{
		volatile T *dst_ptr = static_cast<volatile T *>(dst);
		dst_ptr[i] = value;
	}
}

#endif


#define XMEM_ASSIGN(TYPE, DST, VALUE)                    \
	do                                                     \
	{                                                      \
		volatile TYPE *dst_ptr = (volatile TYPE *)(DST);   \
		*dst_ptr = VALUE;                                  \
	} while (0);


#define XMEM_COPY(TYPE, DST, SRC, COUNT)                 \
	do                                                     \
	{                                                      \
		volatile TYPE *dst_ptr = (volatile TYPE *)(DST);   \
		volatile TYPE *src_ptr = (volatile TYPE *)(SRC);   \
		for (size_t i = 0; i < COUNT; i++)                 \
		{                                                  \
			dst_ptr[i] = src_ptr[i];                       \
		}                                                  \
	} while (0);


#define XMEM_SET(TYPE, DST, VAL, COUNT)                  \
	do                                                     \
	{                                                      \
		volatile TYPE *dst_ptr = (volatile TYPE *)(DST);   \
		for (size_t i = 0; i < COUNT; i++)                 \
		{                                                  \
			dst_ptr[i] = VAL;                              \
		}                                                  \
	} while (0);
/*
// Macros to access and use the xmem_1_t structure for HLS functions
#define XMEM_1_CONVN_VALID(MEM)                                             \
    xmem_1_convn_valid(MEM)

#define XMEM_1_CONV_FPROP(MEM)                                              \
    xmem_1_conv_fprop(MEM)

#define XMEM_1_MAX_POOLING_FPROP(MEM)                                       \
    xmem_1_max_pooling_fprop(MEM)

#define XMEM_1_FULLY_CONNECTED_FPROP(MEM)                                   \
    xmem_1_fully_connected_fprop(MEM)
    */

#endif // _XMEM_1_H_#ifndef _XMEM_H_
#define _XMEM_H_

#include <stdint.h>
#include "common.h"
#include "common_with_hls.h"
#if __riscv
    #include "riscv/hls_long_tail.h"
#endif

#include <stddef.h>
#include "hls_config.h"
#include <math.h>
#include <string.h>

// 定义xmem_t结构体
typedef struct xmem_t {
    // Common arguments section
    //void* common_arg                                  
    
    // convn_valid arguments
  /*  double in_data[MAX_MAP_SIZE]                      _ALIGN; 
    int in_w                                          _ALIGN; 
    int in_h                                          _ALIGN; 
    double kernel[MAX_KERNEL_SIZE]                    _ALIGN; 
    int kernel_w                                      _ALIGN; 
    int kernel_h                                      _ALIGN; 
    double out_data[MAX_MAP_SIZE]                     _ALIGN; 
    int out_w                                         _ALIGN; 
    int out_h                                         _ALIGN; */
    
    // 网络参数结构体
    NetworkParams input_params                          _ALIGN; // 输入层参数
    NetworkParams c1_params                             _ALIGN; // C1卷积层参数
    NetworkParams s2_params                             _ALIGN; // S2池化层参数
    NetworkParams c3_params                             _ALIGN; // C3卷积层参数
    NetworkParams s4_params                             _ALIGN; // S4池化层参数
    NetworkParams c5_params                             _ALIGN; // C5全连接层参数
    NetworkParams output_params                         _ALIGN; // 输出层参数

    uint8_t xxxxx_paddingA[1880];

    // 各层数据结构体
    InputLayer input_layer                              _ALIGN; // 输入层数据
    C1Layer c1_layer                                    _ALIGN; // C1卷积层数据
    S2Layer s2_layer                                    _ALIGN; // S2池化层数据
    C3Layer c3_layer                                    _ALIGN; // C3卷积层数据
    S4Layer s4_layer                                    _ALIGN; // S4池化层数据
    C5Layer c5_layer                                    _ALIGN; // C5全连接层数据
    OutputLayer output_layer                            _ALIGN; // 输出层数据
    uint8_t pconnection[96]                             _ALIGN; // 连接表

// Total scalar size: 168
// Total array size: 19209440
// Total cyclic size: 0

} xmem_t;

#ifdef __cplusplus
extern "C" {
#endif


void xmem_rw_test(void);
/*封装函数
// Function declarations using the xmem_1_t structure
void xmem_1_convn_valid(xmem_1_t* mem);
void xmem_1_conv_fprop(xmem_1_t* mem);
void xmem_1_max_pooling_fprop(xmem_1_t* mem);
void xmem_1_fully_connected_fprop(xmem_1_t* mem);
*/
#ifdef __cplusplus
}
#endif

#ifndef ARRAY_SIZE
    #define ARRAY_SIZE(arr) (sizeof(arr)/sizeof((arr)[0]))
#endif

// Use volatile in xmem_1 here to prevent mismatch load / store instruction being used for memory copy
#ifdef __cplusplus

template <typename T>
static void inline xmem_assign(T *dst, T value)
{
	volatile T *dst_ptr = static_cast<volatile T *>(dst);
	*dst_ptr = value;
}

template <typename T>
static void inline xmem_copy(T *dst, const T *src, size_t num)
{
	for (size_t i = 0; i < num; i++)
	{
		volatile T *dst_ptr = static_cast<volatile T *>(dst);
		dst_ptr[i] = src[i];
	}
}

template <typename T>
static void inline xmem_set(T *dst, T value, size_t num)
{
	for (size_t i = 0; i < num; i++)
	{
		volatile T *dst_ptr = static_cast<volatile T *>(dst);
		dst_ptr[i] = value;
	}
}

#endif


#define XMEM_ASSIGN(TYPE, DST, VALUE)                    \
	do                                                     \
	{                                                      \
		volatile TYPE *dst_ptr = (volatile TYPE *)(DST);   \
		*dst_ptr = VALUE;                                  \
	} while (0);


#define XMEM_COPY(TYPE, DST, SRC, COUNT)                 \
	do                                                     \
	{                                                      \
		volatile TYPE *dst_ptr = (volatile TYPE *)(DST);   \
		volatile TYPE *src_ptr = (volatile TYPE *)(SRC);   \
		for (size_t i = 0; i < COUNT; i++)                 \
		{                                                  \
			dst_ptr[i] = src_ptr[i];                       \
		}                                                  \
	} while (0);


#define XMEM_SET(TYPE, DST, VAL, COUNT)                  \
	do                                                     \
	{                                                      \
		volatile TYPE *dst_ptr = (volatile TYPE *)(DST);   \
		for (size_t i = 0; i < COUNT; i++)                 \
		{                                                  \
			dst_ptr[i] = VAL;                              \
		}                                                  \
	} while (0);
/*
// Macros to access and use the xmem_1_t structure for HLS functions
#define XMEM_1_CONVN_VALID(MEM)                                             \
    xmem_1_convn_valid(MEM)

#define XMEM_1_CONV_FPROP(MEM)                                              \
    xmem_1_conv_fprop(MEM)

#define XMEM_1_MAX_POOLING_FPROP(MEM)                                       \
    xmem_1_max_pooling_fprop(MEM)

#define XMEM_1_FULLY_CONNECTED_FPROP(MEM)                                   \
    xmem_1_fully_connected_fprop(MEM)
    */

#endif // _XMEM_1_H_
