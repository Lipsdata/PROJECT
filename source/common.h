#ifndef COMMON_H
#define COMMON_H

// 定义 MAX 宏
#ifndef MIN
#define MIN(A,B)	(((A) <= (B)) ? (A) : (B))
#endif

#ifndef MAX
#define MAX(A,B)	(((A) >= (B)) ? (A) : (B))
#endif

#define uchar unsigned char

// CNN架构的常量定义
#define MAX_INPUT_SIZE 1024   
#define MAX_C1_SIZE 784       
#define MAX_C1_COUNT 6        
#define MAX_S2_SIZE 196       
#define MAX_S2_COUNT 6        
#define MAX_C3_SIZE 100       
#define MAX_C3_COUNT 16       
#define MAX_S4_SIZE 25        
#define MAX_S4_COUNT 16       
#define MAX_C5_SIZE 1         
#define MAX_C5_COUNT 120      
#define MAX_OUTPUT_COUNT 10   

#define MAX_KERNEL_SIZE 25    
#define MAX_C1_KERNEL_COUNT 6    
#define MAX_C3_KERNEL_COUNT 96   
#define MAX_C5_KERNEL_COUNT 1920 
#define MAX_OUTPUT_KERNEL_COUNT 1200 
#define MAX_SAMPLE_SIZE 1024  
#define MAX_LABEL_SIZE 10     

typedef struct _Sample
{
	float data[MAX_SAMPLE_SIZE];
	float label[MAX_LABEL_SIZE];

	int sample_w;
	int sample_h;
	int sample_count;
} Sample;

// 网络参数结构体
typedef struct _NetworkParams
{
    int map_w;        // 特征图宽度
    int map_h;        // 特征图高度
    int map_count;    // 特征图数量
    int kernel_w;     // 卷积核宽度
    int kernel_h;     // 卷积核高度
    int kernel_count; // 卷积核数量
} NetworkParams;

// 输入层结构体 - 1个通道，32x32
typedef struct _InputLayer
{
    float data[1][MAX_INPUT_SIZE];  // 输入数据
    float map_common[MAX_INPUT_SIZE]; // 公共映射区域
} InputLayer;

// C1卷积层结构体 - 6个通道，28x28
typedef struct _C1Layer
{
    float data[MAX_C1_COUNT][MAX_C1_SIZE];  // 卷积层输出数据
    float b[MAX_C1_COUNT];                  // 偏置值
    float W[MAX_C1_KERNEL_COUNT][MAX_KERNEL_SIZE]; // 卷积核权重
} C1Layer;

// S2池化层结构体 - 6个通道，14x14
typedef struct _S2Layer
{
    float data[MAX_S2_COUNT][MAX_S2_SIZE];  // 池化层输出数据
    float b[MAX_S2_COUNT];                  // 偏置值
    float map_common[MAX_INPUT_SIZE];       // 公共映射区域
} S2Layer;

// C3卷积层结构体 - 16个通道，10x10
typedef struct _C3Layer
{
    float data[MAX_C3_COUNT][MAX_C3_SIZE];  // 卷积层输出数据
    float b[MAX_C3_COUNT];                  // 偏置值
    float W[MAX_C3_KERNEL_COUNT][MAX_KERNEL_SIZE]; // 卷积核权重
} C3Layer;

// S4池化层结构体 - 16个通道，5x5
typedef struct _S4Layer
{
    float data[MAX_S4_COUNT][MAX_S4_SIZE];  // 池化层输出数据
    float b[MAX_S4_COUNT];                  // 偏置值
    float map_common[MAX_INPUT_SIZE];       // 公共映射区域
} S4Layer;

// C5全连接层结构体 - 120个通道，1x1
typedef struct _C5Layer
{
    float data[MAX_C5_COUNT];              // 全连接层输出数据
    float b[MAX_C5_COUNT];                 // 偏置值
    float W[MAX_C5_KERNEL_COUNT][MAX_KERNEL_SIZE]; // 权重
} C5Layer;

// 输出层结构体 - 10个通道，1x1
typedef struct _OutputLayer
{
    float data[MAX_OUTPUT_COUNT];           // 输出数据
    float b[MAX_OUTPUT_COUNT];              // 偏置值
    float W[MAX_OUTPUT_KERNEL_COUNT][MAX_KERNEL_SIZE]; // 权重
} OutputLayer;

#endif // COMMON_H
