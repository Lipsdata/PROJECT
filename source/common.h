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

// Constants for CNN architecture
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

// 第一个结构体包含所有整数参数
typedef struct _Layer1
{
    int map_w;
    int map_h;
    int map_count;
    int kernel_w;
    int kernel_h;
    int kernel_count;
} Layer1;

// 第二个结构体包含所有数组数据 - 仅推理所需
typedef struct _Layer2
{
    // 卷积核权重 - 每层独立存储
    float c1_W[MAX_C1_KERNEL_COUNT][MAX_KERNEL_SIZE];  
    float c3_W[MAX_C3_KERNEL_COUNT][MAX_KERNEL_SIZE];  
    float c5_W[MAX_C5_KERNEL_COUNT][MAX_KERNEL_SIZE];  
    float output_W[MAX_OUTPUT_KERNEL_COUNT][MAX_KERNEL_SIZE]; 
    
    float map_common[MAX_INPUT_SIZE]; 
    
    // 输入层 - 1个通道，32x32
    float input_data[1][MAX_INPUT_SIZE];
    
    // C1层 - 6个通道，28x28
    float c1_data[MAX_C1_COUNT][MAX_C1_SIZE];
    float c1_b[MAX_C1_COUNT];
    
    // S2层 - 6个通道，14x14
    float s2_data[MAX_S2_COUNT][MAX_S2_SIZE];
    float s2_b[MAX_S2_COUNT];
    
    // C3层 - 16个通道，10x10
    float c3_data[MAX_C3_COUNT][MAX_C3_SIZE];
    float c3_b[MAX_C3_COUNT];
    
    // S4层 - 16个通道，5x5
    float s4_data[MAX_S4_COUNT][MAX_S4_SIZE];
    float s4_b[MAX_S4_COUNT];
    
    // C5层 - 120个通道，1x1
    float c5_data[MAX_C5_COUNT];
    float c5_b[MAX_C5_COUNT];
    
    // 输出层 - 10个通道，1x1
    float output_data[MAX_OUTPUT_COUNT];
    float output_b[MAX_OUTPUT_COUNT];
} Layer2;

#endif // COMMON_H
