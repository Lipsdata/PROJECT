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


typedef struct _Layer1
{
    int map_w;
    int map_h;
    int map_count;
    int kernel_w;
    int kernel_h;
    int kernel_count;
} Layer1;


typedef struct _Layer2
{
    
    float c1_W[MAX_C1_KERNEL_COUNT][MAX_KERNEL_SIZE];  
    float c1_dW[MAX_C1_KERNEL_COUNT][MAX_KERNEL_SIZE]; 
    
    float c3_W[MAX_C3_KERNEL_COUNT][MAX_KERNEL_SIZE];  
    float c3_dW[MAX_C3_KERNEL_COUNT][MAX_KERNEL_SIZE]; 
    
    float c5_W[MAX_C5_KERNEL_COUNT][MAX_KERNEL_SIZE];  
    float c5_dW[MAX_C5_KERNEL_COUNT][MAX_KERNEL_SIZE]; 
    
    float output_W[MAX_OUTPUT_KERNEL_COUNT][MAX_KERNEL_SIZE]; 
    float output_dW[MAX_OUTPUT_KERNEL_COUNT][MAX_KERNEL_SIZE];
    
    float map_common[MAX_INPUT_SIZE];  // 共享的特征图临时存储区

    // 输入层 - 1个通道，32x32
    float input_data[1][MAX_INPUT_SIZE];
    float input_error[1][MAX_INPUT_SIZE];
    
    // C1层 - 6个通道，28x28
    float c1_data[MAX_C1_COUNT][MAX_C1_SIZE];
    float c1_error[MAX_C1_COUNT][MAX_C1_SIZE];
    float c1_b[MAX_C1_COUNT];
    float c1_db[MAX_C1_COUNT];
    
    // S2层 - 6个通道，14x14
    float s2_data[MAX_S2_COUNT][MAX_S2_SIZE];
    float s2_error[MAX_S2_COUNT][MAX_S2_SIZE];
    float s2_b[MAX_S2_COUNT];
    float s2_db[MAX_S2_COUNT];
    
    // C3层 - 16个通道，10x10
    float c3_data[MAX_C3_COUNT][MAX_C3_SIZE];
    float c3_error[MAX_C3_COUNT][MAX_C3_SIZE];
    float c3_b[MAX_C3_COUNT];
    float c3_db[MAX_C3_COUNT];
    
    // S4层 - 16个通道，5x5
    float s4_data[MAX_S4_COUNT][MAX_S4_SIZE];
    float s4_error[MAX_S4_COUNT][MAX_S4_SIZE];
    float s4_b[MAX_S4_COUNT];
    float s4_db[MAX_S4_COUNT];
    
    // C5层 - 120个通道，1x1
    float c5_data[MAX_C5_COUNT];
    float c5_error[MAX_C5_COUNT];
    float c5_b[MAX_C5_COUNT];
    float c5_db[MAX_C5_COUNT];
    
    // 输出层 - 10个通道，1x1
    float output_data[MAX_OUTPUT_COUNT];
    float output_error[MAX_OUTPUT_COUNT];
    float output_b[MAX_OUTPUT_COUNT];
    float output_db[MAX_OUTPUT_COUNT];
} Layer2;

#endif // COMMON_H
