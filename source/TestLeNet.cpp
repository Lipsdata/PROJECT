// TestLeNet.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <stdlib.h>
#include <memory.h>
#include <math.h>
#include <time.h>
#include "mt19937ar.h"
#include <string.h> 
//#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "hls.h"
#include "xmem.h"

/*
#define MAX_MAP_SIZE 1024     // 最大特征图大小 (32x32)
#define MAX_MAP_COUNT 120     // 每层最大特征图数量
#define MAX_KERNEL_SIZE 25    // 最大卷积核大小 (5x5)
#define MAX_KERNEL_COUNT 1920 // 最大卷积核数量 (16*120)
#define MAX_SAMPLE_SIZE 1024  // 最大样本大小 (32x32)
#define MAX_LABEL_SIZE 10     // 最大标签大小 (10类)
*/
// 基本参数----------------------------------------------------------------------------------------------/


const int batch_size = 10;
const int classes_count = 10;
const int width  = 32;
const int height = 32;
//const int train_sample_count = 60000;
//const int test_sample_count  = 10000;

//Layer input_layer, output_layer;
//Layer c1_conv_layer, c3_conv_layer, c5_conv_layer;
//Layer s2_pooling_layer, s4_pooling_layer;

//*-------------------------------------------------------------------------------------------------------/

// 初始化------------------------------------------------------------------------------------------------/
void init_layer(Layer1 *layer1, Layer2 *layer2, int prevlayer_map_count, int map_count, 
	int kernel_w, int kernel_h, int map_w, int map_h, bool is_pooling)
{
const float scale = 6.0;
int fan_in = 0;
int fan_out = 0;
if (is_pooling)
{
fan_in  = 4;
fan_out = 1;
}
else
{
fan_in = prevlayer_map_count * kernel_w * kernel_h;
fan_out = map_count * kernel_w * kernel_h;
}
int denominator = fan_in + fan_out;
float weight_base = (denominator != 0) ? sqrt(scale / (float)denominator) : 0.5;

// Initialize Layer1 (integer parameters)
layer1->kernel_count = prevlayer_map_count * map_count;
layer1->kernel_w = kernel_w;
layer1->kernel_h = kernel_h;
layer1->map_count = map_count;
layer1->map_w = map_w;
layer1->map_h = map_h;

// Initialize kernels based on the layer type
if (prevlayer_map_count == 0 && map_count == 1) {
    // 输入层，不需要初始化卷积核
} 
else if (map_count == MAX_C1_COUNT) {
    // C1层卷积核
    if (layer1->kernel_count > MAX_C1_KERNEL_COUNT) {
        printf("Error: C1 kernel_count exceeds MAX_C1_KERNEL_COUNT\n");
        return;
    }
    
    for (int i = 0; i < prevlayer_map_count; i++) {
        for (int j = 0; j < map_count; j++) {
            int idx = i*map_count + j;
            for (int k = 0; k < kernel_w*kernel_h; k++) {
                layer2->c1_W[idx][k] = (genrand_real1() - 0.5) * 2 * weight_base;
                //layer2->c1_dW[idx][k] = 0.0;
            }
        }
    }
} 
else if (map_count == MAX_C3_COUNT) {
    // C3层卷积核
    if (layer1->kernel_count > MAX_C3_KERNEL_COUNT) {
        printf("Error: C3 kernel_count exceeds MAX_C3_KERNEL_COUNT\n");
        return;
    }
    
    for (int i = 0; i < prevlayer_map_count; i++) {
        for (int j = 0; j < map_count; j++) {
            int idx = i*map_count + j;
            for (int k = 0; k < kernel_w*kernel_h; k++) {
                layer2->c3_W[idx][k] = (genrand_real1() - 0.5) * 2 * weight_base;
                //layer2->c3_dW[idx][k] = 0.0;
            }
        }
    }
} 
else if (map_count == MAX_C5_COUNT) {
    // C5层卷积核
    if (layer1->kernel_count > MAX_C5_KERNEL_COUNT) {
        printf("Error: C5 kernel_count exceeds MAX_C5_KERNEL_COUNT\n");
        return;
    }
    
    for (int i = 0; i < prevlayer_map_count; i++) {
        for (int j = 0; j < map_count; j++) {
            int idx = i*map_count + j;
            for (int k = 0; k < kernel_w*kernel_h; k++) {
                layer2->c5_W[idx][k] = (genrand_real1() - 0.5) * 2 * weight_base;
                //layer2->c5_dW[idx][k] = 0.0;
            }
        }
    }
} 
else if (map_count == MAX_OUTPUT_COUNT) {
    // 输出层卷积核
    if (layer1->kernel_count > MAX_OUTPUT_KERNEL_COUNT) {
        printf("Error: Output kernel_count exceeds MAX_OUTPUT_KERNEL_COUNT\n");
        return;
    }
    
    for (int i = 0; i < prevlayer_map_count; i++) {
        for (int j = 0; j < map_count; j++) {
            int idx = i*map_count + j;
            for (int k = 0; k < kernel_w*kernel_h; k++) {
                layer2->output_W[idx][k] = (genrand_real1() - 0.5) * 2 * weight_base;
                //layer2->output_dW[idx][k] = 0.0;
            }
        }
    }
} 
else if (is_pooling) {
    // S2和S4层不需要卷积核
} 
else {
    printf("Error: Unsupported layer configuration for kernel initialization\n");
    return;
}

// 初始化不同层的特征图和偏置
int size = map_w * map_h;

// 输入层
if (prevlayer_map_count == 0 && map_count == 1) {
    if (size > MAX_INPUT_SIZE) {
        printf("Error: input size exceeds MAX_INPUT_SIZE\n");
        return;
    }
    
    for (int k = 0; k < size; k++) {
        layer2->input_data[0][k] = 0.0;
        //layer2->input_error[0][k] = 0.0;
    }
}
// C1层
else if (map_count == MAX_C1_COUNT) {
    if (size > MAX_C1_SIZE) {
        printf("Error: C1 size exceeds MAX_C1_SIZE\n");
        return;
    }
    
    for (int i = 0; i < map_count; i++) {
        layer2->c1_b[i] = 0.0;
       // layer2->c1_db[i] = 0.0;
        
        for (int k = 0; k < size; k++) {
            layer2->c1_data[i][k] = 0.0;
            //layer2->c1_error[i][k] = 0.0;
        }
    }
}
// S2层
else if (map_count == MAX_S2_COUNT && is_pooling) {
    if (size > MAX_S2_SIZE) {
        printf("Error: S2 size exceeds MAX_S2_SIZE\n");
        return;
    }
    
    for (int i = 0; i < map_count; i++) {
        layer2->s2_b[i] = 0.0;
        //layer2->s2_db[i] = 0.0;
        
        for (int k = 0; k < size; k++) {
            layer2->s2_data[i][k] = 0.0;
           // layer2->s2_error[i][k] = 0.0;
        }
    }
}
// C3层
else if (map_count == MAX_C3_COUNT) {
    if (size > MAX_C3_SIZE) {
        printf("Error: C3 size exceeds MAX_C3_SIZE\n");
        return;
    }
    
    for (int i = 0; i < map_count; i++) {
        layer2->c3_b[i] = 0.0;
        //layer2->c3_db[i] = 0.0;
        
        for (int k = 0; k < size; k++) {
            layer2->c3_data[i][k] = 0.0;
            //layer2->c3_error[i][k] = 0.0;
        }
    }
}
// S4层
else if (map_count == MAX_S4_COUNT && is_pooling) {
    if (size > MAX_S4_SIZE) {
        printf("Error: S4 size exceeds MAX_S4_SIZE\n");
        return;
    }
    
    for (int i = 0; i < map_count; i++) {
        layer2->s4_b[i] = 0.0;
        //layer2->s4_db[i] = 0.0;
        
        for (int k = 0; k < size; k++) {
            layer2->s4_data[i][k] = 0.0;
            //layer2->s4_error[i][k] = 0.0;
        }
    }
}
// C5层
else if (map_count == MAX_C5_COUNT) {
    for (int i = 0; i < map_count; i++) {
        layer2->c5_data[i] = 0.0;
        //layer2->c5_error[i] = 0.0;
        layer2->c5_b[i] = 0.0;
        //layer2->c5_db[i] = 0.0;
    }
}
// 输出层
else if (map_count == MAX_OUTPUT_COUNT) {
    for (int i = 0; i < map_count; i++) {
        layer2->output_data[i] = 0.0;
        //layer2->output_error[i] = 0.0;
        layer2->output_b[i] = 0.0;
        //layer2->output_db[i] = 0.0;
    }
}
else {
    printf("Error: Unsupported layer configuration\n");
    return;
}

// 初始化共享的map_common
for (int k = 0; k < MAX_INPUT_SIZE; k++) {
    layer2->map_common[k] = 0.0;
}
}

void release_layer(Layer1 *layer1, Layer2 *layer2)
{
    int kernel_w = layer1->kernel_w;
    int kernel_h = layer1->kernel_h;
    int prevlayer_map_count = 0;
    int map_count = layer1->map_count;
    
    // 计算prevlayer_map_count
    if (layer1->kernel_count > 0 && map_count > 0) {
        prevlayer_map_count = layer1->kernel_count / map_count;
    }
    
    // 释放卷积核权重
    if (map_count == MAX_C1_COUNT) {
        // C1层卷积核
        for (int i = 0; i < prevlayer_map_count; i++) {
            for (int j = 0; j < map_count; j++) {
                int idx = i*map_count + j;
                for (int k = 0; k < kernel_w*kernel_h; k++) {
                    layer2->c1_W[idx][k] = 0.0;
                   // layer2->c1_dW[idx][k] = 0.0;
                }
            }
        }
    }
    else if (map_count == MAX_C3_COUNT) {
        // C3层卷积核
        for (int i = 0; i < prevlayer_map_count; i++) {
            for (int j = 0; j < map_count; j++) {
                int idx = i*map_count + j;
                for (int k = 0; k < kernel_w*kernel_h; k++) {
                    layer2->c3_W[idx][k] = 0.0;
                   // layer2->c3_dW[idx][k] = 0.0;
                }
            }
        }
    }
    else if (map_count == MAX_C5_COUNT) {
        // C5层卷积核
        for (int i = 0; i < prevlayer_map_count; i++) {
            for (int j = 0; j < map_count; j++) {
                int idx = i*map_count + j;
                for (int k = 0; k < kernel_w*kernel_h; k++) {
                    layer2->c5_W[idx][k] = 0.0;
                    //layer2->c5_dW[idx][k] = 0.0;
                }
            }
        }
    }
    else if (map_count == MAX_OUTPUT_COUNT) {
        // 输出层卷积核
        for (int i = 0; i < prevlayer_map_count; i++) {
            for (int j = 0; j < map_count; j++) {
                int idx = i*map_count + j;
                for (int k = 0; k < kernel_w*kernel_h; k++) {
                    layer2->output_W[idx][k] = 0.0;
                    //layer2->output_dW[idx][k] = 0.0;
                }
            }
        }
    }

    int size = layer1->map_w * layer1->map_h;

    // 输入层
    if (map_count == 1 && size <= MAX_INPUT_SIZE) {
        for (int k = 0; k < size; k++) {
            layer2->input_data[0][k] = 0.0;
            //layer2->input_error[0][k] = 0.0;
        }
    }
    // C1层
    else if (map_count == MAX_C1_COUNT && size <= MAX_C1_SIZE) {
        for (int i = 0; i < map_count; i++) {
            layer2->c1_b[i] = 0.0;
            //layer2->c1_db[i] = 0.0;
            
            for (int k = 0; k < size; k++) {
                layer2->c1_data[i][k] = 0.0;
                //layer2->c1_error[i][k] = 0.0;
            }
        }
    }
    // S2层
    else if (map_count == MAX_S2_COUNT && size <= MAX_S2_SIZE) {
        for (int i = 0; i < map_count; i++) {
            layer2->s2_b[i] = 0.0;
            //layer2->s2_db[i] = 0.0;
            
            for (int k = 0; k < size; k++) {
                layer2->s2_data[i][k] = 0.0;
                //layer2->s2_error[i][k] = 0.0;
            }
        }
    }
    // C3层
    else if (map_count == MAX_C3_COUNT && size <= MAX_C3_SIZE) {
        for (int i = 0; i < map_count; i++) {
            layer2->c3_b[i] = 0.0;
            //layer2->c3_db[i] = 0.0;
            
            for (int k = 0; k < size; k++) {
                layer2->c3_data[i][k] = 0.0;
                //layer2->c3_error[i][k] = 0.0;
            }
        }
    }
    // S4层
    else if (map_count == MAX_S4_COUNT && size <= MAX_S4_SIZE) {
        for (int i = 0; i < map_count; i++) {
            layer2->s4_b[i] = 0.0;
            //layer2->s4_db[i] = 0.0;
            
            for (int k = 0; k < size; k++) {
                layer2->s4_data[i][k] = 0.0;
                //layer2->s4_error[i][k] = 0.0;
            }
        }
    }
    // C5层
    else if (map_count == MAX_C5_COUNT) {
        for (int i = 0; i < map_count; i++) {
            layer2->c5_data[i] = 0.0;
            //layer2->c5_error[i] = 0.0;
            layer2->c5_b[i] = 0.0;
           // layer2->c5_db[i] = 0.0;
        }
    }
    // 输出层
    else if (map_count == MAX_OUTPUT_COUNT) {
        for (int i = 0; i < map_count; i++) {
            layer2->output_data[i] = 0.0;
           // layer2->output_error[i] = 0.0;
            layer2->output_b[i] = 0.0;
            //layer2->output_db[i] = 0.0;
        }
    }
    
    // 释放共享的map_common
    for (int k = 0; k < MAX_INPUT_SIZE; k++) {
        layer2->map_common[k] = 0.0;
    }
}
//*-------------------------------------------------------------------------------------------------------/



#if 0
//加速部分

// 激活函数----------------------------------------------------------------------------------------------/
struct activation_func
{
	/* scale: -0.8 ~ 0.8 和label初始值对应 */
	inline static float tan_h(float val)
	{
		float ep = exp(val);
		float em = exp(-val);

		return (ep - em) / (ep + em);
	}

	inline static float dtan_h(float val)
	{
		return 1.0 - val*val;
	}

	/* scale: 0.1 ~ 0.9 和label初始值对应 */
	inline static float relu(float val)
	{
		return val > 0.0 ? val : 0.0;
	}

	inline static float drelu(float val)
	{
		return val > 0.0 ? 1.0 : 0.0;
	}

	/* scale: 0.1 ~ 0.9 和label初始值对应 */
	inline float sigmoid(float val) 
	{ 
		return 1.0 / (1.0 + exp(-val)); 
	}

	float dsigmoid(float val)
	{ 
		return val * (1.0 - val); 
	}
};
//*-------------------------------------------------------------------------------------------------------/


// 卷积--------------------------------------------------------------------------------------------------/
void convn_valid(float in_data[MAX_MAP_SIZE], int in_w, int in_h, 
                float kernel[MAX_KERNEL_SIZE], int kernel_w, int kernel_h, 
                float out_data[MAX_MAP_SIZE], int out_w, int out_h)
{
    float sum = 0.0;
    for (int i = 0; i < out_h; i++)
    {
        for (int j = 0; j < out_w; j++)
        {
            sum = 0.0;
            for (int n = 0; n < kernel_h; n++)
            {
                for (int m = 0; m < kernel_w; m++)
                {
                    sum += in_data[(i + n)*in_w + j + m] * kernel[n*kernel_w + m];
                }
            }
            out_data[i*out_w + j] += sum;
        }
    }
}
//*-------------------------------------------------------------------------------------------------------/

// 正向传播----------------------------------------------------------------------------------------------/
#define O true
#define X false
bool connection_table[6*16] = 
{
	O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O,
	O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O,
	O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
	X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O,
	X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O,
	X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O
};
#undef O
#undef X

void conv_fprop(Layer *prev_layer, Layer *layer, uint8_t *pconnection)
{
	int index = 0;
	int size = layer->map_w * layer->map_h;
	for (int i = 0; i < layer->map_count; i++)
	{
		memset(layer->map_common, 0, size*sizeof(float));
		for (int j = 0; j < prev_layer->map_count; j++)
		{
			index = j*layer->map_count + i;
			if (pconnection != NULL && !pconnection[index])
			{
				continue;
			}
		
			convn_valid(
				prev_layer->data[j][0], prev_layer->map_w, prev_layer->map_h, 
				layer->W[index], layer->kernel_w, layer->kernel_h, 
				layer->map_common, layer->map_w, layer->map_h);
		}

		for (int k = 0; k < size; k++)
		{
			layer->data[i][k] = activation_func::tan_h(layer->map_common[k] + layer->b[i]);
		}
	}
}

/*void avg_pooling_fprop(Layer *prev_layer, Layer *layer)
{
	int map_w = layer->map_w;
	int map_h = layer->map_h;
	int upmap_w = prev_layer->map_w;
	const float scale_factor = 0.25;

	for (int k = 0; k < layer->map_count; k++)
	{
		for (int i = 0; i < map_h; i++)
		{
			for (int j = 0; j < map_w; j++)
			{
				float sum = 0.0;
				for (int n = 2*i; n < 2*(i + 1); n++)
				{
					for (int m = 2*j; m < 2*(j + 1); m++)
					{
						sum += prev_layer->data[k][n*upmap_w + m] * layer->W[k][0];
					}
				}

				sum *= scale_factor;
				sum += layer->b[k];
				layer->data[k][i*map_w + j] = activation_func::tan_h(sum);
			}
		}
	}
}*/

void max_pooling_fprop(Layer *prev_layer, Layer *layer)
{
	int map_w = layer->map_w;
	int map_h = layer->map_h;
	int upmap_w = prev_layer->map_w;

	for (int k = 0; k < layer->map_count; k++)
	{
		for (int i = 0; i < map_h; i++)
		{
			for (int j = 0; j < map_w; j++)
			{
				float max_value = prev_layer->data[k][2*i*upmap_w + 2*j];
				for (int n = 2*i; n < 2*(i + 1); n++)
				{
					for (int m = 2*j; m < 2*(j + 1); m++)
					{
						max_value = MAX(max_value, prev_layer->data[k][n*upmap_w + m]);
					}
				}

				layer->data[k][i*map_w + j] = activation_func::tan_h(max_value);
			}
		}
	}
}

void fully_connected_fprop(Layer *prev_layer, Layer *layer)
{
	for (int i = 0; i < layer->map_count; i++) 
	{
		float sum = 0.0;
		for (int j = 0; j < prev_layer->map_count; j++)
		{
			sum += prev_layer->data[j][0] * layer->W[j*layer->map_count + i][0];
		}

		sum += layer->b[i];
		layer->data[i][0] = activation_func::tan_h(sum);
	}
}

#endif
//到此为止


void forward_propagation(xmem_t *xmem)
{
    memcpy(xmem->pconnection, connection_table, sizeof(connection_table));
    
    conv_fprop1(&xmem->input_layer1, &xmem->input_layer2, 
                &xmem->c1_conv_layer1, &xmem->c1_conv_layer2, NULL);

    max_pooling_fprop1(&xmem->c1_conv_layer1, &xmem->c1_conv_layer2,
                      &xmem->s2_pooling_layer1, &xmem->s2_pooling_layer2);

    conv_fprop2(&xmem->s2_pooling_layer1, &xmem->s2_pooling_layer2,
                &xmem->c3_conv_layer1, &xmem->c3_conv_layer2, xmem->pconnection);

    max_pooling_fprop2(&xmem->c3_conv_layer1, &xmem->c3_conv_layer2,
                      &xmem->s4_pooling_layer1, &xmem->s4_pooling_layer2);

    conv_fprop3(&xmem->s4_pooling_layer1, &xmem->s4_pooling_layer2,
                &xmem->c5_conv_layer1, &xmem->c5_conv_layer2, NULL);

    fully_connected_fprop(&xmem->c5_conv_layer1, &xmem->c5_conv_layer2,
                         &xmem->output_layer1, &xmem->output_layer2);
}

int find_index(Layer1 *layer1, Layer2 *layer2)
{
    int index = 0;
    float max_val = layer2->output_data[0];
    for (int i = 1; i < layer1->map_count; i++)
    {
        if (layer2->output_data[i] > max_val)
        {
            max_val = layer2->output_data[i];
            index = i;
        }
    }
    return index;
}

/*
void save_model(const char* filename) {
    FILE *fp = NULL;
    fp =fopen(filename, "wb");
    if(!fp) {
        printf("Error: Could not open file for writing: %s\n", filename);
        return;
    }
    
    
    const char magic[] = "LENET5";
    fwrite(magic, sizeof(char), 6, fp);
    int version = 1;
    fwrite(&version, sizeof(int), 1, fp);
    
    // 写入每层的维度信息用于验证
    Layer* layers[] = {&c1_conv_layer, &s2_pooling_layer, &c3_conv_layer, 
                      &s4_pooling_layer, &c5_conv_layer, &output_layer};
    
    for(int l = 0; l < 6; l++) {
        Layer* layer = layers[l];
        fwrite(&layer->kernel_count, sizeof(int), 1, fp);
        fwrite(&layer->kernel_w, sizeof(int), 1, fp);
        fwrite(&layer->kernel_h, sizeof(int), 1, fp);
        fwrite(&layer->map_count, sizeof(int), 1, fp);
        
        // 保存卷积核权重
        for(int i = 0; i < layer->kernel_count; i++) {
            int size = layer->kernel_w * layer->kernel_h;
            size_t written = fwrite(layer->W[i], sizeof(float), size, fp);
            if(written != size) {
                printf("Error writing kernel weights for layer %d\n", l);
                fclose(fp);
                return;
            }
        }
        // 保存偏置
        for(int i = 0; i < layer->map_count; i++) {
            fwrite(&layer->b[i], sizeof(float), 1, fp);
        }
    }
    printf("Model saved successfully to %s\n", filename);
    fclose(fp);
}
*/
void load_model(xmem_t *xmem, const char* filename) {
    FILE *fp = NULL;
    fp = fopen(filename, "rb");
    if(!fp) {
        printf("Error: Could not open file for reading: %s\n", filename);
        return;
    }
    
    char magic[7] = {0};
    fread(magic, sizeof(char), 6, fp);
    if(strcmp(magic, "LENET5") != 0) {
        printf("Error: Invalid model file format\n");
        fclose(fp);
        return;
    }
    
    int version;
    fread(&version, sizeof(int), 1, fp);
    if(version != 1) {
        printf("Error: Unsupported model version\n");
        fclose(fp);
        return;
    }
    
    Layer1* layers1[] = {&xmem->c1_conv_layer1, &xmem->s2_pooling_layer1, &xmem->c3_conv_layer1, 
                        &xmem->s4_pooling_layer1, &xmem->c5_conv_layer1, &xmem->output_layer1};
    Layer2* layers2[] = {&xmem->c1_conv_layer2, &xmem->s2_pooling_layer2, &xmem->c3_conv_layer2,
                        &xmem->s4_pooling_layer2, &xmem->c5_conv_layer2, &xmem->output_layer2};
    
    for(int l = 0; l < 6; l++) {
        Layer1* layer1 = layers1[l];
        Layer2* layer2 = layers2[l];
        
        int kernel_count, kernel_w, kernel_h, map_count;
        fread(&kernel_count, sizeof(int), 1, fp);
        fread(&kernel_w, sizeof(int), 1, fp);
        fread(&kernel_h, sizeof(int), 1, fp);
        fread(&map_count, sizeof(int), 1, fp);
        
        if(kernel_count != layer1->kernel_count || 
           kernel_w != layer1->kernel_w ||
           kernel_h != layer1->kernel_h ||
           map_count != layer1->map_count) {
            printf("Error: Model architecture mismatch at layer %d\n", l);
            fclose(fp);
            return;
        }
        
        // 根据不同的层读取卷积核权重
        if (l == 0) { // C1层
            for(int i = 0; i < layer1->kernel_count; i++) {
                int size = layer1->kernel_w * layer1->kernel_h;
                size_t read = fread(layer2->c1_W[i], sizeof(float), size, fp);
                if(read != size) {
                    printf("Error reading kernel weights for layer %d\n", l);
                    fclose(fp);
                    return;
                }
            }
            
            // 读取C1层偏置
            for(int i = 0; i < layer1->map_count; i++) {
                fread(&layer2->c1_b[i], sizeof(float), 1, fp);
            }
        }
        else if (l == 1) { // S2层
            for(int i = 0; i < layer1->kernel_count; i++) {
                int size = layer1->kernel_w * layer1->kernel_h;
                // S2层没有单独的卷积核，跳过它们
                float dummy[MAX_KERNEL_SIZE];
                size_t read = fread(dummy, sizeof(float), size, fp);
                if(read != size) {
                    printf("Error reading kernel weights for layer %d\n", l);
                    fclose(fp);
                    return;
                }
            }
            
            // 读取S2层偏置
            for(int i = 0; i < layer1->map_count; i++) {
                fread(&layer2->s2_b[i], sizeof(float), 1, fp);
            }
        }
        else if (l == 2) { // C3层
            for(int i = 0; i < layer1->kernel_count; i++) {
                int size = layer1->kernel_w * layer1->kernel_h;
                size_t read = fread(layer2->c3_W[i], sizeof(float), size, fp);
                if(read != size) {
                    printf("Error reading kernel weights for layer %d\n", l);
                    fclose(fp);
                    return;
                }
            }
            
            // 读取C3层偏置
            for(int i = 0; i < layer1->map_count; i++) {
                fread(&layer2->c3_b[i], sizeof(float), 1, fp);
            }
        }
        else if (l == 3) { // S4层
            for(int i = 0; i < layer1->kernel_count; i++) {
                int size = layer1->kernel_w * layer1->kernel_h;
                // S4层没有单独的卷积核，跳过它们
                float dummy[MAX_KERNEL_SIZE];
                size_t read = fread(dummy, sizeof(float), size, fp);
                if(read != size) {
                    printf("Error reading kernel weights for layer %d\n", l);
                    fclose(fp);
                    return;
                }
            }
            
            // 读取S4层偏置
            for(int i = 0; i < layer1->map_count; i++) {
                fread(&layer2->s4_b[i], sizeof(float), 1, fp);
            }
        }
        else if (l == 4) { // C5层
            for(int i = 0; i < layer1->kernel_count; i++) {
                int size = layer1->kernel_w * layer1->kernel_h;
                size_t read = fread(layer2->c5_W[i], sizeof(float), size, fp);
                if(read != size) {
                    printf("Error reading kernel weights for layer %d\n", l);
                    fclose(fp);
                    return;
                }
            }
            
            // 读取C5层偏置
            for(int i = 0; i < layer1->map_count; i++) {
                fread(&layer2->c5_b[i], sizeof(float), 1, fp);
            }
        }
        else if (l == 5) { // 输出层
            for(int i = 0; i < layer1->kernel_count; i++) {
                int size = layer1->kernel_w * layer1->kernel_h;
                size_t read = fread(layer2->output_W[i], sizeof(float), size, fp);
                if(read != size) {
                    printf("Error reading kernel weights for layer %d\n", l);
                    fclose(fp);
                    return;
                }
            }
            
            // 读取输出层偏置
            for(int i = 0; i < layer1->map_count; i++) {
                fread(&layer2->output_b[i], sizeof(float), 1, fp);
            }
        }
    }
    fclose(fp);
}
void load_and_preprocess_image(const char* image_path, float* image_data) {
    memset(image_data, 0, width * height * sizeof(float));

    int img_width, img_height, channels;
    unsigned char* img = stbi_load(image_path, &img_width, &img_height, &channels, 1);
    
    if (!img) {
        printf("Error: Could not load image %s\n", image_path);
        return;
    }

    float scale_max =  1.0;
    float scale_min = -1.0;
    int padding = 2;

    for (int i = 0; i < width * height; i++) {
        image_data[i] = scale_min;
    }

    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            if (i < img_height && j < img_width) {
                unsigned char pixel = img[i * img_width + j];
                float normalized = ((float)pixel / 255.0) * (scale_max - scale_min) + scale_min;
                image_data[(i + padding)*width + j + padding] = normalized;
            }
        }
    }

    stbi_image_free(img);
}

int predict_single_image(xmem_t *xmem, const char* image_path) {
    float* image_data = (float*)malloc(width * height * sizeof(float));
    load_and_preprocess_image(image_path, image_data);
    
    memcpy(xmem->input_layer2.input_data[0], image_data, width * height * sizeof(float));
    
    forward_propagation(xmem);
    
    float max_prob = -1;
    int prediction = 0;
    for (int i = 0; i < 10; i++) {
        float prob = xmem->output_layer2.output_data[i];
        if (prob > max_prob) {
            max_prob = prob;
            prediction = i;
        }
    }
    
    free(image_data);
    return prediction;
}

void test_custom_image(xmem_t *xmem, const char* model_path, const char* image_path) {
    int kernel_w = 0, kernel_h = 0;
    
    // Input layer
    init_layer(&xmem->input_layer1, &xmem->input_layer2, 0, 1, kernel_w, kernel_h, width, height, false);

    // C1 conv layer
    kernel_w = 5; kernel_h = 5;
    init_layer(&xmem->c1_conv_layer1, &xmem->c1_conv_layer2, 1, 6, kernel_w, kernel_h, 
              xmem->input_layer1.map_w - kernel_w + 1, xmem->input_layer1.map_h - kernel_h + 1, false);

    // S2 pooling layer
    kernel_w = 1; kernel_h = 1;
    init_layer(&xmem->s2_pooling_layer1, &xmem->s2_pooling_layer2, 1, 6, kernel_w, kernel_h, 
              xmem->c1_conv_layer1.map_w / 2, xmem->c1_conv_layer1.map_h / 2, true);

    // C3 conv layer
    kernel_w = 5; kernel_h = 5;
    init_layer(&xmem->c3_conv_layer1, &xmem->c3_conv_layer2, 6, 16, kernel_w, kernel_h, 
              xmem->s2_pooling_layer1.map_w - kernel_w + 1, xmem->s2_pooling_layer1.map_h - kernel_h + 1, false);

    // S4 pooling layer
    kernel_w = 1; kernel_h = 1;
    init_layer(&xmem->s4_pooling_layer1, &xmem->s4_pooling_layer2, 1, 16, kernel_w, kernel_h, 
              xmem->c3_conv_layer1.map_w / 2, xmem->c3_conv_layer1.map_h / 2, true);

    // C5 conv layer
    kernel_w = 5; kernel_h = 5;
    init_layer(&xmem->c5_conv_layer1, &xmem->c5_conv_layer2, 16, 120, kernel_w, kernel_h, 
              xmem->s4_pooling_layer1.map_w - kernel_w + 1, xmem->s4_pooling_layer1.map_h - kernel_h + 1, false);

    // Output layer
    kernel_w = 1; kernel_h = 1;
    init_layer(&xmem->output_layer1, &xmem->output_layer2, 120, 10, kernel_w, kernel_h, 1, 1, false);

    load_model(xmem, model_path);
  
    int prediction = predict_single_image(xmem, image_path);
    printf("Predicted digit: %d\n", prediction);

    release_layer(&xmem->input_layer1, &xmem->input_layer2);
    release_layer(&xmem->c1_conv_layer1, &xmem->c1_conv_layer2);
    release_layer(&xmem->c3_conv_layer1, &xmem->c3_conv_layer2);
    release_layer(&xmem->c5_conv_layer1, &xmem->c5_conv_layer2);
    release_layer(&xmem->s2_pooling_layer1, &xmem->s2_pooling_layer2);
    release_layer(&xmem->s4_pooling_layer1, &xmem->s4_pooling_layer2);
    release_layer(&xmem->output_layer1, &xmem->output_layer2);
}




int main(int argc, char* argv[])
{
    if (argc < 3) {
        printf("Usage: %s model_path image_path\n", argv[0]);
        return 1;
    }
    #if __riscv && HLS_XMEM
    xmem_t *xmem = (xmem_t*)get_riscv_xmem_base();
#else
    xmem_t *xmem = (xmem_t*)malloc(sizeof(xmem_t));
    if (xmem == NULL){
        printf("cannot allocate xmem\n");
        exit(1);
    }
#endif
    const char* model_path = argv[1];
    const char* image_path = argv[2];
    
    init_genrand((unsigned long)time(NULL));

    test_custom_image(xmem, model_path, image_path);

    system("pause");
    return 0;
}

