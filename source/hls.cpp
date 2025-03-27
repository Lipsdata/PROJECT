#include "hls.h"
#include "hls_config.h"
#include "xmem.h"
#include "common.h"

#ifndef MAX_MAP_SIZE
#define MAX_MAP_SIZE 1024  // 最大特征图大小 (32x32)
#endif

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
    inline static float sigmoid(float val) 
    { 
        return 1.0 / (1.0 + exp(-val)); 
    }

    inline static float dsigmoid(float val)
    { 
        return val * (1.0 - val); 
    }
};

#ifdef __cplusplus
extern "C" {
#endif

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

void IMPL(conv_fprop1)(HLS_COMMON_ARG NetworkParams *input_params, InputLayer *input_layer, 
                      NetworkParams *c1_params, C1Layer *c1_layer, uint8_t pconnection[96])
{
    #pragma HLS disaggregate variable=input_params
    #pragma HLS disaggregate variable=input_layer
    #pragma HLS disaggregate variable=c1_params
    #pragma HLS disaggregate variable=c1_layer

    #pragma HLS INTERFACE mode=ap_memory port=input_layer->data storage_type=RAM_1P latency=3
    #pragma HLS INTERFACE mode=ap_memory port=input_layer->map_common storage_type=RAM_1P latency=3

    #pragma HLS INTERFACE mode=ap_memory port=c1_layer->data storage_type=RAM_1P latency=3
    #pragma HLS INTERFACE mode=ap_memory port=c1_layer->b storage_type=RAM_1P latency=3
    #pragma HLS INTERFACE mode=ap_memory port=c1_layer->W storage_type=RAM_1P latency=3
    #pragma HLS INTERFACE mode=ap_memory port=input_layer->map_common storage_type=RAM_1P latency=3

    #pragma HLS INTERFACE mode=ap_memory port=pconnection storage_type=RAM_1P latency=3

    int index = 0;
    int size = c1_params->map_w * c1_params->map_h;
    for (int i = 0; i < c1_params->map_count; i++)
    {
        memset(input_layer->map_common, 0, size*sizeof(float));
        for (int j = 0; j < input_params->map_count; j++)
        {
            index = j*c1_params->map_count + i;
            if (pconnection != NULL && !pconnection[index])
            {
                continue;
            }
        
            convn_valid(
                input_layer->data[j], input_params->map_w, input_params->map_h, 
                c1_layer->W[index], c1_params->kernel_w, c1_params->kernel_h, 
                input_layer->map_common, c1_params->map_w, c1_params->map_h);
        }

        for (int k = 0; k < size; k++)
        {
            c1_layer->data[i][k] = activation_func::tan_h(input_layer->map_common[k] + c1_layer->b[i]);
        }
    }
}

void IMPL(conv_fprop2)(HLS_COMMON_ARG NetworkParams *s2_params, S2Layer *s2_layer,
                      NetworkParams *c3_params, C3Layer *c3_layer, uint8_t pconnection[96])
{
    #pragma HLS disaggregate variable=s2_params
    #pragma HLS disaggregate variable=s2_layer
    #pragma HLS disaggregate variable=c3_params
    #pragma HLS disaggregate variable=c3_layer

    #pragma HLS INTERFACE mode=ap_memory port=s2_layer->data storage_type=RAM_1P latency=3
    #pragma HLS INTERFACE mode=ap_memory port=s2_layer->b storage_type=RAM_1P latency=3

    #pragma HLS INTERFACE mode=ap_memory port=c3_layer->data storage_type=RAM_1P latency=3
    #pragma HLS INTERFACE mode=ap_memory port=c3_layer->b storage_type=RAM_1P latency=3
    #pragma HLS INTERFACE mode=ap_memory port=c3_layer->W storage_type=RAM_1P latency=3
    #pragma HLS INTERFACE mode=ap_memory port=s2_layer->map_common storage_type=RAM_1P latency=3

    #pragma HLS INTERFACE mode=ap_memory port=pconnection storage_type=RAM_1P latency=3

    int index = 0;
    int size = c3_params->map_w * c3_params->map_h;
    for (int i = 0; i < c3_params->map_count; i++)
    {
        memset(s2_layer->map_common, 0, size*sizeof(float));
        for (int j = 0; j < s2_params->map_count; j++)
        {
            index = j*c3_params->map_count + i;
            if (pconnection != NULL && !pconnection[index])
            {
                continue;
            }
        
            convn_valid(
                s2_layer->data[j], s2_params->map_w, s2_params->map_h, 
                c3_layer->W[index], c3_params->kernel_w, c3_params->kernel_h, 
                s2_layer->map_common, c3_params->map_w, c3_params->map_h);
        }

        for (int k = 0; k < size; k++)
        {
            c3_layer->data[i][k] = activation_func::tan_h(s2_layer->map_common[k] + c3_layer->b[i]);
        }
    }
}

void IMPL(conv_fprop3)(HLS_COMMON_ARG NetworkParams *s4_params, S4Layer *s4_layer,
                      NetworkParams *c5_params, C5Layer *c5_layer, uint8_t pconnection[96])
{
    #pragma HLS disaggregate variable=s4_params
    #pragma HLS disaggregate variable=s4_layer
    #pragma HLS disaggregate variable=c5_params
    #pragma HLS disaggregate variable=c5_layer

    #pragma HLS INTERFACE mode=ap_memory port=s4_layer->data storage_type=RAM_1P latency=3
    #pragma HLS INTERFACE mode=ap_memory port=s4_layer->b storage_type=RAM_1P latency=3
    #pragma HLS INTERFACE mode=ap_memory port=s4_layer->map_common storage_type=RAM_1P latency=3

    #pragma HLS INTERFACE mode=ap_memory port=c5_layer->data storage_type=RAM_1P latency=3
    #pragma HLS INTERFACE mode=ap_memory port=c5_layer->b storage_type=RAM_1P latency=3
    #pragma HLS INTERFACE mode=ap_memory port=c5_layer->W storage_type=RAM_1P latency=3
    #pragma HLS INTERFACE mode=ap_memory port=s4_layer->map_common storage_type=RAM_1P latency=3

    #pragma HLS INTERFACE mode=ap_memory port=pconnection storage_type=RAM_1P latency=3

    int index = 0;
    int size = c5_params->map_w * c5_params->map_h;
    for (int i = 0; i < c5_params->map_count; i++)
    {
        memset(s4_layer->map_common, 0, size*sizeof(float));
        for (int j = 0; j < s4_params->map_count; j++)
        {
            index = j*c5_params->map_count + i;
            if (pconnection != NULL && !pconnection[index])
            {
                continue;
            }
        
            convn_valid(
                s4_layer->data[j], s4_params->map_w, s4_params->map_h, 
                c5_layer->W[index], c5_params->kernel_w, c5_params->kernel_h, 
                s4_layer->map_common, c5_params->map_w, c5_params->map_h);
        }

        c5_layer->data[i] = activation_func::tan_h(s4_layer->map_common[0] + c5_layer->b[i]);
    }
}

void IMPL(max_pooling_fprop1)(HLS_COMMON_ARG NetworkParams *c1_params, C1Layer *c1_layer,
                             NetworkParams *s2_params, S2Layer *s2_layer)
{
    #pragma HLS disaggregate variable=c1_params
    #pragma HLS disaggregate variable=c1_layer
    #pragma HLS disaggregate variable=s2_params
    #pragma HLS disaggregate variable=s2_layer
    
    #pragma HLS INTERFACE mode=ap_memory port=c1_layer->data storage_type=RAM_1P latency=3
    #pragma HLS INTERFACE mode=ap_memory port=c1_layer->b storage_type=RAM_1P latency=3
    #pragma HLS INTERFACE mode=ap_memory port=c1_layer->W storage_type=RAM_1P latency=3

    #pragma HLS INTERFACE mode=ap_memory port=s2_layer->data storage_type=RAM_1P latency=3
    #pragma HLS INTERFACE mode=ap_memory port=s2_layer->b storage_type=RAM_1P latency=3
    #pragma HLS INTERFACE mode=ap_memory port=s2_layer->map_common storage_type=RAM_1P latency=3

    int map_w = s2_params->map_w;
    int map_h = s2_params->map_h;
    int upmap_w = c1_params->map_w;

    for (int k = 0; k < s2_params->map_count; k++)
    {
        for (int i = 0; i < map_h; i++)
        {
            for (int j = 0; j < map_w; j++)
            {
                float max_value = c1_layer->data[k][2*i*upmap_w + 2*j];
                for (int n = 2*i; n < 2*(i + 1); n++)
                {
                    for (int m = 2*j; m < 2*(j + 1); m++)
                    {
                        max_value = MAX(max_value, c1_layer->data[k][n*upmap_w + m]);
                    }
                }

                s2_layer->data[k][i*map_w + j] = activation_func::tan_h(max_value);
            }
        }
    }
}

void IMPL(max_pooling_fprop2)(HLS_COMMON_ARG NetworkParams *c3_params, C3Layer *c3_layer,
                             NetworkParams *s4_params, S4Layer *s4_layer)
{
    #pragma HLS disaggregate variable=c3_params
    #pragma HLS disaggregate variable=c3_layer
    #pragma HLS disaggregate variable=s4_params
    #pragma HLS disaggregate variable=s4_layer
    
    #pragma HLS INTERFACE mode=ap_memory port=c3_layer->data storage_type=RAM_1P latency=3
    #pragma HLS INTERFACE mode=ap_memory port=c3_layer->b storage_type=RAM_1P latency=3
    #pragma HLS INTERFACE mode=ap_memory port=c3_layer->W storage_type=RAM_1P latency=3

    #pragma HLS INTERFACE mode=ap_memory port=s4_layer->data storage_type=RAM_1P latency=3
    #pragma HLS INTERFACE mode=ap_memory port=s4_layer->b storage_type=RAM_1P latency=3
    #pragma HLS INTERFACE mode=ap_memory port=s4_layer->map_common storage_type=RAM_1P latency=3

    int s4_map_w = s4_params->map_w;
    int s4_map_h = s4_params->map_h;
    int c3_map_w = c3_params->map_w;

    for (int k = 0; k < s4_params->map_count; k++)
    {
        for (int i = 0; i < s4_map_h; i++)
        {
            for (int j = 0; j < s4_map_w; j++)
            {
                float max_value = c3_layer->data[k][2*i*c3_map_w + 2*j];
                for (int n = 2*i; n < 2*(i + 1); n++)
                {
                    for (int m = 2*j; m < 2*(j + 1); m++)
                    {
                        max_value = MAX(max_value, c3_layer->data[k][n*c3_map_w + m]);
                    }
                }

                s4_layer->data[k][i*s4_map_w + j] = activation_func::tan_h(max_value);
            }
        }
    }
}

void IMPL(fully_connected_fprop)(HLS_COMMON_ARG NetworkParams *c5_params, C5Layer *c5_layer,
                                NetworkParams *output_params, OutputLayer *output_layer)
{
    #pragma HLS disaggregate variable=c5_params
    #pragma HLS disaggregate variable=c5_layer
    #pragma HLS disaggregate variable=output_params
    #pragma HLS disaggregate variable=output_layer

    #pragma HLS INTERFACE mode=ap_memory port=c5_layer->data storage_type=RAM_1P latency=3
    #pragma HLS INTERFACE mode=ap_memory port=c5_layer->b storage_type=RAM_1P latency=3
    #pragma HLS INTERFACE mode=ap_memory port=c5_layer->W storage_type=RAM_1P latency=3

    #pragma HLS INTERFACE mode=ap_memory port=output_layer->data storage_type=RAM_1P latency=3
    #pragma HLS INTERFACE mode=ap_memory port=output_layer->b storage_type=RAM_1P latency=3
    #pragma HLS INTERFACE mode=ap_memory port=output_layer->W storage_type=RAM_1P latency=3

    for (int i = 0; i < output_params->map_count; i++) 
    {
        float sum = 0.0;
        for (int j = 0; j < c5_params->map_count; j++)
        {
            int idx = j*output_params->map_count + i;
            sum += c5_layer->data[j] * output_layer->W[idx][0];
        }

        sum += output_layer->b[i];
        output_layer->data[i] = activation_func::tan_h(sum);
    }
}

#ifdef __cplusplus
}
#endif

#if CAPTURE_COSIM
    #include "hls_capture.cpp"
#endif
