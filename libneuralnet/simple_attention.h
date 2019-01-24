#ifndef SIMPLE_ATTENTION_H
#define SIMPLE_ATTENTION_H
#include <string>
#include "layer.h"

class SimpleAttention : public Layer {
public:
    SimpleAttention(int input_dim,int out_dim, int time_dim, float* weight,
                    float* biases, float* u, std::string activation);
    void process(float* input,float* output);
    void free();

    int getInputDim() { return input_dim_*time_dim_; }
private:
    float *weights_;
    float *biases_;
    float *u_;
    std::string activation_;
    int time_dim_;
};

#endif // SIMPLE_ATTENTION_H
