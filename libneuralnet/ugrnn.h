#ifndef UGRNN_H
#define UGRNN_H
#include "layer.h"
#include <string>

class UGRnn : public Layer
{
public:
    UGRnn(int input_dim,int out_dim, int time_dim, float* W_context,
          float* b_context,float* W_gate, float* b_gate_, std::string activation, bool use_last_output =false);
    void process(float* input,float* output);
    void free();

    int getOutputDim() { return out_dim_*time_dim_; }
    int getInputDim() { return input_dim_*time_dim_; }
private:
    float *W_context_;
    float *W_gate_;
    float *b_context_;
    float *b_gate_;

    bool use_last_output_;
    int time_dim_;
    std::string activation_;
};

#endif // UGRNN_H
