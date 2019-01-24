#ifndef SRU_RNN_H
#define SRU_RNN_H
#include "layer.h"
#include <string>

class SRURnn : public Layer {
public:
    SRURnn(int input_dim,int out_dim, int time_dim, float* W_context, float* W_out,
           float* f_W_gate, float* f_v_gate, float* f_b_gate,
           float* r_W_gate, float* r_v_gate, float* r_b_gate,
           std::string activation, bool use_last_output =false);

    void process(float* input,float* output);
    void free();

    int getOutputDim() { return out_dim_*time_dim_; }
    int getInputDim() { return input_dim_*time_dim_; }
private:
    float *W_context_;
    float *W_out_;
    float *f_W_gate_;
    float *f_v_gate_;
    float *f_b_gate_;
    float *r_W_gate_;
    float *r_v_gate_;
    float *r_b_gate_;

    bool use_last_output_;
    int time_dim_;
    std::string activation_;
};

#endif // SRU_RNN_H
