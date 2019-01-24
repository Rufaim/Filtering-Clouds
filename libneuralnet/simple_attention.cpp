#include "simple_attention.h"
#include "ranker_utils.h"
#include <math.h>

SimpleAttention::SimpleAttention(int input_dim,int out_dim, int time_dim, float* weight,
                                 float* biases, float* u, std::string activation) :
    weights_(weight),
    biases_(biases),
    u_(u),
    time_dim_(time_dim),
    activation_(activation),
    Layer (input_dim,out_dim)
{}

void SimpleAttention::process(float* input,float* output) {
    float* temp = new float[time_dim_ * out_dim_];
    for(int i = 0; i < time_dim_; i++) {
        for(int j = 0; j < out_dim_; j++) {
            float out = 0;
            for (int k =0; k< input_dim_; k++) {
                out += input[i*input_dim_ + k] * weights_[k*out_dim_+j];
            }
            out += biases_[j];
            temp[i*out_dim_ + j] = tanh(out);
        }
    }

    for(int i=0; i < time_dim_; i++) {
        float out = 0;
        for(int j=0; j < out_dim_; j++) {
            out += temp[i*out_dim_ + j] * u_[j];
        }
        temp[i] = out;
    }

    Stable_Softmax(temp,time_dim_);

    for(int i=0; i<out_dim_; i++) {
        float out = 0;
        for(int j=0; j < time_dim_; j++) {
            out += temp[j]*input[j*input_dim_+i];
        }
        output[i] = out;
    }

    Activation(activation_, output, out_dim_);
    delete [] temp;
}

void SimpleAttention::free() {
    delete [] weights_;
    delete [] biases_;
}
