#include "dense.h"
#include "ranker_utils.h"

Dense::Dense(int input_dim,int out_dim, float* weight, float* biases, std::string activation) :
    weights_(weight),
    biases_(biases),
    activation_(activation),
    Layer (input_dim,out_dim)
{}

void Dense::process(float* input,float* output) {
    for(int i = 0; i < out_dim_; i++) {
        float out = 0;
        for(int j = 0; j < input_dim_; j++) {
            out += weights_[i * input_dim_ + j] * input[j];
        }
        out += biases_[i];
        output[i] = out;
    }
    Activation(activation_, output, out_dim_);
}

void Dense::free() {
    delete [] weights_;
    delete [] biases_;
}

