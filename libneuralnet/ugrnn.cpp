#include "ugrnn.h"
#include "ranker_utils.h"

UGRnn::UGRnn(int input_dim, int out_dim, int time_dim,
             float *W_context, float *b_context, float *W_gate,
             float *b_gate, std::string activation,bool use_last_output) :

    W_context_(W_context),
    b_context_(b_context),
    W_gate_(W_gate),
    b_gate_(b_gate),
    time_dim_(time_dim),
    activation_(activation),
    use_last_output_(use_last_output),
    Layer (input_dim,out_dim)
{}

void UGRnn::process(float *input, float *output) {
    float* state=new float[out_dim_];
    float* temp_state=new float[out_dim_];
    for(int i=0;i<out_dim_;i++) {
        state[i]=0;
        temp_state[i]=0;
    }
    int inp_half = input_dim_/2;
    for(int time=0;time<time_dim_;time++) {
        for(int i = 0; i < out_dim_; i++) {
            float c = 0;
            float gate =0;
            for(int j = 0; j < input_dim_; j++) {
                if (j<inp_half) {
                    c += W_context_[i * input_dim_ + j] * input[time*input_dim_ +j];
                    gate += W_gate_[i * input_dim_ + j] * input[time*input_dim_ +j];
                }
                else {
                    c += W_context_[i * input_dim_ + j] * state[j-inp_half];
                    gate += W_gate_[i * input_dim_ + j] * state[j-inp_half];
                }
            }
            c += b_context_[i];
            gate += b_gate_[i];
            Activation(activation_, &c, 1);
            Activation("S", &gate, 1);

            output[time*input_dim_ +i] = gate * state[i] + (1-gate)* c;
            temp_state[i] = c;
        }
        float* t = state;
        state = temp_state;
        temp_state = t;
    }
    if (use_last_output_) {
        for(int i = 0; i < out_dim_; i++) {
            output[i] = output[(time_dim_-1)*time_dim_+i];
        }
    }
    delete [] state;
    delete [] temp_state;
}

void UGRnn::free() {
    delete [] W_context_;
    delete [] b_context_;
    delete [] W_gate_;
    delete [] b_gate_;
}

