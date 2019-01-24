#include "sru_rnn.h"
#include "ranker_utils.h"

SRURnn::SRURnn(int input_dim, int out_dim, int time_dim, float *W_context, float *W_out,
               float *f_W_gate, float *f_v_gate, float *f_b_gate,
               float *r_W_gate, float *r_v_gate, float *r_b_gate,
               std::string activation, bool use_last_output) :
    W_context_(W_context),
    W_out_(W_out),
    f_W_gate_(f_W_gate),
    f_v_gate_(f_v_gate),
    f_b_gate_(f_b_gate),
    r_W_gate_(r_W_gate),
    r_v_gate_(r_v_gate),
    r_b_gate_(r_b_gate),
    time_dim_(time_dim),
    activation_(activation),
    use_last_output_(use_last_output),
    Layer (input_dim,out_dim)
{}

void SRURnn::process(float *input, float *output) {
    float* state=new float[out_dim_];
    float* temp_state=new float[out_dim_];
    for(int i=0;i<out_dim_;i++) {
        state[i]=0;
        temp_state[i]=0;
    }
    for(int time=0;time<time_dim_;time++) {
        for(int i = 0; i < out_dim_; i++) {
            float c = 0,out=0, f_gate=0,r_gate=0;
            for(int j = 0; j < input_dim_; j++) {
                f_gate += f_W_gate_[i* input_dim_ + j] * input[time*input_dim_ +j];
                r_gate += f_W_gate_[i* input_dim_ + j] * input[time*input_dim_ +j];
                c += W_context_[i* input_dim_ + j] * input[time*input_dim_ +j];
                out += W_out_[i* input_dim_ + j] * input[time*input_dim_ +j];
            }
            f_gate += f_b_gate_[i] + f_v_gate_[i] * state[i];
            r_gate += r_b_gate_[i] + r_v_gate_[i] * state[i];
            Activation(activation_, &c, 1);
            Activation("S", &f_gate, 1);
            Activation("S", &r_gate, 1);

            c = f_gate * state[i] + (1-f_gate) * c;
            out = r_gate * c + (1-r_gate) * out;

            output[time*input_dim_ +i] = out;
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

void SRURnn::free() {
    delete[] W_context_;
    delete[] W_out_;
    delete[] f_W_gate_;
    delete[] f_v_gate_;
    delete[] f_b_gate_;
    delete[] r_W_gate_;
    delete[] r_v_gate_;
    delete[] r_b_gate_;
}
