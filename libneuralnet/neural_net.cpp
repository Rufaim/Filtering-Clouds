#include "neural_net.h"

#include "dense.h"
#include "nalu.h"
#include "simple_attention.h"
#include "sru_rnn.h"
#include "ugrnn.h"

NeuralNet::NeuralNet(json net_config) {
    if (net_config.count("parameters")) {
        json additional_parameters = net_config["parameters"];
        use_last_rnn_out_ = additional_parameters["use_last_rnn_out"].get<bool>();
        num_layers_ = net_config.size()-1;
        assert(num_layers_ >= 1);
    } else {
        use_last_rnn_out_ = false;
        num_layers_ = net_config.size();
    }
    layers_ = new Layer*[num_layers_];
    std::string layer_prefix = "layer";

    for(int i = 0; i < num_layers_; i++) {
        auto curr_layer_name = layer_prefix + std::to_string(i);
        json curr_layer = net_config[curr_layer_name];


        std::string layer_type;
        // for previous versions capabilty
        if (curr_layer.count("type")) {
            layer_type = curr_layer["type"];
        } else {
            layer_type = "DENSE";
        }

        if (layer_type == "DENSE") {
            std::vector<std::vector<float>> W = curr_layer["W"];
            std::vector<float> b = curr_layer["b"];
            int in_dim = curr_layer["in_dim"];
            int out_dim = curr_layer["out_dim"];
            float* biases = new float[out_dim];
            float* weights = new float[out_dim * in_dim];

            for(int j = 0; j < out_dim; j++) {
                biases[j] = b.at(j);
                for(int k = 0; k < in_dim; k++) {
                    weights[j * in_dim + k] = W.at(k).at(j);
                }
            }
            layers_[i] = new Dense(in_dim,out_dim,weights,biases,curr_layer["activation"].get<std::string>());
        } else if (layer_type == "NALU") {
            std::vector<std::vector<float>> W = curr_layer["W"];
            std::vector<std::vector<float>> G = curr_layer["G"];
            int in_dim = curr_layer["in_dim"];
            int out_dim = curr_layer["out_dim"];
            float* gate = new float[out_dim * in_dim];
            float* weights = new float[out_dim * in_dim];

            for(int j = 0; j < out_dim; j++) {
                for(int k = 0; k < in_dim; k++) {
                    weights[j * in_dim + k] = W.at(k).at(j);
                    gate[j * in_dim + k] = G.at(k).at(j);
                }
            }
            layers_[i] = new Nalu(in_dim,out_dim,weights,gate);
        } else if (layer_type == "SIMPLE_ATTENTION") {
            std::vector<std::vector<float>> W = curr_layer["W"];
            std::vector<float> b = curr_layer["b"];
            std::vector<float> u = curr_layer["u"];

            int in_dim = curr_layer["in_dim"];
            int out_dim = curr_layer["out_dim"];
            int time_dim = curr_layer["time_dim"];
            float* weights = new float[out_dim * in_dim];
            float* biases = new float[out_dim];
            float* u_ = new float[out_dim];

            for(int j = 0; j < out_dim; j++) {
                biases[j] = b.at(j);
                u_[j] = u.at(j);
                for(int k = 0; k < in_dim; k++) {
                    weights[j * in_dim + k] = W.at(k).at(j);
                }
            }
            layers_[i] = new SimpleAttention(in_dim,out_dim,time_dim,
                                     weights,biases,u_,curr_layer["activation"].get<std::string>());
        } else if (layer_type == "UGRNN") {
            std::vector<std::vector<float>> W_context = curr_layer["W_context"];
            std::vector<float> b_context = curr_layer["b_context"];
            std::vector<std::vector<float>> W_gate = curr_layer["W_gate"];
            std::vector<float> b_gate = curr_layer["b_gate"];

            int in_dim = curr_layer["in_dim"];
            int out_dim = curr_layer["out_dim"];
            int time_dim = curr_layer["time_dim"];
            float* weights_context = new float[out_dim * in_dim * 2];
            float* biases_context = new float[out_dim];
            float* weights_gate = new float[out_dim * in_dim * 2];
            float* biases_gate = new float[out_dim];

            for(int j = 0; j < out_dim; j++) {
                biases_context[j] = b_context.at(j);
                biases_gate[j] = b_gate.at(j);
                for(int k = 0; k < in_dim; k++) {
                    weights_context[j * in_dim + k] = W_context.at(k).at(j);
                    weights_gate[j * in_dim + k] = W_gate.at(k).at(j);
                }
            }
            layers_[i] = new UGRnn(in_dim,out_dim,time_dim,weights_context,biases_context,
                            weights_gate,biases_gate,curr_layer["activation"].get<std::string>(),use_last_rnn_out_);
        } else if (layer_type == "SRURNN") {
            std::vector<std::vector<float>> W_context = curr_layer["W_context"];
            std::vector<std::vector<float>> W_out = curr_layer["W_out"];
            std::vector<std::vector<float>> f_W_gate = curr_layer["f_W_gate"];
            std::vector<float> f_v_gate = curr_layer["f_v_gate"];
            std::vector<float> f_b_gate = curr_layer["f_b_gate"];
            std::vector<std::vector<float>> r_W_gate = curr_layer["r_W_gate"];
            std::vector<float> r_v_gate = curr_layer["r_v_gate"];
            std::vector<float> r_b_gate = curr_layer["r_b_gate"];

            int in_dim = curr_layer["in_dim"];
            int out_dim = curr_layer["out_dim"];
            int time_dim = curr_layer["time_dim"];
            float* weights_context = new float[out_dim * in_dim];
            float* weights_out = new float[out_dim * in_dim];
            float* weights_f = new float[out_dim * in_dim];
            float* weights_r = new float[out_dim * in_dim];
            float* vector_f = new float[out_dim];
            float* vector_r = new float[out_dim];
            float* biases_f = new float[out_dim];
            float* biases_r = new float[out_dim];

            for(int j = 0; j < out_dim; j++) {
                vector_f[j] = f_v_gate.at(j);
                vector_r[j] = r_v_gate.at(j);
                biases_f[j] = f_b_gate.at(j);
                biases_r[j] = r_b_gate.at(j);
                for(int k = 0; k < in_dim; k++) {
                    weights_context[j * in_dim + k] = W_context.at(k).at(j);
                    weights_out[j * in_dim + k] = W_out.at(k).at(j);
                    weights_f[j * in_dim + k] = f_W_gate.at(k).at(j);
                    weights_r[j * in_dim + k] = r_W_gate.at(k).at(j);
                }
            }
            layers_[i] = new SRURnn(in_dim,out_dim,time_dim,weights_context,weights_out,
                          weights_f,vector_f,biases_f,weights_r,vector_r,biases_r,
                            curr_layer["activation"].get<std::string>(),use_last_rnn_out_);
        }
        if(i == 0) {
            max_dim_ = getInputDim();
        }
        if(layers_[i]->getOutputDim() > max_dim_) {
            max_dim_ = layers_[i]->getOutputDim();
        }
    }
}

NeuralNet::~NeuralNet() {
    free();
}

void NeuralNet::process(float *input, float *output) {
    float *curr_buff = new float[max_dim_];
    float *prev_buff = new float[max_dim_];

    for(int i = 0; i < num_layers_; i++) {
        if (i==0)
            layers_[i]->process(input,curr_buff);
        else if (i == num_layers_-1)
            layers_[i]->process(prev_buff,output);
        else
            layers_[i]->process(prev_buff,curr_buff);
        float *tmp = curr_buff;
        curr_buff = prev_buff;
        prev_buff = tmp;
    }
    delete[] curr_buff;
    delete[] prev_buff;
}

void NeuralNet::free() {
    for(int i = 0; i < num_layers_; i++) {
        layers_[i]->free();
    }
    delete [] layers_;
}
