#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include "layer.h"
#include "json/json.hpp"

using json = nlohmann::json;

class NeuralNet
{
public:
    NeuralNet(json net_config);
    ~NeuralNet();

    void process(float* input,float* output);
    void free();

    int getInputDim() { return layers_[0]->getInputDim(); }
    int getOutputDim() { return layers_[num_layers_-1]->getOutputDim(); }
private:
    Layer** layers_;
    int num_layers_;
    int max_dim_;
    bool use_last_rnn_out_;
};

#endif // NEURAL_NET_H
