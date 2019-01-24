#ifndef DENSE_H
#define DENSE_H
#include <string>
#include "layer.h"

class Dense : public Layer {
public:
    Dense(int input_dim,int out_dim, float* weight, float* biases, std::string activation);
    void process(float* input,float* output);
    void free();
private:
    float *weights_;
    float *biases_;
    std::string activation_;
};

#endif // DENSE_H
