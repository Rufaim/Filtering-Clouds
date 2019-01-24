#ifndef NALU_H
#define NALU_H

#include "layer.h"

class Nalu : public Layer {
public:
    Nalu(int input_dim,int out_dim, float* weight, float* gate);
    void process(float* input,float* output);
    void free();
private:
    float *weights_;
    float *gate_;
};

#endif // NALU_H
