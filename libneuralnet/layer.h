#ifndef LAYER_H
#define LAYER_H


class Layer {
public:
    Layer(int input_dim, int out_dim) : input_dim_(input_dim), out_dim_(out_dim) {}
    virtual void process(float* input,float* output)=0;
    virtual void free()=0;

    virtual int getOutputDim() { return out_dim_; }
    virtual int getInputDim() { return input_dim_; }
protected:
    int input_dim_;
    int out_dim_;
};

#endif // LAYER_H
