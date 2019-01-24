#include "ranker_utils.h"

#include <algorithm>
#include <math.h>

void Activation(std::string func_name, float* val, int dim) {
    if (func_name == "R") { //ReLU
        for(int i = 0; i < dim; i++) {
            val[i] = val[i] > 0 ? val[i] : 0;
        }
    } else if (func_name == "S") { //Sigmoid
        for(int i = 0; i < dim; i++) {
            val[i] = 0.5*tanh(val[i])+0.5;
        }
    } else if (func_name == "T") { //Tanh
        for(int i = 0; i < dim; i++) {
            val[i] = tanh(val[i]);
        }
    } else if (func_name == "E") { //ElU
        for(int i = 0; i < dim; i++) {
            val[i] = val[i] > 0 ? val[i] : exp(val[i])-1;
        }
    } else if (func_name == "SE") { //SELU
        double alpha = 1.6732632423543772848170429916717;
        double scale = 1.0507009873554804934193349852946;
        for(int i = 0; i < dim; i++) {
            val[i] = val[i] > 0 ? val[i] : alpha*(exp(val[i])-1);
            val[i] *= scale;
        }
    }
}

void Stable_Softmax(float* val, int dim) {
    float m = *std::max_element(val,val+dim);
    float den = 0;
    for (int i=0;i<dim;i++){
        val[i] = exp(val[i]-m);
        den += val[i];
    }
    for (int i=0;i<dim;i++){
        val[i] = val[i]/den;
    }
}

float hard_sigmoid(float x) {
    if (x < -2.5)
        return 0.0;
    if (x > 2.5)
        return 1.0;
    return 0.2*x+0.5;
}
