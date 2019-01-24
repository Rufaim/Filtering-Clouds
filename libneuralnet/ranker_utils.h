#ifndef RANKER_UTILS_H
#define RANKER_UTILS_H
#include <string>
#include "assert.h"

void Activation(std::string func_name, float* val, int dim);
void Stable_Softmax(float* val, int dim);
float hard_sigmoid(float x);

#endif // RANKER_UTILS_H
