#ifndef PREPROCESSOR_H
#define PREPROCESSOR_H

#include<opencv2/opencv.hpp>
#include <torch/torch.h>
#include <string>

torch::Tensor read_img_to_tensor(const std::string &path);

#endif