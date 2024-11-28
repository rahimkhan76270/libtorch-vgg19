#ifndef VGG_H
#define VGG_H

#include <torch/torch.h>

class VGGImpl :public torch::nn::Module
{
private:
    torch::nn::Sequential features{nullptr};
    torch::nn::AdaptiveAvgPool2d avgpool{nullptr};
    torch::nn::Sequential classifier{nullptr};

public:
    VGGImpl();
    torch::Tensor forward(torch::Tensor x);
};

// TORCH_MODULE(VGG);

#endif
