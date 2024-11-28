#include "vgg.h"

VGGImpl::VGGImpl()
{
    features = torch::nn::Sequential(
        // Block 1
        torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 3).stride(1).padding(1)),
        torch::nn::ReLU(true),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1)),
        torch::nn::ReLU(true),
        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)),

        // Block 2
        torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(1).padding(1)),
        torch::nn::ReLU(true),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).stride(1).padding(1)),
        torch::nn::ReLU(true),
        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)),

        // Block 3
        torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).stride(1).padding(1)),
        torch::nn::ReLU(true),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1)),
        torch::nn::ReLU(true),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1)),
        torch::nn::ReLU(true),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1)),
        torch::nn::ReLU(true),
        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)),

        // Block 4
        torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 3).stride(1).padding(1)),
        torch::nn::ReLU(true),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
        torch::nn::ReLU(true),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
        torch::nn::ReLU(true),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
        torch::nn::ReLU(true),
        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)),

        // Block 5
        torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
        torch::nn::ReLU(true),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
        torch::nn::ReLU(true),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
        torch::nn::ReLU(true),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
        torch::nn::ReLU(true),
        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
    register_module("features", features);

    avgpool = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({7, 7}));
    register_module("avgpool", avgpool);

    classifier = torch::nn::Sequential(
        torch::nn::Linear(512 * 7 * 7, 4096),
        torch::nn::ReLU(true),
        torch::nn::Dropout(0.5),
        torch::nn::Linear(4096, 4096),
        torch::nn::ReLU(true),
        torch::nn::Dropout(0.5),
        torch::nn::Linear(4096, 1000));
    register_module("classifier", classifier);
}

torch::Tensor VGGImpl::forward(torch::Tensor x)
{
    x = features->forward(x);
    x = avgpool->forward(x);
    x = x.view({x.size(0), -1});
    x = classifier->forward(x);
    return x;
}
