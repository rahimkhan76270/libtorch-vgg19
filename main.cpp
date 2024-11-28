#include <torch/torch.h>
#include "vgg.h"
#include <string>

int main() {
    auto model=std::make_shared<VGGImpl>();
    std::string path="/home/rahim-khan/Work/libtorch-vgg19/vgg19.pt";
    torch::load(model,path);
    model->eval();
    torch::Tensor input = torch::randn({1, 3, 224, 224});

    torch::Tensor output = model->forward(input);

    // Print output shape
    std::cout << "Class " << output.softmax(-1).argmax() << std::endl;

    return 0;
}
