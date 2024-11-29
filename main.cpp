#include <torch/torch.h>
#include <fstream>
#include "vgg.h"
#include "preprocess.h"
#include <string>
#include <iostream>
#include <unordered_map>

int main() {
    std::unordered_map<int,std::string> class_label;
    auto model=std::make_shared<VGGImpl>();
    std::string path="/home/rahim-khan/Work/libtorch-vgg19/vgg19.pt";
    std::string class_file="/home/rahim-khan/Work/libtorch-vgg19/classes.txt";
    std::string img_path="/home/rahim-khan/Downloads/Designer.png";
    torch::load(model,path);
    model->eval();
    std::ifstream file(class_file);
    if (!file) {
        std::cerr << "Unable to open file\n";
        return 1; 
    }
    int i=0;
    std::string line;
    while (std::getline(file, line)) {
        class_label[i]=line;
        i++;
    }
    file.close();
    torch::Tensor input = torch::randn({1, 3, 224, 224});

    torch::Tensor output = model->forward(input);

    // Print output shape
    std::cout << "Class " << output.softmax(-1).argmax() << std::endl;
    for(auto pr:class_label)
    {
        std::cout<<pr.first<<" "<<pr.second<<std::endl;
        break;
    }

    auto img=read_img_to_tensor(img_path);
    std::cout<<img<<std::endl;
    return 0;
}