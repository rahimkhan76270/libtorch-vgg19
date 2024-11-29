#include <torch/torch.h>
#include <fstream>
#include "vgg.h"
#include "preprocess.h"
#include <string>
#include <iostream>
#include <unordered_map>

int main(int argc, char* argv[]) {
    std::unordered_map<int,std::string> class_label;
    auto model=std::make_shared<VGGImpl>();
    std::string path="/home/rahim-khan/Work/libtorch-vgg19/vgg19.pt";
    std::string class_file="/home/rahim-khan/Work/libtorch-vgg19/classes.txt";
    std::string img_path=argv[1];
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
    auto img=read_img_to_tensor(img_path);
    torch::Tensor output = model->forward(img.unsqueeze(0));
    std::cout<<class_label[output.softmax(-1).argmax().item<int>()]<<std::endl;
    // std::cout<<img.sizes()<<std::endl;
    return 0;
}