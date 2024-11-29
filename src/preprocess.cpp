#include<opencv2/opencv.hpp>
#include <torch/torch.h>
#include <string>
#include "preprocess.h"

torch::Tensor read_img_to_tensor(const std::string &path)
{
    cv::Mat img=cv::imread(path,cv::IMREAD_COLOR);
    torch::Tensor tensor_image=torch::from_blob(
        img.data,
        {img.rows,img.cols,img.channels()},
        torch::kUInt8
    );
    tensor_image=tensor_image.permute({2,0,1});
    return tensor_image;
}
