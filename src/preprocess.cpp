#include<opencv2/opencv.hpp>
#include <torch/torch.h>
#include <string>
#include "preprocess.h"

torch::Tensor read_img_to_tensor(const std::string &path)
{
    std::vector<float> mean={0.485, 0.456, 0.406};
    std::vector<float> st_dev={0.229, 0.224, 0.225};
    cv::Mat img=cv::imread(path,cv::IMREAD_COLOR);
    cv::Mat img_resized;
    cv::resize(img,img_resized,{256,256},0,0,cv::INTER_LINEAR);
    cv::Rect rect(0,0,224,224);
    cv::Mat resized_img=img_resized(rect);
    cv::Mat float_img;
    resized_img.convertTo(float_img,CV_32FC3,1.0/255.0);
    std::vector<cv::Mat> channels(3);
    cv::split(float_img,channels);
    for(int i=0;i<3;i++)
    {
        channels[i] = (channels[i] - mean[i]) / st_dev[i];
    }
    cv::Mat merged_img;
    cv::merge(channels,merged_img);
    torch::Tensor tensor_image=torch::from_blob(
        merged_img.data,
        {merged_img.rows,merged_img.cols,merged_img.channels()},
        torch::kFloat32
    );
    tensor_image=tensor_image.permute({2,0,1});
    return tensor_image;
}
