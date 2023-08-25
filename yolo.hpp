#pragma once
#include "base.hpp"
#include <opencv2/core.hpp>
namespace Ort
{
    struct Box
    {
    // public:
        int id = -1; // for sort, useless
        int x1, y1;
        int x2, y2;

        int cls;
        float conf = 0.0;
        bool valid = true;
        Box(int cls, float conf, int x1, int y1, int x2, int y2);
        // Box(const Box&) = default;
        // Box(Box&&) = default;
        // Box& operator=(const Box&) = default;
        // Box& operator=(Box&&) = default;

        float iou(const Box &other) const;
        float area() const;
        cv::Rect2i rect() const;
        friend std::ostream &operator<< (std::ostream &os, const Box &b);
    };

    // void nms_iou(std::vector<Box> &d, float iou_th = 0.45, int maxdet = 20, bool sortbyconf = true);

    cv::Mat tensor2mat(Ort::Value &tensor);

    class YOLO : public BaseOnnx
    {
    public:
        bool v5 = true;
        YOLO(const char *model_path);
        std::vector<Box> detect(const cv::Mat &image, float conf_th = 0.25, float iou_th = 0.45, int maxdet = 20);
    };

    void LetterBox(
        const cv::Mat &image,
        cv::Mat &outImage,
        cv::Rect2i &params,
        const cv::Size &newShape,
        bool autoShape = false,
        bool scaleFill = false,
        bool scaleUp = true,
        int stride = 32,
        const cv::Scalar &color = {114, 114, 114});
};
