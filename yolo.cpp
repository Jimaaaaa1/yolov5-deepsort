#include "yolo.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <numeric>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
void Ort::LetterBox(const cv::Mat &image, cv::Mat &outImage, cv::Rect2i &params, const cv::Size &newShape,
                    bool autoShape, bool scaleFill, bool scaleUp, int stride, const cv::Scalar &color)
{
    using namespace cv;

    cv::Size shape = image.size();
    float r = std::min((float)newShape.height / (float)shape.height,
                       (float)newShape.width / (float)shape.width);
    if (!scaleUp)
        r = std::min(r, 1.0f);

    float ratio[2]{r, r};
    int new_un_pad[2] = {(int)std::round((float)shape.width * r), (int)std::round((float)shape.height * r)};

    auto dw = (float)(newShape.width - new_un_pad[0]);
    auto dh = (float)(newShape.height - new_un_pad[1]);

    if (autoShape)
    {
        dw = (float)((int)dw % stride);
        dh = (float)((int)dh % stride);
    }
    else if (scaleFill)
    {
        dw = 0.0f;
        dh = 0.0f;
        new_un_pad[0] = newShape.width;
        new_un_pad[1] = newShape.height;
        ratio[0] = (float)newShape.width / (float)shape.width;
        ratio[1] = (float)newShape.height / (float)shape.height;
    }

    dw /= 2.0f;
    dh /= 2.0f;

    if (shape.width != new_un_pad[0] && shape.height != new_un_pad[1])
    {
        resize(image, outImage, cv::Size(new_un_pad[0], new_un_pad[1]));
    }
    else
    {
        outImage = image.clone();
    }

    int top = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));
    params = cv::Rect2i(left, top, new_un_pad[0], new_un_pad[1]);
    cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);

    std::cout << "letterbox: " << params << std::endl;
    cv::imwrite("imgs/letterbox.jpg", outImage);
}

Ort::Box::Box(int i, float conf, int x1, int y1, int x2, int y2) : cls(i), conf(conf)
{
    this->x1 = std::max(0, x1);
    this->y1 = std::max(0, y1);
    this->x2 = std::max(0, x2);
    this->y2 = std::max(0, y2);
}

float Ort::Box::iou(const Box &other) const
{
    auto max_x = std::max(x1, other.x1);
    auto min_x = std::min(x2, other.x2);
    auto max_y = std::max(y1, other.y1);
    auto min_y = std::min(y2, other.y2);
    if (min_x <= max_x || min_y <= max_y)
        return 0;
    float over_area = (float)(min_x - max_x) * (min_y - max_y); // 计算重叠面积
    float area_a = area();
    float area_b = other.area();
    float iou = over_area / (area_a + area_b - over_area);
    return iou;
}

float Ort::Box::area() const
{
    return static_cast<float>((x2 - x1) * (y2 - y1));
}

cv::Rect2i Ort::Box::rect() const
{
    return cv::Rect2i{x1, y1, x2 - x1, y2 - y1};
}
/*
void Ort::nms_iou(std::vector<Box> &d, float iou_th, int maxdet, bool sortbyconf)
{
    using namespace std;
    std::sort(d.begin(), d.end(), [](const Box &l, const Box &r)
              { return l.conf > r.conf; });
    auto n = d.size();
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < i; ++j)
        {
            if (d[j].valid && d[i].iou(d[j]) >= iou_th)
            {
                d[i].valid = false;
                break;
            }
        }
    }
    d.erase(remove_if(d.begin(), d.end(),
                      [](const Box &b)
                      { return !b.valid; }),
            d.end());
    if (d.size() > maxdet)
        d.erase(d.begin() + maxdet, d.end());

    if (sortbyconf)
        std::sort(d.begin(), d.end(), [](const Box &l, const Box &r)
                  { return l.conf < r.conf; });
    else
        std::sort(d.begin(), d.end(), [](const Box &l, const Box &r)
                  { return l.id < r.id; });
}
*/
Ort::YOLO::YOLO(const char *model_path) : BaseOnnx(model_path)
{
    assert(input_node_dims.size() == 1);
    assert(output_node_dims.size() >= 1);
    assert(input_node_dims[0].size() == 4);
    assert(input_node_dims[0][0] == 1);
    assert(input_node_dims[0][1] == 3);
    assert(input_node_dims[0][2] > 0);
    assert(input_node_dims[0][3] > 0);
}
std::vector<Ort::Box> Ort::YOLO::detect(const cv::Mat &image, float conf_th, float iou_th, int maxdet)
{
    cv::Mat letterbox;
    cv::Rect2i params;

    // preprocess
    Ort::LetterBox(image, letterbox, params,
                   cv::Size(input_node_dims[0][3], input_node_dims[0][2]));
    int width = letterbox.cols;
    int height = letterbox.rows;
    cv::dnn::blobFromImage(letterbox, letterbox, 1.0 / 255.0, letterbox.size(), cv::Scalar(), true, false);

    // convert to tensor
    Ort::Value input_tensor{nullptr};
    switch (input_node_types[0])
    {
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        letterbox.convertTo(letterbox, CV_16F);
        assert(letterbox.isContinuous());
        input_tensor = Ort::Value::CreateTensor<Float16_t>(
            memory_info, (Float16_t *)letterbox.data, letterbox.total(), input_node_dims[0].data(), input_node_dims[0].size());
        break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        assert(letterbox.isContinuous());
        input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, (float *)letterbox.data, letterbox.total(), input_node_dims[0].data(), input_node_dims[0].size());
        break;
    default:
        throw std::runtime_error("unknown type");
        break;
    }

    // inference
    std::vector<Ort::Value> output_tensor = session->Run(
        Ort::RunOptions{nullptr}, input_node_names_p.data(),
        &input_tensor, 1, output_node_names_p.data(), output_node_names_p.size());

    // get output
    cv::Mat pred = tensor2mat(output_tensor[0]);    // 1 N C or 1 C N
    pred = pred.reshape(1, output_node_dims[0][1]); // N C or C N
    if (!v5) pred = pred.t(); // v8: C N -> N C
    if (output_tensor.size() > 1){
        cv::Mat maskProtos = tensor2mat(output_tensor[1]); // 1 M H W
        int num_mask = maskProtos.size[1];
        pred = pred.colRange(0, pred.size[1] - num_mask); // remove mask weights
    }
    if (!v5){
        cv::Mat boxes = pred.colRange(0, 4);
        cv::Mat confs = pred.colRange(4, pred.cols);
        cv::Mat newconf;
        cv::reduce(confs, newconf, 1, cv::REDUCE_MAX);
        cv::hconcat(boxes, newconf, pred);
        cv::hconcat(pred, confs, pred);
    }
    if (pred.type() != CV_32F)
        pred.convertTo(pred, CV_32F);
    if (!pred.isContinuous())
        pred = pred.clone();
    int classes = pred.cols - 5;

    // post process
    std::vector<cv::Rect> boxes;
    std::vector<float> conf_vec;
    std::vector<int> indices;

    std::vector<int> cls_ids(classes);
    std::iota(cls_ids.begin(), cls_ids.end(), 0);
    for (int k = 0; k < pred.rows; ++k)
    {
        // 0-3: xywh 4: conf 5-84: cls
        float *line = pred.ptr<float>(k);
        if (line[4] < conf_th)
            continue;
        int cls = std::max_element(cls_ids.begin(), cls_ids.end(),
                                   [line](int l, int r)
                                   { return line[l + 5] < line[r + 5]; }) -
                  cls_ids.begin();
        conf_vec.push_back((line[4]) * (line[cls + 5]));

        boxes.emplace_back(
            std::min((line[0] - line[2] / 2 - params.x) / params.width * image.cols, (float)image.cols - 1.0f),
            std::min((line[1] - line[3] / 2 - params.y) / params.height * image.rows, (float)image.rows - 1.0f),
            std::min(line[2] / params.width * image.cols, (float)image.cols),
            std::min(line[3] / params.height * image.rows, (float)image.rows));
    }
    std::cout << "before nms:" << boxes.size() << std::endl;
    cv::dnn::NMSBoxes(boxes, conf_vec, conf_th, iou_th, indices, 1.0f, -1);
    std::cout << "after nms:" << indices.size() << std::endl;

    std::vector<Ort::Box> result;
    for (auto i : indices)
    {
        auto &box = boxes[i];
        std::cout << conf_vec[i] << " ";
        result.emplace_back(cls_ids[i], conf_vec[i], box.x, box.y, box.x + box.width, box.y + box.height);
    }
    std::cout << std::endl;
    std::sort(result.begin(), result.end(), [](const Ort::Box &l, const Ort::Box &r)
              { return l.conf > r.conf; });
    if (result.size() > maxdet)
        result.erase(result.begin() + maxdet, result.end());
    return result;
}

std::ostream& Ort::operator<< (std::ostream &os, const Ort::Box &b)
{
    os << "Box(" << b.x1 << ", " << b.y1 << ", " << b.x2 << ", " << b.y2 << ")";
    return os;
}

cv::Mat Ort::tensor2mat(Ort::Value &tensor)
{
    auto info = tensor.GetTensorTypeAndShapeInfo();
    auto dims = info.GetShape();
    std::vector<int> idims;
    for (auto v : dims)
        idims.push_back(v);
    switch (info.GetElementType())
    {
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        return cv::Mat(idims.size(), idims.data(), CV_32F, tensor.GetTensorMutableData<float>());
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        return cv::Mat(idims.size(), idims.data(), CV_16F, tensor.GetTensorMutableData<Ort::Float16_t>());
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        return cv::Mat(idims.size(), idims.data(), CV_8U, tensor.GetTensorMutableData<uint8_t>());
    default:
        throw std::runtime_error("unknown type");
    }
}