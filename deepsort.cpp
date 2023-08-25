#include <onnxruntime_cxx_api.h>
#include "base.hpp"
#include "yolo.hpp"
#include "CLI11.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <filesystem>
#include "tracker.hpp"

int main(int argc, char **argv)
{
    CLI::App app{"YOLOv5-DeepSort"};
    std::string model_path = "v5.onnx";
    std::string image_path = "imgs/demo5.jpg";
    std::string output_path = "imgs/demo5_out.jpg";
    std::string video = "E:/track-yololabel/imgs";
    float nms_th = 0.45;
    float conf_th = 0.25;
    int max_det = 20;
    float k = 1e-3;
    bool v8 = false;
    float lost = 0.0;
    app.add_option("-m,--model", model_path, "Path to model");
    app.add_option("-i,--image", image_path, "Path to image");
    app.add_option("-o,--output", output_path, "Path to output image");
    app.add_option("-c,--conf", conf_th, "Confidence threshold");
    app.add_option("-t,--nms", nms_th, "NMS threshold");
    app.add_option("-d,--maxdet", max_det, "Max detections");
    app.add_option("-v,--video", video, "Path to video");
    app.add_flag("-8", v8, "yolov8 checkpoint");
    app.add_option("-k", k, "Kalman filter noise");
    app.add_option("-l", lost, "Lost threshold(probability)");
    CLI11_PARSE(app, argc, argv);

    Ort::YOLO yolo(model_path.c_str());
    if (v8)
        yolo.v5 = false;

    AllTrackers trackers;
    trackers.noise = k;

    if (video == "")
    {
        cv::Mat image = cv::imread(image_path);
        auto boxes = yolo.detect(image, conf_th, nms_th, max_det);
        std::cout << "Found " << boxes.size() << " boxes" << std::endl;
        trackers.update(boxes);
        trackers.render(image);
        cv::imwrite(output_path, image);
        return 0;
    }
    if (std::filesystem::status(video).type() == std::filesystem::file_type::directory)
    {
        for (int i = 4; i <= 7000; i = (i + 1) % 7001)
        {
            std::string image_path = video + "/" + std::to_string(i) + ".jpg";
            cv::Mat image = cv::imread(image_path);
            if (image.empty())
                continue;
            auto boxes = yolo.detect(image, conf_th, nms_th, max_det);
            std::cout << "Found " << boxes.size() << " boxes" << std::endl;
            if (!trackers.update(boxes, lost))
            {
                for (auto &b : boxes)
                {
                    cv::putText(image, std::to_string(b.conf), b.rect().br(),
                                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
                    cv::rectangle(image, b.rect(), {255, 0, 0}, 2);
                }
            }
            trackers.render(image);
            cv::imshow("YOLOv5-DeepSort", image);
            if (cv::waitKey(1) == 'q')
                break;
        }
        return 0;
    }
    cv::VideoCapture cap;
    if (video.size() == 1)
    {
        cap.open(video[0] - '0');
    }
    else
    {
        cap.open(video);
    }
    if (!cap.isOpened())
    {
        std::cerr << "Cannot open video" << std::endl;
        return 1;
    }
    cv::Mat frame;
    while (cap.read(frame))
    {
        auto boxes = yolo.detect(frame, conf_th, nms_th, max_det);
        std::cout << "Found " << boxes.size() << " boxes" << std::endl;

        if (!trackers.update(boxes, lost))
        {
            for (auto &b : boxes)
            {
                cv::putText(frame, std::to_string(b.conf), b.rect().br(),
                            cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
                cv::rectangle(frame, b.rect(), {255, 0, 0}, 2);
            }
        }
        trackers.render(frame);
        cv::imshow("YOLOv5-DeepSort", frame);
        if (cv::waitKey(1) == 'q')
            break;
    }
    return 0;
}