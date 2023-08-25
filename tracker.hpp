#pragma once
#include <opencv2/video/tracking.hpp>
#include <iostream>
#include "yolo.hpp"
#include "Hungarian.h"
#include <opencv2/core.hpp>
#include <set>
#include <vector>

class Tracker
{
public:
    cv::KalmanFilter KF;
    Ort::Box s;
    Tracker(const Ort::Box& b, int n = 4, float noise = 1e-3);
    int lost = 0;
    void predict();
    void update(Ort::Box box);
    float cost(Ort::Box box);
};

class AllTrackers
{
public:
    std::vector<Tracker> trackers;
    int states = 8;
    float noise = 1e-3;
    bool update(const std::vector<Ort::Box>& boxes, float lost = 0.0);
    void render(cv::Mat &image);
    HungarianAlgorithm hungarian;
};

std::vector<std::pair<int, int>> hungarian(const cv::Mat &costs);