cmake_minimum_required(VERSION 3.5)
project(deepsort-cpp LANGUAGES CXX)

set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_PREFIX_PATH E:/BaiduNetdiskDownload/opencv/build/x64/vc16/lib)
find_package(OpenCV REQUIRED)

include_directories(../include)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(../lib)
add_executable(deepsort deepsort.cpp yolo.cpp base.cpp tracker.cpp Hungarian.cpp)
target_link_libraries(deepsort ${OpenCV_LIBS} onnxruntime)

