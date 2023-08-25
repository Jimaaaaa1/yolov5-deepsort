#pragma once
#include <onnxruntime_cxx_api.h>
namespace Ort
{
    class BaseOnnx
    {
    public:
        BaseOnnx(const char *model_path);
        ~BaseOnnx();

        BaseOnnx(const BaseOnnx &) = delete;
        BaseOnnx &operator=(const BaseOnnx &) = delete;
        BaseOnnx(BaseOnnx &&) = delete;

    protected:
        Env env = Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "SAM");
        SessionOptions session_options;
        Session *session = nullptr;
        AllocatorWithDefaultOptions allocator;
        MemoryInfo memory_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
        // input info
        std::vector<const char *> input_node_names_p;
        std::vector<std::vector<int64_t>> input_node_dims;
        std::vector<ONNXTensorElementDataType> input_node_types;

        // output info
        std::vector<const char *> output_node_names_p;
        std::vector<std::vector<int64_t>> output_node_dims;
        std::vector<ONNXTensorElementDataType> output_node_types;
    };

    const char* getTypeString(ONNXTensorElementDataType type);
}; // namespace Ort
