#include "base.hpp"
#include <iostream>
Ort::BaseOnnx::BaseOnnx(const char *model_path)
{
    if (model_path == nullptr)
        throw std::runtime_error("model_path is null");
#ifdef _WIN32
    std::wstring model_path_wstr = std::wstring(model_path, model_path + strlen(model_path));
    session = new Ort::Session(env, model_path_wstr.c_str(), session_options);
#else
    session = new Ort::Session(env, model_path.c_str(), session_options);
#endif
    int num_input_nodes = static_cast<int>(session->GetInputCount());
    int num_output_nodes = static_cast<int>(session->GetOutputCount());

    for (int i = 0; i < num_input_nodes; ++i)
    {
        // push name
        auto input_name = session->GetInputNameAllocated(i, allocator);
        char * buffer = new char[strlen(input_name.get()) + 1];
        strcpy(buffer, input_name.get());
        input_node_names_p.push_back(buffer);
        // push shape
        Ort::TypeInfo type_info = session->GetInputTypeInfo(i);
        input_node_types.push_back(type_info.GetTensorTypeAndShapeInfo().GetElementType());
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        input_node_dims.push_back(tensor_info.GetShape());
    }

    for (int i = 0; i < num_output_nodes; ++i)
    {
        // push name
        auto output_name = session->GetOutputNameAllocated(i, allocator);
        char * buffer = new char[strlen(output_name.get()) + 1];
        strcpy(buffer, output_name.get());
        output_node_names_p.push_back(buffer);
        // push shape
        Ort::TypeInfo type_info = session->GetOutputTypeInfo(i);
        output_node_types.push_back(type_info.GetTensorTypeAndShapeInfo().GetElementType());
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        output_node_dims.push_back(tensor_info.GetShape());
    }

    // print name and size
    std::cout << "== Input nodes ==" << std::endl;
    for (int i = 0; i < input_node_names_p.size(); ++i)
    {
        std::cout << "(" << Ort::getTypeString(input_node_types[i]) << ") ";
        std::cout << "Input " << i << ": " << input_node_names_p[i] << ", shape: ";
        for (auto v : input_node_dims[i])
            std::cout << v << ", ";
        std::cout << std::endl;
    }
    std::cout << "== Output nodes ==" << std::endl;
    for (int i = 0; i < output_node_names_p.size(); ++i)
    {
        std::cout << "(" << Ort::getTypeString(output_node_types[i]) << ") ";
        std::cout << "Output " << i << ": " << output_node_names_p[i] << ", shape: ";
        for (auto v : output_node_dims[i])
            std::cout << v << ", ";
        std::cout << std::endl;
    }
    std::cout << "== End ==" << std::endl;
}

Ort::BaseOnnx::~BaseOnnx()
{
    delete session;
}

const char *Ort::getTypeString(ONNXTensorElementDataType type)
{
    switch (type)
    {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        return "float";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        return "float16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        return "double";
    default:
        return "unknown";
    }
}
