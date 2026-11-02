/*
 * @Author: Anonymous anonymous@example.com
 * @Date: 2023-04-23 23:46:27
 * @LastEditors: Anonymous anonymous@example.com
 * @LastEditTime: 2024-08-17 22:55:14
 * @FilePath: /postgres-DB4AI/src/udf/torch_wrapper.h
 * @Description: Multi-model management class, including variable argument passing
 */
#pragma once
#ifndef _MODEL_MANAGER_H_
#define _MODEL_MANAGER_H_

#include "env.h"

#include <stdbool.h>
#include <string>
#include <unordered_map>

// callback parameter
typedef union {
    void* ptr;
    int   integer;
    float8 floating;
} Args;


class ModelManager{
public:
    using PreProcessCallback = bool(*)(std::vector<torch::jit::IValue>&, Args*);
    using OutputProcessFloatCallback = bool(*)(torch::jit::IValue&, Args*, float8&);
    using OutputProcessTextCallback = bool(*)(torch::jit::IValue&, Args*, std::string&);
    
    ModelManager();
    ~ModelManager();
    /**
     * @description: 
     * @event: 
     * @param {string } model_path
     * @return {*}
     */    
    bool LoadModel(std::string model_name, std::string  model_path);

    /**
     * @description: 
     * @event: 
     * @param {string} model_type model category, e.g. classification
     * @param {string} model_name model name
     * @param {string} model_path path of the model file on the local filesystem
     * @param {string} discription model description
     * @return {*}
     */    
    bool CreateModel(const std::string model_name, 
                     const std::string model_path,
                     const std::string base_model, 
                     const std::string discription);

    /**
     * @description: 
     * @event: 
     * @param {string} model_name
     * @param {string} model_path
     * @return {*}
     */    
    bool UpdateModel(const std::string model_name, 
                     const std::string model_path);

    /**
     * @description: 
     * @event: 
     * @param {string} model_name
     * @return {*}
     */    
    bool DropModel(const std::string model_name);

    /**
     * @description: 
     * @event: 
     * @param {string} model_name 
     * @param {string&} model_path
     * @return {*}
     */    
    bool GetModelPath(const std::string model_name, 
                      std::string& model_path);

    bool GetModelDeviceType(const std::string model_name,
                            torch::DeviceType& device_type);

    bool GetBaseModelPathFromBaseModel(const std::string base_model_name, 
                          std::string& base_model_path);

    bool GetBaseModelPaths(std::vector<std::pair<std::string, std::string>>& base_model_paths);

    bool GetBaseModelPathFromModel(const std::string model_name, 
                          std::string& base_model_path);

    bool HaveBaseModel(const std::string model_name);

    bool IsBaseModelExist(const std::string base_model_name);
    /**
     * @description: 
     * @event: 
     * @param {string} model_path
     * @param {string} model_name
     * @param {string&} md5 
     * @return {*}
     */    
    bool GetModelMd5(const std::string model_path, 
                     const std::string model_name,
                     std::string& md5);

    /**
     * @description: 
     * @return {*}
     */    
    bool GetLastModelVersion(const std::string model_name, 
                             int16& version);

    /**
     * @description: 
     * @event: 
     * @param {string&} model_path
     * @return {*}
     */    
    bool SetCuda(const std::string& model_path, const std::string& cuda);

    /**
     * @description: 
     * @event: 
     * @param {string&} model_path
     * @return {*}
     */    
    bool IsCuda(const std::string& model_path);

    /**
     * @description: input preprocessing function
     * @event: 
     * @param {string} model_path
     * @param {Tensor&} img_tensor processed tensor
     * @param {Args*} args variable arguments
     * @return {*}
     */    
    bool PreProcess(const std::string model_path, 
                    std::vector<torch::jit::IValue>& img_tensor, 
                    Args* args);

    /**
     * @description: output processing function; the result is a float8 stored in result
     * @event: 
     * @param {string} model_path
     * @param {Tensor&} output_tensor
     * @param {float8&} result the returned processing result
     * @return {*}
     */    
    bool OutputProcessFloat(const std::string model_path, 
                            torch::jit::IValue& output_tensor, 
                            Args* args,
                            float8& result);

    /**
     * @description: output processing function; the result is text stored in result
     * @event: 
     * @param {string} model_path
     * @param {Tensor&} output_tensor
     * @param {text*} result
     * @return {*}
     */    
    bool OutputProcessText(const std::string model_path, 
                           torch::jit::IValue& output_tensor, 
                           Args* args,
                           std::string& result);
    
    /**
     * @description: register an input processing callback function
     * @event: 
     * @return {*}
     */
    void RegisterPreProcess(const std::string& model_name, 
                            PreProcessCallback func);

    /**
     * @description: register an output processing callback function whose return value is float
     * @event: 
     * @param {string&} path
     * @param {OutputProcessFloatCallback} func
     * @return {*}
     */    
    void RegisterOutoutProcessFloat(const std::string& model_name, 
                                    OutputProcessFloatCallback func);

    /**
     * @description: register an output processing callback function whose return value is text
     * @event: 
     * @param {string&} path
     * @param {OutputProcessTextCallback} func
     * @return {*}
     */    
    void RegisterOutoutProcessText(const std::string& model_name, 
                                   OutputProcessTextCallback func);

    bool Predict(const std::string& model_path,
                 std::vector<torch::jit::IValue>& input, 
                 torch::jit::IValue& output);

    // New method to analyze model without performing inference
    bool AnalyzeModel(const std::string& model_name, 
                      const std::string& model_path,
                      long long& mac_count,
                      long long& param_count,
                      size_t& param_size_bytes);

    bool InitBaseModel();
    void InvalidateModelCache(const std::string& model_name, const std::string& model_path = "");
private:
    ModelManager(const ModelManager &other);  
    ModelManager & operator=(const ModelManager &other);
private:
    std::unordered_map<std::string, torch::jit::script::Module>                                      base_module_handle_;  //key is the path, value is the module handle
    std::unordered_map<std::string, std::pair<torch::jit::script::Module, torch::DeviceType>>        module_handle_;  //key is the path, value is the module handle and whether to use gpu
    std::unordered_map<std::string, std::string>                                                      model_path_cache_; // key=model_name, value=model_path
    std::unordered_map<std::string, std::string>                                                      model_md5_cache_; // key=model_path + '\n' + model_name, value=md5
    std::unordered_map<std::string, PreProcessCallback>                                              module_preprocess_functions_; //key is the model path, value is the registered preprocessing callback function
    std::unordered_map<std::string, OutputProcessFloatCallback>                                      module_outputprocess_functions_float_; //key is the model path, value is the output processing callback function
    std::unordered_map<std::string, OutputProcessTextCallback>                                       module_outputprocess_functions_text_; //key is the model path, value is the output processing callback function
};


#endif
