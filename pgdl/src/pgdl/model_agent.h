#pragma once
#ifndef _MODEL_AGENT_H_
#define _MODEL_AGENT_H_

/**
 * @file model_agent.h
 * @brief Core header file of the model agent system, defining core components such as memory management, agent nodes, and task management
 * 
 * This file contains the core architecture definitions of the model agent system, including:
 * 1. MemoryManager: memory management class responsible for data caching and loading
 * 2. Various agent node classes: responsible for task processing at different stages
 * 3. Task-related data structures: define task types, task information, etc.
 * 4. Data structures and functions related to model analysis and resource monitoring
 */

#include "env.h"
#include "model_manager.h"
#include "model_selection.h"
#include "batch_interface.h"
#include "myfunc.h"
#include "vector.h"
#include "cpu_load_predictor.h"
#include "gpu_load_predictor.h"

#include <stdbool.h>
#include <string>
#include <vector>
#include <unordered_map>

extern "C" {
#include "catalog/pg_type_d.h"
#include "access/htup.h"
#include "access/tupdesc.h"
#include "utils/builtins.h"
#include "miscadmin.h"

#include "postgres.h"
#include "fmgr.h"
#include "access/htup_details.h" // provides heap_getattr
#include "utils/lsyscache.h"     // provides getTypeOutputInfo
#include "utils/syscache.h"
}

/**
 * @brief Maximum cache size
 */
#define MAX_CACHE_SIZE 100000

#include <chrono>
#include <unordered_map>

/**
 * @class MemoryManager
 * @brief Memory management class, responsible for data caching, loading, and management
 * 
 * This class provides the following functions:
 * 1. Load data rows from database tables
 * 2. Convert tuples into vectors
 * 3. Manage input and output caches
 * 4. Track agent execution time
 */
class MemoryManager {
public:
    /**
     * @brief Constructor
     */
    MemoryManager() {}
    
    /**
     * @brief Destructor
     */
    ~MemoryManager() {}
    
    /**
     * @brief Load one row of data from the specified table
     * @param table_name table name
     * @param row_index row index
     * @return the loaded data row
     */
    static Args* LoadOneRow(const std::string& table_name, size_t row_index);

    /**
     * @brief Load one row of data from the specified table using SPI
     * @param tuple output parameter storing the loaded tuple
     * @param tupdesc output parameter storing the tuple descriptor
     * @param table_name table name
     * @param row_index row index
     */
    static void SPILoadOneRow(HeapTuple& tuple, TupleDesc& tupdesc, const std::string& table_name, size_t row_index);
    
    /**
     * @brief Convert a tuple into a vector
     * @param tuple input tuple
     * @param tupdesc tuple descriptor
     * @param start starting column index
     * @param dim vector dimension
     * @return the converted vector
     */
    static Args* Tuple2Vec(HeapTuple tuple, TupleDesc tupdesc, int start, int dim);
    
    /**
     * @brief Load text data (query and context) from the specified table
     * @param table_name table name
     * @param row_index row index
     * @param query output parameter storing the query text
     * @param context output parameter storing the context text
     * @return whether the load succeeded
     */
    static bool LoadTextData(const std::string& table_name, size_t row_index, std::string& query, std::string& context);

    /**
     * @brief Current function call count
     */
    int current_func_call{-1};
    
    /**
     * @brief Whether this is the last call
     * 0: not the last call
     * 1: the last call
     * 2: not set
     */
    int is_last_call{2};

    /**
     * @brief Input cache data
     */
    float** ins_cache_data;
    
    /**
     * @brief Input vector cache
     */
    MVec** ins_cache;
    
    /**
     * @brief Output cache size
     */
    int out_cache_size{0};
    
    /**
     * @brief Output cache data
     */
    float* out_cache_data;
    
    /**
     * @brief Output string cache data
     */
    char** out_cache_string_data;
    
    /**
     * @brief Output index
     */
    int output_index{0};
    
    /**
     * @brief Output type
     * 0: float8
     * 1: string
     */
    int output_type{0};
    
    /**
     * @brief Input buffer
     */
    Args* ins_buffer{nullptr};

    /**
     * @brief Second input cache data
     */
    float** ins2_cache_data;
    
    /**
     * @brief Second input vector cache
     */
    MVec** ins2_cache;

    /**
     * @brief Sample path list
     */
    std::vector<std::string> sample_path;
    
    /**
     * @brief Execution time tracking
     * key: agent name
     * value: cumulative execution time (microseconds)
     */
    std::unordered_map<std::string, long long> execution_time_map;
    
    /**
     * @brief Execution count tracking
     * key: agent name
     * value: execution count
     */
    std::unordered_map<std::string, int> execution_count_map;
    
    /**
     * @brief Print timing statistics
     */
    void PrintTimingStats();

    /**
     * @brief Whether agent timing statistics are enabled
     */
    bool timing_enabled{true};
};

/**
 * @enum AgentAction
 * @brief Agent execution action enumeration
 */
enum class AgentAction {
    START,     ///< start execution
    SUCCESS,   ///< execution succeeded
    FAILURE,   ///< execution failed
    PERCEPTION, ///< perception stage
    ORCHESTRATION, ///< orchestration stage
    OPTIMIZATION, ///< optimization stage
    EXECUTION, ///< execution stage
    EVALUATION, ///< evaluation stage
    SCHEDULE   ///< scheduling stage
};

/**
 * @enum TaskType
 * @brief Task type enumeration
 */
enum class TaskType {
    IMAGE_CLASSIFICATION, ///< image classification task
    SERIES, ///< series task
    NLP, ///< natural language processing task
    REASONING, ///< reasoning task
    MUSIQUE, ///< multi-hop question answering task (musique dataset)
    REASONING_STEP2, ///< reasoning step 2
    REASONING_STEP3, ///< reasoning step 3
    STEP1, ///< step 1 task
    STEP2, ///< step 2 task
    STEP3 ///< step 3 task
};

/**
 * @struct Task
 * @brief Task structure
 * 
 * Defines the detailed information of a task, including task type, model, device, and data range
 */
typedef struct Task {
    TaskType task_type; ///< task type
    char* model; ///< model name
    char* cuda; ///< device name (cpu or gpu)
    char* table_name; ///< table name

    int64_t input_start_index; ///< input start index
    int64_t input_end_index; ///< input end index
    int64_t output_start_index; ///< output start index
    int64_t output_end_index; ///< output end index
} Task;

/**
 * @struct TaskInfo
 * @brief Task information structure
 * 
 * Defines the basic information of a task, used for task initialization and management
 */
typedef struct TaskInfo {
    TaskType task_type; ///< task type
    char* table_name; ///< table name
    char* model_name; ///< model name
    char* cuda_name; ///< device name
} TaskInfo;

/**
 * @struct VecAggState
 * @brief Vector aggregation state structure
 * 
 * Defines the state information during vector aggregation
 */
typedef struct VecAggState {
    MemoryContext ctx; ///< memory context
    List* ins; ///< input list
    List* outs; ///< output list
    int batch_i; ///< batch index
    int prcsd_batch_n; ///< number of processed batches
    char* model; ///< model name
    char* cuda; ///< device name
    int nxt_csr; ///< next CSR index
    int64_t pre_time; ///< preprocessing time (milliseconds)
    int64_t infer_time; ///< inference time (milliseconds)
    int64_t post_time; ///< post-processing time (milliseconds)
} VecAggState;

/**
 * @struct AgentState
 * @brief Agent state structure
 * 
 * Defines the state information during agent execution
 */
typedef struct AgentState {
    FunctionCallInfo fcinfo; ///< function call information
    VecAggState current_state; ///< current state
    AgentAction last_action; ///< last action
    int current_start_index; ///< current start index
    int current_end_index; ///< current end index
    int current_task_id; ///< current task ID (index into task_list)
    List* task_list; ///< task list
    std::vector<TaskInfo> task_info; ///< task information list
} AgentState;

/**
 * @class BaseAgentNode
 * @brief Agent node base class
 * 
 * Base class for all agent nodes, defining the basic interface
 */
class BaseAgentNode {
public:
    /**
     * @brief Virtual destructor
     */
    virtual ~BaseAgentNode() = default;

    /**
     * @brief Core execution interface
     * @param state agent state
     * @return execution action
     */
    virtual AgentAction Execute(std::shared_ptr<AgentState> state) = 0;
    
    /**
     * @brief Get the node name
     * @return node name
     */
    virtual std::string Name() const = 0;
    
    /**
     * @brief Execution wrapper with timing
     * @param state agent state
     * @return execution action
     */
    AgentAction ExecuteWithTiming(std::shared_ptr<AgentState> state);
};

/**
 * @class PerceptionAgent
 * @brief Perception agent
 * 
 * Responsible for converting natural language input into embedding vectors
 */
class PerceptionAgent : public BaseAgentNode {
public:
    /**
     * @brief Execute the perception task
     * @param state agent state
     * @return execution action
     */
    AgentAction Execute(std::shared_ptr<AgentState> state) override;
    
    /**
     * @brief Load vector data
     * @param input input vector
     */
    void LoadMVecData(MVec* input);

    /**
     * @brief Get the node name
     * @return node name
     */
    std::string Name() const override {
        return "PerceptionAgent";
    }
};

/**
 * @class OrchestrationAgent
 * @brief Orchestration agent
 * 
 * Responsible for model selection and resource management
 */
class OrchestrationAgent : public BaseAgentNode {
public:
    /**
     * @brief Execute the orchestration task
     * @param state agent state
     * @return execution action
     */
    AgentAction Execute(std::shared_ptr<AgentState> state) override;
    
    /**
     * @brief Register the SPI process
     */
    void SPIRegisterProcess();
    
    /**
     * @brief Initialize the task
     * @param state agent state
     * @param window_start_index window start index
     * @param window_end_index window end index
     */
    void TaskInit(std::shared_ptr<AgentState> state, int window_start_index = -1, int window_end_index = -1);
    
    /**
     * @brief Select the model
     * @param state agent state
     * @param task_type task type
     * @return model name
     */
    std::string SelectModel(std::shared_ptr<AgentState> state, TaskType task_type);
    
    /**
     * @brief Initialize the model
     * @param model_name model name
     * @param from_select_model whether it comes from model selection
     * @return whether initialization succeeded
     */
    bool InitModel(const char* model_name, bool from_select_model);

    /**
     * @brief Get the node name
     * @return node name
     */
    std::string Name() const override {
        return "OrchestrationAgent";
    }
};

/**
 * @class OptimizationAgent
 * @brief Optimization agent
 * 
 * Responsible for plan tree optimization and resource scheduling
 */
class OptimizationAgent : public BaseAgentNode {
public:
    /**
     * @brief Execute the optimization task
     * @param state agent state
     * @return execution action
     */
    AgentAction Execute(std::shared_ptr<AgentState> state) override;

    /**
     * @brief Get the node name
     * @return node name
     */
    std::string Name() const override {
        return "OptimizationAgent";
    }
};

/**
 * @class ExecutionAgent
 * @brief Execution agent
 * 
 * Responsible for executing the plan
 */
class ExecutionAgent : public BaseAgentNode {
public:
    /**
     * @brief Execute the task
     * @param state agent state
     * @return execution action
     */
    AgentAction Execute(std::shared_ptr<AgentState> state) override;

    /**
     * @brief Get the node name
     * @return node name
     */
    std::string Name() const override {
        return "ExecutionAgent";
    }
};

/**
 * @class EvaluationAgent
 * @brief Evaluation agent
 * 
 * Responsible for evaluating execution results
 */
class EvaluationAgent : public BaseAgentNode {
public:
    /**
     * @brief Execute the evaluation task
     * @param state agent state
     * @return execution action
     */
    AgentAction Execute(std::shared_ptr<AgentState> state) override;

    /**
     * @brief Get the node name
     * @return node name
     */
    std::string Name() const override {
        return "EvaluationAgent";
    }

private:
    /**
     * @brief Perform STEP2 post-processing for REASONING/MUSIQUE tasks
     *
     * Integrates the logic from the original temp_addition_function so that it runs inside EvaluationAgent,
     * and is therefore counted in the Agent runtime by ExecuteWithTiming.
     */
    void RunReasoningPostProcess(std::shared_ptr<AgentState> state, TaskType task_type);
    void RunMusiquePostProcess(std::shared_ptr<AgentState> state);
};

/**
 * @class ScheduleAgent
 * @brief Schedule agent
 * 
 * Responsible for managing the execution order of all agents
 */
class ScheduleAgent : public BaseAgentNode {
public:
    /**
     * @brief Execute the scheduling task
     * @param state agent state
     * @return execution action
     */
    AgentAction Execute(std::shared_ptr<AgentState> state) override;

    /**
     * @brief Get the node name
     * @return node name
     */
    std::string Name() const override {
        return "ScheduleAgent";
    }
};

/**
 * @struct ModelAnalysisResult
 * @brief Model analysis result structure
 * 
 * Stores the feature information of the model
 */
struct ModelAnalysisResult {
    long long mac_count;      ///< number of multiply-add operations
    long long param_count;    ///< total number of parameters
    size_t param_size_bytes;  ///< total parameter size (bytes)
};

/**
 * @struct CpuLoadData
 * @brief CPU load data structure
 * 
 * Stores CPU load information
 */
struct CpuLoadData {
    double cpu_load; ///< CPU load
    int cpu_cores; ///< number of CPU cores
};

/**
 * @struct GpuLoadData
 * @brief GPU load data structure
 * 
 * Stores GPU load information
 */
struct GpuLoadData {
    double cuda_cores; ///< number of CUDA cores
    double gpu_freq; ///< GPU frequency
    double util; ///< GPU utilization
    double mem_used; ///< used memory
    double mem_total; ///< total memory
};

/**
 * @struct GpuStatus
 * @brief GPU status structure
 * 
 * Stores GPU status information
 */
struct GpuStatus {
    int id; ///< GPU ID
    double util; ///< GPU utilization (0-100)
    double mem_used; ///< used memory (MiB)
    double mem_total; ///< total memory (MiB)
};

/**
 * @brief Analyze model features
 * @param model_name model name
 * @return model analysis result
 */
ModelAnalysisResult AnalyzeModelWithInference(const std::string& model_name);

/**
 * @brief Estimate CPU FLOPS
 * @return CPU FLOPS
 */
double estimate_cpu_flops();

/**
 * @brief Get GPU FLOPS
 * @return GPU FLOPS
 */
double get_gpu_flops();

/**
 * @brief Get CPU memory bandwidth
 * @return CPU memory bandwidth
 */
double get_cpu_mem_bandwidth();

/**
 * @brief Get GPU memory bandwidth
 * @return GPU memory bandwidth
 */
double get_gpu_mem_bandwidth();

/**
 * @brief Get CPU load factor
 * @return CPU load factor
 */
double get_cpu_load_factor();

/**
 * @brief Execute a command and return the output
 * @param cmd command
 * @return command output
 */
std::string exec_command(const char* cmd);

/**
 * @brief Trim a string
 * @param s input string
 * @return the trimmed string
 */
std::string trim(const std::string& s);

/**
 * @brief Get all GPU statuses
 * @return GPU status list
 */
std::vector<GpuStatus> get_all_gpu_status();

/**
 * @brief Collect GPU load data
 * @return GPU load data
 */
GpuLoadData CollectGPULoadData();

/**
 * @brief Collect CPU load data
 * @return CPU load data
 */
CpuLoadData CollectCPULoadData();

/**
 * @brief Collect GPU load and runtime data
 * @param shape0 data shape 0
 * @param shape1 data shape 1
 * @param shape2 data shape 2
 * @param shape3 data shape 3
 * @param execution_runtime_us execution runtime (microseconds)
 * @param analysis_result model analysis result
 * @param gpu_data GPU load data
 */
void CollectGPULoadAndRuntimeData(int shape0, int shape1, int shape2, int shape3, long long execution_runtime_us, ModelAnalysisResult analysis_result, GpuLoadData gpu_data);

/**
 * @brief Collect CPU load and runtime data
 * @param shape0 data shape 0
 * @param shape1 data shape 1
 * @param shape2 data shape 2
 * @param shape3 data shape 3
 * @param execution_runtime_us execution runtime (microseconds)
 * @param analysis_result model analysis result
 * @param cpu_data CPU load data
 */
void CollectCPULoadAndRuntimeData(int shape0, int shape1, int shape2, int shape3, long long execution_runtime_us, ModelAnalysisResult analysis_result, CpuLoadData cpu_data);

/**
 * @brief Temporarily added function for handling additional logic of inference tasks
 * @param state agent state
 */
void temp_addition_function(std::shared_ptr<AgentState> state);

#endif
