#include <thread>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <chrono>
#include <sched.h>
#include "ATen/core/TensorBody.h"


#include "model_manager.h"
#include "model_utils.h"
#include "vector.h"

extern "C" {
#include "postgres.h"

#include "catalog/pg_type_d.h"
#include "executor/nodeAgg.h"
#include "executor/spi.h"
#include "fmgr.h"
#include "nodes/pg_list.h"
#include "utils/builtins.h"
#include "utils/palloc.h"

typedef struct TimeFilter {
    int64_t load_model_time; //ms
    int64_t pre_time;   // ms
    int64_t infer_time; // ms
    int64_t post_time;  // ms
} TimeFilter;

// args are "state, model, cuda, vector_elements.."
#define VECTOR_START_ARG_INDEX 3

static bool debug_print_batch_time;
extern ModelManager model_manager;
extern TimeFilter time_filter;

// Slice sampling flag: set to true to collect training data for slice CPU load predictor
#define SLICE_BATCH_SAMPLE_BUTTON false

// Local struct for model analysis (defined in model_agent.h but we avoid including it due to VecAggState conflict)
struct BatchModelAnalysisResult {
    long long mac_count;
    long long param_count;
    size_t param_size_bytes;
};

typedef struct VecAggState {
    MemoryContext ctx;
    List* ins;
    List* outs;
    int batch_i;
    int prcsd_batch_n;
    char* model;
    char* cuda;
    int nxt_csr;
    int64_t pre_time;   // ms
    int64_t infer_time; // ms
    int64_t post_time;  // ms
} VecAggState;

// batch prefict functions
PG_FUNCTION_INFO_V1(predict_batch_accum);
PG_FUNCTION_INFO_V1(predict_batch_accum_inv);
PG_FUNCTION_INFO_V1(predict_batch_final_float);
PG_FUNCTION_INFO_V1(predict_batch_final_text);
PG_FUNCTION_INFO_V1(enable_print_batch_time);
PG_FUNCTION_INFO_V1(predict_batch_dummy);

VecAggState *
makeVecAggState(FunctionCallInfo fcinfo)
{
    VecAggState *state;
    MemoryContext agg_context;
    MemoryContext old_context;

    if (!AggCheckCallContext(fcinfo, &agg_context))
        elog(ERROR, "aggregate function called in non-aggregate context");

    old_context = MemoryContextSwitchTo(agg_context);

    state = (VecAggState *) palloc0(sizeof(VecAggState));
    state->ctx = agg_context;
    state->model = pstrdup(PG_GETARG_CSTRING(1));
    state->cuda = pstrdup(PG_GETARG_CSTRING(2));

    MemoryContextSwitchTo(old_context);

    return state;
}

Args* 
makeVecFromArgs(FunctionCallInfo fcinfo, int start, int dim) 
{
    int mvec_oid = 0;

    Args* vec = (Args*) palloc0(sizeof(Args) * dim);
    if(!get_mvec_oid(mvec_oid)){
        ereport(ERROR, (errmsg("get mvec oid error!")));
    }

    for (int i = start; i < dim + start; i++) {
        Oid argtype = get_fn_expr_argtype(fcinfo->flinfo, i);
        if(argtype == INT4OID || argtype == INT2OID || argtype == INT8OID){
            int cur_int = PG_GETARG_INT32(i);
            vec[i-start].integer = cur_int;
        }else if(argtype == FLOAT4OID || argtype == FLOAT8OID){
            float8 cur_float = PG_GETARG_FLOAT8(i);
            vec[i-start].floating = cur_float;
        }else if(argtype == TEXTOID){
            char* cur_text = TextDatumGetCString(PG_GETARG_DATUM(i));
            vec[i-start].ptr = cur_text;
        }else if(argtype == CSTRINGOID){
            char* cur_cstring = PG_GETARG_CSTRING(i);
            vec[i-start].ptr = cur_cstring;
        }else if(argtype == NUMERICOID){
            Datum numer = PG_GETARG_DATUM(i);
            float8 num_float = DatumGetFloat8(DirectFunctionCall1(numeric_float8, numer));;
            vec[i-start].floating = num_float;
        }else if(argtype == mvec_oid){
            MVec* cur_mvec = DatumGetMVec(PG_GETARG_DATUM(i));
            vec[i-start].ptr = cur_mvec;
        }else{
            ereport(ERROR, (errmsg("%d type don't support!", get_fn_expr_argtype(fcinfo->flinfo, i))));
        }
    }
    return vec;
}

static bool
wait_and_check_error(std::vector<std::thread> &pool, std::vector<int> &res, int parallel_num)
{
    bool has_error = false;
    for (auto &t : pool)
            t.join();

    for (int i = 0; i < parallel_num; i++)
    {
            if (!res[i])
            {
                has_error = true;
                break;
            }
    }

    pool.clear();
    res.clear();
    return has_error;
}

#define CLEAN_UP_CPP_OBJS() \
input_tensors.clear();      \
batch_inputs_tmp.clear();   \
input_batch_tensor.clear(); \
output.~IValue();    \
outputs.clear()

#define WAIT_AND_CHECK_ERROR(stage)                            \
if (wait_and_check_error(pool, res, prcsd_batch_n))            \
{                                                              \
    CLEAN_UP_CPP_OBJS();                                        \
    ereport(ERROR, (errmsg("meet error in " stage " stage"))); \
}

#define CLOCK_START() auto start = std::chrono::system_clock::now()
#define CLOCK_END(type) time_filter.type##_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count()

static std::vector<torch::jit::IValue>
split_results_one_level(torch::jit::IValue res) {
    std::vector<torch::jit::IValue> rets;
    if (res.isTensor()) {
        auto&& tensors = torch::split(res.toTensor(), 1, 0);
        for (auto&& t: tensors) {
            rets.emplace_back(torch::jit::IValue(std::move(t)));
        }
    } else if (res.isList()) {
        auto&& list = res.toList();
        for (auto&& it: list) {
            rets.emplace_back(std::move(it));
        }
    } else if (res.isTuple()) {
        auto&& eles = res.toTuple()->elements();
        for (auto&& e : eles) {
            rets.emplace_back(torch::jit::IValue(std::move(c10::ivalue::Tuple::create(e))));
        }
    }
    return rets;
}

static std::vector<torch::jit::IValue>
split_results(torch::jit::IValue output) {
    std::vector<torch::jit::IValue> ret;
    if (output.isTensor()) {
        return split_results_one_level(output);
    } else if (output.isTuple()) {
        std::vector<std::vector<torch::jit::IValue>> cols;
        for (auto&& it : output.toTuple()->elements()) {
            cols.emplace_back(split_results_one_level(std::move(it)));
        }
        // col to row
        int row_num = cols[0].size();
        std::vector<torch::jit::IValue> row;
        for (int i = 0; i < row_num; i++) {
            for (int j = 0; j < cols.size(); j++) {
                row.emplace_back(std::move(cols[j][i]));
            }
            ret.emplace_back(torch::jit::IValue(c10::ivalue::Tuple::create(std::move(row))));
        }
        return ret;
    } else if (output.isGenericDict()){
        auto generic_dict = output.toGenericDict();

        for (const auto& item : generic_dict) {
            if (item.value().isTensor()) {
                auto tensor = item.value().toTensor();
                for(int i = 0; i < tensor.size(0); ++i){
                    c10::impl::GenericDict dict = c10::impl::GenericDict(generic_dict.keyType(), generic_dict.valueType());
                    dict.insert(item.key(), tensor[i].unsqueeze(0));
                    ret.emplace_back(dict);
                }
            }
        }
    }
    return ret;
}


void
infer_batch_internal(VecAggState *state, bool ret_float8)
{
    std::string model_path;
    int prcsd_batch_n = state->ins->length;
    long long local_load_model_us = 0;
    long long local_preprocess_us = 0;
    long long local_predict_us = 0;
    long long local_postprocess_us = 0;
    std::vector<Args*> input_args;
    
    if(strlen(state->model) == 0){
        ereport(ERROR, (errmsg("model name is empty!")));
    }

    input_args.reserve(prcsd_batch_n);
    for (ListCell* cell = list_head(state->ins); cell != NULL; cell = lnext(state->ins, cell)) {
        input_args.push_back((Args*)lfirst(cell));
    }

    if ((int)input_args.size() != prcsd_batch_n) {
        ereport(ERROR, (errmsg("input batch size mismatch")));
    }

    if(!model_manager.GetModelPath(state->model, model_path)){
        model_path.clear();
        ereport(ERROR, (errmsg("model not exist, can't get path!")));
    }

    // 1. 加载模型
    {
        CLOCK_START();

        if(!model_manager.LoadModel(state->model, model_path)){
            model_path.clear();
            ereport(ERROR, (errmsg("load model error")));
        }

        CLOCK_END(load_model);
        local_load_model_us =
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now() - start).count();
        if (strcmp(state->model, "deberta_reader") == 0) {
            elog(INFO, "infer_batch_internal[%s]: load_model_time_us=%lld batch=%d",
                 state->model, local_load_model_us, prcsd_batch_n);
        }
    }   

    // 2. 设置gpu模式
    // CPU inference stays on CPU 0, where the spin workers are, so the
    // contention is visible. A GPU call drives the device from CPU 20: the
    // CUDA driver sync in Module::to() takes seconds when the calling thread
    // shares CPU 0 with the busy-wait workers.
    bool use_gpu = state->cuda != nullptr && strstr(state->cuda, "gpu") != nullptr;
    cpu_set_t infer_set;
    CPU_ZERO(&infer_set);
    CPU_SET(use_gpu ? 20 : 0, &infer_set);
    sched_setaffinity(0, sizeof(infer_set), &infer_set);
    auto set_cuda_start = std::chrono::high_resolution_clock::now();
    if(use_gpu && model_manager.SetCuda(model_path, state->cuda)){
        // elog(INFO, "set model %s use gpu success!", model_path.c_str());
        //model_path.clear();
        //ereport(ERROR, (errmsg("failed to set gpu mode!")));
    }
    auto set_cuda_end = std::chrono::high_resolution_clock::now();
    auto set_cuda_us = std::chrono::duration_cast<std::chrono::microseconds>(set_cuda_end - set_cuda_start).count();
    if (strcmp(state->model, "deberta_reader") == 0) {
        elog(INFO, "infer_batch_internal[%s]: set_cuda_time_us=%lld batch=%d",
             state->model, (long long)set_cuda_us, prcsd_batch_n);
    }

    std::vector<std::thread> pool;
    std::vector<int> res(prcsd_batch_n, 0);
    std::vector<std::vector<torch::jit::IValue>> input_tensors(prcsd_batch_n, std::vector<torch::jit::IValue>());
    std::vector<std::vector<at::Tensor>> batch_inputs_tmp;
    std::vector<torch::jit::IValue> input_batch_tensor; // batch of tensors
    torch::jit::IValue output;
    std::vector<torch::jit::IValue> outputs; // batch of tuples<tensor|list|tuple> or tensors
    // 3. 输入预处理
    {
        CLOCK_START();
        try{
            for (int i = 0; i < prcsd_batch_n; i++) {
                res[i] = model_manager.PreProcess(model_path, input_tensors[i], input_args[i]);
            }
        }catch (const std::exception& e) {
            elog(INFO, "error message:%s", e.what());
        }
        CLOCK_END(pre);
        local_preprocess_us =
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now() - start).count();
        if (strcmp(state->model, "deberta_reader") == 0) {
            elog(INFO, "infer_batch_internal[%s]: preprocess_time_us=%lld batch=%d",
                 state->model, local_preprocess_us, prcsd_batch_n);
        }
    }
    // 4. 预测
    {
        CLOCK_START();
        try{
            // concat each inputs
            batch_inputs_tmp.resize(input_tensors[0].size());
            input_batch_tensor.reserve(input_tensors.size());

            for (size_t i = 0; i < input_tensors[0].size(); ++i) {
                std::vector<torch::Tensor> one_dim_vecs;
                one_dim_vecs.reserve(input_tensors.size());
                for (auto& vecs : input_tensors) {
                    one_dim_vecs.push_back(vecs[i].toTensor());
                }
                // 使用 torch::cat 替代 torch::concat
                torch::Tensor cat_result = torch::cat(one_dim_vecs, 0);
                // Pad odd batch sizes to even to avoid crash in cross_encoder JIT
                // The TorchScript model has a bug that crashes with odd batch dimensions
                if (cat_result.size(0) % 2 != 0) {
                    torch::Tensor pad = torch::zeros_like(cat_result[0]).unsqueeze(0);
                    cat_result = torch::cat({cat_result, pad}, 0);
                }
                input_batch_tensor.push_back(cat_result);
            }
                
            // infer model
            if(!model_manager.Predict(model_path, input_batch_tensor, output)) {
                CLEAN_UP_CPP_OBJS();
                ereport(ERROR, (errmsg("%s:predict error!", model_path)));
            }
            // If we padded, trim the extra output row
            if (prcsd_batch_n % 2 != 0 && output.isTensor()) {
                auto out_tensor = output.toTensor();
                if (out_tensor.size(0) > prcsd_batch_n) {
                    output = torch::jit::IValue(out_tensor.slice(0, 0, prcsd_batch_n));
                }
            }
        }catch (const std::exception& e) {
            elog(INFO, "error message:%s", e.what());
        }
        CLOCK_END(infer);
        local_predict_us =
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now() - start).count();
        if (strcmp(state->model, "deberta_reader") == 0) {
            elog(INFO, "infer_batch_internal[%s]: predict_time_us=%lld batch=%d",
                 state->model, local_predict_us, prcsd_batch_n);
        }
    }
    

    // 5. 结果处理
    {
        CLOCK_START();
        try{
            outputs = split_results(output);
            if (outputs.empty()) {
                CLEAN_UP_CPP_OBJS();
                ereport(ERROR, (errmsg("cannot handle the result type from model!")));
            }

            std::vector<Args*> output_args;
            output_args.reserve(prcsd_batch_n);
            for (int i = 0; i < prcsd_batch_n; i++) {
                Args* out_arg = (Args*)palloc0(sizeof(Args));
                output_args.push_back(out_arg);
                state->outs = lappend(state->outs, out_arg);
            }

            for (int i = 0; i < prcsd_batch_n; i++) {
                //pool.emplace_back([&, i](){
                    torch::jit::IValue wrapped_out(outputs[i]);
                    if (ret_float8) {
                        float8& out = output_args[i]->floating;
                        res[i] = model_manager.OutputProcessFloat(model_path, wrapped_out, input_args[i], out);
                    } else {
                        std::string result_str;
                        res[i] = model_manager.OutputProcessText(model_path, wrapped_out, input_args[i], result_str);   
                        output_args[i]->ptr = pstrdup(result_str.c_str());
                    }
                //});
            }
            //WAIT_AND_CHECK_ERROR("postprocess");
        }catch (const std::exception& e) {
            elog(INFO, "error message:%s", e.what());
        }
        CLOCK_END(post);
        local_postprocess_us =
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now() - start).count();
        if (strcmp(state->model, "deberta_reader") == 0) {
            elog(INFO, "infer_batch_internal[%s]: postprocess_time_us=%lld batch=%d",
                 state->model, local_postprocess_us, prcsd_batch_n);
        }
    }

    /* Per-query breakdown for the load-test ablation. 10000 rows is 313
     * batches of 32, so print and reset every 313 batches. */
    {
        static long long acc_load = 0, acc_pre = 0, acc_infer = 0, acc_post = 0;
        static int acc_batches = 0;
        acc_load += local_load_model_us;
        acc_pre += local_preprocess_us;
        acc_infer += local_predict_us;
        acc_post += local_postprocess_us;
        acc_batches++;
        if (acc_batches == 313) {
            elog(INFO, "INFER_BREAKDOWN[%s]: batches=%d load=%lld pre=%lld infer=%lld post=%lld us",
                 state->model, acc_batches, acc_load, acc_pre, acc_infer, acc_post);
            acc_load = acc_pre = acc_infer = acc_post = 0;
            acc_batches = 0;
        }
    }

    /* update infered batch size */
    state->prcsd_batch_n = prcsd_batch_n;
    state->batch_i++;
}


Datum
predict_batch_accum(PG_FUNCTION_ARGS)
{
    VecAggState*    state;
    MemoryContext old_context;
    Args*           vec;

    state = PG_ARGISNULL(0) ? NULL : (VecAggState *) PG_GETARG_POINTER(0);
    
    /* Create the state data on the first call */
    if (state == NULL)
        state = makeVecAggState(fcinfo);

    old_context = MemoryContextSwitchTo(state->ctx);

    vec = makeVecFromArgs(fcinfo, VECTOR_START_ARG_INDEX, PG_NARGS() - VECTOR_START_ARG_INDEX);
    state->ins = lappend(state->ins, vec);

    MemoryContextSwitchTo(old_context);

    PG_RETURN_POINTER(state);
}

Datum
predict_batch_accum_inv(PG_FUNCTION_ARGS) 
{
    VecAggState*    state;
    
    state = (VecAggState *) PG_GETARG_POINTER(0);

    /* finilize has been called */
    Assert((state != NULL) && (state->prcsd_batch_n != 0));

    if (state->nxt_csr == state->prcsd_batch_n) 
    {
        for (int i = 0; i < state->prcsd_batch_n; i++) 
        {
            state->ins = list_delete_first(state->ins);
            state->outs = list_delete_first(state->outs);
        }
        state->nxt_csr -= state->prcsd_batch_n;
        state->prcsd_batch_n -= state->prcsd_batch_n;
    }

    PG_RETURN_POINTER(state);
}

static Args*
fetch_next_from_predicted_batch(PG_FUNCTION_ARGS, bool ret_float8) 
{
    VecAggState*    state = NULL;
    Args*           ret = NULL;
    MemoryContext   old_context;

    state = PG_ARGISNULL(0) ? NULL : (VecAggState *) PG_GETARG_POINTER(0);

    /* If there were no non-null inputs, return NULL */
    if (state == NULL || list_length(state->ins) == 0)
        return NULL;

    /* do batch infer once if need */
    if (state->nxt_csr >= state->prcsd_batch_n)
    {
        // Collect CPU load data before inference (for slice sampling)
        double cpu_load_before = 0.0;
        int cpu_cores_before = 1;
        auto infer_start = std::chrono::high_resolution_clock::now();
        
        if (SLICE_BATCH_SAMPLE_BUTTON && strstr(state->cuda, "cpu") != nullptr) {
            try {
                std::ifstream loadavg("/proc/loadavg");
                std::string line;
                if (std::getline(loadavg, line)) {
                    std::istringstream iss(line);
                    iss >> cpu_load_before;
                }
                std::ifstream nproc("/proc/cpuinfo");
                std::string nline;
                int cnt = 0;
                while (std::getline(nproc, nline)) {
                    if (nline.find("processor") != std::string::npos) cnt++;
                }
                if (cnt > 0) cpu_cores_before = cnt;
            } catch (...) {}
        }
        
        old_context = MemoryContextSwitchTo(state->ctx);
        infer_batch_internal(state, ret_float8);
        auto infer_end = std::chrono::high_resolution_clock::now();
        auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(infer_end - infer_start).count();
        MemoryContextSwitchTo(old_context);
        
        // Collect and store training data (for slice sampling)
        if (SLICE_BATCH_SAMPLE_BUTTON && strstr(state->cuda, "cpu") != nullptr) {
            // Only collect for slice model (shape 1,384)
            if (state->ins && state->ins->length > 0) {
                Args* first_in = (Args*)list_nth(state->ins, 0);
                if (first_in && first_in->ptr) {
                    MVec* mvec = (MVec*)first_in->ptr;
                    int shape_size = GET_MVEC_SHAPE_SIZE(mvec);
                    int shape0 = 0, shape1 = 0, shape2 = 0, shape3 = 0;
                    if (shape_size == 2) {
                        shape0 = GET_MVEC_SHAPE_VAL(mvec, 0);
                        shape1 = GET_MVEC_SHAPE_VAL(mvec, 1);
                    }
                    // Only collect slice data (shape 1,384)
                    if (shape0 == 1 && shape1 == 384) {
                        // Get model analysis
                        BatchModelAnalysisResult analysis = {0, 0, 0};
                        std::string mp;
                        if (model_manager.GetModelPath(state->model, mp)) {
                            model_manager.AnalyzeModel(state->model, mp, analysis.mac_count, analysis.param_count, analysis.param_size_bytes);
                        }
                        
                        // Build INSERT SQL directly and execute via SPI_execute
                        char sql_buf[512];
                        snprintf(sql_buf, sizeof(sql_buf),
                            "INSERT INTO slice_cpu_load_training_data "
                            "(cpu_load, cpu_cores, execution_runtime_us, data_shape_1, data_shape_2, data_shape_3, data_shape_4, "
                            "model_mac_count, model_param_count, model_param_size) "
                            "VALUES (%.3f, %d, %lld, %d, %d, %d, %d, %lld, %lld, %zu)",
                            cpu_load_before, cpu_cores_before, (long long)duration_us,
                            shape0, shape1, shape2, shape3,
                            (long long)analysis.mac_count, (long long)analysis.param_count, analysis.param_size_bytes);
                        
                        if (SPI_connect() == SPI_OK_CONNECT) {
                            int ret = SPI_execute(sql_buf, false, 0);
                            if (ret == SPI_OK_INSERT) {
                                elog(NOTICE, "Slice training data inserted via SPI: load=%.3f, runtime=%lld us", cpu_load_before, (long long)duration_us);
                            } else {
                                elog(WARNING, "SPI_execute failed for slice training data: ret=%d", ret);
                            }
                            SPI_finish();
                        } else {
                            elog(WARNING, "SPI_connect failed for slice training data");
                        }
                    }
                }
            }
        }

        if (debug_print_batch_time)
        {
            int64_t total = state->pre_time + state->infer_time + state->post_time;
            ereport(NOTICE, 
                (errmsg("\nbatch %d:\n"
                        " pre process: %ld ms(%.2f%%)\n"
                        " infer: %ld ms(%.2f%%)\n"
                        " post process: %ld ms(%.2f%%)",
                        state->batch_i, 
                        state->pre_time, (state->pre_time / (float)total) * 100, 
                        state->infer_time, (state->infer_time / (float)total)  * 100, 
                        state->post_time, (state->post_time / (float)total  * 100))));
        }
    }

    /* consume one result */ 
    state = (VecAggState *) PG_GETARG_POINTER(0);
    ret = (Args*)list_nth(state->outs, state->nxt_csr);
    state->nxt_csr++;

    return ret;
}

Datum
predict_batch_final_float(PG_FUNCTION_ARGS)
{
    Args* final_ret = fetch_next_from_predicted_batch(fcinfo, true);

    if (final_ret == NULL)
        PG_RETURN_NULL();

    PG_RETURN_FLOAT8(final_ret->floating);
}

Datum
predict_batch_final_text(PG_FUNCTION_ARGS)
{
    Args* final_ret = fetch_next_from_predicted_batch(fcinfo, false);

    if (final_ret == NULL)
        PG_RETURN_NULL();

    PG_RETURN_TEXT_P(cstring_to_text(static_cast<const char*>(final_ret->ptr)));
}

Datum
enable_print_batch_time(PG_FUNCTION_ARGS)
{   
    debug_print_batch_time = PG_ARGISNULL(0) ? true : PG_GETARG_BOOL(0);
    return (Datum)0;
}

Datum
predict_batch_dummy(PG_FUNCTION_ARGS)
{
    return aggregate_dummy(fcinfo);
}

}
