# 项目总览：DBAgent 与 Baseline 系统文档

## 一、DBAgent（本工程：`/home/why/dbagent`）

### 系统简介

DBAgent（代码名 `pgdl`）是一个以 Agent 为核心的 **PostgreSQL 扩展**，将深度学习推理直接嵌入数据库引擎内部。系统采用 C++17 实现，依赖 LibTorch、ONNX Runtime、OpenCV、SentencePiece，底层数据库为 PostgreSQL 13.22。

核心理念：通过声明式 SQL 接口隐藏模型调度、硬件选择、批处理等细节，用户只需指定"任务类型"和"数据来源"，无需手动管理窗口大小、设备类型或模型参数。

### 目录结构

```
/home/why/dbagent/pgdl/
├── src/pgdl/          # 核心 C++ 源码
├── sql/               # PostgreSQL 扩展 DDL（版本迁移脚本）
├── build/             # CMake 构建产物（pgdl.so）
├── paper/
│   ├── sections/      # 论文章节 .tex 文件
│   ├── tables/        # 论文表格 .tex 文件
│   └── figures/       # 图表与 benchmark 结果 JSON
├── test/
│   ├── general_tests/ # 各系统端到端 benchmark 脚本
│   ├── scheduling_tests/ # CPU/GPU 负载调度测试
│   └── throughput_tests/ # 并发吞吐测试
└── visualization/     # 结果可视化脚本
```

### 核心源文件（`src/pgdl/`）

| 文件 | 职责 |
|------|------|
| `agent_interface.cpp/h` | PostgreSQL 入口点，全局状态定义，注册 `db_agent_single` / `db_agent_batch` 聚合函数 |
| `batch_interface.cpp/h` | 窗口函数聚合实现（兼容 `predict_batch_float8` 接口） |
| `model_agent.cpp/h` | 核心推理调度：CPU/GPU 决策、批处理执行、Embedding 处理 |
| `model_manager.cpp/h` | 模型生命周期管理，缓存加载/卸载 |
| `model_selection.cpp/h` | 模型选择逻辑 |
| `model_utils.cpp/h` | 模型分析工具（MACs、参数量统计） |
| `cpu_load_predictor.cpp/h` | CPU 负载预测器（调度决策辅助） |
| `gpu_load_predictor.cpp/h` | GPU 负载预测器 |
| `embedding.cpp/h` | 向量 Embedding 处理 |
| `connection.cpp/h` + `spi_connection.cpp/h` | PostgreSQL SPI/连接管理 |
| `vector.cpp/h` | 向量数据类型操作 |
| `interface.cpp` | 底层 C 接口 |
| `md5.cpp/h` | MD5 工具（模型完整性校验） |

### Agent 架构

系统由以下六个 Agent 通过 **AgentState 黑板**协作：

- **PerceptionAgent**：解析输入数据，提取任务语义
- **OrchestrationAgent**：生成执行计划（线性序列或 DAG）
- **OptimizationAgent**：成本模型优化
- **ExecutionAgent**：模型推理执行（张量准备、GPU Kernel 调度、结果回写）
- **EvaluationAgent**：结果验证与格式化
- **ScheduleAgent**（Control Agent 的一部分）：全局资源调度，CPU/GPU 分配，会话状态管理

执行循环：`ScheduleAgent.decide(S)` -> 分发给对应 Agent -> Agent 执行后返回控制权 -> 循环，直到完成。

### SQL 接口

```sql
-- 进程注册（每个连接调用一次，必须）
SELECT register_process();

-- 主聚合接口（返回 float8[]），自动选择模型和硬件
SELECT unnest(db_agent_single('series', sub_table.data))
FROM (SELECT * FROM slice_test LIMIT 1000) AS sub_table;

SELECT unnest(db_agent_single('nlp', sub_table.comment_vec))
FROM (SELECT * FROM nlp_vector_test LIMIT 1000) AS sub_table;

SELECT unnest(db_agent_single('image_classification', sub_table.image_vector))
FROM (SELECT * FROM imagenet_image_vector_table LIMIT 1000) AS sub_table;

SELECT unnest(db_agent_single('reasoning', sub_table.reasoning_vec))
FROM (SELECT * FROM reasoning_vector_test ORDER BY id LIMIT 10) AS sub_table;
```

支持的任务类型：`'series'`、`'nlp'`、`'image_classification'`、`'reasoning'`

### 构建与安装

```bash
cd /home/why/dbagent/pgdl
mkdir -p build && cd build
cmake ..
make -j$(nproc)
sudo make install
```

依赖：CMake ≥ 3.0、PostgreSQL 13.22 开发头文件、LibTorch、OpenCV、ONNX Runtime、SentencePiece。

安装后在 PostgreSQL 中启用：

```sql
CREATE EXTENSION pgdl;
SELECT load_base_model();  -- 可选，加载基础模型
```

数据库连接配置：`host=localhost, port=5432, user=why, password=123456, dbname=postgres`

### Benchmark 入口脚本

| 脚本 | 用途 |
|------|------|
| `test/general_tests/benchmark_all_baselines.py` | 全量 benchmark：6个系统 × 4个数据集 × 5个 n 值 |
| `test/general_tests/test_single.py` | 单线程延迟测试（DBAgent vs MorphingDB） |
| `test/general_tests/benchmark_reasoning.py` | Reasoning 任务公平对比（HotpotQA） |
| `test/throughput_tests/test_throughput_main.py` | 并发吞吐上限测试 |
| `test/scheduling_tests/test_slice_under_load.py` | 负载下调度决策测试 |
| `paper/figures/generate_barcharts.py` | 从 JSON 结果生成论文图表 |

---

## 二、Baseline 系统（`/home/why/baseline`）

论文将 baseline 分为两类：

- **AI-Native Database Systems (AIDB)**：evadb、morphingdb、neurdb
- **Agent-Oriented Systems (AgenticDB)**：gendb、hermes

---

### 2.1 MorphingDB（AIDB）

**类型**：AI-Native 数据库，PostgreSQL 扩展（与 DBAgent 共存于同一 PostgreSQL 实例）

**特点**：高度优化的任务级 DBMS，支持模型管理、执行位置选择和成本优化。采用固定批大小（窗口 = 32）和 float8 实现，需用户手动指定模型名称、硬件目标和窗口约束。

**SQL 接口**：

```sql
-- 系列数据（CPU，batch window=32）
SELECT predict_batch_float8('slice', 'cpu', data)
OVER (ROWS BETWEEN CURRENT ROW AND 31 FOLLOWING)
FROM slice_test LIMIT 1000;

SELECT predict_batch_float8('swarm', 'cpu', data)
OVER (ROWS BETWEEN CURRENT ROW AND 31 FOLLOWING)
FROM swarm_test LIMIT 1000;

SELECT predict_batch_float8('year_predict', 'cpu', data)
OVER (ROWS BETWEEN CURRENT ROW AND 31 FOLLOWING)
FROM year_predict_test LIMIT 1000;

-- NLP 数据（GPU）
SELECT predict_batch_float8('sst2_vec', 'gpu', comment_vec)
OVER (ROWS BETWEEN CURRENT ROW AND 31 FOLLOWING)
FROM nlp_vector_test LIMIT 1000;

-- 图像数据（GPU）
SELECT predict_batch_float8('defect_vec', 'gpu', image_vector)
OVER (ROWS BETWEEN CURRENT ROW AND 31 FOLLOWING)
FROM imagenet_image_vector_table LIMIT 1000;

SELECT predict_batch_float8('googlenet_cifar10', 'gpu', image_vector)
OVER (ROWS BETWEEN CURRENT ROW AND 31 FOLLOWING)
FROM cifar_image_vector_table LIMIT 1000;
```

**启动方式**：MorphingDB 作为 PostgreSQL 扩展已安装在本机实例中，无需单独启动。直接通过 `psycopg2` 或 `psql` 连接执行 SQL 即可：

```bash
psql -h localhost -p 5432 -U why -d postgres
```

**Benchmark 入口**：`test/general_tests/benchmark_all_baselines.py` 中的 `benchmark_morphingdb()` 函数

---

### 2.2 EvaDB（AIDB）

**类型**：AI-Relational 数据库，Python 库

**特点**：提供类 SQL 接口进行模型推理，通过 UDF（用户自定义函数）注册 Python 推理逻辑，支持结构化和非结构化数据，作为端到端推理的标准对比基线。

**启动方式**：

```python
import evadb

# 每次 benchmark 前需要重新初始化
conn = evadb.connect()
cursor = conn.cursor()

# 1. 注册 PostgreSQL 数据源
cursor.query("""
DROP DATABASE IF EXISTS postgres_data;
""").df()
cursor.query("""
CREATE DATABASE postgres_data WITH ENGINE = 'postgresql',
PARAMETERS = {"database": "postgres", "host": "localhost",
              "port": "5432", "user": "why", "password": "123456"};
""").df()

# 2. 注册推理 UDF（以 slice 为例）
cursor.query("DROP FUNCTION IF EXISTS SliceFunction;").df()
cursor.query("""
CREATE FUNCTION SliceFunction
TYPE Classification
IMPL '/home/why/dbagent/pgdl/test/morphingdb_test/series_test/slice_test/evadb_slice.py';
""").df()

# 3. 执行推理查询
result = cursor.query("""
SELECT SliceFunction(value0, value1, ...) 
FROM postgres_data.slice_origin_test LIMIT 1000;
""").df()
```

UDF 实现文件路径（位于 DBAgent 工程内的测试目录）：

| 数据集 | UDF 文件 |
|--------|----------|
| slice | `pgdl/test/morphingdb_test/series_test/slice_test/evadb_slice.py` |
| sst-2 | `pgdl/test/morphingdb_test/text_test/sst2/evadb_sst2.py` |
| imagenet | `pgdl/test/morphingdb_test/image_test/imagenet/evadb_resnet18.py` |
| cifar-10 | `pgdl/test/morphingdb_test/image_test/cifar10/evadb_cifar.py` |

**Benchmark 入口**：`benchmark_all_baselines.py:benchmark_evadb()`

---

### 2.3 NeurDB（AIDB）

**类型**：AI-Native 数据库，Docker 容器部署

**简介**：整合了学习型索引和自适应模型执行的 AI 原生数据库，集成于 PostgreSQL 引擎内部。源码位于 `/home/why/baseline/neurdb/`。

**启动方式**：

```bash
cd /home/why/baseline/neurdb

# GPU 模式（推荐，需要 NVIDIA GPU + CUDA）
bash build.sh --gpu

# CPU 模式
bash build.sh --cpu
```

等待终端输出 `Press CTRL+C to quit` 后，用 PostgreSQL 客户端连接：

```bash
psql -h localhost -p 5432 -U postgres -d neurdb
```

**关于 Benchmark 中的测量方式**：在论文 benchmark 中，NeurDB 的推理采用**纯 Python 推理路径**（与 Hermes 相同），通过 `torch.jit.load` 加载模型后进行批推理，而非走其 SQL 接口。这是为了保证所有 baseline 在相同的数据和模型上对比。具体逻辑见 `benchmark_all_baselines.py:benchmark_neurdb()`。

---

### 2.4 Hermes（AgenticDB）

**类型**：外部 Agent 编排系统，Python CLI 工具

**简介**：将任务分解为迭代推理和工具驱动的执行步骤，代表外部编排范式。安装位置：`/home/why/.local/bin/hermes`。

**启动方式**：

```bash
# 非交互式：直接传入 prompt 执行任务（-z 参数）
hermes -z "Load the model at /home/why/baseline/models/slice.pt using torch.jit.load on CPU. \
Load 1000 rows from /home/why/baseline/data/series/slice/slice_localization_data.csv, \
take columns 1 to -1 as float32 features. \
Run batched inference with batch_size=32 and report output shape and inference time."

# 交互式聊天
hermes chat

# 指定模型
hermes -z "..." -m <model_name>

# 查看帮助
hermes --help
```

**Benchmark 模式说明**：在端到端 benchmark 中，Hermes 测量分为两条路径：

1. **直接 Python 推理**（`benchmark_all_baselines.py:benchmark_hermes()`）：用 `torch.jit.load` 直接运行批推理，测量纯推理延迟。适用于 Series/NLP/Image 任务。

2. **完整 Agent 调用**（`test/general_tests/hermes_agent_measure.py`）：通过 `subprocess` 调用 `hermes -z "..."` CLI，测量含 LLM 编排的完整端到端延迟。适用于会话复用实验和 Reasoning 任务。

**Benchmark 脚本**：

| 脚本 | 用途 |
|------|------|
| `hermes/benchmark.py` | Hermes 自带的 9个数据集推理 benchmark |
| `hermes/musique_benchmark.py` | Hermes Musique reasoning benchmark |
| `test/general_tests/hermes_agent_measure.py` | Agent 端到端计时（含 LLM） |
| `test/general_tests/hermes_session_reuse.py` | 会话复用延迟测试 |
| `test/general_tests/hermes_session_reuse_full.py` | 完整会话复用实验 |

---

### 2.5 GenDB（AgenticDB）

**类型**：LLM 驱动的生成式查询引擎，Node.js + Python

**简介**：通过五个 LLM Agent（Workload Analyzer、Storage/Index Designer、Query Planner、Code Generator、Query Optimizer）为每个查询生成实例优化的执行代码。支持 Claude 和 OpenAI Codex 作为后端模型。源码位于 `/home/why/baseline/GenDB/`。

**启动方式**（单次 benchmark 执行模式，不需要启动服务）：

```bash
# Slice 推理 benchmark（CPU）
python3 /home/why/baseline/GenDB/benchmarks/slice-inference/benchmark_single_run.py \
    /home/why/baseline/GenDB/benchmarks/slice-inference/subsets/slice_data_10000 \
    /home/why/baseline/models/slice.pt \
    /tmp/gendb_output \
    2000   # num_rows

# SST-2 推理 benchmark（GPU）
python3 /home/why/baseline/GenDB/benchmarks/sst2-inference/benchmark_single_run.py \
    /home/why/baseline/GenDB/benchmarks/sst2-inference/subsets/sst2_10000.csv \
    /home/why/baseline/models/traced_albert_vec.pt \
    /tmp/gendb_output \
    2000

# ImageNet 推理 benchmark（GPU）
python3 /home/why/baseline/GenDB/benchmarks/imagenet-inference/benchmark_single_run.py \
    /home/why/baseline/GenDB/benchmarks/imagenet-inference/subsets/imagenet_10000.csv \
    /home/why/baseline/models/resnet18_imagenet.pt \
    /tmp/gendb_output \
    2000

# CIFAR-10 推理 benchmark（GPU）
python3 /home/why/baseline/GenDB/benchmarks/cifar10-inference/benchmark_single_run.py \
    /home/why/baseline/GenDB/benchmarks/cifar10-inference/subsets/cifar10_10000.csv \
    /home/why/baseline/models/googlenet_cifar10.pt \
    /tmp/gendb_output \
    2000
```

**输出格式**：
- Series/NLP：将时间 JSON 打印到 stdout 最后一行，格式 `{"total_ms": ..., ...}`
- Image：将结果写入 `<output_dir>/result.json`，格式 `{"timing_ms": {"total": ...}}`

**完整 GenDB 生成流程**（生成优化代码，一次性代价）：

```bash
cd /home/why/baseline/GenDB
node src/main.mjs --workload slice-inference --provider claude
```

**Benchmark 入口**：`benchmark_all_baselines.py:benchmark_gendb()`

---

## 三、数据集与模型

### 数据目录（`/home/why/baseline/data/`）

| 任务类型 | 数据集 | 路径 |
|----------|--------|------|
| Series | slice | `data/series/slice/slice_localization_data.csv`（384列特征） |
| Series | swarm | `data/series/swarm/Swarm_Behaviour.csv`（2400列特征） |
| Series | year_predict | `data/series/yead_predict/YearPredictionMSD.csv`（90列特征） |
| NLP | sst-2 | PostgreSQL 表 `nlp_vector_test`（向量化后） |
| NLP | imdb | PostgreSQL 表 `imdb_vector_test` |
| NLP | financial | PostgreSQL 表（financial phrasebank） |
| Image | cifar-10 | `data/image/cifar10/`（32x32 PNG） |
| Image | imagenet | `data/image/imagenet/`（可变尺寸 JPEG） |
| Image | stanford_dogs | `data/image/stanford_dogs/` |
| Reasoning | HotpotQA | PostgreSQL 表 `reasoning_vector_test` + `reasoning_vector_step2` |
| Reasoning | musique | PostgreSQL 表（musique 数据集向量化） |

### 模型文件（`/home/why/baseline/models/`）

| 模型文件 | 用途 |
|----------|------|
| `slice.pt` | Slice 系列推理（CPU，LinearRegressor） |
| `swarm.pt` | Swarm 系列推理（CPU） |
| `year_predict.pt` | YearPredict 系列推理（CPU） |
| `traced_albert_vec.pt` | NLP 文本推理（GPU，ALBERT） |
| `sentiment_analysis_model.pt` | Financial/IMDB 情感分析（GPU） |
| `googlenet_cifar10.pt` | CIFAR-10 图像分类（GPU，GoogLeNet） |
| `resnet18_imagenet.pt` | ImageNet 图像分类（GPU，ResNet18） |
| `alexnet_stanford_dogs.pt` | Stanford Dogs 图像分类（GPU，AlexNet） |

### Reasoning 任务额外模型（`pgdl/test/morphingdb_test/models/`）

| 模型 | 用途 |
|------|------|
| `cross_encoder.pt` | 相关性打分（步骤1） |
| `deberta_reader.pt` | 问答推理（步骤2） |
| `reader_decoder.py` | Token ID 解码（步骤3，subprocess 调用） |
| flan-t5-large（ONNX） | 最终答案生成（步骤4） |

---

## 四、实验设置

### 硬件环境

- CPU：40 核 Intel Xeon Silver 4210R @ 2.40 GHz
- GPU：4 × NVIDIA RTX 3090（每张 24 GB 显存）
- 操作系统：Linux

### 基准测试方法

所有系统统一使用：

1. **1次热身运行**（不计入结果）
2. **3次计时运行**，取最小值作为报告延迟
3. **超时限制**：单次查询最长 3600 秒
4. GPU 计时：在 `time.perf_counter()` 前后各调用 `torch.cuda.synchronize()`

### 规模参数

- **Series/NLP/Image 任务**：`n ∈ {1000, 2000, 5000, 10000}`（对应论文表格中的 n₁-n₄）
- **Reasoning 任务**：`n ∈ {1, 5, 10, 20}`（对应论文表格中的 n₁-n₄）
- **会话复用实验**：`q ∈ {1, 5, 10, 50, 100}` 次查询，每次处理 1000 行

### 吞吐量测试

通过二分法搜索最大并发线程数（上限 160），每个线程独立建立 PostgreSQL 连接，并发执行 `n=1000` 的查询。成功条件：所有线程在超时内完成，结果行数符合预期。

---

## 五、Slice类数据集（SERIES 任务类型）CPU vs GPU 实验结论

### 背景

为了为 slice/swarm/year_predict 调度实验选择测试数据集，需要找一个 DBAgent 运行时 **CPU 比 GPU 更快** 的数据集（即应当将任务调度到 CPU）。

### 测试方法

使用 `predict_batch_float8` 直接指定 `'cpu'` 或 `'gpu:0'` 作为 symbol 参数（该接口不经过 OptimizationAgent，直接绑定设备），对三个数据集各 5 轮测试（跳过第1轮 warmup 求平均）。

### 测试结果（2026-06-14）

| 数据集 | 输入维度 | n=500 CPU | n=500 GPU | n=1000 CPU | n=1000 GPU | n=5000 CPU | n=5000 GPU |
|--------|---------|-----------|-----------|------------|------------|------------|------------|
| **year_predict** | 90 | **0.028s** | 0.038s (+26%) | **0.066s** | 0.068s (+3%) | **0.341s** | 0.343s (+1%) |
| slice | 384 | 0.043s | **0.039s** (-10%) | **0.077s** | 0.080s (+4%) | 0.384s | **0.382s** (~0%) |
| swarm | 2400 | **0.106s** | 0.113s (+6%) | **0.106s** | 0.114s (+7%) | — | — |

### 结论

**选定 `year_predict` 作为测试数据集。** 理由：
- 小批次（n=500）时 CPU 快 26%，差距最显著
- 原因：模型轻量（输入维度仅90），GPU 数据迁移开销大于计算节省
- slice 在 n=500 时 GPU 反而更快，不满足要求
- swarm 的 CPU 优势存在但不如 year_predict 稳定

### 注意事项

- `predict_batch_float8` 的 symbol 参数直接绑定设备，**不经过** OptimizationAgent 调度决策
- `db_agent_single` 才走调度逻辑（目前 SERIES 类型强制 `slice_gpu_model_available = false`，即硬性绑定 CPU）
- 如需测试 `db_agent_single` 的 GPU 路径，需修改 `model_agent.cpp` L1467-1470 并重新 `bash compile.sh && bash run.sh`

---

## 六、调度类实验使用规范

### CPU 资源约束

所有 `scheduling_tests/` 下的调度实验必须将资源限制到机器的**前 20 个 CPU（CPU 0–19）**，其余 CPU 对程序不可见。实现方式：

- 采样脚本主进程：`os.sched_setaffinity(0, range(20))`
- Spin 负载 worker：同上，在 worker 函数内也设置
- PostgreSQL 后端进程：`taskset -p -c 0-19 <backend_pid>`

### 负载等级定义（过载模型）

"中负载"和"高负载"基于**过载比**定义，即 spin worker 数量超过可用 CPU 数，触发 OS 线程调度（无法并行完成任务），刻画 CPU 开销远高于 GPU 的情形：

| 负载等级 | Worker 数 | CPU 数 | 过载比 | 说明 |
|----------|-----------|--------|--------|------|
| 低负载   | 20        | 20     | 1:1    | 恰好占满，无过载 |
| 中负载   | 60        | 20     | 3:1    | 每个 CPU 需分时服务 3 个 worker |
| 高负载   | 120       | 20     | 6:1    | 严重过载，任务时间远超无负载 |

### year_predict 自适应调度实验

以 `year_predict` 数据集（输入形状 1×90，SERIES 任务类型）为调度测试集（CPU 比 GPU 快约 26%@n=500）。调度目标：低负载用 CPU，中/高负载切换 GPU。

**训练数据采样**：

```bash
cd /home/why/dbagent/pgdl/test/scheduling_tests
python3 sample_year_predict_cpu_training_data.py   # CPU 不同负载下采样
python3 sample_year_predict_gpu_training_data.py   # GPU 基准采样
```

训练数据表：`year_predict_cpu_load_training_data`、`year_predict_gpu_load_training_data`

模型文件路径：
- CPU 负载预测器：`pgdl/test/morphingdb_test/models/load_predictors/year_predict_cpu_load_model.pt`
- GPU 负载预测器：`pgdl/test/morphingdb_test/models/load_predictors/year_predict_gpu_load_model.pt`

首次运行 `db_agent_single` 时自动从训练表训练并保存模型（由 `model_agent.cpp` 中 `ADD_OPTIMIZATION_LOGIC=true` 触发）。

---

## 七、一键运行完整 Benchmark

```bash
# 1. 运行所有 baseline（约 4-8 小时）
cd /home/why/dbagent/pgdl/test/general_tests
python3 benchmark_all_baselines.py
# 结果保存至 /home/why/dbagent/pgdl/paper/figures/main_result_benchmark_all.json

# 2. 运行旧版（仅 DBAgent / MorphingDB / Python 推理）
cd /home/why/baseline
python3 benchmark_main.py
# 结果保存至 /home/why/dbagent/pgdl/paper/figures/main_result_benchmark.json

# 3. 生成论文图表
cd /home/why/dbagent/pgdl/paper/figures
python3 generate_barcharts.py
```
