# 基于 DeBERTa-LoRA 的大模型生成文本检测与溯源系统

本仓库是《大数据与数据工程》课程大作业项目，配套报告见
`report/report.tex`。项目选取 CCKS2026 天池“任务六：大模型生成文本检测及溯源”作为案例，围绕原始 JSONL 数据处理、统计分析、可视化展示、基线模型、DeBERTa-LoRA 多卡训练、推理提交和结果归档，完成一个端到端的数据工程实践流程。

实验硬件为 **4 张 NVIDIA RTX 5090**。最终模型在本地验证集上取得检测 Macro-F1 `0.9922`、溯源 Macro-F1 `0.8797`，按官方公式估计综合分数为 `0.9697`。

## 项目任务

系统对输入文本进行两级判断：

1. 检测任务：区分纯人类撰写、纯机器生成、人机协作文本。
2. 溯源任务：当文本被判定为纯机器生成时，进一步识别 8 个大模型家族。

标签定义如下：

| 字段 | 取值 | 含义 |
| --- | --- | --- |
| `label` | `0` | 纯人类撰写文本 |
| `label` | `1` | 纯机器生成文本 |
| `label` | `2` | 人机协作文本 |
| `family` | `0..7` | OpenAI、Alibaba、DeepSeek、ByteDance、Moonshot、Google、Anthropic、xAI |
| `family` | `-1` | 非纯机器生成文本 |

官方评价指标为：

```text
Score = 0.8 * F1_detect + 0.2 * F1_source
```

提交文件为 JSONL 格式，每行包含 `id`、`label`、`family`：

```json
{"id": 1, "label": 2, "family": -1}
{"id": 2, "label": 1, "family": 3}
```

## 数据集

当前仓库使用训练集和测试集 A：

| 数据集 | 文件 | 样本数 | 文件大小 | 平均词数 |
| --- | --- | ---: | ---: | ---: |
| 训练集 | `data/train.jsonl` | 98,800 | 159.0 MB | 258.6 |
| 测试集 A | `data/test_a_release.jsonl` | 12,350 | 19.4 MB | 258.6 |

训练集中三类检测标签分布为：纯人类文本 32,000 条、纯机器文本 32,000 条、人机协作文本 34,800 条。纯机器文本进一步均匀划分为 8 个模型家族，每个家族 4,000 条。

数据画像由以下脚本生成：

```bash
python tool/profile_dataset.py
```

输出位于：

```text
analysis/csv/
analysis/llm_text_data_profile.md
```

## 方法概述

报告中的方法分为两部分。

首先实现 `TF-IDF + SGD` 线性分类器作为基线，用于快速验证数据读取、标签映射、指标计算和提交格式。该基线在本地验证集上取得检测 Macro-F1 `0.8945`、溯源 Macro-F1 `0.7079`。

主模型采用本地 `microsoft/deberta-v3-large` 作为编码器，并通过 LoRA 对注意力层中的 `query_proj`、`key_proj`、`value_proj` 进行参数高效微调。检测任务和溯源任务分别训练独立 adapter：

```text
ckpt/label/     检测三分类 adapter
ckpt/family/    模型家族八分类 adapter
```

分类头不只使用 CLS 向量，而是拼接四类文本表示：

```text
CLS pooling + Mean pooling + Max pooling + Attention pooling
```

随后经过 LayerNorm、全连接层、GELU 和 Multi-sample Dropout 输出分类结果。这个设计与报告中的“风格感知池化分类头”一致，用于同时捕捉全局语义与局部生成风格线索。

## 目录说明

```text
data/               原始训练集和测试集 A
model/              本地 DeBERTa-v3-large 基座模型
ckpt/               已训练 LoRA adapter 和指标元数据
train/              基线、单进程训练、DDP 训练和共享训练逻辑
eval/               推理入口
tool/               数据画像、结果汇总、logo 素材处理
visualization/      报告图生成脚本与绘图样式
Illustration/       报告插图，当前对应 report.tex 中的图 1-4
result/csv/         指标、训练曲线、预测分布等结果表
result/submissions/ 预测明细与提交文件
report/             课程报告 LaTeX 源文件
docs/               赛题要求和课程要求整理
scripts/            环境安装与训练工作流脚本
utils/              共享路径和标签元信息
```

运行后会生成或更新：

```text
analysis/           数据统计分析结果
logs/               DeBERTa-LoRA 训练日志
visualization/figures/
                    TF-IDF baseline 混淆矩阵图
Illustration/logos/
                    报告图使用的模型家族 logo
```

## 环境配置

推荐使用 Python 3.10。PyTorch 不写入 `requirements.txt`，由 `scripts/setup_env.sh` 按 CUDA wheel 源单独安装。RTX 5090 环境建议使用 CUDA 12.8 或更新版本。

```bash
CONDA_ENV=ccks2026 \
PYTHON_VERSION=3.10 \
TORCH_CUDA=cu128 \
TORCH_VERSION=2.8.0 \
TORCH_INDEX_URL=https://mirrors.aliyun.com/pytorch-wheels/cu128 \
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
bash scripts/setup_env.sh

conda activate ccks2026
```

普通 Python 依赖见 `requirements.txt`，主要包括 `transformers`、`peft`、`scikit-learn`、`pandas`、`matplotlib`、`seaborn`、`Pillow`、`requests`、`wandb` 等。

## 实验复现

### 1. 数据统计分析

```bash
python tool/profile_dataset.py
```

### 2. TF-IDF + SGD 基线

```bash
python train/train_tfidf_sgd_baseline.py
```

主要输出：

```text
result/csv/llm_text_baseline_metrics.csv
result/csv/llm_text_baseline_classification_report.csv
result/csv/llm_text_baseline_confusion_matrix.csv
result/csv/llm_text_baseline_source_report.csv
result/submissions/test_a_tfidf_sgd_predictions.csv
result/submissions/submit.jsonl
```

### 3. DeBERTa-LoRA 单机多卡训练

报告正式实验使用 4 卡 DDP，主要超参数如下：

| 配置项 | 取值 |
| --- | --- |
| 训练轮数 | 20 |
| 最大序列长度 | 384 |
| 最大字符截断 | 8000 |
| 单卡训练批大小 | 16 |
| 梯度累积步数 | 1 |
| 评估批大小 | 64 |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA 学习率 | `8e-5` |
| 分类头学习率 | `8e-4` |

复现命令：

```bash
bash scripts/run_deberta_lora_ddp.sh \
  --nproc-per-node 4 \
  --epochs 20 \
  --batch-size 16 \
  --grad-accum 1 \
  --eval-batch-size 64 \
  --max-length 384 \
  --max-chars 8000 \
  --lr 8e-5 \
  --head-lr 8e-4 \
  --lora-r 16 \
  --lora-alpha 32 \
  --num-workers 4 \
  --no-gradient-checkpointing
```

该脚本依次训练检测 adapter、溯源 adapter，并生成测试集 A 预测结果。

### 4. 只运行推理

已有 `ckpt/label/` 和 `ckpt/family/` 时，可直接生成提交：

```bash
python eval/predict_deberta_lora.py \
  --model-dir model \
  --output-dir ckpt \
  --submission-dir result/submissions \
  --max-length 384 \
  --max-chars 8000 \
  --eval-batch-size 64 \
  --num-workers 4
```

输出：

```text
result/submissions/submit_deberta_lora.jsonl
result/submissions/test_a_deberta_lora_predictions.csv
```

若最终提交要求文件名为 `submit.jsonl`：

```bash
cp result/submissions/submit_deberta_lora.jsonl result/submissions/submit.jsonl
```

## 实验结果

本地验证集结果与报告表格保持一致：

| 方法 | 检测 Macro-F1 | 溯源 Macro-F1 | 加权估计分数 |
| --- | ---: | ---: | ---: |
| TF-IDF + SGD | 0.8945 | 0.7079 | 0.8572 |
| DeBERTa-LoRA 初始版 | 0.9178 | 0.3090 | 0.7961 |
| DeBERTa-LoRA + 强分类头 | 0.9922 | 0.8797 | 0.9697 |

当前最佳 checkpoint：

| adapter | Accuracy | Macro-F1 | 最佳轮次 | 验证样本数 |
| --- | ---: | ---: | ---: | ---: |
| `ckpt/label/` | 0.9921 | 0.9922 | 7 | 9,880 |
| `ckpt/family/` | 0.8788 | 0.8797 | 19 | 3,200 |

测试集 A 预测标签分布：

| 预测标签 | 样本数 |
| --- | ---: |
| human | 3,994 |
| machine | 4,003 |
| hybrid | 4,353 |

## 报告与图表

报告源文件：

```text
report/report.tex
```

报告插图：

```text
Illustration/1.png    项目动机
Illustration/2.png    数据集分析
Illustration/3.png    模型结构
Illustration/4.png    实验结果汇总
```

重新生成代码绘制的报告图：

```bash
python visualization/plot_all.py
```

该命令会先汇总结果指标，再生成数据分析图与实验结果图。若缺少模型家族 logo，可先运行：

```bash
python tool/fetch_logo_assets.py
```

注意：`visualization/paper_style.py` 当前会从 `F:\Skills\scientific-paper-figure\scripts` 导入 `figure_primitives`。这只影响报告图重生成，不影响训练、推理和提交文件生成。

## 提交材料

课程归档时建议包含：

```text
train/ eval/ tool/ visualization/ scripts/ utils/
data/ 或数据说明
model/ 或基座模型下载说明
ckpt/
result/
Illustration/
report/report.tex 与编译后的报告 PDF
README.md
requirements.txt
```

最终竞赛提交文件位于：

```text
result/submissions/submit.jsonl
```
