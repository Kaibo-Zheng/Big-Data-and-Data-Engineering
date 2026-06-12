# CCKS2026 赛题要求

本文件整理 CCKS2026 天池“任务六：大模型生成文本检测及溯源”的核心要求，用于项目实现、报告撰写和结果核对。

## 基本信息

- 赛题名称：CCKS2026-任务六：大模型生成文本检测及溯源
- 赛题链接：<https://tianchi.aliyun.com/competition/entrance/532474/>
- 英文简介：Joint evaluation task on LLM-generated text detection and attribution
- 比赛 ID：`532474`

## 任务目标

系统需要对输入文本进行两级判断。第一步是文本来源检测，输出三分类标签：

- `label=0`：纯人类撰写
- `label=1`：纯机器生成
- `label=2`：人机协作产生

当样本预测为 `label=1` 时，还需要进一步识别模型家族来源：

- `family=0`：OpenAI / GPT
- `family=1`：Alibaba / Qwen
- `family=2`：DeepSeek
- `family=3`：ByteDance / Doubao
- `family=4`：Moonshot / Kimi
- `family=5`：Google / Gemini
- `family=6`：Anthropic / Claude
- `family=7`：xAI / Grok

训练集中的 `model` 和 `method` 字段可用于分析和特征工程，但正式提交不需要输出具体模型名或协作方式。

## 数据要求

官方数据包含训练集、测试集 A 和测试集 B，均为 JSONL 格式，每行一个 JSON 对象。

| 数据集 | 文件名 | 纯人类撰写 | 纯机器生成 | 人机协作产生 | 发布时间 |
| --- | --- | ---: | ---: | ---: | --- |
| 训练集 | `train.jsonl` | 32,000 | 32,000 | 34,800 | 5 月 15 日 |
| 测试集 A | `test_a.jsonl` | 4,000 | 4,000 | 4,350 | 5 月 15 日 |
| 测试集 B | `test_b.jsonl` | 4,000 | 4,000 | 4,350 | 7 月 3 日 |

本仓库当前使用 `data/train.jsonl` 和 `data/test_a_release.jsonl`。本地统计结果为训练集 98,800 行，测试集 A 12,350 行。

## 人机协作方式

`label=2` 的训练样本可能包含 `method` 字段，用于标识协作方式：

- `machine-modify-human`：机器对人类文本进行润色或改写
- `machine-continue-human`：机器基于人类文本开头继续写作
- `human-mix-machine`：人类文本与机器文本混合

这些字段不是最终提交字段，但可用于数据分析和错误分析。

## 评价指标

官方最终分数为检测任务与溯源任务的加权 Macro-F1：

```text
FinalScore = 0.8 * F1_detect + 0.2 * F1_source
```

- `F1_detect`：在全部测试样本上，对 `label={0,1,2}` 计算三分类 Macro-F1。
- `F1_source`：只在真实 `label=1` 的测试样本子集上，对 `family={0..7}` 计算 Macro-F1。

该规则说明检测任务是主要得分来源，但模型家族溯源也会影响最终成绩。

## 提交格式

提交文件命名为 `submit.jsonl`，每行包含 `id`、`label`、`family` 三个字段：

```json
{"id": 1, "label": 2, "family": -1}
{"id": 2, "label": 1, "family": 3}
```

提交规则：

- 预测 `label=0` 或 `label=2` 时，`family=-1`。
- 预测 `label=1` 时，`family` 必须属于 `0..7`。

A 榜阶段每天可提交 3 次，B 榜阶段每天可提交 5 次，排行榜每小时整点更新。

## 资格审查材料

B 榜前 10 名队伍需要额外提交资格审查材料，包括：

- 结果文件：`submit.jsonl`
- 系统代码：应能正确运行，并复现提交结果
- 方法及系统描述文档：PDF 或 Markdown，包含算法描述和运行配置

如果使用额外公开数据资源，需要在方法文档中详细说明。
