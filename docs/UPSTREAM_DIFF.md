# Search-QA SDAR：原版来源、最小适配与共同评测

## 固定来源与方法

公开 [ZJU-REAL/SDAR](https://github.com/ZJU-REAL/SDAR)，版本
`29dcb12a55d8fc17ba6ae03fd0a69048f43051f3`；配方直接对应
`examples/sdar_trainer/run_search_3b.sh`，入口 `verl.trainer.main_sdar`。
不从 WebShop/ALFWorld 配方猜测 Search-QA 参数，不重写其训练循环。

沿用 GRPO 原生 advantage/policy loss、response mask、token-mean、optimizer、
precision、gradient accumulation、reference-KL 0.001、entropy 0.001、
invalid-action penalty 0.01、128 questions ×8 rollouts、mini256/micro16、
LR1e-6/warmup0.1、150 updates、四卡 FSDP/TP1/vLLM0.5、prompt4096/
response512/left truncation、history4/max4 actions、top3 Wiki18 检索。

原版 SDAR 的 `gate_beta=5.0`、`sdar_coef=0.01`、`skill_all=false` 均保留。
直接调用原 `compute_sdar_loss`，不换成全词表 KL，不引入 RLSD 或自定义 advantage。
门控为：

```math
g_t=\operatorname{stopgrad}\left[\sigma\left(5\left(
\log\pi_{\theta}^{\mathrm{teacher}}(y_t)
-\log\pi_\theta(y_t)\right)\right)\right].
```

Teacher 是当前 Actor 同权重的 no-grad 评分视角。`skills/search` 中公开的
general skill 与按 data_source 选择的任务 skill 仅加在 Teacher 前缀；
Teacher 仍评分同一批已采样响应，不生成替换答案。其原生左截断逻辑保持不变。
没有独立冻结 Teacher、Analyzer、局部 hint、RETHINK 或 SFT 起点。
Ground-truth target 不添加到 Teacher 输入，保留原 verifier 用途。

## 与官方 GRPO＋OPSD 及已发布 beta5 仓库的关系

固定提交的两个官方 Search-QA 3B launcher 同用 `main_sdar`。除实验标签外，
解析后的参数唯一差异是 `algorithm.sdar.gate_beta`：SDAR 为5，官方 GRPO＋OPSD 为0。
当 beta 为0时，所有有效 token 的门控恒为0.5，蒸馏项并未被关掉。

本项目之前的 GRPO＋OPSD 已配置 beta5，且数据、Teacher、loss、超参数和评测一致，
故同尺寸下与本仓库相同。这里只提供指向官方 SDAR launcher 的独立仓库和来源锁，
不修改旧仓库、不制造方法差异。两份官方脚本均锁定 SHA；
`configs/native_comparison.json` 记录已发布 beta5 版本和等价配置指纹。

## 明确的工程与评测适配

1. 独立公开下载、revision/SHA 锁、相对目录、隔离 Ray、console logger；
   不使用作者本机路径、代理、追踪凭据或全机清理命令。
2. 3B 沿用原基模；独立 1.5B 仓库仅更换同家族 backbone/revision/输出标签。
   两尺寸所有训练与评测设置相同。四卡均为完整 A10080GB；没有宣称40GB适配。
3. **与已有 Search-QA GRPO 两仓库共用逐字节相同的评测/数据/检索文件与逻辑：**
   固定448题（七源各64）于0/20/40/60/80/100/120/140步；最终150步只跑
   完整51,713题。开启第0步评测是相对原脚本 `val_before_train=false` 的明确
   改动；不修改贪心解码、原生归一化EM、搜索预算或训练数据。
4. 共用 GRPO 的只读记录和 loader 切换，保留原生 step/batch 指标，另记题级
   EM/macro/micro 与耗时；两种 scope 分开。共用原 `index` 缺失边界修复：
   先核验题目身份和顺序，再按当前输入批次映射 collector 局部 index。
5. 每20步及最终150步保存；同一原生 FSDP serializer 外加 staging→完整性/SHA
   校验→原子发布→保留最近两份，跨恢复亦生效。最终150在大评测**之前**保存；
   恢复最终断点只补评测，不多训151步，不覆盖已有断点。

上游**只改一行 import**：`main_sdar.py` 选择本仓库的 `SearchTrainer`，
后者继承未改动的 `SkillSDRayTrainer`。其 `fit` 仍调用 `super().fit()`，
validation 仍调用继承的原生 `_validate()`；没有套用纯 GRPO fit 丢掉 Teacher。
`main_ppo.py`、两级原生 Trainer、核心 loss、Actor、worker、环境、采样器、
reward、prompt、parser、检索与六份静态 skill 文件均进行 SHA 校验。
训练设置逐项与官方 SDAR 启动参数比对；未改动 shared evaluation 的代码或题目。

## 共同评测校验与复现边界

`configs/grpo_comparison.json` 记录两个公开 GRPO 版本、共同文件 SHA、
完整 GRPO 实验配置以及允许的具名方法差异；CPU 测试逐项比对。
两个 SearchTrainer 的方法源代码完全相同，仅继承类由 RayPPO 换为 SkillSD。
这保证评测适配没有趁换方法时悄悄更换题目、归一化、聚合或调度。

原始169,615/51,713 train/test 数据中已报告10个规范化题目文本完全匹配，
不擅自重划公开训练集；固定448监测集排除了这些匹配与重复文本。
保留 parser 的原生宽松行为、原生 Teacher 左截断、原生奖励，不额外“优化”。
真实 E5/FAISS 检索、CUDA/NCCL、显存峰值、GPU存取与150步成绩须实机验收；
CPU 测试与 HTTP fixture 不代表这些结果。尚未启动新训练或填入实验分数。
