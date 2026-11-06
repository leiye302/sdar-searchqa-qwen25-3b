# Search-QA SDAR — Qwen2.5-3B-Instruct，四卡 A100 80GB

Standalone reproduction of the [official SDAR Search-QA launcher](https://github.com/ZJU-REAL/SDAR/blob/29dcb12a55d8fc17ba6ae03fd0a69048f43051f3/examples/sdar_trainer/run_search_3b.sh).
Source is pinned to `29dcb12a55d8fc17ba6ae03fd0a69048f43051f3`.
This repository runs **Qwen2.5-3B-Instruct** from its original public weights.
The separate 1.5B repository uses the same recipe, QA files and evaluation.

Built with Qwen. Model weights keep their original license; see THIRD_PARTY.md.

**本仓库从公开基模运行原版 SDAR，不是 RETHINK 方法。**
沿用公开 `main_sdar`、SkillSD Trainer、GRPO policy loss、Teacher 评分和 SDAR
蒸馏实现。保留官方 Search-QA 的 **β＝5**、蒸馏系数 0.01、reference-KL 0.001
与 `skill_all=false`。仅适配独立运行、固定评测和断点管理。

**与现有对照的关系：**官方 GRPO＋OPSD 启动脚本使用 β＝0；两份官方脚本的训练
参数仅此处不同。先前发布的 [GRPO＋OPSD beta5 同尺寸仓库](https://github.com/leiye302/grpo-opsd-searchqa-qwen25-3b)
已使用 β＝5，因此与本仓库是相同算法和训练配方，不能在论文中当作两种独立方法。
源码对照和配置等价性由 `configs/native_comparison.json` 与 CPU 测试锁定。

Teacher 是**当前 Actor 同权重的无梯度评分视角**，不是固定初始模型；Teacher
额外看到 SDAR 公开 `skills/search` 静态 skill，并对 Student 已采样的相同 token
评分，不重新生成答案、不替换动作。Student 不看到 skill。**没有 Route-SFT、
RETHINK、Analyzer、局部 hint 或外部模型 API。** 搜索检索 HTTP 服务是必要环境，
并非付费 LLM API；QA 答案仅供原生 verifier 使用，不拼到 Teacher/Student 输入。

```math
\Delta_t=\log\pi_{\mathrm{teacher}}(y_t)-\log\pi_\theta(y_t),\qquad
g_t=\operatorname{stopgrad}(\sigma(5\Delta_t)),\qquad
\mathcal L_{\mathrm{OPSD}}=\operatorname{masked\_mean}
  [g_t(\log\pi_{\mathrm{teacher}}(y_t)-\log\pi_\theta(y_t))].
```

这描述的是所固定公开代码中的采样 token 目标，不将其改写为另一种 full-vocab KL。
policy loss、reference KL、entropy 和非法动作罚项全部保留。配方及共享评测校验见
[必要适配与对照锁](docs/UPSTREAM_DIFF.md)。

## 1. Fixed monitoring subset; full benchmark only at the end

The pinned public data is `PeterJinGo/nq_hotpotqa_train`, revision
`b7d80abfee334a7a91cb377544f09180d58b34f6`.

| Split | Questions | Sources |
| --- | ---: | --- |
| Train | 169,615 | NQ 79,168; HotpotQA 90,447 |
| Test | 51,713 | The seven datasets below |

| Fixed test dataset | Questions |
| --- | ---: |
| NQ | 3,610 |
| TriviaQA | 11,313 |
| PopQA | 14,267 |
| HotpotQA | 7,405 |
| 2WikiMultiHopQA | 12,576 |
| MuSiQue | 2,417 |
| Bamboogle | 125 |

`val_batch_size=512` is **batch size, not the task count**. This release uses
one fixed **448-question monitoring subset: 64 from each of the seven sources**,
at update0 and every20 updates through140. Both backbone repositories use the
same IDs and order, committed in `configs/validation_subset.json`; it is not
redrawn each time. The final update150 evaluates the complete51,713-question
benchmark once. Native loaders remain `shuffle=false`, `drop_last=false`.
No WebShop/ALFWorld task set is reused. See `configs/evaluation.json`.

Evaluation remains greedy (`do_sample=false`, temperature0, n1); only the
monitoring task set/cadence changes. Each question allows at most four environment actions, history4,
4096 prompt tokens and512 response tokens/action. Search retrieves top3 Wiki18
passages using the unchanged E5-base-v2/FAISS server. The original last
`<answer>...</answer>` extraction and normalized exact match against accepted
answer aliases determine reward0/1. Normalization removes case, punctuation,
articles and excess whitespace. Labels are verifier-only, not Actor input.

The original logger includes decision-row and batch-averaged metrics; these
are preserved, **not silently redefined as question-level EM**. The read-only
adapter additionally deduplicates trajectory steps by input question, reporting
per-dataset EM, the seven-dataset macro-average and whole-test micro-average.
See [evaluation details and aggregation boundary](docs/EVALUATION.md).

A fresh data audit found10 exact question-string overlaps between the original
train and test splits after case folding and trimming only. This release
reports that fact and preserves the official split; it does not claim complete
semantic train/test disjointness or quietly change the training pool. A strict
deduplicated split would be a separately named experiment. The new monitoring
subset excludes known exact normalized training-question matches and repeated
question text; the original train/full-test files are not filtered.

## 2. Native SDAR training settings

| Setting | Value |
| --- | --- |
| Model | Qwen/Qwen2.5-3B-Instruct, pinned revision in models.json |
| Updates | 150 |
| Global rollout | 128 questions x8 trajectories =1,024 episodes/update |
| Learning rate / warmup | 1e-6 / original ratio0.1 |
| PPO | Global decision-row mini256, micro16/GPU, one epoch, token-mean |
| Reference KL | Original low_var_kl, coefficient0.001 |
| SDAR / gate | Native sampled-token loss, coefficient0.01; beta5, detached gate |
| Teacher skill | Unchanged public skills/search; skill_all=false; Teacher-only |
| Other native terms | Entropy0.001, invalid-action penalty0.01 |
| Topology | Four FSDP ranks, four TP1 rollout replicas |
| vLLM | Memory utilization0.5, chunked-prefill=false, eager=false, free-cache=false |
| Interaction | Four actions, history4, prompt4096, response512, left truncation |
| Evaluation adaptation | Fixed448 at0/20/40/60/80/100/120/140; full51,713 only at150; greedy |
| Storage adaptation | Save20/final150; newest two verified complete checkpoints |

The native Search-QA batch is128x8 episodes/update. Do not silently reduce it
for faster training. Native prompts, search action parser,
environment feedback, reward, GRPO, optimizer, model precision, gradient
accumulation and masks are SHA-locked. Only the main_sdar Trainer import is patched to
add scheduled validation loaders, read-only reporting and verified storage.
[Exact adaptations](docs/UPSTREAM_DIFF.md).

## 3. Training environment on a clean Linux server

Use four allocated A10080GB cards and a CUDA12.8-compatible driver. Recommend
at least32 logical CPUs,256GiB host RAM and350GiB free disk, preferably400–500GiB.
The roughly64.56GB Flat index is retained both as downloaded parts and a merged
file; corpus decompression/Arrow cache, two full checkpoints, a third save peak
and final HF export require extra space. These are estimates, not measured
GPU/disk minima. Do not download these large assets onto a small root disk.

```bash
git clone https://github.com/leiye302/sdar-searchqa-qwen25-3b.git
cd sdar-searchqa-qwen25-3b
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python src/setup.py --framework
python -m pip install -r configs/requirements.txt
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
python -m pip install -e .deps/sdar -c configs/requirements.txt
python -m pip install -e .deps/sdar/agent_system/environments/env_package/search/third_party -c configs/requirements.txt
python -m pip check

cp .env.example .env
# Edit only your allocated GPU/CPU IDs and local paths if needed.
python src/setup.py --download-model 3b
python src/data.py --prepare --verify
python src/setup.py --verify
python scripts/check_release.py
python -m pytest tests -q
bash scripts/train.sh --dry-run
```

FlashAttention needs a compatible wheel or a working CUDA Toolkit compiler.
All sources, model weights and assets come from the public pinned locations
in the repository. No author workstation, private account, dataset or remote
server is required. We do not copy upstream's private proxy, local conda
activation, sample tracking credentials or whole-machine cleanup commands.

## 4. Separate original retrieval environment

The original server requires CUDA for the E5 encoder, even when a CPU FAISS
index is explicitly selected. A CPU-only server is **not** silently substituted.
The default is the unchanged GPU-sharded FP16 Flat index and FP16 E5 encoder.

```bash
conda create -n searchqa-retriever python=3.10 -y
conda activate searchqa-retriever
conda install numpy=1.26.4 faiss-gpu=1.8.0 -c pytorch -c nvidia -y
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r configs/requirements-retriever.txt
python -m pip check

# EXPLICIT large public download; no model training starts here.
python src/retrieval.py --prepare
# Keep this foreground process in a dedicated tmux/scheduler session.
bash scripts/start_retriever.sh --gpus 0,1,2,3
```

The retrieval server may share the four allocated training cards as in the
original launcher, but its index/encoder occupy GPU memory. Account for them
alongside vLLM/optimizer states; do not count the full80GB as free for training.
Select only GPUs you own. The native server binds all interfaces: keep port8000
behind a firewall/private network; do not expose it publicly. This repository
does not terminate another user's server or automatically kill GPU processes.

To share assets between the two backbone repositories, explicitly pass
`--directory /path/to/shared-retrieval` to preparation/serve, and the same
`--retrieval-dir /path/to/shared-retrieval` to train/eval. Similarly, use
`--directory /path/to/shared-qa` for data preparation and `--data-dir` for runs.
Both paths must pass the same pinned receipts; sharing is optional, not a
hidden external dependency. Merged files, decompressed corpus and encoder are
hashed before serving/training; verifying large files takes additional I/O.

## 5. Train, resume, evaluate, export

In another session activate the training environment, while the retriever
continues running:

```bash
source .venv/bin/activate
bash scripts/train.sh
# Only after the old controller/workers have exited:
bash scripts/train.sh --resume
# After the complete final update150 save:
python src/run.py export --size 3b

# OPTIONAL extra monitoring of an exported model; training already ran full test at150.
bash scripts/eval.sh --eval-scope subset --model "$PWD/runs/sdar_3b/hf150" \
  --run-dir "$PWD/runs/sdar_3b_final_eval"
python src/report.py --run-dir "$PWD/runs/sdar_3b"
```

Default training output is `runs/sdar_3b/`. Training metrics retain native
names; `evaluation/update_*/attempt-*/summary.json` contains the independently
defined question-level EM and unchanged native logs. Each report explicitly labels `subset` or `full` and checks all IDs/order against
the corresponding receipt. Partial evaluations cannot masquerade as a full-test result. Attempts remain separate, including repeated validation on
resume. No scores are prefilled.

Saving stages a new complete four-rank model/optimizer/RNG/data checkpoint,
checks archive CRCs and SHA, atomically publishes it, then prunes older same-run
saves. Resume re-scans the disk and keeps only the newest two verified saves.
Corrupt, linked, partial or foreign evidence is preserved, not blindly deleted.
Full hashes/recipe/backbone/data/receipt/world size must match for resume.
The final update150 is saved and SHA-verified **before** its full51,713-question
evaluation. The native post-validation save re-verifies that checkpoint instead
of writing it twice. If the full evaluation is interrupted, the same
`bash scripts/train.sh --resume` restores update150 and retries evaluation only;
it does not repeat training or execute update151. Partial attempts remain separate.
Original environment/retrieval state is not part of the FSDP checkpoint, so
uninterrupted and resumed generation are not promised to be bitwise identical.

## 6. Verification boundary

CPU CI checks native source locks, original GRPO and gated SDAR gradients, exact row
preprocessing, fixed balanced-subset/full-test identities and scheduling, native HTTP tool feedback and EM,
last partial-batch accounting, failure-safe checkpoint storage and entrypoints.
Its HTTP fixture smoke is clearly labeled: it does not test a downloaded
Wiki18/E5/FAISS index, Actor capability or model performance.

Actual A100 CUDA/NCCL/vLLM, full-scale retrieval latency/quality, memory fit,
GPU save/load and150-update results still need real hardware acceptance.
Before a long run, confirm the intended backbone, four FSDP ranks/TP1, real
retrieval, finite loss/grad, nonzero updates, memory after optimizer creation
and next vLLM wake-up, and a complete save/recovery. CPU tests are not GPU proof.

[Third-party sources and licenses](THIRD_PARTY.md).

## 7. Monitoring runtime and score scopes

The448-question specification aims at a roughly10-minute monitoring pass on
four A10080GB cards, **not a measured runtime guarantee**. Real cost depends on
model size, generated lengths, retrieval throughput and warm-up. Every complete
evaluation records `wall_seconds` and questions/second. There is no600-second
cutoff and no automatic resampling/size change after seeing scores.

Native validation still runs to completion on the selected loader. The adapter
adds `eval/subset/...` and `eval/full/...` question-EM/time curves while retaining
all original native metric values. Do not mix448-question and51,713-question
micro-averages into an unlabeled curve; the subset gives every source equal
weight, whereas full-test micro-EM weights larger datasets more.

`src/data.py --prepare --verify` also creates/validates the small monitoring
Parquet beside the unchanged train/test files. Existing complete native data
can be reused without re-downloading/reprocessing it. Only subset-local
`extra_info.index` is rebased; prompts, search input and verifier answers are
copied unchanged. No GPU, external model API or private dataset is needed.

The ordinary training run already performs its final full test. Export/report
commands do not trigger a second full pass. An explicit independent
`scripts/eval.sh --eval-scope full` remains available only for deliberate reruns.
The new cadence/subset manifest is part of resume identity; older full-test-only
run bindings are not silently migrated. Use a new run directory for this recipe.
