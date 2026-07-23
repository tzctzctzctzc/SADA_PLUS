# Stage 1 代码实现与复现说明

## 1. 实现依据和版本边界

本实现以工作区 `stage1_final_plan.md` 为唯一方案依据，基于 SADA_PLUS
提交 `c676ba4` 开发，并在首版 Stage 1 提交 `8dfd0c6` 上完成代码审查修订。
Stage 1 只替换 target conditional-alignment 的
routing mask，不修改 source supervised classification / localization loss，
也不修改原 SADA conditional loss 的 source/target 独立计数和平均口径。

Stage 1 默认关闭。运行时不传 `--stage1`，代码走原始 SADA 路径：

```bash
python train_da.py -c CONFIG -n baseline_run --stage1 off
```

两套首轮实验分别运行：

```bash
CUDA_VISIBLE_DEVICES=0 python train_da.py -c CONFIG -n stage1_shift \
  --stage1 shift --stage1-seed 100003

CUDA_VISIBLE_DEVICES=1 python train_da.py -c CONFIG -n stage1_context \
  --stage1 context --stage1-seed 100003
```

两个命令应使用相同的 `CONFIG`、训练 seed、数据 split 和训练预算。Stage 1
扰动使用独立 RNG，不消费 DataLoader 和模型的随机数流。正式训练、source
校准和 target dry run 使用三套彼此独立的 CPU generator。

## 2. 文件改动总览

### `libs/models/SADA/stage1_utils.py`

新增无状态、可单元测试的基础算法：

- `validate_stage1_resume`：在加载 checkpoint 状态前校验 Stage 1 开关、
  扰动模式、`alpha` 和校准状态；
- `population_std`：使用 `correction=0` 的总体标准差；
- `shift_features`：显式 padding 的时间平移，不允许首尾回卷；
- `context_view_on_cpu`：从完整特征切出 Context 视图，并保证进入 batch
  padding/concatenation 前位于 CPU；
- `interpolate_probabilities`：按每层 `points[:, 0]` 做连续时间线性插值，
  查询恰好落在 anchor 时只检查该 anchor；
- `source_action_eligible`：按 multi-hot GT 判断 pseudo class 是否属于任一
  真实正类；
- `erode_mask`：用局部 kernel 传播 strict local-support mask；
- `build_routing_masks`：生成 ReliableAction、StableBackground、Ignore；
- `weighted_quantile`：source-only 层平衡校准使用的加权分位数。

### `libs/models/SADA/stage1_calibration.py`

新增 source-only 阈值校准：

- `layer_balanced_weights`：每层样本总权重相同；
- `calibrate_thresholds`：计算 `beta/tau_act/tau_bg`；
- `save_calibration_artifact`：保存阈值、manifest 与 SHA256；
- `load_calibration_artifact`：核验 manifest hash 和实验元数据后读取阈值；
- `model_state_sha256`：标识校准时使用的精确模型参数。

### `libs/models/SADA/meta_archs.py`

新增模型级 Stage 1 状态和 routing：

- Stage 1 总开关、扰动类型、三套独立 RNG、冻结阈值；
- `_stage1_strict_masks`；
- `_stage1_shift_routing`；
- `_stage1_context_routing`；
- `collect_stage1_calibration_records`；
- `stage1_route_only`；
- `losses(..., stage1_routes=None)` 中仅替换 target mask。

### `libs/datasets/data_utils.py`

`truncate_feats` 额外记录：

```text
metadata.original_feat_len
metadata.crop_start
```

`crop_start` 是 baseline 实际裁剪在完整特征时间轴中的绝对起点。
函数新增可选 `rng` 参数；不传时仍使用原来的全局 `random`，因此 baseline
行为不变；source 校准传入固定 `random.Random`，从而复用完全相同的
`crop_ratio/trunc_thresh/action-preserving/segment clipping/label filtering`
逻辑。

### `libs/datasets/epic_kitchens.py`

新增可选参数 `stage1_context`。仅 Context 实验保存 `full_feats`，用于从完整
视频特征中取得真实第二窗口；Shift 和 baseline 不保存，避免额外显存开销。

### `libs/trainer/base_trainer_da.py`

- 在 `E_cal = warmup_adv`、第一次 conditional update 之前触发校准；
- checkpoint 保存和恢复冻结阈值及三套 perturbation RNG state；
- epoch 结束写入 JSONL、CSV 和 TensorBoard。

### `libs/trainer/da_trainer.py`

- `_calibrate_stage1`：source validation 固定 manifest 校准；
- `_stage1_dry_run`：不读取 target segments/labels 的完整 target 扫描；
- 累加每 epoch 的实际筛选数量；
- 对显式指定的旧校准工件执行配置、代码、split 和模型状态兼容性检查。

### `train_da.py`

新增：

```text
--stage1 {off,shift,context}
--stage1-seed INTEGER
```

并只在 Context + Epic-Kitchens 时为 dataset 打开完整特征返回。
训练期 evaluator 同样接收完整 split list，不再只评估 `split[0]`。

### `tests/test_stage1_utils.py`

覆盖平移、连续/精确插值、互斥路由、多标签 source 判定、真实裁剪复用、
校准工件校验、RNG 恢复、加权分位数和层平衡校准。

## 3. Stage 1 生命周期

### 3.1 Warm-up

从随机初始化开始训练。在 `curr_epoch < warmup_adv` 时：

- 不构造扰动视图；
- 不调用 Stage 1 routing；
- 不消费 `stage1_train_rng`；
- 训练行为与原 SADA 一致。

### 3.2 固定时刻校准

在 `epoch == max(1, warmup_adv)` 且第一次 adversarial update 之前：

1. 临时将模型切为 eval；
2. 用固定 `calibration_seed` 和训练所用的 `truncate_feats` 为 source
   validation 生成 baseline crop；
3. 对 Shift 或 Context 生成双视图；
4. 使用 SADA 自己的 `label_points` 生成 source anchor GT；
5. 收集每层 detached calibration records；
6. 计算 `beta/tau_act/tau_bg`；
7. 保存 `stage1_calibration.json`；
8. 将阈值写回模型并冻结；
9. 执行无 target-GT dry run；
10. 恢复原训练状态并继续当前 epoch。

校准集合为空、无合法 `beta`、action 集合为空或 background 集合为空时直接
抛出异常停止训练，不使用默认常数兜底。

### 3.3 在线 routing

每个 DA iteration：

1. 原始 source/target 有梯度主前向保持不变；
2. Stage 1 使用当前 target 网络权重分别对原视图、扰动视图做两次独立的
   deterministic routing forward；
3. 两次 routing forward 都只运行 target backbone、neck 和
   classification head；
4. 这些模块临时 `eval()` 且整体位于 `torch.no_grad()`；
5. routing 完成后精确恢复各模块原 train/eval 状态；
6. 所有类别、分数、标准差和 mask 都不连接梯度；
7. 仅原始 target feature 进入 conditional discriminator。

## 4. Shift 代码映射

### 4.1 双视图

每 iteration 从独立 CPU generator 均衡选择 `Delta=-2` 或 `Delta=+2`。

`shift_features` 使用切片赋值：

```python
shifted[..., delta:] = features[..., :-delta]  # delta > 0
```

空缺位置保持零，mask 为 False。代码没有使用 `torch.roll`。

### 4.2 连续坐标插值

对原 anchor 坐标：

```python
query = point_grid[:, 0] + delta
```

`interpolate_probabilities` 使用 `torch.searchsorted` 找到左右端点，并对完整
类别 probability vector 线性插值。查询不在网格点上时，只有查询位于坐标
范围内且左右端点都 strict-valid 才有效；查询恰好等于某个网格点时，直接取
该 anchor 的完整 probability vector，并且只要求该 anchor strict-valid。
没有 `anchor_index + Delta` 或最近邻分支。

## 5. Context 代码映射

### 5.1 保持 baseline crop

`truncate_feats` 生成的 `feats` 仍是主训练输入；代码额外保留：

```text
full_feats
metadata.crop_start
```

原始 crop 没有被重新采样。第二窗口直接从 `full_feats` 切片，不会用 padding
伪造上下文。

### 5.2 第二窗口

对有效长度 `W`：

```python
offset = max(1, round((64 / 2304) * W))
```

分别检查左移和右移是否合法；两个方向都合法时使用独立 RNG 均匀选择。若没有
合法方向或 `offset/W > rho_max`，该视频全部 target anchor ignore，但不会
重采样另一视频。

Context 第二视图在进入 `preprocessing` 前统一保持在 CPU：

```python
if context_valid:
    context_start = crop_start + direction * offset
else:
    context_start = crop_start

context = context_view_on_cpu(
    original, full_feats, context_start, length
)
```

有效视频从 CPU `full_feats` 切出偏移窗口；无效视频从同一个 `full_feats`
重建 baseline crop，仅在 `full_feats` 缺失时才将 `original` 转回 CPU。
因此同一 batch 混合“有合法第二窗口”和“无合法第二窗口”的视频时，
`preprocessing` 的 `F.pad` 与 `torch.cat` 输入仍在同一设备，完成拼接后再
整批迁移到 GPU。无效视频的 routing mask 仍全部为 ignore。

### 5.3 绝对时间对应

若方向为 `d`，扰动窗口中的查询坐标为：

```python
query = point_grid[:, 0] - d * offset
```

查询同样通过连续概率插值和 strict-valid 端点检查。

## 6. Strict local-support mask

`_stage1_strict_masks` 从真实输入 mask 开始：

1. 按 embedding conv 的 kernel 连续腐蚀；
2. 按 SGP stem 的最大局部 depthwise kernel 腐蚀；
3. 每层先按真实 pooling kernel/stride 下采样；
4. 再按该 SGP branch 的最大局部 kernel 腐蚀；
5. 按 classification head 的全部卷积继续腐蚀。

SGP 的 global aggregation 不用于扩大 strict-mask 边界，因为不同真实窗口的
全局上下文本来就是 Context 扰动的信号。strict mask 只排除局部卷积、池化和
插值接触人工 padding 的输出。

## 7. Routing 规则

`build_routing_masks` 先计算：

```python
score0, cls0 = original_prob.max(-1)
score1, cls1 = perturbed_prob.max(-1)
action_std = std([p0(cls0), p1(cls0)], correction=0)
background_std = std([score0, score1], correction=0)
```

ReliableAction：

```text
strict-valid
AND score0 >= alpha
AND cls0 == cls1
AND action_std <= tau_act
```

StableBackground：

```text
strict-valid
AND score0 <= beta
AND score1 <= beta
AND background_std <= tau_bg
```

其余全部 Ignore。代码显式保证三个 mask 互斥且覆盖全部 anchor。

在 `losses` 中：

- ReliableAction 根据 deterministic routing 的 `pseudo_class` 写 one-hot；
- StableBackground 单独作为 background branch mask；
- Ignore 不进入任何 target conditional branch；
- Stage 1 关闭或阈值未校准时继续使用原 SADA 单阈值逻辑；
- source routing、counter、权重和跨层归一化没有修改。

## 8. 阈值校准

### 8.1 Layer balance

每个候选集合单独计算：

```python
weight_i = 1 / N_level
```

因此每个非空层总权重为 1，不会由 Level 0/1 的 anchor 数量主导。

### 8.2 Beta

对 strict-valid source anchor：

```python
r = max(score_original, score_perturbed)
```

遍历所有唯一候选分数，取满足 layer-balanced action contamination 不超过
`epsilon_bg=0.01` 的最大值。

校准工件同时保存：

- balanced contamination；
- pooled contamination；
- action leakage；
- background retention；
- selected count。

### 8.3 Tau

`tau_act` 的集合要求 source GT action、原 pseudo class 属于该 anchor 的
任一 multi-hot GT 正类、原分数过 alpha、扰动预测保持同一个 pseudo class
且 strict-valid；取 layer-balanced weighted Q90。代码不使用
`labels.argmax()`，因此重叠动作 anchor 不会被错误压成单标签。

`tau_bg` 的集合要求 source GT background、双视图最大动作分数均不超过 beta
且 strict-valid；取 layer-balanced weighted Q90。

## 9. Dry run

校准后，Trainer 通过 `stage1_route_only` 扫描 target train。这个接口只读取：

```text
feats / full_feats / metadata
```

不会读取 target `segments` 或 `labels`。

输出包括 valid、action candidate、class-consistent、reliable action、
background candidate、all-view-low、stable background、ignore、batch 数和
全空 batch 数。额外输出 reliable action 的逐类别计数；Context 还输出有效
视频数、视频覆盖率和有效重叠 anchor 数。Dry run 使用独立的
`stage1_dryrun_rng`，不会推进正式训练使用的 `stage1_train_rng`，也不会据此
修改阈值。

## 10. 日志和产物

默认输出到本次运行的 `model_results/<name>/<run_id>/`：

```text
saved_ckpt/stage1_calibration.json
stage1_epoch_metrics.jsonl
stage1_epoch_metrics.csv
stage1_tensorboard/
```

epoch 指标包含逐层：

```text
valid
action_candidate
class_consistent
reliable action
background_candidate
all_view_low
stable background
ignore
action/background std
```

校准日志和 `stage1_calibration.json` 还包含用于判断阈值是否失真的：

```text
beta / tau_act / tau_bg
balanced_contamination / pooled_contamination
action_leakage / background_retention
selected_count / eligible_action
```

带 `_total` 的字段是整 epoch 实际累计数量；不带 `_total` 的字段沿用原
MetricTracker，表示 iteration 平均值。epoch 结束还会由累计量计算
`stage1_anchor_valid_ratio`；Context 另有 `stage1_context_video_valid_ratio`，
避免仅看绝对数量而误判覆盖率。

查看 TensorBoard：

```bash
tensorboard --logdir model_results/RUN_NAME/RUN_ID/stage1_tensorboard
```

## 11. Checkpoint 与恢复

checkpoint 的 `stage1` 字段保存：

```text
enabled
perturbation
alpha
train_rng_state
calibration_rng_state
dryrun_rng_state
beta
tau_act
tau_bg
calibrated
```

checkpoint 只保存 `state_dict/ema_state_dict`，不再重复序列化完整的
`model/model_ema` Python 对象。原字段从未被训练恢复或评测入口读取，而且
Stage 1 模型持有三套 `torch.Generator`，旧式完整对象序列化会在 PyTorch 1.9
上报 `TypeError: cannot pickle 'torch._C.Generator' object`。RNG 的可恢复状态
已经以三个独立 ByteTensor 字段保存，因此移除完整对象不会损失恢复信息，同时
会减小 checkpoint 体积。

实际训练入口 `train_da.py` 和 BaseTrainer 的备用恢复入口都会先调用
`validate_stage1_resume`。只有兼容性检查全部通过，才加载模型参数、三套 RNG、
阈值和 calibrated 状态。以下情况直接抛出 `ValueError`：

```text
checkpoint 含 Stage 1 状态，但当前使用 --stage1 off
checkpoint perturbation=shift，当前使用 --stage1 context
checkpoint perturbation=context，当前使用 --stage1 shift
checkpoint alpha 与当前 model.pl_threshold 不同
恢复 epoch 已达到校准时刻，但 checkpoint 没有 Stage 1 状态
恢复 epoch 已达到校准时刻，但 checkpoint 仍标记为未校准
checkpoint 无法确定旧 Stage 1 模式或 alpha
```

所以忘记传 `--stage1` 或给错模式不会再静默关闭模块或混用阈值。恢复尚未到
校准 epoch 的普通 baseline checkpoint 仍被允许，此时 Stage 1 会在后续固定
时刻正常校准。

新 checkpoint 直接保存 `enabled/perturbation/alpha`。首版 Stage 1 checkpoint
尚未保存这三个字段，校验器会从 checkpoint 内部保存的
`config.stage1.perturbation` 和 `config.model.pl_threshold` 回填，因此已有
Shift/Context checkpoint 无需作废；旧字段 `rng_state` 也继续兼容。检查通过
后，连续训练与“保存后恢复”会从相同的正式扰动 RNG 状态继续。

checkpoint 内的 Stage 1 状态是正常断点恢复的权威来源。只有配置中显式给出
`calibration_artifact` 且 checkpoint 没有已校准状态时，Trainer 才尝试加载
外部校准工件。加载前必须同时匹配：

```text
git_commit
perturbation
alpha
source_split
config_sha256
calibration_epoch
calibration_seed
quantile
epsilon_bg
model_state_sha256
manifest_sha256
```

任一项不匹配都会抛出异常，不会静默把 Shift 阈值用于 Context，或把旧模型的
阈值用于新模型。

## 12. 测试

本地可运行：

```bash
python -m unittest tests.test_stage1_utils -v
python -m compileall -q libs/models/SADA libs/trainer libs/datasets train_da.py
git diff --check
```

当前单元测试覆盖：

- 正负平移均不回卷；
- 连续插值数值、越界 mask、精确网格点只检查自身有效性；
- ReliableAction / StableBackground / Ignore 互斥；
- multi-hot GT 中任一正类均可作为正确 pseudo class；
- Context 合法窗口、无效窗口和缺失 `full_feats` fallback 均返回 CPU tensor，
  并验证无效窗口准确重建 baseline crop；
- source 校准裁剪复用 action-preserving 截断、边界裁剪和 label 保留；
- 校准 manifest 与实验元数据不匹配时拒绝加载；
- checkpoint 忘记传 Stage 1、Shift/Context 模式互换、`alpha` 不同、越过
  校准 epoch 却缺少状态时均拒绝恢复；
- 首版 checkpoint 可从内部 config 回填模式和 `alpha` 并正常通过校验；
- 连续 RNG 序列与 checkpoint 恢复后的序列一致；
- weighted quantile；
- layer-balanced beta/tau 校准。

当前共 17 个单元测试。

本地 Python 环境未安装项目 requirements 中的 `einops`，因此没有完成完整
SADA 模型实例化或 GPU 前向；这部分必须在 A40 的项目环境进行 smoke test 和
无 GT dry run 后再启动长训。单元测试和所有 Python 文件静态编译已通过。

## 13. 已知实施边界

1. 首轮按最终计划保持原 SADA conditional-loss 口径，没有实现早期草案中的
   paired-nonempty loss。
2. target-GT 离线机制分析必须在正式 checkpoint 锁定后单独执行；本次训练
   路径没有自动读取或导出 target GT，避免训练泄漏。正式 mAP 继续使用现有
   `evaluate_da.py` 的修复后多子域评测。
3. Context 的 `full_feats` 保持在 CPU；合法偏移窗口和无效视频的 baseline
   fallback 都先在 CPU 组成第二视图 batch，再由 preprocessing 整批移到
   GPU，因此不会把完整长视频常驻显存，也不会出现混合 CPU/GPU 的
   `torch.cat`。Shift 和 Context 都会在原训练前向之外增加两次无梯度 target
   routing forward（原视图一次、扰动视图一次），A40 smoke test 必须确认
   实际显存和耗时。
4. `stage1_calibration.json` 是冻结阈值的权威工件。不得根据 target dry-run
   数量手工修改后继续同一轮实验。

## 14. 推荐复现顺序

1. 安装更新后的 `requirements.txt`；
2. 用 `--stage1 off` 做一次短 B0 回归；
3. 用相同 seed 分别启动 Shift 与 Context；
4. 检查 calibration JSON 是否生成且三个阈值非空；
5. 查看 dry-run 日志是否无 NaN、Inf、插值越界；
6. 检查 `*_total`、逐类别筛选数量、Context 视频覆盖率和全空 batch 比例；
7. 在 A40 分别完成 baseline 回归、Shift 一步、Context 混合有效/无效视频
   一步、全空路由一步和真实 checkpoint 断点一致性 smoke；
8. 确认显存、耗时和 smoke 结果后继续完整训练；
9. 锁定 checkpoint 后运行完整多子域正式评测；
10. 最后才进行 target-GT 离线机制分析。
