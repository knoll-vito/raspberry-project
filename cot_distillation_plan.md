# 树莓派5灾后伤亡风险评估系统 · CoT 蒸馏技术方案

> **文档定位**：本方案用于 dispatch 阶段的工程实施参考，作者为大模型训练工程师视角，聚焦 Claude Code 编排 + DeepSeek-V4 蒸馏 + 边缘部署的完整链路。
>
> **文档版本**：v1.0
> **适用项目**：灾后人员伤亡风险评估（边缘 AI 推理）
> **目标设备**：Raspberry Pi 5（ARM64，无独立 GPU）

---

## 一、项目背景与目标

### 1.1 业务目标

在灾后通信中断的离线环境下，依托树莓派5实现**实时人员伤亡风险评估**，输出连续风险评分（0~100）与风险等级，辅助救援决策。

### 1.2 技术目标

| 指标 | 要求 |
|---|---|
| 推理延迟 | < 5 ms |
| 模型体积 | < 2 MB |
| 运行内存 | < 200 MB |
| 离线可用性 | 100%（无网络依赖） |
| 与 Teacher 一致性 | 皮尔逊相关系数 > 0.85 |

### 1.3 核心方法论

**思维链知识蒸馏（CoT Knowledge Distillation）**——通过云端大模型生成带推理过程的高质量软标签，将"推理能力"迁移到边缘可部署的轻量模型。

> ⚠️ **关于 DeepSeek-V4 的说明**
> 本方案以"DeepSeek-V4 已发布"为前提撰写。由于具体接口规范、定价、context window 等参数需在实施时通过官方文档核验，**实施前请先确认 V4 的 API 兼容性**（预计仍兼容 OpenAI 接口格式）。若 V4 暂未稳定，方案可无缝降级至 DeepSeek-V3.1 或切换至 DeepSeek-R1。

---

## 二、技术方案总览

### 2.1 三方角色定义

```
┌─────────────────────────────────────────────────────┐
│  ① Claude Code（编排层 / Orchestrator）              │
│     职责：搭项目、写代码、跑 Pipeline、debug、验证      │
│     不参与：实际推理、Teacher 标注、Student 训练运算    │
└─────────────────────────────────────────────────────┘
                       ↓ 编排
┌─────────────────────────────────────────────────────┐
│  ② DeepSeek-V4（Teacher / 知识源）                   │
│     职责：对灾情样本输出推理链 + 软标签 + 硬标签        │
│     调用方式：OpenAI 兼容 API                         │
│     运行位置：云端（仅训练阶段使用）                    │
└─────────────────────────────────────────────────────┘
                       ↓ 软标签数据集
┌─────────────────────────────────────────────────────┐
│  ③ LightGBM（Student / 部署模型）                    │
│     职责：学习 Teacher 的输入→评分映射                 │
│     运行位置：树莓派5（永久离线推理）                    │
└─────────────────────────────────────────────────────┘
```

### 2.2 数据流图

```
原始灾情特征
    │
    ▼
[Claude Code 编排] ──→ DeepSeek-V4 API ──→ 推理链 + 软标签
    │                                            │
    │                                            ▼
    │                                    质量过滤模块
    │                                            │
    │                                            ▼
    └──────────→ LightGBM 训练 ←─────── 干净训练集
                       │
                       ▼
                  model.pkl (~500KB)
                       │
                       ▼
              树莓派5 离线推理（<5ms）
```

---

## 三、Student 模型选型分析（核心决策）

这是本方案的关键决策点。下面对 **5 个候选模型**做横向评估：

### 3.1 候选模型对比

| 模型 | 体积 | 推理延迟（Pi5） | 训练复杂度 | 处理软标签 | 树莓派部署难度 | 综合评分 |
|---|---|---|---|---|---|---|
| **LightGBM** | 200KB~1MB | < 1 ms | ⭐⭐ | 原生支持 | 极易（pip 直装） | ⭐⭐⭐⭐⭐ |
| **XGBoost** | 500KB~2MB | 1~3 ms | ⭐⭐ | 原生支持 | 易 | ⭐⭐⭐⭐ |
| **TabNet** | 5~20MB | 20~50 ms | ⭐⭐⭐⭐ | 支持 | 中（需 PyTorch） | ⭐⭐ |
| **小型 MLP** | 100KB~5MB | 1~5 ms | ⭐⭐⭐ | 支持 | 中（需 ONNX） | ⭐⭐⭐ |
| **蒸馏 BERT** | 50~250MB | 200~500 ms | ⭐⭐⭐⭐⭐ | 支持 | 难（内存吃紧） | ⭐ |

### 3.2 选型推荐：**LightGBM**

**入选理由：**

1. **任务匹配度最高**：本项目特征是**结构化表格数据**（震级、倒塌率、温度等数值特征），梯度提升树是这种数据的天然最优解，比深度学习更稳定
2. **软标签原生友好**：直接用回归目标 `objective='regression'` 学习连续 score，无需特殊处理
3. **样本权重内建支持**：`sample_weight=confidence` 可直接利用 Teacher 的置信度
4. **部署零摩擦**：`pip install lightgbm` 在树莓派5上能直接装（已有 ARM64 wheel），无需交叉编译
5. **可解释性强**：特征重要性、SHAP 值现成可用，便于救援领域专家审查模型决策
6. **训练数据需求低**：500~2000 条样本即可获得不错效果，符合本项目数据规模

**备选方案（按场景选择）：**

| 场景 | 推荐模型 | 理由 |
|---|---|---|
| 主方案 | **LightGBM** | 综合最优 |
| 数据量超过 5000 条 | **XGBoost** | 大数据下精度略胜 |
| 需要输出推理过程的端侧 | **小型 MLP + 量化** | 可承载部分语义信息 |
| 引入文本类灾情描述 | **LightGBM + 文本特征工程** | 而非换模型，应当增强特征 |

### 3.3 LightGBM 关键超参（本项目调优起点）

```python
params = {
    "objective": "regression",
    "metric": "rmse",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "num_leaves": 31,
    "min_child_samples": 20,    # 防止过拟合（小数据集关键参数）
    "reg_alpha": 0.1,           # L1 正则
    "reg_lambda": 0.1,          # L2 正则
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}
```

---

## 四、完整技术路线（六阶段）

### 阶段 1：环境准备与项目初始化

**操作平台**：本地开发机（macOS / Linux）+ Claude Code

**关键动作**：
- 在 Claude Code 中下达"项目骨架指令"，自动创建标准化目录结构
- 配置 `.env`（DeepSeek API key）与 `config.yaml`（模型参数、并发设置）
- 初始化 Git 仓库，建立第一个 commit 作为安全网

**产出**：可运行的空骨架项目 + 完整 requirements.txt

---

### 阶段 2：模拟样本生成（特征空间构造）

**操作平台**：Claude Code 编排 + 本地执行

**关键动作**：
- 由 Claude Code 编写 `scripts/generate_samples.py`
- 生成 500~1000 条覆盖各类灾情的模拟样本
- 必须保证特征**正交覆盖**（避免分布偏斜）

**特征矩阵设计**：

| 特征 | 取值范围 | 分桶策略 |
|---|---|---|
| disaster_type | 地震/洪水/火灾/泥石流 | 均衡分布 |
| magnitude | 3.0~8.5 | 每 0.5 一档，均匀采样 |
| building_collapse_rate | 0~1.0 | 5 个区间 |
| estimated_trapped | 0~500 | 对数采样 |
| temperature_c | -10~45 | 极端温度加权 |
| hours_since_disaster | 0~72 | 黄金 72 小时高密度采样 |
| rescue_eta_hours | 0.5~24 | 偏向短时间 |
| road_accessibility | 0~1.0 | 5 个区间 |

**产出**：`data/raw_samples.jsonl`（无标签）

---

### 阶段 3：Teacher 标注（核心成本环节）

**操作平台**：Claude Code 编排 + DeepSeek-V4 API

**关键动作**：

#### 3.1 Prompt 设计（4 个版本）

| 版本 | 用途 | 适用样本占比 |
|---|---|---|
| v1 基础版 | 普通样本 | 60% |
| v2 强制推理链版 | 复杂多因素样本 | 25% |
| v3 对比推理版 | 边界模糊样本 | 10% |
| v4 不确定性感知版 | 数据缺失样本 | 5% |

#### 3.2 标注 Pipeline 关键设计

```python
# 伪代码示意 - 实际由 Claude Code 生成
async def label_sample(sample, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = await deepseek_v4.chat.completions.create(
                model="deepseek-v4",
                messages=[...],
                temperature=0.3,
                timeout=60
            )
            parsed = parse_response(response)
            if validate(parsed):
                save_to_jsonl(parsed)  # 立即落盘 - 断点续标关键
                return parsed
        except RateLimitError:
            await exponential_backoff(attempt)
        except Exception as e:
            log_failure(sample, e)
    return None
```

#### 3.3 必备工程特性

- ✅ 异步并发（`asyncio + Semaphore`，初期 max_concurrent=3）
- ✅ 指数退避重试
- ✅ 断点续标（基于 sample_id 去重）
- ✅ 实时 token 计费统计
- ✅ 失败队列（独立 jsonl 存储未成功样本，二次重试）

**产出**：`data/teacher_labeled.jsonl`（含推理链 + 软标签 + 硬标签 + confidence）

---

### 阶段 4：数据质量过滤

**操作平台**：Claude Code 本地执行

**过滤规则（多重把关）**：

| 规则类型 | 示例 | 预期过滤率 |
|---|---|---|
| **格式合规性** | JSON 解析失败、字段缺失 | 2~5% |
| **置信度门槛** | confidence < 0.5 | 5~10% |
| **领域常识冲突** | 7级地震 + score < 50 | 1~3% |
| **推理链完整性** | reasoning < 100 字符 | 3~5% |
| **重复样本** | 输入特征哈希去重 | 1~2% |

**总过滤率预期**：12%~25%

**产出**：`data/clean_dataset.jsonl`

---

### 阶段 5：Student 训练与评估

**操作平台**：Claude Code 编排 + 本地训练

**训练 Pipeline**：

```
clean_dataset.jsonl
    ↓
特征工程（one-hot、归一化）
    ↓
80/20 划分训练/验证集
    ↓
LightGBM 训练（confidence 加权 + early stopping）
    ↓
特征重要性分析
    ↓
一致性评估（Student vs Teacher）
    ↓
错误样本分析（找出预测分歧最大的 Top10）
    ↓
导出 model.pkl
```

**评估指标**：

| 指标 | 目标值 | 含义 |
|---|---|---|
| MAE | < 5 | 平均绝对误差 |
| RMSE | < 8 | 均方根误差 |
| Pearson r | > 0.85 | 与 Teacher 排序一致性 |
| 边界样本准确率 | > 80% | 风险等级分类正确率 |

**产出**：`student/model.pkl` + 评估报告

---

### 阶段 6：树莓派5 部署与压测

**操作平台**：Pi5 物理设备

**关键动作**：
- 通过 SCP 传输 `model.pkl` 与 `inference.py`
- 在 Pi5 上 `pip install lightgbm joblib`
- 写 systemd 服务实现开机自启
- Benchmark 1000 次推理，统计 P50/P95/P99 延迟
- 监控 CPU 温度、内存占用

**性能验收**：

| 指标 | 验收标准 |
|---|---|
| P50 延迟 | < 1 ms |
| P99 延迟 | < 5 ms |
| 模型大小 | < 2 MB |
| 运行内存 | < 200 MB |
| CPU 占用（推理时） | < 30% |

**产出**：可在 Pi5 上稳定运行的离线推理服务

---

## 五、必须注意的关键细节（基于完整对话总结）

### 5.1 Teacher 一致性陷阱（最易被忽视）

⚠️ **绝对避免中途换 Teacher 模型**：
- 同一数据集**必须使用同一 Teacher**全量标注
- 若需混用 V4 + R1（边界样本重标），必须做**标定对齐测试**
- 测试方法：用 100 条验证样本同时让两个模型标注，皮尔逊相关系数需 > 0.85 才可混用

### 5.2 Prompt 设计要点

- ✅ **Temperature 必须低**（0.3 左右）：保证标注稳定性
- ✅ **强制结构化输出**：用 `<reasoning>` 和 `<score>` 标签包裹
- ✅ **同一样本多次采样取中位数**：对关键样本（边界、高风险）调用 3 次，提高标签质量
- ❌ 不要让 Temperature=0：会损失推理多样性，反而让 Student 学不到泛化能力

### 5.3 数据规模分阶段策略

| 阶段 | 样本量 | 目标 |
|---|---|---|
| **概念验证（PoC）** | 100~200 条 | 跑通 Pipeline，验证 Prompt 质量 |
| **初步可用（v1）** | 500~1000 条 | 模型初步可用，特征覆盖完整 |
| **生产级别（v2）** | 3000~5000 条 | 稳定上线，加入真实历史灾情数据 |

⚠️ **不要一次性标 5000 条**——先 200 条试跑，发现 Prompt/解析/过滤问题成本极低；问题修完再放量。

### 5.4 Claude Code 使用纪律

- ✅ **每条指令限定单一任务范围**：避免一次让它写完所有模块
- ✅ **每个阶段完成后立即 git commit**：建立回滚点
- ✅ **明确"不要过度抽象"**：本项目不需要复杂设计模式
- ✅ **让它自己产出 checklist**：用于自我验证完成度
- ❌ **不要让 Claude Code 替你做领域判断**：风险评分阈值、Prompt 中的灾害术语等必须人工把关

### 5.5 软硬标签结合策略

```python
total_loss = α * loss_soft + (1 - α) * loss_hard
```

- 数据量 < 500：α = 0.8（更依赖软标签的丰富信息）
- 数据量 500~2000：α = 0.6
- 数据量 > 2000：α = 0.4（硬标签更能反映明确边界）

### 5.6 成本控制要点

| 项目 | 估算 |
|---|---|
| DeepSeek-V4 单次调用 | 约 0.001~0.005 USD（取决于 token 量） |
| 1000 条标注总成本 | < 5 USD |
| Claude Code 编排成本 | 取决于使用量 |

🟢 **DeepSeek 系列是目前性价比最高的 Teacher 选择**，1000 条灾情标注成本远低于人工标注。

### 5.7 树莓派部署细节

- **优先用 SSD 而非 SD 卡**：模型加载速度提升 5~10 倍
- **散热是必须的**：持续推理建议加主动风扇，否则降频严重
- **Python 版本统一**：开发机和 Pi5 都用 Python 3.11，避免 pickle 兼容性问题
- **模型加载放在服务启动时**：不要每次推理都重新加载

### 5.8 后续迭代方向

| 方向 | 优先级 | 说明 |
|---|---|---|
| 引入真实历史灾情数据 | 高 | 与模拟数据混合训练，提升真实场景泛化 |
| 多模态扩展（卫星图） | 中 | 需要换 Student 为 CNN+LightGBM 融合 |
| 在线学习能力 | 低 | 边缘端增量学习，需重新设计 Pipeline |
| Hailo-8 AI HAT 加速 | 中 | 如果引入图像特征，可显著提速 |

---

## 六、实施 Checklist（dispatch 阶段直接执行）

### 6.1 环境准备
- [ ] 申请 DeepSeek API key 并充值
- [ ] 验证 V4 接口可用性，确认计费规则
- [ ] 本地装好 Claude Code 客户端
- [ ] 准备 Raspberry Pi 5 + 主动散热 + SSD

### 6.2 项目启动
- [ ] 用项目骨架指令初始化 Claude Code 项目
- [ ] git init 并提交骨架 commit
- [ ] 配置 `.env` 与 `config.yaml`

### 6.3 数据生成
- [ ] 编写 `generate_samples.py`，生成 200 条 PoC 样本
- [ ] 人工抽查 20 条，确认特征分布合理

### 6.4 Teacher 标注（PoC）
- [ ] 实现 `deepseek_client.py`（单线程版）
- [ ] 标注 200 条 PoC 样本
- [ ] 人工抽查推理链质量

### 6.5 Student 训练（PoC）
- [ ] 实现训练脚本
- [ ] 训练第一版模型，跑通完整链路
- [ ] 评估一致性指标

### 6.6 放量到 v1
- [ ] 升级标注 Pipeline 至并发 + 断点续标版
- [ ] 生成并标注 1000 条样本
- [ ] 训练 v1 模型，达到验收指标

### 6.7 部署验证
- [ ] Pi5 上部署 v1 模型
- [ ] 性能压测达标
- [ ] 集成到下游救援决策系统

---

## 七、风险预案

| 风险点 | 影响 | 应对措施 |
|---|---|---|
| DeepSeek-V4 接口不稳定 | 标注中断 | 降级至 V3.1 或 R1，标定对齐 |
| Teacher 输出格式漂移 | 解析失败率高 | 加强 parser 容错 + 失败重试 |
| Student 泛化能力差 | 真实场景失效 | 引入真实历史数据，增加样本多样性 |
| Pi5 性能不足 | 推理超时 | 加 Hailo-8 HAT 或精简模型 |
| 领域知识偏差 | 评分不符救援常识 | 邀请应急管理专家 review 标注样本 |

---

## 八、参考输出形态（项目交付物）

```
cot_distillation/
├── data/
│   ├── raw_samples.jsonl          # 1000 条原始样本
│   ├── teacher_labeled.jsonl      # DeepSeek-V4 标注结果
│   └── clean_dataset.jsonl        # 过滤后训练集
├── teacher/
│   └── [完整标注 Pipeline 代码]
├── student/
│   ├── model.pkl                  # ⭐ 最终交付模型（< 1MB）
│   ├── train.py
│   └── evaluation_report.html     # 评估报告
├── deploy/
│   ├── inference.py               # ⭐ 树莓派推理脚本
│   └── benchmark_report.md        # 性能压测报告
└── docs/
    ├── prompt_design.md           # Prompt 设计文档
    └── domain_knowledge.md        # 领域知识沉淀
```

---

## 九、本方案核心价值总结

> **用云端大模型的智慧，不用云端大模型的身体。**

通过 Claude Code 编排 DeepSeek-V4 的推理能力，将其凝练为一个不到 1MB 的 LightGBM 模型，部署到完全离线的树莓派5上——这不是简单的模型压缩，而是**跨架构的知识迁移**。在灾后通信中断的极端场景下，这套方案可以让一台 75 美元的边缘设备，做出接近大模型水平的风险判断，这就是 CoT 蒸馏在边缘 AI 场景下的工程价值。

---

**文档结束**

> 本文档供 dispatch 阶段实施参考。如需针对某一阶段展开详细 SOP，或在实施过程中遇到具体问题，建议保留本文档作为基线，按章节迭代细化。
