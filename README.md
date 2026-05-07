# 灾后伤亡风险评估系统 · CoT 蒸馏 PoC

> Claude Code 编排 + DeepSeek-V4（Teacher）+ LightGBM（Student）→ 树莓派 5 离线推理

完整方案见 [`cot_distillation_plan.md`](./cot_distillation_plan.md)。本目录是该方案的 PoC 实现。

## 目录结构

```
.
├── config.yaml                # 全局配置
├── .env.example               # API key 模板（拷贝为 .env）
├── requirements.txt
├── data/                      # JSONL 数据（git 忽略）
├── scripts/
│   ├── generate_samples.py    # 阶段2：模拟样本生成
│   ├── run_labeling.py        # 阶段3：Teacher 标注编排
│   └── filter.py              # 阶段4：质量过滤
├── teacher/
│   ├── prompt_templates.py    # 4 个 prompt 版本
│   ├── deepseek_client.py     # 真实 API 客户端（异步并发 + 断点续标）
│   └── mock_labeler.py        # 规则版 Mock，用于无 key 时跑通链路
├── student/
│   ├── train.py               # 阶段5：LightGBM 训练
│   └── evaluate.py            # 一致性评估
├── deploy/
│   ├── inference.py           # 阶段6：单条 / 批量推理
│   └── benchmark.py           # 性能压测
└── tools/
    └── smoke_test.py          # 沙箱无 LightGBM 时的端到端 smoke 验证
```

## 快速开始（最小端到端 demo）

```bash
# 1) 装依赖
pip install -r requirements.txt

# 2) 用 Mock Teacher 跑通端到端（无需 API key）
python scripts/generate_samples.py
python scripts/run_labeling.py --provider mock
python scripts/filter.py
python student/train.py
python student/evaluate.py
python deploy/benchmark.py
```

## 切到真实 DeepSeek

```bash
cp .env.example .env
# 编辑 .env 填入 DEEPSEEK_API_KEY
python scripts/run_labeling.py --provider deepseek
```

## 当前状态

- [x] 项目骨架
- [x] 样本生成
- [x] Mock Teacher（规则版）
- [x] Student 训练 / 评估 / 推理脚本
- [ ] 真实 DeepSeek-V4 标注（需 API key）
- [ ] 树莓派 5 部署验证
