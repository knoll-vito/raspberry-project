# 灾后伤亡风险评估系统

> 基于 CoT 知识蒸馏的轻量化灾后人员伤亡风险评估模型，可部署于树莓派 5 等边缘设备离线运行。

**核心思路**：用大模型（DeepSeek-V4 Teacher）生成带推理链的高质量标注，蒸馏到 LightGBM（Student）小模型，实现毫秒级推理 + 可解释输出。

## 项目特性

- 16 维特征向量（5 灾种 one-hot + 9 连续 + 2 二值），覆盖地震、洪水、山体滑坡、城市火灾、森林火灾
- 四级风险等级输出：一般 / 较大 / 重大 / 特别重大（对齐国家应急响应标准）
- TreeSHAP 可解释性分析 + 动态四段式推理生成
- Isotonic Regression 后校准，RMSE 5.92、等级准确率 96.4%
- 模型体积 ~410KB，树莓派 5 单次推理 < 1ms

## 目录结构

```
.
├── config.yaml                         # 全局配置（路径 / 超参 / API 设置）
├── requirements.txt                    # Python 依赖
├── data/                               # 数据目录（JSONL 格式）
│   ├── starter_real.csv                #   290 条真实历史灾害记录
│   ├── augmented_samples.jsonl         #   300 条 LHS 合成样本
│   ├── merged_train.jsonl              #   690 条合并训练集
│   └── eval_test_set.jsonl             #   138 条 Teacher 生成的评估集
├── scripts/
│   ├── generate_samples.py             # 基础样本生成
│   ├── generate_augmented_samples.py   # Latin Hypercube Sampling 合成样本
│   ├── run_labeling.py                 # Teacher 标注编排（DeepSeek / Mock）
│   └── filter.py                       # 质量过滤
├── teacher/
│   ├── prompt_templates.py             # Teacher prompt 模板（v1~v4）
│   ├── deepseek_client.py              # DeepSeek API 异步客户端
│   ├── mock_labeler.py                 # 规则版 Mock Teacher（零成本标注）
│   ├── recommendations.py             # 动态救援建议生成
│   └── test_generator.py              # Teacher 出题器（双向蒸馏）
├── student/
│   ├── features.py                     # 特征工程（one-hot + 数值 + 二值）
│   ├── train.py                        # LightGBM 训练（confidence 加权）
│   ├── evaluate.py                     # 模型评估
│   ├── calibration.py                  # Isotonic Regression 后校准
│   ├── reasoning_generator.py          # 基于 SHAP 的动态推理生成
│   ├── bidirectional_train.py          # 双向蒸馏训练
│   ├── model.pkl                       # 训练好的 LightGBM 模型
│   └── calibrator.pkl                  # 校准器
├── deploy/
│   ├── inference.py                    # 推理引擎（简洁 / 完整决策包）
│   └── benchmark.py                    # 性能压测
└── tools/
    ├── predict_interactive.py          # 交互式推理（逐字段输入）
    ├── csv_to_jsonl.py                 # CSV → JSONL 转换
    └── smoke_test.py                   # 端到端链路验证
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 使用已训练模型推理

项目已包含训练好的 `student/model.pkl` 和 `student/calibrator.pkl`，安装依赖后即可直接推理。

## 使用方法

### 交互模式

逐字段输入参数，适合快速体验和现场测试：

```bash
python3 tools/predict_interactive.py
```

按提示输入各项参数（直接回车使用默认值），输出 Student 校准分数与 Mock baseline 对比：

```
  灾种 [earthquake/flood/urban_fire/forest_fire/landslide] (默认 earthquake): earthquake
  强度（地震=里氏；其他按映射） [3.0~8.5] (默认 6.5): 7.5
  建筑倒塌率 [0.0~1.0] (默认 0.4): 0.6
  ...

============================================================
  Student (LightGBM):    score= 92.07  level=特别重大
  Mock baseline (规则):   score= 89.30  level=特别重大
  diff (Student - Mock): +2.77
============================================================
```

### 命令行 JSON 模式

适合脚本调用和批量测试：

```bash
# 简洁输出（score + risk_level）
python3 deploy/inference.py --json '{
  "disaster_type": "earthquake",
  "magnitude": 7.5,
  "building_collapse_rate": 0.6,
  "estimated_trapped": 200,
  "temperature_c": 5,
  "hours_since_disaster": 12,
  "rescue_eta_hours": 4,
  "road_accessibility": 0.3,
  "medical_accessibility": 0.4,
  "rescue_skill_level": 0.5,
  "night_time": 1,
  "holiday_event": 0
}'
```

输出：

```json
{
  "score": 92.07,
  "risk_level": "特别重大"
}
```

### 完整决策包模式

加 `--full` 输出含 SHAP 解释、推理链、救援建议的完整 JSON：

```bash
python3 deploy/inference.py --full --json '{
  "disaster_type": "earthquake",
  "magnitude": 7.5,
  "building_collapse_rate": 0.6,
  "estimated_trapped": 200,
  "temperature_c": 5,
  "hours_since_disaster": 12,
  "rescue_eta_hours": 4,
  "road_accessibility": 0.3,
  "medical_accessibility": 0.4,
  "rescue_skill_level": 0.5,
  "night_time": 1,
  "holiday_event": 0
}'
```

可附加设备和位置信息：

```bash
python3 deploy/inference.py --full \
  --device-id RPI5-EM-01 \
  --lat 31.284 --lon 121.503 \
  --json '{ ... }'
```

### 批量推理

传入 JSONL 文件，每行一条样本：

```bash
# 简洁输出
python3 deploy/inference.py --file data/eval_test_set.jsonl

# 完整决策包
python3 deploy/inference.py --file data/eval_test_set.jsonl --full
```

## 输入参数说明

| 参数 | 类型 | 范围 | 说明 |
|------|------|------|------|
| `disaster_type` | string | earthquake / flood / urban_fire / forest_fire / landslide | 灾种 |
| `magnitude` | float | 3.0 ~ 9.5 | 灾害强度（地震为里氏震级） |
| `building_collapse_rate` | float | 0.0 ~ 1.0 | 建筑倒塌率 |
| `estimated_trapped` | int | 0 ~ 10000 | 估计被困人数 |
| `temperature_c` | float | -20 ~ 45 | 环境温度（℃） |
| `hours_since_disaster` | float | 0 ~ 168 | 灾后已过时间（小时） |
| `rescue_eta_hours` | float | 0 ~ 72 | 救援预计到达时间（小时） |
| `road_accessibility` | float | 0.0 ~ 1.0 | 道路可通行性 |
| `medical_accessibility` | float | 0.0 ~ 1.0 | 医疗资源可及性 |
| `rescue_skill_level` | float | 0.0 ~ 1.0 | 救援队伍技能水平 |
| `night_time` | int | 0 / 1 | 是否夜间事件 |
| `holiday_event` | int | 0 / 1 | 是否节假日/大型活动期间 |

## 完整训练流程

如需从零训练模型，按以下步骤执行：

```bash
# 1) 生成合成样本（Latin Hypercube Sampling，300 条）
python3 scripts/generate_augmented_samples.py

# 2) Teacher 标注（Mock 模式无需 API key）
python3 scripts/run_labeling.py --provider mock

# 3) 质量过滤
python3 scripts/filter.py

# 4) 训练 Student 模型
python3 student/train.py

# 5) 模型评估
python3 student/evaluate.py

# 6) 拟合校准器（需要 Teacher 标注的评估数据）
python3 student/calibration.py --fit

# 7) 性能压测
python3 deploy/benchmark.py
```

### 使用真实 DeepSeek Teacher

```bash
cp .env.example .env
# 编辑 .env 填入 DEEPSEEK_API_KEY
python3 scripts/run_labeling.py --provider deepseek
```

## 模型评估结果

基于 138 条 DeepSeek-V4 Teacher 生成的测试集评估：

| 指标 | 校准前 | 校准后 |
|------|--------|--------|
| RMSE | 8.03 | **5.92** |
| MAE | 5.99 | **3.60** |
| Bias | -3.54 | **0.00** |
| Pearson r | 0.4828 | **0.5796** |
| 等级准确率 | 91.3% | **96.4%** |
| 大偏差样本 (>10分) | 25 | **7** |

## 技术栈

- **Teacher**: DeepSeek-V4 / Mock 规则引擎
- **Student**: LightGBM（回归）
- **校准**: scikit-learn IsotonicRegression
- **可解释性**: TreeSHAP
- **部署目标**: Raspberry Pi 5（Python 3.11+）
