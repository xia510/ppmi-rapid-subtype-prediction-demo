# PPMI Rapid Subtype Prediction Demo

一个将 PPMI 队列研究中的 Logistic Regression 模型整理为可复用预测工具的演示项目。

## 项目功能

- 从 JSON 读取一位受试者的 12 项原始特征；
- 使用训练集拟合的 `StandardScaler` 完成特征标准化；
- 使用经过 sigmoid 概率校准的 Logistic Regression 输出 Rapid 亚型概率；
- 按研究阈值输出演示性分类结果；
- 展示全部 12 项特征对该样本线性预测得分的贡献，并列出正向、负向贡献最大的特征。

## 目录说明

```text
app/                 预测与输入校验代码
artifacts/           本地导出的模型文件（不提交到 Git）
examples/            示例输入 JSON
outputs/             本地预测输出（不提交到 Git）
scripts/             从研究训练集导出模型的脚本
tests/               自动化测试
```

## 环境

- Python 3.9+
- pandas 2.3.3
- scikit-learn 1.5.0
- joblib 1.5.1

安装依赖：

```powershell
pip install -r requirements.txt
```

## 使用方式

先在本地从研究训练集导出模型：

```powershell
python scripts/export_model.py
```

再使用示例受试者数据完成预测：

```powershell
python app/predictor.py --input examples/sample_patient.json --artifact artifacts/model_bundle.joblib --output outputs/prediction.json
```

## 结果解释

`rapid_probability` 是校准后的 Rapid 亚型概率；`research_threshold` 是原研究流程中通过交叉验证得到的研究阈值。`contribution` 表示特征对逻辑回归线性得分的贡献，不能被解释为因果效应或临床建议。

## 免责声明

本项目仅用于科研复现与工程学习演示，不可用于临床诊断或医疗决策。

## Learning log

- 2026-09-14: Initialized the Git repository and pushed the project to GitHub.
- 2026-09-14: Practiced the Git command-line workflow.

