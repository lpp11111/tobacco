# predict_res — 评估结果输出目录

本目录存放各模型的评估结果（CSV 格式），按模型类型分类存放。

## 目录结构

| 子目录 | 说明 |
|--------|------|
| `medium/` | medium 模型评估结果 |
| `small/` | small 分割模型评估结果 |
| `large/` | large 分割模型评估结果 |
| `extra_small/` | extra_small 模型评估结果 |
| `finetuned_medium/` | 微调后 medium 模型评估结果 |
| `t_scratch/` | t_scratch 模型评估结果 |
| `t_transfer/` | t_transfer 模型评估结果 |

## 生成方式

运行以下脚本会自动将结果保存到对应子目录：

```bash
python check/check_transfer_medium.py   → predict_res/medium/
python check/check_transfer_small.py    → predict_res/small/
python check/check_transfer_large.py    → predict_res/large/
python check/check_transfer_t.py        → predict_res/ (根目录，对比结果)
```