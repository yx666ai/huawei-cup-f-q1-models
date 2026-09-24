# 华为杯 F 题第一问 Python 代码

本公开代码仓库整理了第一大问三个小问的可复现 Python 实现：

- `q1_part1_quality.py`：22 个质量指标预处理、适宜性转换、熵权法、TOPSIS 与最终质量分 `Q`；
- `q1_part2_conflict.py`：质量冲突定义、成因分析、冲突惩罚与扩展集验证；
- `q1_part3_mixture.py`：17 域二阶混料响应面、验证、方向导数、最优配比与跨尺度分析。

## 数据与隐私

仓库不包含原始数据、处理后的数据、模型输出或本机绝对路径。运行时请自行准备赛题数据，并通过脚本参数或约定目录传入。

## 环境

建议使用 Python 3.10 或更高版本：

```bash
python -m pip install -r requirements.txt
```

第一小问脚本支持命令行参数，可通过以下方式查看：

```bash
python q1_part1_quality.py --help
```

第二、第三小问脚本按当前工作目录下的约定结构读取前序结果与赛题数据，直接运行即可：

```bash
python q1_part2_conflict.py
python q1_part3_mixture.py
```

## 说明

这些脚本来自已经完成并用实际数据验证的建模流程。第一、第二、第三小问之间的数据文件存在依赖时，请按顺序运行，并保持脚本预期的目录结构或显式传入路径。
