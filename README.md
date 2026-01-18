# Music Genre Classifier

基于 Log-Mel 频谱 + CNN 的音乐风格识别项目。

# 一、项目结构与文件说明

| 路径                                | 类型 | 作用                                          |
| ----------------------------------- | ---- | --------------------------------------------- |
| README.md                           | file | 项目说明文档。                                |
| requirements.txt                    | file | Python 依赖列表。                             |
| .gitignore                          | file | Git 忽略规则。                                |
| data/                               | dir  | 数据集根目录。                                |
| data/raw/                           | dir  | 原始音频数据（按类别子目录组织）。            |
| data/features/                      | dir  | 预处理后的特征（如 log-mel）保存目录。        |
| data/splits/                        | dir  | 训练/验证/测试划分文件目录。                  |
| data/bad/                           | dir  | 扫描出的坏音频移动到此目录。                  |
| models/                             | dir  | 训练产物与映射文件目录。                      |
| models/cnn_melspec.pth              | file | 训练好的 CNN 权重。                           |
| models/label_map.json               | file | 类别映射（index -> genre）。                  |
| models/checkpoints/                 | dir  | 训练过程中保存的检查点目录。                  |
| runs/                               | dir  | 训练/评估产物与可视化输出目录。               |
| runs/train_log.csv                  | file | 训练日志（loss/acc/耗时）。                   |
| runs/train_curves.png               | file | 训练曲线图（loss/val acc）。                  |
| runs/test_classification_report.txt | file | 测试集分类报告。                              |
| runs/confusion_matrix_test.png      | file | 测试集混淆矩阵图。                            |
| samples/                            | dir  | 示例音频（用于快速测试/演示）。               |
| src/config.py                       | file | 统一管理音频预处理参数与路径配置。            |
| src/utils_audio.py                  | file | 音频读取、切片、Log-Mel 与归一化工具函数。    |
| src/dataset.py                      | file | 基于目录结构的数据集封装（PyTorch Dataset）。 |
| src/split_dataset.py                | file | 基于 split.json 的数据集封装（可复现划分）。  |
| src/model.py                        | file | CNN 模型定义。                                |
| src/train.py                        | file | 训练入口与日志/曲线保存。                     |
| src/eval.py                         | file | 测试评估、分类报告与混淆矩阵生成。            |
| src/infer.py                        | file | 单文件推理与 Top-K 结果计算。                 |
| tools/scan_bad_audio.py             | file | 扫描不可解码音频并移动到 data/bad/。          |
| tools/make_split.py                 | file | 生成 data/splits/split.json 划分文件。        |
| tools/error_analysis.py             | file | 生成测试集预测结果与难例列表。                |
| tools/plot_cm_numbered.py           | file | 从预测结果绘制带数字标注的混淆矩阵图。        |
| web/app_streamlit.py                | file | Streamlit Web 演示（上传音频并预测）。        |

---

# 二、名词解释

### 2.1、mel 频谱图

它是一种将**音频信号**从 “物理频率域” 映射到 “人类听觉感知域” 的特征表示方法，旨在模拟人耳对声音的非线性感知特性，从而更高效地提取音频中的关键信息（如语音、音乐的特征）

[理解梅尔谱图(Understanding the Mel Spectrogram) - 知乎](https://zhuanlan.zhihu.com/p/606956924)

### 2.2、张量和标签张量

“张量”就是多维数组（PyTorch 里的基本数据结构），用来给模型做计算；“标签张量”是把类别标签也转成张量，方便和模型输出一起参与损失函数计算。

* 输入张量（**x**）：作为模型的特征输入，参与前向传播和梯度计算。
* 标签张量（**y_t**）：作为真实类别，给 **CrossEntropyLoss** 这类损失函数用来计算误差，从而训练模型。

---

# 三、实验数据

### 3.1、关键图标和数据解释

### 表格（核心）

| 实验 | 模型 | 增强 | 优化器/策略 | Params | Test Acc | Macro-F1 |
| ---- | ---- | ---- | ----------- | ------ | -------- | -------- |

 **Params** （模型参数量）这一列很关键：体现“小模型优化”的意义。

### 图（必备）

* 训练曲线（loss/val_acc）
* 混淆矩阵（baseline vs 最优）
* （可选）错误样本分析 hard_cases.csv 的统计（比如 rock 最易混淆到 pop/jazz）

---

# 四、优化流程

### E0：Baseline

输出：test_acc / macro-F1 / confusion matrix / 训练曲线

### E1：只加 SpecAugment（最划算的单点提升）

* 不改模型、不改优化器
* 论文亮点：数据增强提升泛化、尤其改善相近风格

### E2：只改训练策略（让曲线更稳）

* AdamW + Cosine + label smoothing + early stopping
* 论文亮点：训练稳定性提升，val 波动降低，最终模型更可靠

### E3：小模型结构升级（“加深但仍是小模型”）

你可以用我之前给的升级版 CNN block（每个 block 两层 conv 再 pool）：

* 参数量适中
* 更容易写：提取更复杂的局部频谱纹理

### E4：上 ResNet18/EfficientNet（大模型对照）

* 作为论文“上限参考”
* 同样用同一 split、同一评估方式
* 说明迁移学习的优势

### E5：扩大数据规模（最终提升方向）

* 强烈建议离线特征/manifest（工程化）
* 论文里可以写：规模效应、类别混淆的改善
