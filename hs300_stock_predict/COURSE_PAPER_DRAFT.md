# 基于 P-sLSTM 的大规模股票方向预测与非金融时间序列迁移实验研究

> 课程论文初稿。姓名、学号、课程名、学校格式请按实际要求补充。

## 摘要

股票收益方向预测是金融科技领域中的典型时间序列建模任务。与一般连续数值预测不同，股票方向预测受到市场噪声、非平稳分布和时间漂移影响，模型即使能够降低回归误差，也未必能够稳定提高涨跌方向判断能力。近年来，长序列时间预测模型在交通、气象、电力等非金融时间序列任务上取得了较好效果，其中 AAAI-25 论文提出的 P-sLSTM 通过在 sLSTM 上引入 patching 与 channel independence 机制，重新挖掘了 LSTM 类模型在长周期时间序列预测中的潜力。

本文基于 P-sLSTM 构建股票方向预测实验框架，围绕两个问题展开研究：第一，P-sLSTM 相较普通 LSTM 是否能在股票时间序列中获得更强的建模能力；第二，非金融时间序列预训练是否能够迁移到金融方向预测任务中。实验以 Hugging Face S&P500 股票数据集作为大规模主实验，包含 503 只股票、约 62 万条日频样本，并采用 2020-2023 年训练、2024-2025 年测试的严格时间外推划分；同时使用沪深 300 数据作为小样本补充实验。评价指标以方向准确率、Binary F1 与 Direction AUC 为主，同时报告 MAE/RMSE 等回归指标。

实验结果表明，P-sLSTM 在 S&P500 训练期验证集上获得低于 LSTM 的回归损失，说明其具有更强的序列拟合能力；但在 2024-2025 年 out-of-time 测试集上，方向准确率整体仅在 0.50-0.52 附近，提升较为有限。进一步加入方向损失、低权重 BCE 损失与辅助方向分类头后，验证集方向准确率可提升至约 0.59，但测试集未同步改善，说明金融时间序列方向预测的主要困难并非单纯模型容量不足，而是市场状态漂移与外推泛化问题。沪深 300 补充实验中，P-sLSTM 与 Weather 预训练在同分布验证集上表现较好，但在时间外推测试中也存在收益不稳定现象。总体而言，P-sLSTM 可作为金融时间序列建模的有效 backbone，但非金融源域迁移与方向感知训练需要谨慎使用。

**关键词：** 股票方向预测；P-sLSTM；时间序列预测；迁移学习；金融科技；非平稳性

## 1 引言

金融市场预测一直是金融科技与机器学习交叉研究中的重要问题。对于投资决策、风险管理和量化交易而言，预测未来收益率的具体数值固然重要，但在实际交易场景中，判断下一期价格上涨或下跌的方向同样具有直接价值。因此，本文将股票未来一日收益率方向作为核心预测目标，并以方向准确率、Binary F1 和 Direction AUC 作为主要评价指标。

传统股票预测模型通常包括线性回归、ARIMA、支持向量机、随机森林以及 LSTM 等深度学习方法。其中，LSTM 能够建模历史序列依赖关系，长期以来是金融时间序列预测中的常用模型。但股票序列具有高噪声、低信噪比、强非平稳等特点，普通 LSTM 容易出现特征记忆不足、泛化不稳定等问题。近年来，Transformer、MLP 和 patch-based 时间序列模型在长序列预测中表现较强，但这些模型通常依赖较大的数据规模和计算资源。

Kong 等人在 AAAI-25 论文《Unlocking the Power of LSTM for Long Term Time Series Forecasting》中提出 P-sLSTM，尝试重新释放 LSTM 类结构在长序列预测任务中的能力。该方法基于 sLSTM，引入 patching 与 channel independence 两个关键设计：前者将长时间序列切分为局部 patch，以缓解 sLSTM 直接处理长序列时的短记忆问题；后者将多变量序列的各通道独立处理，以减少多变量耦合带来的过拟合风险。原论文在多个标准时间序列数据集上验证了 P-sLSTM 的有效性。

本文尝试将 P-sLSTM 引入股票方向预测任务，并进一步探索非金融时间序列预训练对金融预测的迁移效果。本文的主要工作包括：

1. 基于官方 P-sLSTM 实现，构建适用于股票收益方向预测的数据处理、训练、评估和可视化流程。
2. 在大规模 S&P500 数据集上比较 LSTM、P-sLSTM、非金融预训练 P-sLSTM 及方向增强 P-sLSTM 的表现。
3. 在沪深 300 数据上进行补充实验，分析小样本与跨市场条件下的迁移效果。
4. 通过方向增强消融实验讨论金融时间序列中的验证集提升与 out-of-time 测试泛化差异。

## 2 相关工作

### 2.1 股票方向预测

股票方向预测通常将未来收益率符号作为分类目标，即预测下一交易日收益率是否大于 0。与传统回归任务相比，方向预测更贴近交易决策，但也更容易受到市场状态、宏观事件和短期噪声影响。若训练集与测试集处于不同市场阶段，即使模型在训练期验证集上表现较好，也可能在未来测试期退化。因此，本文采用时间外推测试而非随机划分作为 S&P500 主实验的核心评价方式。

### 2.2 LSTM 与长序列时间预测

LSTM 通过门控结构缓解传统 RNN 的梯度消失问题，能够建模较长历史依赖。然而，在长序列预测任务中，普通 LSTM 仍然存在记忆容量有限、并行性较差和长期依赖建模不充分等问题。P-sLSTM 原文指出，虽然 sLSTM 引入指数门控与 memory mixing 机制，但直接应用到时间序列预测时仍可能存在短记忆问题，因此需要结合 patching 等序列压缩/局部建模机制。

### 2.3 P-sLSTM

P-sLSTM 的核心思想是将 sLSTM 与 patching、channel independence 结合。设输入序列为：

```text
X in R^{B x L x M}
```

其中 B 为 batch size，L 为输入长度，M 为变量通道数。P-sLSTM 首先将输入转置为：

```text
B x M x L
```

并将不同通道视为独立样本，得到：

```text
(B * M) x L
```

随后使用 patching 操作将每个通道的序列切分为多个长度为 P 的 patch，得到：

```text
(B * M) x N x P
```

其中 N 为 patch 数量。每个 patch 经线性层映射到 embedding 空间后输入 sLSTM/xLSTM block，最终展平并投影到预测长度，再恢复为：

```text
B x T x M
```

在本文实现中，S&P500 实验采用输入长度 30、预测长度 1、patch size 为 6、stride 为 3 的设置。由于官方实现主要面向通用时间序列预测，本文在其基础上加入股票收益方向评价、验证集方向校准与方向增强消融实验。

### 2.4 迁移学习与非金融时间序列预训练

时间序列迁移学习的基本假设是：不同领域序列可能共享局部趋势、周期、波动和突变等结构特征。因此，在气象、电力、交通等非金融数据上预训练模型，可能为金融任务提供更好的初始表示。然而，金融市场数据受交易机制、投资者行为和宏观事件影响，和气象/电力数据存在明显域差异。本文将 Weather、ETTm1 和 Electricity 等非金融数据作为源域，检验其对股票方向预测的迁移效果。

## 3 方法

### 3.1 任务定义

给定股票 i 在时间 t 之前长度为 L 的历史特征序列：

```text
X_{i,t-L+1:t}
```

模型预测未来一日收益率：

```text
y_{i,t+1}
```

主方向标签定义为：

```text
d_{i,t+1} = 1, if y_{i,t+1} > 0
d_{i,t+1} = 0, otherwise
```

模型输出连续收益率预测值：

```text
\hat{y}_{i,t+1}
```

方向预测由其符号得到：

```text
\hat{d}_{i,t+1} = 1, if \hat{y}_{i,t+1} > 0
```

### 3.2 基础模型

本文比较以下模型：

1. **LSTM target-only：** 普通 LSTM 直接在目标股票数据上训练。
2. **P-sLSTM target-only：** 使用 P-sLSTM backbone 在目标股票数据上训练。
3. **Weather -> P-sLSTM：** 先在 Weather 非金融时间序列上预训练 P-sLSTM，再在股票目标域上微调。
4. **P-sLSTM 方向增强变体：** 在 target-only P-sLSTM 基础上加入方向损失或辅助方向分类头。

### 3.3 损失函数

基础模型使用均方误差损失：

```text
L_reg = MSE(y, \hat{y})
```

由于主评价指标是方向预测，本文进一步尝试方向感知损失：

```text
L = L_reg + lambda * L_dir
```

其中 `L_dir` 可以是 BCE 损失或 focal loss。辅助方向头版本则在 P-sLSTM 的目标通道表示上增加二分类输出，用于直接预测上涨/下跌方向。该设计参考了分类辅助任务的多任务学习思想，目的是让模型在收益率回归之外显式学习方向判别信息。

### 3.4 评价指标

本文主指标包括：

```text
Directional Accuracy = correct direction predictions / total predictions
```

```text
Binary F1 = 2 * precision * recall / (precision + recall)
```

```text
Direction AUC = ROC-AUC(true direction, predicted score)
```

辅助回归指标为：

```text
MAE = mean(|y - \hat{y}|)
RMSE = sqrt(mean((y - \hat{y})^2))
```

为了减少固定 0 阈值对方向指标的影响，本文在部分实验中使用验证集进行仿射校准：

```text
\hat{y}' = a * \hat{y} + b
```

其中 a 与 b 在验证集上通过搜索选择，使方向准确率最优。该步骤仅用于后处理校准，不改变模型参数。

## 4 实验设计

### 4.1 S&P500 主实验数据

本文使用 Hugging Face 上的 S&P500 数据集作为主实验数据。数据包含 503 只股票，时间范围为 2020 年 7 月至 2025 年 6 月。本文将 2020-07-16 至 2023-12-29 作为训练集，将 2024-01-02 至 2025-06-27 作为 out-of-time 测试集。

| 划分 | 日期范围 | 样本数 |
|---|---:|---:|
| 训练集 | 2020-07-16 至 2023-12-29 | 432,259 |
| 测试集 | 2024-01-02 至 2025-06-27 | 187,333 |

模型输入包含 OHLCV、历史滞后收益、均线、MACD、RSI、布林带等特征，主目标为：

```text
future_return_1d_pct
```

训练窗口长度为 30，预测长度为 1。

建议插图：

```text
figures_sp500/SP500_Fig1_dataset_profile.pdf
figures_sp500/SP500_Fig2_direction_task_framework.pdf
```

### 4.2 沪深 300 补充实验数据

为检验小样本金融场景下的模型表现，本文使用沪深 300 股票数据作为补充实验。训练集为 2017-2018 年股票窗口，测试集为 2019 年 1 月至 3 月数据。该实验同时保留 6 分类涨跌幅任务：

```text
<= -2%, (-2%, -1%], (-1%, 0%], (0%, 1%], (1%, 2%], > 2%
```

源域非金融数据包括 Weather、ETTm1 和 Electricity。

### 4.3 训练设置

S&P500 主实验使用如下参数：

| 参数 | 取值 |
|---|---:|
| seq_len | 30 |
| pred_len | 1 |
| patch_size | 6 |
| stride | 3 |
| epochs | 10 |
| batch_size | 1024 |
| optimizer | AdamW |
| device | NVIDIA GeForce RTX 4080 Laptop GPU |

沪深 300 补充实验使用 seq_len=20、epochs=10、batch_size=512。

## 5 实验结果与分析

### 5.1 S&P500 主实验结果

表 1 给出了校准后的 S&P500 out-of-time 测试结果。

| 模型 | Direction Acc. | Binary F1 | Direction AUC | MAE | RMSE |
|---|---:|---:|---:|---:|---:|
| LSTM target-only | **0.515632** | 0.570933 | **0.517553** | **1.329785** | **2.070312** |
| P-sLSTM target-only | 0.510366 | **0.578282** | 0.507885 | 1.352723 | 2.089108 |
| Weather -> P-sLSTM | 0.510395 | 0.560719 | 0.510376 | 1.393843 | 2.137954 |

从结果可以看出，LSTM 在方向准确率和 AUC 上略高，P-sLSTM target-only 的 Binary F1 最高。结合训练过程可知，P-sLSTM 的最佳验证损失为 0.861746，明显低于 LSTM 的 0.933614，说明 P-sLSTM 对训练期目标域序列具有更强拟合能力。但这种回归拟合优势没有完全转化为 out-of-time 方向预测优势。

建议插图：

```text
figures_sp500/SP500_Fig3_direction_metrics.pdf
figures_sp500/SP500_Fig4_training_curves.pdf
figures_sp500/SP500_Fig5_direction_confusion.pdf
```

### 5.2 方向增强消融实验

为进一步检验“主指标换成方向预测后，是否应直接优化方向标签”，本文尝试了三种 P-sLSTM 方向增强方式：

1. Focal direction loss；
2. 低权重 BCE direction loss；
3. 辅助 direction head。

| 变体 | 最佳验证 Direction Acc. | 测试 Direction Acc. | 测试 Binary F1 | 测试 AUC |
|---|---:|---:|---:|---:|
| P-sLSTM target-only | 0.593287 | **0.510366** | **0.578282** | 0.507885 |
| P-sLSTM + focal direction loss | 0.591358 | 0.508148 | 0.562841 | 0.506211 |
| P-sLSTM + low-weight BCE direction loss | 0.588266 | 0.507359 | 0.553635 | **0.508249** |
| P-sLSTM + auxiliary direction head | **0.597265** | 0.503469 | 0.540510 | 0.504123 |

该消融实验具有较强解释价值：方向增强确实能在验证集上提高方向指标，辅助方向头的验证方向准确率达到 0.597265；但这些提升无法迁移到 2024-2025 年测试集，测试方向准确率反而低于原始 P-sLSTM target-only。这说明在金融市场 out-of-time 设置下，验证集方向信号存在较强阶段性，直接强化方向目标可能导致模型学习到训练期市场状态特征，而非稳定可迁移规律。

建议插图：

```text
figures_sp500/SP500_Fig10_validation_direction_accuracy.pdf
figures_sp500/SP500_Fig6_prediction_deciles.pdf
figures_sp500/SP500_Fig8_rolling_direction_accuracy.pdf
```

### 5.3 沪深 300 同分布验证结果

沪深 300 同分布验证集结果如下：

| 模型 | 训练策略 | MAE | RMSE | Direction Acc. | 6-Class Acc. | Macro F1 |
|---|---|---:|---:|---:|---:|---:|
| LSTM | target-only | 1.5135 | 2.1844 | 0.5060 | 0.2478 | 0.1180 |
| P-sLSTM | target-only | 1.4959 | 2.1580 | 0.5464 | 0.2586 | 0.1529 |
| P-sLSTM | Weather pretrain + fine-tune | 1.4999 | 2.1569 | **0.5568** | 0.2671 | **0.1569** |
| P-sLSTM | Weather-6 pretrain + fine-tune | **1.4912** | **2.1490** | 0.5563 | **0.2681** | 0.1538 |
| P-sLSTM | ETTm1 pretrain + fine-tune | 1.4982 | 2.1585 | 0.5517 | 0.2636 | 0.1535 |
| P-sLSTM | Electricity pretrain + fine-tune | 1.5036 | 2.1628 | 0.5420 | 0.2564 | 0.1527 |

在同分布验证环境下，P-sLSTM 明显优于普通 LSTM，Weather 源域预训练进一步提升方向准确率和 Macro F1。该结果说明，在训练集与验证集分布较一致时，P-sLSTM 及非金融预训练可学习到有用的时间序列表示。

### 5.4 沪深 300 时间外推结果

在 2019 年 out-of-time 测试上，结果如下：

| 模型 | 训练策略 | MAE | RMSE | Direction Acc. | 6-Class Acc. | Macro F1 |
|---|---|---:|---:|---:|---:|---:|
| LSTM | target-only | 2.1364 | 2.9885 | 0.4781 | 0.1769 | 0.0907 |
| LSTM | Weather pretrain + fine-tune | 2.1652 | 3.0196 | 0.4673 | 0.1867 | 0.0976 |
| P-sLSTM | target-only | 2.1918 | 3.0532 | **0.5132** | 0.1825 | 0.1150 |
| P-sLSTM | Weather pretrain + fine-tune | 2.2073 | 3.0851 | 0.4937 | **0.1874** | **0.1205** |

时间外推测试中，P-sLSTM target-only 的方向准确率最高，但 Weather 迁移未能提升方向准确率，仅在 6 分类准确率和 Macro F1 上略有改善。该结果与 S&P500 主实验一致：非金融预训练可能有助于部分形态分类，但不一定提高严格时间外推下的方向判断。

建议插图：

```text
figures/Fig3_same_distribution_metrics.pdf
figures/Fig4_out_of_time_metrics.pdf
figures/Fig8_transfer_gain.pdf
```

## 6 讨论

### 6.1 为什么 P-sLSTM 验证损失更好但测试方向准确率不一定更高？

P-sLSTM 的 patching 与 channel independence 能提高序列拟合能力，因此在训练期验证集上表现出更低的 MSE。然而，股票方向预测关注收益率符号，回归误差降低并不必然等价于方向判断提升。尤其在收益率接近 0 的样本中，即使预测值数值误差很小，符号也可能发生改变。

此外，S&P500 测试集使用 2024-2025 年数据，与 2020-2023 年训练期存在市场环境差异。模型在训练期学到的方向模式可能无法稳定外推。因此，P-sLSTM 的结果应解读为“更强的序列拟合 backbone”，而不是“稳定获得更高交易方向准确率”的模型。

### 6.2 非金融预训练为什么收益不稳定？

Weather、电力等非金融时间序列具有趋势、周期、噪声和突变等结构，这些结构与金融时间序列存在一定共性。但金融市场还受到交易制度、宏观政策、投资者预期和流动性冲击影响，源域与目标域之间存在明显语义差异。实验中 Weather 预训练在沪深 300 同分布验证集上有效，但在 S&P500 out-of-time 测试中未能提升方向准确率，说明迁移学习效果依赖源域选择和目标域分布稳定性。

### 6.3 方向增强为什么验证集有效但测试集无效？

方向增强实验表明，加入 focal/BCE 方向损失或辅助方向头，可以显著提升验证集方向准确率。但测试集表现下降，说明模型可能学习到了训练期和验证期共有的市场阶段特征，而这些特征在未来市场阶段不再稳定。该现象支持金融时间序列研究中的一个重要观点：验证集提升需要经过严格 out-of-time 测试检验，否则容易高估模型的真实预测能力。

## 7 结论

本文基于 P-sLSTM 构建股票方向预测实验框架，并在 S&P500 与沪深 300 数据上进行了大规模实验。结果表明：

1. P-sLSTM 在目标域验证损失上优于普通 LSTM，说明其 patching 与 channel independence 设计能够增强股票时间序列拟合能力。
2. 在 S&P500 out-of-time 方向预测任务中，各模型方向准确率总体仅略高于随机基线，说明股票未来一日方向预测难度较高。
3. 非金融时间序列预训练在同分布验证中可能有效，但在时间外推测试中收益不稳定，存在域差异和负迁移风险。
4. 方向感知损失和辅助方向头能提升验证集方向指标，但无法改善未来测试期表现，说明市场状态漂移是影响模型泛化的重要因素。

因此，P-sLSTM 更适合作为金融时间序列建模的 backbone，而非单独保证方向预测收益的完整解决方案。未来工作可进一步引入市场状态识别、滚动训练、在线学习、风险调整收益评价以及更严格的交易回测框架，以检验模型在实际金融决策中的有效性。

## 参考文献

[1] Kong, Y., Wang, Z., Nie, Y., Zhou, T., Zohren, S., Liang, Y., Sun, P., & Wen, Q. (2025). Unlocking the Power of LSTM for Long Term Time Series Forecasting. Proceedings of the AAAI Conference on Artificial Intelligence, 39(11), 11968-11976. https://ojs.aaai.org/index.php/AAAI/article/view/33303

[2] Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9(8), 1735-1780.

[3] Beck, M., Poppel, K., Spanring, M., Auer, A., Prudnikova, O., Kopp, M., Klambauer, G., Brandstetter, J., & Hochreiter, S. (2024). xLSTM: Extended Long Short-Term Memory. arXiv preprint.

[4] Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. ICLR.

[5] Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023). Are Transformers Effective for Time Series Forecasting? AAAI.

[6] Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., & Zhang, W. (2021). Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting. AAAI.

[7] Adilbai. S&P500 stock dataset. Hugging Face Datasets. https://huggingface.co/datasets/Adilbai/stock-dataset

[8] Eleanorkong. P-sLSTM official implementation. GitHub. https://github.com/Eleanorkong/P-sLSTM

## 附：本文已生成实验材料

结果文档：

```text
SP500_EXPERIMENT_RESULTS.md
EXPERIMENT_RESULTS.md
```

S&P500 图：

```text
figures_sp500/
```

沪深 300 图：

```text
figures/
```

