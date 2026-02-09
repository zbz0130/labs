## 1

### (a) 证明朴素 Softmax 损失等于交叉熵损失

**证明：**
交叉熵损失定义为 $H(\mathbf{y}, \mathbf{\hat{y}}) = -\sum_{w \in \text{Vocab}} y_w \log(\hat{y}_w)$。
已知 $\mathbf{y}$ 是一个 **One-hot 向量**，其中只有在真实上下文词 $o$ 的位置为 $1$（即 $y_o = 1$），其余位置均为 $0$（即当 $w \neq o$ 时，$y_w = 0$）。
将此性质代入求和公式：
$$H(\mathbf{y}, \mathbf{\hat{y}}) = -(0 \cdot \log(\hat{y}_{w_1}) + \dots + 1 \cdot \log(\hat{y}_o) + \dots + 0 \cdot \log(\hat{y}_{w_V})) = -\log(\hat{y}_o)$$
由此得证：$-\sum_{w \in \text{Vocab}} y_w \log(\hat{y}_w) = -\log(\hat{y}_o)$。

---

### (b) 求导与解释

#### i. 计算梯度 $\frac{\partial J}{\partial \mathbf{v}_c}$
已知 $J = -\log \hat{y}_o = -\mathbf{u}_o^\top \mathbf{v}_c + \log \left( \sum_{k \in \text{Vocab}} \exp(\mathbf{u}_k^\top \mathbf{v}_c) \right)$。
对 $\mathbf{v}_c$ 求偏导：
1. 第一项：$\frac{\partial}{\partial \mathbf{v}_c} (-\mathbf{u}_o^\top \mathbf{v}_c) = -\mathbf{u}_o$。
2. 第二项（链式法则）：$\frac{\partial}{\partial \mathbf{v}_c} \log(\dots) = \frac{1}{\sum \exp(\mathbf{u}_k^\top \mathbf{v}_c)} \cdot \sum \left( \exp(\mathbf{u}_k^\top \mathbf{v}_c) \cdot \mathbf{u}_k \right) = \sum_{k \in \text{Vocab}} \hat{y}_k \mathbf{u}_k$。

合并结果得：$\frac{\partial J}{\partial \mathbf{v}_c} = -\mathbf{u}_o + \sum_{k \in \text{Vocab}} \hat{y}_k \mathbf{u}_k$。
利用矩阵 $\mathbf{U}$（各列为 $\mathbf{u}_k$）和向量 $\mathbf{y}, \mathbf{\hat{y}}$ 表示，注意到 $\mathbf{u}_o = \mathbf{U}\mathbf{y}$ 且 $\sum \hat{y}_k \mathbf{u}_k = \mathbf{U}\mathbf{\hat{y}}$：
**最终向量化答案：**
$$\frac{\partial J}{\partial \mathbf{v}_c} = \mathbf{U}(\mathbf{\hat{y}} - \mathbf{y})$$
*(注：若 $\mathbf{v}_c$ 为 $d \times 1$，$\mathbf{U}$ 为 $d \times V$，$\mathbf{\hat{y}}-\mathbf{y}$ 为 $V \times 1$，则结果形状为 $d \times 1$，符合要求)*

#### ii. 梯度何时为零？
数学方程为：
$$\mathbf{\hat{y}} = \mathbf{y}$$
**解释：** 当模型的预测分布 $\mathbf{\hat{y}}$ 与真实的 One-hot 分布 $\mathbf{y}$ 完全一致时（即模型以 100% 的概率预测出正确的上下文词 $o$），梯度为零。

#### iii. 梯度项的物理解释
在梯度下降中，更新公式为 $\mathbf{v}_c^{\text{new}} = \mathbf{v}_c - \alpha \frac{\partial J}{\partial \mathbf{v}_c} = \mathbf{v}_c + \alpha(\mathbf{u}_o - \sum_{w \in \text{Vocab}} \hat{y}_w \mathbf{u}_w)$。

*   **第一项 $\mathbf{u}_o$（观测项）：**
    这一项是正向的。在更新时，它将中心词向量 $\mathbf{v}_c$ 推向真实出现的上下文词向量 $\mathbf{u}_o$。这增加了它们之间的相似度（点积），从而提高了模型预测出正确单词的概率。
*   **第二项 $\sum \hat{y}_w \mathbf{u}_w$（预测期望项）：**
    这一项是负向的。它是模型当前预测的所有上下文词向量的加权平均（期望）。更新时，模型会将 $\mathbf{v}_c$ 从这个“期望中心”推开。这起到了一种“去中心化”或“压制”的作用，防止模型对所有单词都给出高概率，确保概率分布的归一化。

**简而言之：** 梯度下降的过程就是让 $\mathbf{v}_c$ **靠近**实际看到的词 $o$，同时**远离**模型错误地认为可能出现的那些词。

### (c)


### 1. 数学推导（响应 Hint）
首先，根据题目给出的提示（Hint），我们要看看归一化对两个共线向量的影响。
假设有两个词 $x$ 和 $y$，它们的词向量满足关系 $\mathbf{u}_x = \alpha \mathbf{u}_y$，其中 $\alpha$ 是一个大于 0 的标量（这意味着两个向量方向相同，但长度不同）。

当我们对它们进行 L2 归一化时：
*   **对于 $\mathbf{u}_y$：**
    $$ \text{norm}(\mathbf{u}_y) = \frac{\mathbf{u}_y}{\|\mathbf{u}_y\|_2} $$
*   **对于 $\mathbf{u}_x$：**
    $$ \text{norm}(\mathbf{u}_x) = \frac{\mathbf{u}_x}{\|\mathbf{u}_x\|_2} = \frac{\alpha \mathbf{u}_y}{\|\alpha \mathbf{u}_y\|_2} = \frac{\alpha \mathbf{u}_y}{|\alpha| \|\mathbf{u}_y\|_2} $$
    因为 $\alpha > 0$，所以 $|\alpha| = \alpha$，分子分母消去 $\alpha$，得到：
    $$ \text{norm}(\mathbf{u}_x) = \frac{\mathbf{u}_y}{\|\mathbf{u}_y\|_2} $$

**结论：** $\text{norm}(\mathbf{u}_x) = \text{norm}(\mathbf{u}_y)$。
这说明：**L2 归一化会抹除向量的长度（Magnitude）信息，只保留方向（Direction）信息。**

---


#### **(1) 什么时候归一化会丢失有用信息（Take away useful information）？**
当词向量的**长度（Magnitude）代表了某种语义强度或重要性**时，归一化会丢失信息。

*   **情感分析中的程度差异（Intensity）：**
    举个例子，词语 "good"（好）和 "excellent"（极好）可能在向量空间中指向非常相似的方向（都是正向情感），但 "excellent" 的向量长度可能比 "good" 长，以表示更强烈的情感。
    如果我们把它们相加来进行分类（如题目假设的那样）：
    *   未归一化：`vector("excellent")` 会比 `vector("good")` 对总和向量贡献更大的正向值。
    *   归一化后：两者变得完全一样。模型就无法区分“稍微好一点”和“非常棒”的区别了。

#### **(2) 什么时候归一化不会丢失有用信息（When would it not）？**
当任务主要依赖于词的**语义类别或方向**，而长度只是噪音时，归一化不会造成负面影响（甚至可能有帮助）。

*   **基于主题的分类（Topic Classification）：**
    如果任务是判断一个短语是关于“体育”还是“政治”。"Ball" 和 "Stadium" 指向体育的方向，"Vote" 和 "Election" 指向政治的方向。在这种情况下，我们只关心向量指向哪里（方向），而不关心向量有多长。
*   **消除频率偏差：**
    在某些训练方法中，出现频率极高的词（如停用词）或者极低的词可能会导致向量长度异常（过长或过短），但这并不代表它们的语义强度。此时归一化可以让所有词在平等的权重下参与运算，反而能提升效果。


---

### 2. 神经网络优化 (8 分)

#### (a) Adam 优化器 (Adam Optimizer)

**i. (2 分) 解释动量 $m$ 的直觉以及为什么低方差有助于学习：**
*   **直觉：** $m$ 是梯度的滚动平均（类似物理中的惯性）。如果梯度的方向频繁变化（即噪声很大），求平均会抵消掉那些不一致方向的分量，而加强一致方向的分量。这使得更新路径更加平滑，减少了在陡峭峡谷中的剧烈震荡。
*   **帮助：** 较低的方差意味着更新方向更加稳定可靠。这使得学习过程能够以更快的速度朝着最小值移动，而不会因为单个小批量数据的噪声而偏离正确方向，从而加速收敛并提高稳定性。

**ii. (2 分) 自适应学习率：哪些参数会获得更大的更新？为什么这有助于学习？**
*   **谁获得更大更新：** 那些**梯度模长较小**（即 $v$ 的值较小）的参数会获得更大的更新。
*   **为什么有帮助：** 在训练中，有些参数（特征）可能很少出现，导致其梯度长期处于较低水平。通过除以 $\sqrt{v}$，Adam 为这些“冷门”参数提供了更大的有效学习率，让它们有机会更快地学习。同时，对于那些梯度巨大且不稳定的参数，它会通过减小步伐来防止步子迈得太大而跳过最优解。这实现了在所有维度上更均衡的优化。

---

#### (b) Dropout 随机失活

**i. (2 分) $\gamma$ 应该等于什么？给出数学推导。**
*   **答案：** $\gamma = \frac{1}{1 - p_{\text{drop}}}$
*   **推导：**
    根据定义：$E_{p_{\text{drop}}}[h_{\text{drop}}]_i = E[\gamma d_i h_i] = \gamma h_i E[d_i]$。
    掩码 $d_i$ 取值为 0 的概率是 $p_{\text{drop}}$，取值为 1 的概率是 $1 - p_{\text{drop}}$。
    所以 $E[d_i] = 0 \cdot p_{\text{drop}} + 1 \cdot (1 - p_{\text{drop}}) = 1 - p_{\text{drop}}$。
    为了使期望值等于原始值 $h_i$，我们需要：
    $\gamma h_i (1 - p_{\text{drop}}) = h_i \Rightarrow \gamma (1 - p_{\text{drop}}) = 1 \Rightarrow \gamma = \frac{1}{1 - p_{\text{drop}}}$。

**ii. (2 分) 为什么 Dropout 用于训练而非评估？**
*   **训练时使用：** Dropout 随机关闭神经元，防止神经元之间产生过度的“共适应”（co-adaptation），即一个神经元依赖于其他特定神经元的存在。这迫使每个神经元学习更鲁棒、更具代表性的特征，从而起到正则化作用，减少过拟合。
*   **评估时不使用：** 在评估阶段，我们希望利用模型学习到的完整知识来做出最准确的预测。关闭神经元会引入不必要的噪声并降低确定性。通过在训练时使用 $\gamma$ 进行缩放，我们在评估阶段无需额外操作就能保证各层激活值的规模与训练时一致，同时发挥出整个集成网络（Ensemble）的威力。