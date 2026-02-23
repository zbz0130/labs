

### (a) Copying in attention（注意力的复制操作）

**i. 条件描述：**
当查询向量 $q$ 与某个特定的键向量 $k_j$ 的内积远大于它与其他所有键向量的内积时（即 $k_j^T q \gg k_i^T q, \forall i \neq j$），分类分布 $\alpha$ 会将几乎所有的权重集中在 $\alpha_j$ 上。

**ii. 输出 $c$ 的描述：**
在这种条件下，输出向量 $c$ 将近似等于该位置对应的值向量 $v_j$（即 $c \approx v_j$），从而实现了将输入序列中的某个特定信息“复制”到输出的操作。

---

### (b) An average of two（两个向量的平均）

**表达式设计：**
取查询向量 $q = \beta(k_a + k_b)$，其中 $\beta$ 是一个足够大的正数常数。

**理由（Justification）：**
1.  **计算内积：** 根据题目给出的正交单位性条件（$k_i^T k_j = \delta_{ij}$）：
    *   对于 $k_a$：$k_a^T q = \beta(k_a^T k_a + k_a^T k_b) = \beta(1 + 0) = \beta$。
    *   对于 $k_b$：$k_b^T q = \beta(k_b^T k_a + k_b^T k_b) = \beta(0 + 1) = \beta$。
    *   对于其他 $k_i (i \neq a, b)$：$k_i^T q = \beta(k_i^T k_a + k_i^T k_b) = 0$。
2.  **Softmax权重：**
    *   $\alpha_a = \alpha_b = \frac{\exp(\beta)}{\exp(\beta) + \exp(\beta) + (n-2)\exp(0)} = \frac{\exp(\beta)}{2\exp(\beta) + n-2}$。
    *   当 $\beta$ 足够大时，$\exp(\beta)$ 远大于 $n-2$，因此 $\alpha_a \approx \alpha_b \approx \frac{1}{2}$。
    *   此时其他 $\alpha_i \approx 0$。
3.  **结果：** 输出 $c = \sum \alpha_i v_i \approx \frac{1}{2}v_a + \frac{1}{2}v_b$。

---

### (c) Drawbacks of single-headed attention（单头注意力的缺陷）

**i. 设计 $q$ 使得 $c \approx \frac{1}{2}(v_a + v_b)$：**
**设计：** $q = \beta(\mu_a + \mu_b)$，其中 $\beta$ 为较大常数。
**理由：** 因为协方差 $\Sigma_i = \alpha I$ 趋于 0，键向量 $k_i$ 几乎等同于其均值 $\mu_i$。由于 $\mu_i$ 是相互垂直且模长为 1 的，其逻辑与 (b) 部分完全一致。$q$ 在 $\mu_a$ 和 $\mu_b$ 方向上的投影相等且最大，导致权重平分在 $a$ 和 $b$ 两个位置。

**ii. 扰动分析（关于单头注意力的不稳定性）：**
**核心问题：** 单头注意力在面对“模长方差大”的噪声时非常脆弱。
1.  **方差的影响：** 题目设定 $\Sigma_a$ 在 $\mu_a$ 方向上有很大的方差。这意味着即使 $k_a$ 的方向大致指向 $\mu_a$，它的**长度（模长）**会有剧烈波动。
2.  **不稳定性：** 在单头注意力中，$k_a^T q$ 的值会随着 $k_a$ 长度的随机波动而剧烈变化。如果某次采样中 $k_a$ 变得很长，那么 $\alpha_a$ 会瞬间由于 Softmax 的特性占据主导（趋近于 1），导致原本想要实现的“平均 $v_a$ 和 $v_b$”的目标失败，输出只剩下 $v_a$。
3.  **多头的优势：** 这解释了为什么需要**多头注意力**。在多头机制下，即使一个头被某个噪声极大的键向量（如 $k_a$）“带偏”了，其他的头仍然可以捕捉到 $k_b$ 的信息，从而保证了模型在处理现实世界中有噪声的数据时的鲁棒性（Robustness）。
这是一份来自斯坦福大学经典自然语言处理课程 **CS 224N** 的作业。题目探讨了单头自注意力机制在面对“键向量模长扰动”时的不稳定性，从而引出多头注意力的必要性。



在不同样本中，输出向量 $c$ 将不再稳定地等于两个向量的平均值（即不再是 $c \approx \frac{1}{2}v_a + \frac{1}{2}v_b$），而是会在 $v_a$ 和 $v_b$ 之间**剧烈摆动（jump/flicker）**。在大多数采样实例中，$c$ 会表现为“赢家通吃”的状态，要么几乎完全等于 $v_a$，要么几乎完全等于 $v_b$。


*   **第 (i) 小题的情况：** 所有的 $k_i$ 模长都非常接近 1，因此 $k_a^\top q$ 和 $k_b^\top q$ 几乎相等，Softmax 分配到的权重是平衡的（各 0.5）。
*   **本小题的情况（扰动）：**
    *   由于 $\Sigma_a$ 在 $\mu_a$ 方向上具有大方差，$k_a$ 的采样结果可能是 $(1+\epsilon)\mu_a$，其中 $\epsilon$ 是一个较大的随机扰动。
    *   计算内积：$k_a^\top q \approx \beta(1+\epsilon)$，而 $k_b^\top q \approx \beta$。
    *   **Softmax 的放大效应：** Softmax 函数对内积的微小差异非常敏感（因为是指数级放大）。如果 $\epsilon > 0$（即 $k_a$ 稍微长一点），$\exp(\beta(1+\epsilon))$ 会远大于 $\exp(\beta)$，导致权重 $\alpha_a \to 1$，此时 $c \approx v_a$。
    *   如果 $\epsilon < 0$（即 $k_a$ 稍微短一点），则 $\alpha_b \to 1$，导致 $c \approx v_b$。

#### . 与第 (i) 小题的区别
*   **均值 vs 采样：** 虽然 $c$ 的“期望值”可能仍接近 $\frac{1}{2}(v_a + v_b)$，但在**单次采样（不同样本）**中，它几乎从不等于这个平均值。
*   **稳定性：** 第 (i) 小题中的 $c$ 是稳定的；而这里的 $c$ 具有**极大的方差（High Variance）**。

#### 4. 结论与动机（为什么需要多头）
这个现象展示了**单头注意力（Single-head attention）的局限性**：
在单头机制下，模型很难同时稳定地关注多个对象。如果其中一个对象的信号（键向量模长）因为噪声而增强，它会迅速“淹没”其他所有对象。


#### (d) 部分答案

**i. 查询向量设计：**
我们可以设计每个“头”专注于其中一个目标向量。
令 **$q_1 = \beta \mu_a$** 且 **$q_2 = \beta \mu_b$**（其中 $\beta$ 是一个足够大的正标量）。
*   **理由：** $q_1$ 只与 $k_a$ 有很大的内积，因此 $c_1 \approx v_a$；同理，$q_2$ 只与 $k_b$ 有很大内积，因此 $c_2 \approx v_b$。最终 $c = \frac{1}{2}(c_1 + c_2) \approx \frac{1}{2}(v_a + v_b)$。

**ii. 定性描述与方差分析：**
*   **预期表现：** 即使在不同样本中，输出 **$c$ 将会非常稳定（低方差）**，始终保持在 $\frac{1}{2}(v_a + v_b)$ 附近。
*   **分析：**
    *   **对于 $c_1$：** 虽然 $k_a$ 的模长有很大方差，但由于 $q_1$ 专门指向 $\mu_a$，且与其他均值正交，所以 $k_a^\top q_1$ 始终会远大于其他 $k_j^\top q_1$。正如 (a) 小题所述，这仅仅是一个稳健的“复制”操作，$c_1$ 稳定地输出 $v_a$。
    *   **对于 $c_2$：** $q_2$ 指向 $\mu_b$。由于正交性，即使 $k_a$ 的模长剧烈波动，它在 $q_2$ 方向上的投影（内积）依然接近于 0。因此，$k_b^\top q_2$ 始终占主导，$c_2$ 稳定地输出 $v_b$。
    *   **结论：** $c_1$ 和 $c_2$ 的方差都很小，因此它们的平均值 $c$ 也非常稳定。

---

#### (e) 部分总结

**多头注意力如何克服单头注意力的缺点：**

1.  **解耦与专注：** 单头注意力在试图同时关注多个目标时，必须在一个查询向量中平衡多个内积，这使得它对任何一个键向量的模长噪声（方差）都极其敏感，容易导致“赢家通吃”的不稳定现象。
2.  **鲁棒性：** 多头注意力通过允许不同的头在不同的子空间内进行**专门化（Specialization）**来解决这个问题。每个头可以独立且稳健地捕捉一个特定的信号（如“复制”一个向量），而不受其他位置噪声的干扰。
3.  **方差降低：** 通过将多个稳定头的输出进行结合（平均或拼接），模型能够以极低的方差同时整合多个来源的信息，从而在处理现实世界中有噪声的数据时表现出更强的可靠性。

---

### (a) 输入排列 (Permuting the input)

**i. (3分) 证明 $\mathbf{Z}_{\text{perm}} = \mathbf{P}\mathbf{Z}$：**

这个问题的核心是证明 Transformer 的各个组件都是置换等价的。

1.  **自注意力层 (Self-Attention)：**
    设原始输入为 $\mathbf{X}$，其生成的 Query, Key, Value 矩阵为 $\mathbf{Q}, \mathbf{K}, \mathbf{V}$。
    对于排列后的输入 $\mathbf{X}_{\text{perm}} = \mathbf{P}\mathbf{X}$，相应的矩阵变为 $\mathbf{Q}_{\text{perm}} = \mathbf{P}\mathbf{Q}, \mathbf{K}_{\text{perm}} = \mathbf{P}\mathbf{K}, \mathbf{V}_{\text{perm}} = \mathbf{P}\mathbf{V}$（因为它们只是 $\mathbf{X}$ 的线性变换）。
    注意力输出为：
    $$\mathbf{H}_{\text{perm}} = \text{softmax}\left(\frac{\mathbf{P}\mathbf{Q}(\mathbf{P}\mathbf{K})^\top}{\sqrt{d}}\right)\mathbf{P}\mathbf{V} = \text{softmax}\left(\frac{\mathbf{P}\mathbf{Q}\mathbf{K}^\top\mathbf{P}^\top}{\sqrt{d}}\right)\mathbf{P}\mathbf{V}$$
    利用题目给出的性质 $\text{softmax}(\mathbf{P}\mathbf{A}\mathbf{P}^\top) = \mathbf{P}\text{softmax}(\mathbf{A})\mathbf{P}^\top$：
    $$\mathbf{H}_{\text{perm}} = \mathbf{P}\text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}}\right)\mathbf{P}^\top\mathbf{P}\mathbf{V}$$
    由于 $\mathbf{P}$ 是置换矩阵，满足 $\mathbf{P}^\top\mathbf{P} = \mathbf{I}$，因此：
    $$\mathbf{H}_{\text{perm}} = \mathbf{P}\text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}}\right)\mathbf{V} = \mathbf{P}\mathbf{H}$$

2.  **前馈网络层 (FFN)：**
    $\mathbf{Z}_{\text{perm}} = \text{ReLU}(\mathbf{H}_{\text{perm}}\mathbf{W}_1 + \mathbf{1}b_1)\mathbf{W}_2 + \mathbf{1}b_2 = \text{ReLU}(\mathbf{P}\mathbf{H}\mathbf{W}_1 + \mathbf{1}b_1)\mathbf{W}_2 + \mathbf{1}b_2$
    注意全 1 向量满足 $\mathbf{1} = \mathbf{P}\mathbf{1}$（重新排列一堆 1 还是那堆 1），所以：
    $$\mathbf{Z}_{\text{perm}} = \text{ReLU}(\mathbf{P}(\mathbf{H}\mathbf{W}_1 + \mathbf{1}b_1))\mathbf{W}_2 + \mathbf{P}\mathbf{1}b_2$$
    利用性质 $\text{ReLU}(\mathbf{P}\mathbf{A}) = \mathbf{P}\text{ReLU}(\mathbf{A})$：
    $$\mathbf{Z}_{\text{perm}} = \mathbf{P}\text{ReLU}(\mathbf{H}\mathbf{W}_1 + \mathbf{1}b_1)\mathbf{W}_2 + \mathbf{P}\mathbf{1}b_2 = \mathbf{P}\mathbf{Z}$$
    **证明完毕。**

**ii. (1分) 说明为什么该性质在处理文本时有问题：**
该性质说明 Transformer 具有**置换等价性**。这意味着如果你打乱句子中单词的顺序，输出的向量也会以同样的方式打乱，但每个单词对应的特征向量数值本身不会改变。换句话说，Transformer **本质上将输入视为一个“词袋”（集合），而不是一个有序的序列**。在自然语言中，语序对意义至关重要（例如“狗咬人”和“人咬狗”），如果没有位置信息，模型将无法区分不同语序带来的语义差别。

---

### (b) 位置编码 (Position embeddings)

**i. (1分) 位置编码是否有助于解决 (a) 中的问题？**
**是，有帮助。** 位置编码 $\Phi$ 为序列中的每个位置提供了一个唯一的向量。通过将 $\Phi$ 加到单词嵌入 $\mathbf{X}$ 上，得到的 $\mathbf{X}_{\text{pos}} = \mathbf{X} + \Phi$ 使得相同的单词在不同的位置具有不同的数值表示。这打破了纯粹的置换等价性，使得模型能够感知并利用单词之间的顺序关系。

**ii. (1分) 两个不同位置的编码是否可能相同？**
**不，几乎不可能。** 公式中使用了不同频率的 $\sin$ 和 $\cos$ 函数（频率范围跨越了 $1$ 到 $10000$）。由于维度 $d$ 通常很大（如 512），对于两个不同的位置 $t_1 \neq t_2$，要在所有的维度上同时取得完全相同的正弦/余弦值在数学上几乎是不可能的。这种设计确保了序列中每一个时间步都有一个唯一的“位置签名”。