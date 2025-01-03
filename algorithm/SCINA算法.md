# SCINA（**Semi-supervised Category Identification and Assignment**）算法

## 标记定义

正态化和对数转换后的表达矩阵$E_{i,j}$，其中$i(i=1..I)$为基因代号；$j(j=1..J)$为细胞代号

待分配的细胞类型代号$r=1..R$，每一个细胞均有且只有一个细胞类型

对应每个细胞类型的基因特征$S_r$，基因特征之间不鼓励重叠

每个细胞对应的真实细胞类型命名为$z_j = 0,1,...R$，后文为了简略可以用$s$代表

## 算法目标

根据签名基因确定每个轮廓细胞j的细胞类型。默认情况下，假设基因签名被激活。与其他细胞类型相比，在一种细胞类型中具有特征性低表达的特征基因的表达可以被倒置，因此该基因在该细胞类型中的伪表达量较高。每个细胞，j，都属于R细胞类型之一或其他细胞类型（新的未知细胞类型）（zj0）。每一种类型的细胞，r，都以特征基因，Sr的激活来标记。它可以假定未知的细胞类型没有激活任何基因特征。

## 模型假设

1. 假设所有标记基因表达量符合双峰分布，即：
   $$
   (\overrightarrow{e_{S_r,j}}|z_j = r) \sim N(\overrightarrow{\mu_{r,1}}, \Sigma^{r}_{1})\\
   (\overrightarrow{e_{S_r,j}}|z_j \neq r) \sim N(\overrightarrow{\mu_{r,2}}, \Sigma^{r}_{2})
   $$
   其中$\overrightarrow{\mu_{r,1}}$和$\overrightarrow{\mu_{r,2}}$是由所有细胞对于每个基因的平均值，向量长度为细胞类型为$r$的标记基因向量$S_r$的长度，并且

   令均值向量$\overrightarrow{\mu_{r,1}}$的每个元素均大于$\overrightarrow{\mu_{r,2}}$中对应位次的元素（此处将r类型细胞在r类型mark基因遵循的分布称之为**一类突出分布**，对于其他细胞在r类型mark基因上遵循的分布称之为**二类平凡分布**）；

   令协方差矩阵$\Sigma^r_1$和$\Sigma^r_2$（均为$J \times J$的对角矩阵）为对角矩阵（即主对角线其他元素均为0——这个假设事实上认为每个细胞的表达量向量是独立同正态分布的，其目的是降低了估计问题的维数。也可以用非对角线项来指定其他的协方差结构。但对于这种情况，可能需要正则化技术，特别是当签名基因的数量比可用的细胞数量较大时）

   **note：**对于一系列随机变量$X_i,i=1..n$其协方差矩阵$Cov=\{c_{ij}|i,j=1..n\}，其中c_{ij}=E\{[X_i-E(X_i)][X_j-E(X_j)]\}$

2. 进一步有：
   $$
   如果z_j=s,s>0,则有\overrightarrow{e_{S_s,j}} \sim N(\overrightarrow{\mu_{s,1}}, \Sigma^{s}_{1})和\overrightarrow{e_{S_r,j}} \sim N(\overrightarrow{\mu_{r,2}}, \Sigma^{s}_{2})\\
   如果z_j=0,则有\overrightarrow{e_{S_s,j}} \sim N(\overrightarrow{\mu_{s,2}}, \Sigma^{s}_{2})，其中r=1..R
   $$
   也就是说对于j细胞而言，如果其真实类型为s，那么该细胞所有s类型对应的mark基因均遵循一类突出分布，其余的基因均遵循二类平凡分布

## E-M算法

该算法采取极大似然估计的原理，即**在已知观测数据的情况下，找到一组参数值，使得这些数据出现的概率最大**

### 似然函数

令$P(z_j=r)=\tau_r,其中r=0..R,\sum\limits_{r=0}^{R} \tau_r = 1$

需要估计的参数包括$\Theta=(\tau_r,\overrightarrow{\mu_{r,1}},\overrightarrow{\mu_{r,2}},\Sigma^r_1,\Sigma^r_2)$，其中所有的$r=1..R$，这里参数依次是细胞类型r对应的概率向量，一类突出分布和二类平凡分布的均值$\mu_r$，一类突出分布和二类平凡分布的协方差矩阵$\Sigma_r$

$$
L(\Theta; E) = \prod\limits_{j=1}^{J}\{\tau_0 \prod\limits_{r=1}^{R}{f(\overrightarrow{e_{S_r,j}};\overrightarrow{\mu_{r,2}}, \Sigma_{2}^{r})}+\sum\limits_{s=1}^{R}[\tau_s f(\overrightarrow{e_{S_s,j}};\overrightarrow{\mu_{s,1}},\Sigma_1^s)\prod\limits_{r=1,r\neq s}^R{f(\overrightarrow{e_{S_r,j}};\overrightarrow{\mu_{r,2}},\Sigma_2^r)}]\}
$$
**note：**函数里的分号标记用于分隔随机变量和参数，例如$P(X;\theta)$表示给定参数$\theta$时，随机变量$X$的概率分布

极大似然函数里的函数$f$应该是概率密度函数，其分布由平均值和协方差决定，这里的似然函数事实上是基于测序结果的每一细胞类型的概率密度函数的期望。

### E步骤

对于类型为未知类型的概率：
$$
\begin{align}
P^t(z_j=0|E;\Theta^t) &= \frac{\tau_0^t \prod\limits_{r=1}^{R}{f(\overrightarrow{e_{S_r,j}};\overrightarrow{\mu_{r,2}^t}, \Sigma_{2}^{r,t})}}{\tau_0^t \prod\limits_{r=1}^{R}{f(\overrightarrow{e_{S_r,j}};\overrightarrow{\mu_{r,2}^t}, \Sigma_{2}^{r,t})}+\sum\limits_{s=1}^{R}[\tau_s f(\overrightarrow{e_{S_s,j}};\overrightarrow{\mu_{s,1}^t},\Sigma_1^{s,t})\prod\limits_{r=1,r\neq s}^R{f(\overrightarrow{e_{S_r,j}};\overrightarrow{\mu_{r,2}^t},\Sigma_2^{r,t})}]}\\
&= \frac{\tau_0^t}{\tau_0^t+\sum\limits_{s=1}^{R}{[\tau_s^t\frac{f(\overrightarrow{e_{S_s,j}};\overrightarrow{\mu^t_{s,1}},\Sigma^{s,t}_1)}{f(\overrightarrow{e_{S_s,j}};\overrightarrow{\mu^t_{s,2}},\Sigma^{s,t}_2)}]}}
\end{align}
$$
对于类型为已知类型$s_0$的概率
$$
\begin{align}
P^t(z_j=0|E;\Theta^t) &= \frac{\tau_0^t \prod\limits_{r=1}^{R}{f(\overrightarrow{e_{S_r,j}};\overrightarrow{\mu_{r,2}^t}, \Sigma_{2}^{r,t})}}{\tau_0^t \prod\limits_{r=1}^{R}{f(\overrightarrow{e_{S_r,j}};\overrightarrow{\mu_{r,2}^t}, \Sigma_{2}^{r,t})}+\sum\limits_{s=1}^{R}[\tau_s f(\overrightarrow{e_{S_s,j}};\overrightarrow{\mu_{s,1}^t},\Sigma_1^{s,t})\prod\limits_{r=1,r\neq s}^R{f(\overrightarrow{e_{S_r,j}};\overrightarrow{\mu_{r,2}^t},\Sigma_2^{r,t})}]}\\
&= \frac{\tau_{s_0}^t\frac{f(\overrightarrow{e_{S_{s_0},j}};\overrightarrow{\mu^t_{{s_0},1}},\Sigma^{{s_0},t}_1)}{f(\overrightarrow{e_{S_{s_0},j}};\overrightarrow{\mu^t_{{s_0},2}},\Sigma^{{s_0},t}_2)}}{\tau_0^t+\sum\limits_{s=1}^{R}{[\tau_s^t\frac{f(\overrightarrow{e_{S_s,j}};\overrightarrow{\mu^t_{s,1}},\Sigma^{s,t}_1)}{f(\overrightarrow{e_{S_s,j}};\overrightarrow{\mu^t_{s,2}},\Sigma^{s,t}_2)}]}}
\end{align}
$$
以上的推导均为不限定分布情况下的推导。如果基于上两式的结果，并假设表达量遵循正态分布，那么就有
$$
\begin{align}
\frac{f(\overrightarrow{e};\overrightarrow{\mu_1},\Sigma_1)}{f(\overrightarrow{e};\overrightarrow{\mu_2},\Sigma_2)}
&= \frac{\frac{1}{|\Sigma_1|^{0.5}\sqrt{2\pi}} e^{-\frac{||\overrightarrow{e}-\overrightarrow{\mu_1}||^2}{2\Sigma_1}} }{\frac{1}{|\Sigma_2|^{0.5}\sqrt{2\pi}} e^{-\frac{||\overrightarrow{e}-\overrightarrow{\mu_2}||^2}{2\Sigma_2}}}\\
&=\frac{|\Sigma_2|^{0.5}}{|\Sigma_1|^{0.5}} e^{-\frac{1}{2}(\overrightarrow{e}-\overrightarrow{\mu_1})^T\Sigma_1^{-1}(\overrightarrow{e}-\overrightarrow{\mu_1})+\frac{1}{2}(\overrightarrow{e}-\overrightarrow{\mu_2})^T\Sigma_2^{-1}(\overrightarrow{e}-\overrightarrow{\mu_2})}
\end{align}
$$

注意这里是协方差矩阵是，所以需要开方

### M步骤

每一轮的概率更新
$$
\tau^{t+1}_r =  \frac{\sum\limits_{j=1..J}P^t(z_j=r|E;\Theta^t)}{J}
$$
并且定义
$$
Q(\Theta|\Theta^t)=\sum\limits_{j=1}^J\{P^t(z_j=0|E;\Theta^t)[\log\tau_0+\sum\limits_{r=1}^R\log f(\overrightarrow{e_{S_r,j}};\overrightarrow{\mu_{r,2}}, \Sigma_{2}^{r})]\\+\sum\limits_{s=1}^RP^t(z_j=s|E;\Theta^t)[\log\tau_s+\log f(\overrightarrow{e_{S_r,j}};\overrightarrow{\mu_{r,1}},\Sigma_1^r)+\sum\limits_{r=1..R,r\neq s}\log f(\overrightarrow{e_{S_r,j}};\overrightarrow{\mu_{r,2}},\Sigma_2^r)]\}
$$

**note：**

1. 条件概率$P(A|B)=\frac{P(A\cap B)}{P(B)}$

2. 贝叶斯公式：设实验$E$的样本空间为$S$，$A$为$E$的事件，$B_1,B_2,..B_n$为$S$的一个**划分**（即对于每次实验$E$，这些事件中必然有且只有一个事件会发生），且$P(A)>0,P(B_i)>0\,(i=1,2,...,n)$，则有
   $$
   P(B_i|A)=\frac{P(A|B_i)P(B_i)}{\sum\limits_{j=1}^{n}{P(A|B_j)P(B_j)}},\,i=1,2,..,n.
   $$

因为对于每一细胞类型$s$均与其他类型
$$
\begin{align}
(\overrightarrow{\mu_{s,1}^{t+1}},\overrightarrow{\mu_{s,2}^{t+1}},\Sigma_1^{s,t+1},\Sigma_2^{s,t+1})\\
=\underset{\overrightarrow{\mu_{s,1}},\overrightarrow{\mu_{s,2}},\Sigma_1^{s},\Sigma_2^{s}}{\mathrm{argmax}} \sum\limits_{j=1}^J\{
% 细胞类型s
&P^t(z_j=s|E;\Theta^t)\log{f(\overrightarrow{e_{S_s,j}};\overrightarrow{\mu_{s,1}},\Sigma_1^s})+\\
% 细胞类型非s
&\sum\limits_{r=0,r\neq s}^{R}{P^t(z_j=r|E;\Theta^t)\log{f(\overrightarrow{e_{S_s,j}};\overrightarrow{\mu_{s,2}},\Sigma_2^s})}
\}\\
=\underset{\overrightarrow{\mu_{s,1}},\overrightarrow{\mu_{s,2}},\Sigma_1^{s},\Sigma_2^{s}}{\mathrm{argmax}} \sum\limits_{j=1}^J\{
% 细胞类型s
&P^t(z_j=s|E;\Theta^t)[-\frac{1}{2}\log{|\Sigma^s_1|}-\frac{1}{2}(\overrightarrow{e_{S_s,j}}-\overrightarrow{\mu_{s,1}})^T\Sigma^{s,-1}_1(\overrightarrow{e_{S_s,j}}-\overrightarrow{\mu_{s,1}})]\\
% 细胞类型非s
&\sum\limits_{r=0,r\neq s}^{R}{P^t(z_j=r|E;\Theta^t)[-\frac{1}{2}\log{|\Sigma^s_2|}-\frac{1}{2}(\overrightarrow{e_{S_s,j}}-\overrightarrow{\mu_{s,2}})^T\Sigma^{s,-1}_2(\overrightarrow{e_{S_s,j}}-\overrightarrow{\mu_{s,2}})]}
\}
\end{align}
$$

对参数进行更新后有
$$
\begin{flalign}
&\overrightarrow{\mu^{t+1}_{s,1}}=\frac{\sum\limits_{j=1}^{J}{P^t(z_j=s|E;\Theta^t)\overrightarrow{e_{S_s,j}}}}{\sum\limits_{j=1}^{J}{P^t(z_j=s|E;\Theta^t)}}\\
&\overrightarrow{\mu^{t+1}_{s,2}}=\frac{\sum\limits_{j=1}^{J}{\sum\limits_{r=0,r\neq s}^{R}{P^t(z_j=s|E;\Theta^t)\overrightarrow{e_{S_s,j}}}}}{\sum\limits_{j=1}^{J}{\sum\limits_{r=0,r\neq s}^{R}{P^t(z_j=s|E;\Theta^t)}}}
=\frac{\sum\limits_{j=1}^{J}{[1-P^t(z_j=s|E;\Theta^t)]\overrightarrow{e_{S_s,j}}}}{\sum\limits_{j=1}^{J}{[1-P^t(z_j=s|E;\Theta^t)]}}
\end{flalign}
$$
用这种方法计算后，如果基因集$S_r$中的任何基因$i$满足$\mu^{t+1}_{s,1}(i)<\mu^{t+1}_{s,2}(i)$，那么令
$$
\mu^{t+1}_{s,1}=\mu^{t+1}_{s,2}=\frac{\sum\limits_{j=1}^{J}{e_{i,j}}}{J}
$$
更新协方差矩阵
$$
D^{t+1}_{s,1}=diag((\overrightarrow{e_{S_s,j}}-\overrightarrow{\mu_{S_s,1}^{t+1}})\odot(\overrightarrow{e_{S_s,j}}-\overrightarrow{\mu_{S_s,1}^{t+1}}))\\
D^{t+1}_{s,2}=diag((\overrightarrow{e_{S_s,j}}-\overrightarrow{\mu_{S_s,2}^{t+1}})\odot(\overrightarrow{e_{S_s,j}}-\overrightarrow{\mu_{S_s,2}^{t+1}}))\\
\Sigma^{t+1}_{s,1}=\Sigma^{t+1}_{s,2}=\frac{\sum\limits_{j=1}^{J}{[1-P^t(z_j=s|E;\Theta^t)]D^{t+1}_{s,2}+P^t(z_j=s|E;\Theta^t)D^{t+1}_{s,1}}}{J}
$$


### 参数初始化

$\tau^0_r=\frac{1}{R+1},r=0..R$

$\overrightarrow{\mu^0_{r,1}}, \overrightarrow{\mu^0_{r,2}}, r=1...R$ 根据经验分布的分位数赋值

$\Sigma^{r,0},r=1..R$是一个对角矩阵的对角元素对应于每个基因的方差乘以一个常数

### 终止条件

指定的细胞类型标签的稳定

## 标记基因重叠的情况

这个版本的SCINA以一种启发式的方式处理基因签名之间的重叠。当存在重叠时，算法“虚拟”在重叠基因的表达式矩阵中创建多行。虚拟行的数量对应于该基因出现在基因签名集中的次数。然后，实际上，每个签名中的这个基因被分配给每个重复的行，而EM算法假设这些行是相互独立的。

Con：建模假设要求每个重复行有两种模式，较大的模式只对应来自一种细胞类型的表达贡献，而较小的模式只包含来自非表达细胞的贡献。这与重叠的基因签名相反。然而，当这些基因只由一小部分细胞类型与整个细胞类型共享时，偏差应该很小

Pro：此设置支持在不同单元格类型的签名之间的干净分离。因此，SCINA受益于r中向量化计算的巨大加速。此外，我们下面的经验分析表明，当签名中只有中等水平的重叠时，细胞分型性能的下降确实很低

