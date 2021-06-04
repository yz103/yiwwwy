---
layout: post
title:  "Notes for Gradient Boosting"
date:   2021-06-03 14:37:02 -0500
tags: ML/DL
---



I don't know why I wrote it in Chinese..... Lately I will change it to English.....

最近准备面试，又复习了这一块的知识，想当初老师讲的也极其简单粗暴（如图1。。。），找了些中文和英文的资料，发现中西结合更有效得搞出了这一篇。我也有一模一样的全英文版本，想要的小伙伴可以私信我。**Gradient boosting uses an ensemble of decision trees to predict a target label.** 简单的说，**Gradient boosing 是利用决策树的集成来预测目标值的一种方法**。提示一下，在学习gradient boosting之前，需要先了解decision tree（决策树），和ensemble（模型集成，不知是不是这么翻译的）。

<p align="center">
  <img src="https://yz103.github.io/yiwwwy/assets/gradient_boosting_figure1_instructornotes.png" align="center" width="300px" height="200px"/>
<center>图1: 老师的简单粗暴手稿。。</center>
</p>


接下来文章主要分为三个部分：**第一个部分，用一个具体例子定性的解释一下gradient boosting 训练过程；第二个部分，用严谨的数学公式来表示；第三个部分，code**！

####  <u>手动train一个gradient boosting</u>

假设我们有这样一个训练任务，客人下了外卖之后，一般外卖平台会显示一个预计送达时间，我们的任务是预测从下单开始到外卖送到大概要花多少时间。数据如下（纯属虚构）

| 餐厅ID | 订单价格 | 订单件数 | 花费时间 |
| ------ | -------- | -------- | :------: |
| 2      | 36       | 2        |    30    |
| 3      | 68       | 3        |    45    |
| 4      | 53       | 3        |    42    |
| 5      | 21       | 1        |    15    |
| 6      | 90       | 3        |    36    |

- 步骤1： 用目标值的**平均值**作为初始的regressor 或者 第一个树

$$
\text{average value} = \frac{20+45+42+15+36}{5}=33.6
$$

- 步骤2： 计算一下residual（残差）= 目标值 - average value， 然后我们的表格变成了下面样子,

  | 餐厅ID | 订单价格 | 订单件数 | 花费时间 | 残差  |
  | ------ | -------- | -------- | :------: | ----- |
  | 2      | 36       | 2        |    30    | -3.6  |
  | 3      | 68       | 3        |    45    | 11.4  |
  | 4      | 53       | 3        |    42    | 8.4   |
  | 5      | 21       | 1        |    15    | -18.6 |
  | 6      | 90       | 3        |    36    | 2.4   |

- 步骤3：训练一个决策树 $F_1(x)$ 来预测这个residuals（参差），比如它可以长成酱紫（纯属虚构）

  <p align="center">
    <img src="https://yz103.github.io/yiwwwy/assets/gradient_boosting_figure2_tree1.png" align="center" width="300px" height="200px"/>
  </p>
  
  有2个数据点分到了一个叶子里面，那么预测的时候，就取叶子里面数据点目标值的平均数，所以最后这个决策树长这个样子，
  
  <p align="center">
    <img src="https://yz103.github.io/yiwwwy/assets/gradient_boosting_figure3_tree2.png" align="center" width="300px" height="200px"/>
  </p>
  
  
  
- 步骤4： 计算新的预测值
  $$
  F(x) = F_0(x)+ \gamma F_1(x) = \text{average value + learning rate *}F_1(x)
  $$
  假设learning rate 是$$0.1$$的话，比如第一个数据点的预测值就是 $$33.6 + 0.1*-3.6 = 33.24$$

  然后我们的table就变成酱紫

  | 餐厅ID | 订单价格 | 订单件数 | 花费时间 | 残差 after $F_0(x)$ | 新残差 |
  | ------ | -------- | -------- | :------: | ------------------- | ------ |
  | 2      | 36       | 2        |    30    | -3.6                | -3.24  |
  | 3      | 68       | 3        |    45    | 11.4                | 10.41  |
  | 4      | 53       | 3        |    42    | 8.4                 | 7.41   |
  | 5      | 21       | 1        |    15    | -18.6               | -16.74 |
  | 6      | 90       | 3        |    36    | 2.4                 | 2.16   |

- 步骤5：接下来再训练一个决策树 $$F_2(x)$$ 来预测这个新的参差。相当于回到步骤3 开始循环往复直到第M个决策树。也就是我们最终的模型：average value + learning rate * 第一个决策树 + learning rate* 第二个决策树...一直加到learning rate*第M个决策树。

#### <u>严谨的数学公式（维基百科抄的）</u>

在训练之前我们要确定至少以下几点：

- 训练的数据${\displaystyle \{(x_{i},y_{i})\}_{i=1}^{n}}$
- Loss function (损失函数) $${\displaystyle L(y,F(x))}$$
- 训练决策树数量 $$M$$

算法如下:

- 初始化第一个树, 下面的公式看起来繁琐，其实就是**找一个常数使loss最小**，一般就可以像第一部分那个例子一样选择目标值的平均值来作为第一个树
  $$
  {\hat {F}}={\underset {F}{\arg \min }}\,\mathbb {E} _{x,y}[L(y,F(x))]
  $$

- For $$m = 1 \text{ to } M$$:

  1. 计算 **pseudo-residuals**(“假”残差):

     $r_{im} = -\Big[{\frac{\partial L(y_i,F(x_i))}{\partial F(x_i)}}\Big]_{F(x)=F_{m-1}(x)}\quad for\quad i=1,\dots,n$

     如果loss function是mean square error 的话，手动推到一下会发现
     $$
     -\Big[{\frac{\partial L(y_i,F(x_i))}{\partial F(x_i)}}\Big] = y-F(x_i)
     $$
     也就是说这个求导正好就是残差，那为什么又叫它“假”残差呢，我猜是因为如果用别的loss function可能没这么正好。但无论如何我们这一步的任务只是想个办法来得到残差。因为接下来的决策树是基于残差的训练。

  2. 基于这个新的数据$$\{(x_{i},r_{im})\}_{i=1}^{n}$$，训练决策树 $${\displaystyle h_{m}(x)}$$ 

     $$h_m(x)=\sum_{j=1}^{J}C_{mj}I(x_i\in R_{mj})$$

     学过决策树的筒子们应该很清楚这个表达式，无非就是把数据分到$J$ 个叶子里面。对于一个数据点如果它属于叶子$$j$$，那么它的预测值就是$$C_{mj}$$

  3. 接下来是选择一个合适的决策树$$h_m(x)$$的learning rate $$\gamma_m$$
     $$
     \gamma_m = argmin_{\gamma}\sum_{i=1}^{n}L(y_i,F_{m-1}(x_i)+\gamma h_m(x_i))
     $$

  4. 更新一下目前的模型

     $${\displaystyle F_{m}(x)=F_{m-1}(x)+\gamma *{m}h*{m}(x).}$$

- 输出 $${\displaystyle F_{M}(x).}$$

#### <u>Coding！</u>

这个作者太懒还没有写code。。。之后会更新。（主要是没有找到不需要预处理直接可以做训练的数据，参考文献的第一篇里面有code！但是好像没有数据库连接）。





参考资料：

- Cory Marlin. [Gradient Boosting Decision Tree Algorithm Explained](https://towardsdatascience.com/machine-learning-part-18-boosting-algorithms-gradient-boosting-in-python-ef5ae6965be4)
- Wikipedia. [Gradient boosting](https://en.wikipedia.org/wiki/Gradient_boosting#:~:text=When%20a%20decision%20tree%20is,an%20arbitrary%20differentiable%20loss%20function)





















