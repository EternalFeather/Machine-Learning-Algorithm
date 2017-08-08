# 十大常见机器学习算法
常用的机器学习算法，几乎可以用在所有的数据问题上：

## 线性回归（Linear Regression）

线性回归通常用于根据**连续变量**估计实际数值等问题上。通过拟合最佳的**直线**来建立**自变量（X，features）** 和 **因变量（Y，labels）** 的关系。这条直线也叫做回归线，并用**Y = a* X + b**来表示。

在这个等式中：
- `Y` : 因变量（也就是Labels）
- `a` : 斜率（也就是Weights）
- `X` : 自变量（也就是Features）
- `b` : 截距（也就是Bias）

![](https://i.imgur.com/PSM7e7e.png)

系数 `a` 和 `b` 可以通过**最小二乘法**（即让所有pairs带入线性表达式等号两边的方差和最小）获得。

## 逻辑回归（Logistic Regression）

逻辑回归虽然名字中带有**回归**字样，但其实是一个**分类**算法而不是回归算法。该算法根据已知的一系列因变量估计**离散的数值**（0或1，代表假和真）。该算法通过将数据拟合进一个逻辑函数来预估一个事件发生的**概率**。由于其估计的对象是概率，所以输出的值大都在0和1之间。

逻辑回归通常用于解决二分类的问题，例如判断人是男是女等。逻辑回归就是通过人的一些基本性状特征来判断属于男女的概率。

从数学角度看，几率的对数使用的是**预测变量的线性组合**模型。
```Python
# Probability of event occurence / not occurence
odds = p / (1 - p)
ln(odds) = ln(p / (1 - p))
logit(p) = ln(p / (1 - p)) = b0 + b1X1 + b2X2 + ... + bnXn
```

式子中 `p` 指的是特征出现的概率，它选用使观察样本可能性最大的值（**极大似然估计**）作为参数，而不是通过最小二乘法得到。

- 那么为什么要取对数log呢？

    - 简而言之就是对数这种方式是复制阶梯函数最好的方法之一。

![](https://i.imgur.com/hq1q9Z5.png)

- 关于改进模型的方法：
    - 加入交互项（**X1 * X2**等）
    - 对输入输出进行正规化
    - 使用非线性模型

## 决策树（Decision Tree）

该算法属于监督式学习的一部分，主要用来处理分类的问题，它能够适用于分类连续因变量。我们将主体分成两个或者更多的类群，根据重要的属性或者自变量来尽可能多地区分开来。

![](https://i.imgur.com/8Nj3E0r.png)

- 根据不同的决策属性，我们可以依次将输入进行分类，最终会得到一个标签（Label）。为了把总体分成不同组别，需要用到许多技术，比如**Gini、Information Gain** 和 **Entropy** 等。

### Gini

![](https://i.imgur.com/ltVHIxt.png)

图中的实际分配曲线（红线）和绝对平衡线（绿线）之间的**面积**为A，和绝对不平衡线（蓝线）之间的面积为B，则横纵坐标之间的比例的**Gini系数**为：

<img src="http://chart.googleapis.com/chart?cht=tx&chl=\Large {A \over A + B}" style="border:none;">

- A为零时，Gini系数为0，表示完全平衡。B为零时，Gini系数为1，表示完全不平衡。

### Information Gain & Entropy
在我们建立决策树的时候，常常会有许多属性，那么用哪一个属性作为数的根节点呢？这个时候就需要用到 **信息增益（Information Gain）** 来衡量一个属性区分以上数据样本的能力强弱。信息增益越大的属性作为数的根节点，就能使得这棵树更加简洁。

![](https://i.imgur.com/9vwwsJt.png)

- 以图中数据为例，要想知道信息增益，就必须先算出分类系的**熵值（Entropy）**。最终结果的label是yes或者no，所以统计数量之后共有9个yes和5个no。这时候**P（“yes”） = 9 / 14，P（“no”） = 5 / 14**。这里的熵值计算公式为：

<img src="http://chart.googleapis.com/chart?cht=tx&chl=\Large Entropy(S) = {-(9 / 14) * log2(9 / 14) - (5 / 14) * log2(5 / 14)}" style="border:none;">

- 之后就可以计算每一个属性特征的信息增益（Gain）了。以wind属性为例，Wind为Weak的共有8条，其中yes的有6条，no的有2条；为Strong的共有6条，其中yes的有3条，no的也有3条。因此相应的熵值为：

<img src="http://chart.googleapis.com/chart?cht=tx&chl=\Large Entropy(Weak) = {-(6 / 8) * log2(6 / 8) - (2 / 8) * log2(2 / 8)}" style="border:none;">

<img src="http://chart.googleapis.com/chart?cht=tx&chl=\Large Entropy(Strong) = {-(3 / 6) * log2(3 / 6) - (3 / 6) * log2(3 / 6)}" style="border:none;">
- 现在就可以计算Wind属性的**信息增益**了：

<img src="http://chart.googleapis.com/chart?cht=tx&chl=\Large Gain(Wind) = {Entropy(S) -(8 / 14) * Entropy(Weak) - (6 / 14) * Entropy(Strong)}" style="border:none;">

## 支持向量机（Support vector machine,SVM）
SVM是一种常用的机器学习分类方式。在这个算法过程中，我们将每一笔数据在**N维度的空间中用点表示（N为特征总数，Features）**，每个特征的值是一个坐标的值。

如果以二维空间为例，此时有两个特征变量，我们会在空间中画出这两个变量的分布情况，每个点都有两个坐标（分别为tuples所具有的特征值组合）。

![](https://i.imgur.com/Ea3Jb95.png)

- 现在我们找一条直线将两组不同的数据在维度空间中分开。分割的曲线满足让两个分组中的距离最近的两个点到直线的距离**动态最优化**（都尽可能最近）。

![](https://i.imgur.com/NGsSXtM.png)

- 那么看到这里一定很多人和我一样有一个疑问，那就是这种线性分类的SVM和之前提到的逻辑回归（Logistic Regression）有什么**区别**呢？

其实他们在二维空间的**线性分类**中都扮演了重要的角色，其主要区别大致可分为两类：
- 寻找最优超平面的方式不同。
    - 形象来说就是Logistic模型找的超平面（二维中就是线）是尽可能让所有点都远离它。而SVM寻找的超平面，是只让最靠近的那些点远离，这些点也因此被称为**支持向量样本**，因此模型才叫**支持向量机**。

- SVM可以处理非线性的情况。
    - 比Logistic更强大的是，SVM还可以处理**非线性**的情况（经过优化之后的Logistic也可以，但是却更为复杂）。

![](https://i.imgur.com/5seIoZJ.png)

## 朴素贝叶斯（Naive Bayesian）
在假设变量间**相互独立**的前提下，根据贝叶斯定理（Bayesian Theorem）可以推得朴素贝叶斯这个分类方法。通俗来说，一个朴素贝叶斯分类器假设分类的特性和其他特性不相关。朴素贝叶斯模型容易创建，而且在非监督式学习的大型数据样本集中非常有用，虽然简单，却能超越复杂的分类方法。其基本思想就是：对于给出的待分类项，求解**在此项出现的条件下各个目标类别出现的概率**，哪个最大，就认为此待分类项属于哪个类别。

贝叶斯定理提供了从P（c）、P（x）和P（x | c）计算后验概率P（c | x）的方法:

<img src="http://chart.googleapis.com/chart?cht=tx&chl=\Large P(c | x) = {P(x | c) P(c) \over P(x)}" style="border:none;">

式子中的变量表示如下：
- P（c | x）是已知预测变量（属性特征）的前提下，目标发生的后验概率。
- P（c）是目标发生的先验概率。
- P（x | c）是已知目标发生的前提下，预测变量发生的概率。
- P（x）是预测变量的先验概率。

举一个例子：

![](https://i.imgur.com/gBuFCBd.png)

- 这是一个训练资料集，提供一些身体特征，用来预测人的性别。此时假设特征之间独立且满足高斯分布，则得到下表：

![](https://i.imgur.com/eSwuOJV.png)

- 通过计算方差、均值等参数，同时确认Label出现的频率来判断训练集的样本分布概率，P（male） = P（female） = 0.5。

![](https://i.imgur.com/qZPw7xC.png)

- 此时给出测试资料，我们希望通过计算得到性别的后验概率从而判断样本的类型：

**男子的后验概率**:

<img src="http://chart.googleapis.com/chart?cht=tx&chl=\Large Posterior(male) = {P(male) P(height | male) P(weight | male) P(footsize | male) \over evidence}" style="border:none;">

**女子的后验概率**:

<img src="http://chart.googleapis.com/chart?cht=tx&chl=\Large Posterior(female) = {P(female) P(height | female) P(weight | female) P(footsize | female) \over evidence}" style="border:none;">

证据因子（evidence）通常为常数，是用来对结果进行归一化的参数。

<img src="http://chart.googleapis.com/chart?cht=tx&chl=\Large Evidence = {(Posterior(female) + Posterior(male)) * evidence}" style="border:none;">

- 因此我们可以计算出相应结果：

<img src="http://chart.googleapis.com/chart?cht=tx&chl=\Large P(height | male) = {1 \over \sqrt{2\pi\sigma^2}}exp({-(6 - \mu^2) \over 2\sigma^2})" style="border:none;">

<img src="http://chart.googleapis.com/chart?cht=tx&chl=\Large P(weight | male) = ..." style="border:none;">

- 最后可以得出后验概率:

<img src="http://chart.googleapis.com/chart?cht=tx&chl=\Large Posterior Numerator(male) = {6.1984e^{-09}}" style="border:none;">

<img src="http://chart.googleapis.com/chart?cht=tx&chl=\Large Posterior Numerator(female) = {5.3778e^{-04}}" style="border:none;">

- 因此女性的概率较大，我们估计结果为女性。

## K近邻（K Nearest Neighbors）
该算法可以用于分类和回归问题，然而我们更常将其被用于解决分类问题上。KNN能够存储所有的案例，通过对比周围K个样本中的大概率情况，从而决定新的对象应该分配在哪一个类别。新的样本会被分配到它的K个最近最普遍的类别中去，因此KNN算法也是一个基于距离函数的算法。

这些**距离函数**可以是欧氏距离、曼哈顿距离、明氏距离或是汉明距离。前三个距离函数用于**连续函数**，最后一个用于**分类变量**。如果K = 1，新的样本就会被直接分到距离最近的那个样本所属的类别中。因此选择K是一个关系到模型精确度的问题。

![](https://i.imgur.com/7sGrxz0.png)

- 如图所示，如果我们取K = 3，即为中间的圆圈内，我们可以直观地看出此时绿点应该被归为红三角的一类。而如果K = 5，此时延伸到虚线表示的圆，则此时绿点应该被归为蓝色的类。

在选择KNN之前，我们需要考虑的事情有：

- KNN在K数量大的时候的计算成本很高。
- 变量（Features）应该先标准化（normalized），不然会被更高数量单位级别的范围带偏。
- 越是**干净**的资料效果越好，如果存在偏离度较高的杂讯噪声，那么在类别判断时就会收到干扰。

### 欧式距离
空间中点X = （X1，X2，X3，...，Xn）与点Y = （Y1，Y2，Y3，...，Yn）的欧氏距离为：

<img src="http://chart.googleapis.com/chart?cht=tx&chl=\Large d(x, y) := {\sqrt{(X1 - Y1)^2 + (X2 - Y2)^2 + ... + (Xn - Yn)^2}}" style="border:none;">

### 曼哈顿距离
在平面上，坐标（X1，X2，...，Xn）的点和坐标（Y1，Y2，...，Yn）的点之间的曼哈顿距离为:

<img src="http://chart.googleapis.com/chart?cht=tx&chl=\Large {|X1 - Y1| + |X2 - Y2| + ... + |Xn - Yn|}" style="border:none;">

### 明氏距离
两点 P = (X1，X2，...，Xn) 和 Q = （Y1，Y2，...，Yn）之间的明氏距离为:

<img src="http://chart.googleapis.com/chart?cht=tx&chl=\Large {(|X1 - Y1|^p + |X2 - Y2|^p + ... + |Xn - Yn|^p)^{1 \over p}}" style="border:none;">

- 其中p取1时为曼哈顿距离，p取2时为欧氏距离。

### 汉明距离
对于固定长度n，汉明距离是该长度字符串向量空间上的度量，即表示长度n中不同字符串的个数。

例子：
- **"toned"** 和 **“roses”** 之间的汉明距离就是3。因为其中 **t - > r，n -> s，d -> s** 三个字符不相同。

## K均值（K-means）
K-means方法是一种**非监督式学习**的算法，能够解决**聚类**问题。使用K-means算法将一个数据样本归入一定数量的集群中（假设有K个）中，每一个集群的数据点都是均匀齐次的，并且异于其它集群。

![](https://i.imgur.com/WQlIGo4.png)

K-means算法如何形成**集群**？
- 给一个集群选择K个点，这些点称为质心。
- 给每一个数据点与距离最接近的质心形成一个集群，也就是K个集群。
- 根据现有的类别成员，找出每个类别的质心。
- 当有新的样本输入后，找到距离每个数据点最近的质心，并与质心对应的集群归为一类，计算新的质心位置，重复这个过程直到数据收敛，即质心位置不再改变。
- 如果新的数据点到多个质心的距离相同，则将这个数据点作为**新的质心**。

如何决定K值？
- K-means算法涉及到集群问题，每个集群都有自己的质心。一个集群的内的质心和个数据点之间的距离的平方和形成了这个集群的平方值之和。我们能够直观地想象出当集群的内部的数据点增加时，K值会跟着下降（数据点越多，分散开来每个质心能够包揽的范围就变大了，这时候其他的集群就会被吞并或者分解）。**集群元素数量的最优值**也就是在集群的平方值之和最小的时候取得（每个点到质心的距离和最小，分类最精确）。

## 随机森林（Random Forest）
Random Forest是表示**决策树总体**的一个专有名词。在算法中我们有一系列的决策树（因此为**森林**）。为了根据一个新的对象特征将其分类，每一个决策树都有一个分类结果，称之为这个决策树**投票**给某一个分类群。这个森林选择获得其中（所有决策树）**投票数最多**的分类。

![](https://i.imgur.com/xViexYM.png)

Random Forest中的Decision Tree是如何形成的？
- 如果训练集的样本数量为N，则从N个样本中用重置抽样的方式随机抽取样本。这个样本将作为决策树的训练资料。
- 假如有N个输入特征变量，则定义一个数字**m << M**。m表示从M中随机选中的变量，这m个变量中最好的切分特征会被用来当成节点的决策特征（利用Information Gain等方式）。在构建其他决策树的时候，m的值**保持不变**。
- 尽可能大地建立每一个数的节点分支。

## 降维（Dimensionality reduction）
当今的社会中信息的捕捉量都是呈上升的趋势。各种研究信息数据都在尽可能地捕捉完善，生怕遗漏一些关键的特征值。对于这些数据中包含许多特征变量的数据而言，看似为我们的模型建立提供了充足的**训练材料**。但是这里却存在一个问题，那就是**如何从上百甚至是上千种特征中区分出样本的类别呢？**样本特征的**重要程度**又该如何评估呢？

- 其实随着输入数据特征变量的增多，模型很难拟合众多样本变量（高维度）的数据分类规则。这样训练出来的模型不但**效果差**，而且**消耗大量的时间**。
- 这个时候，降维算法和别的一些算法（比如**Decision Tree**、**Random Forest**、**主成分分析（PCA）** 和 **因子分析**）就能帮助我们实现根据相关矩阵，压缩维度空间之后总结特征规律，最终再逐步还原到高维度空间的训练模式。

### 主成分分析（PCA）
在多元统计分析中，PCA是一种分析、简化数据集的技术，经常用于减少数据集的维数，同时保留数据集中的**对方差贡献最大**的那些特征变量。

- 该算法会根据不同维度的压缩（在这个维度上的**投影**）来测试**各个维度对方差的影响**，从而对每一个维度进行重新排序（影响最大的放在第一维度）。之后只需要取有限个数的维度进行训练，就能够保证模型拟合最佳的数据特征了。

### 因子分析
该算法主要是从关联矩阵内部的依赖关系出发，把一些重要信息重叠，将错综复杂的变量归结为少数几个不相关的综合因子的多元统计方法。基本思想是：根据**相关性大小**把变量分租，使得同组内的变量之间相关性高，但不同组的变量不相关或者相关性低。每组变量代表一个基本结构，即公共因子。

## Gradient Boost & Adaboost
当我们想要处理很多数据来做一个具有高度预测能力的预测模型时，我们会用到Gradient Boost和AdaBoost这两种Boosting算法。**Boosting算法**是一种集成学习算法，它结合了建立在多个基础估计值上的预测结果，来增强单个估计值的准确度。

![](https://i.imgur.com/eOKOw6J.png)

### Adaboost

Bossting能够对一份数据建立多个模型（如分类模型），通常这些模型都比较简单，称为**弱分类器（Weak Learner）**。每次分类都将上一次分错的数据权重值调大（放大的圆圈），然后再次进行分类，最终得到更好的结果。最终所有学习器（在这里值分类器）共同组成完整的模型。

### Gradient Boost
与Adaboost不同的是，Gradient Boost在迭代的时候选择梯度下降的方向来保证最后的结果最好。损失函数（Loss function）用来描述模型的误差程度，如果模型没有Over fitting，那么loss的值越大则误差越高。如果我们的模型能够让损失函数值下降，说明它在不断改进，而最好的方式就是让函数在**梯度的方向**上改变。（类似神经网络的**Gradient Descend**）
