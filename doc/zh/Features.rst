Features
========

This is a short introduction for the features and algorithms used in LightGBM.

This page doesn't contain detailed algorithms, please refer to cited papers or source code if you are interested.

Optimization in Speed and Memory Usage
--------------------------------------

Many boosting tools use pre-sorted based algorithms\ `[1, 2] <#references>`__ (e.g. default algorithm in xgboost) for decision tree learning. It is a simple solution, but not easy to optimize.

LightGBM uses the histogram based algorithms\ `[3, 4, 5] <#references>`__, which bucketing continuous feature(attribute) values into discrete bins, to speed up training procedure and reduce memory usage.
Following are advantages for histogram based algorithms:

-  **Reduce calculation cost of split gain**

   -  Pre-sorted based algorithms need ``O(#data)`` times calculation

   -  Histogram based algorithms only need to calculate ``O(#bins)`` times, and ``#bins`` is far smaller than ``#data``

      -  It still needs ``O(#data)`` times to construct histogram, which only contain sum-up operation

-  **Use histogram subtraction for further speed-up**

   -  To get one leaf's histograms in a binary tree, can use the histogram subtraction of its parent and its neighbor

   -  So it only need to construct histograms for one leaf (with smaller ``#data`` than its neighbor), then can get histograms of its neighbor by histogram subtraction with small cost(``O(#bins)``)
-  **Reduce memory usage**

   -  Can replace continuous values to discrete bins. If ``#bins`` is small, can use small data type, e.g. uint8\_t, to store training data

   -  No need to store additional information for pre-sorting feature values

-  **Reduce communication cost for parallel learning**

Sparse Optimization
-------------------

-  Only need ``O(2 * #non_zero_data)`` to construct histogram for sparse features

Optimization in Accuracy
------------------------

Leaf-wise (Best-first) Tree Growth
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most decision tree learning algorithms grow tree by level(depth)-wise, like the following image:

.. image:: ./_static/images/level-wise.png
   :align: center

LightGBM grows tree by leaf-wise (best-first)\ `[6] <#references>`__. It will choose the leaf with max delta loss to grow.
When growing same ``#leaf``, leaf-wise algorithm can reduce more loss than level-wise algorithm.

Leaf-wise may cause over-fitting when ``#data`` is small.
So, LightGBM can use an additional parameter ``max_depth`` to limit depth of tree and avoid over-fitting (tree still grows by leaf-wise).

.. image:: ./_static/images/leaf-wise.png
   :align: center

Optimal Split for Categorical Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We often convert the categorical features into one-hot coding.
However, it is not a good solution in tree learner.
The reason is, for the high cardinality categorical features, it will grow the very unbalance tree, and needs to grow very deep to achieve the good accuracy.

Actually, the optimal solution is partitioning the categorical feature into 2 subsets, and there are ``2^(k-1) - 1`` possible partitions.
But there is a efficient solution for regression tree\ `[7] <#references>`__. It needs about ``k * log(k)`` to find the optimal partition.

The basic idea is reordering the categories according to the relevance of training target.
More specifically, reordering the histogram (of categorical feature) according to it's accumulate values (``sum_gradient / sum_hessian``), then find the best split on the sorted histogram.

Optimization in Network Communication
-------------------------------------

It only needs to use some collective communication algorithms, like "All reduce", "All gather" and "Reduce scatter", in parallel learning of LightGBM.
LightGBM implement state-of-art algorithms\ `[8] <#references>`__.
These collective communication algorithms can provide much better performance than point-to-point communication.

并行学习的优化
---------------------------------

LightGBM提供以下并行学习优化算法：

特征并行
~~~~~~~~~~~~~~~~

传统算法
^^^^^^^^^^^^^^^^^^^^^

Feature parallel aims to parallel the "Find Best Split" in the decision tree. The procedure of traditional feature parallel is:
传统的特征并行算法旨在于在并行化决策树中的“Find Best Split”。主要流程如下：

1. Partition data vertically (different machines have different feature set)
1. 垂直划分数据（不同的机器有不同的特征集）

2. Workers find local best split point {feature, threshold} on local feature set
2. 在本地特征集寻找最佳划分点｛特征， 阈值｝

3. Communicate local best splits with each other and get the best one
3. 本地进行各个划分的通信整合并得到最佳划分

4. Worker with best split to perform split, then send the split result of data to other workers
4. 以最佳划分方法对数据进行划分，并将数据划分结果传递给其他线程

5. Other workers split data according received data
5. 其他线程对接受到的数据进一步划分

The shortage of traditional feature parallel:
传统的特征并行方法主要不足：

-  Has computation overhead, since it cannot speed up "split", whose time complexity is ``O(#data)``.
   Thus, feature parallel cannot speed up well when ``#data`` is large.
-  存在计算上的局限，传统特征并行无法加速 “split”（时间复杂度为 “O（#data）”）。
   因此，当数据量很大的时候，难以加速。

-  Need communication of split result, which cost about ``O(#data / 8)`` (one bit for one data).
-  需要对划分的结果进行通信整合，其额外的时间复杂度约为 “O（#data/8）”（一个数据一个字节）

Feature Parallel in LightGBM
LightGBM中的特征并行
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Since feature parallel cannot speed up well when ``#data`` is large, we make a little change here: instead of partitioning data vertically, every worker holds the full data.
Thus, LightGBM doesn't need to communicate for split result of data since every worker know how to split data.
And ``#data`` won't be larger, so it is reasonable to hold full data in every machine.
既然在数据量很大时，传统数据并行方法无法有效地加速，我们做了一些改变：不再垂直划分数据，即每个线程都持有全部数据。
因此，LighetGBM中没有数据划分结果之间通信的开销，各个线程都知道如何划分数据。
而且，“#data” 不会变得更大，所以，在使每天机器都持有全部数据是合理的。

The procedure of feature parallel in LightGBM:
LightGBM 中特征并行的流程如下：

1. Workers find local best split point {feature, threshold} on local feature set
1. 每个线程都在本地数据集上寻找最佳划分点｛特征， 阈值｝

2. Communicate local best splits with each other and get the best one
2. 本地进行各个划分的通信整合并得到最佳划分

3. Perform best split
3. 执行最佳划分

However, this feature parallel algorithm still suffers from computation overhead for "split" when ``#data`` is large.
So it will be better to use data parallel when ``#data`` is large.
然而，该特征并行算法在数据量很大时仍然存在计算上的局限。因此，建议在数据量很大时使用数据并行。

Data Parallel
数据并行
~~~~~~~~~~~~~

Traditional Algorithm
传统算法
^^^^^^^^^^^^^^^^^^^^^

Data parallel aims to parallel the whole decision learning. The procedure of data parallel is:
数据并行旨在于并行化整个决策学习过程。数据并行的主要流程如下：

1. Partition data horizontally
1. 水平划分数据

2. Workers use local data to construct local histograms
2. 线程以本地数据构建本地直方图

3. Merge global histograms from all local histograms
3. 将本地直方图整合成全局整合图

4. Find best split from merged global histograms, then perform splits
4. 在全局直方图中寻找最佳划分，然后执行此划分

The shortage of traditional data parallel:
传统数据划分的不足：

-  High communication cost.
   If using point-to-point communication algorithm, communication cost for one machine is about ``O(#machine * #feature * #bin)``.
   If using collective communication algorithm (e.g. "All Reduce"), communication cost is about ``O(2 * #feature * #bin)`` (check cost of "All Reduce" in chapter 4.5 at `[8] <#references>`__).
-  高通讯开销。
   如果使用点对点的通讯算法，一个机器的通讯开销大约为 “O(#machine * #feature * #bin)” 。
   如果使用集成的通讯算法（例如， “All Reduce”等），通讯开销大约为 “O(2 * #feature * #bin)”[8] 。
Data Parallel in LightGBM
LightGBM中的数据并行
^^^^^^^^^^^^^^^^^^^^^^^^^

We reduce communication cost of data parallel in LightGBM:
LightGBM中采用以下方法较少数据并行中的通讯开销：

1. Instead of "Merge global histograms from all local histograms", LightGBM use "Reduce Scatter" to merge histograms of different(non-overlapping) features for different workers.
   Then workers find local best split on local merged histograms and sync up global best split.
1. 不同于“整合所有本地直方图以形成全局直方图”的方式，LightGBM 使用分散规约(Reduce scatter)的方式对不同线程的不同特征（不重叠的）进行整合。
   然后线程从本地整合直方图中寻找最佳划分并同步到全局的最佳划分中。
   
2. As aforementioned, LightGBM use histogram subtraction to speed up training.
   Based on this, we can communicate histograms only for one leaf, and get its neighbor's histograms by subtraction as well.
2. 如上所述。LightGBM 通过直方图做差法加速训练。
   基于此，我们可以进行单叶子的直方图通讯，并且在相邻直方图上使用做差法。
   
Above all, we reduce communication cost to ``O(0.5 * #feature * #bin)`` for data parallel in LightGBM.
通过上述方法，LightGBM 将数据并行中的通讯开销减少到 “O(0.5 * #feature * #bin)”。

Voting Parallel
投票并行
~~~~~~~~~~~~~~~

Voting parallel further reduce the communication cost in `Data Parallel <#data-parallel>`__ to constant cost.
投票并行未来将致力于将“数据并行”中的通讯开销减少至常数级别。
It uses two stage voting to reduce the communication cost of feature histograms\ `[9] <#references>`__.
其将会通过两阶段的投票过程较少特征直方图的通讯开销\ `[9] <#references>`__ 。

GPU Support
GPU 支持
-----------

Thanks `@huanzhang12 <https://github.com/huanzhang12>`__ for contributing this feature. Please read `[10] <#references>`__ to get more details.
感谢 “@huanzhang12 <https://github.com/huanzhang12>” 对此项特性的贡献。相关细节请阅读 `[10] <#references>`__ 。

- `GPU Installation <./Installatn-ioGuide.rst#build-gpu-version>`__
- `GPU 安装 <./Installatn-ioGuide.rst#build-gpu-version>`__

- `GPU Tutorial <./GPU-Tutorial.rst>`__
- `GPU 训练 <./GPU-Tutorial.rst>`__

Applications and Metrics
应用和度量
------------------------

Support following application:
支持以下应用：

-  regression, the objective function is L2 loss
-  回归，目标函数为 L2 loss

-  binary classification, the objective function is logloss
-  二分类， 目标函数为 logloss

-  multi classification
-  多分类

-  lambdarank, the objective function is lambdarank with NDCG
-  lambdarank,目标函数为基于 NDCG 的 lambdarank

Support following metrics:
支持的度量

-  L1 loss

-  L2 loss

-  Log loss

-  Classification error rate

-  AUC

-  NDCG

-  Multi class log loss

-  Multi class error rate

For more details, please refer to `Parameters <./Parameters.rst#metric-parameters>`__.
获取更多详情，请至 `Parameters <./Parameters.rst#metric-parameters>`__。

Other Features
其他特性
--------------

-  Limit ``max_depth`` of tree while grows tree leaf-wise

-  `DART <https://arxiv.org/abs/1505.01866>`__

-  L1/L2 regularization

-  Bagging

-  Column(feature) sub-sample

-  Continued train with input GBDT model

-  Continued train with the input score file

-  Weighted training

-  Validation metric output during training

-  Multi validation data

-  Multi metrics

-  Early stopping (both training and prediction)

-  Prediction for leaf index

For more details, please refer to `Parameters <./Parameters.rst>`__.
获取更多详情，请至 `Parameters <./Parameters.rst>`__。

References
----------

[1] Mehta, Manish, Rakesh Agrawal, and Jorma Rissanen. "SLIQ: A fast scalable classifier for data mining." International Conference on Extending Database Technology. Springer Berlin Heidelberg, 1996.

[2] Shafer, John, Rakesh Agrawal, and Manish Mehta. "SPRINT: A scalable parallel classifier for data mining." Proc. 1996 Int. Conf. Very Large Data Bases. 1996.

[3] Ranka, Sanjay, and V. Singh. "CLOUDS: A decision tree classifier for large datasets." Proceedings of the 4th Knowledge Discovery and Data Mining Conference. 1998.

[4] Machado, F. P. "Communication and memory efficient parallel decision tree construction." (2003).

[5] Li, Ping, Qiang Wu, and Christopher J. Burges. "Mcrank: Learning to rank using multiple classification and gradient boosting." Advances in neural information processing systems. 2007.

[6] Shi, Haijian. "Best-first decision tree learning." Diss. The University of Waikato, 2007.

[7] Walter D. Fisher. "`On Grouping for Maximum Homogeneity`_." Journal of the American Statistical Association. Vol. 53, No. 284 (Dec., 1958), pp. 789-798.

[8] Thakur, Rajeev, Rolf Rabenseifner, and William Gropp. "`Optimization of collective communication operations in MPICH`_." International Journal of High Performance Computing Applications 19.1 (2005): 49-66.

[9] Qi Meng, Guolin Ke, Taifeng Wang, Wei Chen, Qiwei Ye, Zhi-Ming Ma, Tieyan Liu. "`A Communication-Efficient Parallel Algorithm for Decision Tree`_." Advances in Neural Information Processing Systems 29 (NIPS 2016).

[10] Huan Zhang, Si Si and Cho-Jui Hsieh. "`GPU Acceleration for Large-scale Tree Boosting`_." arXiv:1706.08359, 2017.

.. _On Grouping for Maximum Homogeneity: http://amstat.tandfonline.com/doi/abs/10.1080/01621459.1958.10501479

.. _Optimization of collective communication operations in MPICH: http://wwwi10.lrr.in.tum.de/~gerndt/home/Teaching/HPCSeminar/mpich_multi_coll.pdf

.. _A Communication-Efficient Parallel Algorithm for Decision Tree: http://papers.nips.cc/paper/6381-a-communication-efficient-parallel-algorithm-for-decision-tree

.. _GPU Acceleration for Large-scale Tree Boosting: https://arxiv.org/abs/1706.08359
