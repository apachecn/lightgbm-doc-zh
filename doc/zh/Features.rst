Features
========

This is a short introduction for the features and algorithms used in LightGBM.

This page doesn't contain detailed algorithms, please refer to cited papers or source code if you are interested.

Optimization in Speed and Memory Usage
--------------------------------------

特点
====

这篇文档是对LightGBM特点和其中用到的算法的简单介绍

本页不包含算法的细节，如果你对这些算法感兴趣可以查阅引用的论文或者源代码

速度和内存使用的优化
-------------------

Many boosting tools use pre-sorted based algorithms\ `[1, 2] <#references>`__ (e.g. default algorithm in xgboost) for decision tree learning. It is a simple solution, but not easy to optimize.
许多增强工具对于决策树的学习利用基于算法\ `[1, 2] <#references>`__ (例如，在xgboost中默认的算法)的预排序，这是一个简单的解决方案，但是不易于最优化。

LightGBM uses the histogram based algorithms\ `[3, 4, 5] <#references>`__, which bucketing continuous feature(attribute) values into discrete bins, to speed up training procedure and reduce memory usage.
Following are advantages for histogram based algorithms:

LightGBM 利用基于histogram算法\ `[3, 4, 5] <#references>`__，通过将连续特征（属性）值分段为离散的值来加快训练的速度并减少内存的使用。
如下的是基于histogram算法的优点：

-  **Reduce calculation cost of split gain**
-  **减少分割增益的计算量**
   -  Pre-sorted based algorithms need ``O(#data)`` times calculation
   -  Pre-sorted算法需要 ``O(#data)`` 次的计算

   -  Histogram based algorithms only need to calculate ``O(#bins)`` times, and ``#bins`` is far smaller than ``#data``
   -  histogram算法只需要计算 ``O(#bins)`` 次, 而 ``#bins`` 远少于 ``#data`` 

      -  It still needs ``O(#data)`` times to construct histogram, which only contain sum-up operation
      -  这个仍然需要 ``O(#data)`` 次来构建直方图, 而这仅仅包含总结操作

-  **Use histogram subtraction for further speed-up**
-  **通过对直方图的相减来进行进一步的加速**

   -  To get one leaf's histograms in a binary tree, can use the histogram subtraction of its parent and its neighbor
   -  在二叉树中可以通过利用叶节点的父节点和相邻节点的直方图的相减来获得该叶节点的直方图

   -  So it only need to construct histograms for one leaf (with smaller ``#data`` than its neighbor), then can get histograms of its neighbor by histogram subtraction with small cost(``O(#bins)``)
   -  所以仅仅需要为一个叶节点建立直方图 (用与其相邻节点少的多的 ``#data`` )就可以通过直方图的相减，花费很小的代价(``O(#bins)``)，来获得相邻节点的直方图

-  **Reduce memory usage**
-  **减少内存的使用**

   -  Can replace continuous values to discrete bins. If ``#bins`` is small, can use small data type, e.g. uint8\_t, to store training data
   -  可以将连续的值替换为 discrete bins。 如果 ``#bins`` 较小, 可以利用较小的数据类型, 如 uint8\_t, 来存储训练数据。

   -  No need to store additional information for pre-sorting feature values
   -  无需为 pre-sorting 特征值存储额外的信息

-  **Reduce communication cost for parallel learning**
-  **减少并行学习的通信代价**

Sparse Optimization
-------------------

稀疏优化
--------

-  Only need ``O(2 * #non_zero_data)`` to construct histogram for sparse features
-  对于稀疏的特征仅仅需要 ``O(2 * #non_zero_data)`` 来建立直方图

Optimization in Accuracy
------------------------
准确率的优化
-----------

Leaf-wise (Best-first) Tree Growth
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Leaf-wise (Best-first) Tree Growth
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most decision tree learning algorithms grow tree by level(depth)-wise, like the following image:
大部分决策树的学习算法通过 level(depth)-wise 策略增长树，就像如下的图像一样：

.. image:: ./_static/images/level-wise.png
   :align: center

LightGBM grows tree by leaf-wise (best-first)\ `[6] <#references>`__. It will choose the leaf with max delta loss to grow.
LightGBM 通过 leaf-wise (best-first)\ `[6] <#references>`__策略来增长树。它将选取具有最大 delta loss 的叶节点来分裂。

When growing same ``#leaf``, leaf-wise algorithm can reduce more loss than level-wise algorithm.
当增长相同的 ``#leaf``， 相较于 level-wise 算法，leaf-wise 算法可以减少更多的损失。

Leaf-wise may cause over-fitting when ``#data`` is small.
当 ``#data`` 较小的时候，leaf-wise 可能会造成过拟合。

So, LightGBM can use an additional parameter ``max_depth`` to limit depth of tree and avoid over-fitting (tree still grows by leaf-wise).
所以，LightGBM 可以利用额外的参数 ``max_depth`` 来限制树的深度并避免过拟合（树的生长仍然通过 leaf-wise 策略）。

.. image:: ./_static/images/leaf-wise.png
   :align: center

Optimal Split for Categorical Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

类别特征值的最优分割
~~~~~~~~~~~~~~~~~~~

We often convert the categorical features into one-hot coding.
我们通常将类别特征转化为 one-hot coding。
However, it is not a good solution in tree learner.
然而，对于学习树来说这不是个好的解决方案。
The reason is, for the high cardinality categorical features, it will grow the very unbalance tree, and needs to grow very deep to achieve the good accuracy.
原因是，对于一个基数较大的类别特征，学习树会变的非常不平衡，并且需要非常深的深度才能来达到较好的准确率。

Actually, the optimal solution is partitioning the categorical feature into 2 subsets, and there are ``2^(k-1) - 1`` possible partitions.
其实，最好的解决方案是将类别特征划分为两个子集，总共有 ``2^(k-1) - 1`` 中可能的划分
But there is a efficient solution for regression tree\ `[7] <#references>`__. It needs about ``k * log(k)`` to find the optimal partition.
但是对于回归树\ `[7] <#references>`__有个有效的解决方案。为了寻找最优的划分需要大约 ``k * log(k)`` 。
The basic idea is reordering the categories according to the relevance of training target.
基本的思想是根据训练目标的相关性对类别进行重排序。
More specifically, reordering the histogram (of categorical feature) according to it's accumulate values (``sum_gradient / sum_hessian``), then find the best split on the sorted histogram.
更具体的说，根据累加值(``sum_gradient / sum_hessian``)重新对直方图（类别特征）进行排序，然后在排好序的直方图中寻找最好的分割点。


Optimization in Network Communication
-------------------------------------
网络通信的优化
-------------


It only needs to use some collective communication algorithms, like "All reduce", "All gather" and "Reduce scatter", in parallel learning of LightGBM.
在 LightGBM 中的并行学习，仅仅需要使用一些聚合通信算法，例如"All reduce", "All gather" 和 "Reduce scatter"
LightGBM implement state-of-art algorithms\ `[8] <#references>`__.
LightGBM实现了 state-of-art 算法\ `[8] <#references>`__。
These collective communication algorithms can provide much better performance than point-to-point communication.
这些聚合通信算法可以提供比点对点通信更好的性能。












Optimization in Parallel Learning
---------------------------------

LightGBM provides following parallel learning algorithms.

Feature Parallel
~~~~~~~~~~~~~~~~

Traditional Algorithm
^^^^^^^^^^^^^^^^^^^^^

Feature parallel aims to parallel the "Find Best Split" in the decision tree. The procedure of traditional feature parallel is:

1. Partition data vertically (different machines have different feature set)

2. Workers find local best split point {feature, threshold} on local feature set

3. Communicate local best splits with each other and get the best one

4. Worker with best split to perform split, then send the split result of data to other workers

5. Other workers split data according received data

The shortage of traditional feature parallel:

-  Has computation overhead, since it cannot speed up "split", whose time complexity is ``O(#data)``.
   Thus, feature parallel cannot speed up well when ``#data`` is large.

-  Need communication of split result, which cost about ``O(#data / 8)`` (one bit for one data).

Feature Parallel in LightGBM
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Since feature parallel cannot speed up well when ``#data`` is large, we make a little change here: instead of partitioning data vertically, every worker holds the full data.
Thus, LightGBM doesn't need to communicate for split result of data since every worker know how to split data.
And ``#data`` won't be larger, so it is reasonable to hold full data in every machine.

The procedure of feature parallel in LightGBM:

1. Workers find local best split point {feature, threshold} on local feature set

2. Communicate local best splits with each other and get the best one

3. Perform best split

However, this feature parallel algorithm still suffers from computation overhead for "split" when ``#data`` is large.
So it will be better to use data parallel when ``#data`` is large.

Data Parallel
~~~~~~~~~~~~~

Traditional Algorithm
^^^^^^^^^^^^^^^^^^^^^

Data parallel aims to parallel the whole decision learning. The procedure of data parallel is:

1. Partition data horizontally

2. Workers use local data to construct local histograms

3. Merge global histograms from all local histograms

4. Find best split from merged global histograms, then perform splits

The shortage of traditional data parallel:

-  High communication cost.
   If using point-to-point communication algorithm, communication cost for one machine is about ``O(#machine * #feature * #bin)``.
   If using collective communication algorithm (e.g. "All Reduce"), communication cost is about ``O(2 * #feature * #bin)`` (check cost of "All Reduce" in chapter 4.5 at `[8] <#references>`__).

Data Parallel in LightGBM
^^^^^^^^^^^^^^^^^^^^^^^^^

We reduce communication cost of data parallel in LightGBM:

1. Instead of "Merge global histograms from all local histograms", LightGBM use "Reduce Scatter" to merge histograms of different(non-overlapping) features for different workers.
   Then workers find local best split on local merged histograms and sync up global best split.

2. As aforementioned, LightGBM use histogram subtraction to speed up training.
   Based on this, we can communicate histograms only for one leaf, and get its neighbor's histograms by subtraction as well.

Above all, we reduce communication cost to ``O(0.5 * #feature * #bin)`` for data parallel in LightGBM.

Voting Parallel
~~~~~~~~~~~~~~~

Voting parallel further reduce the communication cost in `Data Parallel <#data-parallel>`__ to constant cost.
It uses two stage voting to reduce the communication cost of feature histograms\ `[9] <#references>`__.

GPU Support
-----------

Thanks `@huanzhang12 <https://github.com/huanzhang12>`__ for contributing this feature. Please read `[10] <#references>`__ to get more details.

- `GPU Installation <./Installation-Guide.rst#build-gpu-version>`__

- `GPU Tutorial <./GPU-Tutorial.rst>`__

Applications and Metrics
------------------------

Support following application:

-  regression, the objective function is L2 loss

-  binary classification, the objective function is logloss

-  multi classification

-  lambdarank, the objective function is lambdarank with NDCG

Support following metrics:

-  L1 loss

-  L2 loss

-  Log loss

-  Classification error rate

-  AUC

-  NDCG

-  Multi class log loss

-  Multi class error rate

For more details, please refer to `Parameters <./Parameters.rst#metric-parameters>`__.

Other Features
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
