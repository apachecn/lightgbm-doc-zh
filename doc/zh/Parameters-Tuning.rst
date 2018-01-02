Parameters Tuning
参数的调整
=================

This is a page contains all parameters in LightGBM.
本页包含了 LightGBM 中所有的参数。

**List of other helpful links**
**其他有帮助的链接列表**

-  `Parameters <./Parameters.rst>`__
-  `Python API <./Python-API.rst>`__

Tune Parameters for the Leaf-wise (Best-first) Tree
Leaf-wise (最优优先) 树的参数调整
---------------------------------------------------

LightGBM uses the `leaf-wise <./Features.rst#leaf-wise-best-first-tree-growth>`__ tree growth algorithm, while many other popular tools use depth-wise tree growth.
Compared with depth-wise growth, the leaf-wise algorithm can convenge much faster.
However, the leaf-wise growth may be over-fitting if not used with the appropriate parameters.
LightGBM 使用`leaf-wise <./Features.rst#leaf-wise-best-first-tree-growth>`__ 的树生长策略，而很多其他流行的算法采用depth-wise 的树生长策略。
与 depth-wise 的树生长策略相较，leaf-wise 算法可以收敛的更快。
但是，如果参数选择不当的话，leaf-wise 算法有可能导致过拟合。

To get good results using a leaf-wise tree, these are some important parameters:
想要在使用 leaf-wise 算法时得到好的结果，这里有几个重要的参数值得注意：

1. ``num_leaves``. This is the main parameter to control the complexity of the tree model.
   Theoretically, we can set ``num_leaves = 2^(max_depth)`` to convert from depth-wise tree.
   However, this simple conversion is not good in practice.
   The reason is, when number of leaves are the same, the leaf-wise tree is much deeper than depth-wise tree. As a result, it may be over-fitting.
   Thus, when trying to tune the ``num_leaves``, we should let it be smaller than ``2^(max_depth)``.
   For example, when the ``max_depth=6`` the depth-wise tree can get good accuracy,
   but setting ``num_leaves`` to ``127`` may cause over-fitting, and setting it to ``70`` or ``80`` may get better accuracy than depth-wise.
   Actually, the concept ``depth`` can be forgotten in leaf-wise tree, since it doesn't have a correct mapping from ``leaves`` to ``depth``.
1. ``叶子数目``。这是控制树模型复杂度的主要参数。
理论上，借鉴 depth-wise 树，我们可以设置 ``叶子数目 = 2^(树的最大深度)`` 
但是，这种简单的转化在实际应用中表现不佳。
这是因为，当叶子数目相同时，leaf-wise 树要比 depth-wise 树深得多，这就有可能导致过拟合。
因此，当我们试着调整 ``叶子数目`` 的取值时，应该让其小于 ``2^(树的最大深度)``。
举个例子，当 ``树的最大深度=6`` 时(这里译者认为例子中，树的最大深度应为7)，depth-wise 树可以达到较高的准确率。但是如果设置 ``叶子数目`` 为 ``127`` 时，有可能会导致过拟合，而将其设置为 ``70`` 或 ``80`` 时可能会得到比 depth-wise 树更高的准确率。
其实，``深度`` 的概念在 leaf-wise 树中并没有多大作用，因为并不存在一个从 ``叶子数目`` 到 ``深度`` 的合理映射。
 
2. ``min_data_in_leaf``. This is a very important parameter to deal with over-fitting in leaf-wise tree.
   Its value depends on the number of training data and ``num_leaves``.
   Setting it to a large value can avoid growing too deep a tree, but may cause under-fitting.
   In practice, setting it to hundreds or thousands is enough for a large dataset.
2. ``叶子所包含的最少数据``。这是处理 leaf-wise 树的过拟合问题中一个非常重要的参数。 它的值取决于训练数据的样本个树和 ``叶子数目``。将其设置的较大可以避免生成一个过深的树，但有可能导致欠拟合。实际应用中，对于大数据集，设置其为几百或几千就足够了。

3. ``max_depth``. You also can use ``max_depth`` to limit the tree depth explicitly.
3. ``树的最大深度``。你也可以利用 ``树的最大深度`` 来显式地限制树的深度。

For Faster Speed
----------------
针对更快的训练速度
----------------
-  Use bagging by setting ``bagging_fraction`` and ``bagging_freq``

-  Use feature sub-sampling by setting ``feature_fraction``

-  Use small ``max_bin``

-  Use ``save_binary`` to speed up data loading in future learning

-  Use parallel learning, refer to `Parallel Learning Guide <./Parallel-Learning-Guide.rst>`__

- 通过设置 ``bagging 比例`` 和 ``bagging 频率`` 参数来使用 bagging 方法

- 通过设置 ``特征比例`` 参数来使用特征的子抽样

- 使用较小的 ``最大bin``

- 使用 ``save_binary`` 在未来的学习过程对数据加载进行加速

- 使用并行学习，可参考 `Parallel Learning Guide <./Parallel-Learning-Guide.rst>`__


For Better Accuracy
-------------------
针对更好的准确率
-------------------

-  Use large ``max_bin`` (may be slower)

-  Use small ``learning_rate`` with large ``num_iterations``

-  Use large ``num_leaves`` (may cause over-fitting)

-  Use bigger training data

-  Try ``dart``

- 使用较大的 ``最大 bin`` （学习速度可能变慢）

- 使用较小的 ``学习速率`` 和较大的 ``迭代次数``

- 使用较大的 ``叶子数目``（可能导致过拟合）

- 使用更大的训练数据

- 尝试 ``dart``

Deal with Over-fitting
----------------------
处理过拟合
----------------------

-  Use small ``max_bin``

-  Use small ``num_leaves``

-  Use ``min_data_in_leaf`` and ``min_sum_hessian_in_leaf``

-  Use bagging by set ``bagging_fraction`` and ``bagging_freq``

-  Use feature sub-sampling by set ``feature_fraction``

-  Use bigger training data

-  Try ``lambda_l1``, ``lambda_l2`` and ``min_gain_to_split`` for regularization

-  Try ``max_depth`` to avoid growing deep tree

- 使用较小的 ``最大 bin``

- 使用较小的 ``叶子数目``

- 使用 ``叶子包含的最小数据`` 和 ``min_sum_hessian_in_leaf``

- 通过设置``bagging 比例`` 和``bagging 频率``来使用 bagging

- 通过设置 ``特征比例`` 来使用特征子抽样

- 使用更大的训练数据

- 使用 ``lambda_l1``, ``lambda_l2`` 和 ``节点分裂最小增益`` 来使用正则

- 尝试 ``最大深度`` 来避免生成过深的树

