参数
==========

这个页面包含了 LightGBM 的所有参数.

**一些有用的链接列表**

- `Python API <./Python-API.rst>`__

- `Parameters Tuning <./Parameters-Tuning.rst>`__

**外部链接**

- `Laurae++ Interactive Documentation`_

**更新于 08/04/2017**

以下参数的默认值已经修改:

-  ``min_data_in_leaf`` = 100 => 20
-  ``min_sum_hessian_in_leaf`` = 10 => 1e-3
-  ``num_leaves`` = 127 => 31
-  ``num_iterations`` = 10 => 100

参数格式
-----------------

参数的格式为 ``key1=value1 key2=value2 ...``.
并且，在配置文件和命令行中均可以设置参数.
使用命令行设置参数时，在 ``=`` 前后都不应该有空格.
使用配置文件设置参数时, 一行只能包含一个参数. 你可以使用 ``#`` 进行注释.

如果一个参数在命令行和配置文件中均出现了, LightGBM 将会使用命令行中的该参数.

核心参数
---------------

-  ``config``, default=\ ``""``, type=string, alias=\ ``config_file``

   -  配置文件的路径

-  ``task``, default=\ ``train``, type=enum, options=\ ``train``, ``predict``, ``convert_model``

   -  ``train``, alias=\ ``training``, for training

   -  ``predict``, alias=\ ``prediction``, ``test``, for prediction.

   -  ``convert_model``, 要将模型文件转换成 if-else 格式, 可以查看这个链接获取更多信息 `Convert model parameters <#convert-model-parameters>`__

-  ``application``, default=\ ``regression``, type=enum,
   options=\ ``regression``, ``regression_l1``, ``huber``, ``fair``, ``poisson``, ``quantile``, ``quantile_l2``,
   ``binary``, ``multiclass``, ``multiclassova``, ``xentropy``, ``xentlambda``, ``lambdarank``,
   alias=\ ``objective``, ``app``

   -  regression application

      -  ``regression_l2``, L2 loss, alias=\ ``regression``, ``mean_squared_error``, ``mse``

      -  ``regression_l1``, L1 loss, alias=\ ``mean_absolute_error``, ``mae``

      -  ``huber``, `Huber loss`_

      -  ``fair``, `Fair loss`_

      -  ``poisson``, `Poisson regression`_

      -  ``quantile``, `Quantile regression`_

      -  ``quantile_l2``, 类似于 ``quantile``, 但是使用了 L2 loss 

   -  ``binary``, binary `log loss`_ classification application

   -  multi-class classification application

      -  ``multiclass``, `softmax`_ 目标函数, 应该设置好 ``num_class`` 

      -  ``multiclassova``, `One-vs-All`_ 二分类目标函数, 应该设置好 ``num_class`` 

   -  cross-entropy application

      -  ``xentropy``, 目标函数为 cross-entropy (同时有可选择的线性权重), alias=\ ``cross_entropy``

      -  ``xentlambda``, 替代参数化的 cross-entropy, alias=\ ``cross_entropy_lambda``

      -  标签是 [0, 1] 间隔内的任意值

   -  ``lambdarank``, `lambdarank`_ application

      -  在 lambdarank 任务中标签应该为 ``int`` 类型, 数值越大代表相关性越高 (e.g. 0:bad, 1:fair, 2:good, 3:perfect)

      -  ``label_gain`` 可以被用来设置 ``int`` 标签的增益 (权重)

-  ``boosting``, default=\ ``gbdt``, type=enum,
   options=\ ``gbdt``, ``rf``, ``dart``, ``goss``,
   alias=\ ``boost``, ``boosting_type``

   -  ``gbdt``, 传统的梯度提升决策树

   -  ``rf``, Random Forest (随机森林)

   -  ``dart``, `Dropouts meet Multiple Additive Regression Trees`_

   -  ``goss``, Gradient-based One-Side Sampling (基于梯度的单侧采样)

-  ``data``, default=\ ``""``, type=string, alias=\ ``train``, ``train_data``

   -  训练数据, LightGBM 将会使用这个数据进行训练

-  ``valid``, default=\ ``""``, type=multi-string, alias=\ ``test``, ``valid_data``, ``test_data``

   -  验证/测试 数据, LightGBM 将输出这些数据的度量

   -  支持多验证数据集, 以 ``,`` 分割

-  ``num_iterations``, default=\ ``100``, type=int,
   alias=\ ``num_iteration``, ``num_tree``, ``num_trees``, ``num_round``, ``num_rounds``, ``num_boost_round``

   -  boosting 的迭代次数

   -  **Note**: 对于 Python/R 包, **这个参数是被忽略的**,
      使用 ``train`` and ``cv`` 的输入参数 ``num_boost_round`` (Python) or ``nrounds`` (R) 来代替

   -  **Note**: 在内部, LightGBM 对于 ``multiclass`` 问题设置 ``num_class * num_iterations`` 棵树

-  ``learning_rate``, default=\ ``0.1``, type=double, alias=\ ``shrinkage_rate``

   -  shrinkage rate (收缩率)

   -  在 ``dart`` 中, 它还影响了 dropped trees 的归一化权重

-  ``num_leaves``, default=\ ``31``, type=int, alias=\ ``num_leaf``

   -  一棵树上的叶子数

-  ``tree_learner``, default=\ ``serial``, type=enum, options=\ ``serial``, ``feature``, ``data``, ``voting``, alias=\ ``tree``

   -  ``serial``, 单台机器的 tree learner

   -  ``feature``, alias=\ ``feature_parallel``, 特征并行的 tree learner

   -  ``data``, alias=\ ``data_parallel``, 数据并行的 tree learner

   -  ``voting``, alias=\ ``voting_parallel``, 投票并行的 tree learner

   -  请阅读 `Parallel Learning Guide <./Parallel-Learning-Guide.rst>`__ 来了解更多细节

-  ``num_threads``, default=\ ``OpenMP_default``, type=int, alias=\ ``num_thread``, ``nthread``

   -  LightGBM 的线程数

   -  为了更快的速度，将此设置为真正的CPU内核数，而不是线程的数量 (大多数CPU使用超线程来使每个CPU内核生成2个线程)

   -  当你的数据集小的时候不要将它设置的过大 (比如，当数据集有10,000行时不要使用64线程)

   -  请注意，任务管理器或任何类似的CPU监视工具可能会报告未被充分利用的内核. **这是正常的**

   -  对于并行学习，不应该使用全部的CPU内核，因为这会导致网络性能不佳

-  ``device``, default=\ ``cpu``, options=\ ``cpu``, ``gpu``

   -  为树学习选择设备，你可以使用 GPU 来获得更快的学习速度

   -  **Note**: 建议使用较小的 ``max_bin`` (e.g. 63) 来获得更快的速度

   -  **Note**: 为了加快学习速度， GPU 默认使用32位浮点数来求和.
      你可以设置 ``gpu_use_dp=true`` 来启用64位浮点数, 但是它会使训练速度降低

   -  **Note**: 请参考 `Installation Guide <./Installation-Guide.rst#build-gpu-version>`__ 来构建 GPU 版本

学习控制参数
---------------------------

-  ``max_depth``, default=\ ``-1``, type=int

   -  限制树模型的最大深度. 这可以在 ``#data`` 小的情况下防止过拟合. 树仍然可以通过 leaf-wise 生长.

   -  ``< 0`` 意味着没有限制.

-  ``min_data_in_leaf``, default=\ ``20``, type=int, alias=\ ``min_data_per_leaf`` , ``min_data``, ``min_child_samples``

   -  一个叶子上数据的最小数量. 可以用来处理过拟合.

-  ``min_sum_hessian_in_leaf``, default=\ ``1e-3``, type=double,
   alias=\ ``min_sum_hessian_per_leaf``, ``min_sum_hessian``, ``min_hessian``, ``min_child_weight``

   -  一个叶子上的最小 hessian 和. 类似于 ``min_data_in_leaf``, 可以用来处理过拟合.

-  ``feature_fraction``, default=\ ``1.0``, type=double, ``0.0 < feature_fraction < 1.0``, alias=\ ``sub_feature``, ``colsample_bytree``

   -  如果 ``feature_fraction`` 小于 ``1.0``， LightGBM 将会在每次迭代中随机选择部分特征.
      例如, 如果设置为 ``0.8``, 将会在每棵树训练之前选择 80% 的特征

   -  可以用来加速训练

   -  可以用来处理过拟合

-  ``feature_fraction_seed``, default=\ ``2``, type=int

   -  ``feature_fraction`` 的随机数种子

-  ``bagging_fraction``, default=\ ``1.0``, type=double, ``0.0 < bagging_fraction < 1.0``, alias=\ ``sub_row``, ``subsample``

   -  类似于 ``feature_fraction``, 但是它将在不进行重采样的情况下随机选择部分数据

   -  可以用来加速训练

   -  可以用来处理过拟合

   -  **Note**: 为了启用 bagging, ``bagging_freq`` 应该设置为非零值

-  ``bagging_freq``, default=\ ``0``, type=int, alias=\ ``subsample_freq``

   -  bagging 的频率, ``0`` 意味着禁用 bagging. ``k`` 意味着每 ``k`` 次迭代执行bagging

   -  **Note**: 为了启用 bagging, ``bagging_fraction`` 设置适当

-  ``bagging_seed`` , default=\ ``3``, type=int, alias=\ ``bagging_fraction_seed``

   -  bagging 随机数种子

-  ``early_stopping_round``, default=\ ``0``, type=int, alias=\ ``early_stopping_rounds``, ``early_stopping``

   -  如果一个验证集的度量在 ``early_stopping_round`` 循环中没有提升，将停止训练

-  ``lambda_l1``, default=\ ``0``, type=double, alias=\ ``reg_alpha``

   -  L1 正则

-  ``lambda_l2``, default=\ ``0``, type=double, alias=\ ``reg_lambda``

   -  L2 正则

-  ``min_split_gain``, default=\ ``0``, type=double, alias=\ ``min_gain_to_split``

   -  执行切分的最小增益

-  ``drop_rate``, default=\ ``0.1``, type=double

   -  仅仅在 ``dart`` 时使用

-  ``skip_drop``, default=\ ``0.5``, type=double

   -  仅仅在 ``dart`` 时使用, 跳过 drop 的概率

-  ``max_drop``, default=\ ``50``, type=int

   -  仅仅在 ``dart`` 时使用, 一次迭代中删除树的最大数量
   
   -  ``<=0`` 意味着没有限制

-  ``uniform_drop``, default=\ ``false``, type=bool

   -  仅仅在 ``dart`` 时使用, 如果想要均匀的删除，将它设置为 ``true`` 

-  ``xgboost_dart_mode``, default=\ ``false``, type=bool

   -  仅仅在 ``dart`` 时使用, 如果想要使用 xgboost dart 模式，将它设置为 ``true``  

-  ``drop_seed``, default=\ ``4``, type=int

   -  仅仅在 ``dart`` 时使用, 选择 dropping models 的随机数种子

-  ``top_rate``, default=\ ``0.2``, type=double

   -  仅仅在 ``goss`` 时使用, 大梯度数据的保留比例

-  ``other_rate``, default=\ ``0.1``, type=int

   -  仅仅在 ``goss`` 时使用, 小梯度数据的保留比例

-  ``min_data_per_group``, default=\ ``100``, type=int

   -  每个分类组的最小数据量

-  ``max_cat_threshold``, default=\ ``32``, type=int

   -  用于分类特征

   -  限制分类特征的最大阈值

-  ``cat_smooth``, default=\ ``10``, type=double

   -  用于分类特征

   -  这可以降低噪声在分类特征中的影响, 尤其是对数据很少的类别

-  ``cat_l2``, default=\ ``10``, type=double

   -  分类切分中的 L2 正则

-  ``max_cat_to_onehot``, default=\ ``4``, type=int

   -  当一个特征的类别数小于或等于 ``max_cat_to_onehot`` 时, one-vs-other 切分算法将会被使用

-  ``top_k``, default=\ ``20``, type=int, alias=\ ``topk``

   -  被使用在 `Voting parallel <./Parallel-Learning-Guide.rst#choose-appropriate-parallel-algorithm>`__ 中

   -  将它设置为更大的值可以获得更精确的结果，但会减慢训练速度

IO Parameters
-------------

-  ``max_bin``, default=\ ``255``, type=int

   -  max number of bins that feature values will be bucketed in.
      Small number of bins may reduce training accuracy but may increase general power (deal with over-fitting)

   -  LightGBM will auto compress memory according ``max_bin``.
      For example, LightGBM will use ``uint8_t`` for feature value if ``max_bin=255``

-  ``min_data_in_bin``, default=\ ``3``, type=int

   -  min number of data inside one bin, use this to avoid one-data-one-bin (may over-fitting)

-  ``data_random_seed``, default=\ ``1``, type=int

   -  random seed for data partition in parallel learning (not include feature parallel)

-  ``output_model``, default=\ ``LightGBM_model.txt``, type=string, alias=\ ``model_output``, ``model_out``

   -  file name of output model in training

-  ``input_model``, default=\ ``""``, type=string, alias=\ ``model_input``, ``model_in``

   -  file name of input model

   -  for ``prediction`` task, this model will be used for prediction data

   -  for ``train`` task, training will be continued from this model

-  ``output_result``, default=\ ``LightGBM_predict_result.txt``,
   type=string, alias=\ ``predict_result``, ``prediction_result``

   -  file name of prediction result in ``prediction`` task

-  ``model_format``, default=\ ``text``, type=multi-enum, options=\ ``text``, ``proto``

   -  format to save and load model

   -  if ``text``, text string will be used

   -  if ``proto``, Protocol Buffer binary format will be used

   -  you can save in multiple formats by joining them with comma, like ``text,proto``. In this case, ``model_format`` will be add as suffix after ``output_model``

   -  **Note**: loading with multiple formats is not supported

   -  **Note**: to use this parameter you need to `build version with Protobuf Support <./Installation-Guide.rst#protobuf-support>`__

-  ``pre_partition``, default=\ ``false``, type=bool, alias=\ ``is_pre_partition``

   -  used for parallel learning (not include feature parallel)

   -  ``true`` if training data are pre-partitioned, and different machines use different partitions

-  ``is_sparse``, default=\ ``true``, type=bool, alias=\ ``is_enable_sparse``, ``enable_sparse``

   -  used to enable/disable sparse optimization. Set to ``false`` to disable sparse optimization

-  ``two_round``, default=\ ``false``, type=bool, alias=\ ``two_round_loading``, ``use_two_round_loading``

   -  by default, LightGBM will map data file to memory and load features from memory.
      This will provide faster data loading speed. But it may run out of memory when the data file is very big

   -  set this to ``true`` if data file is too big to fit in memory

-  ``save_binary``, default=\ ``false``, type=bool, alias=\ ``is_save_binary``, ``is_save_binary_file``

   -  if ``true`` LightGBM will save the dataset (include validation data) to a binary file.
      Speed up the data loading for the next time

-  ``verbosity``, default=\ ``1``, type=int, alias=\ ``verbose``

   -  ``<0`` = Fatal,
      ``=0`` = Error (Warn),
      ``>0`` = Info

-  ``header``, default=\ ``false``, type=bool, alias=\ ``has_header``

   -  set this to ``true`` if input data has header

-  ``label``, default=\ ``""``, type=string, alias=\ ``label_column``

   -  specify the label column

   -  use number for index, e.g. ``label=0`` means column\_0 is the label

   -  add a prefix ``name:`` for column name, e.g. ``label=name:is_click``

-  ``weight``, default=\ ``""``, type=string, alias=\ ``weight_column``

   -  specify the weight column

   -  use number for index, e.g. ``weight=0`` means column\_0 is the weight

   -  add a prefix ``name:`` for column name, e.g. ``weight=name:weight``

   -  **Note**: index starts from ``0``.
      And it doesn't count the label column when passing type is Index, e.g. when label is column\_0, and weight is column\_1, the correct parameter is ``weight=0``

-  ``query``, default=\ ``""``, type=string, alias=\ ``query_column``, ``group``, ``group_column``

   -  specify the query/group id column

   -  use number for index, e.g. ``query=0`` means column\_0 is the query id

   -  add a prefix ``name:`` for column name, e.g. ``query=name:query_id``

   -  **Note**: data should be grouped by query\_id.
      Index starts from ``0``.
      And it doesn't count the label column when passing type is Index, e.g. when label is column\_0 and query\_id is column\_1, the correct parameter is ``query=0``

-  ``ignore_column``, default=\ ``""``, type=string, alias=\ ``ignore_feature``, ``blacklist``

   -  specify some ignoring columns in training

   -  use number for index, e.g. ``ignore_column=0,1,2`` means column\_0, column\_1 and column\_2 will be ignored

   -  add a prefix ``name:`` for column name, e.g. ``ignore_column=name:c1,c2,c3`` means c1, c2 and c3 will be ignored

   -  **Note**: works only in case of loading data directly from file

   -  **Note**: index starts from ``0``. And it doesn't count the label column

-  ``categorical_feature``, default=\ ``""``, type=string, alias=\ ``categorical_column``, ``cat_feature``, ``cat_column``

   -  specify categorical features

   -  use number for index, e.g. ``categorical_feature=0,1,2`` means column\_0, column\_1 and column\_2 are categorical features

   -  add a prefix ``name:`` for column name, e.g. ``categorical_feature=name:c1,c2,c3`` means c1, c2 and c3 are categorical features

   -  **Note**: only supports categorical with ``int`` type. Index starts from ``0``. And it doesn't count the label column

   -  **Note**: the negative values will be treated as **missing values**

-  ``predict_raw_score``, default=\ ``false``, type=bool, alias=\ ``raw_score``, ``is_predict_raw_score``

   -  only used in ``prediction`` task

   -  set to ``true`` to predict only the raw scores

   -  set to ``false`` to predict transformed scores

-  ``predict_leaf_index``, default=\ ``false``, type=bool, alias=\ ``leaf_index``, ``is_predict_leaf_index``

   -  only used in ``prediction`` task

   -  set to ``true`` to predict with leaf index of all trees

-  ``predict_contrib``, default=\ ``false``, type=bool, alias=\ ``contrib``, ``is_predict_contrib``

   -  only used in ``prediction`` task

   -  set to ``true`` to estimate `SHAP values`_, which represent how each feature contributs to each prediction.
      Produces number of features + 1 values where the last value is the expected value of the model output over the training data

-  ``bin_construct_sample_cnt``, default=\ ``200000``, type=int, alias=\ ``subsample_for_bin``

   -  number of data that sampled to construct histogram bins

   -  will give better training result when set this larger, but will increase data loading time

   -  set this to larger value if data is very sparse

-  ``num_iteration_predict``, default=\ ``-1``, type=int

   -  only used in ``prediction`` task
   -  use to specify how many trained iterations will be used in prediction

   -  ``<= 0`` means no limit

-  ``pred_early_stop``, default=\ ``false``, type=bool

   -  if ``true`` will use early-stopping to speed up the prediction. May affect the accuracy

-  ``pred_early_stop_freq``, default=\ ``10``, type=int

   -  the frequency of checking early-stopping prediction

-  ``pred_early_stop_margin``, default=\ ``10.0``, type=double

   -  the threshold of margin in early-stopping prediction

-  ``use_missing``, default=\ ``true``, type=bool

   -  set to ``false`` to disable the special handle of missing value

-  ``zero_as_missing``, default=\ ``false``, type=bool

   -  set to ``true`` to treat all zero as missing values (including the unshown values in libsvm/sparse matrics)

   -  set to ``false`` to use ``na`` to represent missing values

-  ``init_score_file``, default=\ ``""``, type=string

   -  path to training initial score file, ``""`` will use ``train_data_file`` + ``.init`` (if exists)

-  ``valid_init_score_file``, default=\ ``""``, type=multi-string

   -  path to validation initial score file, ``""`` will use ``valid_data_file`` + ``.init`` (if exists)

   -  separate by ``,`` for multi-validation data

Objective Parameters
--------------------

-  ``sigmoid``, default=\ ``1.0``, type=double

   -  parameter for sigmoid function. Will be used in ``binary`` classification and ``lambdarank``

-  ``alpha``, default=\ ``0.9``, type=double

   -  parameter for `Huber loss`_ and `Quantile regression`_. Will be used in ``regression`` task

-  ``fair_c``, default=\ ``1.0``, type=double

   -  parameter for `Fair loss`_. Will be used in ``regression`` task

-  ``gaussian_eta``, default=\ ``1.0``, type=double

   -  parameter to control the width of Gaussian function. Will be used in ``regression_l1`` and ``huber`` losses

-  ``poisson_max_delta_step``, default=\ ``0.7``, type=double

   -  parameter for `Poisson regression`_ to safeguard optimization

-  ``scale_pos_weight``, default=\ ``1.0``, type=double

   -  weight of positive class in ``binary`` classification task

-  ``boost_from_average``, default=\ ``true``, type=bool

   -  only used in ``regression`` task

   -  adjust initial score to the mean of labels for faster convergence

-  ``is_unbalance``, default=\ ``false``, type=bool, alias=\ ``unbalanced_sets``

   -  used in ``binary`` classification
   
   -  set this to ``true`` if training data are unbalance

-  ``max_position``, default=\ ``20``, type=int

   -  used in ``lambdarank``

   -  will optimize `NDCG`_ at this position

-  ``label_gain``, default=\ ``0,1,3,7,15,31,63,...``, type=multi-double

   -  used in ``lambdarank``

   -  relevant gain for labels. For example, the gain of label ``2`` is ``3`` if using default label gains

   -  separate by ``,``

-  ``num_class``, default=\ ``1``, type=int, alias=\ ``num_classes``

   -  only used in ``multiclass`` classification

-  ``reg_sqrt``, default=\ ``false``, type=bool

   -  only used in ``regression``

   -  will fit ``sqrt(label)`` instead and prediction result will be also automatically converted to ``pow2(prediction)``

Metric Parameters
-----------------

-  ``metric``, default={``l2`` for regression}, {``binary_logloss`` for binary classification}, {``ndcg`` for lambdarank}, type=multi-enum,
   options=\ ``l1``, ``l2``, ``ndcg``, ``auc``, ``binary_logloss``, ``binary_error`` ...

   -  ``l1``, absolute loss, alias=\ ``mean_absolute_error``, ``mae``

   -  ``l2``, square loss, alias=\ ``mean_squared_error``, ``mse``

   -  ``l2_root``, root square loss, alias=\ ``root_mean_squared_error``, ``rmse``

   -  ``quantile``, `Quantile regression`_

   -  ``huber``, `Huber loss`_

   -  ``fair``, `Fair loss`_

   -  ``poisson``, `Poisson regression`_

   -  ``ndcg``, `NDCG`_

   -  ``map``, `MAP`_

   -  ``auc``, `AUC`_

   -  ``binary_logloss``, `log loss`_

   -  ``binary_error``, for one sample: ``0`` for correct classification, ``1`` for error classification

   -  ``multi_logloss``, log loss for mulit-class classification

   -  ``multi_error``, error rate for mulit-class classification

   -  ``xentropy``, cross-entropy (with optional linear weights), alias=\ ``cross_entropy``

   -  ``xentlambda``, "intensity-weighted" cross-entropy, alias=\ ``cross_entropy_lambda``

   -  ``kldiv``, `Kullback-Leibler divergence`_, alias=\ ``kullback_leibler``

   -  support multi metrics, separated by ``,``

-  ``metric_freq``, default=\ ``1``, type=int

   -  frequency for metric output

-  ``train_metric``, default=\ ``false``, type=bool, alias=\ ``training_metric``, ``is_training_metric``

   -  set this to ``true`` if you need to output metric result of training

-  ``ndcg_at``, default=\ ``1,2,3,4,5``, type=multi-int, alias=\ ``ndcg_eval_at``, ``eval_at``

   -  `NDCG`_ evaluation positions, separated by ``,``

Network Parameters
------------------

Following parameters are used for parallel learning, and only used for base (socket) version.

-  ``num_machines``, default=\ ``1``, type=int, alias=\ ``num_machine``

   -  used for parallel learning, the number of machines for parallel learning application

   -  need to set this in both socket and mpi versions

-  ``local_listen_port``, default=\ ``12400``, type=int, alias=\ ``local_port``

   -  TCP listen port for local machines

   -  you should allow this port in firewall settings before training

-  ``time_out``, default=\ ``120``, type=int

   -  socket time-out in minutes

-  ``machine_list_file``, default=\ ``""``, type=string, alias=\ ``mlist``

   -  file that lists machines for this parallel learning application

   -  each line contains one IP and one port for one machine. The format is ``ip port``, separate by space

GPU Parameters
--------------

-  ``gpu_platform_id``, default=\ ``-1``, type=int

   -  OpenCL platform ID. Usually each GPU vendor exposes one OpenCL platform.

   -  default value is ``-1``, means the system-wide default platform

-  ``gpu_device_id``, default=\ ``-1``, type=int

   -  OpenCL device ID in the specified platform. Each GPU in the selected platform has a unique device ID

   -  default value is ``-1``, means the default device in the selected platform

-  ``gpu_use_dp``, default=\ ``false``, type=bool

   -  set to ``true`` to use double precision math on GPU (default using single precision)
  
Convert Model Parameters
------------------------

This feature is only supported in command line version yet.

-  ``convert_model_language``, default=\ ``""``, type=string

   -  only ``cpp`` is supported yet

   -  if ``convert_model_language`` is set when ``task`` is set to ``train``, the model will also be converted

-  ``convert_model``, default=\ ``"gbdt_prediction.cpp"``, type=string

   -  output file name of converted model

Others
------

Continued Training with Input Score
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LightGBM supports continued training with initial scores. It uses an additional file to store these initial scores, like the following:

::

    0.5
    -0.1
    0.9
    ...

It means the initial score of the first data row is ``0.5``, second is ``-0.1``, and so on.
The initial score file corresponds with data file line by line, and has per score per line.
And if the name of data file is ``train.txt``, the initial score file should be named as ``train.txt.init`` and in the same folder as the data file.
In this case LightGBM will auto load initial score file if it exists.

Weight Data
~~~~~~~~~~~

LightGBM supporta weighted training. It uses an additional file to store weight data, like the following:

::

    1.0
    0.5
    0.8
    ...

It means the weight of the first data row is ``1.0``, second is ``0.5``, and so on.
The weight file corresponds with data file line by line, and has per weight per line.
And if the name of data file is ``train.txt``, the weight file should be named as ``train.txt.weight`` and in the same folder as the data file.
In this case LightGBM will auto load weight file if it exists.

**update**:
You can specific weight column in data file now. Please refer to parameter ``weight`` in above.

Query Data
~~~~~~~~~~

For LambdaRank learning, it needs query information for training data.
LightGBM use an additional file to store query data, like the following:

::

    27
    18
    67
    ...

It means first ``27`` lines samples belong one query and next ``18`` lines belong to another, and so on.

**Note**: data should be ordered by the query.

If the name of data file is ``train.txt``, the query file should be named as ``train.txt.query`` and in same folder of training data.
In this case LightGBM will load the query file automatically if it exists.

**update**:
You can specific query/group id in data file now. Please refer to parameter ``group`` in above.

.. _Laurae++ Interactive Documentation: https://sites.google.com/view/lauraepp/parameters

.. _Huber loss: https://en.wikipedia.org/wiki/Huber_loss

.. _Quantile regression: https://en.wikipedia.org/wiki/Quantile_regression

.. _Fair loss: https://www.kaggle.com/c/allstate-claims-severity/discussion/24520

.. _Poisson regression: https://en.wikipedia.org/wiki/Poisson_regression

.. _lambdarank: https://papers.nips.cc/paper/2971-learning-to-rank-with-nonsmooth-cost-functions.pdf

.. _Dropouts meet Multiple Additive Regression Trees: https://arxiv.org/abs/1505.01866

.. _hyper-threading: https://en.wikipedia.org/wiki/Hyper-threading

.. _SHAP values: https://arxiv.org/abs/1706.06060

.. _NDCG: https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG

.. _MAP: https://en.wikipedia.org/wiki/Information_retrieval#Mean_average_precision

.. _AUC: https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve

.. _log loss: https://www.kaggle.com/wiki/LogLoss

.. _softmax: https://en.wikipedia.org/wiki/Softmax_function

.. _One-vs-All: https://en.wikipedia.org/wiki/Multiclass_classification#One-vs.-rest

.. _Kullback-Leibler divergence: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
