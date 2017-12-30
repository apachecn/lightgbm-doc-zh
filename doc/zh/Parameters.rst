Parameters
==========

This page contains all parameters in LightGBM.

**List of other helpful links**

- `Python API <./Python-API.rst>`__

- `Parameters Tuning <./Parameters-Tuning.rst>`__

**External Links**

- `Laurae++ Interactive Documentation`_

**Update of 08/04/2017**

Default values for the following parameters have changed:

-  ``min_data_in_leaf`` = 100 => 20
-  ``min_sum_hessian_in_leaf`` = 10 => 1e-3
-  ``num_leaves`` = 127 => 31
-  ``num_iterations`` = 10 => 100

Parameters Format
-----------------

The parameters format is ``key1=value1 key2=value2 ...``.
And parameters can be set both in config file and command line.
By using command line, parameters should not have spaces before and after ``=``.
By using config files, one line can only contain one parameter. You can use ``#`` to comment.

If one parameter appears in both command line and config file, LightGBM will use the parameter in command line.

Core Parameters
---------------

-  ``config``, default=\ ``""``, type=string, alias=\ ``config_file``

   -  path of config file

-  ``task``, default=\ ``train``, type=enum, options=\ ``train``, ``predict``, ``convert_model``

   -  ``train``, alias=\ ``training``, for training

   -  ``predict``, alias=\ ``prediction``, ``test``, for prediction.

   -  ``convert_model``, for converting model file into if-else format, see more information in `Convert model parameters <#convert-model-parameters>`__

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

      -  ``quantile_l2``, like the ``quantile``, but L2 loss is used instead

   -  ``binary``, binary `log loss`_ classification application

   -  multi-class classification application

      -  ``multiclass``, `softmax`_ objective function, ``num_class`` should be set as well

      -  ``multiclassova``, `One-vs-All`_ binary objective function, ``num_class`` should be set as well

   -  cross-entropy application

      -  ``xentropy``, objective function for cross-entropy (with optional linear weights), alias=\ ``cross_entropy``

      -  ``xentlambda``, alternative parameterization of cross-entropy, alias=\ ``cross_entropy_lambda``

      -  the label is anything in interval [0, 1]

   -  ``lambdarank``, `lambdarank`_ application

      -  the label should be ``int`` type in lambdarank tasks, and larger number represent the higher relevance (e.g. 0:bad, 1:fair, 2:good, 3:perfect)

      -  ``label_gain`` can be used to set the gain(weight) of ``int`` label

-  ``boosting``, default=\ ``gbdt``, type=enum,
   options=\ ``gbdt``, ``rf``, ``dart``, ``goss``,
   alias=\ ``boost``, ``boosting_type``

   -  ``gbdt``, traditional Gradient Boosting Decision Tree

   -  ``rf``, Random Forest

   -  ``dart``, `Dropouts meet Multiple Additive Regression Trees`_

   -  ``goss``, Gradient-based One-Side Sampling

-  ``data``, default=\ ``""``, type=string, alias=\ ``train``, ``train_data``

   -  training data, LightGBM will train from this data

-  ``valid``, default=\ ``""``, type=multi-string, alias=\ ``test``, ``valid_data``, ``test_data``

   -  validation/test data, LightGBM will output metrics for these data

   -  support multi validation data, separate by ``,``

-  ``num_iterations``, default=\ ``100``, type=int,
   alias=\ ``num_iteration``, ``num_tree``, ``num_trees``, ``num_round``, ``num_rounds``, ``num_boost_round``

   -  number of boosting iterations

   -  **Note**: for Python/R package, **this parameter is ignored**,
      use ``num_boost_round`` (Python) or ``nrounds`` (R) input arguments of ``train`` and ``cv`` methods instead

   -  **Note**: internally, LightGBM constructs ``num_class * num_iterations`` trees for ``multiclass`` problems

-  ``learning_rate``, default=\ ``0.1``, type=double, alias=\ ``shrinkage_rate``

   -  shrinkage rate

   -  in ``dart``, it also affects on normalization weights of dropped trees

-  ``num_leaves``, default=\ ``31``, type=int, alias=\ ``num_leaf``

   -  number of leaves in one tree

-  ``tree_learner``, default=\ ``serial``, type=enum, options=\ ``serial``, ``feature``, ``data``, ``voting``, alias=\ ``tree``

   -  ``serial``, single machine tree learner

   -  ``feature``, alias=\ ``feature_parallel``, feature parallel tree learner

   -  ``data``, alias=\ ``data_parallel``, data parallel tree learner

   -  ``voting``, alias=\ ``voting_parallel``, voting parallel tree learner

   -  refer to `Parallel Learning Guide <./Parallel-Learning-Guide.rst>`__ to get more details

-  ``num_threads``, default=\ ``OpenMP_default``, type=int, alias=\ ``num_thread``, ``nthread``

   -  number of threads for LightGBM

   -  for the best speed, set this to the number of **real CPU cores**,
      not the number of threads (most CPU using `hyper-threading`_ to generate 2 threads per CPU core)

   -  do not set it too large if your dataset is small (do not use 64 threads for a dataset with 10,000 rows for instance)

   -  be aware a task manager or any similar CPU monitoring tool might report cores not being fully utilized. **This is normal**

   -  for parallel learning, should not use full CPU cores since this will cause poor performance for the network

-  ``device``, default=\ ``cpu``, options=\ ``cpu``, ``gpu``

   -  choose device for the tree learning, you can use GPU to achieve the faster learning

   -  **Note**: it is recommended to use the smaller ``max_bin`` (e.g. 63) to get the better speed up

   -  **Note**: for the faster speed, GPU use 32-bit float point to sum up by default, may affect the accuracy for some tasks.
      You can set ``gpu_use_dp=true`` to enable 64-bit float point, but it will slow down the training

   -  **Note**: refer to `Installation Guide <./Installation-Guide.rst#build-gpu-version>`__ to build with GPU

Learning Control Parameters
---------------------------

-  ``max_depth``, default=\ ``-1``, type=int

   -  limit the max depth for tree model. This is used to deal with over-fitting when ``#data`` is small. Tree still grows by leaf-wise

   -  ``< 0`` means no limit

-  ``min_data_in_leaf``, default=\ ``20``, type=int, alias=\ ``min_data_per_leaf`` , ``min_data``, ``min_child_samples``

   -  minimal number of data in one leaf. Can be used to deal with over-fitting

-  ``min_sum_hessian_in_leaf``, default=\ ``1e-3``, type=double,
   alias=\ ``min_sum_hessian_per_leaf``, ``min_sum_hessian``, ``min_hessian``, ``min_child_weight``

   -  minimal sum hessian in one leaf. Like ``min_data_in_leaf``, it can be used to deal with over-fitting

-  ``feature_fraction``, default=\ ``1.0``, type=double, ``0.0 < feature_fraction < 1.0``, alias=\ ``sub_feature``, ``colsample_bytree``

   -  LightGBM will randomly select part of features on each iteration if ``feature_fraction`` smaller than ``1.0``.
      For example, if set to ``0.8``, will select 80% features before training each tree

   -  can be used to speed up training

   -  can be used to deal with over-fitting

-  ``feature_fraction_seed``, default=\ ``2``, type=int

   -  random seed for ``feature_fraction``

-  ``bagging_fraction``, default=\ ``1.0``, type=double, ``0.0 < bagging_fraction < 1.0``, alias=\ ``sub_row``, ``subsample``

   -  like ``feature_fraction``, but this will randomly select part of data without resampling

   -  can be used to speed up training

   -  can be used to deal with over-fitting

   -  **Note**: To enable bagging, ``bagging_freq`` should be set to a non zero value as well

-  ``bagging_freq``, default=\ ``0``, type=int, alias=\ ``subsample_freq``

   -  frequency for bagging, ``0`` means disable bagging. ``k`` means will perform bagging at every ``k`` iteration

   -  **Note**: to enable bagging, ``bagging_fraction`` should be set as well

-  ``bagging_seed`` , default=\ ``3``, type=int, alias=\ ``bagging_fraction_seed``

   -  random seed for bagging

-  ``early_stopping_round``, default=\ ``0``, type=int, alias=\ ``early_stopping_rounds``, ``early_stopping``

   -  will stop training if one metric of one validation data doesn't improve in last ``early_stopping_round`` rounds

-  ``lambda_l1``, default=\ ``0``, type=double, alias=\ ``reg_alpha``

   -  L1 regularization

-  ``lambda_l2``, default=\ ``0``, type=double, alias=\ ``reg_lambda``

   -  L2 regularization

-  ``min_split_gain``, default=\ ``0``, type=double, alias=\ ``min_gain_to_split``

   -  the minimal gain to perform split

-  ``drop_rate``, default=\ ``0.1``, type=double

   -  only used in ``dart``

-  ``skip_drop``, default=\ ``0.5``, type=double

   -  only used in ``dart``, probability of skipping drop

-  ``max_drop``, default=\ ``50``, type=int

   -  only used in ``dart``, max number of dropped trees on one iteration
   
   -  ``<=0`` means no limit

-  ``uniform_drop``, default=\ ``false``, type=bool

   -  only used in ``dart``, set this to ``true`` if want to use uniform drop

-  ``xgboost_dart_mode``, default=\ ``false``, type=bool

   -  only used in ``dart``, set this to ``true`` if want to use xgboost dart mode

-  ``drop_seed``, default=\ ``4``, type=int

   -  only used in ``dart``, random seed to choose dropping models

-  ``top_rate``, default=\ ``0.2``, type=double

   -  only used in ``goss``, the retain ratio of large gradient data

-  ``other_rate``, default=\ ``0.1``, type=int

   -  only used in ``goss``, the retain ratio of small gradient data

-  ``min_data_per_group``, default=\ ``100``, type=int

   -  min number of data per categorical group

-  ``max_cat_threshold``, default=\ ``32``, type=int

   -  use for the categorical features

   -  limit the max threshold points in categorical features

-  ``cat_smooth``, default=\ ``10``, type=double

   -  used for the categorical features

   -  this can reduce the effect of noises in categorical features, especially for categories with few data

-  ``cat_l2``, default=\ ``10``, type=double

   -  L2 regularization in categorcial split

-  ``max_cat_to_onehot``, default=\ ``4``, type=int

   -  when number of categories of one feature smaller than or equal to ``max_cat_to_onehot``, one-vs-other split algorithm will be used

-  ``top_k``, default=\ ``20``, type=int, alias=\ ``topk``

   -  used in `Voting parallel <./Parallel-Learning-Guide.rst#choose-appropriate-parallel-algorithm>`__

   -  set this to larger value for more accurate result, but it will slow down the training speed

IO 参数
-------------
-  ``max_bin``, 默认值=\ ``255``, 类型=int

   -  工具箱的最大数特征值决定了容量
      工具箱的最小数特征值可能会降低训练的准确性，但是可能会增加一些一般的影响（处理过度学习）

   -  LightGBM将根据``max_bin``自动压缩内存。
      例如，如果maxbin=255，那么LightGBM将使用uint8t的特性值

-  ``max_bin``, 默认值=\ ``255``, 类型=int

-  ``min_data_in_bin``, 默认值=\ ``3``, 类型=int
   -  单个数据箱的最小数，使用此方法避免one-data-one-bin（可能会过度学习）

-  ``data_r和om_seed``, 默认值=\ ``1``, 类型=int

   -  并行学习数据分隔中的随机种子 (不包括并行功能)

-  ``output_model``, 默认值=\ ``LightGBM_model.txt``, 类型=string, 别名=\ ``model_output``, ``model_out``

   -  培训中输出的模型文件名

-  ``input_model``, 默认值=\ ``""``, 类型=string, 别名=\ ``model_input``, ``model_in``

   -  输入模型的文件名

   -  对于``prediction`` 任务, 该模型将用于预测数据

   -  对于 ``train`` 任务, 培训将从该模型继续

-  ``output_result``, 默认值=\ ``LightGBM_predict_result.txt``,
   类型=string, 别名=\ ``predict_result``, ``prediction_result``

   -  ``prediction`` 任务的预测结果文件名

-  ``model_format``, 默认值=\ ``text``, 类型=multi-enum, 可选项=\ ``text``, ``proto``

   -  保存和加载模型的格式

   -   ``text``, 使用文本字符串

   -   ``proto``, 使用协议缓冲二进制格式

   -  您可以通过使用逗号来进行多种格式的保存，例如 ``text,proto``. 在这种情况下, ``model_format`` 将作为后缀添加 ``output_model``

   -  **Note**: 不支持多种格式的加载

   -  **Note**: 要使用这个参数，您需要使用build 版本 <./Installation-Guide.rst#protobuf-support>`__

-  ``pre_partition``, 默认值=\ ``false``, 类型=bool, 别名=\ ``is_pre_partition``

   -  用于并行学习(不包括功能并行)

   -  ``true`` 如果训练数据 pre-partitioned, 不同的机器使用不同的分区

-  ``is_sparse``, 默认值=\ ``true``, 类型=bool, 别名=\ ``is_enable_sparse``, ``enable_sparse``

   -  用于 enable/disable 稀疏优化. 设置 ``false``就禁用稀疏优化

-  ``two_round``, 默认值=\ ``false``, 类型=bool, 别名=\ ``two_round_loading``, ``use_two_round_loading``

   -  默认情况下，LightGBM将把数据文件映射到内存，并从内存加载特性。
      这将提供更快的数据加载速度。但当数据文件很大时，内存可能会耗尽
   -  如果数据文件太大，不能放在内存中，就把它设置为``true``

-  ``save_binary``, 默认值=\ ``false``, 类型=bool, 别名=\ ``is_save_binary``, ``is_save_binary_file``

   -  如果设置为 ``true`` LightGBM则将数据集(包括验证数据)保存到二进制文件中。
      可以加快数据加载速度。

-  ``verbosity``, 默认值=\ ``1``, 类型=int, 别名=\ ``verbose``

   -  ``<0`` = 致命的,
      ``=0`` = 错误 (警告),
      ``>0`` = 信息

-  ``header``, 默认值=\ ``false``, 类型=bool, 别名=\ ``has_header``

   -  如果输入数据有标识头，则在此处设置``true``

-  ``label``, 默认值=\ ``""``, 类型=string, 别名=\ ``label_column``

   -  指定标签列

   -  用于索引的数字, e.g. ``label=0`` 意味着 column\_0 是标签列

   -  为列名添加前缀 ``name:`` , e.g. ``label=name:is_click``

-  ``weight``, 默认值=\ ``""``, 类型=string, 别名=\ ``weight_column``

   -  列的指定

   -  用于索引的数字, e.g. ``weight=0`` 表示 column\_0 是权重点

   -  为列名添加前缀 ``name:``, e.g. ``weight=name:weight``

   -  **Note**: 索引从 ``0`` 开始.
      当传递类型为索引时，它不计算标签列，例如当标签为0时，权重为列1，正确的参数是权重值为0

-  ``query``, 默认值=\ ``""``, 类型=string, 别名=\ ``query_column``, ``group``, ``group_column``

   -  指定 query/group ID列

   -  用数字做索引, e.g. ``query=0`` 意味着 column\_0 是这个查询的Id

   -  为列名添加前缀 ``name:`` , e.g. ``query=name:query_id``

   -  **Note**: 数据应按照 query\_id.
      索引从 ``0``开始.
      当传递类型为索引时，它不计算标签列，例如当标签为列0，查询id为列1时，正确的参数是查询=0

-  ``ignore_column``, 默认值=\ ``""``, 类型=string, 别名=\ ``ignore_feature``, ``blacklist``

   -  在培训中指定一些忽略的列

   -  用数字做索引, e.g. ``ignore_column=0,1,2`` 意味着 column\_0, column\_1 和 column\_2 将被忽略

   -  为列名添加前缀 ``name:`` , e.g. ``ignore_column=name:c1,c2,c3`` 意味着 c1, c2 和 c3 将被忽略

   -  **Note**: 只在从文件直接加载数据的情况下工作

   -  **Note**: 索引从 ``0`` 开始. 它不包括标签栏

-  ``categorical_feature``, 默认值=\ ``""``, 类型=string, 别名=\ ``categorical_column``, ``cat_feature``, ``cat_column``

   -  指定分类特征

   -  用数字做索引, e.g. ``categorical_feature=0,1,2`` 意味着 column\_0, column\_1 和 column\_2 是分类特征

   -  为列名添加前缀 ``name:``, e.g. ``categorical_feature=name:c1,c2,c3`` 意味着 c1, c2 和 c3 是分类特征

   -  **Note**: 只支持分类与 ``int`` 类型. 索引从 ``0`` 开始. 同时它不包括标签栏

   -  **Note**: 负值的值将被视为 **missing values**

-  ``predict_raw_score``, 默认值=\ ``false``, 类型=bool, 别名=\ ``raw_score``, ``is_predict_raw_score``

   -   只用于``prediction`` 任务

   -  设置为 ``true``只预测原始分数

   -  设置为 ``false`` 只预测分数

-  ``predict_leaf_index``, 默认值=\ ``false``, 类型=bool, 别名=\ ``leaf_index``, ``is_predict_leaf_index``

   -  只用于 ``prediction`` 任务

   -  设置为 ``true`` to predict with leaf index of all trees

-  ``predict_contrib``, 默认值=\ ``false``, 类型=bool, 别名=\ ``contrib``, ``is_predict_contrib``

   -  只用于 ``prediction`` 任务

   -  设置为 ``true`` 预估`SHAP values`_, 这代表了每个特性对每个预测的贡献。
      生成的特征+1的值，其中最后一个值是模型输出的预期值，而不是训练数据

-  ``bin_construct_sample_cnt``, 默认值=\ ``200000``, 类型=int, 别名=\ ``subsample_for_bin``

   -  用来构建直方图的数据的数量

   -  在设置更大的数据时，会提供更好的培训效果，但会增加数据加载时间

   -  如果数据非常稀疏，则将其设置为更大的值

-  ``num_iteration_predict``, 默认值=\ ``-1``, 类型=int

   -  只用于 ``prediction`` 任务
   -  用于指定在预测中使用多少经过培训的迭代

   -  ``<= 0`` 意味着没有限制

-  ``pred_early_stop``, 默认值=\ ``false``, 类型=bool

   - 如果``true``将使用提前停止来加速预测。可能影响精度

-  ``pred_early_stop_freq``, 默认值=\ ``10``, 类型=int

   - 检查早期early-stopping的频率

-  ``pred_early_stop_margin``, 默认值=\ ``10.0``, 类型=double

   -  t提前early-stopping的边际阈值

-  ``use_missing``, 默认值=\ ``true``, 类型=bool

   -  设置为 ``false`` 禁用丢失值的特殊句柄

-  ``zero_as_missing``, 默认值=\ ``false``, 类型=bool

   -  设置为 ``true`` 将所有的0都视为缺失的值 (包括 libsvm/sparse 矩阵中未显示的值)

   -  设置为 ``false`` 使用 ``na`` 代表缺失值

-  ``init_score_file``, 默认值=\ ``""``, 类型=string

   -  训练初始分数文件的路径, ``""`` 将使用 ``train_data_file`` + ``.init`` (如果存在)

-  ``valid_init_score_file``, 默认值=\ ``""``, 类型=multi-string

   -  验证初始分数文件的路径, ``""`` 将使用 ``valid_data_file`` + ``.init`` (如果存在)

   -  通过 ``,`` 对multi-validation进行分离

目标参数
--------------------

-  ``sigmoid``, 默认值=\ ``1.0``, 类型=double

   -  sigmoid 函数的参数. 将用于 ``binary`` 分类 和 ``lambdarank``

-  ``alpha``, 默认值=\ ``0.9``, 类型=double

   -   `Huber loss`_ 和 `Quantile regression`_ 的参数. 将用于``regression`` 任务

-  ``fair_c``, 默认值=\ ``1.0``, 类型=double

   -   `Fair loss`_ 的参数. 将用于 ``regression`` 任务

-  ``gaussian_eta``, 默认值=\ ``1.0``, 类型=double

   -  控制高斯函数的宽度的参数. 将用于``regression_l1`` 和 ``huber`` losses

-  ``poisson_max_delta_step``, 默认值=\ ``0.7``, 类型=double

   -  `Poisson regression`_ 的参数用于维护优化

-  ``scale_pos_weight``, 默认值=\ ``1.0``, 类型=double

   -  正值的权重 ``binary`` 分类 任务

-  ``boost_from_average``, 默认值=\ ``true``, 类型=bool

   -  只用于 ``regression`` 任务

   -  将初始分数调整为更快收敛速度的平均值

-  ``is_unbalance``, 默认值=\ ``false``, 类型=bool, 别名=\ ``unbalanced_sets``

   -  用于 ``binary`` 分类
   
   - 如果培训数据不平衡 设置为 ``true``

-  ``max_position``, 默认值=\ ``20``, 类型=int

   -  用于 ``lambdarank``

   -  将在这个`NDCG`_位置优化

-  ``label_gain``, 默认值=\ ``0,1,3,7,15,31,63,...``, 类型=multi-double

   -  用于 ``lambdarank``

   -  有关获得标签. 列如, 如果使用默认标签增益 这个``2``的标签则是``3``

   -  使用 ``,`` 分隔

-  ``num_class``, 默认值=\ ``1``, 类型=int, 别名=\ ``num_classes``

   -  只用于 ``multiclass`` 分类

-  ``reg_sqrt``, 默认值=\ ``false``, 类型=bool

   -  只用于 ``regression``
   
   -  适合``sqrt(label)``相反，预测结果也会自动转换成``pow2(prediction)``

度量参数
-----------------

-  ``metric``, 默认值={``l2`` for regression}, {``binary_logloss`` for binary classification}, {``ndcg`` for lambdarank}, 类型=multi-enum,
   options=\ ``l1``, ``l2``, ``ndcg``, ``auc``, ``binary_logloss``, ``binary_error`` ...

   -  ``l1``, absolute loss, 别名=\ ``mean_absolute_error``, ``mae``

   -  ``l2``, square loss, 别名=\ ``mean_squared_error``, ``mse``

   -  ``l2_root``, root square loss, 别名=\ ``root_mean_squared_error``, ``rmse``

   -  ``quantile``, `Quantile regression`_

   -  ``huber``, `Huber loss`_

   -  ``fair``, `Fair loss`_

   -  ``poisson``, `Poisson regression`_

   -  ``ndcg``, `NDCG`_

   -  ``map``, `MAP`_

   -  ``auc``, `AUC`_

   -  ``binary_logloss``, `log loss`_

   -  ``binary_error``, 样本: ``0`` 的正确分类, ``1`` 错误分类

   -  ``multi_logloss``, mulit-class 损失日志分类

   -  ``multi_error``, error rate for mulit-class 出错率分类

   -  ``xentropy``, cross-entropy (与可选的线性权重), 别名=\ ``cross_entropy``

   -  ``xentlambda``, "intensity-weighted" 交叉熵, 别名=\ ``cross_entropy_lambda``

   -  ``kldiv``, `Kullback-Leibler divergence`_, 别名=\ ``kullback_leibler``

   -  支持多指标, 使用 ``,``分隔

-  ``metric_freq``, 默认值=\ ``1``, 类型=int

   -  频率指标输出

-  ``train_metric``, 默认值=\ ``false``, 类型=bool, 别名=\ ``training_metric``, ``is_training_metric``

   - 如果你需要输出训练的度量结果则设置 ``true``

-  ``ndcg_at``, 默认值=\ ``1,2,3,4,5``, 类型=multi-int, 别名=\ ``ndcg_eval_at``, ``eval_at``

   -  `NDCG`_ 职位评估, 使用 ``,``分隔

网络参数
------------------

以下参数用于并行学习，只用于基本(socket)版本。

-  ``num_machines``, 默认值=\ ``1``, 类型=int, 别名=\ ``num_machine``

   -  用于并行学习的并行学习应用程序的数量

   -  需要在socket和mpi版本中设置这个

-  ``local_listen_port``, 默认值=\ ``12400``, 类型=int, 别名=\ ``local_port``

   -  监听本地机器的TCP端口

   -  在培训之前，您应该再防火墙设置中放开该端口

-  ``time_out``, 默认值=\ ``120``, 类型=int

   -    允许socket几分钟内超时

-  ``machine_list_file``, 默认值=\ ``""``, 类型=string, 别名=\ ``mlist``

   -  为这个并行学习应用程序列出机器的文件

   -  每一行包含一个IP和一个端口为一台机器。格式是ip port，由空格分隔

GPU 参数
--------------

-  ``gpu_platform_id``, 默认值=\ ``-1``, 类型=int

   -  OpenCL 平台 ID. 通常每个GPU供应商都会公开一个OpenCL平台。

   -  默认值为 ``-1``, 意味着整个系统平台

-  ``gpu_device_id``, 默认值=\ ``-1``, 类型=int

   -  OpenCL设备ID在指定的平台上。 在选定的平台上的每一个GPU都有一个唯一的设备ID

   -  默认值为``-1``, 这个默认值意味着选定平台上的设备

-  ``gpu_use_dp``, 默认值=\ ``false``, 类型=bool

   -  设置为 ``true`` 在GPU上使用双精度GPU (默认使用单精度)
  
模型参数
------------------------

该特性仅在命令行版本中得到支持。

-  ``convert_model_language``, 默认值=\ ``""``, 类型=string

   -  只支持``cpp``

   -  如果 ``convert_model_language`` 设置为 ``task``时 该模型也将转换为 ``train``, 

-  ``convert_model``, 默认值=\ ``"gbdt_prediction.cpp"``, 类型=string

   -  转换模型的输出文件名

其他
------

持续训练输入分数
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LightGBM支持对初始得分进行持续的培训。它使用一个附加的文件来存储这些初始值，如下:

::

    0.5
    -0.1
    0.9
    ...

它意味着最初的得分第一个数据行是``0.5`,第二个是``-0.1``等等。
初始得分文件与数据文件逐行对应，每一行有一个分数。
如果数据文件的名称是``train.txt`，最初的分数文件应该被命名为``train.txt.init``与作为数据文件在同一文件夹。
在这种情况下，LightGBM将自动加载初始得分文件，如果它存在的话。

权重数据
~~~~~~~~~~~

LightGBM 加权训练。它使用一个附加文件来存储权重数据，如下:

::

    1.0
    0.5
    0.8
    ...

它意味的重压着第一个数据行是``1.0``,第二个是``0.5``,等等。
权重文件按行与数据文件行相对应，每行的权重为。
如果数据文件的名称是``train.txt``，应该将重量文件命名为``train.txt.weight` 与数据文件相同的文件夹。
在这种情况下，LightGBM将自动加载权重文件，如果它存在的话。

**update**:
现在可以在数据文件中指定``weight``列。请参阅以上参数的参数。

查询数据
~~~~~~~~~~

对于LambdaRank的学习，它需要查询信息来训练数据。
LightGBM使用一个附加文件来存储查询数据，如下:
::

    27
    18
    67
    ...

它意味着第一个“27”“行样本属于一个查询和下一个``18``行属于另一个,等等。
**Note**: 数据应该由查询来排序.

如果数据文件的名称是``train.txt`,这个查询文件应该被命名为``train.txt.query``查询在相同的培训数据文件夹中。
在这种情况下，LightGBM将自动加载查询文件，如果它存在的话。

**update**:
现在可以在数据文件中指定特定的 query/group id。请参阅上面的参数组。

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
