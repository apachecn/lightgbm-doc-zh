GPU SDK 相关以及设备对应表
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GPU 对应表
=================

当使用 OpenCL SDKs 时, 同时面向 CPU 和 GPU 有时是可能的。
尤其是对于 Intel OpenCL SDK 和 AMD APP SDK.

你可以在下表中找到相关信息：

+---------------------------+-----------------+-----------------+-----------------+--------------+
| SDK                       | CPU Intel/AMD   | GPU Intel       | GPU AMD         | GPU NVIDIA   |
+===========================+=================+=================+=================+==============+
| `Intel SDK for OpenCL`_   | Supported       | Supported \*    | Supported       | Untested     |
+---------------------------+-----------------+-----------------+-----------------+--------------+
| `AMD APP SDK`_            | Supported       | Untested \*     | Supported       | Untested     |
+---------------------------+-----------------+-----------------+-----------------+--------------+
| `NVIDIA CUDA Toolkit`_    | Untested \*\*   | Untested \*\*   | Untested \*\*   | Supported    |
+---------------------------+-----------------+-----------------+-----------------+--------------+

说明:

-  \* 不能直接使用.
-  \*\* Reported as unsupported in public forums（表示在公共论坛上不支持）.

AMD GPUs 使用 Intel SDK for OpenCL 并非有误, 是因为 AMD APP SDK 不兼容 CPUs.

--------------

对应表
===============

我们展示了如下情境：

-  CPU, no GPU
-  单 CPU 以及 GPU (甚至包括集成显卡)
-  多 CPU/GPU

我们提供了测试用的 R 语言代码如下，但是你可以使用任意一种语言作为测试用例：

.. code:: r

    library(lightgbm)
    data(agaricus.train, package = "lightgbm")
    train <- agaricus.train
    train$data[, 1] <- 1:6513
    dtrain <- lgb.Dataset(train$data, label = train$label)
    data(agaricus.test, package = "lightgbm")
    test <- agaricus.test
    dtest <- lgb.Dataset.create.valid(dtrain, test$data, label = test$label)
    valids <- list(test = dtest)

    params <- list(objective = "regression",
                   metric = "rmse",
                   device = "gpu",
                   gpu_platform_id = 0,
                   gpu_device_id = 0,
                   nthread = 1,
                   boost_from_average = FALSE,
                   num_tree_per_iteration = 10,
                   max_bin = 32)
    model <- lgb.train(params,
                       dtrain,
                       2,
                       valids,
                       min_data = 1,
                       learning_rate = 1,
                       early_stopping_rounds = 10)

使用不好的 ``gpu_device_id`` 是不谨慎的, 因为它会导致：

-  ``gpu_device_id = 0`` if using ``gpu_platform_id = 0``
-  ``gpu_device_id = 1`` if using ``gpu_platform_id = 1``

然而，使用不好的 ``gpu_platform_id`` 和 ``gpu_device_id`` 的组合会导致 **冲突** （你可能会丢失整个 session 内容）。

CPU Only 架构
----------------------

当你使用单一设备时 (one CPU), OpenCL 的使用很直接： ``gpu_platform_id = 0``, ``gpu_device_id = 0``

这将会使用 CPU with OpenCL, even though it says it says GPU（尽管它说这是 GPU）。

例如:

.. code:: r

    > params <- list(objective = "regression",
    +                metric = "rmse",
    +                device = "gpu",
    +                gpu_platform_id = 0,
    +                gpu_device_id = 0,
    +                nthread = 1,
    +                boost_from_average = FALSE,
    +                num_tree_per_iteration = 10,
    +                max_bin = 32)
    > model <- lgb.train(params,
    +                    dtrain,
    +                    2,
    +                    valids,
    +                    min_data = 1,
    +                    learning_rate = 1,
    +                    early_stopping_rounds = 10)
    [LightGBM] [Info] This is the GPU trainer!!
    [LightGBM] [Info] Total Bins 232
    [LightGBM] [Info] Number of data: 6513, number of used features: 116
    [LightGBM] [Info] Using requested OpenCL platform 0 device 1
    [LightGBM] [Info] Using GPU Device: Intel(R) Core(TM) i7-4600U CPU @ 2.10GHz, Vendor: GenuineIntel
    [LightGBM] [Info] Compiling OpenCL Kernel with 16 bins...
    [LightGBM] [Info] GPU programs have been built
    [LightGBM] [Info] Size of histogram bin entry: 12
    [LightGBM] [Info] 40 dense feature groups (0.12 MB) transfered to GPU in 0.004540 secs. 76 sparse feature groups.
    [LightGBM] [Info] No further splits with positive gain, best gain: -inf
    [LightGBM] [Info] Trained a tree with leaves=16 and max_depth=8
    [1]:    test's rmse:1.10643e-17 
    [LightGBM] [Info] No further splits with positive gain, best gain: -inf
    [LightGBM] [Info] Trained a tree with leaves=7 and max_depth=5
    [2]:    test's rmse:0

单 CPU 以及 GPU (甚至包括集成显卡)
--------------------------------------------------

如果你有一块集成显卡（Intel HD Graphics）和一块独立显卡（AMD, NVIDIA），独立显卡可能会自动覆盖集成显卡。
解决办法是中断独显从而使用你的集显。

当你拥有多个设备时（一个 CPU，一个 GPU），通常按如下顺序：

-  GPU: ``gpu_platform_id = 0``, ``gpu_device_id = 0``,
   sometimes it is usable using ``gpu_platform_id = 1``, ``gpu_device_id = 1`` but at your own risk!

-  CPU: ``gpu_platform_id = 0``, ``gpu_device_id = 1``

GPU 的例子(``gpu_platform_id = 0``, ``gpu_device_id = 0``):

.. code:: r

    > params <- list(objective = "regression",
    +                metric = "rmse",
    +                device = "gpu",
    +                gpu_platform_id = 0,
    +                gpu_device_id = 0,
    +                nthread = 1,
    +                boost_from_average = FALSE,
    +                num_tree_per_iteration = 10,
    +                max_bin = 32)
    > model <- lgb.train(params,
    +                    dtrain,
    +                    2,
    +                    valids,
    +                    min_data = 1,
    +                    learning_rate = 1,
    +                    early_stopping_rounds = 10)
    [LightGBM] [Info] This is the GPU trainer!!
    [LightGBM] [Info] Total Bins 232
    [LightGBM] [Info] Number of data: 6513, number of used features: 116
    [LightGBM] [Info] Using GPU Device: Oland, Vendor: Advanced Micro Devices, Inc.
    [LightGBM] [Info] Compiling OpenCL Kernel with 16 bins...
    [LightGBM] [Info] GPU programs have been built
    [LightGBM] [Info] Size of histogram bin entry: 12
    [LightGBM] [Info] 40 dense feature groups (0.12 MB) transfered to GPU in 0.004211 secs. 76 sparse feature groups.
    [LightGBM] [Info] No further splits with positive gain, best gain: -inf
    [LightGBM] [Info] Trained a tree with leaves=16 and max_depth=8
    [1]:    test's rmse:1.10643e-17 
    [LightGBM] [Info] No further splits with positive gain, best gain: -inf
    [LightGBM] [Info] Trained a tree with leaves=7 and max_depth=5
    [2]:    test's rmse:0

CPU 的例子 (``gpu_platform_id = 0``, ``gpu_device_id = 1``):

.. code:: r

    > params <- list(objective = "regression",
    +                metric = "rmse",
    +                device = "gpu",
    +                gpu_platform_id = 0,
    +                gpu_device_id = 1,
    +                nthread = 1,
    +                boost_from_average = FALSE,
    +                num_tree_per_iteration = 10,
    +                max_bin = 32)
    > model <- lgb.train(params,
    +                    dtrain,
    +                    2,
    +                    valids,
    +                    min_data = 1,
    +                    learning_rate = 1,
    +                    early_stopping_rounds = 10)
    [LightGBM] [Info] This is the GPU trainer!!
    [LightGBM] [Info] Total Bins 232
    [LightGBM] [Info] Number of data: 6513, number of used features: 116
    [LightGBM] [Info] Using requested OpenCL platform 0 device 1
    [LightGBM] [Info] Using GPU Device: Intel(R) Core(TM) i7-4600U CPU @ 2.10GHz, Vendor: GenuineIntel
    [LightGBM] [Info] Compiling OpenCL Kernel with 16 bins...
    [LightGBM] [Info] GPU programs have been built
    [LightGBM] [Info] Size of histogram bin entry: 12
    [LightGBM] [Info] 40 dense feature groups (0.12 MB) transfered to GPU in 0.004540 secs. 76 sparse feature groups.
    [LightGBM] [Info] No further splits with positive gain, best gain: -inf
    [LightGBM] [Info] Trained a tree with leaves=16 and max_depth=8
    [1]:    test's rmse:1.10643e-17 
    [LightGBM] [Info] No further splits with positive gain, best gain: -inf
    [LightGBM] [Info] Trained a tree with leaves=7 and max_depth=5
    [2]:    test's rmse:0

当错误使用 ``gpu_device_id``, 它会自动设置为 ``gpu_device_id = 0``:

.. code:: r

    > params <- list(objective = "regression",
    +                metric = "rmse",
    +                device = "gpu",
    +                gpu_platform_id = 0,
    +                gpu_device_id = 9999,
    +                nthread = 1,
    +                boost_from_average = FALSE,
    +                num_tree_per_iteration = 10,
    +                max_bin = 32)
    > model <- lgb.train(params,
    +                    dtrain,
    +                    2,
    +                    valids,
    +                    min_data = 1,
    +                    learning_rate = 1,
    +                    early_stopping_rounds = 10)
    [LightGBM] [Info] This is the GPU trainer!!
    [LightGBM] [Info] Total Bins 232
    [LightGBM] [Info] Number of data: 6513, number of used features: 116
    [LightGBM] [Info] Using GPU Device: Oland, Vendor: Advanced Micro Devices, Inc.
    [LightGBM] [Info] Compiling OpenCL Kernel with 16 bins...
    [LightGBM] [Info] GPU programs have been built
    [LightGBM] [Info] Size of histogram bin entry: 12
    [LightGBM] [Info] 40 dense feature groups (0.12 MB) transfered to GPU in 0.004211 secs. 76 sparse feature groups.
    [LightGBM] [Info] No further splits with positive gain, best gain: -inf
    [LightGBM] [Info] Trained a tree with leaves=16 and max_depth=8
    [1]:    test's rmse:1.10643e-17 
    [LightGBM] [Info] No further splits with positive gain, best gain: -inf
    [LightGBM] [Info] Trained a tree with leaves=7 and max_depth=5
    [2]:    test's rmse:0

Do not ever run under the following scenario as it is known to crash even if it says it is using the CPU because it is NOT the case:

-  One CPU and one GPU
-  ``gpu_platform_id = 1``, ``gpu_device_id = 0``

.. code:: r

    > params <- list(objective = "regression",
    +                metric = "rmse",
    +                device = "gpu",
    +                gpu_platform_id = 1,
    +                gpu_device_id = 0,
    +                nthread = 1,
    +                boost_from_average = FALSE,
    +                num_tree_per_iteration = 10,
    +                max_bin = 32)
    > model <- lgb.train(params,
    +                    dtrain,
    +                    2,
    +                    valids,
    +                    min_data = 1,
    +                    learning_rate = 1,
    +                    early_stopping_rounds = 10)
    [LightGBM] [Info] This is the GPU trainer!!
    [LightGBM] [Info] Total Bins 232
    [LightGBM] [Info] Number of data: 6513, number of used features: 116
    [LightGBM] [Info] Using requested OpenCL platform 1 device 0
    [LightGBM] [Info] Using GPU Device: Intel(R) Core(TM) i7-4600U CPU @ 2.10GHz, Vendor: Intel(R) Corporation
    [LightGBM] [Info] Compiling OpenCL Kernel with 16 bins...
    terminate called after throwing an instance of 'boost::exception_detail::clone_impl<boost::exception_detail::error_info_injector<boost::compute::opencl_error> >'
      what():  Invalid Program

    This application has requested the Runtime to terminate it in an unusual way.
    Please contact the application's support team for more information.

多 CPU 和 GPU
--------------------

如果你有多个设备（多 CPU 和多 GPU），
你需要测试不同的 ``gpu_device_id`` 和不同的 ``gpu_platform_id`` 值以找出适用于你想使用的 CPU/GPU 的值。
记住不中断其他独立显卡的话，可能不能直接使用集成显卡。

.. _Intel SDK for OpenCL: https://software.intel.com/en-us/articles/opencl-drivers

.. _AMD APP SDK: http://developer.amd.com/amd-accelerated-parallel-processing-app-sdk/

.. _NVIDIA CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
