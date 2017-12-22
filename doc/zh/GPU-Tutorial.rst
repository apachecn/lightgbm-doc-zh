LightGBM GPU 教程
=====================

本文档的目的在于一步步教你快速上手 GPU 训练。

对于 Windows, 请参阅 `GPU Windows 教程 <./GPU-Windows.rst>`__.

我们将用 `Microsoft Azure 云计算平台`_ 上的 GPU 实例做演示，
但你可以使用具有现代 AMD 或 NVIDIA GPU 的任何机器。

GPU 安装
---------

你需要在 Azure （East US, North Central US, South Central US, West Europe 以及 Southeast Asia 等区域都可用）上启动一个 ``NV`` 类型的实例
并选择 Ubuntu 16.04 LTS 作为操作系统。

经测试, ``NV6`` 类型的虚拟机是满足最小需求的, 这种虚拟机包括 1/2 M60 GPU， 8 GB 内存, 180 GB/s 的内存带宽以及 4,825 GFLOPS 的峰值计算能力。
不要使用 ``NC`` 类型的实例，因为这些 GPU (K80) 是基于较老的架构 (Kepler).

首先我们需要安装精简版的 NVIDIA 驱动和 OpenCL 开发环境：

::

    sudo apt-get update
    sudo apt-get install --no-install-recommends nvidia-375
    sudo apt-get install --no-install-recommends nvidia-opencl-icd-375 nvidia-opencl-dev opencl-headers

安装完驱动以后需要重新启动服务器。

::

    sudo init 6

大约30秒后，服务器可以重新运转。

如果你正在使用 AMD GPU, 你需要下载并安装 `AMDGPU-Pro`_ 驱动，同时安装 ``ocl-icd-libopencl1`` 和 ``ocl-icd-opencl-dev`` 两个包。

编译 LightGBM
--------------

现在安装必要的编译工具和依赖项：

::

    sudo apt-get install --no-install-recommends git cmake build-essential libboost-dev libboost-system-dev libboost-filesystem-dev

``NV6`` GPU 实例自带一个 320 GB 的极速 SSD，挂载在 ``/mnt`` 目录下。
我们把它作为我们的工作环境（如果你正在使用自己的机器，可以跳过该步）：

::

    sudo mkdir -p /mnt/workspace
    sudo chown $(whoami):$(whoami) /mnt/workspace
    cd /mnt/workspace

现在我们可以准备好校验 LightGBM 并使用 GPU 支持来编译它：

::

    git clone --recursive https://github.com/Microsoft/LightGBM
    cd LightGBM
    mkdir build ; cd build
    cmake -DUSE_GPU=1 .. 
    # if you have installed the NVIDIA OpenGL, please using following instead
    # sudo cmake -DUSE_GPU=1 -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so -OpenCL_INCLUDE_DIR=/usr/local/cuda/include/ ..
    make -j$(nproc)
    cd ..

你可以看到有两个二进制文件生成了，``lightgbm`` 和 ``lib_lightgbm.so`` 

如果你正在 OSX 系统上编译，你可能需要在 ``src/treelearner/gpu_tree_learner.h`` 中移除 ``BOOST_COMPUTE_USE_OFFLINE_CACHE`` 宏指令以避免 Boost.Compute 中的冲突错误。

安装 Python 接口 (可选)
-----------------------------------

如果你希望使用 LightGBM 的 Python 接口，你现在可以安装它（同时包括一些必要的 Python 依赖包）：

::

    sudo apt-get -y install python-pip
    sudo -H pip install setuptools numpy scipy scikit-learn -U
    cd python-package/
    sudo python setup.py install --precompile
    cd ..

你需要设置一个额外的参数 ``"device" : "gpu"`` （同时也包括其他选项如 ``learning_rate`` ，``num_leaves`` ，等等）来在 Python 中使用 GPU.

你可以阅读我们的 `Python Package Examples`_ 来获取更多关于如何使用 Python 接口的信息。

数据集准备
-------------------

使用如下命令来准备 Higgs 数据集

::

    git clone https://github.com/guolinke/boosting_tree_benchmarks.git
    cd boosting_tree_benchmarks/data
    wget "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
    gunzip HIGGS.csv.gz
    python higgs2libsvm.py
    cd ../..
    ln -s boosting_tree_benchmarks/data/higgs.train
    ln -s boosting_tree_benchmarks/data/higgs.test

现在我们可以通过运行如下命令来为 LightGBM 创建一个配置文件（请复制整段代码块并作为一个整体来运行它）：

::

    cat > lightgbm_gpu.conf <<EOF
    max_bin = 63
    num_leaves = 255
    num_iterations = 50
    learning_rate = 0.1
    tree_learner = serial
    task = train
    is_training_metric = false
    min_data_in_leaf = 1
    min_sum_hessian_in_leaf = 100
    ndcg_eval_at = 1,3,5,10
    sparse_threshold = 1.0
    device = gpu
    gpu_platform_id = 0
    gpu_device_id = 0
    EOF
    echo "num_threads=$(nproc)" >> lightgbm_gpu.conf

我们可以通过在配置文件中设置 ``device=gpu`` 来使 GPU 处于可用状态。
默认将使用系统安装的第一个 GPU（ ``gpu_platform_id=0`` 以及 ``gpu_device_id=0`` ）。

在 GPU 上运行你的第一个学习任务
-----------------------------------

现在我们可以准备开始用 GPU 做训练了！

首先我们希望确保 GPU 能够正确工作。
运行如下代码来在 GPU 上训练，并记录下50次迭代后的 AUC。

::

    ./lightgbm config=lightgbm_gpu.conf data=higgs.train valid=higgs.test objective=binary metric=auc

现在用如下代码在 CPU 上训练相同的数据集。你应该能观察到相似的 AUC：

::

    ./lightgbm config=lightgbm_gpu.conf data=higgs.train valid=higgs.test objective=binary metric=auc device=cpu

现在我们可以不计算 AUC，每次迭代后进行 GPU 上的速度测试。

::

    ./lightgbm config=lightgbm_gpu.conf data=higgs.train objective=binary metric=auc

CPU 的速度测试：

::

    ./lightgbm config=lightgbm_gpu.conf data=higgs.train objective=binary metric=auc device=cpu

你可以观察到在该 GPU 上加速了超过三倍.

GPU 加速也可以用于其他任务/指标上（回归，多类别分类器，排序，等等）。
比如，我们可以在一个回归任务下训练 Higgs 数据集：

::

    ./lightgbm config=lightgbm_gpu.conf data=higgs.train objective=regression_l2 metric=l2

同样地，你也可以比较 CPU 上的训练速度：

::

    ./lightgbm config=lightgbm_gpu.conf data=higgs.train objective=regression_l2 metric=l2 device=cpu

进一步阅读
---------------

- `GPU Tuning Guide and Performance Comparison <./GPU-Performance.rst>`__

- `GPU SDK Correspondence and Device Targeting Table <./GPU-Targets.rst>`__

- `GPU Windows Tutorial <./GPU-Windows.rst>`__

参考
---------

如果您觉得 GPU 加速很有用，希望您在著作中能够引用如下文章；

Huan Zhang, Si Si and Cho-Jui Hsieh. "`GPU Acceleration for Large-scale Tree Boosting`_." arXiv:1706.08359, 2017.

.. _Microsoft Azure cloud computing platform: https://azure.microsoft.com/

.. _AMDGPU-Pro: http://support.amd.com/en-us/download/linux

.. _Python Package Examples: https://github.com/Microsoft/LightGBM/tree/master/examples/python-guide

.. _GPU Acceleration for Large-scale Tree Boosting: https://arxiv.org/abs/1706.08359
