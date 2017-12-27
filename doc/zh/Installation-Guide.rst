安装指南
==================

该页面是 LightGBM CLI 版本的构建指南.

要构建 Python 和 R 的软件包, 请分别参阅 `Python-package`_ 和 `R-package`_ 文件夹.

Windows
~~~~~~~

LightGBM 可以使用 Visual Studio, MSBuild 与 CMake 或 MinGW 来在  Windows 上构建.

Visual Studio (or MSBuild)
^^^^^^^^^^^^^^^^^^^^^^^^^^

使用 GUI
********

1. 安装 `Visual Studio`_ (2015 或更新版本).

2. 下载 `zip archive`_ 并且 unzip（解压）它.

3. 定位到 ``LightGBM-master/windows`` 文件夹.

4. 使用 Visual Studio 打开 ``LightGBM.sln`` 文件, 选择 ``Release`` 配置并且点击 ``BUILD``->\ ``Build Solution (Ctrl+Shift+B)``.

   如果出现有关 **Platform Toolset** 的错误, 定位到 ``PROJECT``->\ ``Properties``->\ ``Configuration Properties``->\ ``General`` 然后选择 toolset 安装到你的机器.

该 exe 文件可以在 ``LightGBM-master/windows/x64/Release`` 文件夹中找到.

使用命令行
*****************

1. 安装 `Git for Windows`_, `CMake`_ (3.8 或更新版本) 以及 `MSBuild`_ (**MSBuild** 是非必要的, 如果已安装 **Visual Studio** (2015 或更新版本) 的话).

2. 运行以下命令:

   .. code::

     git clone --recursive https://github.com/Microsoft/LightGBM
     cd LightGBM
     mkdir build
     cd build
     cmake -DCMAKE_GENERATOR_PLATFORM=x64 ..
     cmake --build . --target ALL_BUILD --config Release

这些 exe 和 dll 文件可以在 ``LightGBM/Release`` 文件夹中找到.

MinGW64
^^^^^^^

1. 安装 `Git for Windows`_, `CMake`_ 和 `MinGW-w64`_.

2. 运行以下命令:

   .. code::

     git clone --recursive https://github.com/Microsoft/LightGBM
     cd LightGBM
     mkdir build
     cd build
     cmake -G "MinGW Makefiles" ..
     mingw32-make.exe -j4

这些 exe 和 dll 文件可以在 ``LightGBM/`` 文件夹中找到.

**注意**: 也许你需要再一次运行 ``cmake -G "MinGW Makefiles" ..`` 命令, 如果遇到 ``sh.exe was found in your PATH`` 错误的话.

也许你还想要参阅 `gcc 建议 <./gcc-Tips.rst>`__.

Linux
~~~~~

LightGBM 使用 **CMake** 来构建. 运行以下命令:

.. code::

  git clone --recursive https://github.com/Microsoft/LightGBM ; cd LightGBM
  mkdir build ; cd build
  cmake ..
  make -j4

**注意**: glibc >= 2.14 是必须的.

也许你还想要参阅 `gcc 建议 <./gcc-Tips.rst>`__.

OSX
~~~

LightGBM 依赖于 **OpenMP** 进行编译, 然而 Apple Clang 不支持它.

请使用以下命令来安装 **gcc/g++** :

.. code::

  brew install cmake
  brew install gcc --without-multilib

然后安装 LightGBM:

.. code::

  git clone --recursive https://github.com/Microsoft/LightGBM ; cd LightGBM
  export CXX=g++-7 CC=gcc-7
  mkdir build ; cd build
  cmake ..
  make -j4

也许你还想要参阅 `gcc 建议 <./gcc-Tips.rst>`__.

Docker
~~~~~~

请参阅 `Docker 文件夹 <https://github.com/Microsoft/LightGBM/tree/master/docker>`__.

Build MPI 版本
~~~~~~~~~~~~~~~~~

LightGBM 默认的构建版本是基于 socket 的的.
LightGBM 也支持 `MPI`_.
MPI 是一种与 `RDMA`_ 支持的高性能通信方法.

如果您需要运行具有高性能通信的并行学习应用程序, 则可以构建带有 MPI 支持的 LightGBM.

Windows
^^^^^^^

使用 GUI
********

1. 需要先安装 `MS MPI`_ . 需要 ``msmpisdk.msi`` 和 ``MSMpiSetup.exe``.

2. 安装 `Visual Studio`_ (2015 或更新版本).

3. 下载 `zip archive`_ 并且 unzip（解压）它.

4. 定位到 ``LightGBM-master/windows`` 文件夹.

5. 使用 Visual Studio 打开 ``LightGBM.sln`` 文件, 选择 ``Release_mpi`` 配置并且点击 ``BUILD``->\ ``Build Solution (Ctrl+Shift+B)``.

   如果遇到有关 **Platform Toolset** 的错误, 定位到 ``PROJECT``->\ ``Properties``->\ ``Configuration Properties``->\ ``General`` 并且选择安装 toolset 到你的机器上.

该 exe 文件可以在 ``LightGBM-master/windows/x64/Release_mpi`` 文件夹中找到.

使用命令行
*****************

1. 需要先安装 `MS MPI`_ . 需要 ``msmpisdk.msi`` 和 ``MSMpiSetup.exe``.

2. 安装 `Git for Windows`_, `CMake`_ (3.8 或更新版本) 和 `MSBuild`_ (MSBuild 是非必要的, 如果已安装 **Visual Studio** (2015 或更新版本)).

3. 运行以下命令:

   .. code::

     git clone --recursive https://github.com/Microsoft/LightGBM
     cd LightGBM
     mkdir build
     cd build
     cmake -DCMAKE_GENERATOR_PLATFORM=x64 -DUSE_MPI=ON ..
     cmake --build . --target ALL_BUILD --config Release

这些 exe 和 dll 文件可以在 ``LightGBM/Release`` 文件夹中找到.

**注意**: Build MPI version 通过 **MinGW** 来构建 MPI 版本的不支持的, 由于它里面缺失了 MPI 库.

Linux
^^^^^

需要先安装 `Open MPI`_ .

然后运行以下命令:

.. code::

  git clone --recursive https://github.com/Microsoft/LightGBM ; cd LightGBM
  mkdir build ; cd build
  cmake -DUSE_MPI=ON ..
  make -j4

**Note**: glibc >= 2.14 是必要的.

OSX
^^^

先安装 **gcc** 和 **Open MPI** :

.. code::

  brew install openmpi
  brew install cmake
  brew install gcc --without-multilib

然后运行以下命令:

.. code::

  git clone --recursive https://github.com/Microsoft/LightGBM ; cd LightGBM
  export CXX=g++-7 CC=gcc-7
  mkdir build ; cd build
  cmake -DUSE_MPI=ON ..
  make -j4

Build GPU 版本
~~~~~~~~~~~~~~~~~

Linux
^^^^^

在编译前应该先安装以下依赖:

-  OpenCL 1.2 headers and libraries, 它们通常由 GPU 制造商提供.

   The generic OpenCL ICD packages (for example, Debian package ``cl-icd-libopencl1`` and ``cl-icd-opencl-dev``) can also be used.

-  libboost 1.56 或更新版本 (1.61 或最新推荐的版本).

   We use Boost.Compute as the interface to GPU, which is part of the Boost library since version 1.61. However, since we include the source code of Boost.Compute as a submodule, we only require the host has Boost 1.56 or later installed. We also use Boost.Align for memory allocation. Boost.Compute requires Boost.System and Boost.Filesystem to store offline kernel cache.

   The following Debian packages should provide necessary Boost libraries: ``libboost-dev``, ``libboost-system-dev``, ``libboost-filesystem-dev``.

-  CMake 3.2 或更新版本.

要构建 LightGBM GPU 版本, 运行以下命令:

.. code::

  git clone --recursive https://github.com/Microsoft/LightGBM ; cd LightGBM
  mkdir build ; cd build
  cmake -DUSE_GPU=1 ..
  # if you have installed the NVIDIA OpenGL, please using following instead
  # sudo cmake -DUSE_GPU=1 -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so -OpenCL_INCLUDE_DIR=/usr/local/cuda/include/ ..
  make -j4

Windows
^^^^^^^

如果使用 **MinGW**, 该构建过程类似于 Linux 上的构建. 相关的更多细节请参阅 `GPU Windows 平台上的编译 <./GPU-Windows.rst>`__ .

以下构建过程适用于 MSVC (Microsoft Visual C++) 构建.

1. 安装 `Git for Windows`_, `CMake`_ (3.8 or higher) 和 `MSBuild`_ (MSBuild 是非必要的, 如果已安装 **Visual Studio** (2015 或更新版本)).

2. 针对 Windows 平台安装 **OpenCL** . 安装取决于你的 GPU 显卡品牌 (NVIDIA, AMD, Intel).

   - 要运行在 Intel 上, 获取 `Intel SDK for OpenCL`_.

   - 要运行在 AMD 上, 获取 `AMD APP SDK`_.

   - 要运行在 NVIDIA 上, 获取 `CUDA Toolkit`_.

3. 安装 `Boost Binary`_.

   **注意**: 要匹配你的 Visual C++ 版本:
   
   Visual Studio 2015 -> ``msvc-14.0-64.exe``,

   Visual Studio 2017 -> ``msvc-14.1-64.exe``.

4. 运行以下命令:

   .. code::

     Set BOOST_ROOT=C:\local\boost_1_64_0\
     Set BOOST_LIBRARYDIR=C:\local\boost_1_64_0\lib64-msvc-14.0
     git clone --recursive https://github.com/Microsoft/LightGBM
     cd LightGBM
     mkdir build
     cd build
     cmake -DCMAKE_GENERATOR_PLATFORM=x64 -DUSE_GPU=1 ..
     cmake --build . --target ALL_BUILD --config Release

   **注意**: ``C:\local\boost_1_64_0\`` 和 ``C:\local\boost_1_64_0\lib64-msvc-14.0`` 是你 Boost 二进制文件的位置. 你还可以将它们设置为环境变量, 以在构建时避免 ``Set ...`` 命令.

Protobuf 支持
^^^^^^^^^^^^^^^^

如果想要使用 protobuf 来保存和加载模型, 请先安装 `protobuf c++ version <https://github.com/google/protobuf/blob/master/src/README.md>`__ .
然后使用 USE_PROTO=ON 配置来运行 cmake 命令, 例如:

.. code::

  cmake -DUSE_PROTO=ON ..

然后在保存或加载模型时, 可以在参数中使用 ``model_format=proto``.

**注意**: 针对 windows 用户, 它只对 mingw 进行了测试. 

Docker
^^^^^^

请参阅 `GPU Docker 文件夹 <https://github.com/Microsoft/LightGBM/tree/master/docker/gpu>`__.

.. _Python-package: https://github.com/Microsoft/LightGBM/tree/master/python-package

.. _R-package: https://github.com/Microsoft/LightGBM/tree/master/R-package

.. _zip archive: https://github.com/Microsoft/LightGBM/archive/master.zip

.. _Visual Studio: https://www.visualstudio.com/downloads/

.. _Git for Windows: https://git-scm.com/download/win

.. _CMake: https://cmake.org/

.. _MSBuild: https://www.visualstudio.com/downloads/#build-tools-for-visual-studio-2017

.. _MinGW-w64: https://mingw-w64.org/doku.php/download

.. _MPI: https://en.wikipedia.org/wiki/Message_Passing_Interface

.. _RDMA: https://en.wikipedia.org/wiki/Remote_direct_memory_access

.. _MS MPI: https://www.microsoft.com/en-us/download/details.aspx?id=49926

.. _Open MPI: https://www.open-mpi.org/

.. _Intel SDK for OpenCL: https://software.intel.com/en-us/articles/opencl-drivers

.. _AMD APP SDK: http://developer.amd.com/amd-accelerated-parallel-processing-app-sdk/

.. _CUDA Toolkit: https://developer.nvidia.com/cuda-downloads

.. _Boost Binary: https://sourceforge.net/projects/boost/files/boost-binaries/1.64.0/
