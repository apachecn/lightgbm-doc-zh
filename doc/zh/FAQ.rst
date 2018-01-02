LightGBM FAQ
============
LightGBM 常见问题解答
Contents
~~~~~~~~
内容
-  `Critical <#critical>`__
-  `关键问题  <#关键问题》`__
-  `LightGBM <#lightgbm>`__
-  `LightBGM <#lightgbm>
-  `R-package <#r-package>`__
-  `R包 <#R包>`__
-  `Python-package <#python-package>`__
-  `Python包 <#Python包>`__
--------------

Critical
~~~~~~~~
关键问题
You encountered a critical issue when using LightGBM (crash, prediction error, non sense outputs...). Who should you contact?
在使用LightGBM遇到关键问题时（程序奔溃，预测结果错误，无意义输出...）,你应该联系谁？
If your issue is not critical, just post an issue in `Microsoft/LightGBM repository <https://github.com/Microsoft/LightGBM/issues>`__.
如果你的问题不是那么紧急，可以把问题放到`Microsoft/LightGBM repository <https://github.com/Microsoft/LightGBM/issues>`__。
If it is a critical issue, identify first what error you have:
如果你的问题急需要解决，首先要明确你有哪些错误：
-  Do you think it is reproducible on CLI (command line interface), R, and/or Python?
-  你认为问题会不会复现在CLI（命令行接口），R或者Python上？
-  Is it specific to a wrapper? (R or Python?)
-  还是只会在某个特定的包（R或者Python）上出现?
-  Is it specific to the compiler? (gcc versions? MinGW versions?)
-  还是会在某个特定的编译器（gcc或者MinGW）上出现？
-  Is it specific to your Operating System? (Windows? Linux?)
-  还是会在某个特定的操作系统（Windows或者Linux）上出现？
-  Are you able to reproduce this issue with a simple case?
-  你能用一个简单的例子复现这些问题吗？
-  Are you able to (not) reproduce this issue after removing all optimization flags and compiling LightGBM in debug mode?
-  你能（或者不能）在去掉所有的优化信息和在debug模式下编译LightGBM时复现这些问题吗？
Depending on the answers, while opening your issue, feel free to ping (just mention them with the arobase (@) symbol) appropriately so we can attempt to solve your problem faster:
当出现问题的时候，根据上述答案，随时可以@我们（不同的问题可以@不同的人，下面是各种不同类型问题的负责人），这样我们就能更快地帮助你解决问题。
-  `@guolinke <https://github.com/guolinke>`__ (C++ code / R-package / Python-package)
-  `@chivee <https://github.com/chivee>`__ (C++ code / Python-package)
-  `@Laurae2 <https://github.com/Laurae2>`__ (R-package)
-  `@wxchan <https://github.com/wxchan>`__ (Python-package)
-  `@henry0312 <https://github.com/henry0312>`__ (Python-package)
-  `@StrikerRUS <https://github.com/StrikerRUS>`__ (Python-package)
-  `@huanzhang12 <https://github.com/huanzhang12>`__ (GPU support)

Remember this is a free/open community support. We may not be available 24/7 to provide support.
记住这是一个免费的/开放的社区支持，我们可能不能做到全天候的提供帮助。
--------------

LightGBM
~~~~~~~~
LightBGM
-  **Question 1**: Where do I find more details about LightGBM parameters?
-  **问题 1**：我可以去哪里找到关于LightBGM参数的更多详细内容？
-  **Solution 1**: Take a look at `Parameters <./Parameters.rst>`__ and `Laurae++/Parameters <https://sites.google.com/view/lauraepp/parameters>`__ website.
-  **方法 1**：可以看一下这个`Parameters <./Parameters.rst>`__ and `Laurae++/Parameters <https://sites.google.com/view/lauraepp/parameters>`__网站。
--------------

-  **Question 2**: On datasets with million of features, training do not start (or starts after a very long time).
-  **问题 2**：在一个有百万个特征的数据集中，（要在很长一段时间后才开始训练或者）训练根本没有开始。
-  **Solution 2**: Use a smaller value for ``bin_construct_sample_cnt`` and a larger value for ``min_data``.
-  **方法 2**：对``bin_construct_sample_cnt``用一个较小的值和对``min_data``用一个较大的值。
--------------

-  **Question 3**: When running LightGBM on a large dataset, my computer runs out of RAM.
-  **问题 3**：当在一个很大的数据集上使用LightBGM，我的电脑会耗尽内存。
-  **Solution 3**: Multiple solutions: set ``histogram_pool_size`` parameter to the MB you want to use for LightGBM (histogram\_pool\_size + dataset size = approximately RAM used),
   lower ``num_leaves`` or lower ``max_bin`` (see `Microsoft/LightGBM#562 <https://github.com/Microsoft/LightGBM/issues/562>`__).
-  **方法 3**：很多方法啊：将``histogram_pool_size``参数设置成你想为LightGBM分配的MB(histogram\_pool\_size + dataset size = approximately RAM used),
   减少``num_leaves``或减少 ``max_bin``（点这里`Microsoft/LightGBM#562 <https://github.com/Microsoft/LightGBM/issues/562>`__）。
--------------

-  **Question 4**: I am using Windows. Should I use Visual Studio or MinGW for compiling LightGBM?
-  **问题 4**：我使用Windows系统。我应该使用Visual Studio或者MinGW编译LightBGM吗？
-  **Solution 4**: It is recommended to `use Visual Studio <https://github.com/Microsoft/LightGBM/issues/542>`__ as its performance is higher for LightGBM.
-  **方法 4**：推荐使用Visual Studio <https://github.com/Microsoft/LightGBM/issues/542>`__，因为它的性能更好。
--------------

-  **Question 5**: When using LightGBM GPU, I cannot reproduce results over several runs.
-  **问题 5**：当使用LightBGM，我每次运行得到的结果都不同（结果不能复现）。
-  **Solution 5**: It is a normal issue, there is nothing we/you can do about,
   you may try to use ``gpu_use_dp = true`` for reproducibility (see `Microsoft/LightGBM#560 <https://github.com/Microsoft/LightGBM/pull/560#issuecomment-304561654>`__).
   You may also use CPU version.
-  **方法 5**：这是一个很正常的问题，我们/你也无能为力。
   你可以试试使用 ``gpu_use_dp = true`` 来复现结果（点这里`Microsoft/LightGBM#560 <https://github.com/Microsoft/LightGBM/pull/560#issuecomment-304561654>`__）。
   你也可以使用CPU的版本试试。
--------------

-  **Question 6**: Bagging is not reproducible when changing the number of threads.
-  **问题 6**：Bagging在改变线程的数量时，是不能复现的。
-  **Solution 6**: As LightGBM bagging is running multithreaded, its output is dependent on the number of threads used.
   There is `no workaround currently <https://github.com/Microsoft/LightGBM/issues/632>`__.
-  **方法 6**：由于LightBGM Bagging是多线程运行的，它的输出依赖于使用线程的数量。
   There is `no workaround currently <https://github.com/Microsoft/LightGBM/issues/632>`__。
--------------

-  **Question 7**: I tried to use Random Forest mode, and LightGBM crashes!
-  **问题 7**：我试过使用随机森林模式，LightBGM奔溃啦！
-  **Solution 7**: It is by design.
   You must use ``bagging_fraction`` and ``feature_fraction`` different from 1, along with a ``bagging_freq``.
   See `this thread <https://github.com/Microsoft/LightGBM/issues/691>`__ as an example.
-  **方法 7**：这是设计的问题。
   你必须使用 ``bagging_fraction``和``feature_fraction`` 与1不同，要和``bagging_freq``结合使用。
   看这个例子 `this thread <https://github.com/Microsoft/LightGBM/issues/691>`__ 。
--------------

-  **Question 8**: CPU are not kept busy (like 10% CPU usage only) in Windows when using LightGBM on very large datasets with many core systems.
-  **问题 8**：当在一个很大的数据集上和很多核心系统使用LightBGMWindows系统时，CPU不是满负荷运行（例如只使用了10%的CPU）。
-  **Solution 8**: Please use `Visual Studio <https://www.visualstudio.com/downloads/>`__
   as it may be `10x faster than MinGW <https://github.com/Microsoft/LightGBM/issues/749>`__ especially for very large trees.
-  **方法 8**：请使用`Visual Studio <https://www.visualstudio.com/downloads/>`__，
   因为Visual Studio可能`10x faster than MinGW <https://github.com/Microsoft/LightGBM/issues/749>`__，尤其是在很大的树上。
--------------

R-package
~~~~~~~~~
R包
-  **Question 1**: Any training command using LightGBM does not work after an error occurred during the training of a previous LightGBM model.
-  **问题 1**：在训练先前的LightBGM模型时一个错误出现后，任何使用LightBGM的训练命令都不会起作用。
-  **Solution 1**: Run ``lgb.unloader(wipe = TRUE)`` in the R console, and recreate the LightGBM datasets (this will wipe all LightGBM-related variables).
   Due to the pointers, choosing to not wipe variables will not fix the error.
   This is a known issue: `Microsoft/LightGBM#698 <https://github.com/Microsoft/LightGBM/issues/698>`__.
-  **方法 1**：在R控制台中运行 ``lgb.unloader(wipe = TRUE)``，再重新创建LightBGM数据集（这会消除所有与LightBGM相关的变量）。
   由于这些指针，选择不去消除这些变量不会修复这些错误。
   这是一个已知的问题: `Microsoft/LightGBM#698 <https://github.com/Microsoft/LightGBM/issues/698>`__。
--------------

-  **Question 2**: I used ``setinfo``, tried to print my ``lgb.Dataset``, and now the R console froze!
-  **问题 2**：我使用过``setinfo``,试过打印我的``lgb.Dataset``,结果R控制台无响应。
-  **Solution 2**: Avoid printing the ``lgb.Dataset`` after using ``setinfo``.
   This is a known bug: `Microsoft/LightGBM#539 <https://github.com/Microsoft/LightGBM/issues/539>`__.
-  **方法 2**：在使用``setinfo``后避免打印``lgb.Dataset``.
   这是一个已知的bug：`Microsoft/LightGBM#539 <https://github.com/Microsoft/LightGBM/issues/539>`__。
--------------

Python-package
~~~~~~~~~~~~~~
Python包
-  **Question 1**: I see error messages like this when install from GitHub using ``python setup.py install``.
-  **问题 1**：当从GitHub使用``python setup.py install``安装，我看到如下错误信息。
   ::
   ::
       error: Error: setup script specifies an absolute path:
       /Users/Microsoft/LightGBM/python-package/lightgbm/../../lib_lightgbm.so
       setup() arguments must *always* be /-separated paths relative to the setup.py directory, *never* absolute paths.
       error：错误：安装脚本指定绝对路径：
       /Users/Microsoft/LightGBM/python-package/lightgbm/../../lib_lightgbm.so
       setup()参数必须*一直*是/-分离路径相对于setup.py目录，*从不*是绝对路径。
-  **Solution 1**: This error should be solved in latest version.
   If you still meet this error, try to remove ``lightgbm.egg-info`` folder in your Python-package and reinstall,
   or check `this thread on stackoverflow <http://stackoverflow.com/questions/18085571/pip-install-error-setup-script-specifies-an-absolute-path>`__.
-  **方法 1**：这个错误在新版本中应该会被解决。
   如果你还会遇到这个问题，试着在你的Python包中去掉``lightgbm.egg-info``文件夹，再重装一下，
   或者对照一下这个`this thread on stackoverflow <http://stackoverflow.com/questions/18085571/pip-install-error-setup-script-specifies-an-absolute-path>`__。
--------------

-  **Question 2**: I see error messages like
-  **问题 2**：我看到错误信息如下
   ::
   ::
       Cannot get/set label/weight/init_score/group/num_data/num_feature before construct dataset
       在构建数据集前不能 get/set label/weight/init_score/group/num_data/num_feature。
   but I've already constructed dataset by some code like
   但是我已经使用下面的代码构建数据集
   ::
   ::
       train = lightgbm.Dataset(X_train, y_train)
        train = lightgbm.Dataset(X_train, y_train)
   or error messages like
   或如下错误信息
   ::
   ::
       Cannot set predictor/reference/categorical feature after freed raw data, set free_raw_data=False when construct Dataset to avoid this.
       在释放原始数据后，不能设置predictor/reference/categorical特征。可以在创建数据集时设置free_raw_data=False避免上面的问题。
-  **Solution 2**: Because LightGBM constructs bin mappers to build trees, and train and valid Datasets within one Booster share the same bin mappers,
   categorical features and feature names etc., the Dataset objects are constructed when construct a Booster.
   And if you set ``free_raw_data=True`` (default), the raw data (with Python data struct) will be freed.
   So, if you want to:
-  **方法2**: 因为LightBGM创建bin mappers来构建树，在一个Booster内的train和valid数据集共享同一个bin mappers，类别特征和特征名等信息，数据集对象在创建Booster时候被创建。
   如果你设置 ``free_raw_data=True`` (默认)，原始数据（在Python数据结构中的）将会被释放。
   所以，如果你想要：
   -  get label(or weight/init\_score/group) before construct dataset, it's same as get ``self.label``
   -  在创建数据集前get label(or weight/init\_score/group)，这和get ``self.label``操作相同。
   -  set label(or weight/init\_score/group) before construct dataset, it's same as ``self.label=some_label_array``
   -  在创建数据集前set label(or weight/init\_score/group)，这和``self.label=some_label_array``操作相同。
   -  get num\_data(or num\_feature) before construct dataset, you can get data with ``self.data``,
      then if your data is ``numpy.ndarray``, use some code like ``self.data.shape``
   -  在创建数据集前get num\_data(or num\_feature)，你可以使用``self.data``得到数据，然后如果你的数据是``numpy.ndarray``，使用一些类似 ``self.data.shape``的代码。
   -  set predictor(or reference/categorical feature) after construct dataset,
      you should set ``free_raw_data=False`` or init a Dataset object with the same raw data
   -  在构建数据集之后set predictor(or reference/categorical feature)，你应该设置``free_raw_data=False``或使用同样的原始数据初始化数据集对象。