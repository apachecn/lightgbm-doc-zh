LightGBM 常见问题解答
============

内容
~~~~~~~~

-  `关键问题  <#关键问题>`__

-  `LightGBM <#lightgbm>`__

-  `R包 <#R包>`__

-  `Python包 <#Python包>`__

--------------

关键问题
~~~~~~~~

在使用LightGBM遇到关键问题时（程序奔溃，预测结果错误，无意义输出...）,你应该联系谁？

如果你的问题不是那么紧急，可以把问题放到 `Microsoft/LightGBM repository <https://github.com/Microsoft/LightGBM/issues>`__。

如果你的问题急需要解决，首先要明确你有哪些错误：

-  你认为问题会不会复现在CLI（命令行接口），R或者Python上？

-  还是只会在某个特定的包（R或者Python）上出现?

-  还是会在某个特定的编译器（gcc或者MinGW）上出现？

-  还是会在某个特定的操作系统（Windows或者Linux）上出现？

-  你能用一个简单的例子复现这些问题吗？

-  你能（或者不能）在去掉所有的优化信息和在debug模式下编译LightGBM时复现这些问题吗？

当出现问题的时候，根据上述答案，随时可以@我们（不同的问题可以@不同的人，下面是各种不同类型问题的负责人），这样我们就能更快地帮助你解决问题。

-  `@guolinke <https://github.com/guolinke>`__ (C++ code / R-package / Python-package)
-  `@chivee <https://github.com/chivee>`__ (C++ code / Python-package)
-  `@Laurae2 <https://github.com/Laurae2>`__ (R-package)
-  `@wxchan <https://github.com/wxchan>`__ (Python-package)
-  `@henry0312 <https://github.com/henry0312>`__ (Python-package)
-  `@StrikerRUS <https://github.com/StrikerRUS>`__ (Python-package)
-  `@huanzhang12 <https://github.com/huanzhang12>`__ (GPU support)

记住这是一个免费的/开放的社区支持，我们可能不能做到全天候的提供帮助。
--------------

LightGBM
~~~~~~~~

-  **问题 1**：我可以去哪里找到关于LightBGM参数的更多详细内容？

-  **方法 1**：可以看一下这个 `Parameters <./Parameters.rst>`__ and `Laurae++/Parameters <https://sites.google.com/view/lauraepp/parameters>`__网站。
--------------

-  **问题 2**：在一个有百万个特征的数据集中，（要在很长一段时间后才开始训练或者）训练根本没有开始。

-  **方法 2**：对 ``bin_construct_sample_cnt`` 用一个较小的值和对 ``min_data`` 用一个较大的值。
--------------

-  **问题 3**：当在一个很大的数据集上使用LightBGM，我的电脑会耗尽内存。

-  **方法 3**：很多方法啊：将 ``histogram_pool_size`` 参数设置成你想为LightGBM分配的MB(histogram\_pool\_size + dataset size = approximately RAM used),
   减少 ``num_leaves`` 或减少 ``max_bin``（点这里 `Microsoft/LightGBM#562 <https://github.com/Microsoft/LightGBM/issues/562>`__）。
--------------

-  **问题 4**：我使用Windows系统。我应该使用Visual Studio或者MinGW编译LightBGM吗？

-  **方法 4**：推荐使用 `Visual Studio <https://github.com/Microsoft/LightGBM/issues/542>`__，因为它的性能更好。
--------------

-  **问题 5**：当使用LightBGM，我每次运行得到的结果都不同（结果不能复现）。

-  **方法 5**：这是一个很正常的问题，我们/你也无能为力。
   你可以试试使用 ``gpu_use_dp = true`` 来复现结果（点这里 `Microsoft/LightGBM#560 <https://github.com/Microsoft/LightGBM/pull/560#issuecomment-304561654>`__）。
   你也可以使用CPU的版本试试。
--------------

-  **问题 6**：Bagging在改变线程的数量时，是不能复现的。

-  **方法 6**：由于LightBGM Bagging是多线程运行的，它的输出依赖于使用线程的数量。
   There is `no workaround currently <https://github.com/Microsoft/LightGBM/issues/632>`__。
--------------

-  **问题 7**：我试过使用随机森林模式，LightBGM奔溃啦！

-  **方法 7**：这是设计的问题。
   你必须使用 ``bagging_fraction`` 和 ``feature_fraction`` 与1不同，要和 ``bagging_freq`` 结合使用。
   看这个例子 `this thread <https://github.com/Microsoft/LightGBM/issues/691>`__。
--------------

-  **问题 8**：当在一个很大的数据集上和很多核心系统使用LightBGMWindows系统时，CPU不是满负荷运行（例如只使用了10%的CPU）。

-  **方法 8**：请使用 `Visual Studio <https://www.visualstudio.com/downloads/>`__，
   因为Visual Studio可能 `10x faster than MinGW <https://github.com/Microsoft/LightGBM/issues/749>`__，尤其是在很大的树上。
--------------

R包
~~~~~~~~~

-  **问题 1**：在训练先前的LightBGM模型时一个错误出现后，任何使用LightBGM的训练命令都不会起作用。

-  **方法 1**：在R控制台中运行 ``lgb.unloader(wipe = TRUE)``，再重新创建LightBGM数据集（这会消除所有与LightBGM相关的变量）。
   由于这些指针，选择不去消除这些变量不会修复这些错误。
   这是一个已知的问题: `Microsoft/LightGBM#698 <https://github.com/Microsoft/LightGBM/issues/698>`__。
--------------

-  **问题 2**：我使用过``setinfo``,试过打印我的``lgb.Dataset``,结果R控制台无响应。

-  **方法 2**：在使用 ``setinfo`` 后避免打印 ``lgb.Dataset``.
   这是一个已知的bug：`Microsoft/LightGBM#539 <https://github.com/Microsoft/LightGBM/issues/539>`__。
--------------

Python包
~~~~~~~~~~~~~~

-  **问题 1**：当从GitHub使用 ``python setup.py install`` 安装，我看到如下错误信息。

   ::

       error：错误：安装脚本指定绝对路径：
       /Users/Microsoft/LightGBM/python-package/lightgbm/../../lib_lightgbm.so
       setup()参数必须 *一直* 是/-分离路径相对于setup.py目录， *从不* 是绝对路径。

-  **方法 1**：这个错误在新版本中应该会被解决。
   如果你还会遇到这个问题，试着在你的Python包中去掉 ``lightgbm.egg-info`` 文件夹，再重装一下，
   或者对照一下这个 `this thread on stackoverflow <http://stackoverflow.com/questions/18085571/pip-install-error-setup-script-specifies-an-absolute-path>`__。
--------------

-  **问题 2**：我看到错误信息如下

   ::
       在构建数据集前不能 get/set label/weight/init_score/group/num_data/num_feature。

   但是我已经使用下面的代码构建数据集

   ::
       train = lightgbm.Dataset(X_train, y_train)
       
   或如下错误信息

   ::

       在释放原始数据后，不能设置predictor/reference/categorical特征。可以在创建数据集时设置free_raw_data=False避免上面的问题。

-  **方法2**: 因为LightBGM创建bin mappers来构建树，在一个Booster内的train和valid数据集共享同一个bin mappers，类别特征和特征名等信息，数据集对象在创建Booster时候被创建。
   如果你设置 ``free_raw_data=True`` (默认)，原始数据（在Python数据结构中的）将会被释放。
   所以，如果你想要：

   -  在创建数据集前get label(or weight/init\_score/group)，这和get  ``self.label`` 操作相同。
   
   -  在创建数据集前set label(or weight/init\_score/group)，这和 ``self.label=some_label_array`` 操作相同。
   
   -  在创建数据集前get num\_data(or num\_feature)，你可以使用 ``self.data`` 得到数据，然后如果你的数据是 ``numpy.ndarray``，使用一些类似  ``self.data.shape`` 的代码。
   
   -  在构建数据集之后set predictor(or reference/categorical feature)，你应该设置 ``free_raw_data=False`` 或使用同样的原始数据初始化数据集对象。