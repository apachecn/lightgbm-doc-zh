# LightGBM 中文文档

LightGBM 是一个梯度 boosting 框架, 使用基于学习算法的决策树.
它是分布式的, 高效的, 装逼的, 它具有以下优势:
* 速度和内存使用的优化
  * 减少分割增益的计算量
  * 通过直方图的相减来进行进一步的加速
  * 减少内存的使用
  减少并行学习的通信代价
* 稀疏优化
* 准确率的优化
  * Leaf-wise (Best-first) 的决策树生长策略
  * 类别特征值的最优分割
* 网络通信的优化
* 并行学习的优化
  * 特征并行
  * 数据并行
  * 投票并行
* GPU 支持可处理大规模数据

更多有关 LightGBM 特性的详情, 请参阅: [LightGBM 特性]().

## 文档地址

+   [在线阅读](http://lightgbm.apachecn.org)

## 项目负责人

*   [@那伊抹微笑](https://github.com/wangyangting)

## 项目贡献者

*   [@那伊抹微笑](https://github.com/apachecn/lightgbm-doc-zh)
*   [@陈瑶](https://github.com/apachecn/lightgbm-doc-zh)
*   [@胡世昌](https://github.com/apachecn/lightgbm-doc-zh)
*   [@王金树](https://github.com/apachecn/lightgbm-doc-zh)
*   [@谢家柯](https://github.com/apachecn/lightgbm-doc-zh)
*   [@方振影](https://github.com/apachecn/lightgbm-doc-zh)
*   [@臧艺](https://github.com/apachecn/lightgbm-doc-zh)
*   [@冯斐](https://github.com/apachecn/lightgbm-doc-zh)
*   [@黄志浩](https://github.com/apachecn/lightgbm-doc-zh)
*   [@刘陆琛](https://github.com/apachecn/lightgbm-doc-zh)
*   [@周立刚](https://github.com/apachecn/lightgbm-doc-zh)
*   [@陈洪](https://github.com/apachecn/lightgbm-doc-zh)
*   [@孙永杰](https://github.com/apachecn/lightgbm-doc-zh)
*   [@王贤才](https://github.com/apachecn/lightgbm-doc-zh)

## 下载

### PYPI

```
pip install lightgbm-doc-zh
lightgbm-doc-zh <port>
# 访问 http://localhost:{port} 查看文档
```

### NPM

```
npm install -g lightgbm-doc-zh
lightgbm-doc-zh <port>
# 访问 http://localhost:{port} 查看文档
```

## 贡献指南

为了使项目更加便于维护，我们将文档格式全部转换成了 Markdown，同时更换了页面生成器。后续维护工作将完全在 Markdown 上进行。

小部分格式仍然存在问题，主要是链接和表格。需要大家帮忙找到，并提 PullRequest 来修复。

## 建议反馈

*   联系项目负责人 [@那伊抹微笑](https://github.com/wangyangting).
*   在我们的 [apachecn/lightgbm-doc-zh](https://github.com/apachecn/lightgbm-doc-zh) github 上提 issue.
*   发送到 Email: lightgbm#apachecn.org（#替换成@）.
*   在我们的 [组织学习交流群](./apachecn-learning-group.rst) 中联系群主/管理员即可.

## 组织学习交流群

机器学习交流群: [629470233](http://shang.qq.com/wpa/qunwpa?idkey=bcee938030cc9e1552deb3bd9617bbbf62d3ec1647e4b60d9cd6b6e8f78ddc03) （2000人）

大数据交流群: [214293307](http://shang.qq.com/wpa/qunwpa?idkey=bcee938030cc9e1552deb3bd9617bbbf62d3ec1647e4b60d9cd6b6e8f78ddc03) （2000人）

了解我们: [http://www.apachecn.org/organization/209.html](http://www.apachecn.org/organization/209.html)

加入组织: [http://www.apachecn.org/organization/209.html](http://www.apachecn.org/organization/209.html)

更多信息请参阅: [http://www.apachecn.org/organization/348.html](http://www.apachecn.org/organization/348.html)
