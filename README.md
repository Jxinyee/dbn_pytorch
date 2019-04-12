# dbn_pytorch
用Pytorch 实现dbn 如果有别的问题，请联系我

要求的包 pytorch, numpy ,tqdm


关于rbm的原理可以看[这里](https://blog.csdn.net/itplus/article/details/19168937)

dbn只不过是将rbm进行固定住而已，然后推往下一层
其中 有几点比较重要的他没有仔细提到的就是关于MCMC中的Metropolis–Hastings算法与吉布斯采样

这几点着实也困扰了我好长时间 可以通过下面俩篇博客进行学习


[白马博客](https://blog.csdn.net/baimafujinji/article/details/53946367)

[刘建平博客](https://www.cnblogs.com/pinard/p/6638955.html)

关于数据集的描述[查看](https://blog.csdn.net/com_stu_zhang/article/details/6987632)


关于代码 的运行步骤请看exercise.ipynb，里面有详细的代码
另外data.npz.npy 、adddata、testdata.npz.npy 分别是经过预处理的训练数据
加强后smote生成了5000个点的数据 testdata.npz.npy 是测试数据

数据增强 -> smote..py


在这里我把 最外层的softmax层设置成了5个单元,当进行2分类的时候请修改

## Tips

另外： 虽然用了gibbs采样代码还是非常慢，请合理设置epoch！

刚开始我没用gibbs而是直接进行训练，跑了我整整一天，sad!!!!

后来我又用了svd，train了一发，发现同样不行 sad*2！！！！！

效果烂的快哭了，请不要和我一样心态失衡！

对了，如果exercise.ipynb 你对dbn的参数进行修改，然后运行ipynb的时候，记得重启下。

我也不知道为什么ipynb对缓存没有消除，就酱！

