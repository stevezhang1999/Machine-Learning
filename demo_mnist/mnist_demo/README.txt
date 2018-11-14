0、非常简单的一些tensorflow操作
1、tensorflow低层api实现的一个神经网络和简单的可视化图
2、建议用jupyter notebook打开.ipynb那两个文件... .py那两个有点丑。。
3、the_graph存放着一个只记录了结构图的文件，可以用tensorboard直接打开：
	在命令行模式中键入以下代码：
	tensorboard --logdir=你储存的文件夹路径名
	注意：使用绝对路径可以保证读取到，不要引号，不要引号，不要引号！！！
	之后根据提示打开网页即可。
4、ckpt存放着一个训练过的模型（跟主代码里的一样），有兴趣可以搜下怎么加载...（看看tf.Saver()的文档）