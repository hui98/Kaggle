# 1.背景
	利物浦大学研究者希望你能帮助他通过细胞膜内外电位差预测离子通道数，详情见
	https://www.kaggle.com/c/liverpool-ion-switching/overview

# 2.数据集
	训练集为5000000*3的数组，每一列分别代表时间(s), 电位差和离子通道数。
	测试集为2000000*2的数组，每一列代表时间和电位差，预测给出离子通道数，比赛截止前评分采用前30%的测试集。
![image](https://github.com/hui98/Kaggle/blob/master/ion-switching/pics/%E6%95%B0%E6%8D%AE%E9%9B%86.png)
	
	下载地址：https://www.kaggle.com/c/liverpool-ion-switching/data

# 3.思路&模型

	问题本身是一个时间序列预测问题，首先想到的是RNN。同时观察到数据集变化规律：电位差增加，通道数总是不降的；电位差
	减小，通道数总是不增的，这种‘变化’让我想到，或许可以求出原数据的多阶差分和原数据同时作为网络的输入，同时输出也去
	拟合通道数及其差分，或许就能捕捉到更加精细的变化。差分可以通过与一个每行相邻元素分别为-1和1的矩阵相乘实现，这个也
	不难实现，但是我想到或许可以用更加NN的方式去捕捉这些特征。
	
	假设有一序列 x = [500,600,700]
	设矩阵A = [[-1,1,0],
	        [0,-1,1]]
	其差分x' = x*A.T() = [100,100]
	
	上述过程等价于使用kernel = 3,padding = 0的两个卷积核k1 = [-1,1,0]，k2 = [0,-1,1]对x进行一维卷积操作
	通过CNN来学习特征或许是一个更好的解决方案。
	
	
	最终模型如下图：(Netron是根据类名来生成计算图的，和实际模型有些偏差，仅作为参考，详情见models.py)
![](https://github.com/hui98/Kaggle/blob/master/ion-switching/pics/%E6%89%B9%E6%B3%A8%202020-03-26%20235238.png)
![](https://github.com/hui98/Kaggle/blob/master/ion-switching/pics/lstm.png)
	
# 4.训练与调参
	
	lr设为0.002，优化器选择RMSprop,数据集划分的random_seed设置为321，
	迭代120次后选择在val set中交叉熵损失最小的解输出到submission.csv中。

# 5.总结
	1.最好提交成绩为0.941(最高0.943)，排名38,前4%。目前还是银牌。
	2.Resnet确实提高了网络表现(0.932~0.941)，但单纯靠叠层数似乎不是银弹，三层以上效果都差不多。
  根据讨论https://www.kaggle.com/c/liverpool-ion-switching/discussion/135576，
  或许是单纯使用CV中分数最高。
  
