# -*- encoding : utf-8 -*-
# @File : 2hello_knn.py
# @Time : 2025/01/26 23:08:18
# @Author : haizhen 
# @Software : VScode

from sklearn.neighbors import KNeighborsClassifier


# 获取数据
x = [[1],[2],[0],[0]]
y = [1,1,0,0]

# 机器学习
# 1.实例化一个训练模型
estimator = KNeighborsClassifier(n_neighbors=2)

# 调用fit方法进行训练
estimator.fit(x,y)

# 预测其他值
# -1 0 1 10
ret = estimator.predict([[10]])
print(ret)