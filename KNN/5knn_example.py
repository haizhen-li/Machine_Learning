# -*- encoding : utf-8 -*-
# @File : 5knn_example.py
# @Time : 2025/01/28 21:40:02
# @Author : haizhen 
# @Software : VScode


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# 1.获取数据集
iris = load_iris()

# 2.数据基本处理
# 2.1 数据分割
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=22,test_size=0.2)


# 3.特征工程
# 3.1 实例化一个转换器
transfer = StandardScaler()
# 3.2 调用fit_transform()
x_train = transfer.fit_transform(x_train)
x_test = transfer.fit_transform(x_test)

# 4.机器学习（模型训练）
# 4.1 实例化一个估计器
estimator = KNeighborsClassifier(n_neighbors=5)
# 4.2 模型训练
estimator.fit(x_train, y_train)

# 5.模型评估
# 5.1 输出预测值
y_pre = estimator.predict(x_test)
print("预测值是：\n", y_pre)
print("预测值和真实值对比：\n", y_pre==y_test)

# 5.2 输出准确率
ret = estimator.score(x_test, y_test)
print("准确率是：\n", ret)