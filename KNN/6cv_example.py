# -*- encoding : utf-8 -*-
# @File : 6cv_example.py
# @Time : 2025/01/28 22:10:21
# @Author : haizhen 
# @Software : VScode


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# 1.获取数据集
iris = load_iris()

# 2.数据基本处理
# 2.1 数据分割
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=20,test_size=0.2)


# 3.特征工程
# 3.1 实例化一个转换器
transfer = StandardScaler()
# 3.2 调用fit_transform()
x_train = transfer.fit_transform(x_train)
x_test = transfer.fit_transform(x_test)

# 4.机器学习（模型训练）
# 4.1 实例化一个估计器
estimator = KNeighborsClassifier(n_neighbors=1)

# 4.2 调用交叉验证网格搜索
param_grid = {"n_neighbors": [1,3,5,7,9]}
estimator = GridSearchCV(estimator, param_grid=param_grid,cv=10, n_jobs=-1)


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

# 5.3 其他评价指标
print("最好的模型：\n", estimator.best_estimator_)
print("最好的结果：\n", estimator.best_score_)
print("整体模型结果：\n", estimator.cv_results_)
