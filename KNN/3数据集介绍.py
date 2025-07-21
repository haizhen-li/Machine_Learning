# -*- encoding : utf-8 -*-
# @File : 3数据集介绍.py
# @Time : 2025/01/27 21:59:42
# @Author : haizhen 
# @Software : VScode

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris, fetch_20newsgroups
from sklearn.model_selection import train_test_split

# 1.数据集获取
# 1.1小数据集获取
# load 从本地获取
iris = load_iris()
# print(iris)

# 1.2 大数据集获取
# fetch 从网上下载
news = fetch_20newsgroups()
# print(news)

# 2.数据集属性描述 shift + option + A
""" print("鸢尾花的特征值:\n", iris["data"])
print("鸢尾花的目标值：\n", iris.target)
print("鸢尾花特征的名字：\n", iris.feature_names)
print("鸢尾花目标值的名字：\n", iris.target_names)
print("鸢尾花的描述：\n", iris.DESCR) """

# 3.数据可视化
# 3.1 数据类型转换DataFrame
iris_data = pd.DataFrame(data=iris.data, columns=['Sepal_Length', 'Sepal_Width','Petal_Length','Petal_Width'])
iris_data["target"] = iris.target
# print(iris_data)


def iris_plot(data, col1, col2):
    sns.lmplot(x=col1, y=col2, data=data,hue="target",fit_reg=False)
    plt.title("鸢尾花数据展示")
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.show()

#iris_plot(iris_data, "Sepal_Length","Petal_Width")
# iris_plot(iris_data, "Sepal_Width","Petal_Length")

# 4.数据集的划分 random_state是随机数种子，不同的种子会造成不同的随机采样结果。
x_train, x_test,y_train,y_test = train_test_split(iris.data, iris.target,test_size=0.2,random_state=22)
# 返回值：这四个都是
""" print("训练集的特征值是：\n", x_train)
print("测试集的特征值是：\n", x_test)
print("训练集的特征值是：\n", y_train)
print("测试集的特征值是：\n", y_test) """

print("训练集的特征值形状：\n", y_train.shape)
print("测试集的特征值形状：\n", y_test.shape)