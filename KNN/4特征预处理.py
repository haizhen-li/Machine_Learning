# -*- encoding : utf-8 -*-
# @File : 4特征预处理.py
# @Time : 2025/01/28 20:37:25
# @Author : haizhen 
# @Software : VScode

from sklearn.preprocessing import MinMaxScaler,StandardScaler
import pandas as pd

data = pd.read_csv("./data/dating.txt")
print(data)

""" # 1.归一化
# 1.1 实例化一个转换器
transfer = MinMaxScaler(feature_range=(0,10))
# 2.调用fit_transform(data)
minmax_data = transfer.fit_transform(data[['milage','Liters','Consumtime']])
print("经过归一化处理之后的数据为：\n", minmax_data) """

# 数据中异常点可能多，（1.2 1.3 134.0）的归一化就会出现不稳定异常值
# 鲁棒性比较差

# 2.标准化
# 2.1 实例化一个转换器
transfer = StandardScaler()
# 2.调用fit_transform(data)
minmax_data = transfer.fit_transform(data[['milage','Liters','Consumtime']])
print("经过标准化处理之后的数据为：\n", minmax_data)

