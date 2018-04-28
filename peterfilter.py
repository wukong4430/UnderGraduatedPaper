# -*- coding: utf-8 -*-
# @Author: Kicc Shen
# @Date:   2018-04-26 12:39:35
# @Last Modified by:   Kicc Shen
# @Last Modified time: 2018-04-26 15:48:50

from Processing import Processing
import numpy as np
import math


class peter_filter:

    def __init__(self, X_src, y_src, X_tar, y_tar):
        self.X_src = X_src
        self.y_src = y_src
        self.X_tar = X_tar
        self.y_tar = y_tar

    def CalculateD(self, row_i, row_j):
        size = row_i.shape[0]
        sum_distance = 0.0
        for i in range(size):
            sum_distance += pow((row_i[i] - row_j[i]), 2)
        distance = math.sqrt(sum_distance)
        return distance

    def transform(self):
        X_train, Y_train = self.X_src, self.y_src
        test_X_array, test_Y_array = self.X_tar, self.y_tar
        X_first_train, Y_first_train = self.X_src, self.y_src
        # 至此我们得到了测试集和需要筛选的训练集，现在就是要遍历筛选出我们需要的和测试集相似的训练集
        all_group_pair = []  # all_group用来保存测试集和训练集的簇，以便于下一步的筛选
        for train_data_i in range(X_train.shape[0]):
            min_distance = float("inf")  # 将初始距离设置为正无穷
            save_test_x = []
            save_test_y = []
            each_group = []
            each_x_group = []
            each_y_group = []
            each_x_group.append(list(X_first_train[train_data_i]))
            each_y_group.append(Y_first_train[train_data_i])
            for test_data_j in range(test_X_array.shape[0]):
                # 计算训练集中每一个元素与每一个测试集之间的距离大小，选最小的，保存为改测试集的“粉丝”
                distance = self.CalculateD(
                    X_first_train[train_data_i], test_X_array[test_data_j])
                if distance < min_distance:
                    save_test_x = list(test_X_array[test_data_j])
                    save_test_y = test_Y_array[test_data_j]
                    min_distance = distance
            each_x_group.append(save_test_x)
            each_y_group.append(save_test_y)

            each_group.append(each_x_group)
            each_group.append(each_y_group)
            all_group_pair.append(each_group)
        # all_group_pair:[[[[1.0, 1.0], [2.0, 3.0]], [1.0, 0.0]]] 第一个是训练集，第二个是测试集，第三个是两个集对应的标签
        # 至此，得到每一个彩色球与他最近的白球的组合对，接下来，反着求白球与这些组队中最近的彩色球，这些彩色球将作为训练集
        second_train_X_data = []
        second_train_Y_data = []
        for i in range(test_X_array.shape[0]):
            min_distance = float("inf")  # 将初始距离设置为正无穷
            save_train_x = []
            save_train_y = []
            for j in range(len(all_group_pair)):
                if list(test_X_array[i]) in all_group_pair[j][0]:
                    distance = self.CalculateD(np.array(
                        all_group_pair[j][0][0]), np.array(all_group_pair[j][0][1]))
                    if distance < min_distance:
                        save_train_x = all_group_pair[j][0][0]
                        save_train_y = all_group_pair[j][1][0]
                        min_distance = distance
            if len(save_train_x) > 0:
                second_train_X_data.append(save_train_x)
                second_train_Y_data.append(save_train_y)

        second_train_X_data = np.array(second_train_X_data)
        second_train_Y_data = np.array(second_train_Y_data)
        # print(second_train_X_data.shape)
        # print(second_train_Y_data.shape)
        # print(test_X_array.shape)
        # print(test_Y_array.shape)
        return second_train_X_data, second_train_Y_data, test_X_array, test_Y_array
